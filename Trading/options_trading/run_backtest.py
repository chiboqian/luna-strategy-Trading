#!/usr/bin/env python3
"""
Runs a backtest for a given strategy.
Opens position on the first business day of the week (usually Monday).
Closes position either on the last business day of the week (default)
or after a specified number of business days.

Usage:
    python util/run_backtest.py --start-date 2023-01-01 --end-date 2023-06-30 \
        --strategy synthetic_long --symbol SPY --historical data/options.parquet \
        --output-dir results/spy_backtest --strategy-args "--days 30"
"""

import argparse
import subprocess
import sys
import json
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pathlib import Path
import shlex

# Strategy Defaults (TP, SL)
STRATEGY_DEFAULTS = {
    "synthetic_long": {"tp": 0.5, "sl": 0.4},
    "bull_put_spread": {"tp": 0.5, "sl": 1.0},
    "iron_condor": {"tp": 0.5, "sl": 1.0},
    "straddle": {"tp": 0.25, "sl": 0.15},
    "long_call": {"tp": 0.5, "sl": 0.3},
    "long_stock": {"tp": 0.1, "sl": 0.05}
}

def get_backtest_schedule(start_date, end_date, hold_days=None, every_day=False):
    """
    Returns a list of (open_date, close_date) tuples.
    Handles US holidays.
    """
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    bdays = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    
    if every_day:
        schedule = []
        for day in bdays:
            open_date = day
            if hold_days is not None:
                close_date = open_date + (hold_days * us_bd)
            else:
                close_date = open_date  # Default to same-day close for daily mode
            schedule.append((open_date, close_date))
        return schedule
    
    weeks = {}
    for day in bdays:
        # isocalendar returns (year, week, day)
        iso_year, iso_week, _ = day.isocalendar()
        key = (iso_year, iso_week)
        if key not in weeks:
            weeks[key] = []
        weeks[key].append(day)
        
    schedule = []
    for key in sorted(weeks.keys()):
        days = sorted(weeks[key])
        if not days:
            continue
        
        # First business day of the week
        open_date = days[0]
        
        if hold_days is not None:
            close_date = open_date + (hold_days * us_bd)
        else:
            # Last business day of the week
            close_date = days[-1]
        
        schedule.append((open_date, close_date))
        
    return schedule

def run_command(cmd, verbose=False):
    print(f"Exec: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            if result.stdout: print(result.stdout)
            if result.stderr: print(result.stderr, file=sys.stderr)
        return result
    except Exception as e:
        print(f"Error executing command: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run backtest cycle")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--strategy", required=True, choices=["synthetic_long", "bull_put_spread", "iron_condor", "straddle", "long_call", "long_stock"], help="Strategy to run")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. SPY)")
    parser.add_argument("--historical", required=True, help="Path to historical data file or dataset directory")
    parser.add_argument("--underlying", help="Path to underlying data file or dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--strategy-args", default="", help="Additional args for strategy script (e.g. '--days 45')")
    parser.add_argument("--hold-days", type=int, default=None, help="Number of business days to hold position (default: close on last day of week)")
    parser.add_argument("--entry-time", default="09:30:00", help="Time of entry (HH:MM:SS)")
    parser.add_argument("--exit-time", default="15:55:00", help="Time of exit (HH:MM:SS)")
    parser.add_argument("--stop-loss", type=float, help="Stop Loss % (e.g. 0.5 for 50%)")
    parser.add_argument("--take-profit", type=float, help="Take Profit % (e.g. 0.5 for 50%)")
    parser.add_argument("--every-day", action="store_true", help="Run backtest every trading day (instead of weekly)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent
    strategy_script = base_dir / f"{args.strategy}.py"
    if not strategy_script.exists():
        strategy_script = base_dir / "options" / f"{args.strategy}.py"

    close_script = base_dir / "close_mock_order.py"
    if not close_script.exists():
        close_script = base_dir / "options" / "close_mock_order.py"
    
    if not strategy_script.exists():
        print(f"Error: Strategy script {strategy_script} not found.", file=sys.stderr)
        sys.exit(1)
    if not close_script.exists():
        print(f"Error: Close script {close_script} not found.", file=sys.stderr)
        sys.exit(1)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine TP/SL defaults if not provided
    defaults = STRATEGY_DEFAULTS.get(args.strategy, {})
    tp_pct = args.take_profit if args.take_profit is not None else defaults.get("tp")
    sl_pct = args.stop_loss if args.stop_loss is not None else defaults.get("sl")
    
    print(f"Strategy: {args.strategy}")
    print(f"Management: TP={tp_pct}, SL={sl_pct}")
    
    # Generate Schedule
    print(f"Generating schedule from {args.start_date} to {args.end_date}...")
    try:
        schedule = get_backtest_schedule(args.start_date, args.end_date, args.hold_days, args.every_day)
    except Exception as e:
        print(f"Error generating schedule: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(schedule)} trading sessions.")
    
    results = []
    
    for open_dt, close_dt in schedule:
        open_str = open_dt.strftime("%Y-%m-%d")
        close_str = close_dt.strftime("%Y-%m-%d")
        if args.entry_time:
            open_str = f"{open_str} {args.entry_time}"
        if args.exit_time:
            close_str = f"{close_str} {args.exit_time}"
        
        print(f"Processing Session: Open {open_str} -> Close {close_str}")
        
        # 1. Open Position
        open_str_safe = open_str.replace(" ", "_").replace(":", "")
        order_file = output_dir / f"order_{open_str_safe}.json"
        
        # Build Open Command
        cmd_open = [
            sys.executable, str(strategy_script),
            args.symbol,
            "--historical", args.historical,
            "--date", open_str,
            "--save-order", str(order_file),
            "--json"
        ]
        
        if args.underlying:
            cmd_open.extend(["--underlying", args.underlying])
        
        # Default to limit order (mid-price) for long_call and long_stock
        if args.strategy in ["long_call", "long_stock"]:
            cmd_open.append("--limit-order")
        
        # Add strategy specific args
        if args.strategy_args:
            cmd_open.extend(shlex.split(args.strategy_args))
            
        # Ensure quantity is set if not in args (default to 1 for backtest consistency)
        if "--quantity" not in args.strategy_args and "--amount" not in args.strategy_args:
            cmd_open.extend(["--quantity", "1"])
            
        res_open = run_command(cmd_open, args.verbose)
        
        if not res_open or res_open.returncode != 0:
            print(f"  ❌ Open failed on {open_str}")
            if res_open:
                print(f"     Stderr: {res_open.stderr.strip()}")
            continue
            
        if not order_file.exists():
            print(f"  ❌ Order file not created for {open_str}")
            continue
        
        # Debug: Print entry info
        try:
            with open(order_file, 'r') as f:
                od = json.load(f)
                qty = float(od.get('quantity', 0))
                filled_at = od.get('filled_at', open_str).replace('T', ' ')
                print(f"  ✅ Open: {od.get('id')} @ {filled_at} (Qty: {qty}, Entry Cost: ${od.get('entry_cash_flow', 0):.2f})")
                if qty == 0:
                    print(f"     ⚠️ Warning: Quantity is 0. Check --amount vs strategy risk.")
                    if res_open and res_open.stderr:
                        print(f"     ℹ️  Strategy Log: {res_open.stderr.strip()}")
        except:
            pass
            
        # 2. Close Position
        close_str_safe = close_str.replace(" ", "_").replace(":", "")
        pnl_file = output_dir / f"pnl_{open_str_safe}_to_{close_str_safe}.json"
        
        cmd_close = [
            sys.executable, str(close_script),
            "--order", str(order_file),
            "--historical", args.historical,
            "--date", close_str,
            "--output", str(pnl_file),
            "--json"
        ]
        
        if args.underlying:
            cmd_close.extend(["--underlying", args.underlying])
        
        # Pass management params
        if tp_pct is not None:
            cmd_close.extend(["--take-profit-pct", str(tp_pct)])
        if sl_pct is not None:
            cmd_close.extend(["--stop-loss-pct", str(sl_pct)])
        # Pass begin time (entry time) to start monitoring immediately
        cmd_close.extend(["--begin-time", open_str])
        
        res_close = run_command(cmd_close, args.verbose)
        
        if not res_close or res_close.returncode != 0:
            print(f"  ❌ Close failed on {close_str}")
            if res_close:
                print(f"     Stderr: {res_close.stderr.strip()}")
            continue
            
        # 3. Record Result
        try:
            with open(pnl_file, 'r') as f:
                pnl_data = json.load(f)
                total_pnl = pnl_data.get('total_pnl', 0.0)
                qty = float(pnl_data.get('quantity', 0))
                close_ts = pnl_data.get('close_date', close_str)
                reason = pnl_data.get('close_reason', 'Exit Time')
                print(f"  ✅ Close @ {close_ts} ({reason}) | P&L: ${total_pnl:.2f} (Qty: {qty})")
                
                if 'legs' in pnl_data:
                    for leg in pnl_data['legs']:
                        l_sym = leg.get('symbol')
                        l_price = leg.get('close_price', 0.0)
                        l_side = leg.get('side')
                        close_action = "Sell" if l_side == 'buy' else "Buy"
                        print(f"     Close {l_sym} ({close_action}): ${l_price:.2f}")

                results.append({
                    "open_date": open_str,
                    "close_date": close_str,
                    "pnl": total_pnl,
                    "entry_cost": pnl_data.get('entry_cash_flow', 0),
                    "exit_credit": pnl_data.get('exit_cash_flow', 0),
                    "quantity": pnl_data.get('quantity', 0)
                })
        except Exception as e:
            print(f"  ⚠️ Error reading P&L file: {e}")

    # Final Summary
    if results:
        df = pd.DataFrame(results)
        summary_file = output_dir / "backtest_summary.csv"
        df.to_csv(summary_file, index=False)
        
        total_pnl = df['pnl'].sum()
        wins = len(df[df['pnl'] > 0])
        losses = len(df[df['pnl'] <= 0])
        win_rate = wins / len(df) if len(df) > 0 else 0
        
        print("\n" + "="*40)
        print(f"Backtest Complete: {args.symbol} ({args.strategy})")
        print(f"Total Trades: {len(df)}")
        print(f"Total P&L:    ${total_pnl:.2f}")
        print(f"Win Rate:     {win_rate:.1%} ({wins}W / {losses}L)")
        print(f"Summary saved to: {summary_file}")
        print("="*40)
    else:
        print("\nNo trades completed.")

if __name__ == "__main__":
    main()
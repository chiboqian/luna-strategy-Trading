#!/usr/bin/env python3
"""
Analyzes backtest P&L JSON files in a directory.
Generates a CSV summary and prints performance metrics.

Usage:
    python util/analyze_backtest_results.py --dir results/spy_weekly
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path

def analyze_results(directory: str, output_csv: str = None, max_profit: float = None, max_loss: float = None, max_profit_pct: float = None, max_loss_pct: float = None):
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        print(f"Error: Directory {directory} not found.", file=sys.stderr)
        sys.exit(1)
    
    files = list(p.glob("pnl_*.json"))
    if not files:
        print(f"No pnl_*.json files found in {directory}", file=sys.stderr)
        sys.exit(0)
        
    data = []
    print(f"Processing {len(files)} files in {directory}...")
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                d = json.load(fp)
                
                # Try to parse open date from filename: pnl_OPEN_to_CLOSE.json
                open_date = "N/A"
                try:
                    # Remove pnl_ prefix and .json suffix
                    name_parts = f.stem[4:].split('_to_')
                    if len(name_parts) >= 1:
                        open_date = name_parts[0].replace('_', ' ')
                except:
                    pass
                
                row = {
                    'file': f.name,
                    'close_date': d.get('close_date'),
                    'close_reason': d.get('close_reason', 'N/A'),
                    'total_pnl': float(d.get('total_pnl', 0.0)),
                    'open_date': open_date,
                    'entry_cost': float(d.get('entry_cash_flow', 0.0)),
                    'exit_credit': float(d.get('exit_cash_flow', 0.0)),
                    'quantity': float(d.get('quantity', 0)),
                    'unit_pnl': float(d.get('unit_pnl', 0.0))
                }
                data.append(row)
        except Exception as e:
            print(f"Warning: Error reading {f.name}: {e}", file=sys.stderr)
            
    if not data:
        print("No valid data extracted.")
        sys.exit(0)
        
    df = pd.DataFrame(data)
    
    # Calculate Basis and Multiplier for Percentage Caps
    # Heuristic: Derive multiplier from Total PnL / (Unit PnL * Qty) if possible
    def get_multiplier(row):
        if abs(row['unit_pnl']) > 0.0001 and row['quantity'] > 0:
            m = row['total_pnl'] / (row['unit_pnl'] * row['quantity'])
            if abs(m - 1) < 0.1: return 1.0
            if abs(m - 100) < 10.0: return 100.0
        # Default to 100 (Options) if undetermined, unless entry cost looks like a stock price (> $10 and no decimals? hard to say)
        return 100.0

    df['multiplier'] = df.apply(get_multiplier, axis=1)
    # Basis is the absolute initial cash flow (Cost for Debit, Credit for Credit strategies)
    df['basis'] = (df['entry_cost'] * df['quantity'] * df['multiplier']).abs()

    # Apply scenario caps if provided
    if max_profit is not None:
        print(f"Scenario: Capping max profit at ${max_profit:.2f}")
        df['total_pnl'] = df['total_pnl'].apply(lambda x: min(x, max_profit) if x > 0 else x)
        
    if max_loss is not None:
        loss_limit = -abs(max_loss)
        print(f"Scenario: Capping max loss at ${abs(max_loss):.2f}")
        df['total_pnl'] = df['total_pnl'].apply(lambda x: max(x, loss_limit) if x < 0 else x)

    if max_profit_pct is not None:
        print(f"Scenario: Capping max profit at {max_profit_pct*100:.1f}% of basis")
        df['total_pnl'] = df.apply(lambda row: min(row['total_pnl'], row['basis'] * max_profit_pct) if row['total_pnl'] > 0 else row['total_pnl'], axis=1)

    if max_loss_pct is not None:
        print(f"Scenario: Capping max loss at {max_loss_pct*100:.1f}% of basis")
        df['total_pnl'] = df.apply(lambda row: max(row['total_pnl'], -(row['basis'] * max_loss_pct)) if row['total_pnl'] < 0 else row['total_pnl'], axis=1)
    
    # Sort by date
    if 'close_date' in df.columns:
        df['close_date'] = pd.to_datetime(df['close_date'], format='mixed', utc=True)
        df = df.sort_values('close_date')
    
    # Save CSV
    if output_csv:
        out_path = Path(output_csv)
    else:
        out_path = p / "analysis_summary.csv"
        
    df.to_csv(out_path, index=False)
    print(f"Saved summary CSV to: {out_path}")
    
    # Trade History
    print("\n" + "="*110)
    print("Trade History")
    print("="*110)
    print(f"{'Open Date':<19} | {'Close Date':<19} | {'Cost/Credit':>12} | {'P&L':>12} | {'%':>8} | {'Result':<6} | {'Reason':<20}")
    print("-" * 110)
    
    for _, row in df.iterrows():
        date_str = "N/A"
        if pd.notnull(row['close_date']):
            ts = row['close_date']
            if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
                date_str = ts.strftime('%Y-%m-%d')
            else:
                date_str = ts.strftime('%Y-%m-%d %H:%M')
        
        open_str = row.get('open_date', 'N/A')
        
        initial_cash = row['entry_cost'] * row['quantity'] * row['multiplier']
        result = "WIN" if row['total_pnl'] > 0 else "LOSS" if row['total_pnl'] < 0 else "FLAT"
        pct = (row['total_pnl'] / abs(initial_cash)) * 100 if initial_cash != 0 else 0.0
        reason = str(row.get('close_reason', 'N/A'))
        print(f"{open_str:<19} | {date_str:<19} | ${initial_cash:>11.2f} | ${row['total_pnl']:>11.2f} | {pct:>7.1f}% | {result:<6} | {reason:<20}")

    # Analysis
    total_trades = len(df)
    total_pnl = df['total_pnl'].sum()
    wins = df[df['total_pnl'] > 0]
    losses = df[df['total_pnl'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    avg_win = wins['total_pnl'].mean() if win_count > 0 else 0
    avg_loss = losses['total_pnl'].mean() if loss_count > 0 else 0
    
    gross_profit = wins['total_pnl'].sum()
    gross_loss = abs(losses['total_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    expectancy = total_pnl / total_trades if total_trades > 0 else 0
    
    # Drawdown
    df['cumulative_pnl'] = df['total_pnl'].cumsum()
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()
    
    # Biggest Win/Loss
    best_trade = df.loc[df['total_pnl'].idxmax()] if not df.empty else None
    worst_trade = df.loc[df['total_pnl'].idxmin()] if not df.empty else None

    lines = []
    lines.append("\n" + "="*40)
    lines.append(f"Backtest Analysis Report")
    lines.append("="*40)
    lines.append(f"Total Trades:    {total_trades}")
    lines.append(f"Total P&L:       ${total_pnl:,.2f}")
    lines.append(f"Expectancy:      ${expectancy:,.2f} per trade")
    lines.append(f"Win Rate:        {win_rate:.1%} ({win_count}W / {loss_count}L)")
    lines.append(f"Avg Win:         ${avg_win:,.2f}")
    lines.append(f"Avg Loss:        ${avg_loss:,.2f}")
    lines.append(f"Profit Factor:   {profit_factor:.2f}")
    lines.append(f"Max Drawdown:    ${max_drawdown:,.2f}")
    
    if best_trade is not None:
        date_str = best_trade['close_date'].strftime('%Y-%m-%d') if pd.notnull(best_trade['close_date']) else "N/A"
        lines.append(f"Biggest Win:     ${best_trade['total_pnl']:,.2f} ({date_str})")
    
    if worst_trade is not None:
        date_str = worst_trade['close_date'].strftime('%Y-%m-%d') if pd.notnull(worst_trade['close_date']) else "N/A"
        lines.append(f"Biggest Loss:    ${worst_trade['total_pnl']:,.2f} ({date_str})")
    
    # Monthly breakdown
    if 'close_date' in df.columns:
        lines.append("\n--- Monthly P&L ---")
        df['month'] = df['close_date'].dt.tz_localize(None).dt.to_period('M')
        monthly = df.groupby('month')['total_pnl'].sum()
        for period, pnl in monthly.items():
            lines.append(f"{period}: ${pnl:,.2f}")

    report_text = "\n".join(lines)
    print(report_text)

    # Save report to file
    report_path = p / "analysis_report.txt"
    try:
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"\nSaved report to: {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Analyze backtest P&L files")
    parser.add_argument("--dir", required=True, help="Directory containing pnl_*.json files")
    parser.add_argument("--output", help="Path to save summary CSV (default: analysis_summary.csv in dir)")
    parser.add_argument("--max-profit", type=float, help="Scenario: Cap max profit per trade")
    parser.add_argument("--max-loss", type=float, help="Scenario: Cap max loss per trade (positive number)")
    parser.add_argument("--max-profit-pct", type=float, help="Scenario: Cap max profit as %% of basis (e.g. 0.5 for 50%%)")
    parser.add_argument("--max-loss-pct", type=float, help="Scenario: Cap max loss as %% of basis (e.g. 0.2 for 20%%)")
    args = parser.parse_args()
    
    analyze_results(args.dir, args.output, args.max_profit, args.max_loss, args.max_profit_pct, args.max_loss_pct)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Creates a Long Straddle position for a given symbol.

A Long Straddle is a neutral strategy that profits from significant volatility
in either direction.

Structure (2 Legs):
1. Buy ATM Call
2. Buy ATM Put

Profits if the stock moves significantly (more than the premium paid) in either direction.
Max Risk: Limited to the premium paid.
Max Profit: Unlimited (upside), Substantial (downside).

Usage:
    python util/straddle.py SPY --days 30 --quantity 1
    python util/straddle.py AAPL --amount 2000 --dry-run
"""

import sys
import argparse
import json
import yaml
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import pandas as pd
except ImportError:
    pd = None

# Add parent directory to path to import Trading modules
sys.path.insert(0, str(Path(__file__).parent.parent / "Trading"))
try:
    from alpaca_client import AlpacaClient
except ImportError:
    # Fallback if running from proper context
    try:
        from Trading.alpaca_client import AlpacaClient
    except ImportError:
        print("Error: Could not import AlpacaClient. Check python path.", file=sys.stderr)
        sys.exit(1)

try:
    from mock_client import MockOptionClient
except ImportError:
    # Fallback if running from root
    try:
        from options_trading.mock_client import MockOptionClient
    except ImportError:
        pass

def get_closest_contract(contracts: List[Dict], target_strike: float, option_type: str) -> Optional[Dict]:
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
    filtered.sort(key=lambda x: abs(float(x['strike_price']) - target_strike))
    return filtered[0]

def select_best_expiration(expirations: Dict, target_dte: int) -> str:
    sorted_exps = sorted(expirations.keys())
    friday_exps = []
    for exp in sorted_exps:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            if exp_date.weekday() == 4:
                friday_exps.append(exp)
        except:
            pass
    if friday_exps: return friday_exps[0]
    return sorted_exps[0] if sorted_exps else None

def print_payoff_diagram(current_price: float, strike: float, debit: float):
    """Prints an ASCII payoff diagram for Long Straddle."""
    # Range: +/- 20%
    low_price = strike * 0.80
    high_price = strike * 1.20
    price_step = (high_price - low_price) / 20
    
    break_even_low = strike - debit
    break_even_high = strike + debit
    
    print("\n--- Payoff Diagram (at Expiration) ---")
    print(f"       Risk | Profit")
    
    prices = []
    curr = high_price
    while curr >= low_price:
        prices.append(curr)
        curr -= price_step
        
    for p in prices:
        # Value of Call: max(0, p - strike)
        # Value of Put: max(0, strike - p)
        # Total Value: abs(p - strike)
        # PnL: Total Value - Debit
        pnl = abs(p - strike) - debit
        
        # Draw bar
        # Scale: max risk is debit.
        scale = debit if debit > 0 else 1.0
        bar_len = int(abs(pnl) / scale * 5) # Scale factor
        bar_len = min(bar_len, 20)
        
        marker = " "
        if abs(p - current_price) < price_step/2: marker = "← NOW"
        elif abs(p - break_even_low) < price_step/2: marker = "← BE"
        elif abs(p - break_even_high) < price_step/2: marker = "← BE"
        elif abs(p - strike) < price_step/2: marker = "← STRIKE"
        
        if pnl > 0:
            bar = " " * 10 + "|" + "#" * bar_len + " " + f"+${pnl:.2f} {marker}" 
        elif pnl < 0:
            bar = " " * (10 - bar_len) + "#" * bar_len + "|" + " " * 11 + f"-${abs(pnl):.2f} {marker}"
        else:
            bar = " " * 10 + "|" + " " * 11 + f" $0.00 {marker}"
            
        print(f"${p:7.2f} {bar}")

def main():
    # Load Config
    config_path = Path(__file__).parent.parent / "config" / "Options.yaml"
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass

    defaults = config.get("options", {}).get("straddle", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 14)
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Long Straddle Strategy")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of contracts")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days), default {default_window}")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount (max risk)")
    parser.add_argument("--limit-order", action="store_true", help="Use limit order at mid-price")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--historical", type=str, help="Path to historical data file (mock mode)")
    parser.add_argument("--underlying", type=str, help="Path to underlying data file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    
    args = parser.parse_args()

    if args.quantity is not None:
        args.amount = 0.0
    elif args.amount is not None:
        pass
    elif default_amount > 0:
        args.amount = default_amount
    else:
        args.quantity = 1

    if args.date:
        try:
            reference_date = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            reference_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        reference_date = datetime.now()

    if args.historical:
        client = MockOptionClient(args.historical, reference_date, args.save_order, args.underlying)
    else:
        client = AlpacaClient()

    symbol = args.symbol.upper()
    
    # Check for existing positions
    if not args.historical:
        try:
            positions = client.get_all_positions()
            for p in positions:
                if p['symbol'] == symbol:
                    msg = f"Existing stock position in {symbol}. Skipping."
                    if args.json: print(json.dumps({"status": "skipped", "reason": msg}))
                    else: print(msg)
                    return
                if p.get('asset_class') == 'us_option' and len(p['symbol']) >= 15:
                    if p['symbol'][:-15] == symbol:
                        msg = f"Existing option position in {symbol} ({p['symbol']}). Skipping."
                        if args.json: print(json.dumps({"status": "skipped", "reason": msg}))
                        else: print(msg)
                        return
        except Exception as e:
            if not args.json: print(f"Warning: Check for existing positions failed: {e}", file=sys.stderr)

    # 1. Get Spot Price
    current_price = 0.0
    try:
        trade = client.get_stock_latest_trade(symbol)
        if trade:
            current_price = float(trade.get('p', 0))
        if current_price <= 0:
            snap = client.get_stock_snapshot(symbol)
            if snap:
                current_price = float(snap.get('latestTrade', {}).get('p') or 
                                      snap.get('dailyBar', {}).get('c') or 0)
    except Exception as e:
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(symbol)
                if hasattr(ticker, 'fast_info') and ticker.fast_info.last_price:
                    current_price = float(ticker.fast_info.last_price)
            except:
                pass

    if current_price <= 0:
        err = {"error": f"Could not determine current price for {symbol}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    if not args.json:
        print(f"Symbol: {symbol}")
        print(f"Current Price: ${current_price:.2f}")

    # 2. Find Options
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    if not args.json:
        print(f"Searching for contracts expiring >= {start_date}...")

    try:
        # Narrow range for ATM
        min_strike = current_price * 0.90
        max_strike = current_price * 1.10
        
        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            limit=10000,
            status='active'
        )
    except Exception as e:
        err = {"error": f"Failed to fetch contracts: {e}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    if not contracts:
        err = {"error": f"No contracts found in window {start_date} to {end_date}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Group by expiration
    expirations = {}
    for c in contracts:
        exp = c['expiration_date']
        if exp not in expirations: expirations[exp] = []
        expirations[exp].append(c)
    
    selected_exp = select_best_expiration(expirations, args.days)
    if not selected_exp:
        if args.json: print(json.dumps({"error": "No valid expirations"}))
        return
    
    exp_contracts = expirations[selected_exp]
    
    # Find ATM Strike (closest to current price) that has both Call and Put
    strikes = {}
    for c in exp_contracts:
        k = float(c['strike_price'])
        if k not in strikes: strikes[k] = {}
        strikes[k][c['type']] = c
        
    # Filter for strikes with both legs
    valid_strikes = [k for k, v in strikes.items() if 'call' in v and 'put' in v]
    
    if valid_strikes:
        best_strike = min(valid_strikes, key=lambda x: abs(x - current_price))
        atm_call = strikes[best_strike]['call']
        atm_put = strikes[best_strike]['put']
    else:
        err = {"error": "Could not find ATM contracts with both legs"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Fetch snapshots for pricing
    leg_symbols = [atm_call['symbol'], atm_put['symbol']]
    joined_symbols = ",".join(leg_symbols)
    snapshots = client.get_option_snapshot(joined_symbols)
    if 'latestQuote' in snapshots: snapshots = {leg_symbols[0]: snapshots}

    total_cost = 0.0
    legs = []
    
    for contract in [atm_call, atm_put]:
        sym = contract['symbol']
        snap = snapshots.get(sym, {})
        quote = snap.get('latestQuote', {})
        ask = float(quote.get('ap') or 0)
        last = float(snap.get('latestTrade', {}).get('p') or 0)
        price = ask if ask > 0 else last
        
        total_cost += price
        legs.append({
            "symbol": sym,
            "side": "buy",
            "position_intent": "buy_to_open",
            "ratio_qty": 1,
            "estimated_price": price
        })

    # Auto-Quantity
    if args.amount and args.amount > 0:
        cost_per_straddle = total_cost * 100
        if cost_per_straddle > 0:
            args.quantity = max(1, int(args.amount // cost_per_straddle))

    if not args.json:
        print(f"\n--- Straddle Analysis ---")
        print(f"Expiration: {selected_exp}")
        print(f"Strike:     ${float(atm_call['strike_price']):.2f}")
        print(f"Net Debit:  ${total_cost:.2f}")
        print(f"Break Even: ${float(atm_call['strike_price']) - total_cost:.2f} / ${float(atm_call['strike_price']) + total_cost:.2f}")
        print(f"Quantity:   {args.quantity}")
        
        print_payoff_diagram(current_price, float(atm_call['strike_price']), total_cost)

    if args.dry_run:
        if args.json:
            print(json.dumps({"status": "dry_run", "legs": legs, "cost": total_cost}, indent=2))
        else:
            print("\nDry Run Complete.")
        return

    # Execute
    if not args.json:
        print(f"\nSubmitting order...")
        
    try:
        entry_cash_flow = -total_cost # Debit
        kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
        if isinstance(client, MockOptionClient):
            kwargs['entry_cash_flow'] = entry_cash_flow
            kwargs['underlying_price'] = current_price
        
        response = client.place_option_limit_order(**kwargs) if args.limit_order else client.place_option_market_order(**kwargs) # Debit is negative cash flow
            
        if args.json:
            print(json.dumps({"status": "executed", "order": response}, indent=2))
        else:
            print(f"Order Submitted: {response.get('id')}")
            
    except Exception as e:
        err = {"error": str(e)}
        if args.json: print(json.dumps(err))
        else: print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
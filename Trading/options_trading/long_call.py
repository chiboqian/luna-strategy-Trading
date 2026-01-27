#!/usr/bin/env python3
"""
Creates a Long Call position for a given symbol.

A Long Call is a bullish strategy that profits from a rise in the underlying stock price.
Risk is limited to the premium paid. Profit potential is unlimited.

Structure (1 Leg):
1. Buy ATM (or slightly OTM/ITM) Call

Usage:
    python util/long_call.py SPY --days 30 --quantity 1
    python util/long_call.py AAPL --amount 2000 --dry-run
"""

import sys
import argparse
import json
import yaml
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

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

    defaults = config.get("options", {}).get("long_call", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 14)
    default_amount = defaults.get("default_amount", 0.0)
    default_moneyness = defaults.get("moneyness", 1.0) # 1.0 = ATM

    parser = argparse.ArgumentParser(description="Execute Long Call Strategy")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of contracts")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days), default {default_window}")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount (max risk)")
    parser.add_argument("--moneyness", type=float, default=default_moneyness, help=f"Moneyness (1.0 = ATM, >1.0 = OTM, <1.0 = ITM for calls)")
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
        target_strike = current_price * args.moneyness
        min_strike = target_strike * 0.90
        max_strike = target_strike * 1.10
        
        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            type='call',
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
    call_contract = get_closest_contract(exp_contracts, target_strike, 'call')
    
    if not call_contract:
        err = {"error": "Could not find suitable call contract"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    sym = call_contract['symbol']
    snapshots = client.get_option_snapshot(sym)
    if 'latestQuote' in snapshots: snapshots = {sym: snapshots}
    
    snap = snapshots.get(sym, {})
    quote = snap.get('latestQuote', {})
    ask = float(quote.get('ap') or 0)
    bid = float(quote.get('bp') or 0)
    last = float(snap.get('latestTrade', {}).get('p') or 0)
    
    mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else last
    price = mid if args.limit_order else (ask if ask > 0 else last)
    total_cost = price
    
    legs = [{
        "symbol": sym,
        "side": "buy",
        "position_intent": "buy_to_open",
        "ratio_qty": 1,
        "estimated_price": price
    }]

    if args.amount and args.amount > 0:
        cost_per_contract = total_cost * 100
        if cost_per_contract > 0:
            args.quantity = max(1, int(args.amount // cost_per_contract))

    if not args.json:
        print(f"\n--- Long Call Analysis ---")
        print(f"Expiration: {selected_exp}")
        print(f"Strike:     ${float(call_contract['strike_price']):.2f}")
        print(f"Premium:    ${price:.2f}")
        print(f"Total Cost: ${total_cost * 100:.2f} per contract")
        print(f"Quantity:   {args.quantity}")

    if args.dry_run:
        if args.json:
            print(json.dumps({"status": "dry_run", "legs": legs, "cost": total_cost}, indent=2))
        else:
            print("\nDry Run Complete.")
        return

    if not args.json:
        print(f"\nSubmitting order...")
        
    try:
        entry_cash_flow = -total_cost
        kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'simple'}
        
        if args.limit_order:
             kwargs['limit_price'] = round(price, 2)
             kwargs['type'] = 'limit'
        else:
             kwargs['type'] = 'market'

        if isinstance(client, MockOptionClient):
            kwargs['entry_cash_flow'] = entry_cash_flow
            response = client.place_option_market_order(**kwargs)
        else:
            response = client.place_option_market_order(**kwargs)
            
        if args.json:
            print(json.dumps({"status": "executed", "order": response}, indent=2))
        else:
            print(f"Order Submitted: {response.get('id')}")
            print(f"Purchase Price: ${price:.2f}")
            
    except Exception as e:
        err = {"error": str(e)}
        if args.json: print(json.dumps(err))
        else: print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
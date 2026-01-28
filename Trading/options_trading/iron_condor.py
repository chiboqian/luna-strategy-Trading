#!/usr/bin/env python3
"""
Creates an Iron Condor position for a given symbol.

An Iron Condor is a neutral, defined-risk strategy that profits from:
- Time decay (Theta)
- Drop in Implied Volatility (Vega)
- Stock price staying within a range

Structure (4 Legs):
1. Buy OTM Put (Long Put) - Protection
2. Sell OTM Put (Short Put) - Income
3. Sell OTM Call (Short Call) - Income
4. Buy OTM Call (Long Call) - Protection

Usage:
    python util/iron_condor.py SPY --days 45 --short-delta 0.16 --width 5 --quantity 1
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
    import numpy as np
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

def get_contract_by_delta(contracts: List[Dict], snapshots: Dict, target_delta: float, 
                          option_type: str, tolerance: float = 0.15) -> Optional[Dict]:
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
    
    best_contract = None
    best_diff = float('inf')
    
    for contract in filtered:
        sym = contract['symbol']
        snap = snapshots.get(sym, {})
        greeks = snap.get('greeks', {})
        if not greeks: continue
            
        delta = float(greeks.get('delta') or 0)
        diff = abs(abs(delta) - abs(target_delta))
        
        if diff < best_diff and diff <= tolerance:
            best_diff = diff
            best_contract = contract
            best_contract['_delta'] = delta
    
    return best_contract

def check_liquidity(snapshot: Dict, max_spread_pct: float = 0.20) -> Tuple[bool, float, str]:
    quote = snapshot.get('latestQuote', {})
    bid = float(quote.get('bp') or 0)
    ask = float(quote.get('ap') or 0)
    if bid <= 0 or ask <= 0: return False, 1.0, "No valid quote"
    mid = (bid + ask) / 2
    spread = ask - bid
    spread_pct = spread / mid if mid > 0 else 1.0
    if spread_pct > max_spread_pct:
        return False, spread_pct, f"Wide spread ({spread_pct:.1%} > {max_spread_pct:.0%})"
    return True, spread_pct, "OK"

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

def print_payoff_diagram(current_price: float, short_put: float, long_put: float, 
                         short_call: float, long_call: float, credit: float):
    """Prints an ASCII payoff diagram for Iron Condor."""
    put_width = short_put - long_put
    call_width = long_call - short_call
    max_width = max(put_width, call_width)
    max_risk = max_width - credit
    
    be_low = short_put - credit
    be_high = short_call + credit
    
    low_price = long_put * 0.95
    high_price = long_call * 1.05
    price_step = (high_price - low_price) / 20
    
    print("\n--- Payoff Diagram (at Expiration) ---")
    print(f"       Risk | Profit")
    
    prices = []
    curr = high_price
    while curr >= low_price:
        prices.append(curr)
        curr -= price_step
        
    for p in prices:
        pnl = 0.0
        if p > short_call:
            loss = min(p - short_call, call_width)
            pnl -= loss
        if p < short_put:
            loss = min(short_put - p, put_width)
            pnl -= loss
        pnl += credit
        
        scale_factor = max(max_risk, credit)
        if scale_factor == 0: scale_factor = 1
        bar_len = int(abs(pnl) / scale_factor * 10)
        bar_len = min(bar_len, 20)
        
        marker = " "
        if abs(p - current_price) < price_step/2: marker = "← NOW"
        elif abs(p - be_low) < price_step/2: marker = "← BE"
        elif abs(p - be_high) < price_step/2: marker = "← BE"
        elif abs(p - short_put) < price_step/2: marker = "← S.PUT"
        elif abs(p - short_call) < price_step/2: marker = "← S.CALL"
        
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

    defaults = config.get("options", {}).get("iron_condor", {})
    default_days = defaults.get("min_days_to_expiration", 45)
    default_window = defaults.get("search_window_days", 14)
    default_short_delta = defaults.get("default_short_delta", 0.16) # ~1 SD
    default_width = defaults.get("default_width", 5.0)
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Iron Condor Strategy")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of spreads")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days), default {default_window}")
    parser.add_argument("--short-delta", type=float, default=default_short_delta, help=f"Target delta for short legs, default {default_short_delta}")
    parser.add_argument("--width", type=float, default=default_width, help=f"Wing width in dollars, default ${default_width}")
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
        # Fallback to YFinance for Indices (SPX, NDX, etc) if Alpaca fails
        if HAS_YFINANCE:
            yf_map = {'SPX': '^SPX', 'NDX': '^NDX', 'RUT': '^RUT', 'VIX': '^VIX'}
            yf_sym = yf_map.get(symbol, symbol)
            if not yf_sym.startswith('^') and symbol in ['SPX', 'NDX', 'RUT', 'VIX']:
                 yf_sym = f"^{symbol}"
            try:
                ticker = yf.Ticker(yf_sym)
                if hasattr(ticker, 'fast_info') and ticker.fast_info.last_price:
                    current_price = float(ticker.fast_info.last_price)
                if current_price <= 0:
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
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
        print(f"Target Delta: {args.short_delta}")
        print(f"Wing Width: ${args.width}")

    # 2. Find Options
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    if not args.json:
        print(f"Searching for contracts expiring >= {start_date}...")

    try:
        # Wide range to catch wings
        min_strike = current_price * 0.7
        max_strike = current_price * 1.3
        
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
    
    # Fetch snapshots for greeks
    contract_symbols = [c['symbol'] for c in exp_contracts]
    all_snapshots = {}
    try:
        batch_size = 100
        for i in range(0, len(contract_symbols), batch_size):
            batch = contract_symbols[i:i+batch_size]
            joined = ",".join(batch)
            snaps = client.get_option_snapshot(joined)
            if isinstance(snaps, dict):
                if 'latestQuote' in snaps or 'greeks' in snaps:
                    all_snapshots[batch[0]] = snaps
                else:
                    all_snapshots.update(snaps)
    except Exception as e:
        if not args.json: print(f"Warning: Snapshot fetch failed: {e}")

    # Find Legs
    # 1. Short Put (Delta ~ args.short_delta)
    short_put = get_contract_by_delta(exp_contracts, all_snapshots, args.short_delta, 'put')
    # 2. Short Call (Delta ~ args.short_delta)
    short_call = get_contract_by_delta(exp_contracts, all_snapshots, args.short_delta, 'call')
    
    if not short_put or not short_call:
        err = {"error": "Could not find short legs with target delta"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    s_put_strike = float(short_put['strike_price'])
    s_call_strike = float(short_call['strike_price'])
    
    # 3. Long Put (Short Put - Width)
    l_put_target = s_put_strike - args.width
    long_put = get_closest_contract(exp_contracts, l_put_target, 'put')
    
    # 4. Long Call (Short Call + Width)
    l_call_target = s_call_strike + args.width
    long_call = get_closest_contract(exp_contracts, l_call_target, 'call')
    
    if not long_put or not long_call:
        err = {"error": "Could not find long legs with target width"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return
        
    l_put_strike = float(long_put['strike_price'])
    l_call_strike = float(long_call['strike_price'])
    
    # Validate Structure
    if l_put_strike >= s_put_strike or l_call_strike <= s_call_strike:
        err = {"error": "Invalid strike hierarchy found"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    legs = [
        {"symbol": long_put['symbol'], "side": "buy", "position_intent": "buy_to_open", "ratio_qty": 1},
        {"symbol": short_put['symbol'], "side": "sell", "position_intent": "sell_to_open", "ratio_qty": 1},
        {"symbol": short_call['symbol'], "side": "sell", "position_intent": "sell_to_open", "ratio_qty": 1},
        {"symbol": long_call['symbol'], "side": "buy", "position_intent": "buy_to_open", "ratio_qty": 1}
    ]
    
    # Metrics
    metrics = {
        "net_credit": 0.0,
        "max_loss": 0.0,
        "max_profit": 0.0,
        "break_even_low": 0.0,
        "break_even_high": 0.0,
        "pop": 0.0
    }
    
    total_credit = 0.0
    mid_credit = 0.0
    
    for leg in legs:
        sym = leg['symbol']
        snap = all_snapshots.get(sym, {})
        quote = snap.get('latestQuote', {})
        bid = float(quote.get('bp') or 0)
        ask = float(quote.get('ap') or 0)
        mid = (bid + ask) / 2
        
        if leg['side'] == 'sell':
            total_credit += bid
            mid_credit += mid
            leg['estimated_price'] = bid
        else:
            total_credit -= ask
            mid_credit -= mid
            leg['estimated_price'] = ask
            
    metrics['net_credit'] = total_credit
    metrics['mid_credit'] = mid_credit
    metrics['max_profit'] = total_credit
    
    # Max Loss = Max Width - Credit
    put_width = s_put_strike - l_put_strike
    call_width = l_call_strike - s_call_strike
    max_width = max(put_width, call_width)
    metrics['max_loss'] = max_width - total_credit
    
    metrics['break_even_low'] = s_put_strike - total_credit
    metrics['break_even_high'] = s_call_strike + total_credit
    
    # POP approx
    s_put_delta = abs(float(all_snapshots.get(short_put['symbol'], {}).get('greeks', {}).get('delta') or 0))
    s_call_delta = abs(float(all_snapshots.get(short_call['symbol'], {}).get('greeks', {}).get('delta') or 0))
    metrics['pop'] = 1.0 - (s_put_delta + s_call_delta)
    
    # Auto-Quantity
    if args.amount and args.amount > 0:
        risk_per_spread = metrics['max_loss'] * 100
        if risk_per_spread > 0:
            args.quantity = max(1, int(args.amount // risk_per_spread))
            
    if not args.json:
        print(f"\n--- Iron Condor Analysis ---")
        print(f"Expiration: {selected_exp}")
        print(f"Strikes: {l_put_strike:.1f}/{s_put_strike:.1f} P  ---  {s_call_strike:.1f}/{l_call_strike:.1f} C")
        print(f"Net Credit: ${metrics['net_credit']:.2f} (Mid: ${metrics['mid_credit']:.2f})")
        print(f"Max Loss:   ${metrics['max_loss']:.2f}")
        print(f"Max Profit: ${metrics['max_profit']:.2f}")
        print(f"POP:        {metrics['pop']:.1%}")
        print(f"Quantity:   {args.quantity}")
        
        print_payoff_diagram(current_price, s_put_strike, l_put_strike, s_call_strike, l_call_strike, metrics['net_credit'])

    if args.dry_run:
        if args.json:
            print(json.dumps({
                "status": "dry_run",
                "legs": legs,
                "metrics": metrics
            }, indent=2))
        else:
            print("\nDry Run Complete.")
        return

    # Execute
    if not args.json:
        print(f"\nSubmitting order...")
        
    try:
        if args.limit_order:
            limit_price = round(metrics['mid_credit'], 2)
            entry_cash_flow = limit_price
            kwargs = {'legs': legs, 'quantity': args.quantity, 'limit_price': limit_price, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow
                kwargs['underlying_price'] = current_price
            response = client.place_option_limit_order(**kwargs)
        else:
            entry_cash_flow = metrics['net_credit']
            kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow
                kwargs['underlying_price'] = current_price
            response = client.place_option_market_order(**kwargs)
            
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
#!/usr/bin/env python3
"""
Creates a Synthetic Short with Protective Call position for a given symbol.

This strategy mimics shorting the stock with a protective cap (defined risk).
Structure (3 Legs):
1. Buy ATM Put (Synthetic Short leg 1)
2. Sell ATM Call (Synthetic Short leg 2)
3. Buy OTM Call (Protective leg, e.g. 5% OTM)

The combination of Leg 2 (Short Call) and Leg 3 (Long Call) forms a Bear Call Spread.
The combination of Leg 1 (Long Put) provides the downside profit.
Net result: "Synthetic Short Stock" with upside risk capped at the OTM strike.

Usage:
    python util/synthetic_short.py SPY --days 30 --with-protection --protection-pct 5 --quantity 1
"""

import sys
import argparse
import json
import yaml
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
    """Finds the contract with strike price closest to target."""
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
    
    # Sort by distance to target strike
    # strike_price is usually a string in API response
    filtered.sort(key=lambda x: abs(float(x['strike_price']) - target_strike))
    return filtered[0]

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

    defaults = config.get("options", {}).get("synthetic_short", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 35)
    default_prot_pct = defaults.get("default_protection_pct", 5.0)
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Synthetic Short with Protective Call")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of contract sets")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days) after min days, default {default_window}")
    parser.add_argument("--protection-pct", type=float, default=default_prot_pct, help=f"Protective Call distance from spot (%%), default {default_prot_pct}")
    parser.add_argument("--with-protection", action="store_true", help="Enable protective call (default: pure synthetic short)")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount to invest (default ${default_amount})")
    parser.add_argument("--limit-order", action="store_true", help="Use limit order at mid-price instead of market order")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Max bid/ask spread percent (default: 5.0)")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--historical", type=str, help="Path to historical data file (mock mode)")
    parser.add_argument("--underlying", type=str, help="Path to underlying data file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()

    # Resolve defaults priority
    if args.quantity is not None:
        args.amount = 0.0 # Force ignore amount logic
    elif args.amount is not None:
        pass # Use provided amount
    elif default_amount > 0:
        args.amount = default_amount
    else:
        args.quantity = 1 # Fallback default

    # Determine Reference Date
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
        
        # If price is 0 (market closed/no data), try snapshot
        if current_price <= 0:
            snap = client.get_stock_snapshot(symbol)
            if snap:
                # Try getting from latestTrade, then dailyBar
                current_price = float(snap.get('latestTrade', {}).get('p') or 
                                      snap.get('dailyBar', {}).get('c') or 0)
    except Exception as e:
        err = {"error": f"Failed to get price for {symbol}: {e}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error getting price: {e}", file=sys.stderr)
        return

    if current_price <= 0:
        err = {"error": f"Could not determine current price for {symbol}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Could not determine current price for {symbol}", file=sys.stderr)
        return

    # 2. Determine Targets
    atm_strike_target = current_price
    prot_strike_target = None
    if args.with_protection:
        # Protection is on the upside (Call), so strike > current price
        prot_strike_target = current_price * (1.0 + (args.protection_pct / 100.0))
    
    if not args.json:
        print(f"Symbol: {symbol}")
        print(f"Current Price: ${current_price:.2f}")
        if prot_strike_target:
            print(f"Targets -> ATM: ${atm_strike_target:.2f}, Prot: ${prot_strike_target:.2f} (+{args.protection_pct}%)")
        else:
            print(f"Targets -> ATM: ${atm_strike_target:.2f}, Prot: None (Pure Synthetic Short)")

    # 3. Find Options
    # Look for contracts expiring after roughly args.days
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    # Window of opportunity to find liquid expiration
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    if not args.json:
        print(f"Searching for contracts expiring >= {start_date}...")

    # Fetch contracts
    try:
        # Optimization: Narrow search to relevant strikes
        # We need strikes around ATM and Protection target (Upside)
        min_strike = atm_strike_target * 0.8
        if prot_strike_target:
            max_strike = prot_strike_target * 1.2
        else:
            max_strike = atm_strike_target * 1.2

        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            limit=10000, # Max limit to get all relevant ones
            status='active'
        )
    except Exception as e:
        err = {"error": f"Failed to fetch contracts: {e}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error fetching contracts: {e}", file=sys.stderr)
        return

    if not contracts:
        err = {"error": f"No contracts found in window {start_date} to {end_date}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"No contracts found in window {start_date} to {end_date}", file=sys.stderr)
        return

    # Group by expiration
    expirations = {}
    for c in contracts:
        exp = c['expiration_date']
        if exp not in expirations:
            expirations[exp] = []
        expirations[exp].append(c)
    
    # Sort expirations and pick the first one
    sorted_exps = sorted(expirations.keys())
    if not sorted_exps:
        if args.json:
            print(json.dumps({"error": "No valid expirations parsed"}))
        return
        
    selected_exp = sorted_exps[0]
    
    # Calculate days out roughly
    try:
        d_out = (datetime.strptime(selected_exp, "%Y-%m-%d") - reference_date).days
    except:
        d_out = "?"

    if not args.json:
        print(f"Selected Expiration: {selected_exp} (~{d_out} days out)")
    
    exp_contracts = expirations[selected_exp]
    
    # Find Legs
    # 1. ATM Put (Buy)
    atm_put = get_closest_contract(exp_contracts, atm_strike_target, 'put')
    # 2. ATM Call (Sell)
    atm_call = get_closest_contract(exp_contracts, atm_strike_target, 'call')
    # 3. Protective Call (Buy)
    prot_call = None
    if args.with_protection:
        prot_call = get_closest_contract(exp_contracts, prot_strike_target, 'call')
    
    # Validate legs
    missing = []
    if not atm_put: missing.append("ATM Put")
    if not atm_call: missing.append("ATM Call")
    if args.with_protection and not prot_call: missing.append("Protective Call")
    
    if missing:
        err = {"error": f"Could not find required legs: {', '.join(missing)}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Check for strike logic
    put_strike = float(atm_put['strike_price'])
    call_strike = float(atm_call['strike_price'])
    prot_strike = float(prot_call['strike_price']) if prot_call else 0.0
    
    if not args.json:
        print(f"Leg 1 (Long Put):  {atm_put['symbol']}  Strike=${put_strike}")
        print(f"Leg 2 (Short Call): {atm_call['symbol']} Strike=${call_strike}")
        if prot_call:
            print(f"Leg 3 (Prot Call): {prot_call['symbol']} Strike=${prot_strike}")
        else:
            print(f"Leg 3 (Prot Call): None")

    # Build Legs List
    legs = []
    
    if args.dry_run and not args.json:
        print("\n--- Strategy Analysis ---")
        print(f"Synthetic Short: Buy Put ${put_strike} / Sell Call ${call_strike}")
        if prot_call:
            print(f"Protective Call: Buy Call ${prot_strike}")
            # Max Risk Analysis
            # Risk from Short Call is capped by Long Call
            risk_per_share = max(0, prot_strike - call_strike)
            print(f"Max Upside Risk from Structure (excluding premiums): ${risk_per_share:.2f}/share")
        else:
            print("Protective Call: None")
            print(f"Max Upside Risk from Structure: Unlimited (Stock -> Infinity)")
        
    # Construct Legs
    # 1. Buy Put
    legs.append({
        "symbol": atm_put['symbol'],
        "side": "buy",
        "position_intent": "buy_to_open",
        "ratio_qty": 1
    })
    
    # 2. Sell Call (only if different from prot call)
    if not prot_call or call_strike != prot_strike:
        legs.append({
            "symbol": atm_call['symbol'],
            "side": "sell",
            "position_intent": "sell_to_open",
            "ratio_qty": 1
        })
    else:
        if not args.json:
            print("Notice: ATM Call and Protective Call strikes are identical. Canceling Short Call leg.")

    # 3. Buy Protective Call
    if prot_call:
        legs.append({
            "symbol": prot_call['symbol'],
            "side": "buy",
            "position_intent": "buy_to_open",
            "ratio_qty": 1
        })
    
    # If call_strike == prot_strike, we have Sell Call + Buy Call -> Cancel out.
    # We are left with Buy Put (Long Put strategy).
    if prot_call and atm_call['symbol'] == prot_call['symbol']:
        # They cancel perfectly
        legs = [{
            "symbol": atm_put['symbol'],
            "side": "buy",
            "position_intent": "buy_to_open",
            "ratio_qty": 1
        }]
    
    # --- METRICS CALCULATION ---
    metrics = {
        "net_cost": 0.0,
        "net_mid_cost": 0.0,
        "max_loss": 0.0,
        "break_even": 0.0,
        "contracts_data": {},
        "max_leg_spread": 0.0
    }
    warnings = []
    
    try:
        leg_symbols = [l['symbol'] for l in legs]
        if leg_symbols:
            joined_symbols = ",".join(leg_symbols)
            # Fetch Snapshots
            snapshots = client.get_option_snapshot(joined_symbols)
            if len(legs) == 1:
                if joined_symbols in snapshots:
                     pass
                else:
                     snapshots = {joined_symbols: snapshots}

            total_premium = 0.0
            total_mid_premium = 0.0
            net_delta = 0.0
            
            for leg in legs:
                sym = leg['symbol']
                side = leg['side']
                snap = snapshots.get(sym, {})
                metrics["contracts_data"][sym] = snap
                
                # Get Price
                quote = snap.get('latestQuote', {})
                ask = float(quote.get('ap') or 0)
                bid = float(quote.get('bp') or 0)
                last = float(snap.get('latestTrade', {}).get('p') or 0)
                mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else last
                
                if ask > 0 and bid > 0:
                    spread_pct = (ask - bid) / mid if mid > 0 else 0
                    if spread_pct > metrics['max_leg_spread']:
                        metrics['max_leg_spread'] = spread_pct
                    if spread_pct > (args.max_spread / 100.0):
                        warnings.append(f"Wide spread on {sym}: {spread_pct:.1%} > {args.max_spread}%")
                
                price = 0.0
                if 'buy' in side:
                    price = ask if ask > 0 else last
                    total_premium += price # Debit
                    total_mid_premium += mid
                else:
                    price = bid if bid > 0 else last
                    total_premium -= price # Credit (reduces cost)
                    total_mid_premium -= mid
                
                leg['estimated_price'] = price
                
                # Delta
                greeks = snap.get('greeks')
                if greeks:
                    d = float(greeks.get('delta') or 0)
                    if 'sell' in side: d = -d
                    net_delta += d
            
            metrics['net_cost'] = total_premium
            metrics['net_mid_cost'] = total_mid_premium
            metrics['net_delta'] = net_delta
            
            # Max Loss = (ProtCallStrike - CallStrike) + NetPremium
            # If pure synthetic short: Unlimited
            if prot_call:
                width = max(0, prot_strike - call_strike)
                metrics['max_loss'] = width + total_premium
            else:
                metrics['max_loss'] = float('inf')
            
            # Break Even = PutStrike - NetPremium (approx)
            metrics['break_even'] = put_strike - total_premium
            
            # --- Auto-Calculate Quantity if --amount is used ---
            if args.amount and args.amount > 0:
                # Calculate capital required per contract structure
                if args.with_protection:
                    # Defined Risk: Use Max Loss
                    risk_per_contract = metrics['max_loss'] * 100.0
                else:
                    # Undefined Risk (Pure Synthetic Short): Use Margin Requirement
                    # Reg T Margin for Short Call ~ 20% of Underlying
                    # We use 20% of strike as a proxy + net cost
                    margin_req = (call_strike * 0.20) * 100.0
                    risk_per_contract = margin_req + (metrics['net_cost'] * 100.0)
                
                if risk_per_contract > 0:
                    auto_qty = int(args.amount // risk_per_contract)
                    if auto_qty < 1:
                        msg = f"Warning: Amount ${args.amount} is insufficient for 1 contract (Requires ~${risk_per_contract:.2f})"
                        if not args.json:
                            print(f"\n{msg}")
                        else:
                            print(msg, file=sys.stderr)
                        args.quantity = 0
                    else:
                        args.quantity = auto_qty
                        if not args.json:
                            print(f"\nAuto-calculated Quantity: {args.quantity}")
                            print(f"  Based on Amount: ${args.amount}")
                            print(f"  Risk Per Contract: ${risk_per_contract:.2f}")

            if not args.json:
                print("\n--- Financial Analysis (Per Share) ---")
                print(f"Est. Net Premium: ${metrics['net_cost']:.2f} (Positive=Debit, Negative=Credit)")
                print(f"Mid Net Premium:  ${metrics['net_mid_cost']:.2f}")
                if prot_call:
                    print(f"Max Loss Risk:    ${metrics['max_loss']:.2f} (Cap @ ${prot_strike:.2f})")
                else:
                    print(f"Max Loss Risk:    Unlimited (No Cap)")
                
                print(f"Break Even:       ${metrics['break_even']:.2f}")
                print(f"Position Delta:   {metrics['net_delta']:.2f}")
                
                if prot_call:
                    print(f"Cap Efficiency:   Upside protection cost = ${metrics['net_cost']:.2f} vs Short Stock")
                
                print(f"Max Leg Spread:   {metrics['max_leg_spread']:.2%}")

            if warnings and not args.json:
                print("\n⚠️  Liquidity Warnings:")
                for w in warnings:
                    print(f"  - {w}")

    except Exception as e:
        if not args.json:
            print(f"Warning: Could not calculate metrics: {e}", file=sys.stderr)
        metrics['error'] = str(e)

    if args.dry_run:
        # Add per-leg details for JSON
        detailed_legs = []
        for leg in legs:
            sym = leg['symbol']
            snap = metrics.get('contracts_data', {}).get(sym, {})
            quote = snap.get('latestQuote', {})
            ask = float(quote.get('ap') or 0)
            bid = float(quote.get('bp') or 0)
            last = float(snap.get('latestTrade', {}).get('p') or 0)
            mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else last
            detailed_leg = dict(leg)
            detailed_leg.update({"bid": bid, "ask": ask, "mid": mid})
            detailed_legs.append(detailed_leg)
            
        result = {
            "status": "dry_run",
            "scan": {
                "current_price": current_price,
                "expiration": selected_exp,
            },
            "legs": detailed_legs,
            "metrics": metrics,
            "warnings": warnings
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\nDry Run Complete. Order not submitted.")
        return

    # Execute
    if not args.json:
        order_type = "LIMIT" if args.limit_order else "MARKET"
        print(f"\nSubmitting {order_type} order for {args.quantity}x structures...")
        
    try:
        if args.limit_order:
            limit_price = round(metrics['net_mid_cost'], 2)
            # Pass entry_cash_flow for mock orders (Debit = negative cash flow)
            # net_cost is positive for debit, negative for credit. entry_cash_flow should be negative for debit.
            entry_cash_flow = -limit_price
            kwargs = {'legs': legs, 'quantity': args.quantity, 'limit_price': limit_price, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow
                kwargs['underlying_price'] = current_price
            
            response = client.place_option_limit_order(**kwargs)
        else:
            entry_cash_flow = -metrics['net_cost'] # Debit is negative cash flow
            kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow
                kwargs['underlying_price'] = current_price

            response = client.place_option_market_order(**kwargs)
        
        result = {
            "status": "executed",
            "order": response,
            "warnings": warnings
        }
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Order Submitted Successfully!")
            print(f"Order ID: {response.get('id')}")
            print(f"Status: {response.get('status')}")
    except Exception as e:
        err = {"error": str(e)}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
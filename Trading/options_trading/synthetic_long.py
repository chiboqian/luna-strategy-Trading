#!/usr/bin/env python3
"""
Creates a Synthetic Long with Protective Put position for a given symbol.

This strategy mimics owning the stock with a protective floor (defined risk).
Structure (3 Legs):
1. Buy ATM Call (Synthetic Long leg 1)
2. Sell ATM Put (Synthetic Long leg 2)
3. Buy OTM Put (Protective leg, e.g. 5% OTM)

The combination of Leg 2 (Short Put) and Leg 3 (Long Put) forms a Bull Put Spread.
The combination of Leg 1 (Long Call) provides the unlimited upside.
Net result: "Synthetic Long Stock" with downside capped at the OTM strike.

Usage:
    python util/synthetic_long.py SPY --days 30 --with-protection --protection-pct 5 --quantity 1
"""

import sys
import argparse
import json
import yaml
import logging
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
    from logging_config import setup_logging
except ImportError:
    # Fallback if running from proper context
    try:
        from Trading.alpaca_client import AlpacaClient
        from Trading.logging_config import setup_logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SyntheticLong")

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

    defaults = config.get("options", {}).get("synthetic_long", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 35)
    default_prot_pct = defaults.get("default_protection_pct", 5.0)
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Synthetic Long with Protective Put")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of contract sets")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days) after min days, default {default_window}")
    parser.add_argument("--protection-pct", type=float, default=default_prot_pct, help=f"Protective Put distance from spot (%%), default {default_prot_pct}")
    parser.add_argument("--with-protection", action="store_true", help="Enable protective put (default: pure synthetic long)")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount to invest (default ${default_amount})")
    parser.add_argument("--limit-order", action="store_true", help="Use limit order at mid-price instead of market order")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Max bid/ask spread percent (default: 5.0)")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--historical", type=str, help="Path to historical data file (mock mode)")
    parser.add_argument("--underlying", type=str, help="Path to underlying data file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/options)")
    parser.add_argument("--log-file", help="Log file name (default: synthetic_long.log)")
    
    args = parser.parse_args()

    setup_logging(args.log_dir, args.log_file, default_dir='trading_logs/options', default_file='synthetic_long.log')

    # Resolve defaults priority:
    # 1. Explicit Quantity (--quantity 5) -> Ignore amount (even default)
    # 2. Explicit Amount (--amount 1000) -> Calculate quantity
    # 3. Default Amount (from config) -> Calculate quantity
    # 4. Default Quantity (fallback=1)
    
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
                    else: logger.info(msg)
                    return
                if p.get('asset_class') == 'us_option' and len(p['symbol']) >= 15:
                    if p['symbol'][:-15] == symbol:
                        msg = f"Existing option position in {symbol} ({p['symbol']}). Skipping."
                        if args.json: print(json.dumps({"status": "skipped", "reason": msg}))
                        else: logger.info(msg)
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
        prot_strike_target = current_price * (1.0 - (args.protection_pct / 100.0))
    
    if not args.json:
        print(f"Symbol: {symbol}")
        print(f"Current Price: ${current_price:.2f}")
        if prot_strike_target:
            print(f"Targets -> ATM: ${atm_strike_target:.2f}, Prot: ${prot_strike_target:.2f} (-{args.protection_pct}%)")
        else:
            print(f"Targets -> ATM: ${atm_strike_target:.2f}, Prot: None (Pure Synthetic)")

    # 3. Find Options
    # Look for contracts expiring after roughly args.days
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    # Window of opportunity to find liquid expiration
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    if not args.json:
        print(f"Searching for contracts expiring >= {start_date}...")

    # Fetch contracts
    try:
        # Optimization: Narrow search to relevant strikes to avoid fetching thousands of contracts
        # We need strikes around ATM and Protection target
        if prot_strike_target:
            min_strike = prot_strike_target * 0.8
        else:
            min_strike = atm_strike_target * 0.8
        max_strike = atm_strike_target * 1.2

        # Debug: Print parameters for API call
        print("[DEBUG] Fetching option contracts with parameters:", file=sys.stderr)
        print(f"  underlying_symbol: {symbol}", file=sys.stderr)
        print(f"  expiration_date_gte: {start_date}", file=sys.stderr)
        print(f"  expiration_date_lte: {end_date}", file=sys.stderr)
        print(f"  strike_price_gte: {min_strike}", file=sys.stderr)
        print(f"  strike_price_lte: {max_strike}", file=sys.stderr)
        print(f"  limit: 10000", file=sys.stderr)
        print(f"  status: active", file=sys.stderr)

        # Try to print the raw API response if possible
        try:
            raw_response = client.raw_option_contracts_response if hasattr(client, 'raw_option_contracts_response') else None
        except Exception as e:
            raw_response = f"[DEBUG] Could not access raw response: {e}"

        contracts = None
        api_error = None
        try:
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
            api_error = str(e)

        print(f"[DEBUG] Raw API response (if available): {raw_response}", file=sys.stderr)
        if api_error:
            print(f"[DEBUG] Exception from get_option_contracts: {api_error}", file=sys.stderr)
        if contracts is not None:
            print(f"[DEBUG] Number of contracts returned: {len(contracts)}", file=sys.stderr)
            if contracts:
                print(f"[DEBUG] First contract: {json.dumps(contracts[0], indent=2, default=str)}", file=sys.stderr)
            else:
                print("[DEBUG] No contracts returned.", file=sys.stderr)
        else:
            print("[DEBUG] No contracts object returned from API call.", file=sys.stderr)
        if api_error:
            err = {"error": f"Failed to fetch contracts: {api_error}"}
            if args.json:
                print(json.dumps(err))
            else:
                print(f"Error fetching contracts: {api_error}", file=sys.stderr)
            return
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
    # 1. ATM Call
    atm_call = get_closest_contract(exp_contracts, atm_strike_target, 'call')
    # 2. ATM Put
    atm_put = get_closest_contract(exp_contracts, atm_strike_target, 'put')
    # 3. Protective Put
    prot_put = None
    if args.with_protection:
        prot_put = get_closest_contract(exp_contracts, prot_strike_target, 'put')
    
    # Validate legs
    missing = []
    if not atm_call: missing.append("ATM Call")
    if not atm_put: missing.append("ATM Put")
    if args.with_protection and not prot_put: missing.append("Protective Put")
    
    if missing:
        err = {"error": f"Could not find required legs: {', '.join(missing)}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Check for strike logic
    call_strike = float(atm_call['strike_price'])
    put_strike = float(atm_put['strike_price'])
    prot_strike = float(prot_put['strike_price']) if prot_put else 0.0
    
    if not args.json:
        print(f"Leg 1 (Long Call): {atm_call['symbol']} Strike=${call_strike}")
        print(f"Leg 2 (Short Put): {atm_put['symbol']}  Strike=${put_strike}")
        if prot_put:
            print(f"Leg 3 (Prot Put):  {prot_put['symbol']}  Strike=${prot_strike}")
        else:
            print(f"Leg 3 (Prot Put):  None")

    # Build Legs List
    legs = []
    
    if args.dry_run and not args.json:
        print("\n--- Strategy Analysis ---")
        print(f"Syntheitc Long: Buy Call ${call_strike} / Sell Put ${put_strike}")
        if prot_put:
            print(f"Protective Put: Buy Put ${prot_strike}")
            # Max Risk Analysis
            # Risk from Short Put is capped by Long Put
            risk_per_share = max(0, put_strike - prot_strike)
            print(f"Max Downside Risk from Structure (excluding premiums): ${risk_per_share:.2f}/share")
        else:
            print("Protective Put: None")
            print(f"Max Downside Risk from Structure (excluding premiums): ${put_strike:.2f}/share (Stock -> 0)")
        
    # Construct Legs
    # 1. Buy Call
    legs.append({
        "symbol": atm_call['symbol'],
        "side": "buy",
        "position_intent": "buy_to_open",
        "ratio_qty": 1
    })
    
    # 2. Sell Put (only if different from prot put)
    if not prot_put or put_strike != prot_strike:
        legs.append({
            "symbol": atm_put['symbol'],
            "side": "sell",
            "position_intent": "sell_to_open",
            "ratio_qty": 1
        })
    else:
        if not args.json:
            print("Notice: ATM Put and Protective Put strikes are identical. Canceling Short Put leg.")

    # 3. Buy Protective Put
    if prot_put:
        legs.append({
            "symbol": prot_put['symbol'],
            "side": "buy",
            "position_intent": "buy_to_open",
            "ratio_qty": 1
        })
    
    # If put_strike == prot_strike, we have Buy Call + Buy Put (Straddle/Strangle if strikes differ slightly, or just Married Put logic)
    # But here we specifically want Synthetic Long (Short Put + Long Call).
    # If protection is 0%, Short Put and Long Put cancel. We just buy the call.
    
    if prot_put and atm_put['symbol'] == prot_put['symbol']:
        # They cancel perfectly
        legs = [{
            "symbol": atm_call['symbol'],
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
            # Handle single vs multiple return format
            if len(legs) == 1:
                # If only 1 leg, response might be the snapshot dict itself, but get_option_snapshot logic
                # for multiple symbols (comma) returns a dict of snapshots.
                # However, if we pass a single symbol without comma, it returns single snapshot.
                # If we pass "SYM", get_option_snapshot returns snapshot. 
                # If we pass "SYM1,SYM2", it returns {SYM1: ..., SYM2: ...}
                # To be safe, let's normalize.
                if joined_symbols in snapshots: # It's a dict of snapshots
                     pass
                else: # It's a single snapshot object, wrap it
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
                # Use Ask for Buy, Bid for Sell
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
            
            # Max Loss = (CallStrike - ProtStrike) + NetPremium
            # If standard structure: CallStrike == PutStrike
            width = max(0, call_strike - prot_strike) # If prot_strike is 0 (no protection), width is call_strike (stock -> 0)
            metrics['max_loss'] = width + total_premium
            
            # Break Even = CallStrike + NetPremium
            metrics['break_even'] = call_strike + total_premium
            
            # --- Auto-Calculate Quantity if --amount is used ---
            if args.amount and args.amount > 0:
                # Calculate capital required per contract structure
                if args.with_protection:
                    # Defined Risk: Use Max Loss
                    risk_per_contract = metrics['max_loss'] * 100.0
                else:
                    # Undefined Risk (Pure Synthetic): Use Margin Requirement
                    # Reg T Margin for Short Put ~ 20% of Underlying
                    # We use 20% of strike as a proxy + net cost
                    margin_req = (put_strike * 0.20) * 100.0
                    risk_per_contract = margin_req + (metrics['net_cost'] * 100.0)
                
                if risk_per_contract > 0:
                    auto_qty = int(args.amount // risk_per_contract)
                    if auto_qty < 1:
                        msg = f"Warning: Amount ${args.amount} is insufficient for 1 contract (Requires ~${risk_per_contract:.2f})"
                        if not args.json:
                            print(f"\n{msg}")
                        else:
                            # Print to stderr so backtester captures it
                            print(msg, file=sys.stderr)
                        args.quantity = 0 # Will likely fail or just skip
                    else:
                        args.quantity = auto_qty
                        if not args.json:
                            print(f"\nAuto-calculated Quantity: {args.quantity}")
                            print(f"  Based on Amount: ${args.amount}")
                            print(f"  Risk Per Contract: ${risk_per_contract:.2f}")

            if not args.json:
                print("\n--- Financial Analysis (Per Share) ---")
                print(f"Est. Net Premium: ${metrics['net_cost']:.2f} (Natural)")
                print(f"Mid Net Premium:  ${metrics['net_mid_cost']:.2f}")
                if prot_put:
                    print(f"Max Loss Risk:    ${metrics['max_loss']:.2f} (Floor @ ${prot_strike:.2f})")
                else:
                    print(f"Max Loss Risk:    ${metrics['max_loss']:.2f} (No Floor)")
                max_loss_ratio = metrics['max_loss'] / current_price if current_price > 0 else 0
                print(f"Max Loss/Price:   {max_loss_ratio:.1%}")
                if metrics['net_cost'] > 0:
                    loss_prem_ratio = metrics['max_loss'] / metrics['net_cost']
                    print(f"Max Loss/Prem:    {loss_prem_ratio:.2f}x")
                print(f"Break Even:       ${metrics['break_even']:.2f}")
                print(f"Position Delta:   {metrics['net_delta']:.2f}")
                if prot_put:
                    print(f"Cap Efficiency:   Floor protection cost = ${metrics['net_cost']:.2f} vs Stock Price ${current_price:.2f}")
                else:
                    print(f"Cap Efficiency:   Synthetic cost = ${metrics['net_cost']:.2f} vs Stock Price ${current_price:.2f}")
                print(f"Max Leg Spread:   {metrics['max_leg_spread']:.2%}")

            if warnings and not args.json:
                print("\n⚠️  Liquidity Warnings:")
                for w in warnings:
                    print(f"  - {w}")

    except Exception as e:
        if not args.json:
            print(f"Warning: Could not calculate metrics: {e}", file=sys.stderr)
        metrics['error'] = str(e)

    # --- Generate Email Text ---
    email_lines = []
    email_lines.append(f"Synthetic Long Strategy: {symbol}")
    email_lines.append("=" * 30)
    email_lines.append(f"Date: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}")
    if current_price > 0:
        email_lines.append(f"Current Price: ${current_price:.2f}")
    email_lines.append(f"Expiration: {selected_exp}")
    email_lines.append("")
    email_lines.append("Structure:")
    for leg in legs:
        price = leg.get('estimated_price', 0)
        sym = leg['symbol']
        snap = metrics.get('contracts_data', {}).get(sym, {})
        quote = snap.get('latestQuote', {})
        ask = float(quote.get('ap') or 0)
        bid = float(quote.get('bp') or 0)
        last = float(snap.get('latestTrade', {}).get('p') or 0)
        mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else last
        spread = ask - bid if (ask > 0 and bid > 0) else 0
        spread_pct = (spread / mid) if (mid > 0) else 0
        price_str = f"~${price:.2f} (Bid: ${bid:.2f}, Ask: ${ask:.2f}, Mid: ${mid:.2f}, Spread: ${spread:.2f}, Spread%: {spread_pct:.2%})"
        email_lines.append(f"- {leg.get('position_intent', leg.get('side'))}: {leg['symbol']} ({price_str})")
        # Print to console as well
        if not args.json:
            print(f"  {leg.get('position_intent', leg.get('side'))}: {leg['symbol']}\n    Price: ${price:.2f}\n    Bid:   ${bid:.2f}\n    Ask:   ${ask:.2f}\n    Mid:   ${mid:.2f}\n    Spread: ${spread:.2f} ({spread_pct:.2%})")
    email_lines.append("")
    email_lines.append("Financial Analysis (Per Share):")
    email_lines.append(f"Est. Net Premium: ${metrics.get('net_cost', 0):.2f}")
    if metrics.get('max_loss'):
        email_lines.append(f"Max Loss Risk:    ${metrics.get('max_loss', 0):.2f}")
        max_loss_ratio = metrics['max_loss'] / current_price if current_price > 0 else 0
        email_lines.append(f"Max Loss/Price:   {max_loss_ratio:.1%}")
        if metrics.get('net_cost', 0) > 0:
             loss_prem_ratio = metrics['max_loss'] / metrics['net_cost']
             email_lines.append(f"Max Loss/Prem:    {loss_prem_ratio:.2f}x")
    if metrics.get('break_even'):
        email_lines.append(f"Break Even:       ${metrics.get('break_even', 0):.2f}")
    if metrics.get('net_delta'):
        email_lines.append(f"Position Delta:   {metrics.get('net_delta', 0):.2f}")
    email_lines.append("")
    email_lines.append(f"Target Quantity: {args.quantity}")
    
    if metrics.get('error'):
        email_lines.append("")
        email_lines.append(f"⚠️ Metrics Warning: {metrics.get('error')}")

    email_text_base = "\n".join(email_lines)

    # Save plan to log
    try:
        log_dir = Path("trading_logs/strategies")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"synthetic_long_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, "w") as f:
            f.write(email_text_base)
            f.write(f"\n\nContext: Dry Run={args.dry_run}, Limit={args.limit_order}")
        if not args.json:
            print(f"\n📄 Plan Summary:  {log_file}")
    except Exception:
        pass

    if args.dry_run:
        # Add per-leg bid/ask/mid/spread info to each leg in JSON output
        detailed_legs = []
        for leg in legs:
            sym = leg['symbol']
            snap = metrics.get('contracts_data', {}).get(sym, {})
            quote = snap.get('latestQuote', {})
            ask = float(quote.get('ap') or 0)
            bid = float(quote.get('bp') or 0)
            last = float(snap.get('latestTrade', {}).get('p') or 0)
            mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else last
            spread = ask - bid if (ask > 0 and bid > 0) else 0
            spread_pct = (spread / mid) if (mid > 0) else 0
            detailed_leg = dict(leg)
            detailed_leg.update({
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_pct": spread_pct
            })
            detailed_legs.append(detailed_leg)
        result = {
            "status": "dry_run",
            "scan": {
                "current_price": current_price,
                "expiration": selected_exp,
            },
            "legs": detailed_legs,
            "metrics": metrics,
            "email_text": email_text_base + "\n\nStatus: Dry Run (No Order Submitted)",
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
        logger.info(f"Submitting {order_type} order for {args.quantity}x structures...")
        
    try:
        if args.limit_order:
            limit_price = round(metrics['net_mid_cost'], 2)
            # Pass entry_cash_flow for mock orders (Debit = negative cash flow for Long strategies usually,
            # but here net_cost is positive for debit. entry_cash_flow should be negative for debit.)
            entry_cash_flow = -limit_price
            kwargs = {'legs': legs, 'quantity': args.quantity, 'limit_price': limit_price, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow
            
            response = client.place_option_limit_order(**kwargs)
        else:
            # Determine order class: 'mleg' for multi-leg strategies.
            # Pass entry_cash_flow for mock orders (Debit = negative cash flow)
            entry_cash_flow = -metrics['net_cost']
            kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow

            response = client.place_option_market_order(**kwargs)
        
        result = {
            "status": "executed",
            "order": response,
            "email_text": email_text_base + f"\n\nStatus: Order Submitted\nOrder ID: {response.get('id')}\nAlpaca Order Status: {response.get('status')}",
            "warnings": warnings
        }
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            logger.info(f"Order Submitted Successfully!")
            logger.info(f"Order ID: {response.get('id')}")
            logger.info(f"Status: {response.get('status')}")
            # print(json.dumps(response, indent=2))
    except Exception as e:
        err = {"error": str(e), "email_text": email_text_base + f"\n\n❌ Execution Error: {str(e)}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

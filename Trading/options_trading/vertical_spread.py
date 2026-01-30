#!/usr/bin/env python3
"""
Creates a Vertical Spread position (Bull/Bear, Call/Put).

Strategies:
1. Bull Call Spread (Debit): Buy Lower Call, Sell Higher Call. (Bullish)
2. Bear Call Spread (Credit): Sell Lower Call, Buy Higher Call. (Bearish)
3. Bull Put Spread (Credit): Buy Lower Put, Sell Higher Put. (Bullish)
4. Bear Put Spread (Debit): Sell Lower Put, Buy Higher Put. (Bearish)

Usage:
    python util/vertical_spread.py SPY --type call --direction bull --width 5 --quantity 1
    python util/vertical_spread.py QQQ --type put --direction bear --anchor-pct 0 --width 10
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
logger = logging.getLogger("VerticalSpread")

def get_closest_contract(contracts: List[Dict], target_strike: float, option_type: str) -> Optional[Dict]:
    """Finds the contract with strike price closest to target."""
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
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

    defaults = config.get("options", {}).get("vertical_spread", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 14)
    default_width = defaults.get("default_width", 5.0)
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Vertical Spread Strategy")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--type", choices=['call', 'put'], required=True, help="Option type (call/put)")
    parser.add_argument("--direction", choices=['bull', 'bear'], required=True, help="Strategy direction (bull/bear)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of spreads")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days), default {default_window}")
    parser.add_argument("--width", type=float, default=default_width, help=f"Spread width in dollars, default ${default_width}")
    parser.add_argument("--anchor-pct", type=float, default=0.0, help="Distance of anchor leg from spot (%%). Positive = OTM for Credit, ITM for Debit (usually). Default 0 (ATM).")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount (max risk)")
    parser.add_argument("--limit-order", action="store_true", help="Use limit order at mid-price")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--historical", type=str, help="Path to historical data file (mock mode)")
    parser.add_argument("--underlying", type=str, help="Path to underlying data file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/options)")
    parser.add_argument("--log-file", help="Log file name (default: vertical_spread.log)")
    
    args = parser.parse_args()

    setup_logging(args.log_dir, args.log_file, default_dir='trading_logs/options', default_file='vertical_spread.log')

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

    # 2. Determine Strikes
    # Logic:
    # Call Spreads: Strikes are usually above price (Bear) or around price (Bull)
    # Put Spreads: Strikes are usually below price (Bull) or around price (Bear)
    
    # Anchor Leg: The leg closer to the money (or the one defining the entry)
    # For Bull Call (Debit): Buy Leg (Lower) is Anchor.
    # For Bear Call (Credit): Sell Leg (Lower) is Anchor.
    # For Bull Put (Credit): Sell Leg (Higher) is Anchor.
    # For Bear Put (Debit): Buy Leg (Higher) is Anchor.
    
    # Wait, standard convention for "Anchor" usually implies the Short leg for Credit, Long leg for Debit?
    # Let's use "Lower Strike" and "Higher Strike" logic.
    
    # Anchor Pct:
    # If Call: Target = Price * (1 + pct/100)
    # If Put:  Target = Price * (1 - pct/100)
    
    # But for Bull Put (Credit), we sell OTM Put (Below price). So pct should be positive to mean "Below".
    # Let's simplify:
    # Call: Anchor = Price * (1 + pct/100)
    # Put:  Anchor = Price * (1 - pct/100)
    
    anchor_strike = current_price
    if args.type == 'call':
        anchor_strike = current_price * (1.0 + (args.anchor_pct / 100.0))
    else:
        anchor_strike = current_price * (1.0 - (args.anchor_pct / 100.0))
        
    # Determine Second Leg
    # Call: Higher Strike = Anchor + Width
    # Put:  Lower Strike = Anchor - Width
    
    if args.type == 'call':
        strike1 = anchor_strike
        strike2 = anchor_strike + args.width
        # Bull Call: Buy Low (1), Sell High (2)
        # Bear Call: Sell Low (1), Buy High (2)
        if args.direction == 'bull':
            long_target = strike1
            short_target = strike2
        else:
            short_target = strike1
            long_target = strike2
    else: # Put
        strike1 = anchor_strike
        strike2 = anchor_strike - args.width
        # Bull Put: Sell High (1), Buy Low (2)
        # Bear Put: Buy High (1), Sell Low (2)
        if args.direction == 'bull':
            short_target = strike1
            long_target = strike2
        else:
            long_target = strike1
            short_target = strike2

    if not args.json:
        logger.info(f"Symbol: {symbol} (${current_price:.2f})")
        logger.info(f"Strategy: {args.direction.title()} {args.type.title()} Spread")
        logger.info(f"Targets: Long ${long_target:.2f}, Short ${short_target:.2f}")

    # 3. Find Options
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    try:
        min_strike = min(long_target, short_target) * 0.9
        max_strike = max(long_target, short_target) * 1.1
        
        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            type=args.type,
            limit=10000,
            status='active'
        )
    except Exception as e:
        err = {"error": f"Failed to fetch contracts: {e}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    if not contracts:
        err = {"error": "No contracts found"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Group by expiration and select best
    expirations = {}
    for c in contracts:
        exp = c['expiration_date']
        if exp not in expirations: expirations[exp] = []
        expirations[exp].append(c)
    
    sorted_exps = sorted(expirations.keys())
    # Prefer Friday
    friday_exps = [e for e in sorted_exps if datetime.strptime(e, "%Y-%m-%d").weekday() == 4]
    selected_exp = friday_exps[0] if friday_exps else sorted_exps[0]
    
    exp_contracts = expirations[selected_exp]
    
    long_leg = get_closest_contract(exp_contracts, long_target, args.type)
    short_leg = get_closest_contract(exp_contracts, short_target, args.type)
    
    if not long_leg or not short_leg:
        err = {"error": "Could not find suitable legs"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Fetch Prices
    joined = f"{long_leg['symbol']},{short_leg['symbol']}"
    snapshots = client.get_option_snapshot(joined)
    if 'latestQuote' in snapshots: snapshots = {joined: snapshots}
    
    def get_price(sym, side):
        snap = snapshots.get(sym, {})
        quote = snap.get('latestQuote', {})
        bid = float(quote.get('bp') or 0)
        ask = float(quote.get('ap') or 0)
        last = float(snap.get('latestTrade', {}).get('p') or 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
        
        if args.limit_order: return mid
        return ask if side == 'buy' else bid if bid > 0 else last

    long_price = get_price(long_leg['symbol'], 'buy')
    short_price = get_price(short_leg['symbol'], 'sell')
    
    net_cost = long_price - short_price # Positive = Debit, Negative = Credit
    
    legs = [
        {"symbol": long_leg['symbol'], "side": "buy", "position_intent": "buy_to_open", "ratio_qty": 1, "estimated_price": long_price},
        {"symbol": short_leg['symbol'], "side": "sell", "position_intent": "sell_to_open", "ratio_qty": 1, "estimated_price": short_price}
    ]
    
    if args.amount and args.amount > 0:
        # Risk:
        # Debit Spread: Risk = Net Debit
        # Credit Spread: Risk = Width - Net Credit
        if net_cost > 0: # Debit
            risk = net_cost * 100
        else: # Credit
            width = abs(float(long_leg['strike_price']) - float(short_leg['strike_price']))
            risk = (width - abs(net_cost)) * 100
            
        if risk > 0:
            args.quantity = max(1, int(args.amount // risk))

    if not args.json:
        logger.info(f"Expiration: {selected_exp}")
        logger.info(f"Long:  {long_leg['symbol']} (${long_price:.2f})")
        logger.info(f"Short: {short_leg['symbol']} (${short_price:.2f})")
        logger.info(f"Net {'Debit' if net_cost > 0 else 'Credit'}: ${abs(net_cost):.2f}")
        logger.info(f"Quantity: {args.quantity}")

    if args.dry_run:
        if args.json:
            print(json.dumps({"status": "dry_run", "legs": legs, "net_cost": net_cost}, indent=2))
        return

    try:
        entry_cash_flow = -net_cost # Debit is negative cash flow
        kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
        
        if args.limit_order:
             kwargs['limit_price'] = round(abs(net_cost), 2)
             # kwargs['type'] = 'limit' # Removed: place_option_limit_order sets this internally
        else:
             # kwargs['type'] = 'market' # Removed: place_option_market_order sets this internally
             pass

        if isinstance(client, MockOptionClient):
            kwargs['entry_cash_flow'] = entry_cash_flow
            kwargs['underlying_price'] = current_price
            response = client.place_option_limit_order(**kwargs) if args.limit_order else client.place_option_market_order(**kwargs)
        else:
            response = client.place_option_limit_order(**kwargs) if args.limit_order else client.place_option_market_order(**kwargs)
            
        if args.json:
            print(json.dumps({"status": "executed", "order": response}, indent=2))
        else:
            logger.info(f"Order Submitted: {response.get('id')}")
            
    except Exception as e:
        err = {"error": str(e)}
        if args.json: print(json.dumps(err))
        else: print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
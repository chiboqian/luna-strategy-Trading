#!/usr/bin/env python3
"""
Creates a Long Stock position for a given symbol.

Structure:
1. Buy Stock (1 Leg)

Usage:
    python util/long_stock.py SPY --quantity 100
    python util/long_stock.py AAPL --amount 10000 --dry-run
"""

import sys
import argparse
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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
logger = logging.getLogger("LongStock")

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

    defaults = config.get("options", {}).get("long_stock", {})
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Long Stock Strategy")
    parser.add_argument("symbol", help="Stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of shares")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount (max risk)")
    parser.add_argument("--limit-order", action="store_true", help="Use limit order")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--historical", type=str, help="Path to historical data file (mock mode)")
    parser.add_argument("--underlying", type=str, help="Path to underlying data file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/options)")
    parser.add_argument("--log-file", help="Log file name (default: long_stock.log)")
    
    args = parser.parse_args()

    setup_logging(args.log_dir, args.log_file, default_dir='trading_logs/options', default_file='long_stock.log')

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

    # For long_stock, we primarily need underlying data. 
    # Prioritize args.underlying to avoid loading large options files from args.historical
    primary_file = args.underlying if args.underlying else args.historical
    if primary_file:
        secondary_file = None # No need for secondary if we have what we need
        client = MockOptionClient(primary_file, reference_date, args.save_order, secondary_file)
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
    bid = 0.0
    ask = 0.0

    try:
        if hasattr(client, 'get_stock_latest_quote'):
            quote = client.get_stock_latest_quote(symbol)
            if quote:
                bid = float(quote.get('bp', 0) or 0)
                ask = float(quote.get('ap', 0) or 0)
    except Exception:
        pass

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
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Current Price: ${current_price:.2f}")

    # Determine execution price
    exec_price = current_price
    if args.limit_order:
        if bid > 0 and ask > 0:
            exec_price = (bid + ask) / 2
        elif current_price > 0:
            exec_price = current_price
    
    # Auto-Quantity
    if args.amount and args.amount > 0:
        if exec_price > 0:
            args.quantity = max(1, int(args.amount // exec_price))

    if not args.json:
        logger.info(f"Quantity: {args.quantity}")
        logger.info(f"Price:    ${exec_price:.2f} ({'Limit/Mid' if args.limit_order else 'Market'})")
        logger.info(f"Total Cost: ${exec_price * args.quantity:.2f}")

    legs = [{
        "symbol": symbol,
        "side": "buy",
        "qty": args.quantity,
        "estimated_price": exec_price
    }]

    if args.dry_run:
        if args.json:
            print(json.dumps({"status": "dry_run", "legs": legs, "cost": exec_price * args.quantity}, indent=2))
        else:
            logger.info("Dry Run Complete.")
        return

    # Execute
    try:
        # Entry cash flow should be per-unit (per share) for consistency with closing logic
        entry_cash_flow = -1 * exec_price
        
        # For mock client, we simulate the order placement since place_stock_order might not be mocked
        if isinstance(client, MockOptionClient):
            response = {
                "id": "mock_stock_order",
                "status": "filled",
                "asset_class": "us_equity",
                "filled_at": reference_date.isoformat(),
                "symbol": symbol,
                "quantity": args.quantity,
                "filled_avg_price": exec_price,
                "side": "buy",
                "legs": legs, # Include legs for close_mock_order compatibility
                "entry_cash_flow": entry_cash_flow,
                "underlying_price": current_price
            }
            if args.save_order:
                with open(args.save_order, 'w') as f:
                    json.dump(response, f, indent=2)
                if not args.json:
                    logger.info(f"Mock order saved to {args.save_order}")
        else:
            response = client.place_stock_order(
                symbol=symbol, 
                side="buy", 
                quantity=args.quantity, 
                order_type="limit" if args.limit_order else "market",
                limit_price=exec_price if args.limit_order else None,
                time_in_force="day"
            )
            
        if args.json:
            print(json.dumps({"status": "executed", "order": response}, indent=2))
        else:
            logger.info(f"Order Submitted: {response.get('id')}")
            logger.info(f"Purchased Share Price: ${exec_price:.2f}")
            logger.info(f"Total Cost: ${exec_price * args.quantity:.2f}")
            
    except Exception as e:
        err = {"error": str(e)}
        if args.json: print(json.dumps(err))
        else: print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
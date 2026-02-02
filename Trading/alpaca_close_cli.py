#!/usr/bin/env python3
"""
CLI to close positions for a list of stocks or options via Alpaca.

Usage:
    python Trading/alpaca_close_cli.py AAPL MSFT
    python Trading/alpaca_close_cli.py --all
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import List

# Add parent directory to path to import Trading modules if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from Trading.alpaca_client import AlpacaClient
    from Trading.logging_config import setup_logging
except ImportError:
    # Fallback if running from root or if Trading package is not resolved as expected
    try:
        # If we are in Trading/ and Trading/Trading exists as a package
        from Trading.alpaca_client import AlpacaClient
        from Trading.logging_config import setup_logging
    except ImportError:
        print("Error: Could not import AlpacaClient. Check python path.", file=sys.stderr)
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlpacaCloseCLI")

def main():
    parser = argparse.ArgumentParser(
        description="Close positions for specific symbols or all positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("symbols", nargs="*", help="List of symbols to close (e.g. AAPL MSFT)")
    parser.add_argument("--all", action="store_true", help="Close ALL open positions")
    parser.add_argument("--qty", type=float, help="Quantity to close (for single symbol only)")
    parser.add_argument("--pct", type=float, help="Percentage to close (e.g. 50 for 50%)")
    parser.add_argument("--cancel-orders", action="store_true", help="Cancel open orders for these symbols before closing")
    parser.add_argument("--scan-rules", action="store_true", help="Scan streaming_rules directory for symbols to close")
    parser.add_argument("--rules-dir", help="Directory containing streaming rules (default: Trading/config/streaming_rules)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without executing")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/cli)")
    parser.add_argument("--log-file", help="Log file name (default: alpaca_close.log)")

    args = parser.parse_args()

    setup_logging(args.log_dir, args.log_file, default_dir='trading_logs/cli', default_file='alpaca_close.log')

    if args.scan_rules:
        rules_dir = args.rules_dir
        if not rules_dir:
            # Try relative to script location
            script_path = Path(__file__).resolve()
            default_path = script_path.parent / "config" / "streaming_rules"
            
            if default_path.exists():
                rules_dir = default_path
            else:
                # Fallback to cwd
                default_path_cwd = Path("Trading/config/streaming_rules")
                if default_path_cwd.exists():
                    rules_dir = default_path_cwd
        
        if rules_dir:
            rules_path = Path(rules_dir)
            if rules_path.exists() and rules_path.is_dir():
                logger.info(f"Scanning rules in {rules_path}...")
                found_symbols = set()
                for yaml_file in rules_path.glob("*.yaml"):
                    try:
                        with open(yaml_file, 'r') as f:
                            data = yaml.safe_load(f)
                            if not data: continue
                            
                            rules_list = data.get('rules', []) if isinstance(data, dict) and 'rules' in data else [data] if isinstance(data, dict) else data if isinstance(data, list) else []
                                
                            for rule in rules_list:
                                if isinstance(rule, dict) and 'symbol' in rule:
                                    found_symbols.add(rule['symbol'].upper())
                    except Exception as e:
                        logger.warning(f"Error reading {yaml_file}: {e}")
                
                if found_symbols:
                    logger.info(f"Found symbols in rules: {', '.join(found_symbols)}")
                    if not args.symbols:
                        args.symbols = []
                    for s in found_symbols:
                        if s not in args.symbols:
                            args.symbols.append(s)
            else:
                logger.warning(f"Rules directory {rules_path} not found or not a directory.")
        else:
            logger.warning("Could not locate streaming_rules directory.")

    if not args.symbols and not args.all:
        parser.error("Must specify symbols, --all, or --scan-rules")

    if args.all and args.symbols:
        parser.error("Cannot specify both symbols (or --scan-rules) and --all")

    if (args.qty or args.pct) and (args.all or len(args.symbols) > 1):
        parser.error("--qty and --pct can only be used with a single symbol")

    client = AlpacaClient()
    results = []

    # Path 1: Close All
    if args.all:
        if args.dry_run:
            logger.info("[Dry Run] Would close ALL positions and cancel orders if requested.")
            results.append({"status": "dry_run", "action": "close_all"})
        else:
            try:
                # Alpaca API close_all_positions takes cancel_orders boolean
                responses = client.close_all_positions(cancel_orders=args.cancel_orders)
                for resp in responses:
                    sym = resp.get('symbol')
                    logger.info(f"Close order submitted for {sym}: {resp.get('id')}")
                    results.append({"symbol": sym, "status": "submitted", "order": resp})
                if not responses:
                    logger.info("No positions found to close.")
            except Exception as e:
                logger.error(f"Failed to close all positions: {e}")
                results.append({"status": "failed", "error": str(e)})
        
        if args.json:
            print(json.dumps(results, indent=2))
        return

    # Path 2: Close Specific Symbols
    for symbol in args.symbols:
        symbol = symbol.upper()
        
        if args.cancel_orders:
            if not args.dry_run:
                try:
                    client.cancel_orders_for_symbol(symbol)
                    logger.info(f"Canceled open orders for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cancel orders for {symbol}: {e}")
            else:
                logger.info(f"[Dry Run] Would cancel open orders for {symbol}")

        if args.dry_run:
            logger.info(f"[Dry Run] Would close position: {symbol} (Qty: {args.qty}, Pct: {args.pct})")
            results.append({"symbol": symbol, "status": "dry_run"})
            continue

        try:
            logger.info(f"Closing position: {symbol}...")
            response = client.close_position(symbol, qty=args.qty, percentage=args.pct)
            logger.info(f"Close order submitted for {symbol}: {response.get('id')}")
            results.append({"symbol": symbol, "status": "submitted", "order": response})
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                logger.warning(f"Position not found for {symbol}")
                results.append({"symbol": symbol, "status": "not_found"})
            else:
                logger.error(f"Failed to close {symbol}: {error_msg}")
                results.append({"symbol": symbol, "status": "failed", "error": error_msg})

    if args.json:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
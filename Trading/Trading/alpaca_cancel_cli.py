#!/usr/bin/env python3
"""
CLI to cancel all open orders for a given symbol via Alpaca.
"""

import argparse
import sys

from alpaca_client import AlpacaClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cancel all open orders for a stock symbol (Alpaca)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("symbol", help="Stock symbol to cancel orders for (e.g., AAPL)")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information for each canceled order"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbol = args.symbol.upper()

    # Initialize client
    try:
        client = AlpacaClient()
    except Exception as e:
        print(f"❌ Failed to initialize Alpaca client: {e}", file=sys.stderr)
        sys.exit(1)

    # Cancel all orders for the symbol
    try:
        canceled = client.cancel_orders_for_symbol(symbol)
        
        if not canceled:
            print(f"ℹ️ No open orders found for {symbol}")
            sys.exit(0)
        
        success_count = sum(1 for c in canceled if c.get('status') == 'canceled')
        failed_count = len(canceled) - success_count
        
        print(f"✅ Canceled {success_count} order(s) for {symbol}")
        if failed_count > 0:
            print(f"⚠️ Failed to cancel {failed_count} order(s)")
        
        if args.verbose:
            print("\nDetails:")
            for c in canceled:
                status_icon = "✓" if c.get('status') == 'canceled' else "✗"
                print(f"   {status_icon} Order {c['id']}: {c['status']}")
                if c.get('error'):
                    print(f"      Error: {c['error']}")
                    
    except Exception as e:
        print(f"❌ Error canceling orders: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

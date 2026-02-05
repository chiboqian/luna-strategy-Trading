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
    parser.add_argument("symbol", nargs="?", help="Stock symbol to cancel orders for (e.g., AAPL)")
    parser.add_argument("--all", action="store_true", help="Cancel ALL open orders for all symbols")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate cancellation without executing"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information for each canceled order"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.symbol and not args.all:
        print("Error: Must specify a symbol or --all", file=sys.stderr)
        sys.exit(1)

    # Initialize client
    try:
        client = AlpacaClient()
    except Exception as e:
        print(f"❌ Failed to initialize Alpaca client: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch positions to filter orders
    try:
        positions = client.get_all_positions()
        position_symbols = {p['symbol'] for p in positions}
    except Exception as e:
        print(f"❌ Failed to fetch positions: {e}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        try:
            # Fetch all open orders (limit 500 to catch most)
            orders = client.get_orders(status='open', limit=500)
            
            # Filter out orders for symbols where we have a position
            to_cancel = [o for o in orders if o['symbol'] not in position_symbols]
            skipped_count = len(orders) - len(to_cancel)
            
            if not to_cancel:
                print(f"ℹ️ No orders to cancel (Skipped {skipped_count} orders for symbols with open positions)")
                sys.exit(0)
            
            if args.dry_run:
                print(f"ℹ️ [DRY RUN] Would cancel {len(to_cancel)} orders (Skipping {skipped_count} with open positions):")
                for order in to_cancel:
                    print(f"   [DRY RUN] Would cancel {order['symbol']} order {order['id']}")
                sys.exit(0)
            
            print(f"Canceling {len(to_cancel)} orders (Skipping {skipped_count} with open positions)...")
            
            success_count = 0
            for order in to_cancel:
                try:
                    client.cancel_order_by_id(order['id'])
                    success_count += 1
                    if args.verbose:
                        print(f"   ✓ Canceled {order['symbol']} order {order['id']}")
                except Exception as e:
                    print(f"   ❌ Failed to cancel {order['symbol']} order {order['id']}: {e}", file=sys.stderr)
            
            print(f"✅ Canceled {success_count} orders")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error canceling all orders: {e}", file=sys.stderr)
            sys.exit(1)

    symbol = args.symbol.upper()
    
    if symbol in position_symbols:
        print(f"⚠️ Skipping cancellation for {symbol}: Open position exists.")
        sys.exit(0)

    if args.dry_run:
        try:
            orders = client.get_orders(status='open', symbols=[symbol])
            symbol_orders = [o for o in orders if o.get('symbol', '').upper() == symbol]
            
            if not symbol_orders:
                print(f"ℹ️ [DRY RUN] No open orders found for {symbol}")
                sys.exit(0)
                
            print(f"ℹ️ [DRY RUN] Would cancel {len(symbol_orders)} order(s) for {symbol}:")
            for order in symbol_orders:
                print(f"   [DRY RUN] Would cancel Order {order['id']}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error fetching orders: {e}", file=sys.stderr)
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

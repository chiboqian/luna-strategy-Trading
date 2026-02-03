#!/usr/bin/env python3
"""
Get account summary and open positions.
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import Trading modules if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from alpaca_client import AlpacaClient
except ImportError:
    # Fallback if running from root
    try:
        from Trading.alpaca_client import AlpacaClient
    except ImportError:
        print("Error: Could not import AlpacaClient. Check python path.", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Get account summary and positions")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    try:
        client = AlpacaClient()
        account = client.get_account_info()
        positions = client.get_all_positions()
        
        # Calculate total P/L
        total_pl = sum(float(p.get('unrealized_pl', 0)) for p in positions)
        total_pl_pct = 0.0
        equity = float(account.get('equity', 0))
        last_equity = float(account.get('last_equity', 0))
        if last_equity > 0:
            total_pl_pct = (equity - last_equity) / last_equity

        summary = {
            "account": account,
            "positions": positions,
            "metrics": {
                "total_pl": total_pl,
                "day_change_pct": total_pl_pct
            }
        }
        
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Equity: ${equity:,.2f}")
            print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
            print(f"Day Change: {total_pl_pct:.2%}")
            print(f"Positions: {len(positions)}")
            for p in positions:
                print(f"  {p['symbol']}: {p['qty']} @ ${float(p['avg_entry_price']):.2f} (P/L: ${float(p['unrealized_pl']):.2f})")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Get account summary and positions.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure Trading module import path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Trading'))
from alpaca_client import AlpacaClient

def main():
    parser = argparse.ArgumentParser(description='Get account summary and positions')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    try:
        client = AlpacaClient()
        account = client.get_account_info()
        positions = client.get_all_positions()
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Format email text
    lines = []
    lines.append("Account Summary Report")
    lines.append("======================")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("Account Overview:")
    lines.append("-----------------")
    equity = float(account.get('equity', 0))
    cash = float(account.get('cash', 0))
    buying_power = float(account.get('buying_power', 0))
    
    # Calculate P/L
    last_equity = float(account.get('last_equity', 0))
    pl = equity - last_equity
    pl_pct = (pl / last_equity) * 100 if last_equity != 0 else 0
    
    lines.append(f"Equity:       ${equity:,.2f}")
    lines.append(f"Cash:         ${cash:,.2f}")
    lines.append(f"Buying Power: ${buying_power:,.2f}")
    lines.append(f"Today's P/L:  ${pl:,.2f} ({pl_pct:+.2f}%)")
    lines.append("")

    lines.append(f"Positions ({len(positions)}):")
    lines.append("-----------------")
    
    if not positions:
        lines.append("No open positions.")
    else:
        # Sort by symbol
        positions.sort(key=lambda x: x.get('symbol'))
        
        for pos in positions:
            symbol = pos.get('symbol')
            qty = float(pos.get('qty', 0))
            avg_entry = float(pos.get('avg_entry_price', 0))
            current_price = float(pos.get('current_price', 0))
            market_value = float(pos.get('market_value', 0))
            unrealized_pl = float(pos.get('unrealized_pl', 0))
            unrealized_plpc = float(pos.get('unrealized_plpc', 0)) * 100
            
            lines.append(f"{symbol:<5} {qty:>6} shares @ ${avg_entry:>7.2f} | Curr: ${current_price:>7.2f} | Val: ${market_value:>9.2f} | P/L: ${unrealized_pl:>8.2f} ({unrealized_plpc:>+6.2f}%)")

    email_text = "\n".join(lines)

    result = {
        "account": account,
        "positions": positions,
        "email_text": email_text
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(email_text)

if __name__ == '__main__':
    main()

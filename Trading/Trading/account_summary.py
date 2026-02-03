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

def generate_html_report(account, positions, metrics):
    equity = float(account.get('equity', 0))
    buying_power = float(account.get('buying_power', 0))
    
    html = [
        "<html>",
        "<head>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        .positive { color: green; }",
        "        .negative { color: red; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h2>Account Summary</h2>",
        f"    <p><strong>Equity:</strong> ${equity:,.2f}</p>",
        f"    <p><strong>Buying Power:</strong> ${buying_power:,.2f}</p>",
        f"    <p><strong>Day Change:</strong> <span class=\"{'positive' if metrics['day_change_pct'] >= 0 else 'negative'}\">{metrics['day_change_pct']:.2%}</span></p>",
        f"    <p><strong>Total Unrealized P/L:</strong> <span class=\"{'positive' if metrics['total_pl'] >= 0 else 'negative'}\">${metrics['total_pl']:,.2f}</span></p>",
        f"    <h3>Positions ({len(positions)})</h3>",
        "    <table>",
        "        <tr>",
        "            <th>Symbol</th>",
        "            <th>Qty</th>",
        "            <th>Avg Entry</th>",
        "            <th>Current Price</th>",
        "            <th>Market Value</th>",
        "            <th>Unrealized P/L</th>",
        "        </tr>"
    ]
    
    for p in positions:
        pl = float(p.get('unrealized_pl', 0))
        pl_class = "positive" if pl >= 0 else "negative"
        html.append("        <tr>")
        html.append(f"            <td>{p['symbol']}</td>")
        html.append(f"            <td>{p['qty']}</td>")
        html.append(f"            <td>${float(p.get('avg_entry_price', 0)):.2f}</td>")
        html.append(f"            <td>${float(p.get('current_price', 0)):.2f}</td>")
        html.append(f"            <td>${float(p.get('market_value', 0)):.2f}</td>")
        html.append(f"            <td class=\"{pl_class}\">${pl:.2f}</td>")
        html.append("        </tr>")
    
    html.append("    </table>")
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)

def generate_text_report(account, positions, metrics):
    equity = float(account.get('equity', 0))
    buying_power = float(account.get('buying_power', 0))
    
    lines = []
    lines.append("Account Summary")
    lines.append("===============")
    lines.append(f"Equity:       ${equity:,.2f}")
    lines.append(f"Buying Power: ${buying_power:,.2f}")
    lines.append(f"Day Change:   {metrics['day_change_pct']:.2%}")
    lines.append(f"Total P/L:    ${metrics['total_pl']:,.2f}")
    lines.append("")
    lines.append(f"Positions ({len(positions)})")
    lines.append("-" * 65)
    
    if not positions:
        lines.append("No open positions.")
    else:
        # Header
        lines.append(f"{'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P/L':>12} {'P/L %':>8}")
        lines.append("-" * 65)
        for p in positions:
            symbol = p['symbol']
            qty = p['qty']
            avg_entry = float(p.get('avg_entry_price', 0))
            current = float(p.get('current_price', 0))
            pl = float(p.get('unrealized_pl', 0))
            pl_pct = float(p.get('unrealized_plpc', 0))
            lines.append(f"{symbol:<8} {qty:>6} {avg_entry:>10.2f} {current:>10.2f} {pl:>12.2f} {pl_pct:>8.2%}")
            
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Get account summary and positions")
    parser.add_argument("--json", action="store_true", help="Output JSON (deprecated, use --format json)")
    parser.add_argument("--format", choices=['text', 'json', 'html'], default='text', help="Output format")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    if args.json:
        args.format = 'json'

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
            
        metrics = {
            "total_pl": total_pl,
            "day_change_pct": total_pl_pct
        }

        output_content = ""

        if args.format == 'json':
            # Generate text report to include in JSON for convenience
            text_report = generate_text_report(account, positions, metrics)
            summary = {
                "account": account,
                "positions": positions,
                "metrics": metrics,
                "email_text": text_report
            }
            output_content = json.dumps(summary, indent=2)
        
        elif args.format == 'html':
            output_content = generate_html_report(account, positions, metrics)
            
        else: # text
            output_content = generate_text_report(account, positions, metrics)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_content)
            print(f"Report saved to {args.output}")
        else:
            print(output_content)

    except Exception as e:
        if args.format == 'json':
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Lists all open option positions for a given underlying symbol.
Supports checking against Take Profit (TP) and Stop Loss (SL) thresholds.

Usage:
    python options_trading/list_option_positions.py SPY
    python options_trading/list_option_positions.py SPY --tp 0.50 --sl 0.50
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Path setup
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "Trading"))

try:
    from alpaca_client import AlpacaClient
except ImportError:
    try:
        from Trading.alpaca_client import AlpacaClient
    except ImportError:
        print("Error: Could not import AlpacaClient. Check PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

def parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parses an OCC option symbol string into its components.
    Example: AAPL230616C00150000
    """
    if len(symbol) < 15:
        return None
    try:
        suffix = symbol[-15:]
        root = symbol[:-15].strip()
        date_str = suffix[:6]
        type_char = suffix[6]
        strike_str = suffix[7:]
        
        expiry = datetime.strptime(date_str, "%y%m%d")
        strike = float(strike_str) / 1000.0
        return {
            'root': root,
            'expiry': expiry,
            'type': 'call' if type_char.upper() == 'C' else 'put',
            'strike': strike,
            'symbol': symbol
        }
    except (ValueError, IndexError):
        return None

def main():
    parser = argparse.ArgumentParser(description="List open option positions for a given underlying symbol.")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g., SPY, QQQ)")
    parser.add_argument("--tp", type=float, help="Take Profit threshold (decimal, e.g. 0.50 for 50%%)")
    parser.add_argument("--sl", type=float, help="Stop Loss threshold (decimal, e.g. 0.50 for 50%%)")
    args = parser.parse_args()

    symbol_to_find = args.symbol.upper()

    try:
        client = AlpacaClient()
        positions = client.get_all_positions()
    except Exception as e:
        print(f"Error fetching positions from Alpaca: {e}", file=sys.stderr)
        sys.exit(1)

    if not positions:
        print("No open positions found.")
        return

    option_positions = []
    for pos in positions:
        if pos.get('asset_class') != 'us_option':
            continue
        
        parsed = parse_option_symbol(pos['symbol'])
        if parsed and parsed['root'] == symbol_to_find:
            option_positions.append({**pos, **parsed})

    if not option_positions:
        print(f"No open option positions found for underlying symbol '{symbol_to_find}'.")
        return

    option_positions.sort(key=lambda x: (x['expiry'], x['strike']))

    print(f"--- Open Option Positions for {symbol_to_find} ---")
    header = f"{'Symbol':<22} {'Qty':>6} {'Type':>5} {'Strike':>8} {'Expiry':<12} {'Avg Cost':>10} {'Market Val':>12} {'P/L':>12} {'P/L %':>8}"
    if args.tp or args.sl:
        header += " {'Status':<12}"
    print(header)
    print("-" * len(header))

    total_pl = 0.0

    for pos in option_positions:
        qty = float(pos.get('qty', 0))
        display_qty = f"({abs(qty):.0f})" if pos.get('side') == 'short' else f"{qty:.0f}"
        
        avg_entry = float(pos.get('avg_entry_price', 0))
        market_val = float(pos.get('market_value', 0))
        unrealized_pl = float(pos.get('unrealized_pl', 0))
        unrealized_plpc = float(pos.get('unrealized_plpc', 0))
        
        total_pl += unrealized_pl
        
        # Status check
        status = ""
        if args.tp is not None and unrealized_plpc >= args.tp:
            status = "TARGET HIT (TP)"
        elif args.sl is not None and unrealized_plpc <= -args.sl:
            status = "TARGET HIT (SL)"
        
        line = f"{pos['symbol']:<22} {display_qty:>6} {pos['type']:>5} {pos['strike']:>8.2f} {pos['expiry'].strftime('%Y-%m-%d'):<12} ${avg_entry:>9.2f} ${abs(market_val):>11.2f} ${unrealized_pl:>11.2f} {unrealized_plpc:>7.1%}"
        if args.tp or args.sl:
            line += f" {status:<12}"
        print(line)

    print("-" * len(header))
    print(f"Total P/L for {symbol_to_find}: ${total_pl:,.2f}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Execute a single short sell order.

Usage:
  python util/execute_single_sell.py AAPL --dollars 5000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Import from trading_utils
sys.path.insert(0, str(Path(__file__).parent))

def run_short_cli(symbol: str, dollars: float = None, market: bool = False, use_bid: bool = False, mid_price: bool = False, price_offset: float = 0.0, verbose: bool = False, dry_run: bool = False) -> Dict:
    script_path = Path(__file__).parent.parent / "Trading" / "alpaca_short_sell_cli.py"
    if not script_path.exists():
        raise FileNotFoundError(f"alpaca_short_sell_cli.py not found at {script_path}")
    cmd = [str(script_path), symbol]
    if dollars is not None:
        cmd.append(str(dollars))
    if market:
        cmd.append("--market")
    if use_bid:
        cmd.append("--use-bid")
    if mid_price:
        cmd.append("--mid-price")
    if price_offset:
        cmd += ["--price-offset", str(price_offset)]
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out_text = result.stdout
        try:
            # Parse full or trailing JSON (skip any preface text)
            try:
                parsed = json.loads(out_text)
            except json.JSONDecodeError:
                start = out_text.rfind('\n{')
                if start != -1:
                    parsed = json.loads(out_text[start+1:])
                else:
                    raise
            return {"success": True, "output": parsed, "exit_code": result.returncode}
        except Exception:
            return {"success": result.returncode == 0, "output": out_text or result.stderr, "exit_code": result.returncode}
    except Exception as e:
        return {"success": False, "error": str(e), "exit_code": -1}

def main():
    parser = argparse.ArgumentParser(description='Execute single short sell order')
    parser.add_argument('symbol', help='Stock symbol to short')
    parser.add_argument('--dollars', type=float, help='Dollar amount')
    parser.add_argument('--market', action='store_true', help='Use market order')
    parser.add_argument('--use-bid', action='store_true', help='Use bid price')
    parser.add_argument('--mid-price', action='store_true', help='Use mid price')
    parser.add_argument('--price-offset', type=float, default=0.0, help='Price offset')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    parser.add_argument('--recommendation', help='Recommendation type (e.g. STRONG_SELL)')
    args = parser.parse_args()
    
    try:
        result = run_short_cli(
            args.symbol,
            dollars=args.dollars,
            market=args.market,
            use_bid=args.use_bid,
            mid_price=args.mid_price,
            price_offset=args.price_offset,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
        
        print(json.dumps(result, indent=2))
        if not result['success']:
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Execute a single buy order.

Usage:
  python util/execute_single_buy.py AAPL --dollars 5000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Import from trading_utils
sys.path.insert(0, str(Path(__file__).parent))

def execute_buy_order(symbol: str, dollars: float = None, market: bool = False, mid_price: bool = False, verbose: bool = False, dry_run: bool = False) -> Dict:
    """Execute a buy order using alpaca_buy_cli.py."""
    script_path = Path(__file__).parent.parent / "Trading" / "alpaca_buy_cli.py"
    if not script_path.exists():
        raise FileNotFoundError(f"alpaca_buy_cli.py not found at {script_path}")
    
    cmd = [str(script_path), symbol]
    if dollars is not None:
        cmd.append(str(dollars))
    if market:
        cmd.append("--market")
    if mid_price:
        cmd.append("--mid-price")
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        output_text = result.stdout
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError:
            json_start = output_text.rfind('\n{')
            if json_start != -1:
                try:
                    output = json.loads(output_text[json_start+1:])
                except json.JSONDecodeError:
                    return {
                        "symbol": symbol,
                        "success": result.returncode == 0,
                        "output": output_text or result.stderr,
                        "exit_code": result.returncode
                    }
            else:
                return {
                    "symbol": symbol,
                    "success": result.returncode == 0,
                    "output": output_text or result.stderr,
                    "exit_code": result.returncode
                }
        
        return {
            "symbol": symbol,
            "success": output.get("success", False),
            "output": output,
            "exit_code": result.returncode
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "success": False,
            "error": str(e),
            "exit_code": -1
        }

def main():
    parser = argparse.ArgumentParser(description='Execute single buy order')
    parser.add_argument('symbol', help='Stock symbol to buy')
    parser.add_argument('--dollars', type=float, help='Dollar amount')
    parser.add_argument('--market', action='store_true', help='Use market order')
    parser.add_argument('--mid-price', action='store_true', help='Use mid price')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Simulate execution')
    parser.add_argument('--recommendation', help='Recommendation type (e.g. STRONG_BUY)')
    args = parser.parse_args()
    
    try:
        result = execute_buy_order(
            args.symbol,
            dollars=args.dollars,
            market=args.market,
            mid_price=args.mid_price,
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

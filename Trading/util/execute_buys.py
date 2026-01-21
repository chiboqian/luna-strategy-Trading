#!/usr/bin/env python3
"""
Execute buy orders for top recommended stocks.

Reads recommendations from a JSON file or payload,
fetches the top buy recommendations, and executes buy orders using alpaca_buy_cli.py.

Usage:
  python util/execute_buys.py --json-file /path/to/recommendations.json
  python util/execute_buys.py --json-payload '{"research": [...]}'

Optional:
  --dollars AMOUNT   Dollar amount per stock (default: from Trading.yaml)
  --dry-run          Show what would be executed without running orders
  --market           Use market orders instead of limit orders
  --mid-price        Use midpoint of bid and ask price
  --verbose          Print detailed information
  -q, --quiet        Suppress non-error output
"""

import argparse
import json
import sys
import subprocess
import os
import yaml
from pathlib import Path
from typing import List, Dict

# Import from trading_utils
sys.path.insert(0, str(Path(__file__).parent))

def load_min_score_from_config() -> float:
    cfg_path = Path(__file__).parent.parent / 'config' / 'Trading.yaml'
    try:
        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            rec = (data.get('recommendations') or {})
            val = rec.get('min_score')
            if isinstance(val, (int, float)):
                return float(val)
    except Exception:
        pass
    return 22.0

def main():
    parser = argparse.ArgumentParser(
        description='Execute buy orders for top recommended stocks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dollars', type=float, help='Dollar amount per stock (uses default from Trading.yaml if not specified)')
    parser.add_argument('--dry-run', action='store_true', help='Show orders without executing')
    parser.add_argument('--market', action='store_true', help='Use market orders instead of limit')
    parser.add_argument('--mid-price', action='store_true', help='Use midpoint of bid and ask price')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    parser.add_argument('--min-score', type=float, help='Minimum total conviction score for recommendations')
    parser.add_argument('--json-file', help='Path to JSON file with recommendations')
    parser.add_argument('--json-payload', help='JSON string with recommendations')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--alpaca-api-key', help='Alpaca API Key')
    parser.add_argument('--alpaca-api-secret', help='Alpaca API Secret')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress all non-error output')
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.alpaca_api_key:
        os.environ['ALPACA_API_KEY'] = args.alpaca_api_key
    if args.alpaca_api_secret:
        os.environ['ALPACA_API_SECRET'] = args.alpaca_api_secret
    
    min_score = args.min_score if args.min_score is not None else load_min_score_from_config()
    
    top_buys = []
    all_buys = []

    if args.json_file or args.json_payload:
        try:
            if args.json_file:
                with open(args.json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = json.loads(args.json_payload)
            
            # Handle different structures
            items = []
            if isinstance(data, list):
                if len(data) > 0 and 'research' in data[0]:
                    # Structure: [{"research": [...]}]
                    for entry in data:
                        items.extend(entry.get('research', []))
                else:
                    # Structure: [...] (list of items)
                    items = data
            elif isinstance(data, dict):
                if 'research' in data:
                    items = data['research']
                else:
                    # Single item?
                    items = [data]
            
            # Filter for buys and aggregate scores
            buy_scores = {}
            
            for item in items:
                rating_val = item.get('analysis_rating')
                rating = str(rating_val).upper() if rating_val else ""
                if 'BUY' in rating:
                    score = item.get('conviction_level')
                    try:
                        score_val = float(score)
                    except (ValueError, TypeError):
                        score_val = 0.0
                    
                    symbol = item.get('symbol') or item.get('ticker')
                    if symbol:
                        if symbol not in buy_scores:
                            buy_scores[symbol] = {'score': 0.0, 'recommendation': rating}
                        buy_scores[symbol]['score'] += score_val
            
            # Convert to list and filter by min_score
            for symbol, data in buy_scores.items():
                item = {
                    'symbol': symbol,
                    'recommendation': data['recommendation'],
                    'score': data['score']
                }
                all_buys.append(item)
                
                if data['score'] >= min_score:
                    top_buys.append(item)
            
            # Sort by score descending
            top_buys.sort(key=lambda x: x['score'], reverse=True)
            all_buys.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            print(f"Error parsing JSON input: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Error: Either --json-file or --json-payload must be provided", file=sys.stderr)
        sys.exit(1)
        
    if not top_buys:
        if args.json:
            print(json.dumps({
                "recommendations": [], 
                "all_recommendations": all_buys,
                "executions": [], 
                "summary": {"total": 0, "successful": 0}
            }))
        elif not args.quiet:
            print("No buy recommendations found.")
        sys.exit(0)
        
    if not args.json and not args.quiet:
        print(f"\nTop buy recommendations:")
        for i, b in enumerate(top_buys, 1):
            score = b.get('score')
            score_str = f" [score: {score:.1f}]" if isinstance(score, (int, float)) else ""
            print(f"  {i}. {b['symbol']} ({b['recommendation']}){score_str}")

    # 2. Execute orders
    if not args.json and not args.quiet:
        if args.dry_run:
            print("\n🔍 Dry run mode - simulating orders...")
        else:
            print(f"\n📈 Executing buy orders...")
        
    execute_script = Path(__file__).parent / "execute_single_buy.py"
    results = []
    
    for b in top_buys:
        if not args.json and not args.quiet:
            print(f"\n→ Buying {b['symbol']}...")
            
        cmd = [sys.executable, str(execute_script), b['symbol']]
        if args.dollars:
            cmd.extend(["--dollars", str(args.dollars)])
        if args.market:
            cmd.append("--market")
        if args.mid_price:
            cmd.append("--mid-price")
        if args.verbose:
            cmd.append("--verbose")
        if args.dry_run:
            cmd.append("--dry-run")
        if b.get('recommendation'):
            cmd.extend(["--recommendation", b['recommendation']])
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse output
            try:
                output_json = json.loads(result.stdout)
                if isinstance(output_json, dict):
                    output_json['symbol'] = b['symbol']
                    output_json['action'] = 'BUY'
                results.append(output_json)
                
                if output_json.get('success'):
                    if not args.json and not args.quiet:
                        print(f"  ✅ Success")
                        if args.verbose:
                            print(json.dumps(output_json, indent=2))
                else:
                    if not args.json and not args.quiet:
                        print(f"  ❌ Failed")
                        if output_json.get('error'):
                            print(f"     Error: {output_json['error']}")
            except json.JSONDecodeError:
                if not args.json:
                    print(f"  ❌ Failed (Invalid JSON output)")
                    print(f"     Output: {result.stdout}")
                results.append({
                    "success": False, 
                    "error": "Invalid JSON output", 
                    "raw_output": result.stdout,
                    "symbol": b['symbol'],
                    "action": "BUY"
                })
                
        except Exception as e:
            if not args.json:
                print(f"  ❌ Failed to execute script: {e}")
            results.append({
                "success": False, 
                "error": str(e),
                "symbol": b['symbol'],
                "action": "BUY"
            })

    # Summary
    successful = sum(1 for r in results if r.get('success'))
    
    if args.json:
        print(json.dumps({
            "recommendations": top_buys,
            "all_recommendations": all_buys,
            "executions": results,
            "summary": {
                "total": len(results),
                "successful": successful
            }
        }, indent=2))
    else:
        if not args.quiet:
            print("\n" + "="*60)
            print(f"Summary: {successful}/{len(results)} orders executed successfully")
    
    if successful < len(results):
        sys.exit(1)

if __name__ == '__main__':
    main()
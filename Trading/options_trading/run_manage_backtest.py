#!/usr/bin/env python3
"""
Runs manage_options.py in mock mode for a directory of order files.
Generates a summary CSV of management results.
"""
import argparse
import subprocess
import sys
import json
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run management backtest on order files")
    parser.add_argument("--orders-dir", required=True, help="Directory containing order_*.json files")
    parser.add_argument("--historical", required=True, help="Path to historical data")
    parser.add_argument("--output", default="management_results.csv", help="Output CSV file")
    parser.add_argument("--tp", type=float, default=0.5, help="Take Profit %")
    parser.add_argument("--sl", type=float, default=0.5, help="Stop Loss %")
    parser.add_argument("--dte", type=int, default=5, help="Close DTE")
    args = parser.parse_args()
    
    orders_dir = Path(args.orders_dir)
    if not orders_dir.exists():
        print(f"Error: {orders_dir} does not exist")
        sys.exit(1)
        
    order_files = list(orders_dir.glob("order_*.json"))
    print(f"Found {len(order_files)} order files in {orders_dir}")
    
    results = []
    
    manage_script = Path(__file__).parent / "manage_options.py"
    if not manage_script.exists():
        print(f"Error: {manage_script} not found")
        sys.exit(1)
        
    for i, order_file in enumerate(order_files):
        print(f"[{i+1}/{len(order_files)}] Processing {order_file.name}...")
        
        cmd = [
            sys.executable, str(manage_script),
            "--historical", args.historical,
            "--mock-position", str(order_file),
            "--tp", str(args.tp),
            "--sl", str(args.sl),
            "--dte", str(args.dte),
            "--json"
        ]
        
        try:
            # Capture stdout only, let stderr flow to console for progress updates
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            if res.returncode == 0:
                try:
                    # The script might print other things, find the JSON line
                    output = res.stdout.strip()
                    if output:
                        # Parse the last line or try to find JSON object
                        lines = output.split('\n')
                        # Try to find the JSON line (it should be the last one if successful)
                        json_line = None
                        for line in reversed(lines):
                            if line.strip().startswith('{') and line.strip().endswith('}'):
                                json_line = line
                                break
                        
                        if json_line:
                            data = json.loads(json_line)
                            status = data.get('status')
                            if status == 'closed':
                                results.append({
                                    'order_file': order_file.name,
                                    'close_date': data['close_date'],
                                    'reason': data['reason'],
                                    'pnl': data['pnl'],
                                    'pnl_pct': data['pnl_pct']
                                })
                                print(f"  -> Closed: {data['reason']} (PnL: ${data['pnl']:.2f})")
                            elif status == 'expired':
                                results.append({
                                    'order_file': order_file.name,
                                    'close_date': data['close_date'],
                                    'reason': data['reason'],
                                    'pnl': data['pnl'],
                                    'pnl_pct': data['pnl_pct']
                                })
                                print(f"  -> Expired: {data['reason']} (PnL: ${data['pnl']:.2f})")
                            else:
                                print(f"  -> No exit triggered (Status: {status}).")
                        else:
                            print(f"  -> No valid JSON output found.")
                    else:
                        print(f"  -> No output.")
                        
                except json.JSONDecodeError:
                    print(f"  -> Error parsing JSON output.")
            else:
                print(f"  -> Error running script (exit code {res.returncode})")
        except Exception as e:
            print(f"  -> Exception: {e}")
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
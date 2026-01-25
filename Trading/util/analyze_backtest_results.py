#!/usr/bin/env python3
"""
Analyzes backtest P&L JSON files in a directory.
Generates a CSV summary and prints performance metrics.

Usage:
    python util/analyze_backtest_results.py --dir results/spy_weekly
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path

def analyze_results(directory: str, output_csv: str = None):
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        print(f"Error: Directory {directory} not found.", file=sys.stderr)
        sys.exit(1)
    
    files = list(p.glob("pnl_*.json"))
    if not files:
        print(f"No pnl_*.json files found in {directory}", file=sys.stderr)
        sys.exit(0)
        
    data = []
    print(f"Processing {len(files)} files in {directory}...")
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                d = json.load(fp)
                
                row = {
                    'file': f.name,
                    'close_date': d.get('close_date'),
                    'total_pnl': float(d.get('total_pnl', 0.0)),
                    'entry_cost': float(d.get('entry_cash_flow', 0.0)),
                    'exit_credit': float(d.get('exit_cash_flow', 0.0)),
                    'quantity': float(d.get('quantity', 0)),
                    'unit_pnl': float(d.get('unit_pnl', 0.0))
                }
                data.append(row)
        except Exception as e:
            print(f"Warning: Error reading {f.name}: {e}", file=sys.stderr)
            
    if not data:
        print("No valid data extracted.")
        sys.exit(0)
        
    df = pd.DataFrame(data)
    
    # Sort by date
    if 'close_date' in df.columns:
        df['close_date'] = pd.to_datetime(df['close_date'])
        df = df.sort_values('close_date')
    
    # Save CSV
    if output_csv:
        out_path = Path(output_csv)
    else:
        out_path = p / "analysis_summary.csv"
        
    df.to_csv(out_path, index=False)
    print(f"Saved summary CSV to: {out_path}")
    
    # Analysis
    total_trades = len(df)
    total_pnl = df['total_pnl'].sum()
    wins = df[df['total_pnl'] > 0]
    losses = df[df['total_pnl'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    avg_win = wins['total_pnl'].mean() if win_count > 0 else 0
    avg_loss = losses['total_pnl'].mean() if loss_count > 0 else 0
    
    gross_profit = wins['total_pnl'].sum()
    gross_loss = abs(losses['total_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    expectancy = total_pnl / total_trades if total_trades > 0 else 0
    
    # Drawdown
    df['cumulative_pnl'] = df['total_pnl'].cumsum()
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()
    
    print("\n" + "="*40)
    print(f"Backtest Analysis Report")
    print("="*40)
    print(f"Total Trades:    {total_trades}")
    print(f"Total P&L:       ${total_pnl:,.2f}")
    print(f"Expectancy:      ${expectancy:,.2f} per trade")
    print(f"Win Rate:        {win_rate:.1%} ({win_count}W / {loss_count}L)")
    print(f"Avg Win:         ${avg_win:,.2f}")
    print(f"Avg Loss:        ${avg_loss:,.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Max Drawdown:    ${max_drawdown:,.2f}")
    
    # Monthly breakdown
    if 'close_date' in df.columns:
        print("\n--- Monthly P&L ---")
        df['month'] = df['close_date'].dt.to_period('M')
        monthly = df.groupby('month')['total_pnl'].sum()
        for period, pnl in monthly.items():
            print(f"{period}: ${pnl:,.2f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze backtest P&L files")
    parser.add_argument("--dir", required=True, help="Directory containing pnl_*.json files")
    parser.add_argument("--output", help="Path to save summary CSV (default: analysis_summary.csv in dir)")
    args = parser.parse_args()
    
    analyze_results(args.dir, args.output)

if __name__ == "__main__":
    main()
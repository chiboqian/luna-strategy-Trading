#!/usr/bin/env python3
"""
Orchestrator script to:
1. Download high-volume options from Yahoo Finance.
2. Feed the underlying symbols into the Option Strategy Scanner.
"""

import sys
import argparse
import subprocess
import shlex
from pathlib import Path
import pandas as pd

# Import the download function from the sibling script
try:
    from get_yahoo_option_volume import download_yahoo_options
except ImportError:
    # Handle case where script is run from root directory
    sys.path.append(str(Path(__file__).parent))
    from get_yahoo_option_volume import download_yahoo_options

def main():
    parser = argparse.ArgumentParser(description="Scan High Volume Options from Yahoo")
    parser.add_argument("--min-volume", type=int, default=20000, help="Minimum volume filter (default: 20000)")
    parser.add_argument("--scanner-args", type=str, default="", help="Additional arguments for scanner (e.g. '--strategy synthetic_long --verbose')")
    parser.add_argument("--max-spread", type=float, help="Max bid/ask spread percent for scanner (e.g. 2.0)")
    parser.add_argument("--input-file", type=str, help="Path to offline CSV file to use instead of downloading")
    
    args = parser.parse_args()

    df = None
    if args.input_file:
        print(f"--- Step 1: Loading offline data from {args.input_file} ---")
        try:
            df = pd.read_csv(args.input_file)
            # Apply volume filter if column exists
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                df = df[df['volume'] >= args.min_volume]
                print(f"Filtered to {len(df)} items with volume >= {args.min_volume}")
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        # 1. Get Data
        print(f"--- Step 1: Fetching Options with Volume >= {args.min_volume} ---")
        df = download_yahoo_options(min_volume=args.min_volume)
    
    # 2. Run Scanner
    if df is not None and not df.empty:
        symbol_col = None
        if 'underlyingSymbol' in df.columns:
            symbol_col = 'underlyingSymbol'
        elif 'symbol' in df.columns:
            symbol_col = 'symbol'
        elif 'Symbol' in df.columns:
            symbol_col = 'Symbol'

        if symbol_col:
            symbols = df[symbol_col].unique().tolist()
            symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
            
            if symbols:
                print(f"\n--- Step 2: Feeding {len(symbols)} symbols to Option Strategy Scanner ---")
                scanner_script = Path(__file__).parent / "optionStrategyScanner.py"
                
                # Construct command: python optionStrategyScanner.py --symbol SYM1 SYM2 ... [extra_args]
                cmd = [sys.executable, str(scanner_script), "--symbol"] + symbols
                
                if args.scanner_args:
                    cmd.extend(shlex.split(args.scanner_args))
                
                if args.max_spread:
                    cmd.extend(["--max-spread", str(args.max_spread)])
                
                print(f"Executing Scanner...")
                subprocess.run(cmd)
            else:
                print("No valid symbols found in data.")
        else:
            print("Error: Neither 'underlyingSymbol' nor 'symbol' column found in data.")
    else:
        print("No data returned from Yahoo Finance.")

if __name__ == "__main__":
    main()
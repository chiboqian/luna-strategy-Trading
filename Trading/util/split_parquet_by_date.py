#!/usr/bin/env python3
"""
Splits a Parquet file into multiple CSV files grouped by date.
Creates subfolders for each year (e.g., output_dir/2023/2023-01-01.csv).

Usage:
    python util/split_parquet_by_date.py data.parquet --output-dir exported_data
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import os

def split_parquet(input_file, output_dir, date_col=None):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    # Auto-detect date column if not provided
    if not date_col:
        # Priority: 'date', 'quote_date', 'timestamp', columns with 'date' in name
        candidates = ['date', 'quote_date', 'timestamp', 'time', 'created_at']
        
        # Check exact matches first
        for c in candidates:
            if c in df.columns:
                date_col = c
                break
        
        # Check partial matches or types if not found
        if not date_col:
            for c in df.columns:
                if 'date' in c.lower():
                    date_col = c
                    break
        
        if not date_col:
            # Check for datetime objects
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_col = c
                    break

    if not date_col or date_col not in df.columns:
        print("Error: Could not auto-detect a date column. Please specify with --date-col.")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    print(f"Splitting by column: {date_col}")

    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if df[date_col].isna().any():
                print(f"Warning: Dropping {df[date_col].isna().sum()} rows with invalid dates.")
                df = df.dropna(subset=[date_col])
        except Exception as e:
            print(f"Error converting column '{date_col}' to datetime: {e}")
            sys.exit(1)

    unique_dates = df[date_col].dt.date.unique()
    print(f"Found {len(unique_dates)} unique dates. Starting export...")

    count = 0
    for d in sorted(unique_dates):
        date_str = str(d) # YYYY-MM-DD
        year_str = str(d.year)
        
        year_dir = output_base / year_str
        year_dir.mkdir(exist_ok=True)
        
        daily_df = df[df[date_col].dt.date == d]
        daily_df.to_csv(year_dir / f"{date_str}.csv", index=False)
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count}/{len(unique_dates)} days...", end='\r')

    print(f"\nSuccessfully created {count} CSV files in {output_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a Parquet file into daily CSV files organized by year.")
    parser.add_argument("input_file", help="Path to the input Parquet file")
    parser.add_argument("--output-dir", required=True, help="Directory where output folders will be created")
    parser.add_argument("--date-col", help="Name of the date column to split by (optional, auto-detected)")
    args = parser.parse_args()
    
    split_parquet(args.input_file, args.output_dir, args.date_col)
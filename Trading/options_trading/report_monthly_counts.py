#!/usr/bin/env python3
"""
Generates a report of counts per year and per month from a data file.
Can count total rows or unique values of a specific column.

Usage:
    ./util/report_monthly_counts.py data/options.parquet
    ./util/report_monthly_counts.py data/trades.csv --date-col timestamp --unique-col account_id
"""

import pandas as pd
import argparse
import sys
import os

def report_counts(file_path, date_col=None, unique_col=None):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    print(f"Reading {file_path}...")
    try:
        try:
            df = pd.read_parquet(file_path)
        except Exception:
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                # Fallback for simple single-column text files
                df = pd.read_csv(file_path, header=None, names=['data'])
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Auto-detect date column if not provided
    if not date_col:
        candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'created_at' in c.lower()]
        if candidates:
            date_col = candidates[0]
            print(f"Auto-detected date column: {date_col}")
        else:
            print("Error: Could not detect a date column. Please specify with --date-col")
            sys.exit(1)
    
    if date_col not in df.columns:
        print(f"Error: Column '{date_col}' not found in file.")
        sys.exit(1)

    # Convert to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        print(f"Error converting column '{date_col}' to datetime: {e}")
        sys.exit(1)

    print(f"\nReport for {file_path}")
    print(f"Total Records: {len(df)}")
    
    # Group by Month
    period_group = df[date_col].dt.to_period("M")
    
    if unique_col:
        if unique_col not in df.columns:
            print(f"Error: Unique column '{unique_col}' not found.")
            sys.exit(1)
        print(f"Counting unique '{unique_col}' per month...")
        counts = df.groupby(period_group)[unique_col].nunique().reset_index(name='Count')
    else:
        print("Counting total rows per month...")
        counts = df.groupby(period_group).size().reset_index(name='Count')

    counts.columns = ['Period', 'Count']
    counts = counts.sort_values('Period')

    print("-" * 30)
    print(f"{'Period':<10} | {'Count':>10}")
    print("-" * 30)
    
    for _, row in counts.iterrows():
        print(f"{str(row['Period']):<10} | {row['Count']:>10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report counts per year and month from a data file.")
    parser.add_argument("file", help="Path to parquet or csv file")
    parser.add_argument("--date-col", help="Name of the date column (optional, auto-detected)")
    parser.add_argument("--unique-col", help="Column to count unique values for (e.g. account_id)")
    args = parser.parse_args()

    report_counts(args.file, args.date_col, args.unique_col)
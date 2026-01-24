#!/usr/bin/env python3
import pandas as pd
import sys
import warnings
import os


def query_parquet(file_path, query=None, columns=None, limit=10, describe=False, unique=None):
    """
    Reads an Apache Parquet file and optionally queries/filter the data.
    :param file_path: Path to the Parquet file
    :param query: Optional pandas query string (e.g., 'col > 5')
    :param columns: Optional list of columns to select
    :param limit: Number of rows to display
    :param describe: If True, print dataframe info
    :param unique: If set, print unique values for this column
    :return: Filtered DataFrame
    """
    if os.path.exists(file_path):
        print(f"Reading {file_path} ({os.path.getsize(file_path)} bytes)...")

    try:
        df = pd.read_parquet(file_path, columns=columns)
    except ImportError:
        print("Error: Missing parquet engine. Please run: pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    # Adjust pandas display options to show all rows requested
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if describe:
        df.info()
        return df

    if unique:
        if unique in df.columns:
            values = df[unique].unique()
            print(f"Unique values for '{unique}' ({len(values)}):")
            print(pd.Series(values).to_string(index=False))
        else:
            print(f"Error: Column '{unique}' not found.", file=sys.stderr)
        return df

    if query:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                df = df.query(query)
        except Exception as e:
            print(f"Error executing query '{query}': {e}", file=sys.stderr)
            print("Hint: Use '==' for equality and quotes for strings (e.g., \"type == 'call'\")", file=sys.stderr)
            sys.exit(1)

    if df.empty:
        print("Warning: The dataset is empty (0 rows).")

    print(f"Total rows: {len(df)}")
    print(df.head(limit))
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Query an Apache Parquet file.",
        epilog="Examples:\n  ./util/query_parquet.py data.parquet --query \"symbol == 'AAPL' and price > 150\"\n  ./util/query_parquet.py data.parquet --query \"type == 'call' or type == 'put'\"",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to the Parquet file")
    parser.add_argument("--query", help="Pandas query string (e.g., 'col > 5 and col2 == \"val\"')", default=None)
    parser.add_argument("--columns", nargs="*", help="Columns to select", default=None)
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to display")
    parser.add_argument("--describe", action="store_true", help="Show dataset schema and info")
    parser.add_argument("--unique", help="Show unique values for a specific column")
    args = parser.parse_args()

    query_parquet(args.file, args.query, args.columns, args.limit, args.describe, args.unique)


if __name__ == "__main__":
    main()

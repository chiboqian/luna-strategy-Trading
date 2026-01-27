#!/usr/bin/env python3
"""
Script to query Databento DBN files (compressed or uncompressed).
Requires 'databento' package (pip install databento).
"""

import argparse
import sys
import os
import warnings
import pandas as pd

try:
    import databento
except ImportError:
    print("❌ Error: 'databento' module not found. Please install it using 'pip install databento'.", file=sys.stderr)
    sys.exit(1)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

def query_dbn(file_path, query=None, columns=None, limit=10, describe=False, unique=None, sql=None):
    """
    Reads a Databento DBN file (.dbn, .dbn.zst) and queries it.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {file_path} ({os.path.getsize(file_path)} bytes)...")

    try:
        # DBNStore handles compression transparently
        # to_df() converts to pandas DataFrame
        df = databento.DBNStore.from_file(file_path).to_df()
        
        # Reset index so fields like 'ts_event' are available as columns
        df.reset_index(inplace=True)
        
    except Exception as e:
        print(f"❌ Error reading DBN file: {e}", file=sys.stderr)
        sys.exit(1)

    # SQL Query (DuckDB)
    if sql:
        if not HAS_DUCKDB:
            print("❌ Error: 'duckdb' module not found. Please install it using 'pip install duckdb'.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Executing SQL: {sql}")
        try:
            # Register df as a view for DuckDB
            duckdb.register('df', df)
            
            # Replace TABLE with df
            sql_query = sql.replace('TABLE', 'df')
            
            # Handle implicit SELECT *
            sql_upper = sql.strip().upper()
            if not any(sql_upper.startswith(k) for k in ["SELECT", "DESCRIBE", "SHOW", "WITH"]):
                cols_str = "*"
                if columns:
                    cols_str = ", ".join(columns)
                sql_query = f"SELECT {cols_str} FROM df WHERE {sql}"
            
            df = duckdb.query(sql_query).df()
            
        except Exception as e:
            print(f"❌ SQL Error: {e}", file=sys.stderr)
            sys.exit(1)
            
    # Pandas Query
    elif query:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                df = df.query(query, engine='python')
        except Exception as e:
            print(f"❌ Query Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Column Selection (if not handled by SQL)
    if columns and not sql:
        try:
            # Check for missing columns
            missing = [c for c in columns if c not in df.columns]
            if missing:
                print(f"⚠️ Warning: Columns not found: {missing}", file=sys.stderr)
            
            # Select existing columns
            existing = [c for c in columns if c in df.columns]
            if existing:
                df = df[existing]
        except Exception as e:
            print(f"❌ Column Selection Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Display Settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Describe
    if describe:
        print("\n=== Schema ===")
        df.info()
        print("\n=== Stats ===")
        print(df.describe())
        return

    # Unique Values
    if unique:
        if unique in df.columns:
            vals = df[unique].unique()
            print(f"\nUnique values for '{unique}' ({len(vals)}):")
            for v in vals[:limit]:
                print(v)
            if len(vals) > limit:
                print(f"... and {len(vals) - limit} more")
        else:
            print(f"❌ Column '{unique}' not found.")
        return

    # Show Data
    if df.empty:
        print("⚠️ Result is empty.")
    else:
        print(f"\nTotal rows: {len(df)}")
        print(df.head(limit))


def main():
    parser = argparse.ArgumentParser(
        description="Query a Databento DBN file (.dbn, .dbn.zst).",
        epilog="Examples:\n  ./query_dbn.py data.dbn.zst --query \"price > 100\"\n  ./query_dbn.py data.dbn.zst --sql \"SELECT * FROM TABLE LIMIT 5\"",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to the .dbn or .dbn.zst file")
    parser.add_argument("--query", help="Pandas query string")
    parser.add_argument("--sql", help="SQL query (DuckDB). Use 'TABLE' as placeholder.")
    parser.add_argument("--columns", nargs="*", help="Columns to select")
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to display")
    parser.add_argument("--describe", action="store_true", help="Show schema and stats")
    parser.add_argument("--unique", help="Show unique values for a column")
    
    args = parser.parse_args()
    
    query_dbn(args.file, args.query, args.columns, args.limit, args.describe, args.unique, args.sql)

if __name__ == "__main__":
    main()
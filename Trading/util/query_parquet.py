#!/usr/bin/env python3
import pandas as pd
import sys
import warnings
import os

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

def query_parquet(file_path, query=None, columns=None, limit=10, describe=False, unique=None, sql=None):
    """
    Reads an Apache Parquet file and optionally queries/filter the data.
    :param file_path: Path to the Parquet file
    :param query: Optional pandas query string (e.g., 'col > 5')
    :param sql: Optional SQL query string (DuckDB) - use 'TABLE' as placeholder for file
    :param columns: Optional list of columns to select
    :param limit: Number of rows to display
    :param describe: If True, print dataframe info
    :param unique: If set, print unique values for this column
    :return: Filtered DataFrame
    """
    if os.path.exists(file_path):
        print(f"Reading {file_path} ({os.path.getsize(file_path)} bytes)...")

    df = None

    if sql:
        if not HAS_DUCKDB:
            print("Error: duckdb is not installed. Please run: pip install duckdb", file=sys.stderr)
            sys.exit(1)
        
        # Construct SQL
        # If it looks like a full query, replace TABLE placeholder
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("SELECT") or sql_upper.startswith("DESCRIBE") or sql_upper.startswith("SHOW"):
            final_sql = sql.replace("TABLE", f"'{file_path}'")
        else:
            # Assume it's a WHERE clause
            cols = "*"
            if columns:
                cols = ", ".join(columns)
            final_sql = f"SELECT {cols} FROM '{file_path}' WHERE {sql}"
            
        print(f"Executing DuckDB SQL: {final_sql}")
        try:
            df = duckdb.sql(final_sql).df()
        except Exception as e:
            print(f"DuckDB Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if HAS_DUCKDB:
            cols = "*"
            if columns:
                cols = ", ".join([f'"{c}"' for c in columns])
            try:
                df = duckdb.sql(f"SELECT {cols} FROM '{file_path}'").df()
            except Exception as e:
                print(f"Warning: DuckDB read failed ({e}), falling back to pandas.", file=sys.stderr)

        if df is None:
            try:
                df = pd.read_parquet(file_path, columns=columns)
            except ImportError:
                print("Error: Missing parquet engine. Please run: pip install pyarrow", file=sys.stderr)
                sys.exit(1)

        if query:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    df = df.query(query)
            except Exception as e:
                print(f"Error executing query '{query}': {e}", file=sys.stderr)
                print("Hint: Use '==' for equality and quotes for strings (e.g., \"type == 'call'\")", file=sys.stderr)
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
    parser.add_argument("--sql", help="DuckDB SQL query (e.g., 'SELECT * FROM TABLE WHERE price > 100')", default=None)
    parser.add_argument("--columns", nargs="*", help="Columns to select", default=None)
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to display")
    parser.add_argument("--describe", action="store_true", help="Show dataset schema and info")
    parser.add_argument("--unique", help="Show unique values for a specific column")
    args = parser.parse_args()

    query_parquet(args.file, args.query, args.columns, args.limit, args.describe, args.unique, args.sql)


if __name__ == "__main__":
    main()

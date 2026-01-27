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

try:
    import databento
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False

def query_zst(file_path, query=None, columns=None, limit=10, describe=False, unique=None, sql=None, file_type=None):
    """
    Reads a Zstandard compressed file (CSV/JSON) and optionally queries/filter the data.
    :param file_path: Path to the .zst file
    :param query: Optional pandas query string (e.g., 'col > 5')
    :param sql: Optional SQL query string (DuckDB) - use 'TABLE' as placeholder for file
    :param columns: Optional list of columns to select
    :param limit: Number of rows to display
    :param describe: If True, print dataframe info
    :param unique: If set, print unique values for this column
    :param file_type: Optional 'csv' or 'json'. If None, inferred from filename.
    :return: Filtered DataFrame
    """
    if os.path.exists(file_path):
        print(f"Reading {file_path} ({os.path.getsize(file_path)} bytes)...")
    else:
        print(f"Error: File {file_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Infer file type if not provided
    if not file_type:
        if ".json" in file_path.lower():
            file_type = 'json'
        elif ".dbn" in file_path.lower():
            file_type = 'dbn'
        else:
            file_type = 'csv' # Default to CSV for .csv.zst or just .zst

    df = None

    # Handle DBN files (Databento)
    if file_type == 'dbn':
        if not HAS_DATABENTO:
            print("Error: databento is not installed. Please run: pip install databento", file=sys.stderr)
            sys.exit(1)
        try:
            print(f"Reading DBN file {file_path}...")
            # DBNStore handles compression transparently
            df = databento.DBNStore.from_file(file_path).to_df()
            # Reset index to make fields accessible as columns
            df.reset_index(inplace=True)
        except Exception as e:
            print(f"Error reading DBN file: {e}", file=sys.stderr)
            sys.exit(1)

    if sql:
        if not HAS_DUCKDB:
            print("Error: duckdb is not installed. Please run: pip install duckdb", file=sys.stderr)
            sys.exit(1)
        
        # Construct SQL
        sql_upper = sql.strip().upper()
        
        if df is not None:
            # Query the in-memory DataFrame
            if sql_upper.startswith("SELECT") or sql_upper.startswith("DESCRIBE") or sql_upper.startswith("SHOW"):
                final_sql = sql.replace("TABLE", "df")
            else:
                cols = "*"
                if columns:
                    cols = ", ".join(columns)
                final_sql = f"SELECT {cols} FROM df WHERE {sql}"
            print(f"Executing DuckDB SQL on DataFrame: {final_sql}")
        else:
            # Query the file directly
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
        # Optimization: Try to push down pandas query to DuckDB SQL if possible
        if query and HAS_DUCKDB and df is None:
            # Heuristic: Convert pandas '==' to SQL '='
            sql_where = query.replace("==", "=")
            cols_select = "*"
            if columns:
                cols_select = ", ".join([f'"{c}"' for c in columns])
            
            try:
                # DuckDB can read zst directly
                df = duckdb.sql(f"SELECT {cols_select} FROM '{file_path}' WHERE {sql_where}").df()
                query = None # Filter already applied
            except Exception:
                # Fallback to loading full DF if SQL translation fails
                pass

        if HAS_DUCKDB and df is None:
            cols = "*"
            if columns:
                cols = ", ".join([f'"{c}"' for c in columns])
            try:
                df = duckdb.sql(f"SELECT {cols} FROM '{file_path}'").df()
            except Exception as e:
                print(f"Warning: DuckDB read failed ({e}), falling back to pandas.", file=sys.stderr)

        if df is None:
            try:
                # Pandas requires zstandard for .zst
                if file_type == 'json':
                    df = pd.read_json(file_path, lines=True, compression='zstd')
                else:
                    use_cols = columns if columns else None
                    df = pd.read_csv(file_path, usecols=use_cols, compression='zstd')
            except ImportError:
                print("Error: Missing zstandard library. Please run: pip install zstandard", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading file with pandas: {e}", file=sys.stderr)
                sys.exit(1)

        if query:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    df = df.query(query, engine='python')
            except Exception as e:
                print(f"Error executing query '{query}': {e}", file=sys.stderr)
                print("Hint: Use '==' for equality and quotes for strings (e.g., \"type == 'call'\")", file=sys.stderr)
                sys.exit(1)

    # Filter columns for DBN if not handled by SQL
    if file_type == 'dbn' and columns and not sql:
        try:
            df = df[columns]
        except KeyError as e:
            print(f"Error: Column not found: {e}", file=sys.stderr)
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
        description="Query a Zstandard compressed file (CSV/JSON).",
        epilog="Examples:\n  ./util/query_zst.py data.csv.zst --query \"symbol == 'AAPL' and price > 150\"\n  ./util/query_zst.py data.json.zst --sql \"SELECT * FROM TABLE LIMIT 5\"",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to the .zst file")
    parser.add_argument("--query", help="Pandas query string (e.g., 'col > 5 and col2 == \"val\"')", default=None)
    parser.add_argument("--sql", help="DuckDB SQL query (e.g., 'SELECT * FROM TABLE WHERE price > 100')", default=None)
    parser.add_argument("--columns", nargs="*", help="Columns to select", default=None)
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to display")
    parser.add_argument("--describe", action="store_true", help="Show dataset schema and info")
    parser.add_argument("--unique", help="Show unique values for a specific column")
    parser.add_argument("--type", choices=['csv', 'json', 'dbn'], help="Force file type (csv, json, dbn). Default: inferred from extension")
    args = parser.parse_args()

    query_zst(args.file, args.query, args.columns, args.limit, args.describe, args.unique, args.sql, args.type)


if __name__ == "__main__":
    main()
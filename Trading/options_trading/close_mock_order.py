#!/usr/bin/env python3
"""
Closes a mock order using historical data and calculates P&L.

Usage:
    python util/close_mock_order.py --order mock_order.json --parquet data/options.parquet --date 2023-06-30
"""
import argparse
import json
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    import databento
except ImportError:
    databento = None

def main():
    parser = argparse.ArgumentParser(description="Close Mock Order and Calculate P&L")
    parser.add_argument("--order", required=True, help="Path to mock order JSON file")
    parser.add_argument("--parquet", required=True, help="Path to historical options parquet")
    parser.add_argument("--date", required=True, help="Close date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    # Load order
    try:
        with open(args.order, 'r') as f:
            order = json.load(f)
    except Exception as e:
        print(f"Error loading order file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load data
    try:
        path_input = Path(args.parquet)
        if path_input.is_dir():
            # New structure: folder -> year -> date.parquet
            year_str = args.date.split('-')[0]
            file_path = path_input / year_str / f"{args.date}.parquet"
            
            # Check for Databento format
            if not file_path.exists():
                dbn_date = args.date.replace('-', '')
                file_path = path_input / f"{dbn_date}.dbn.zst"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Daily data file not found in {path_input}")
                
            print(f"Loading {file_path}...")
            if str(file_path).endswith('.dbn.zst'):
                if not databento: raise ImportError("databento module required")
                df = databento.DBNStore.from_file(file_path).to_df()
                df.reset_index(inplace=True)
            else:
                df = pd.read_parquet(file_path)
        else:
            df = pd.read_parquet(args.parquet)
    except Exception as e:
        print(f"Error loading parquet file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Identify Date Column
    date_col = next((c for c in ['date', 'quote_date', 'timestamp', 'time', 'ts_event'] if c in df.columns), None)
    if not date_col:
        print("Error: No date column in parquet", file=sys.stderr)
        sys.exit(1)
        
    # Filter for Close Date
    # Convert to string for matching YYYY-MM-DD
    # Databento ts_event is datetime64[ns, UTC]
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        # Ensure we match the date part
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        
    day_df = df[df[date_col].astype(str).str.startswith(args.date)].copy()
    
    if day_df.empty:
        print(f"Error: No data found for date {args.date}", file=sys.stderr)
        sys.exit(1)

    # Identify Columns
    cols = day_df.columns
    col_map = {
        'symbol': 'symbol', # Option symbol
        'bid': 'bid',
        'ask': 'ask',
        'mark': 'mark',
        'last': 'last'
    }
    if 'contract_id' in cols:
        col_map['symbol'] = 'contract_id'
    
    # Databento mapping
    if 'bid_px_00' in cols: col_map['bid'] = 'bid_px_00'
    if 'ask_px_00' in cols: col_map['ask'] = 'ask_px_00'
    
    legs = order.get('legs', [])
    quantity = float(order.get('quantity', 1))
    entry_cash_flow = float(order.get('entry_cash_flow', 0.0))
    
    exit_cash_flow = 0.0
    leg_details = []
    
    for leg in legs:
        symbol = leg['symbol']
        side = leg['side'] # buy or sell
        
        # Find row
        row = day_df[day_df[col_map['symbol']] == symbol]
        
        if row.empty:
            print(f"Warning: No data for {symbol} on {args.date}. Assuming 0 price.", file=sys.stderr)
            close_price = 0.0
        else:
            r = row.iloc[0]
            bid = float(r.get(col_map['bid'], 0))
            ask = float(r.get(col_map['ask'], 0))
            mark = float(r.get(col_map['mark'], 0))
            last = float(r.get(col_map['last'], 0))
            
            # Close Price Logic
            # Long (Buy) -> Sell at Bid
            # Short (Sell) -> Buy at Ask
            if side == 'buy':
                close_price = bid if bid > 0 else mark if mark > 0 else last
            else:
                close_price = ask if ask > 0 else mark if mark > 0 else last
        
        # Calculate Cash Flow impact
        # Long leg: Selling gives +Cash
        # Short leg: Buying costs -Cash
        if side == 'buy':
            leg_flow = close_price
        else:
            leg_flow = -close_price
            
        exit_cash_flow += leg_flow
        
        leg_details.append({
            "symbol": symbol,
            "side": side,
            "close_price": close_price,
            "leg_cash_flow": leg_flow
        })

    # Total PnL
    # Multiplier 100 for options
    unit_pnl = entry_cash_flow + exit_cash_flow
    total_pnl = unit_pnl * quantity * 100
    
    result = {
        "order_id": order.get('id'),
        "close_date": args.date,
        "entry_cash_flow": entry_cash_flow,
        "exit_cash_flow": exit_cash_flow,
        "unit_pnl": unit_pnl,
        "quantity": quantity,
        "total_pnl": total_pnl,
        "legs": leg_details
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {args.output}")
        
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Close Date: {args.date}")
        print(f"Entry Cash Flow: ${entry_cash_flow:.2f}")
        print(f"Exit Cash Flow:  ${exit_cash_flow:.2f}")
        print("Leg Prices at Close:")
        for leg in leg_details:
            print(f"  {leg['symbol']} ({leg['side']}): ${leg['close_price']:.2f}")
        print(f"Unit P&L:        ${unit_pnl:.2f}")
        print(f"Quantity:        {quantity}")
        print(f"Total P&L:       ${total_pnl:.2f}")

if __name__ == "__main__":
    main()
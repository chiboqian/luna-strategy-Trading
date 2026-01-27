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
    parser.add_argument("--historical", required=True, help="Path to historical options data")
    parser.add_argument("--date", required=True, help="Close date (YYYY-MM-DD) or datetime (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--begin-time", help="Start time for monitoring (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--stop-loss-pct", type=float, help="Stop Loss percentage (e.g., 0.5 for 50%)")
    parser.add_argument("--take-profit-pct", type=float, help="Take Profit percentage (e.g., 0.5 for 50%)")
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
        path_input = Path(args.historical)
        if path_input.is_dir():
            # New structure: folder -> year -> date.parquet
            date_part = args.date.split(' ')[0]
            year_str = date_part.split('-')[0]
            file_path = path_input / year_str / f"{date_part}.parquet"
            
            # Check for Databento format
            if not file_path.exists():
                dbn_date = date_part.replace('-', '')
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
            df = pd.read_parquet(args.historical)
    except Exception as e:
        print(f"Error loading parquet file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Identify Date Column
    date_col = next((c for c in ['date', 'quote_date', 'timestamp', 'time', 'ts_event'] if c in df.columns), None)
    if not date_col:
        print("Error: No date column in parquet", file=sys.stderr)
        sys.exit(1)
        
    # Parse target date to check for time component
    try:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
        has_time = True
    except ValueError:
        target_dt = datetime.strptime(args.date, "%Y-%m-%d")
        has_time = False

    # Parse begin time if provided
    begin_dt = None
    if args.begin_time:
        try:
            begin_dt = datetime.strptime(args.begin_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                begin_dt = datetime.strptime(args.begin_time, "%H:%M:%S")
                # If only time provided, combine with target date
                begin_dt = datetime.combine(target_dt.date(), begin_dt.time())
            except ValueError:
                pass

    # Filter Data
    if has_time:
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception:
                pass # Fallback to string matching if conversion fails

        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Handle Timezone (assume input is ET if data is TZ-aware)
            compare_dt = target_dt
            if df[date_col].dt.tz is not None and target_dt.tzinfo is None:
                compare_dt = pd.Timestamp(target_dt).tz_localize('America/New_York').tz_convert(df[date_col].dt.tz)
            
            start_dt = compare_dt.normalize()
            if begin_dt:
                if df[date_col].dt.tz is not None and begin_dt.tzinfo is None:
                    # Assume begin_time is also ET if data is TZ aware
                    start_dt = pd.Timestamp(begin_dt).tz_localize('America/New_York').tz_convert(df[date_col].dt.tz)
                else:
                    start_dt = begin_dt
            
            print(f"[DEBUG] Filtering data <= {compare_dt} (Timezone: {df[date_col].dt.tz})", file=sys.stderr)
            
            # Filter: start_dt <= time <= compare_dt
            day_df = df[(df[date_col] >= start_dt) & (df[date_col] <= compare_dt)].copy()
            day_df = day_df.sort_values(date_col)
        else:
            day_df = df[df[date_col].astype(str).str.startswith(args.date.split()[0])].copy()
    else:
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        day_df = df[df[date_col].astype(str).str.startswith(args.date)].copy()
    
    if day_df.empty:
        print(f"Error: No data found for date {args.date}", file=sys.stderr)
        sys.exit(1)
        
    # Debug: Print data range
    min_t = day_df[date_col].min()
    max_t = day_df[date_col].max()
    print(f"Data loaded for {args.date}: {len(day_df)} rows, Range: {min_t} to {max_t}", file=sys.stderr)

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
    if 'price' in cols: col_map['last'] = 'price'
    if 'last_price' in cols: col_map['last'] = 'last_price'
    
    # If we are NOT stepping through prices (no SL/TP), we can deduplicate to just the last row per symbol
    # But if we have SL/TP, we need the time series.
    # For simplicity, let's keep the time series if SL/TP is requested, otherwise deduplicate.
    use_step_logic = args.stop_loss_pct is not None or args.take_profit_pct is not None

    legs = order.get('legs', [])
    quantity = float(order.get('quantity', 1))
    entry_cash_flow = float(order.get('entry_cash_flow', 0.0))
    print(f"[DEBUG] Entry Cash Flow from Order: {entry_cash_flow}", file=sys.stderr)
    
    exit_cash_flow = 0.0
    close_reason = "Exit Time"
    close_timestamp = args.date
    leg_details = []
    
    # Determine Strategy Type (Credit vs Debit) for PnL % calculation
    # Entry Cash Flow > 0 => Credit Strategy (Max Profit = Entry Credit)
    # Entry Cash Flow < 0 => Debit Strategy (Cost = -Entry Cash Flow)
    is_credit_strategy = entry_cash_flow > 0
    initial_basis = abs(entry_cash_flow) if entry_cash_flow != 0 else 1.0 # Avoid div by zero

    if use_step_logic:
        # Step through time
        # Filter for relevant symbols only to speed up iteration
        relevant_symbols = [leg['symbol'] for leg in legs]
        step_df = day_df[day_df[col_map['symbol']].isin(relevant_symbols)].copy()
        step_df = step_df.sort_values(date_col)
        
        triggered = False
        last_known_prices = {} # {symbol: {price_data}}
        
        # Iterate through every row (tick/bar) in chronological order
        for idx, row in step_df.iterrows():
            ts = row[date_col]
            sym = row[col_map['symbol']]
            
            # Update price data for this symbol
            bid = float(row.get(col_map['bid'], 0))
            ask = float(row.get(col_map['ask'], 0))
            mark = float(row.get(col_map['mark'], 0))
            last = float(row.get(col_map['last'], 0))
            
            last_known_prices[sym] = {
                'bid': bid,
                'ask': ask,
                'mark': mark,
                'last': last,
                'ts': ts
            }
            
            # Check if we have prices for all legs to calculate PnL
            # (Optimization: We could check this only if we have at least one price for every leg)
            if len(last_known_prices) < len(legs):
                continue
                
            # Check if we have all specific symbols needed (in case of duplicates/supersets)
            if not all(leg['symbol'] in last_known_prices for leg in legs):
                continue

            current_exit_cash_flow = 0.0
            temp_leg_details = []
            
            for leg in legs:
                sym = leg['symbol']
                side = leg['side']
                prices = last_known_prices[sym]
                
                if side == 'buy':
                    price = prices['bid'] if prices['bid'] > 0 else prices['mark'] if prices['mark'] > 0 else prices['last']
                    leg_flow = price
                else:
                    price = prices['ask'] if prices['ask'] > 0 else prices['mark'] if prices['mark'] > 0 else prices['last']
                    leg_flow = -price
                
                current_exit_cash_flow += leg_flow
                temp_leg_details.append({
                    "symbol": sym, "side": side, "close_price": price, "leg_cash_flow": leg_flow,
                    "market_data": prices
                })
                
            # Calculate PnL
            # Unit PnL = Entry + Exit
            unit_pnl = entry_cash_flow + current_exit_cash_flow
            
            # Check Triggers
            # PnL % = Unit PnL / Initial Basis
            # For Credit: Profit is positive PnL. Max Profit = Initial Basis.
            # For Debit: Profit is positive PnL. Cost = Initial Basis.
            
            pnl_pct = unit_pnl / initial_basis
            
            # Take Profit
            if args.take_profit_pct and pnl_pct >= args.take_profit_pct:
                triggered = True
                close_reason = f"Take Profit ({pnl_pct:.1%})"
            
            # Stop Loss
            # Stop loss is usually defined as losing X% of the basis.
            # So PnL % <= -X%
            elif args.stop_loss_pct and pnl_pct <= -args.stop_loss_pct:
                triggered = True
                close_reason = f"Stop Loss ({pnl_pct:.1%})"
            
            if triggered:
                exit_cash_flow = current_exit_cash_flow
                leg_details = temp_leg_details
                close_timestamp = str(ts)
                break
        
        if not triggered:
            # If loop finishes without trigger, use the last calculated values from the loop
            if temp_leg_details:
                exit_cash_flow = current_exit_cash_flow
                leg_details = temp_leg_details
                # close_timestamp is already set to args.date (exit time) by default
            else:
                # Fallback if no valid timestamps found
                use_step_logic = False

    if not use_step_logic:
        # Standard logic: Take last available price in the window
        if col_map['symbol'] in day_df.columns:
            day_df = day_df.drop_duplicates(subset=[col_map['symbol']], keep='last')
            
        for leg in legs:
            symbol = leg['symbol']
            side = leg['side'] # buy or sell
            
            # Find row
            row = day_df[day_df[col_map['symbol']] == symbol]
            
            if row.empty:
                available = day_df[col_map['symbol']].unique()[:10]
                print(f"Error: Symbol {symbol} not found in data for {args.date}.", file=sys.stderr)
                print(f"Available symbols (first 10): {available}", file=sys.stderr)
                sys.exit(1)
            else:
                r = row.iloc[0]
                bid = float(r.get(col_map['bid'], 0))
                ask = float(r.get(col_map['ask'], 0))
                mark = float(r.get(col_map['mark'], 0))
                last = float(r.get(col_map['last'], 0))
                
                print(f"[DEBUG] {symbol} @ {args.date}: Bid={bid}, Ask={ask}, Mark={mark}, Last={last}", file=sys.stderr)
                
                # Close Price Logic
                if side == 'buy':
                    close_price = bid if bid > 0 else mark if mark > 0 else last
                else:
                    close_price = ask if ask > 0 else mark if mark > 0 else last
                
                if close_price <= 0:
                    print(f"Error: Found symbol {symbol} but valid close price is 0.0.", file=sys.stderr)
                    print(f"  Prices found: Bid={bid}, Ask={ask}, Mark={mark}, Last={last}", file=sys.stderr)
                    sys.exit(1)
                
                print(f"[DEBUG] {symbol} Close Price: {close_price} (Side: {side})", file=sys.stderr)
            
            # Calculate Cash Flow impact
            if side == 'buy':
                leg_flow = close_price
            else:
                leg_flow = -close_price
                
            exit_cash_flow += leg_flow
            
            leg_details.append({
                "symbol": symbol,
                "side": side,
                "close_price": close_price,
                "leg_cash_flow": leg_flow,
                "market_data": {
                    "bid": bid,
                    "ask": ask,
                    "mark": mark,
                    "last": last
                }
            })

    # Total PnL
    # Multiplier 100 for options
    unit_pnl = entry_cash_flow + exit_cash_flow
    total_pnl = unit_pnl * quantity * 100
    
    result = {
        "order_id": order.get('id'),
        "close_date": str(close_timestamp),
        "close_reason": close_reason,
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
        print(f"Close Date: {close_timestamp} ({close_reason})")
        print(f"Entry Cash Flow: ${entry_cash_flow:.2f}")
        print("Leg Prices at Open:")
        for leg in legs:
            open_price = float(leg.get('estimated_price', 0.0))
            print(f"  {leg.get('symbol')} ({leg.get('side')}): ${open_price:.2f}")
        print(f"Exit Cash Flow:  ${exit_cash_flow:.2f}")
        print("Leg Prices at Close:")
        for leg in leg_details:
            print(f"  {leg['symbol']} ({leg['side']}): ${leg['close_price']:.2f}")
        print(f"Unit P&L:        ${unit_pnl:.2f}")
        print(f"Quantity:        {quantity}")
        print(f"Total P&L:       ${total_pnl:.2f}")

if __name__ == "__main__":
    main()
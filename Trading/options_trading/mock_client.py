import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import databento
except ImportError:
    databento = None

class MockOptionClient:
    """Mock client to simulate AlpacaClient using historical Parquet data."""
    def __init__(self, parquet_file: str, target_date: datetime, save_order_file: str = None, underlying_file: str = None, deduplicate: bool = True):
        if pd is None:
            raise ImportError("pandas is required for MockOptionClient")
        
        self.target_date = target_date
        self.save_order_file = save_order_file

        # Load Options Data
        self.df = self._load_dataset(parquet_file, target_date, required=True)
        
        # Detect date column if not already set (e.g. from directory load)
        date_col = None
        if date_col is None:
            for col in ['ts_event', 'date', 'quote_date', 'timestamp', 'time']:
                if col in self.df.columns:
                    date_col = col
                    break

        # Ensure date column is datetime for later time filtering
        if date_col and not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            try:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
            except Exception:
                pass
        
        # Ensure required columns exist or map them
        self.col_map = {
            'symbol': 'symbol', # Option symbol
            'root': 'root',     # Underlying symbol
            'strike': 'strike',
            'expiration': 'expiration',
            'type': 'type',
            'bid': 'bid',
            'ask': 'ask',
            'underlying': 'underlying_last' # Common name, fallback checked later
        }
        
        # Adjust map based on actual columns
        cols = self.df.columns
        if 'contract_id' in cols:
            self.col_map['symbol'] = 'contract_id'
            if 'symbol' in cols:
                self.col_map['root'] = 'symbol'
        
        if 'underlying_price' in cols: self.col_map['underlying'] = 'underlying_price'
        elif 'underlying_last' in cols: self.col_map['underlying'] = 'underlying_last'
        elif 'spot' in cols: self.col_map['underlying'] = 'spot'
        elif 'price' in cols: self.col_map['underlying'] = 'price'
        elif 'last' in cols: self.col_map['underlying'] = 'last'
        elif 'close' in cols: self.col_map['underlying'] = 'close'
        elif 'mark' in cols: self.col_map['underlying'] = 'mark'
        
        if 'strike_price' in cols: self.col_map['strike'] = 'strike_price'
        if 'expiration_date' in cols: self.col_map['expiration'] = 'expiration_date'
        if 'option_type' in cols: self.col_map['type'] = 'option_type'
        
        # If loading stock data directly (no contract_id), map root to symbol for filtering
        if 'symbol' in cols and 'contract_id' not in cols and 'root' not in cols:
            self.col_map['root'] = 'symbol'
        
        # Databento mapping
        if 'bid_px_00' in cols: self.col_map['bid'] = 'bid_px_00'
        if 'ask_px_00' in cols: self.col_map['ask'] = 'ask_px_00'
        
        # Parse OCC symbols if metadata columns are missing
        missing_meta = any(self.col_map[k] not in cols for k in ['root', 'strike', 'expiration', 'type'])
        if missing_meta and 'symbol' in cols:
            self._parse_occ_symbols()
            
        # Time-based filtering (Minute data support)
        if date_col:
            if target_date.time() != datetime.min.time():
                 compare_date = target_date
                 # Handle timezone mismatch
                 if pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                     df_tz = self.df[date_col].dt.tz
                     if df_tz is not None and target_date.tzinfo is None:
                         # Data is TZ-aware (likely UTC), Input is Naive.
                         # Assume Input is ET (Market Time) -> Convert to Data TZ
                         try:
                             # Attempt to treat input as NY time
                             ts = pd.Timestamp(target_date).tz_localize('America/New_York')
                             compare_date = ts.tz_convert(df_tz)
                             print(f"Assuming input time {target_date} is ET. Converted to {compare_date} for filtering.")
                         except Exception:
                             # Fallback: Treat as UTC
                             compare_date = pd.Timestamp(target_date).tz_localize('UTC')
                     elif df_tz is None and target_date.tzinfo is not None:
                         compare_date = target_date.replace(tzinfo=None)
                 # Filter <= target_date if a specific time is provided
                 self.df = self.df[self.df[date_col] <= compare_date]
                 print(f"Filtered data <= {compare_date}: {len(self.df)} rows remaining")
            
            # Always deduplicate to ensure unique contracts (keep latest snapshot)
            symbol_col = self.col_map['symbol']
            if deduplicate and symbol_col in self.df.columns:
                 self.df = self.df.sort_values(date_col)
                 self.df = self.df.drop_duplicates(subset=[symbol_col], keep='last')
        
        # Load Underlying Data if provided
        self.underlying_df = None
        if underlying_file:
            self.underlying_df = self._load_dataset(underlying_file, target_date, required=False)
            if self.underlying_df is not None:
                # Apply time filtering to underlying data
                u_date_col = next((c for c in ['ts_event', 'date', 'quote_date', 'timestamp', 'time'] if c in self.underlying_df.columns), None)
                if u_date_col and target_date.time() != datetime.min.time():
                    compare_date = target_date
                    if pd.api.types.is_datetime64_any_dtype(self.underlying_df[u_date_col]):
                        df_tz = self.underlying_df[u_date_col].dt.tz
                        if df_tz is not None and target_date.tzinfo is None:
                            try:
                                ts = pd.Timestamp(target_date).tz_localize('America/New_York')
                                compare_date = ts.tz_convert(df_tz)
                            except:
                                compare_date = pd.Timestamp(target_date).tz_localize('UTC')
                        elif df_tz is None and target_date.tzinfo is not None:
                            compare_date = target_date.replace(tzinfo=None)
                    self.underlying_df = self.underlying_df[self.underlying_df[u_date_col] <= compare_date]

    def _parse_occ_symbols(self):
        """Parses OCC symbols (e.g. AAPL230616C00150000) into metadata columns."""
        # Regex: Root(chars) + Date(6 digits) + Type(C/P) + Strike(8 digits)
        # Make regex case insensitive for root and type
        # Allow spaces between root and date (common in some data feeds)
        extracted = self.df['symbol'].str.extract(r'^([a-zA-Z]+)\s*(\d{6})([cCpP])(\d{8})$')
        if not extracted.empty:
            if self.col_map['root'] not in self.df.columns:
                self.df[self.col_map['root']] = extracted[0].str.upper()
            if self.col_map['expiration'] not in self.df.columns:
                # Convert to string YYYY-MM-DD for consistency
                self.df[self.col_map['expiration']] = pd.to_datetime(extracted[1], format='%y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
            if self.col_map['type'] not in self.df.columns:
                self.df[self.col_map['type']] = extracted[2].str.upper().map({'C': 'call', 'P': 'put'})
            if self.col_map['strike'] not in self.df.columns:
                self.df[self.col_map['strike']] = extracted[3].astype(float) / 1000.0
            
            if self.col_map['root'] in self.df.columns and self.df[self.col_map['root']].isna().all() and not self.df.empty:
                print(f"Debug: Symbol parsing failed for all rows. Sample symbol: '{self.df['symbol'].iloc[0]}'")

    def _load_dataset(self, path_str: str, target_date: datetime, required: bool = True) -> Optional[pd.DataFrame]:
        path_input = Path(path_str)
        file_path = path_input
        
        # Directory handling: Look for daily files
        if path_input.is_dir():
            year_str = str(target_date.year)
            date_str = target_date.strftime('%Y-%m-%d')
            # Try Parquet structure: folder/YYYY/YYYY-MM-DD.parquet
            p1 = path_input / year_str / f"{date_str}.parquet"
            if p1.exists():
                file_path = p1
            else:
                # Try Databento structure: folder/YYYYMMDD.dbn.zst
                dbn_date = target_date.strftime('%Y%m%d')
                file_path = path_input / f"{dbn_date}.dbn.zst"
        
        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"Data file not found: {file_path}")
            else:
                print(f"Warning: Data file {file_path} not found")
                return None
                
        print(f"Loading data from {file_path}...")
        try:
            if str(file_path).endswith('.dbn.zst'):
                if not databento:
                    raise ImportError("databento module required for .dbn.zst files")
                df = databento.DBNStore.from_file(file_path).to_df()
                df.reset_index(inplace=True)
            else:
                df = pd.read_parquet(file_path)
        except Exception as e:
            if required:
                raise e
            print(f"Error loading data: {e}")
            return None
            
        # Filter by date if single file (legacy support for bulk files loaded directly)
        if not path_input.is_dir():
            date_col = next((c for c in ['date', 'quote_date', 'timestamp', 'time', 'ts_event'] if c in df.columns), None)
            if date_col:
                target_str = target_date.strftime('%Y-%m-%d')
                try:
                    # Check if we need to filter (simple string check)
                    mask = df[date_col].astype(str).str.startswith(target_str)
                    if not mask.all():
                        df = df[mask].copy()
                        print(f"Filtered data for {target_str}: {len(df)} rows")
                except Exception:
                    pass

        return df

    def get_stock_latest_trade(self, symbol: str):
        if self.underlying_df is not None and not self.underlying_df.empty:
            # Check for symbol match if column exists
            subset = self.underlying_df
            if 'symbol' in self.underlying_df.columns:
                subset = self.underlying_df[self.underlying_df['symbol'] == symbol]
            
            if not subset.empty:
                # Get last row (latest time)
                row = subset.iloc[-1]
                for col in ['price', 'last', 'close', 'mark']:
                    if col in row:
                        return {'p': float(row[col])}

        # Try to find underlying price from option data
        # Look for rows matching the root symbol
        
        # 1. Try explicit column
        if self.col_map['underlying'] in self.df.columns:
            if self.col_map['root'] in self.df.columns:
                subset = self.df[self.df[self.col_map['root']] == symbol]
                if not subset.empty:
                    price = subset[self.col_map['underlying']].iloc[0]
                    return {'p': float(price)}
            
            price = self.df[self.col_map['underlying']].iloc[0]
            return {'p': float(price)}
            
        # 2. Infer from Deep ITM Call (Delta ~ 1)
        # or Put-Call Parity (ATM)
        if self.col_map['root'] in self.df.columns:
            subset = self.df[self.df[self.col_map['root']] == symbol]
        else:
            subset = self.df
            
        # Try Put-Call Parity first (S = K + C - P)
        if not subset.empty:
            strike_col = self.col_map['strike']
            type_col = self.col_map['type']
            bid_col = self.col_map['bid']
            ask_col = self.col_map['ask']
            
            if strike_col in subset.columns and type_col in subset.columns:
                # Calculate mid price
                bids = subset[bid_col].fillna(0)
                asks = subset[ask_col].fillna(0)
                mids = (bids + asks) / 2
                
                df_calc = subset.copy()
                df_calc['mid'] = mids
                df_calc = df_calc[df_calc['mid'] > 0]
                
                calls = df_calc[df_calc[type_col] == 'call'][[strike_col, 'mid', self.col_map['expiration']]]
                puts = df_calc[df_calc[type_col] == 'put'][[strike_col, 'mid', self.col_map['expiration']]]
                
                merged = pd.merge(calls, puts, on=[strike_col, self.col_map['expiration']], suffixes=('_c', '_p'))
                if not merged.empty:
                    # S = K + C - P
                    merged['implied_S'] = merged[strike_col] + merged['mid_c'] - merged['mid_p']
                    inferred_price = merged['implied_S'].median()
                    print(f"Inferred underlying price for {symbol}: {inferred_price:.2f} (Put-Call Parity)")
                    return {'p': inferred_price}

        # Debugging if price not found
        if subset.empty and not self.df.empty:
             # Fallback: if subset is empty (root mismatch), try using the whole dataframe if it's small or we are desperate
             # This handles cases where 'root' column parsing failed but the file contains the data
             subset = self.df
             # Retry parity logic with full df? No, too risky if multiple symbols.

        if not subset.empty:
            calls = subset[subset[self.col_map['type']] == 'call']
            if not calls.empty:
                best_call = None
                if 'delta' in calls.columns:
                    deep_itm = calls[calls['delta'] > 0.9]
                    if not deep_itm.empty:
                        best_call = deep_itm.sort_values('delta', ascending=False).iloc[0]
                
                if best_call is None:
                    best_call = calls.sort_values(self.col_map['strike'], ascending=True).iloc[0]
                
                strike = float(best_call[self.col_map['strike']])
                opt_price = 0.0
                
                if 'mark' in best_call:
                    opt_price = float(best_call['mark'])
                elif self.col_map['bid'] in best_call and self.col_map['ask'] in best_call:
                    bid = float(best_call[self.col_map['bid']])
                    ask = float(best_call[self.col_map['ask']])
                    opt_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
                
                if opt_price > 0:
                    inferred_price = strike + opt_price
                    print(f"Inferred underlying price for {symbol}: {inferred_price:.2f} (from ITM Call)")
                    return {'p': inferred_price}
        
        print(f"Warning: Could not determine underlying price for {symbol}.")
        print(f"  Dataframe size: {len(self.df)}")
        print(f"  Subset size (root={symbol}): {len(subset)}")
        if not subset.empty:
             print(f"  Subset columns: {list(subset.columns)}")
             # print(f"  Sample row: {subset.iloc[0].to_dict()}")

        return {'p': 0.0}

    def get_stock_latest_quote(self, symbol: str):
        if self.underlying_df is not None and not self.underlying_df.empty:
            subset = self.underlying_df
            if 'symbol' in self.underlying_df.columns:
                subset = self.underlying_df[self.underlying_df['symbol'] == symbol]
            
            if not subset.empty:
                row = subset.iloc[-1]
                bp = 0.0
                ap = 0.0
                # Check for Databento or standard columns
                if 'bid_px_00' in row: bp = float(row['bid_px_00'])
                elif 'bid' in row: bp = float(row['bid'])
                
                if 'ask_px_00' in row: ap = float(row['ask_px_00'])
                elif 'ask' in row: ap = float(row['ask'])
                
                return {'bp': bp, 'ap': ap}
        
        # Check self.df (if it contains stock data)
        if self.df is not None and not self.df.empty:
             subset = self.df
             if 'symbol' in self.df.columns:
                 subset = self.df[self.df['symbol'] == symbol]
             
             if not subset.empty:
                row = subset.iloc[-1]
                bp = 0.0
                ap = 0.0
                if 'bid_px_00' in row: bp = float(row['bid_px_00'])
                elif 'bid' in row: bp = float(row['bid'])
                if 'ask_px_00' in row: ap = float(row['ask_px_00'])
                elif 'ask' in row: ap = float(row['ask'])
                return {'bp': bp, 'ap': ap}

        return {}

    def get_stock_snapshot(self, symbol: str):
        trade = self.get_stock_latest_trade(symbol)
        return {'latestTrade': trade, 'dailyBar': {'c': trade['p']}}

    def get_option_contracts(self, underlying_symbol, expiration_date_gte, expiration_date_lte, strike_price_gte, strike_price_lte, limit=None, status=None, type=None):
        # Filter dataframe
        mask = pd.Series(True, index=self.df.index)
        
        # 0. Root Symbol
        if self.col_map['root'] in self.df.columns:
            mask &= (self.df[self.col_map['root']] == underlying_symbol)
        
        # Ensure expiration column is string for comparison
        exp_col = self.col_map['expiration']
        if exp_col in self.df.columns:
            # If it's not object/string, convert it temporarily or ensure comparison works
            # Safest is to assume the column is YYYY-MM-DD string as set by _parse_occ_symbols
            # If it came from parquet as datetime, we might need to convert.
            if pd.api.types.is_datetime64_any_dtype(self.df[exp_col]):
                 mask &= (self.df[exp_col] >= pd.to_datetime(expiration_date_gte)) & (self.df[exp_col] <= pd.to_datetime(expiration_date_lte))
            else:
                 mask &= (self.df[exp_col].astype(str) >= expiration_date_gte) & (self.df[exp_col].astype(str) <= expiration_date_lte)

        # 1. Expiration
        # (Handled above with type check)
        
        # 2. Strike
        mask &= (self.df[self.col_map['strike']] >= strike_price_gte) & (self.df[self.col_map['strike']] <= strike_price_lte)
        
        # 3. Type
        if type:
            mask &= (self.df[self.col_map['type']] == type.lower())

        filtered = self.df[mask]
        contracts = []
        for _, row in filtered.iterrows():
            contracts.append({
                'symbol': row[self.col_map['symbol']],
                'expiration_date': str(row[self.col_map['expiration']]),
                'strike_price': row[self.col_map['strike']],
                'type': row[self.col_map['type']],
                'open_interest': row.get('open_interest', 0)
            })
            
        if not contracts:
            print(f"Debug: No contracts found for {underlying_symbol} in range.")
            print(f"  Filters: Exp {expiration_date_gte}-{expiration_date_lte}, Strike {strike_price_gte}-{strike_price_lte}, Type {type}")
            print(f"  Dataframe size: {len(self.df)}")
            if self.col_map['root'] in self.df.columns:
                roots = self.df[self.col_map['root']].unique()
                print(f"  Available Roots: {roots[:10]}")
                
        return contracts

    def get_option_snapshot(self, symbol_or_symbols):
        symbols = symbol_or_symbols.split(',')
        snapshots = {}
        for sym in symbols:
            row = self.df[self.df[self.col_map['symbol']] == sym]
            if not row.empty:
                r = row.iloc[0]
                snapshots[sym] = {
                    'latestQuote': {'ap': r.get(self.col_map['ask'], 0), 'bp': r.get(self.col_map['bid'], 0)},
                    'latestTrade': {'p': (r.get(self.col_map['ask'], 0) + r.get(self.col_map['bid'], 0)) / 2}, # Mid as trade
                    'greeks': {
                        'delta': r.get('delta'),
                        'gamma': r.get('gamma'),
                        'theta': r.get('theta'),
                        'vega': r.get('vega'),
                        'implied_volatility': r.get('implied_volatility')
                    }
                }
        
        if len(symbols) == 1:
            return snapshots[symbols[0]]
        return snapshots

    def place_option_limit_order(self, **kwargs):
        return self._mock_order(**kwargs)

    def place_option_market_order(self, **kwargs):
        return self._mock_order(**kwargs)

    def _mock_order(self, **kwargs):
        order = {'id': 'mock_order_id', 'status': 'filled', 'filled_at': self.target_date.isoformat(), **kwargs}
        if self.save_order_file:
            with open(self.save_order_file, 'w') as f:
                json.dump(order, f, indent=2, default=str)
            print(f"Mock order saved to {self.save_order_file}")
        return order
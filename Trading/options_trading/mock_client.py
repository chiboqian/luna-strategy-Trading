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
    def __init__(self, parquet_file: str, target_date: datetime, save_order_file: str = None):
        if pd is None:
            raise ImportError("pandas is required for MockOptionClient")
        
        self.target_date = target_date
        self.save_order_file = save_order_file
        
        path_input = Path(parquet_file)
        if path_input.is_dir():
            # New structure: folder -> year -> date.parquet
            year_str = str(target_date.year)
            date_str = target_date.strftime('%Y-%m-%d')
            file_path = path_input / year_str / f"{date_str}.parquet"
            
            # Fallback to Databento format: YYYYMMDD.dbn.zst
            if not file_path.exists():
                dbn_date = target_date.strftime('%Y%m%d')
                file_path = path_input / f"{dbn_date}.dbn.zst"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Daily parquet file not found: {file_path}")
                
            print(f"Loading historical data from {file_path}...")
            if str(file_path).endswith('.dbn.zst'):
                if not databento:
                    raise ImportError("databento module required for .dbn.zst files")
                self.df = databento.DBNStore.from_file(file_path).to_df()
                self.df.reset_index(inplace=True)
            else:
                self.df = pd.read_parquet(file_path)
        else:
            print(f"Loading historical data from {parquet_file}...")
            self.df = pd.read_parquet(parquet_file)
            
            # Filter by date if 'date' or 'quote_date' column exists
            date_col = None
            for col in ['date', 'quote_date', 'timestamp', 'time']:
                if col in self.df.columns:
                    date_col = col
                    break
            
            if date_col:
                # Ensure date column matches target_date (string or datetime comparison)
                # Assuming format YYYY-MM-DD in target_date
                target_str = target_date.strftime('%Y-%m-%d')
                # Try string conversion for filtering
                mask = self.df[date_col].astype(str).str.startswith(target_str)
                self.df = self.df[mask].copy()
                print(f"Filtered data for {target_str}: {len(self.df)} rows")
                
                # Ensure date column is datetime for later time filtering
                if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
        
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
        
        if 'strike_price' in cols: self.col_map['strike'] = 'strike_price'
        if 'expiration_date' in cols: self.col_map['expiration'] = 'expiration_date'
        if 'option_type' in cols: self.col_map['type'] = 'option_type'
        
        # Databento mapping
        if 'bid_px_00' in cols: self.col_map['bid'] = 'bid_px_00'
        if 'ask_px_00' in cols: self.col_map['ask'] = 'ask_px_00'
        
        # Parse OCC symbols if metadata columns are missing
        if self.col_map['strike'] not in cols and 'symbol' in cols:
            self._parse_occ_symbols()
            
        # Time-based filtering (Minute data support)
        if date_col and target_date.time() != datetime.min.time():
             # Filter <= target_date
             self.df = self.df[self.df[date_col] <= target_date]
             # Keep last entry per symbol
             symbol_col = self.col_map['symbol']
             if symbol_col in self.df.columns:
                 self.df = self.df.sort_values(date_col).groupby(symbol_col).last().reset_index()

    def _parse_occ_symbols(self):
        """Parses OCC symbols (e.g. AAPL230616C00150000) into metadata columns."""
        # Regex: Root(chars) + Date(6 digits) + Type(C/P) + Strike(8 digits)
        extracted = self.df['symbol'].str.extract(r'^([A-Z]+)(\d{6})([CP])(\d{8})$')
        if not extracted.empty:
            self.df[self.col_map['root']] = extracted[0]
            self.df[self.col_map['expiration']] = pd.to_datetime(extracted[1], format='%y%m%d').dt.strftime('%Y-%m-%d')
            self.df[self.col_map['type']] = extracted[2].map({'C': 'call', 'P': 'put'})
            self.df[self.col_map['strike']] = extracted[3].astype(float) / 1000.0

    def get_stock_latest_trade(self, symbol: str):
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
        # Price = Strike + OptionPrice
        if self.col_map['root'] in self.df.columns:
            subset = self.df[self.df[self.col_map['root']] == symbol]
        else:
            subset = self.df
            
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
                
        return {'p': 0.0}

    def get_stock_snapshot(self, symbol: str):
        trade = self.get_stock_latest_trade(symbol)
        return {'latestTrade': trade, 'dailyBar': {'c': trade['p']}}

    def get_option_contracts(self, underlying_symbol, expiration_date_gte, expiration_date_lte, strike_price_gte, strike_price_lte, limit=None, status=None, type=None):
        # Filter dataframe
        mask = pd.Series(True, index=self.df.index)
        
        # 0. Root Symbol
        if self.col_map['root'] in self.df.columns:
            mask &= (self.df[self.col_map['root']] == underlying_symbol)
            
        # 1. Expiration
        mask &= (self.df[self.col_map['expiration']] >= expiration_date_gte) & (self.df[self.col_map['expiration']] <= expiration_date_lte)
        # 2. Strike
        mask &= (self.df[self.col_map['strike']] >= strike_price_gte) & (self.df[self.col_map['strike']] <= strike_price_lte)
        
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
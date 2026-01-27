#!/usr/bin/env python3
"""
Creates a Long Straddle position for a given symbol.

A Long Straddle is a neutral strategy that profits from significant volatility
in either direction.

Structure (2 Legs):
1. Buy ATM Call
2. Buy ATM Put

Profits if the stock moves significantly (more than the premium paid) in either direction.
Max Risk: Limited to the premium paid.
Max Profit: Unlimited (upside), Substantial (downside).

Usage:
    python util/straddle.py SPY --days 30 --quantity 1
    python util/straddle.py AAPL --amount 2000 --dry-run
"""

import sys
import argparse
import json
import yaml
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import pandas as pd
except ImportError:
    pd = None

# Add parent directory to path to import Trading modules
sys.path.insert(0, str(Path(__file__).parent.parent / "Trading"))
try:
    from alpaca_client import AlpacaClient
except ImportError:
    # Fallback if running from proper context
    try:
        from Trading.alpaca_client import AlpacaClient
    except ImportError:
        print("Error: Could not import AlpacaClient. Check python path.", file=sys.stderr)
        sys.exit(1)

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
            
            if not file_path.exists():
                raise FileNotFoundError(f"Daily parquet file not found: {file_path}")
                
            print(f"Loading historical data from {file_path}...")
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
                target_str = target_date.strftime('%Y-%m-%d')
                mask = self.df[date_col].astype(str).str.startswith(target_str)
                self.df = self.df[mask].copy()
                print(f"Filtered data for {target_str}: {len(self.df)} rows")
        
        # Ensure required columns exist or map them
        self.col_map = {
            'symbol': 'symbol', # Option symbol
            'root': 'root',     # Underlying symbol
            'strike': 'strike',
            'expiration': 'expiration',
            'type': 'type',
            'bid': 'bid',
            'ask': 'ask',
            'underlying': 'underlying_last'
        }
        
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

    def get_stock_latest_trade(self, symbol: str):
        if self.col_map['underlying'] in self.df.columns:
            if self.col_map['root'] in self.df.columns:
                subset = self.df[self.df[self.col_map['root']] == symbol]
                if not subset.empty:
                    price = subset[self.col_map['underlying']].iloc[0]
                    return {'p': float(price)}
            price = self.df[self.col_map['underlying']].iloc[0]
            return {'p': float(price)}
            
        # Infer from Deep ITM Call (Delta ~ 1)
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
        mask = pd.Series(True, index=self.df.index)
        if self.col_map['root'] in self.df.columns:
            mask &= (self.df[self.col_map['root']] == underlying_symbol)
        mask &= (self.df[self.col_map['expiration']] >= expiration_date_gte) & (self.df[self.col_map['expiration']] <= expiration_date_lte)
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
                    'latestTrade': {'p': (r.get(self.col_map['ask'], 0) + r.get(self.col_map['bid'], 0)) / 2},
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

def get_closest_contract(contracts: List[Dict], target_strike: float, option_type: str) -> Optional[Dict]:
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
    filtered.sort(key=lambda x: abs(float(x['strike_price']) - target_strike))
    return filtered[0]

def select_best_expiration(expirations: Dict, target_dte: int) -> str:
    sorted_exps = sorted(expirations.keys())
    friday_exps = []
    for exp in sorted_exps:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            if exp_date.weekday() == 4:
                friday_exps.append(exp)
        except:
            pass
    if friday_exps: return friday_exps[0]
    return sorted_exps[0] if sorted_exps else None

def print_payoff_diagram(current_price: float, strike: float, debit: float):
    """Prints an ASCII payoff diagram for Long Straddle."""
    # Range: +/- 20%
    low_price = strike * 0.80
    high_price = strike * 1.20
    price_step = (high_price - low_price) / 20
    
    break_even_low = strike - debit
    break_even_high = strike + debit
    
    print("\n--- Payoff Diagram (at Expiration) ---")
    print(f"       Risk | Profit")
    
    prices = []
    curr = high_price
    while curr >= low_price:
        prices.append(curr)
        curr -= price_step
        
    for p in prices:
        # Value of Call: max(0, p - strike)
        # Value of Put: max(0, strike - p)
        # Total Value: abs(p - strike)
        # PnL: Total Value - Debit
        pnl = abs(p - strike) - debit
        
        # Draw bar
        # Scale: max risk is debit.
        scale = debit if debit > 0 else 1.0
        bar_len = int(abs(pnl) / scale * 5) # Scale factor
        bar_len = min(bar_len, 20)
        
        marker = " "
        if abs(p - current_price) < price_step/2: marker = "← NOW"
        elif abs(p - break_even_low) < price_step/2: marker = "← BE"
        elif abs(p - break_even_high) < price_step/2: marker = "← BE"
        elif abs(p - strike) < price_step/2: marker = "← STRIKE"
        
        if pnl > 0:
            bar = " " * 10 + "|" + "#" * bar_len + " " + f"+${pnl:.2f} {marker}" 
        elif pnl < 0:
            bar = " " * (10 - bar_len) + "#" * bar_len + "|" + " " * 11 + f"-${abs(pnl):.2f} {marker}"
        else:
            bar = " " * 10 + "|" + " " * 11 + f" $0.00 {marker}"
            
        print(f"${p:7.2f} {bar}")

def main():
    # Load Config
    config_path = Path(__file__).parent.parent / "config" / "Options.yaml"
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass

    defaults = config.get("options", {}).get("straddle", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 14)
    default_amount = defaults.get("default_amount", 0.0)

    parser = argparse.ArgumentParser(description="Execute Long Straddle Strategy")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of contracts")
    parser.add_argument("--days", type=int, default=default_days, help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, help=f"Search window (days), default {default_window}")
    parser.add_argument("--amount", type=float, default=None, help=f"Notional dollar amount (max risk)")
    parser.add_argument("--limit-order", action="store_true", help="Use limit order at mid-price")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--parquet", type=str, help="Path to historical parquet file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    
    args = parser.parse_args()

    if args.quantity is not None:
        args.amount = 0.0
    elif args.amount is not None:
        pass
    elif default_amount > 0:
        args.amount = default_amount
    else:
        args.quantity = 1

    if args.date:
        reference_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        reference_date = datetime.now()

    if args.parquet:
        client = MockOptionClient(args.parquet, reference_date, args.save_order)
    else:
        client = AlpacaClient()

    symbol = args.symbol.upper()
    
    # 1. Get Spot Price
    current_price = 0.0
    try:
        trade = client.get_stock_latest_trade(symbol)
        if trade:
            current_price = float(trade.get('p', 0))
        if current_price <= 0:
            snap = client.get_stock_snapshot(symbol)
            if snap:
                current_price = float(snap.get('latestTrade', {}).get('p') or 
                                      snap.get('dailyBar', {}).get('c') or 0)
    except Exception as e:
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(symbol)
                if hasattr(ticker, 'fast_info') and ticker.fast_info.last_price:
                    current_price = float(ticker.fast_info.last_price)
            except:
                pass

    if current_price <= 0:
        err = {"error": f"Could not determine current price for {symbol}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    if not args.json:
        print(f"Symbol: {symbol}")
        print(f"Current Price: ${current_price:.2f}")

    # 2. Find Options
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    if not args.json:
        print(f"Searching for contracts expiring >= {start_date}...")

    try:
        # Narrow range for ATM
        min_strike = current_price * 0.90
        max_strike = current_price * 1.10
        
        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            limit=10000,
            status='active'
        )
    except Exception as e:
        err = {"error": f"Failed to fetch contracts: {e}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    if not contracts:
        err = {"error": f"No contracts found in window {start_date} to {end_date}"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Group by expiration
    expirations = {}
    for c in contracts:
        exp = c['expiration_date']
        if exp not in expirations: expirations[exp] = []
        expirations[exp].append(c)
    
    selected_exp = select_best_expiration(expirations, args.days)
    if not selected_exp:
        if args.json: print(json.dumps({"error": "No valid expirations"}))
        return
    
    exp_contracts = expirations[selected_exp]
    
    # Find ATM Strike (closest to current price) that has both Call and Put
    strikes = {}
    for c in exp_contracts:
        k = float(c['strike_price'])
        if k not in strikes: strikes[k] = {}
        strikes[k][c['type']] = c
        
    # Filter for strikes with both legs
    valid_strikes = [k for k, v in strikes.items() if 'call' in v and 'put' in v]
    
    if valid_strikes:
        best_strike = min(valid_strikes, key=lambda x: abs(x - current_price))
        atm_call = strikes[best_strike]['call']
        atm_put = strikes[best_strike]['put']
    else:
        err = {"error": "Could not find ATM contracts with both legs"}
        if args.json: print(json.dumps(err))
        else: print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Fetch snapshots for pricing
    leg_symbols = [atm_call['symbol'], atm_put['symbol']]
    joined_symbols = ",".join(leg_symbols)
    snapshots = client.get_option_snapshot(joined_symbols)
    if 'latestQuote' in snapshots: snapshots = {leg_symbols[0]: snapshots}

    total_cost = 0.0
    legs = []
    
    for contract in [atm_call, atm_put]:
        sym = contract['symbol']
        snap = snapshots.get(sym, {})
        quote = snap.get('latestQuote', {})
        ask = float(quote.get('ap') or 0)
        last = float(snap.get('latestTrade', {}).get('p') or 0)
        price = ask if ask > 0 else last
        
        total_cost += price
        legs.append({
            "symbol": sym,
            "side": "buy",
            "position_intent": "buy_to_open",
            "ratio_qty": 1,
            "estimated_price": price
        })

    # Auto-Quantity
    if args.amount and args.amount > 0:
        cost_per_straddle = total_cost * 100
        if cost_per_straddle > 0:
            args.quantity = max(1, int(args.amount // cost_per_straddle))

    if not args.json:
        print(f"\n--- Straddle Analysis ---")
        print(f"Expiration: {selected_exp}")
        print(f"Strike:     ${float(atm_call['strike_price']):.2f}")
        print(f"Net Debit:  ${total_cost:.2f}")
        print(f"Break Even: ${float(atm_call['strike_price']) - total_cost:.2f} / ${float(atm_call['strike_price']) + total_cost:.2f}")
        print(f"Quantity:   {args.quantity}")
        
        print_payoff_diagram(current_price, float(atm_call['strike_price']), total_cost)

    if args.dry_run:
        if args.json:
            print(json.dumps({"status": "dry_run", "legs": legs, "cost": total_cost}, indent=2))
        else:
            print("\nDry Run Complete.")
        return

    # Execute
    if not args.json:
        print(f"\nSubmitting order...")
        
    try:
        entry_cash_flow = -total_cost # Debit
        kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
        if isinstance(client, MockOptionClient):
            kwargs['entry_cash_flow'] = entry_cash_flow
        
        response = client.place_option_market_order(**kwargs)
            
        if args.json:
            print(json.dumps({"status": "executed", "order": response}, indent=2))
        else:
            print(f"Order Submitted: {response.get('id')}")
            
    except Exception as e:
        err = {"error": str(e)}
        if args.json: print(json.dumps(err))
        else: print(f"Execution Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
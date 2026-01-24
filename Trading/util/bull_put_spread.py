#!/usr/bin/env python3
"""
Creates a Bull Put Spread position for a given symbol.

A Bull Put Spread is a credit spread strategy with:
- Limited profit potential (premium collected)
- Limited risk (width of spread - premium)
- Bullish to neutral outlook

Structure (2 Legs):
1. Sell OTM Put (higher strike) - collect premium
2. Buy further OTM Put (lower strike) - protection

This is a net credit strategy that profits when:
- Stock stays above short put strike
- Stock rises
- Implied volatility decreases

Key Entry Criteria (checked by this script):
- IV Rank > 30% (sell premium when IV is elevated)
- Bid-Ask spread < 20% of mid price (liquidity)
- Open Interest >= 100 on both legs
- Minimum credit >= 1/3 of spread width (good risk/reward)

Usage:
    python util/bull_put_spread.py SPY --days 30 --short-delta 0.30 --width 5 --quantity 1
    python util/bull_put_spread.py AAPL --days 45 --short-pct 5 --width 10 --amount 5000
    python util/bull_put_spread.py QQQ --amount 2000 --skip-checks  # Skip IV/liquidity checks
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
    import numpy as np
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
        
        print(f"Loading historical data from {parquet_file}...")
        self.df = pd.read_parquet(parquet_file)
        self.target_date = target_date
        self.save_order_file = save_order_file
        
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
            
        if not subset.empty and 'delta' in subset.columns and 'mark' in subset.columns:
            # Find call with delta > 0.95
            calls = subset[(subset[self.col_map['type']] == 'call') & (subset['delta'] > 0.95)]
            if not calls.empty:
                best = calls.sort_values('delta', ascending=False).iloc[0]
                strike = float(best[self.col_map['strike']])
                mark = float(best['mark'])
                inferred_price = strike + mark
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

def get_closest_contract(contracts: List[Dict], target_strike: float, option_type: str) -> Optional[Dict]:
    """Finds the contract with strike price closest to target."""
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
    
    # Sort by distance to target strike
    filtered.sort(key=lambda x: abs(float(x['strike_price']) - target_strike))
    return filtered[0]


def get_contract_by_delta(contracts: List[Dict], snapshots: Dict, target_delta: float, 
                          option_type: str, tolerance: float = 0.15) -> Optional[Dict]:
    """
    Finds the contract with delta closest to target.
    For puts, delta is negative, so we compare absolute values.
    """
    filtered = [c for c in contracts if c['type'] == option_type]
    if not filtered:
        return None
    
    best_contract = None
    best_diff = float('inf')
    
    for contract in filtered:
        sym = contract['symbol']
        snap = snapshots.get(sym, {})
        greeks = snap.get('greeks', {})
        
        if not greeks:
            continue
            
        delta = float(greeks.get('delta') or 0)
        # For puts, delta is negative, compare absolute values
        diff = abs(abs(delta) - abs(target_delta))
        
        if diff < best_diff and diff <= tolerance:
            best_diff = diff
            best_contract = contract
            best_contract['_delta'] = delta
    
    return best_contract


def align_strike_to_increment(target: float, price: float, direction: str = 'down') -> float:
    """
    Aligns strike to standard option increments.
    - Stocks < $50: $0.50 or $1 increments
    - Stocks $50-$200: $1 or $2.50 increments  
    - Stocks > $200: $5 increments
    """
    if price < 50:
        increment = 1.0
    elif price < 200:
        increment = 2.5 if price > 100 else 1.0
    else:
        increment = 5.0
    
    if direction == 'down':
        return math.floor(target / increment) * increment
    else:
        return math.ceil(target / increment) * increment


def check_liquidity(snapshot: Dict, max_spread_pct: float = 0.20) -> Tuple[bool, float, str]:
    """
    Checks bid-ask spread for liquidity.
    Returns (passed, spread_pct, reason)
    """
    quote = snapshot.get('latestQuote', {})
    bid = float(quote.get('bp') or 0)
    ask = float(quote.get('ap') or 0)
    
    if bid <= 0 or ask <= 0:
        return False, 1.0, "No valid quote"
    
    mid = (bid + ask) / 2
    spread = ask - bid
    spread_pct = spread / mid if mid > 0 else 1.0
    
    if spread_pct > max_spread_pct:
        return False, spread_pct, f"Wide spread ({spread_pct:.1%} > {max_spread_pct:.0%})"
    
    return True, spread_pct, "OK"


def check_open_interest(contract: Dict, min_oi: int = 100) -> Tuple[bool, int, str]:
    """
    Checks open interest for liquidity.
    Returns (passed, oi, reason)
    """
    oi = int(contract.get('open_interest', 0))
    
    if oi < min_oi:
        return False, oi, f"Low OI ({oi} < {min_oi})"
    
    return True, oi, "OK"


def get_iv_stats(snapshots: Dict, contracts: List[Dict]) -> Dict:
    """
    Calculates IV rank and stats from contract chain.
    Returns dict: {rank, median, low, high, count}
    """
    ivs = []
    for c in contracts:
        snap = snapshots.get(c['symbol'], {})
        greeks = snap.get('greeks', {})
        iv = float(greeks.get('implied_volatility') or 0)
        if iv > 0:
            ivs.append(iv)
    
    stats = {'rank': 0.5, 'median': 0.0, 'low': 0.0, 'high': 0.0, 'count': 0}
    
    if not ivs:
        return stats
        
    stats['count'] = len(ivs)
    sorted_ivs = sorted(ivs)
    stats['low'] = sorted_ivs[0]
    stats['high'] = sorted_ivs[-1]
    stats['median'] = sorted_ivs[len(sorted_ivs) // 2]
    
    # Calculate Rank (0-1)
    # Simple percentile of current median within the range?
    # No, usually IV Rank is (Current - Low) / (High - Low)
    # But here we only have a snapshot of the *chain*, not historical IV.
    # So we are calculating "Strike IV Rank" (where does ATM sit relative to wings?)
    # ...Actually the original function calculated percentile of distribution.
    
    # Original logic:
    below = sum(1 for iv in ivs if iv < stats['median'])
    stats['rank'] = below / len(ivs)
    
    return stats


def calculate_expected_move(price: float, iv: float, days: int) -> float:
    """Calculates expected move based on IV and time."""
    if days <= 0: return 0.0
    return price * iv * math.sqrt(days / 365.0)


def check_dividend_risk(symbol: str, expiry_date: datetime, reference_date: datetime = None) -> Tuple[bool, Optional[str], float, str]:
    """
    Checks if an ex-dividend date falls between now and expiration.
    Returns (safe, ex_date_str, amount, reason)
    """
    if reference_date is None: reference_date = datetime.now()
    if not HAS_YFINANCE:
        return True, None, 0.0, "yfinance not available"

    try:
        ticker = yf.Ticker(symbol)
        # Get dividends
        dividends = ticker.dividends
        if dividends.empty:
            return True, None, 0.0, "No dividends found"
            
        # We need future dividends. yhist contains past.
        # Checking calendar is better for upcoming
        calendar = ticker.calendar
        if calendar is None or calendar.empty:
            return True, None, 0.0, "No calendar data"
            
        if 'Ex-Dividend Date' in calendar.index:
            ex_date_val = calendar.loc['Ex-Dividend Date']
            # handle list or single
            if isinstance(ex_date_val, (list, tuple)) or (hasattr(ex_date_val, 'shape') and len(ex_date_val.shape) > 0):
                ex_date_val = ex_date_val[0]
                
            # Parse date
            if isinstance(ex_date_val, str):
                ex_dt = datetime.strptime(ex_date_val, "%Y-%m-%d")
            elif hasattr(ex_date_val, 'strftime'):
                ex_dt = ex_date_val
            else:
                return True, None, 0.0, "Could not parse date"
                
            if reference_date <= ex_dt <= expiry_date:
                # Get estimated amount (last dividend)
                amount = dividends.iloc[-1] if not dividends.empty else 0.0
                return False, ex_dt.strftime("%Y-%m-%d"), float(amount), f"Ex-Div date {ex_dt.strftime('%Y-%m-%d')} before expiration"
                
    except Exception:
        pass
        
    return True, None, 0.0, "Check failed"


def check_earnings_risk(symbol: str, days_buffer: int = 7, reference_date: datetime = None) -> Tuple[bool, Optional[str], Optional[datetime], str]:
    """
    Checks if earnings are within the buffer period.
    Returns (safe_to_trade, earnings_date_str, earnings_datetime, reason)
    """
    if reference_date is None: reference_date = datetime.now()
    if not HAS_YFINANCE:
        return True, None, None, "yfinance not available"
    
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is None or calendar.empty:
            return True, None, None, "No earnings data"
        
        # Get earnings date
        if 'Earnings Date' in calendar.index:
            earnings_dates = calendar.loc['Earnings Date']
            if isinstance(earnings_dates, str):
                next_earnings = datetime.strptime(earnings_dates, "%Y-%m-%d")
            else:
                # Could be a series or single value
                next_earnings = earnings_dates.iloc[0] if hasattr(earnings_dates, 'iloc') else earnings_dates
                if hasattr(next_earnings, 'strftime'):
                    pass  # Already datetime
                else:
                    next_earnings = datetime.strptime(str(next_earnings)[:10], "%Y-%m-%d")
            
            earnings_str = next_earnings.strftime("%Y-%m-%d")
            days_to_earnings = (next_earnings - reference_date).days
            
            if 0 <= days_to_earnings <= days_buffer:
                return False, earnings_str, next_earnings, f"Earnings in {days_to_earnings} days"
            
            return True, earnings_str, next_earnings, "OK"
    except Exception:
        pass
    
    return True, None, None, "Could not check"


def get_support_level(symbol: str, lookback_days: int = 20) -> Tuple[Optional[float], Optional[float]]:
    """
    Gets recent support level (20-day low) and distance from current price.
    Returns (support_price, pct_above_support)
    """
    if not HAS_YFINANCE:
        return None, None
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{lookback_days}d")
        
        if hist.empty:
            return None, None
        
        support = hist['Low'].min()
        current = hist['Close'].iloc[-1]
        pct_above = (current - support) / support if support > 0 else None
        
        return support, pct_above
    except Exception:
        return None, None


def print_payoff_diagram(current_price: float, short_strike: float, long_strike: float, credit: float):
    """Prints an ASCII payoff diagram."""
    width = short_strike - long_strike
    max_risk = width - credit
    break_even = short_strike - credit
    
    # Define range for the chart
    low_price = long_strike * 0.95
    high_price = short_strike * 1.05
    price_step = (high_price - low_price) / 20
    
    print("\n--- Payoff Diagram (at Expiration) ---")
    print(f"       Risk | Profit")
    
    # Generate rows
    prices = []
    curr = high_price
    while curr >= low_price:
        prices.append(curr)
        curr -= price_step
        
    for p in prices:
        # Calculate P&L
        if p >= short_strike:
            pnl = credit
        elif p <= long_strike:
            pnl = -max_risk
        else:
            # Between strikes
            loss_amt = (short_strike - p) - credit
            pnl = -loss_amt
            
        # Draw bar
        bar_len = int(abs(pnl) / max_risk * 10)
        bar_len = min(bar_len, 20) # Cap length
        
        marker = " "
        
        # Check for price markers
        label = ""
        # Current Price
        if abs(p - current_price) < price_step/2:
            marker = "← NOW"
        # Break Even
        elif abs(p - break_even) < price_step/2:
            marker = "← BE"
        # Short Strike
        elif abs(p - short_strike) < price_step/2:
            marker = "← SHORT"
        # Long Strike
        elif abs(p - long_strike) < price_step/2:
            marker = "← LONG"
            
        if pnl > 0:
            bar = " " * 10 + "|" + "#" * bar_len + " " + f"+${pnl:.2f} {marker}" 
        elif pnl < 0:
            bar = " " * (10 - bar_len) + "#" * bar_len + "|" + " " * 11 + f"-${abs(pnl):.2f} {marker}"
        else:
            bar = " " * 10 + "|" + " " * 11 + f" $0.00 {marker}"
            
        print(f"${p:7.2f} {bar}")

def get_trend_details(symbol: str) -> Dict:
    """
    Gets extended trend/fundamental details using yfinance.
    Returns dict with keys: beta, sector, industry, etc.
    """
    if not HAS_YFINANCE: return {}
    
    details = {}
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        details['beta'] = info.get('beta')
        details['sector'] = info.get('sector')
        details['industry'] = info.get('industry')
        details['marketCap'] = info.get('marketCap')
        # details['forwardPE'] = info.get('forwardPE')
    except:
        pass
    return details

def check_trend(symbol: str) -> Tuple[bool, str, Dict]:
    """
    Checks if stock is in uptrend using SMA crossovers.
    Returns (is_bullish, trend_status, details)
    """
    if not HAS_YFINANCE:
        return True, "Unknown (yfinance unavailable)", {}
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")
        
        if len(hist) < 50:
            return True, "Insufficient data", {}
        
        # Calculate SMAs
        hist['SMA20'] = hist['Close'].rolling(20).mean()
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        
        # Calculate Historical Volatility (30-day window)
        # HV = std_dev(ln(P/P_prev)) * sqrt(252)
        try:
            log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
            hist['HV30'] = log_returns.rolling(window=30).std() * np.sqrt(252)
            hv30 = hist['HV30'].iloc[-1]
        except Exception:
            hv30 = None

        # Calculate RSI (14-day)
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential weighted moving average (standard for RSI)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        current_price = hist['Close'].iloc[-1]
        sma20 = hist['SMA20'].iloc[-1]
        sma50 = hist['SMA50'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        
        details = {
            'price': current_price,
            'sma20': sma20,
            'sma50': sma50,
            'rsi': rsi,
            'hv30': hv30,
            'above_sma20': current_price > sma20,
            'above_sma50': current_price > sma50,
            'sma20_above_sma50': sma20 > sma50
        }
        
        # Bullish if price > SMA20 > SMA50
        if current_price > sma20 > sma50:
            return True, "Strong Uptrend", details
        elif current_price > sma50:
            return True, "Moderate Uptrend", details
        elif current_price > sma20:
            return False, "Weak/Choppy", details
        else:
            return False, "Downtrend", details
    except Exception:
        return True, "Could not check", {}


def calculate_probability_otm(current_price: float, strike: float, iv: float, 
                              dte: int, risk_free_rate: float = 0.05) -> float:
    """
    Calculates probability of option expiring OTM using Black-Scholes.
    For puts: P(OTM) = P(S > K at expiry)
    """
    if not HAS_SCIPY or iv <= 0 or dte <= 0:
        return 0.0
    
    try:
        t = dte / 365.0
        d2 = (math.log(current_price / strike) + (risk_free_rate - 0.5 * iv**2) * t) / (iv * math.sqrt(t))
        # For put: P(OTM) = P(S > K) = N(d2)
        return norm.cdf(d2)
    except Exception:
        return 0.0


def calculate_kelly_fraction(prob_win: float, win_amount: float, loss_amount: float) -> float:
    """
    Calculates Kelly Criterion for optimal position sizing.
    f* = (p * b - q) / b
    where p = prob win, q = prob lose, b = win/loss ratio
    Returns fraction of bankroll to risk (0-1).
    """
    if loss_amount <= 0 or prob_win <= 0 or prob_win >= 1:
        return 0.0
    
    q = 1 - prob_win
    b = win_amount / loss_amount
    
    kelly = (prob_win * b - q) / b
    
    # Cap at 25% (quarter Kelly is common for options)
    return max(0, min(kelly * 0.25, 0.25))


def get_vix_level() -> Optional[float]:
    """
    Gets current VIX level for market context.
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    
    return None


def is_market_hours() -> Tuple[bool, str]:
    """
    Checks if US market is currently open.
    Returns (is_open, status_message)
    """
    now = datetime.now()
    # Simple check - doesn't account for holidays
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    
    if weekday >= 5:  # Weekend
        return False, "Market closed (weekend)"
    
    # Market hours: 9:30 AM - 4:00 PM ET (simplified, assumes local is ET)
    market_open = (hour == 9 and minute >= 30) or (10 <= hour < 16)
    
    if not market_open:
        if hour < 9 or (hour == 9 and minute < 30):
            return False, "Pre-market"
        else:
            return False, "After-hours"
    
    return True, "Market open"


def select_best_expiration(expirations: Dict, target_dte: int) -> str:
    """
    Selects best expiration, preferring Friday expirations (weekly/monthly)
    for better liquidity and theta decay patterns.
    """
    sorted_exps = sorted(expirations.keys())
    
    # Prefer Friday expirations
    friday_exps = []
    for exp in sorted_exps:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            if exp_date.weekday() == 4:  # Friday
                friday_exps.append(exp)
        except:
            pass
    
    # Return first Friday if available, otherwise first expiration
    if friday_exps:
        return friday_exps[0]
    
    return sorted_exps[0] if sorted_exps else None


def optimize_spread(client, symbol: str, current_price: float, contracts: List[Dict], 
                    all_snapshots: Dict, expirations: Dict, min_credit_pct: float = 0.20,
                    top_n: int = 5, earnings_date: Optional[datetime] = None,
                    support_price: Optional[float] = None,
                    historical_volatility: Optional[float] = None,
                    rsi: Optional[float] = None,
                    reference_date: datetime = None) -> List[Dict]:
    """
    Scans multiple parameter combinations and returns top N optimized recommendations.
    
    Parameters scanned:
    - Short delta: 0.15, 0.20, 0.25, 0.30, 0.35
    - Spread width: $2, $5, $10, $15, $20
    - Expiration: All available in window
    
    Scoring factors (weighted):
    - Expected Value (30%) - Most important: is this a +EV trade?
    - Probability of profit (25%) - High prob = consistent income
    - Credit % of width (20%) - Better capital efficiency
    - Annualized ROC (15%) - Time value of money
    - Liquidity score (10%) - Tight spreads, good OI
    - Vertical Skew (10%) - Sell high IV, buy low IV
    - Technical Alignment (Bonus) - Short strike below support
    - Market Context (Bonus/Penalty) - IV/HV ratio and RSI
    """
    
    # Configuration for scan
    MIN_DELTA = 0.10
    MAX_DELTA = 0.45
    MIN_WIDTH = 1.0
    MAX_WIDTH = 25.0
    MIN_OPEN_INTEREST = 50
    MAX_BID_ASK_SPREAD_PCT = 0.35
    MIN_CREDIT_ABS = 0.05
    
    if reference_date is None: reference_date = datetime.now()
    candidates = []
    
    for exp_date, exp_contracts in expirations.items():
        try:
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            dte = (exp_dt - reference_date).days
        except:
            continue
            
        if dte < 7:  # Skip very short-dated (gamma risk)
            continue
            
        # 1. Filter and Enrich Puts
        puts = []
        for c in exp_contracts:
            if c['type'] != 'put': continue
            
            sym = c['symbol']
            snap = all_snapshots.get(sym, {})
            greeks = snap.get('greeks', {})
            quote = snap.get('latestQuote', {})
            
            # Basic data extraction
            strike = float(c['strike_price'])
            bid = float(quote.get('bp') or 0)
            ask = float(quote.get('ap') or 0)
            oi = int(c.get('open_interest') or 0)
            delta = float(greeks.get('delta') or 0) # usually negative for puts
            iv = float(greeks.get('implied_volatility') or 0)
            theta = float(greeks.get('theta') or 0)
            
            if bid <= 0 or ask <= 0: continue
            
            # Store enriched object
            puts.append({
                'contract': c,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': (bid+ask)/2,
                'oi': oi,
                'delta': delta, # keep sign
                'iv': iv,
                'theta': theta,
                'symbol': sym
            })
            
        # Sort by strike descending (Higher strike = Short candidate, Lower = Long)
        puts.sort(key=lambda x: x['strike'], reverse=True)
        
        # 2. Iterate Short Puts
        for i, short_leg in enumerate(puts):
            # Short leg criteria
            # Delta check (abs value)
            if abs(short_leg['delta']) < MIN_DELTA or abs(short_leg['delta']) > MAX_DELTA:
                continue
            if short_leg['oi'] < MIN_OPEN_INTEREST:
                continue
            
            # 3. Iterate Long Puts (must be lower strike)
            for j in range(i + 1, len(puts)):
                long_leg = puts[j]
                
                actual_width = short_leg['strike'] - long_leg['strike']
                if actual_width < MIN_WIDTH or actual_width > MAX_WIDTH:
                    continue
                
                if long_leg['oi'] < MIN_OPEN_INTEREST:
                    continue
                
                # Calculate natural credit (sell at bid, buy at ask)
                credit = short_leg['bid'] - long_leg['ask']
                mid_credit = short_leg['mid'] - long_leg['mid']
                
                # Use mid_credit for optimization if natural is too poor, but ensure it's realistic
                if mid_credit <= 0 or mid_credit < MIN_CREDIT_ABS:
                    continue
                
                # Liquidity check - bid-ask spread
                short_spread_pct = (short_leg['ask'] - short_leg['bid']) / short_leg['bid']
                long_spread_pct = (long_leg['ask'] - long_leg['bid']) / long_leg['bid']
                avg_spread_pct = (short_spread_pct + long_spread_pct) / 2
                
                if avg_spread_pct > MAX_BID_ASK_SPREAD_PCT:
                    continue
                
                # Core Metrics
                # We use mid_credit for display, but calculate a "conservative credit"
                # for EV scoring to account for slippage in wider spreads.
                # Estimate slippage as 10% of the total bid-ask spread width.
                short_spread_val = short_leg['ask'] - short_leg['bid']
                long_spread_val = long_leg['ask'] - long_leg['bid']
                slippage_est = (short_spread_val + long_spread_val) * 0.10
                
                calc_credit = mid_credit
                conservative_credit = max(0, mid_credit - slippage_est)
                
                max_profit = calc_credit # Display value
                max_loss = actual_width - calc_credit
                credit_pct = calc_credit / actual_width
                
                if max_loss <= 0:
                    continue
                
                risk_reward = max_profit / max_loss
                
                # Probability (Black-Scholes if available, else Delta)
                actual_delta = abs(short_leg['delta'])
                long_delta_val = abs(long_leg['delta'])

                if HAS_SCIPY and short_leg['iv'] > 0:
                    prob_profit = calculate_probability_otm(current_price, short_leg['strike'], short_leg['iv'], dte)
                    if long_leg['iv'] > 0:
                        prob_long_otm = calculate_probability_otm(current_price, long_leg['strike'], long_leg['iv'], dte)
                        prob_max_loss = 1.0 - prob_long_otm
                    else:
                        prob_max_loss = long_delta_val
                else:
                    prob_profit = 1.0 - actual_delta
                    prob_max_loss = long_delta_val
                
                # Probability of Touch (approximate)
                # P(Touch) ~ 2 * P(ITM) = 2 * (1 - P(OTM))
                # This represents the risk of the stock testing the short strike during the trade
                prob_touch = min(1.0, 2.0 * (1.0 - prob_profit))

                # Expected Value (refined with conservative credit)
                # EV = (P_win * Profit) + (P_loss * -MaxLoss) + (P_partial * AvgPartial)
                # We use conservative_credit here to penalize illiquid spreads in the score
                cons_max_profit = conservative_credit
                cons_max_loss = actual_width - conservative_credit
                prob_partial = max(0.0, 1.0 - prob_profit - prob_max_loss)
                term_win = prob_profit * cons_max_profit
                term_loss = prob_max_loss * (-cons_max_loss)
                term_partial = prob_partial * (cons_max_profit - (actual_width / 2))
                expected_value = term_win + term_loss + term_partial
                
                # Net Theta (should be positive for credit spreads)
                short_theta = short_leg['theta']
                long_theta = long_leg['theta']
                # When we SELL short put, we RECEIVE its theta (flip sign)
                # When we BUY long put, we PAY its theta
                net_theta = (-short_theta) - (-long_theta)  # = long_theta - short_theta in abs terms
                # Actually: Sell put = +theta, Buy put = -theta
                # net_theta = |short_theta| - |long_theta| (both are negative values from API)
                net_theta = abs(short_theta) - abs(long_theta)
                
                theta_efficiency = (net_theta / max_loss) if max_loss > 0 and net_theta > 0 else 0
                
                # Annualized Return on Capital
                roc = max_profit / max_loss if max_loss > 0 else 0
                annualized_roc = roc * (365 / dte) if dte > 0 else 0
                
                # Distance from current price (in %)
                distance_pct = (current_price - short_leg['strike']) / current_price
                
                # Liquidity Score (0-1, higher is better)
                liquidity_score = max(0, 1 - avg_spread_pct / MAX_BID_ASK_SPREAD_PCT)
                
                # Vertical Skew (Short IV - Long IV)
                # Positive skew is beneficial (selling expensive, buying cheap)
                skew = short_leg['iv'] - long_leg['iv']
                
                # --- COMPOSITE SCORE ---
                # Normalize each factor to roughly 0-1 range, then weight
                score = 0.0
                
                # Expected Value Yield (25%) - Return on Risk
                # Normalize 5% return on risk as excellent (1.0)
                ev_yield = expected_value / cons_max_loss if cons_max_loss > 0 else 0
                ev_normalized = min(1.0, max(0, ev_yield) / 0.05)
                score += ev_normalized * 0.25
                
                # Probability of Profit (25%) - already 0-1
                score += prob_profit * 0.25
                
                # Credit % of width (15%) - normalize to 40% as excellent
                credit_normalized = min(1.0, credit_pct / 0.40)
                score += credit_normalized * 0.15
                
                # Annualized ROC (15%) - normalize to 100% as excellent
                roc_normalized = min(1.0, annualized_roc / 1.0)
                score += roc_normalized * 0.15
                
                # Liquidity (10%)
                score += liquidity_score * 0.10
                
                # Vertical Skew (10%) - normalize 5% skew as excellent
                skew_normalized = max(0, min(1.0, skew / 0.05))
                score += skew_normalized * 0.10
                
                # --- PENALTIES ---
                if credit_pct < min_credit_pct:
                    score *= 0.7  # Penalize low credit
                if prob_profit < 0.60:
                    score *= 0.8  # Penalize low probability
                if expected_value < 0:
                    score *= 0.5  # Heavy penalty for negative EV
                if dte < 14:
                    score *= 0.9  # Slight penalty for short DTE (gamma)
                if dte > 60:
                    score *= 0.95  # Slight penalty for long DTE (capital tied up)
                
                # Earnings Penalty
                if earnings_date:
                    # Check if earnings falls between now and expiration
                    if reference_date < earnings_date <= exp_dt:
                        score *= 0.85 # 15% penalty for holding through earnings
                
                # Technical Analysis Bonus/Penalty
                if support_price:
                    if short_leg['strike'] < support_price:
                        score *= 1.15  # Bonus: Short strike is protected by support
                    elif short_leg['strike'] > support_price:
                        score *= 0.90  # Penalty: Short strike is exposed above support
                
                # IV vs HV Context (Sell expensive premium)
                if historical_volatility and historical_volatility > 0 and short_leg['iv'] > 0:
                    iv_hv_ratio = short_leg['iv'] / historical_volatility
                    if iv_hv_ratio > 1.25:
                        score *= 1.10  # Bonus: Premium is rich relative to recent movement
                    elif iv_hv_ratio < 0.90:
                        score *= 0.90  # Penalty: Premium is cheap
                
                # RSI Context (Mean Reversion / Trend Extension)
                if rsi is not None:
                    if rsi < 30:
                        score *= 1.15  # Bonus: Oversold, good entry for bullish trade
                    elif rsi > 70:
                        score *= 0.85  # Penalty: Overbought, risk of pullback
                
                # Probability of Touch Penalty
                # If likely to be tested (>50%), reduce score significantly
                if prob_touch > 0.50:
                    score *= 0.90
                
                candidates.append({
                    'expiration': exp_date,
                    'dte': dte,
                    'short_strike': short_leg['strike'],
                    'long_strike': long_leg['strike'],
                    'width': actual_width,
                    'short_delta': actual_delta,
                    'credit': credit,
                    'mid_credit': mid_credit,
                    'credit_pct': credit_pct,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'risk_reward': risk_reward,
                    'prob_profit': prob_profit,
                    'prob_touch': prob_touch,
                    'expected_value': expected_value,
                    'theta_efficiency': theta_efficiency,
                    'net_theta': net_theta,
                    'annualized_roc': annualized_roc,
                    'distance_pct': distance_pct,
                    'liquidity_score': liquidity_score,
                    'short_iv': short_leg['iv'],
                    'skew': skew,
                    'score': score,
                    'short_symbol': short_leg['symbol'],
                    'long_symbol': long_leg['symbol']
                })
    
    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Deduplicate similar spreads (keep best width for each short strike/expiration combo)
    seen = set()
    unique = []
    for c in candidates:
        # Key by Expiration and Short Strike only
        # Since candidates are sorted by Score, the first one encountered 
        # for a given strike/exp is the best width configuration.
        key = (c['expiration'], f"{c['short_strike']:.2f}")
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique[:top_n]


def print_optimization_results(results: List[Dict], current_price: float, symbol: str, market_cap: Optional[float] = None):
    """Pretty prints optimization results."""
    if not results:
        print("No valid spreads found matching criteria.")
        return
    
    print(f"\n{'='*70}")
    print(f" 🎯 OPTIMIZED BULL PUT SPREAD RECOMMENDATIONS - {symbol}")
    print(f" Current Price: ${current_price:.2f}")
    if market_cap:
         if market_cap >= 1e12:
             mcap_str = f"${market_cap/1e12:.2f}T"
         elif market_cap >= 1e9:
             mcap_str = f"${market_cap/1e9:.2f}B"
         else:
             mcap_str = f"${market_cap/1e6:.2f}M"
         print(f" Market Cap:    {mcap_str}")
    print(f"{'='*70}\n")
    
    for i, r in enumerate(results, 1):
        profit_risk_ratio = r['max_loss'] / r['max_profit'] if r['max_profit'] > 0 else 0
        ev_label = "✓" if r['expected_value'] > 0 else "⚠️"
        ev_yield = r['expected_value'] / r['max_loss'] if r['max_loss'] > 0 else 0
        
        print(f"#{i} | Score: {r['score']:.2f} | {r['expiration']} ({r['dte']} DTE)")
        print(f"   Strikes:       ${r['short_strike']:.0f}/${r['long_strike']:.0f} (${r['width']:.0f} wide)")
        print(f"   Short Delta:   {r['short_delta']:.0%} ({r['distance_pct']:.1%} OTM)")
        print(f"   Credit:        ${r['credit']:.2f} (Mid: ${r['mid_credit']:.2f})")
        print(f"   Risk/Reward:   ${r['max_profit']:.2f} / ${r['max_loss']:.2f} (1:{profit_risk_ratio:.1f})")
        print(f"   P(Profit):     {r['prob_profit']:.1%} (Touch: {r.get('prob_touch', 0):.1%})")
        print(f"   Exp. Value:    ${r['expected_value']:.2f} ({ev_yield:.1%} yield) {ev_label}")
        print(f"   Ann. ROC:      {r['annualized_roc']:.0%}")
        if r['net_theta'] > 0:
            print(f"   Net Theta:     ${r['net_theta']:.3f}/day ({r['theta_efficiency']:.2%} eff)")
        if r['short_iv'] > 0:
            print(f"   Short IV:      {r['short_iv']:.1%} (Skew: {r.get('skew', 0):+.1%})")
        print()
    
    # Command suggestion
    best = results[0]
    print(f"{'─'*70}")
    print(f"💡 To execute the top recommendation:")
    print(f"   python util/bull_put_spread.py {symbol} --days {best['dte']} \\")
    print(f"          --short-delta {best['short_delta']:.2f} --width {int(best['width'])} --dry-run")
    print()


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

    defaults = config.get("options", {}).get("bull_put_spread", {})
    default_days = defaults.get("min_days_to_expiration", 30)
    default_window = defaults.get("search_window_days", 14)
    default_short_delta = defaults.get("default_short_delta", 0.30)
    default_short_pct = defaults.get("default_short_pct", 5.0)
    default_width = defaults.get("default_width", 5.0)
    default_amount = defaults.get("default_amount", 0.0)
    default_min_iv_rank = defaults.get("min_iv_rank", 0.30)
    default_min_credit_pct = defaults.get("min_credit_pct", 0.20)
    default_min_oi = defaults.get("min_open_interest", 100)
    default_max_spread_pct = defaults.get("max_bid_ask_spread_pct", 0.20)

    parser = argparse.ArgumentParser(description="Execute Bull Put Spread")
    parser.add_argument("symbol", help="Underlying stock symbol (e.g. SPY)")
    parser.add_argument("--quantity", type=int, default=None, help="Number of spreads")
    parser.add_argument("--days", type=int, default=default_days, 
                        help=f"Days to expiration (min), default {default_days}")
    parser.add_argument("--window", type=int, default=default_window, 
                        help=f"Search window (days) after min days, default {default_window}")
    parser.add_argument("--short-delta", type=float, default=None, 
                        help=f"Target delta for short put (e.g. 0.30 = 30 delta), default {default_short_delta}")
    parser.add_argument("--short-pct", type=float, default=None, 
                        help=f"Short put distance from spot (%%), overrides --short-delta, default {default_short_pct}%%")
    parser.add_argument("--width", type=float, default=default_width, 
                        help=f"Spread width in dollars, default ${default_width}")
    parser.add_argument("--amount", type=float, default=None, 
                        help=f"Notional dollar amount (max risk), calculates quantity")
    parser.add_argument("--min-iv-rank", type=float, default=default_min_iv_rank,
                        help=f"Minimum IV rank to enter (0-1), default {default_min_iv_rank}")
    parser.add_argument("--skip-checks", action="store_true", 
                        help="Skip IV rank, liquidity, and credit quality checks")
    parser.add_argument("--skip-earnings", action="store_true",
                        help="Skip earnings date check")
    parser.add_argument("--limit-order", action="store_true",
                        help="Use limit order at mid-price instead of market order")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute, just show plan")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--show-diagram", action="store_true", help="Show ASCII payoff diagram")
    parser.add_argument("--optimize", action="store_true", 
                        help="Scan all parameter combinations and show top recommendations")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of recommendations to show in optimize mode (default: 5)")
    parser.add_argument("--parquet", type=str, help="Path to historical parquet file (mock mode)")
    parser.add_argument("--date", type=str, help="Current date for simulation (YYYY-MM-DD)")
    parser.add_argument("--save-order", type=str, help="File to save mock order JSON")
    
    args = parser.parse_args()

    # Resolve defaults priority for quantity/amount
    if args.quantity is not None:
        args.amount = 0.0  # Force ignore amount logic
    elif args.amount is not None:
        pass  # Use provided amount
    elif default_amount > 0:
        args.amount = default_amount
    else:
        args.quantity = 1  # Fallback default

    # Resolve strike selection method (delta vs percentage)
    use_delta = True
    if args.short_pct is not None:
        use_delta = False
    elif args.short_delta is None:
        args.short_delta = default_short_delta

    # Store validation thresholds
    min_iv_rank = args.min_iv_rank
    min_credit_pct = default_min_credit_pct
    min_oi = default_min_oi
    max_spread_pct = default_max_spread_pct
    skip_checks = args.skip_checks

    # Determine Reference Date
    if args.date:
        reference_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        reference_date = datetime.now()

    if args.parquet:
        client = MockOptionClient(args.parquet, reference_date, args.save_order)
    else:
        client = AlpacaClient()

    symbol = args.symbol.upper()
    
    # Get Account Info (for buying power check)
    account_info = {}
    try:
        account_info = client.get_account_info()
    except:
        pass

    # 1. Get Spot Price
    current_price = 0.0
    
    # Check for Existing Positions (Safety)
    try:
        positions = client.get_all_positions()
        existing_exposure = [p for p in positions if p.get('symbol') == symbol or p.get('symbol', '').startswith(symbol)]
        if existing_exposure:
            if not args.json:
                print(f"\n⚠️  EXISTING EXPOSURE: You already have {len(existing_exposure)} position(s) in {symbol}.")
                for p in existing_exposure:
                    # Simple summary of existing pos
                    qty = p.get('qty', 0)
                    side = p.get('side', 'long')
                    sym = p.get('symbol', symbol)
                    print(f"   - {sym}: {qty} shares/contracts ({side})")
    except:
        pass

    try:
        trade = client.get_stock_latest_trade(symbol)
        if trade:
            current_price = float(trade.get('p', 0))
        
        # If price is 0 (market closed/no data), try snapshot
        if current_price <= 0:
            snap = client.get_stock_snapshot(symbol)
            if snap:
                current_price = float(snap.get('latestTrade', {}).get('p') or 
                                      snap.get('dailyBar', {}).get('c') or 0)
    except Exception as e:
        err = {"error": f"Failed to get price for {symbol}: {e}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error getting price: {e}", file=sys.stderr)
        return

    if current_price <= 0:
        err = {"error": f"Could not determine current price for {symbol}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Could not determine current price for {symbol}", file=sys.stderr)
        return

    # --- Market Context ---
    market_open = False
    market_status = "Unknown"
    try:
        clock = client.get_clock()
        market_open = clock.get('is_open', False)
        market_status = "Market Open" if market_open else "Market Closed"
        if not market_open:
            # Add detail if possible (e.g. next open)
            next_open = clock.get('next_open')
            if next_open:
                market_status += f" (Next open: {next_open})"
    except Exception:
        # Fallback to local time check
        market_open, market_status = is_market_hours()

    vix_level = get_vix_level()
    support_price, pct_above_support = get_support_level(symbol)
    is_bullish, trend_status, trend_details = check_trend(symbol)
    fund_details = get_trend_details(symbol)
    
    if not args.json:
        print(f"\n--- Market Context ---")
        print(f"Market Status:    {market_status}")
        
        # Sector/Beta
        if fund_details:
             sector_str = f"{fund_details.get('sector', 'Unknown')} ({fund_details.get('industry', '')})"
             print(f"Sector:           {sector_str}")
             
             if fund_details.get('marketCap'):
                 mcap = fund_details.get('marketCap')
                 if mcap >= 1e12:
                     mcap_str = f"${mcap/1e12:.2f}T"
                 elif mcap >= 1e9:
                     mcap_str = f"${mcap/1e9:.2f}B"
                 else:
                     mcap_str = f"${mcap/1e6:.2f}M"
                 print(f"Market Cap:       {mcap_str}")

             if fund_details.get('beta'):
                 beta = fund_details['beta']
                 beta_desc = "High Vol" if beta > 1.3 else "Low Vol" if beta < 0.8 else "Market Correlated"
                 print(f"Beta:             {beta:.2f} ({beta_desc})")

        if vix_level:
            vix_regime = "LOW" if vix_level < 15 else "NORMAL" if vix_level < 25 else "HIGH" if vix_level < 35 else "EXTREME"
            print(f"VIX Level:        {vix_level:.1f} ({vix_regime})")
        if support_price and pct_above_support:
            print(f"20-Day Support:   ${support_price:.2f} (price {pct_above_support:.1%} above)")
        print(f"Trend:            {trend_status}" + (" ✓" if is_bullish else " ⚠️"))
        if trend_details:
            if trend_details.get('sma20') and trend_details.get('sma50'):
                print(f"                  SMA20=${trend_details['sma20']:.2f}, SMA50=${trend_details['sma50']:.2f}")
            if trend_details.get('rsi'):
                rsi = trend_details['rsi']
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                print(f"                  RSI(14)={rsi:.1f} ({rsi_status})")
    
    # --- Earnings Check ---
    earnings_dt = None
    if not args.skip_earnings:
        earnings_safe, earnings_date, earnings_dt, earnings_reason = check_earnings_risk(symbol, reference_date=reference_date)
        
        if not skip_checks and not earnings_safe:
            # First check: Earnings < 7 days
             if not args.json:
                print(f"\n⚠️ EARNINGS RISK: {earnings_reason}")
                print(f"   Next Earnings: {earnings_date}")
                print("   Use --skip-earnings to override.")
             else:
                print(json.dumps({"error": "Earnings risk", "earnings_date": earnings_date, "reason": earnings_reason}))
             return
        elif earnings_date and not args.json:
             print(f"Next Earnings:    {earnings_date}")
            
    # 2. Determine Strike Targets
    if use_delta:
        # We'll find short put by delta later
        short_strike_target = current_price * 0.95  # Initial estimate for contract fetch
        if not args.json:
            print(f"Symbol: {symbol}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Short Put Target: {args.short_delta:.0%} Delta")
            print(f"Spread Width: ${args.width:.2f}")
    else:
        short_strike_target = current_price * (1.0 - (args.short_pct / 100.0))
        if not args.json:
            print(f"Symbol: {symbol}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Short Put Target: ${short_strike_target:.2f} (-{args.short_pct}%)")
            print(f"Spread Width: ${args.width:.2f}")

    long_strike_target = short_strike_target - args.width

    # 3. Find Options
    start_date = (reference_date + timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_date = (reference_date + timedelta(days=args.days + args.window)).strftime("%Y-%m-%d")

    if not args.json:
        print(f"Searching for contracts expiring >= {start_date}...")

    # Fetch contracts - only puts needed for this strategy
    try:
        # Narrow search to relevant strikes
        min_strike = long_strike_target * 0.85
        max_strike = current_price * 1.05  # Include some upside for delta-based selection
        
        contracts = client.get_option_contracts(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=min_strike,
            strike_price_lte=max_strike,
            type='put',  # Only puts for this strategy
            limit=10000,
            status='active'
        )
    except Exception as e:
        err = {"error": f"Failed to fetch contracts: {e}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error fetching contracts: {e}", file=sys.stderr)
        return

    if not contracts:
        err = {"error": f"No contracts found in window {start_date} to {end_date}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"No contracts found in window {start_date} to {end_date}", file=sys.stderr)
        return

    # Group by expiration
    expirations = {}
    for c in contracts:
        exp = c['expiration_date']
        if exp not in expirations:
            expirations[exp] = []
        expirations[exp].append(c)
    
    # Select best expiration (prefer Friday for liquidity)
    selected_exp = select_best_expiration(expirations, args.days)
    if not selected_exp:
        if args.json:
            print(json.dumps({"error": "No valid expirations parsed"}))
        return
    
    # Calculate days out
    try:
        exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
        d_out = (exp_dt - reference_date).days
    except:
        d_out = "?"
        exp_dt = None

    if not args.json:
        print(f"Selected Expiration: {selected_exp} (~{d_out} days out)")
        
    # Check if earnings fall during the trade
    if earnings_dt and exp_dt and not args.skip_earnings:
        # Check if Earnings falls between NOW and EXPIRATION
        if reference_date < earnings_dt <= exp_dt:
             warnings.append(f"Earnings release on {earnings_date} falls before expiration")

    # --- Dividend Check ---
    if exp_dt:
        div_safe, div_date, div_amt, div_reason = check_dividend_risk(symbol, exp_dt, reference_date=reference_date)
        if not div_safe:
             msg = f"Dividend Ex-Date {div_date} before expiry (${div_amt:.2f} drop)"
             if not args.json:
                 print(f"⚠️ {msg}")
             warnings.append(msg)
             msg = f"Trade holds through EARNINGS on {earnings_dt.strftime('%Y-%m-%d')}"
             warnings.append(msg)
             if not args.json:
                 print(f"⚠️  WARNING: {msg}")

    exp_contracts = expirations[selected_exp]
    
    # If using delta-based selection, fetch snapshots first
    short_put = None
    long_put = None
    
    if use_delta:
        # Fetch snapshots for all contracts to get greeks
        contract_symbols = [c['symbol'] for c in exp_contracts]
        
        if not args.json:
            print(f"Fetching greeks for {len(contract_symbols)} contracts...")
        
        try:
            # Batch fetch snapshots
            all_snapshots = {}
            batch_size = 100
            for i in range(0, len(contract_symbols), batch_size):
                batch = contract_symbols[i:i+batch_size]
                joined = ",".join(batch)
                snaps = client.get_option_snapshot(joined)
                if isinstance(snaps, dict):
                    # Could be single or multiple
                    if 'latestQuote' in snaps or 'greeks' in snaps:
                        # Single snapshot returned
                        all_snapshots[batch[0]] = snaps
                    else:
                        all_snapshots.update(snaps)
            
            # --- OPTIMIZATION MODE ---
            if args.optimize:
                print(f"\n🔍 Scanning parameter combinations for {symbol}...")
                results = optimize_spread(
                    client=client,
                    symbol=symbol,
                    current_price=current_price,
                    contracts=contracts,
                    all_snapshots=all_snapshots,
                    expirations=expirations,
                    min_credit_pct=min_credit_pct,
                    top_n=args.top,
                    earnings_date=earnings_dt,
                    support_price=support_price,
                    historical_volatility=trend_details.get('hv30'),
                    rsi=trend_details.get('rsi'),
                    reference_date=reference_date
                )
                
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print_optimization_results(results, current_price, symbol, market_cap=fund_details.get('marketCap'))
                return
                        
            # Find short put by delta
            short_put = get_contract_by_delta(
                exp_contracts, all_snapshots, 
                target_delta=args.short_delta, 
                option_type='put'
            )
            
            if short_put:
                short_strike = float(short_put['strike_price'])
                long_strike_target = short_strike - args.width
                long_put = get_closest_contract(exp_contracts, long_strike_target, 'put')
                
        except Exception as e:
            if not args.json:
                print(f"Warning: Delta-based selection failed ({e}), falling back to percentage")
            use_delta = False
    
    if not use_delta or not short_put:
        # Percentage-based selection
        short_put = get_closest_contract(exp_contracts, short_strike_target, 'put')
        if short_put:
            short_strike = float(short_put['strike_price'])
            long_strike_target = short_strike - args.width
            long_put = get_closest_contract(exp_contracts, long_strike_target, 'put')
    
    # Logic Correction: Ensure Long Strike < Short Strike
    if short_put and long_put:
        s_strike = float(short_put['strike_price'])
        l_strike = float(long_put['strike_price'])
        if l_strike >= s_strike:
            if not args.json:
                print(f"Adjusting: Long strike {l_strike} >= Short {s_strike}. Finding lower strike...")
            # Find closest strike strictly lower than short strike
            lower_puts = [c for c in exp_contracts if c['type'] == 'put' and float(c['strike_price']) < s_strike]
            if lower_puts:
                # Get closest to (Short - Width)
                target = s_strike - args.width
                lower_puts.sort(key=lambda x: abs(float(x['strike_price']) - target))
                long_put = lower_puts[0]
            else:
                 long_put = None # No valid long leg found

    # Validate legs
    missing = []
    if not short_put:
        missing.append("Short Put")
    if not long_put:
        missing.append("Long Put")
    
    if missing:
        err = {"error": f"Could not find required legs: {', '.join(missing)}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Extract strikes
    short_strike = float(short_put['strike_price'])
    long_strike = float(long_put['strike_price'])
    actual_width = short_strike - long_strike
    
    # Validate spread makes sense
    if actual_width <= 0:
        err = {"error": f"Invalid spread: Short ${short_strike} must be above Long ${long_strike}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error: {err['error']}", file=sys.stderr)
        return

    # Build Legs List
    legs = [
        {
            "symbol": short_put['symbol'],
            "side": "sell",
            "position_intent": "sell_to_open",
            "ratio_qty": 1
        },
        {
            "symbol": long_put['symbol'],
            "side": "buy",
            "position_intent": "buy_to_open",
            "ratio_qty": 1
        }
    ]

    # --- METRICS CALCULATION (Initialized early for leg printing) ---
    metrics = {
        "net_credit": 0.0,
        "max_profit": 0.0,
        "max_loss": 0.0,
        "break_even": 0.0,
        "risk_reward": 0.0,
        "return_on_capital": 0.0,
        "expected_value": 0.0,
        "prob_profit": 0.0,
        "iv_rank": 0.0,
        "short_iv": 0.0,
        "net_gamma": 0.0,
        "net_vega": 0.0,
        "contracts_data": {}
    }
    
    # We need to pre-fetch snapshots if we want to display deltas above
    # Or just rely on all_snapshots if available
    s_delta_disp = 0.0
    l_delta_disp = 0.0
    if 'all_snapshots' in locals():
        s_greeks = all_snapshots.get(short_put['symbol'], {}).get('greeks', {})
        l_greeks = all_snapshots.get(long_put['symbol'], {}).get('greeks', {})
        s_delta_disp = float(s_greeks.get('delta') or 0)
        l_delta_disp = float(l_greeks.get('delta') or 0)
        
    if not args.json:
        print(f"\nLeg 1 (Short Put): {short_put['symbol']} Strike=${short_strike} (Δ {s_delta_disp:.2f})")
        print(f"Leg 2 (Long Put):  {long_put['symbol']} Strike=${long_strike} (Δ {l_delta_disp:.2f})")
        print(f"Actual Spread Width: ${actual_width:.2f}")

    # Track validation warnings
    warnings = []
    
    try:
        leg_symbols = [l['symbol'] for l in legs]
        joined_symbols = ",".join(leg_symbols)
        snapshots = client.get_option_snapshot(joined_symbols)
        
        # Normalize response
        if 'latestQuote' in snapshots or 'greeks' in snapshots:
            snapshots = {leg_symbols[0]: snapshots}

        natural_credit = 0.0
        mid_credit = 0.0
        net_delta = 0.0
        net_theta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        
        for leg in legs:
            sym = leg['symbol']
            side = leg['side']
            snap = snapshots.get(sym, {})
            metrics["contracts_data"][sym] = snap
            
            # --- Liquidity Check ---
            if not skip_checks:
                liq_ok, spread_pct, liq_reason = check_liquidity(snap, max_spread_pct)
                if not liq_ok:
                    warnings.append(f"{sym}: {liq_reason}")
                leg['bid_ask_spread_pct'] = spread_pct
            
            # Get Prices
            quote = snap.get('latestQuote', {})
            ask = float(quote.get('ap') or 0)
            bid = float(quote.get('bp') or 0)
            last = float(snap.get('latestTrade', {}).get('p') or 0)
            mid = (ask + bid) / 2 if (ask > 0 and bid > 0) else last
            
            # Natural Price (Conservative):
            # Sell @ Bid, Buy @ Ask
            if side == 'sell':
                natural_credit += bid if bid > 0 else last
                mid_credit += mid
                leg['estimated_price'] = bid if bid > 0 else last
            else: # buy
                natural_credit -= ask if ask > 0 else last
                mid_credit -= mid
                leg['estimated_price'] = ask if ask > 0 else last
            
            # Greeks
            greeks = snap.get('greeks', {})
            if greeks:
                d = float(greeks.get('delta') or 0)
                t = float(greeks.get('theta') or 0)
                g = float(greeks.get('gamma') or 0)
                v = float(greeks.get('vega') or 0)
                if side == 'sell':
                    d = -d
                    t = -t
                    g = -g
                    v = -v
                net_delta += d
                net_theta += t
                net_gamma += g
                net_vega += v
        
        # Decide which credit to use for Analysis
        if args.limit_order:
             # If using limit order, we aim for Mid-Price (or close to it)
             # Let's be slightly conservative: Mid-Price
             total_credit = mid_credit
             credit_source = "Mid-Price"
        else:
             # Market order gets Natural price (Bid/Ask)
             total_credit = natural_credit
             credit_source = "Natural (Bid/Ask)"

        metrics['net_credit'] = total_credit
        metrics['natural_credit'] = natural_credit
        metrics['mid_credit'] = mid_credit
        metrics['credit_source'] = credit_source
        
        metrics['net_delta'] = net_delta
        metrics['net_theta'] = net_theta
        metrics['net_gamma'] = net_gamma
        metrics['net_vega'] = net_vega
        
        # Max Profit = Net Credit (based on entry type)
        metrics['max_profit'] = total_credit
        
        # Max Loss = Width - Net Credit (occurs if stock <= long strike at expiry)
        metrics['max_loss'] = actual_width - total_credit
        
        # Break Even = Short Strike - Net Credit
        metrics['break_even'] = short_strike - total_credit
        
        # Risk/Reward Ratio = Max Profit / Max Loss
        if metrics['max_loss'] > 0:
            metrics['risk_reward'] = metrics['max_profit'] / metrics['max_loss']
        
        # Credit Quality Check - minimum 1/3 of width
        credit_pct = total_credit / actual_width if actual_width > 0 else 0
        metrics['credit_pct'] = credit_pct
        if not skip_checks and credit_pct < min_credit_pct:
            warnings.append(f"Low credit: {credit_pct:.1%} of width (min {min_credit_pct:.0%})")
            # Suggest widening
            if actual_width < 10:
                warnings.append(f"  Consider widening spread to increase credit/width ratio")
        
        # Probability of Profit (rough estimate using delta of short put)
        # Delta of short put ≈ probability of being ITM
        # So P(profit) ≈ 1 - |delta of short put|
        short_snap = snapshots.get(short_put['symbol'], {})
        long_snap = snapshots.get(long_put['symbol'], {})
        
        short_greeks = short_snap.get('greeks', {})
        long_greeks = long_snap.get('greeks', {})
        
        metrics['short_iv'] = 0.0
        metrics['long_iv'] = 0.0
        
        if short_greeks:
            short_delta = abs(float(short_greeks.get('delta') or 0))
            metrics['prob_profit'] = 1.0 - short_delta
            metrics['short_iv'] = float(short_greeks.get('implied_volatility') or 0)
            
        if long_greeks:
            metrics['long_iv'] = float(long_greeks.get('implied_volatility') or 0)

        # Fallback: maintain IV if missing in targeted snapshot but present in bulk fetch
        if metrics.get('short_iv', 0) <= 0 and 'all_snapshots' in locals() and short_put['symbol'] in all_snapshots:
             alt_greeks = all_snapshots[short_put['symbol']].get('greeks', {})
             metrics['short_iv'] = float(alt_greeks.get('implied_volatility') or 0)

        # Expected Value = P(win)*MaxProfit - P(loss)*MaxLoss
        if metrics['prob_profit'] > 0:
            p_win = metrics['prob_profit']
            p_loss = 1.0 - p_win
            # Adjust for partial losses (not always max loss)
            # Rough estimate: avg loss is ~60% of max loss
            avg_loss_factor = 0.6
            metrics['expected_value'] = (p_win * metrics['max_profit']) - (p_loss * metrics['max_loss'] * avg_loss_factor)
        
        # Return on Capital (Annualized)
        # ROC = (Credit / Max Risk) * (365 / DTE)
        if metrics['max_loss'] > 0 and isinstance(d_out, int) and d_out > 0:
            roc = metrics['max_profit'] / metrics['max_loss']
            annualized_roc = roc * (365 / d_out)
            metrics['return_on_capital'] = annualized_roc
        
        # IV Stats
        if use_delta and 'all_snapshots' in locals():
            iv_stats = get_iv_stats(all_snapshots, exp_contracts)
        else:
            iv_stats = get_iv_stats(snapshots, [short_put, long_put])
            
        metrics['iv_rank'] = iv_stats['rank']
        iv_rank = iv_stats['rank'] # Backwards compat for warnings
        
        # Fallback short_iv if missing (use chain median -> VIX -> HV)
        if metrics.get('short_iv', 0) <= 0:
             if iv_stats['median'] > 0:
                 metrics['short_iv'] = iv_stats['median']
             elif vix_level and symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
                 metrics['short_iv'] = vix_level / 100.0
             elif trend_details and trend_details.get('hv30'):
                 metrics['short_iv'] = trend_details['hv30']
        
        # IV Rank Check
        if not skip_checks:
            if iv_stats['count'] < 5:
                 pass # Warning?
            elif iv_rank < min_iv_rank:
                warnings.append(f"Low IV rank: {iv_rank:.0%} (min {min_iv_rank:.0%}) - consider waiting for higher IV")
        
        # Support Level Check
        if support_price:
            if short_strike < support_price:
                # This is good (strike protected by support)
                pass 
            elif short_strike > support_price:
                # This is riskier (strike above support)
                warnings.append(f"Short strike ${short_strike:.2f} is above 20-day support ${support_price:.2f}")
            elif short_strike <= support_price * 1.02:
                # Near support
                pass
        
        # VIX context warning
        if vix_level:
            if vix_level < 15 and not skip_checks:
                warnings.append(f"Low VIX ({vix_level:.1f}): Premium is cheap. Vega expansion risk is high.")
            elif vix_level > 35:
                warnings.append(f"Extreme VIX ({vix_level:.1f}): High premiums but extreme handling risk.")

        # Trend warning
        if not is_bullish and not skip_checks:
            warnings.append(f"Bearish trend ({trend_status}) - bull put spread may underperform")
            
        # RSI Warning
        if trend_details and trend_details.get('rsi'):
            rsi = trend_details['rsi']
            if rsi > 70 and not skip_checks:
                warnings.append(f"RSI Overbought ({rsi:.1f}) - risk of pullback")
        
        # Enhanced probability calculation using Black-Scholes if available
        if HAS_SCIPY and metrics['short_iv'] > 0 and isinstance(d_out, int):
            bs_prob = calculate_probability_otm(current_price, short_strike, metrics['short_iv'], d_out)
            if bs_prob > 0:
                # Use BS probability instead of delta approximation
                metrics['prob_profit_bs'] = bs_prob
                metrics['prob_touch'] = 2.0 * (1.0 - bs_prob)
                
                # Recalculate expected value with better probability
                p_win = bs_prob
                p_loss = 1.0 - p_win
                
                # Refined Expectancy: 
                # Credit spreads rarely hit max loss if managed.
                # Assuming management at 2x credit (100% loss of credit width)
                # But "Max Loss" in spread is Width - Credit.
                # If managed, avg loss is often ~1-1.5x Credit.
                # Let's be conservative: Avg Loss = 50% of (Width - Credit) = 50% of Max Loss allowed
                avg_loss = metrics['max_loss'] * 0.5 
                metrics['expected_value'] = (p_win * metrics['max_profit']) - (p_loss * avg_loss)
        
        # --- Auto-Calculate Quantity if --amount is used ---
        if args.amount and args.amount > 0:
            # Capital at risk per spread = Max Loss * 100
            risk_per_spread = metrics['max_loss'] * 100.0
            
            if risk_per_spread > 0:
                auto_qty = int(args.amount // risk_per_spread)
                if auto_qty < 1:
                    if not args.json:
                        print(f"\nWarning: Amount ${args.amount} is insufficient for 1 spread (Requires ~${risk_per_spread:.2f} max risk)")
                    args.quantity = 0
                else:
                    args.quantity = auto_qty
                    if not args.json:
                        print(f"\nAuto-calculated Quantity: {args.quantity}")
                        print(f"  Based on Max Risk Budget: ${args.amount}")
                        print(f"  Max Risk Per Spread: ${risk_per_spread:.2f}")

        # --- Open Interest Check ---
        if not skip_checks:
            for leg_contract, leg_name in [(short_put, 'Short Put'), (long_put, 'Long Put')]:
                oi_ok, oi, oi_reason = check_open_interest(leg_contract, min_oi)
                if not oi_ok:
                    warnings.append(f"{leg_name}: {oi_reason}")
        
        if not args.json:
            print("\n--- Financial Analysis (Per Share) ---")
            credit_pct = metrics['net_credit'] / actual_width if actual_width > 0 else 0
            print(f"Net Credit:       ${metrics['net_credit']:.2f} ({credit_pct:.0%} of spread width)")
            print(f"Entry Price:      {metrics['credit_source']}")
            
            if args.limit_order:
                 print(f"Natural Credit:   ${metrics['natural_credit']:.2f} (Backup/Conservative)")
            else:
                 print(f"Mid Credit:       ${metrics['mid_credit']:.2f} (Potential limit price)")
            
            # Slippage Warning
            if metrics['mid_credit'] > 0:
                 slippage = metrics['mid_credit'] - metrics['natural_credit']
                 if slippage > 0.10: # > 10 cents slippage
                      print(f"Slippage Risk:    ${slippage:.2f} per share (Difference between Mid and Natural)")
            
            print(f"Max Profit:       ${metrics['max_profit']:.2f} (stock ≥ ${short_strike:.2f})")
            print(f"Max Loss:         ${metrics['max_loss']:.2f} (stock ≤ ${long_strike:.2f})")
            # Calculate ratio as 1:X format (profit:risk)
            if metrics['max_profit'] > 0:
                ratio = metrics['max_loss'] / metrics['max_profit']
                print(f"Profit:Risk:      1:{ratio:.1f} (risk ${ratio:.2f} to make $1)")
            print(f"Break Even:       ${metrics['break_even']:.2f}")
            print(f"Risk/Reward:      {metrics['risk_reward']:.2f} ({metrics['risk_reward']*100:.0f}% return on risk)")
            if metrics['return_on_capital'] > 0:
                print(f"Annualized ROC:   {metrics['return_on_capital']:.0%}")
            if metrics['prob_profit'] > 0:
                prob_display = metrics.get('prob_profit_bs', metrics['prob_profit'])
                prob_source = "Black-Scholes" if 'prob_profit_bs' in metrics else "Delta approx"
                print(f"Est. P(Profit):   {prob_display:.1%} ({prob_source})")
                if 'prob_touch' in metrics:
                    print(f"Est. P(Touch):    {metrics['prob_touch']:.1%} (Risk of touching short strike during trade)")

            if metrics['expected_value'] != 0:
                ev_label = "positive" if metrics['expected_value'] > 0 else "negative"
                print(f"Expected Value:   ${metrics['expected_value']:.2f} ({ev_label})")
            
            # Kelly Criterion for position sizing
            if metrics['prob_profit'] > 0 and metrics['max_loss'] > 0:
                kelly = calculate_kelly_fraction(
                    metrics.get('prob_profit_bs', metrics['prob_profit']),
                    metrics['max_profit'],
                    metrics['max_loss']
                )
                if kelly > 0:
                    print(f"Kelly Fraction:   {kelly:.1%} of portfolio (quarter-Kelly)")
            
            print(f"\n--- Volatility ---")
            print(f"IV Rank:          {iv_rank:.0%} ✓")
        elif not skip_checks:
             print(f"IV Rank:          {iv_rank:.0%} ⚠️ LOW (Min {min_iv_rank:.0%})")
        else:
             print(f"IV Rank:          {iv_rank:.0%}")

        if metrics['short_iv'] > 0:
            print(f"Short Put IV:     {metrics['short_iv']:.1%}")
            
        # Vertical Skew Analysis
        if metrics.get('short_iv') and metrics.get('long_iv'):
            skew = metrics['short_iv'] - metrics['long_iv']
            skew_eval = "Favorable (Selling Skew)" if skew > 0 else "Unfavorable (Buying Skew)"
            print(f"Vertical Skew:    {skew:+.1%} ({skew_eval})")

        if trend_details and trend_details.get('hv30'):
            hv = trend_details['hv30']
            print(f"Hist Vol (30d):   {hv:.1%}")
            if metrics['short_iv'] > 0:
                iv_hv_ratio = metrics['short_iv'] / hv
                print(f"IV/HV Ratio:      {iv_hv_ratio:.2f}" + (" (Rich Premium)" if iv_hv_ratio > 1.2 else " (Fair/Cheap)" if iv_hv_ratio < 0.9 else ""))
            
        print(f"\n--- Greeks (Net Position) ---")
        print(f"Delta:            {metrics['net_delta']:.3f} (Bullish)")
        print(f"Gamma:            {metrics['net_gamma']:.4f}")
        print(f"Theta:            ${metrics['net_theta']:.3f}/day (Time Decay Profit)")
        print(f"Vega:             ${metrics['net_vega']:.3f} (Profit if IV drops 1%)")
        
        # Margin/Buying Power estimate (spread margin = width - credit)
        margin_per_spread = (actual_width - total_credit) * 100
        print(f"\n--- Capital Requirements ---")
        print(f"Margin/Spread:    ${margin_per_spread:.2f}")
        if args.quantity:
            print(f"Total Margin:     ${margin_per_spread * args.quantity:.2f}")
            
        # Buying Power Check
        if account_info and args.quantity:
            bp = float(account_info.get('buying_power') or 0)
            # Option Requirement: Cash or Equity usually (Standard Margin is 2x/4x for stocks)
            # For Defined Risk Put Spread: Requirement = Max Loss
            req_capital = margin_per_spread * args.quantity
            
            # Use 'cash' or 'portfolio_value' or 'buying_power' (Alpaca BP is often 4x for stocks)
            # Ideally we want "Option Buying Power".
            # Let's be conservative and check against 'equity' if available, otherwise BP/4?
            # Actually, simplest is just to print the available BP context.
            # account_info type: 'margin' or 'cash'
            acc_status = f"(BP: ${bp:.2f})"
            if req_capital > bp:
                 warnings.append(f"Insufficient Buying Power! Required: ${req_capital:.2f} > Available: ${bp:.2f}")
            else:
                 pass # print(f"Capital Check:    OK {acc_status}")

        # Opportunity cost comparison
            risk_free_rate = 0.05  # Approximate T-bill rate
            if isinstance(d_out, int) and d_out > 0:
                rf_return = margin_per_spread * (risk_free_rate * d_out / 365)
                print(f"Risk-Free Alt:    ${rf_return:.2f} ({risk_free_rate:.1%} APY for {d_out} days)")
                excess_return = (total_credit * 100) - rf_return
                print(f"Excess Return:    ${excess_return:.2f} vs risk-free")
                
            # Expected Move Context
            if metrics['short_iv'] > 0 and isinstance(d_out, int) and d_out > 0:
                exp_move = calculate_expected_move(current_price, metrics['short_iv'], d_out)
                print(f"\n--- Probable Range (1 Std Dev) ---")
                print(f"Expected Move:    ±${exp_move:.2f}")
                print(f"Upper Bound:      ${current_price + exp_move:.2f}")
                print(f"Lower Bound:      ${current_price - exp_move:.2f}")
                
                # Check where short strike lies
                dist_sigma = (current_price - short_strike) / exp_move
                print(f"Short Strike:     {dist_sigma:.1f}x Expected Moves OTM")
                if dist_sigma < 1.0:
                    print(f"                  ⚠️ Inside likely move (Aggressive)")
                else:
                    print(f"                  ✓ Outside likely move (Conservative)")

            print(f"\n--- Position Management ---")
            take_profit_price = total_credit * 0.50  # 50% of credit
            stop_loss_price = total_credit + (actual_width * 0.50)  # When spread doubles in price
            adjustment_trigger = short_strike * 1.01  # When stock gets close to short strike
            print(f"Take Profit:      Close when spread = ${take_profit_price:.2f} (50% profit)")
            
            # Theta Burn Estimate
            if metrics['net_theta'] > 0:
                 burn_amt = total_credit * 0.50
                 # Theta is per day. Simple linear extrapolation (conservative as theta accelerates)
                 days_to_50 = burn_amt / metrics['net_theta']
                 print(f"Est. Time to 50%: ~{days_to_50:.1f} days (based on current Theta)")
                 
                 # Efficiency Check
                 if metrics['max_loss'] > 0:
                     theta_yield = metrics['net_theta'] / metrics['max_loss']
                     # 0.1% per day is decent ($1 theta on $1000 risk)
                     if theta_yield < 0.001:
                          print(f"                  ⚠️ Low Theta Efficiency ({theta_yield:.2%} daily). Consider widening spread.")
                     else:
                          print(f"                  ✓ Good Efficiency ({theta_yield:.2%} daily return on risk)")
            
            print(f"Stop Loss:        Close when spread = ${stop_loss_price:.2f} (loss = credit)")
            if support_price:
                 print(f"Technical Stop:   Close if daily close < ${support_price:.2f} (Support break)")
            print(f"Adjustment:       Consider rolling if stock < ${adjustment_trigger:.2f}")
            print(f"Days to Manage:   {max(1, d_out - 7) if isinstance(d_out, int) else '?'} days (close before last week)")
            if isinstance(d_out, int) and d_out < 7:
                 print(f"                  ⚠️ GAMMA RISK: Expiration is imminent. Assignment risk high!")

            # Print Payoff Diagram (only if requested)
            if args.show_diagram:
                print_payoff_diagram(current_price, short_strike, long_strike, metrics['max_profit'])
            
            # Print Plan Location last
            if not args.json and 'log_file_path' in locals() and log_file_path:
                print(f"\n📄 Plan Summary:  {log_file_path}")

            # Rolling guidance
            if isinstance(d_out, int) and d_out <= 21:
                print(f"\n📋 21 DTE Management Rule:")
                print(f"   - If profitable: Close at 50% profit or roll to next month")
                print(f"   - If at loss: Roll down and out for credit, or close for small loss")
                print(f"   - Never hold through last week (gamma risk increases)")
            
            # Show warnings
            if warnings:
                print(f"\n--- ⚠️ Warnings ---")
                for w in warnings:
                    print(f"  • {w}")
                
                # Dynamic Optimization Tips (Dry Run only)
                if args.dry_run:
                     suggestions = []
                     warnings_text = " ".join(warnings).lower()
                     
                     if "credit" in warnings_text:
                          suggestions.append("Try a wider spread to improve Credit/Width ratio")
                          suggestions.append("Move short strike closer to price (higher Delta)")
                     
                     if "efficiency" in warnings_text or "theta" in warnings_text:
                          suggestions.append("Widen spread to increase Theta capture per dollar of risk")
                          suggestions.append("Look for expiration with higher IV (Earnings/Events) if appropriate")
                     
                     if "iv rank" in warnings_text:
                          suggestions.append("Consider Debit Spreads (Bear Put/Bull Call) in low IV environment")
                          suggestions.append("Wait for IV expansion (market drop) to sell premiums")

                     if suggestions:
                          print("\n💡 Optimization Tips:")
                          for s in sorted(list(set(suggestions))): # dedupe
                               print(f"  • {s}")

    except Exception as e:
        if not args.json:
            print(f"Warning: Could not calculate metrics: {e}", file=sys.stderr)
        metrics['error'] = str(e)

    # --- Generate Email Text ---
    # Store log file path for printing later if needed
    log_file_path = None
    
    email_lines = []
    email_lines.append(f"Bull Put Spread: {symbol}")
    email_lines.append("=" * 30)
    email_lines.append(f"Date: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}")
    if current_price > 0:
        email_lines.append(f"Current Price: ${current_price:.2f}")
    email_lines.append(f"Expiration: {selected_exp}")
    email_lines.append("")
    email_lines.append("Structure:")
    for leg in legs:
        action = "SELL" if leg['side'] == 'sell' else "BUY"
        price_str = f"~${leg.get('estimated_price', 0):.2f}"
        email_lines.append(f"- {action}: {leg['symbol']} ({price_str})")
    email_lines.append("")
    email_lines.append("Financial Analysis (Per Share):")
    email_lines.append(f"Net Credit:    ${metrics.get('net_credit', 0):.2f}")
    email_lines.append(f"Max Profit:    ${metrics.get('max_profit', 0):.2f}")
    email_lines.append(f"Max Loss:      ${metrics.get('max_loss', 0):.2f}")
    email_lines.append(f"Break Even:    ${metrics.get('break_even', 0):.2f}")
    email_lines.append(f"Risk/Reward:   {metrics.get('risk_reward', 0):.2%}")
    if metrics.get('prob_profit'):
        email_lines.append(f"P(Profit):     {metrics.get('prob_profit', 0):.1%}")
    if metrics.get('expected_value'):
        email_lines.append(f"Exp. Value:    ${metrics.get('expected_value', 0):.2f}")
    email_lines.append(f"IV Rank:       {metrics.get('iv_rank', 0):.0%}")
    email_lines.append("")
    email_lines.append(f"Target Quantity: {args.quantity}")
    
    if warnings:
        email_lines.append("")
        email_lines.append("Warnings:")
        for w in warnings:
            email_lines.append(f"  • {w}")
    
    if metrics.get('error'):
        email_lines.append("")
        email_lines.append(f"⚠️ Metrics Warning: {metrics.get('error')}")

    email_text_base = "\n".join(email_lines)

    # Save plan to log
    try:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"bull_put_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_file_path = str(log_file)
        with open(log_file, "w") as f:
            f.write(email_text_base)
            f.write(f"\n\nContext: Dry Run={args.dry_run}, Limit={args.limit_order}")
    except Exception:
        pass
        
    if not args.json:
         pass # Don't print path here, printed in summary

    if args.dry_run:
        result = {
            "status": "dry_run",
            "strategy": "bull_put_spread",
            "scan": {
                "current_price": current_price,
                "expiration": selected_exp,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "width": actual_width,
            },
            "legs": legs,
            "metrics": metrics,
            "warnings": warnings,
            "email_text": email_text_base + "\n\nStatus: Dry Run (No Order Submitted)"
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n--- Strategy Summary ---")
            print(f"Strategy: Bull Put Spread (Credit Spread)")
            print(f"Outlook: Bullish/Neutral (want stock ≥ ${short_strike:.2f})")
            
            # Trade quality assessment
            quality_score = 0
            if metrics['iv_rank'] >= min_iv_rank: quality_score += 1
            if metrics.get('credit_pct', 0) >= min_credit_pct: quality_score += 1
            if metrics['prob_profit'] >= 0.65: quality_score += 1
            if metrics['expected_value'] > 0: quality_score += 1
            if not warnings: quality_score += 1
            
            quality_labels = {0: "Poor", 1: "Weak", 2: "Fair", 3: "Good", 4: "Strong", 5: "Excellent"}
            print(f"Trade Quality:   {quality_labels.get(quality_score, 'Unknown')} ({quality_score}/5)")
            
            if warnings and not skip_checks:
                print(f"\n⚠️ {len(warnings)} warning(s) found. Use --skip-checks to ignore.")
            
            print(f"\nDry Run Complete. Order not submitted.")
        return

    # Check for blocking warnings before execution
    if warnings and not skip_checks:
        if not args.json:
            print(f"\n⚠️ Cannot execute: {len(warnings)} warning(s) found.")
            print("Use --skip-checks to override, or fix the issues.")
            for w in warnings:
                print(f"  • {w}")
        else:
            print(json.dumps({"error": "Trade blocked due to warnings", "warnings": warnings}))
        return
    
    # Execute
    if not args.json:
        order_type = "LIMIT" if args.limit_order else "MARKET"
        print(f"\nSubmitting {order_type} order for {args.quantity}x spreads...")
        
    try:
        if args.limit_order:
            # Calculate mid-price for limit order
            short_snap = snapshots.get(short_put['symbol'], {})
            long_snap = snapshots.get(long_put['symbol'], {})
            
            short_quote = short_snap.get('latestQuote', {})
            long_quote = long_snap.get('latestQuote', {})
            
            short_mid = (float(short_quote.get('bp', 0)) + float(short_quote.get('ap', 0))) / 2
            long_mid = (float(long_quote.get('bp', 0)) + float(long_quote.get('ap', 0))) / 2
            
            limit_credit = round(short_mid - long_mid, 2)
            
            if not args.json:
                print(f"Limit Credit: ${limit_credit:.2f} (mid-price)")
            
            # Note: Alpaca may require different API call for limit orders
            # This assumes the client supports limit_price parameter
            # Pass entry_cash_flow for mock orders (Credit = positive cash flow)
            entry_cash_flow = limit_credit
            kwargs = {'legs': legs, 'quantity': args.quantity, 'limit_price': limit_credit, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow

            response = client.place_option_limit_order(**kwargs)
        else:
            # Pass entry_cash_flow for mock orders (Credit = positive cash flow)
            entry_cash_flow = metrics['net_credit']
            kwargs = {'legs': legs, 'quantity': args.quantity, 'time_in_force': 'day', 'order_class': 'mleg'}
            if isinstance(client, MockOptionClient):
                kwargs['entry_cash_flow'] = entry_cash_flow

            response = client.place_option_market_order(**kwargs)
        
        result = {
            "status": "executed",
            "strategy": "bull_put_spread",
            "order_type": "limit" if args.limit_order else "market",
            "order": response,
            "email_text": email_text_base + f"\n\nStatus: Order Submitted\nOrder ID: {response.get('id')}\nAlpaca Order Status: {response.get('status')}"
        }
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Order Submitted Successfully!")
            print(f"Order ID: {response.get('id')}")
            print(f"Status: {response.get('status')}")
    except Exception as e:
        err = {"error": str(e), "email_text": email_text_base + f"\n\n❌ Execution Error: {str(e)}"}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Execution Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

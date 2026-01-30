#!/usr/bin/env python3
"""
Alpaca Synthetic Long + Protective Put Scanner
----------------------------------------------
A comprehensive scanning engine to identify candidates for synthetic long strategies
based on Liquidity, Dividend Risk, Volatility Skew, and Momentum.

Attributes:
    High Liquidity: Top 100 Most Active Stocks
    Low Dividend: No ex-dates in next 90 days, Yield < 1.5%
    High Momentum: RSI > 50, Price > SMA50 > SMA200
    Flat Skew: |IV_Put_OTM - IV_Call_OTM| < Threshold
    Low IV: ATM IV < 40% (Configurable)
"""

import os
import asyncio
import datetime
import math
import concurrent.futures
import requests
import yaml
import argparse
import re
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from dotenv import load_dotenv

# Alpaca SDK Imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, 
    StockLatestQuoteRequest,
    CorporateActionsRequest,
    OptionChainRequest,
    StockLatestTradeRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------

load_dotenv()

# API Credentials (Load from Environment for Security)
# Users must set these in their system environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
IS_PAPER = "paper" in BASE_URL

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY and ALPACA_API_SECRET must be set in environment.")
    # We don't exit here to allow import, but script execution will fail if main is run.

# Strategy Parameters
MIN_PRICE = 15.0            # Minimum stock price
MIN_VOLUME = 2_000_000      # Daily Volume Floor
MAX_SPREAD_PCT = 0.005      # Max Bid-Ask Spread (0.5%)
MAX_DIV_YIELD = 0.015       # Max Dividend Yield (1.5%)
RSI_MIN = 50.0              # Momentum Floor
RSI_MAX = 75.0              # Momentum Ceiling (Avoid Overbought)
IV_MAX_THRESHOLD = 0.45     # Max Implied Volatility
SKEW_TOLERANCE = 0.04       # Max difference between Put/Call IV
TARGET_DTE_MIN = 30         # Min Days to Expiration
TARGET_DTE_MAX = 60         # Max Days to Expiration

# -----------------------------------------------------------------------------
# ANALYTICAL UTILITIES (GREEKS & SOLVERS)
# -----------------------------------------------------------------------------

class BlackScholesSolver:
    """
    Local implementation of Black-Scholes-Merton model to solve for
    Implied Volatility when API returns null or to validate skew.
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculates d1 parameter for Black-Scholes."""
        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(d1_val: float, sigma: float, T: float) -> float:
        """Calculates d2 parameter."""
        return d1_val - sigma * np.sqrt(T)

    @staticmethod
    def price_option(S: float, K: float, T: float, r: float, sigma: float, flag: str = 'c') -> float:
        """Pricing function for Calls ('c') and Puts ('p')."""
        d1 = BlackScholesSolver.d1(S, K, T, r, sigma)
        d2 = BlackScholesSolver.d2(d1, sigma, T)
        
        if flag.lower() == 'c':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, flag: str = 'c') -> float:
        """Calculates Option Delta."""
        d1 = BlackScholesSolver.d1(S, K, T, r, sigma)
        if flag.lower() == 'c':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculates Option Gamma (same for calls and puts)."""
        if sigma <= 0 or T <= 0 or S <= 0:
            return 0.0
        d1 = BlackScholesSolver.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, flag: str = 'c') -> float:
        """
        Calculates Option Theta (daily decay).
        Returns negative value (time decay costs money for long positions).
        """
        if sigma <= 0 or T <= 0 or S <= 0:
            return 0.0
        d1 = BlackScholesSolver.d1(S, K, T, r, sigma)
        d2 = BlackScholesSolver.d2(d1, sigma, T)
        
        # First term (common to both)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if flag.lower() == 'c':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            return (term1 + term2) / 365.0  # Daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365.0  # Daily theta

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculates Option Vega (sensitivity to IV change of 1%)."""
        if sigma <= 0 or T <= 0 or S <= 0:
            return 0.0
        d1 = BlackScholesSolver.d1(S, K, T, r, sigma)
        return (S * norm.pdf(d1) * np.sqrt(T)) / 100.0  # Per 1% IV change

    @staticmethod
    def implied_volatility(
        market_price: float, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        flag: str = 'c'
    ) -> Optional[float]:
        """
        Newton-Raphson solver to derive IV from Market Price.
        Handles the 'Null Greeks' issue inherent in API snapshots.
        """
        MAX_ITER = 100
        PRECISION = 1.0e-5
        sigma = 0.5  # Initial guess
        
        for _ in range(MAX_ITER):
            price = BlackScholesSolver.price_option(S, K, T, r, sigma, flag)
            diff = market_price - price
            
            if abs(diff) < PRECISION:
                return sigma
            
            # Calculate Vega (Derivative of Price w.r.t Sigma)
            d1 = BlackScholesSolver.d1(S, K, T, r, sigma)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            if abs(vega) < 1.0e-8:
                break # Avoid division by zero
                
            sigma = sigma + diff / vega
            
        return None # Failed to converge

class RiskFreeRate:
    """
    Fetches dynamic risk-free rate using Treasury Yields via yfinance.[15]
    Uses 3-Month T-Bill (^IRX) as proxy.
    """
    _cached_rate = None
    _last_fetch = None

    @classmethod
    def get_rate(cls) -> float:
        now = datetime.datetime.now()
        # Cache for 24 hours
        if cls._cached_rate and cls._last_fetch and (now - cls._last_fetch).days < 1:
            return cls._cached_rate
            
        try:
            # ^IRX is the CBOE Interest Rate 13 Week T Bill
            ticker = yf.Ticker("^IRX")
            # Divide by 100 as data is typically 5.25 for 5.25%
            rate = ticker.history(period="1d")['Close'].iloc[-1] / 100.0
            cls._cached_rate = rate
            cls._last_fetch = now
            return rate
        except Exception as e:
            print(f"Warning: Could not fetch Risk Free Rate ({e}). Defaulting to 4.5%")
            return 0.045

# -----------------------------------------------------------------------------
# MAIN SCANNER LOGIC
# -----------------------------------------------------------------------------

class OptionStrategyScanner:
    def __init__(self):
        # Initialize Alpaca Clients
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=IS_PAPER)
        self.stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.option_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
        self.rf_rate = RiskFreeRate.get_rate()
        self.config = self.load_config()
        self.scan_opts = self.config.get('option_scan', {})
        
        # Check VIX regime for market context
        self.vix_level = self._get_vix_level()
        self.market_regime = self._classify_market_regime()
        
        print(f"Initialized Scanner. Risk-Free Rate: {self.rf_rate:.2%}")
        print(f"Market Regime: {self.market_regime} (VIX: {self.vix_level:.1f})")

    def _get_vix_level(self) -> float:
        """Fetches current VIX level for market regime classification."""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            print(f"Warning: Could not fetch VIX ({e}). Defaulting to 20.")
        return 20.0
    
    def _classify_market_regime(self) -> str:
        """Classifies market based on VIX level."""
        if self.vix_level < 15:
            return "LOW_VOL (Complacent)"
        elif self.vix_level < 20:
            return "NORMAL"
        elif self.vix_level < 30:
            return "ELEVATED"
        else:
            return "HIGH_VOL (Fear)"

    def load_config(self) -> Dict:
        """Loads configuration from YAML file."""
        try:
            # Resolve path relative to this script: ../config/Options.yaml
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, '../config/Options.yaml')
            
            if os.path.exists(config_path):
                print(f"Loading config from {config_path}")
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(f"Config file not found at {config_path}")
        except Exception as e:
            print(f"Warning loading config: {e}")
        return {}

    def get_top_market_cap_universe(self, limit: int = 200) -> List[str]:
        """
        Fetches S&P 500, retrieves market caps, and returns the top N.
        Caches results for 7 days to avoid repeated scraping and API calls.
        """
        cache_dir = Path("trading_logs/scanner")
        cache_dir.mkdir(parents=True, exist_ok=True)
        CACHE_FILE = cache_dir / "sp500_market_cap_cache.csv"
        CACHE_DURATION_DAYS = 7
        
        # Check Cache
        if CACHE_FILE.exists():
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
            if (datetime.datetime.now() - mtime).days < CACHE_DURATION_DAYS:
                print(f"Loading S&P 500 list from cache ({CACHE_FILE}, age: {(datetime.datetime.now() - mtime).days} days)...")
                try:
                    df_cache = pd.read_csv(CACHE_FILE)
                    # Ensure sorted by mcap descending
                    df_cache = df_cache.sort_values(by='market_cap', ascending=False)
                    top_n = df_cache['symbol'].head(limit).tolist()
                    print(f"Retrieved Top {len(top_n)} from Cache. Top 5: {top_n[:5]}")
                    return top_n
                except Exception as e:
                    print(f"Error reading cache: {e}. Proceeding to fetch fresh data.")

        print(f"Fetching S&P 500 list and Market Caps to filter for Top {limit}...")
        try:
            # 1. Scraping S&P 500 Tickers
            # Use requests with User-Agent to avoid 403 Forbidden
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
            }
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            table = pd.read_html(StringIO(response.text))
            df_sp = table[0]
            tickers = df_sp['Symbol'].tolist()
            
            # 2. Fetch Market Caps via YFinance (using FastInfo)
            print(f"Fetching market cap data for {len(tickers)} symbols via YFinance...")
            
            def get_mcap(symbol):
                try:
                    # Conversion for YF: BRK.B -> BRK-B
                    yf_sym = symbol.replace('.', '-')
                    # Accessing fast_info is efficient (often no web request if cached or light request)
                    # Note: accessing 'market_cap' usually triggers the fetch
                    info = yf.Ticker(yf_sym).fast_info
                    mcap = info.market_cap
                    return symbol, mcap
                except Exception:
                    return symbol, 0.0

            data = []
            # Use threading to speed up 500 HTTP requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = executor.map(get_mcap, tickers)
                for sym, mcap in results:
                    if mcap and mcap > 0:
                        data.append((sym, mcap))
            
            # 3. Sort and Save to Cache
            data.sort(key=lambda x: x[1], reverse=True)
            
            df_save = pd.DataFrame(data, columns=['symbol', 'market_cap'])
            df_save.to_csv(CACHE_FILE, index=False)
            print(f"Saved S&P 500 market cap data to cache ({CACHE_FILE}).")

            top_n = [x[0] for x in data[:limit]]
            
            print(f"Retrieved Top {len(top_n)} by Market Cap. Top 5: {top_n[:5]}")
            return top_n
            
        except Exception as e:
            print(f"Error fetching market caps: {e}")
            # Fallback to cache if scrape fails but cache exists (even if old)
            if CACHE_FILE.exists():
                print("Scrape failed, falling back to existing cache (potentially expired).")
                try:
                    df_cache = pd.read_csv(CACHE_FILE)
                    df_cache = df_cache.sort_values(by='market_cap', ascending=False)
                    return df_cache['symbol'].head(limit).tolist()
                except:
                    pass
            return []

    async def get_universe(self, limit_override: Optional[int] = None) -> List[str]:
        """
        Stage 1: Filter for Top N Market Cap + Optionable + Tradable.
        """
        # Prioritize override, then config, then default
        if limit_override:
            limit = limit_override
            print(f"Using CLI limit: {limit}")
        else:
            limit = self.config.get('option_scan', {}).get('max_scan_count', 200)
            print(f"Using Config limit: {limit}")
        
        # 1. Get Target List (Top N Market Cap)
        target_symbols = self.get_top_market_cap_universe(limit=limit)
        
        candidates = []
        
        # 2. Get all active US equities with options from Alpaca to validate
        print("Validating symbols against Alpaca Universe (Optionable check)...")
        search_params = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, 
            status=AssetStatus.ACTIVE,
            attributes="has_options"
        )
        assets = self.trading_client.get_all_assets(search_params)
        
        # Create a lookup map
        valid_alpaca_map = {a.symbol: a for a in assets}
        
        # 3. Intersect lists
        for sym in target_symbols:
            # Check direct match
            found_asset = valid_alpaca_map.get(sym)
            
            # Check slash format (Alpaca sometimes uses / for classes)
            if not found_asset:
                 found_asset = valid_alpaca_map.get(sym.replace('.', '/'))
            
            if found_asset:
                a = found_asset
                is_tradable = getattr(a, 'tradable', False)
                is_marginable = getattr(a, 'marginable', False)
                is_shortable = getattr(a, 'shortable', False)
                
                if is_tradable and is_marginable and is_shortable:
                     candidates.append(a.symbol)

        print(f"Final Universe Size filtered by Liquidity/Options: {len(candidates)}")
        return candidates

    def get_market_data_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Retrieves price and volume to finalize the liquidity filter.
        """
        # Chunking requests to avoid URI length limits
        chunk_size = 50
        valid_data = []
        min_price = self.scan_opts.get('min_price', 15.0)
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            try:
                # Get latest trades for price
                request = StockLatestTradeRequest(symbol_or_symbols=chunk)
                trades = self.stock_client.get_stock_latest_trade(request)
                
                for sym, trade in trades.items():
                    if trade.price > min_price:
                        valid_data.append({'symbol': sym, 'price': trade.price})
            except Exception as e:
                print(f"Error fetching snapshot for chunk: {e}")
                
        return pd.DataFrame(valid_data)

    def analyze_momentum(self, symbol: str, trend_view: str = "bullish", verbose: bool = False) -> Tuple[bool, float, float]:
        """
        Calculates RSI, Moving Averages, MACD, ADX, and Volume.
        Returns (Pass/Fail, RSI Value, HV20).
        """
        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=365) # Fetch enough for SMA200
        
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            limit=300
        )
        
        try:
            bars = self.stock_client.get_stock_bars(req).df
            # Multi-index handling (symbol, timestamp)
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol)
                
            if len(bars) < 200:
                if verbose: print(f"  [Detail] Not enough history: {len(bars)} bars")
                return False, 0.0, 0.0

            # Calculate SMA
            bars['sma50'] = bars['close'].rolling(window=50).mean()
            bars['sma200'] = bars['close'].rolling(window=200).mean()
            
            # Calculate RSI (Wilder's Smoothing)
            delta = bars['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            bars['rsi'] = 100 - (100 / (1 + rs))
            
            current = bars.iloc[-1]
            
            # Logic: Price > SMA50 > SMA200 (Bullish Trend) AND RSI in range
            price = current['close']
            sma50 = current['sma50']
            sma200 = current['sma200']
            rsi = current['rsi']
            
            # Calculate MACD (12, 26, 9) for trend confirmation
            ema12 = bars['close'].ewm(span=12, adjust=False).mean()
            ema26 = bars['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            macd_current = macd_line.iloc[-1]
            signal_current = signal_line.iloc[-1]
            histogram_current = macd_histogram.iloc[-1]
            # Check if MACD crossed above signal in last 5 days (recent bullish)
            macd_bullish_cross = False
            for i in range(-5, 0):
                if macd_line.iloc[i-1] < signal_line.iloc[i-1] and macd_line.iloc[i] >= signal_line.iloc[i]:
                    macd_bullish_cross = True
                    break
            # MACD above signal line is bullish
            macd_bullish = macd_current > signal_current
            
            # Calculate Relative Volume (current volume vs 20-day average)
            vol_sma20 = bars['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = current['volume']
            rel_volume = current_volume / vol_sma20 if vol_sma20 > 0 else 1.0
            
            # Calculate HV (Historical Volatility) - 20 Day Annualized
            bars['log_ret'] = np.log(bars['close'] / bars['close'].shift(1))
            hv_window = 20
            if len(bars) >= hv_window:
                hv20 = bars['log_ret'].tail(hv_window).std() * np.sqrt(252)
            else:
                hv20 = 0.0

            # Calculate ADX (Average Directional Index) for Trend Strength
            # ADX > 25 indicates strong trend, > 40 very strong
            high = bars['high']
            low = bars['low']
            close = bars['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff().abs() * -1
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr14 = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = dx.rolling(window=14).mean().iloc[-1]

            # Calculate IV Percentile (using HV as proxy over 1 year)
            # This shows where current volatility sits relative to the past year
            rolling_hv = bars['log_ret'].rolling(window=20).std() * np.sqrt(252)
            rolling_hv = rolling_hv.dropna()
            if len(rolling_hv) > 50:
                iv_percentile = (rolling_hv < hv20).sum() / len(rolling_hv) * 100
            else:
                iv_percentile = 50.0  # Default to middle
            
            # Store for later use
            self._last_iv_percentile = iv_percentile
            
            # Calculate SPY correlation (20-day) for diversification awareness
            try:
                spy_req = StockBarsRequest(
                    symbol_or_symbols="SPY",
                    timeframe=TimeFrame.Day,
                    start=start_dt,
                    end=end_dt,
                    limit=50
                )
                spy_bars = self.stock_client.get_stock_bars(spy_req).df
                if isinstance(spy_bars.index, pd.MultiIndex):
                    spy_bars = spy_bars.xs("SPY")
                
                # Align and calculate correlation
                stock_rets = bars['log_ret'].tail(20)
                spy_rets = np.log(spy_bars['close'] / spy_bars['close'].shift(1)).tail(20)
                
                if len(stock_rets) == len(spy_rets):
                    spy_corr = stock_rets.corr(spy_rets)
                else:
                    spy_corr = 0.0
            except Exception:
                spy_corr = 0.0
            
            self._last_spy_corr = spy_corr
            
            # Get sector for tracking
            try:
                tk = yf.Ticker(symbol)
                sector = tk.info.get('sector', 'Unknown')
            except Exception:
                sector = 'Unknown'
            self._last_sector = sector

            rsi_min = self.scan_opts.get('rsi_min', 50.0)
            rsi_max = self.scan_opts.get('rsi_max', 75.0)
            min_adx = self.scan_opts.get('min_adx', 20.0)  # Min trend strength
            min_rel_volume = self.scan_opts.get('min_rel_volume', 0.8)  # Min relative volume (80% of avg)
            require_macd_bullish = self.scan_opts.get('require_macd_bullish', True)  # MACD above signal

            if trend_view == "bullish":
                trend_check = (price > sma50) and (sma50 > sma200)
                rsi_check = rsi_min < rsi < rsi_max
                macd_check = macd_bullish if require_macd_bullish else True
            elif trend_view == "bearish":
                trend_check = (price < sma50) and (sma50 < sma200)
                # For bearish, we want RSI < 50 (downtrend) but not oversold (<30) ideally, or just < 60
                rsi_check = rsi < 60 
                macd_check = not macd_bullish if require_macd_bullish else True
            else:
                trend_check = True; rsi_check = True; macd_check = True

            adx_check = adx >= min_adx if not np.isnan(adx) else True  # Pass if ADX unavailable
            volume_check = rel_volume >= min_rel_volume
            
            if verbose:
                print(f"  [Detail] Momentum Metrics for {symbol}:")
                print(f"    Price: ${price:.2f}")
                print(f"    SMA50: ${sma50:.2f} (Price > SMA50: {price > sma50})")
                print(f"    SMA200: ${sma200:.2f} (SMA50 > SMA200: {sma50 > sma200})")
                print(f"    RSI: {rsi:.1f} (Allowed: {rsi_min}-{rsi_max})")
                print(f"    ADX: {adx:.1f} (Min: {min_adx}, Trend Strength: {'Strong' if adx >= 25 else 'Weak'})")
                print(f"    MACD: {macd_current:.3f} (Signal: {signal_current:.3f}, Histogram: {histogram_current:.3f})")
                print(f"    MACD Status: {'BULLISH' if macd_bullish else 'BEARISH'} (Recent Crossover: {'Yes' if macd_bullish_cross else 'No'})")
                print(f"    Rel Volume: {rel_volume:.2f}x (Min: {min_rel_volume}x, Today: {current_volume:,.0f})")
                print(f"    HV20: {hv20:.2%} (Historical Volatility)")
                print(f"    IV Percentile: {iv_percentile:.0f}% (Current vol rank vs 1yr)")
                print(f"    SPY Correlation: {spy_corr:.2f} (Diversification: {'High' if abs(spy_corr) > 0.7 else 'Moderate' if abs(spy_corr) > 0.4 else 'Low'})")
                print(f"    Sector: {sector}")
                print(f"    Trend Result: {'PASS' if trend_check else 'FAIL'}")
                print(f"    RSI Result: {'PASS' if rsi_check else 'FAIL'}")
                print(f"    ADX Result: {'PASS' if adx_check else 'FAIL'}")
                print(f"    MACD Result: {'PASS' if macd_check else 'FAIL'}")
                print(f"    Volume Result: {'PASS' if volume_check else 'FAIL'}")

            # Store MACD info for results
            self._last_macd_bullish = macd_bullish
            self._last_macd_cross = macd_bullish_cross
            self._last_rel_volume = rel_volume
            self._last_adx = adx if not np.isnan(adx) else 0.0

            return (trend_check and rsi_check and adx_check and volume_check and macd_check), rsi, hv20
            
        except Exception as e:
            if verbose: print(f"  [Detail] Momentum Analysis Error: {e}")
            return False, 0.0, 0.0

    def _parse_option_symbol(self, symbol_str: str) -> Optional[Dict]:
        """
        Parses Alpaca/OCC option symbol string: Root + YYMMDD + T + Strike
        Example: AAPL230616C00150000
        """
        try:
            # We assume suffix format: 15 chars (6+1+8) at the end.
            if len(symbol_str) < 15:
                return None
                
            suffix = symbol_str[-15:]
            date_str = suffix[:6]   # YYMMDD
            type_char = suffix[6]   # C or P
            strike_str = suffix[7:] # 00150000
            
            expiry = datetime.datetime.strptime(date_str, "%y%m%d").date()
            otype = 'call' if type_char.upper() == 'C' else 'put'
            strike = float(strike_str) / 1000.0
            
            return {
                'type': otype,
                'expiration': expiry,
                'strike': strike
            }
        except Exception:
            return None

    def check_dividend_risk(self, symbol: str, verbose: bool = False) -> bool:
        """
        Checks for upcoming dividends to avoid assignment on Short Put.
        Uses yfinance as fallback since Alpaca Client issue.
        Note: ETFs (SPY, QQQ, etc.) don't have traditional dividend calendars.
        """
        if verbose: print(f"  [Detail] Checking dividends for {symbol}...")
        
        try:
            tk = yf.Ticker(symbol)
            
            # Check if this is an ETF (quoteType)
            try:
                info = tk.info
                quote_type = info.get('quoteType', 'EQUITY')
                if quote_type == 'ETF':
                    if verbose: print("    -> ETF detected, skipping dividend calendar check.")
                    return True
            except Exception:
                pass  # If info fails, continue with calendar check
            
            cal = tk.calendar
            
            found_risk = False
            today = datetime.date.today()
            limit_date = today + datetime.timedelta(days=90)
            ex_date = None
            
            # Handle different yfinance response structures
            if isinstance(cal, dict) and 'Ex-Dividend Date' in cal:
                 ex_date = cal['Ex-Dividend Date']
            elif isinstance(cal, pd.DataFrame) and 'Ex-Dividend Date' in cal.index:
                ex_date = cal.loc['Ex-Dividend Date']
                if isinstance(ex_date, pd.Series):
                    ex_date = ex_date.iloc[0]

            if ex_date is not None:
                if isinstance(ex_date, (list, tuple)):
                    ex_date = ex_date[0]
                if isinstance(ex_date, (datetime.date, datetime.datetime)):
                    ex_dt = ex_date.date() if isinstance(ex_date, datetime.datetime) else ex_date
                    if today <= ex_dt <= limit_date:
                        if verbose: print(f"    -> FOUND DIVIDEND Risk: Ex-Date {ex_dt}")
                        found_risk = True

            if found_risk:
                return False 
            
            if verbose: print("    -> No dividend risk found.")
            return True
            
        except Exception as e:
            # Suppress 404 errors for ETFs/symbols without fundamentals
            error_str = str(e)
            if '404' in error_str or 'No fundamentals' in error_str:
                if verbose: print("    -> No dividend calendar (ETF or N/A). Passing.")
                return True
            if verbose: print(f"  [Detail] Dividend Check Failed (Ignored): {e}")
            return True

    def check_earnings_risk(self, symbol: str, verbose: bool = False) -> bool:
        """
        Checks for upcoming earnings to avoid volatility crush or gap risk.
        Uses yfinance calendar.
        Note: ETFs (SPY, QQQ, etc.) don't have earnings dates.
        """
        if verbose: print(f"  [Detail] Checking earnings for {symbol}...")
        try:
            tk = yf.Ticker(symbol)
            
            # Check if this is an ETF (quoteType)
            try:
                info = tk.info
                quote_type = info.get('quoteType', 'EQUITY')
                if quote_type == 'ETF':
                    if verbose: print("    -> ETF detected, no earnings risk.")
                    return True
            except Exception:
                pass  # If info fails, continue with calendar check
            
            cal = tk.calendar
            
            earnings_date = None
            found_risk = False
            today = datetime.date.today()
            limit_date = today + datetime.timedelta(days=60) # 60 Day risk window
            
            # yfinance structure varies
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                earnings_date = cal['Earnings Date']
            elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
                earnings_date = cal.loc['Earnings Date']
                if isinstance(earnings_date, pd.Series):
                    earnings_date = earnings_date.iloc[0]
            
            if earnings_date is not None:
                # Can be list of dates
                if isinstance(earnings_date, (list, tuple)):
                    earnings_date = earnings_date[0]
                
                if isinstance(earnings_date, (datetime.date, datetime.datetime)):
                     e_dt = earnings_date.date() if isinstance(earnings_date, datetime.datetime) else earnings_date
                     if today <= e_dt <= limit_date:
                         if verbose: print(f"    -> FOUND EARNINGS Risk: {e_dt}")
                         found_risk = True
            
            if found_risk:
                return False
                
            if verbose: print("    -> No earnings risk found.")
            return True
        except Exception as e:
            # Suppress 404 errors for ETFs/symbols without fundamentals
            error_str = str(e)
            if '404' in error_str or 'No fundamentals' in error_str:
                if verbose: print("    -> No earnings calendar (ETF or N/A). Passing.")
                return True
            if verbose: print(f"  [Detail] Earnings Check Failed (Ignored): {e}")
            return True

    def analyze_volatility_skew(self, symbol: str, spot_price: float, hv20: float = 0.0, verbose: bool = False, strategy: str = "synthetic_long") -> Optional[Dict]:
        """
        The Core Engine: Fetches Option Chain, solves for IV, calculates Skew.
        Target: Flat Skew & Low IV.
        """
        target_dte_min = self.scan_opts.get('target_dte_min', 30)
        target_dte_max = self.scan_opts.get('target_dte_max', 60)
        iv_max_threshold = self.scan_opts.get('iv_max_threshold', 0.45)
        skew_tolerance = self.scan_opts.get('skew_tolerance', 0.04)
        max_opt_spread = self.scan_opts.get('max_option_spread_pct', 0.05)
        max_iv_hv_ratio = self.scan_opts.get('iv_hv_ratio_threshold', 1.25)
        min_oi = self.scan_opts.get('min_open_interest', 50)

        # 1. Define Expiration Window
        start_date = datetime.date.today() + datetime.timedelta(days=target_dte_min)
        end_date = datetime.date.today() + datetime.timedelta(days=target_dte_max)
        
        if verbose: print(f"  [Detail] Fetching Options Chain ({target_dte_min}-{target_dte_max} days out)...")

        # 2. Fetch Chain (Filtered) 
        req = OptionChainRequest(
            underlying_symbol=symbol,
            expiration_date_gte=start_date,
            expiration_date_lte=end_date,
            strike_price_gte=spot_price * 0.85, # Wide enough to catch OTM Puts
            strike_price_lte=spot_price * 1.15
        )
        
        try:
            # Note: This returns a generator or paginated list. 
            # We must iterate to find a suitable expiration cycle (e.g., Monthly)
            chain_response = self.option_client.get_option_chain(req)
            
            if not chain_response:
                if verbose: print(f"  [Detail] Alpaca returned 0 contracts for query (Check strike range/dates).")
                return None
            
            # Parse into DataFrame for easier manipulation
            # Structure: symbol, type, strike, expiration, latest_quote (bid/ask)
            
            contracts = []
            skipped_counts = {"parse": 0, "no_quote": 0, "zero_price": 0, "wide_spread": 0}
            
            for contract_sym, data in chain_response.items():
                # We need snapshots to get pricing. The chain endpoint implies snapshots
                # if configured, or we must fetch them separately.
                # Assuming data contains 'latest_quote'
                
                # FIX: 'OptionsSnapshot' object has no attribute 'type/strike/expiration'
                # Must parse from symbol key
                parsed = self._parse_option_symbol(contract_sym)
                if not parsed:
                    skipped_counts["parse"] += 1
                    continue

                quote = data.latest_quote
                if not quote or quote.bid_price is None or quote.ask_price is None:
                    skipped_counts["no_quote"] += 1
                    continue
                    
                mid_price = (quote.bid_price + quote.ask_price) / 2
                if mid_price == 0:
                    skipped_counts["zero_price"] += 1
                    continue
                
                # SPREAD CHECK
                spread = quote.ask_price - quote.bid_price
                spread_pct = spread / mid_price
                if spread_pct > max_opt_spread:
                     # Skip illiquid options
                     skipped_counts["wide_spread"] += 1
                     continue

                # Check Open Interest (if available) - Fallback to 0
                open_int = getattr(data, 'open_interest', 0) or 0
                # Some API instances might have it nested or different key
                
                contracts.append({
                    'symbol': contract_sym,
                    'type': parsed['type'],
                    'strike': parsed['strike'],
                    'expiration': parsed['expiration'],
                    'mid_price': mid_price,
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'spread_pct': spread_pct,
                    'open_interest': open_int
                })
                
            df = pd.DataFrame(contracts)
            if df.empty:
                if verbose: 
                    print(f"  [Detail] No valid option contracts found.")
                    print(f"    Stats: Total={len(chain_response)}, Skipped: Spread={skipped_counts['wide_spread']} (>{max_opt_spread:.1%}), NoQuote={skipped_counts['no_quote']}, ZeroP={skipped_counts['zero_price']}")
                return None
            
            # Filter by Open Interest (soft filter, maybe only for chosen strikes?)
            # Let's filter globally to ensure we are in a liquid chain
            # But high strikes might have low OI.
            # We will check OI specifically on the chosen contracts later.

            # 3. Select Best Expiration (Most Volume/OI, or just closest to target)
            # Simplification: Take the expiration with the most contracts
            best_expiry = df['expiration'].mode().iloc[0]
            if verbose: print(f"  [Detail] Selected Expiration: {best_expiry} (Most Contracts)")

            df_expiry = df[df['expiration'] == best_expiry].copy()
            
            T = (pd.to_datetime(best_expiry) - pd.Timestamp.now()).days / 365.0
            
            # 4. Identify Strikes with Configurable Moneyness
            # target_moneyness: 1.0 = ATM, 0.98 = 2% ITM (for higher delta), 1.02 = 2% OTM
            target_moneyness = self.scan_opts.get('target_moneyness', 1.0)
            target_strike_price = spot_price * target_moneyness
            
            # Find the closest strike to our target moneyness
            call_strikes = df_expiry[df_expiry['type'] == 'call']['strike'].unique()
            if len(call_strikes) == 0:
                if verbose: print("  [Detail] No call strikes available.")
                return None
                
            target_strike = min(call_strikes, key=lambda x: abs(x - target_strike_price))
            
            # Also get true ATM for reference
            atm_strike = min(call_strikes, key=lambda x: abs(x - spot_price))
            
            if verbose:
                moneyness_pct = (target_strike / spot_price - 1) * 100
                print(f"  [Detail] Target Moneyness: {target_moneyness} -> Strike ${target_strike} ({moneyness_pct:+.1f}% from spot)")
            
            # OTM (approx 5-10% OTM for skew check)
            otm_put_strike_target = spot_price * 0.95
            otm_call_strike_target = spot_price * 1.05
            
            try:
                otm_put_strike = df_expiry[df_expiry['type'] == 'put'].iloc[(df_expiry[df_expiry['type'] == 'put']['strike'] - otm_put_strike_target).abs().argsort()[:1]]['strike'].values[0]
                otm_call_strike = df_expiry[df_expiry['type'] == 'call'].iloc[(df_expiry[df_expiry['type'] == 'call']['strike'] - otm_call_strike_target).abs().argsort()[:1]]['strike'].values[0]
                
                if verbose:
                    print(f"  [Detail] Strikes Selected -> ATM: {atm_strike}, Put: {otm_put_strike}, Call: {otm_call_strike}")
            except IndexError:
                if verbose: print("  [Detail] Could not find suitable OTM strikes.")
                return None

            # 5. Solve for IV (Using Local Solver)
            # ATM Call IV
            try:
                atm_call_row = df_expiry[(df_expiry['strike'] == atm_strike) & (df_expiry['type'] == 'call')].iloc[0]
                atm_iv = BlackScholesSolver.implied_volatility(
                    atm_call_row['mid_price'], spot_price, atm_strike, T, self.rf_rate, 'c'
                )
                # Store for failed candidate tracking
                self._last_atm_iv = atm_iv
            except IndexError:
                if verbose: print("  [Detail] ATM Call Analysis Failed.")
                return None
            
            if verbose: print(f"  [Detail] ATM IV: {atm_iv if atm_iv else 'N/A'} (Limit: {iv_max_threshold})")

            # Check vs Raw Limit
            pass_iv = True
            if atm_iv is None or atm_iv > iv_max_threshold:
                if verbose: print("    -> FAIL: IV too high or undef.")
                pass_iv = False
                if not verbose: return None # Failed IV check
            
            # Check vs HV (Relative Value)
            pass_iv_hv = True
            if hv20 > 0:
                iv_hv_ratio = atm_iv / hv20 if atm_iv else 0
                if verbose: print(f"  [Detail] IV/HV Ratio: {iv_hv_ratio:.2f} (Limit: {max_iv_hv_ratio})")
                if iv_hv_ratio > max_iv_hv_ratio:
                     if verbose: print(f"    -> FAIL: Options expensive relative to history (IV >> HV).")
                     pass_iv_hv = False
                     if not verbose: return None
            else:
                 iv_hv_ratio = 0.0

            # OTM Put IV
                
            # OTM Put IV
            try:
                otm_put_row = df_expiry[(df_expiry['strike'] == otm_put_strike) & (df_expiry['type'] == 'put')].iloc[0]
                otm_put_iv = BlackScholesSolver.implied_volatility(
                    otm_put_row['mid_price'], spot_price, otm_put_strike, T, self.rf_rate, 'p'
                )
                
                # OTM Call IV
                otm_call_row = df_expiry[(df_expiry['strike'] == otm_call_strike) & (df_expiry['type'] == 'call')].iloc[0]
                otm_call_iv = BlackScholesSolver.implied_volatility(
                    otm_call_row['mid_price'], spot_price, otm_call_strike, T, self.rf_rate, 'c'
                )

                # Target Strike Call/Put for Synthetic Long Position (may differ from ATM)
                target_call_row = df_expiry[(df_expiry['strike'] == target_strike) & (df_expiry['type'] == 'call')].iloc[0]
                target_put_row = df_expiry[(df_expiry['strike'] == target_strike) & (df_expiry['type'] == 'put')].iloc[0]
                
            except IndexError:
                if verbose: print("  [Detail] OTM/Target Strike Analysis Failed.")
                return None

            if otm_put_iv is None or otm_call_iv is None:
                if verbose: print(f"  [Detail] Could not solve OTM IVs. PutIV={otm_put_iv}, CallIV={otm_call_iv}")
                return None
            
            # 5b. Open Interest Check on Target Strike (where we'll trade)
            target_call_oi = target_call_row.get('open_interest', 0)
            target_put_oi  = target_put_row.get('open_interest', 0)
            
            if verbose:
                print(f"  [Detail] Target Strike Open Interest: Call={target_call_oi}, Put={target_put_oi} (Min: {min_oi})")
            
            pass_oi = True
            if target_call_oi < min_oi or target_put_oi < min_oi:
                 if verbose: print("    -> FAIL: Low Open Interest on target strike contracts.")
                 pass_oi = False
                 if not verbose: return None

            # 5c. Synthetic Long Metrics (Debit/Credit & Delta)
            # Long Call @ Target Strike, Short Put @ Target Strike
            # Cost = Call Ask - Put Bid
            # Delta = Call Delta - Put Delta (Short Put -> -(-Delta) -> +Delta)
            
            if strategy == "synthetic_long":
                # Long Call, Short Put
                entry_cost = target_call_row['ask'] - target_put_row['bid']
            elif strategy == "synthetic_short":
                # Long Put, Short Call
                entry_cost = target_put_row['ask'] - target_call_row['bid']
            else:
                entry_cost = 0.0
            
            # Solve IV for target strike options
            target_call_iv = BlackScholesSolver.implied_volatility(
                target_call_row['mid_price'], spot_price, target_strike, T, self.rf_rate, 'c'
            ) or atm_iv  # Fallback to ATM IV
            
            target_put_iv = BlackScholesSolver.implied_volatility(
                target_put_row['mid_price'], spot_price, target_strike, T, self.rf_rate, 'p'
            ) or atm_iv  # Fallback to ATM IV
            
            # Calculate Deltas
            target_call_delta = BlackScholesSolver.delta(spot_price, target_strike, T, self.rf_rate, target_call_iv, 'c')
            target_put_delta = BlackScholesSolver.delta(spot_price, target_strike, T, self.rf_rate, target_put_iv, 'p')
                
            if strategy == "synthetic_long":
                net_delta = target_call_delta - target_put_delta # Long Call (+), Short Put (-) -> Delta - (-Delta) = Sum
            elif strategy == "synthetic_short":
                net_delta = target_put_delta - target_call_delta # Long Put (-), Short Call (-) -> -Delta - Delta = -Sum
            else:
                net_delta = 0.0
            
            # 5d. Calculate Greeks for Risk Assessment
            # Theta: Daily time decay (Long Call loses, Short Put gains)
            call_theta = BlackScholesSolver.theta(spot_price, target_strike, T, self.rf_rate, target_call_iv, 'c')
            put_theta = BlackScholesSolver.theta(spot_price, target_strike, T, self.rf_rate, target_put_iv, 'p')
            
            if strategy == "synthetic_long":
                net_theta = call_theta - put_theta  # Short put theta is negated
            elif strategy == "synthetic_short":
                net_theta = put_theta - call_theta # Long Put (neg), Short Call (pos)
            else:
                net_theta = 0.0
            
            # Gamma: Acceleration of delta (same for both legs, additive)
            call_gamma = BlackScholesSolver.gamma(spot_price, target_strike, T, self.rf_rate, target_call_iv)
            put_gamma = BlackScholesSolver.gamma(spot_price, target_strike, T, self.rf_rate, target_put_iv)
            
            if strategy in ["synthetic_long", "synthetic_short"]:
                net_gamma = call_gamma + put_gamma
            else:
                net_gamma = 0.0
            
            # Vega: IV sensitivity (Long Call + Short Put both have vega exposure)
            call_vega = BlackScholesSolver.vega(spot_price, target_strike, T, self.rf_rate, target_call_iv)
            put_vega = BlackScholesSolver.vega(spot_price, target_strike, T, self.rf_rate, target_put_iv)
            
            if strategy == "synthetic_long":
                net_vega = call_vega - put_vega
            elif strategy == "synthetic_short":
                net_vega = put_vega - call_vega
            else:
                net_vega = 0.0
            
            # 5e. Put-Call Parity Check (Arbitrage Detection)
            # C - P = S - K*e^(-rT) (for same strike)
            # If mispriced, there's arbitrage opportunity or liquidity issue
            theoretical_diff = spot_price - target_strike * np.exp(-self.rf_rate * T)
            actual_diff = target_call_row['mid_price'] - target_put_row['mid_price']
            parity_error = abs(actual_diff - theoretical_diff)
            parity_error_pct = parity_error / spot_price * 100
            
            # 5f. Margin Estimate (Short Put requires margin)
            # Typical margin = max(20% of underlying - OTM amount, 10% of strike)
            otm_amount = max(0, target_strike - spot_price)  # For ATM, this is ~0
            margin_estimate = max(0.20 * spot_price - otm_amount, 0.10 * target_strike)
            
            # 5g. Break-even Analysis
            # Break-even = Strike + Net Debit (for synthetic)
            breakeven = target_strike + entry_cost
            breakeven_pct = (breakeven - spot_price) / spot_price * 100
            
            # 5h. Cost Efficiency vs Stock
            # Compare capital required: Synthetic vs buying 100 shares
            stock_cost = spot_price * 100
            synthetic_capital = (entry_cost * 100) + (margin_estimate * 100)  # Debit + Margin
            capital_efficiency = (1 - synthetic_capital / stock_cost) * 100 if stock_cost > 0 else 0
            
            # 5i. Max Loss Calculation (for risk management)
            # Max Loss on Synthetic Long = Strike Price (if stock goes to $0)
            # But practical max loss with stop = Strike - Stop_Price + Debit
            # For now, calculate max loss if stock drops 20% (protective scenario)
            protective_drop_pct = 0.20
            max_loss_scenario = (spot_price * protective_drop_pct * 100) + (entry_cost * 100)
            
            # 5j. Risk/Reward Ratio
            # Expected gain if stock rises 10% vs max loss scenario
            expected_gain_pct = 0.10
            expected_gain = spot_price * expected_gain_pct * 100  # Delta ~1, so gain ≈ stock gain
            risk_reward = expected_gain / max_loss_scenario if max_loss_scenario > 0 else 0
            
            # 5k. Probability of Profit (using Call Delta as proxy)
            # Delta of call ≈ probability stock finishes above strike
            prob_profit = target_call_delta * 100  # Convert to percentage
            
            # 5l. Days to Expiration for reference
            dte = (pd.to_datetime(best_expiry) - pd.Timestamp.now()).days
            
            # 5m. Position Sizing (based on max risk per trade)
            # Recommended contracts based on account size and risk tolerance
            account_size = self.scan_opts.get('account_size', 100000)  # Default $100k
            max_risk_pct = self.scan_opts.get('max_risk_per_trade', 0.02)  # Default 2%
            max_risk_dollars = account_size * max_risk_pct
            
            # Risk per contract = max loss scenario / 100 shares
            risk_per_contract = max_loss_scenario
            if risk_per_contract > 0:
                suggested_contracts = int(max_risk_dollars / risk_per_contract)
                suggested_contracts = max(1, suggested_contracts)  # At least 1
            else:
                suggested_contracts = 1
            
            # 5n. Greeks-based Quality Checks (configurable filters)
            min_delta = self.scan_opts.get('min_net_delta', 0.90)  # Min delta for synthetic
            max_theta_pct = self.scan_opts.get('max_theta_pct', 0.005)  # Max daily theta as % of position
            
            # Theta check: daily decay shouldn't exceed X% of synthetic capital
            theta_pct = abs(net_theta) / (synthetic_capital / 100) if synthetic_capital > 0 else 0
            pass_theta = theta_pct <= max_theta_pct # For long strategies, we pay theta. For short, we earn it.
            if strategy == "synthetic_short":
                pass_delta = net_delta <= -min_delta # Expect negative delta
            else:
                pass_delta = net_delta >= min_delta
            
            # 5o. Term Structure Analysis (Contango vs Backwardation)
            # Compare ATM IV of selected expiration vs a longer-dated expiration
            # Contango (normal): far-dated IV > near-dated IV -> favorable for long positions
            # Backwardation: near-dated IV > far-dated IV -> unfavorable, suggests event risk
            term_structure = "UNKNOWN"
            term_structure_spread = 0.0
            
            # Check if we have contracts at different expirations
            available_expiries = sorted(df['expiration'].unique())
            if len(available_expiries) > 1 and best_expiry != available_expiries[-1]:
                far_expiry = available_expiries[-1]
                df_far = df[df['expiration'] == far_expiry]
                
                # Find ATM call at far expiry
                far_calls = df_far[df_far['type'] == 'call']
                if len(far_calls) > 0:
                    far_atm_row = far_calls.iloc[(far_calls['strike'] - spot_price).abs().argsort()[:1]]
                    if len(far_atm_row) > 0:
                        far_T = (pd.to_datetime(far_expiry) - pd.Timestamp.now()).days / 365.0
                        far_iv = BlackScholesSolver.implied_volatility(
                            far_atm_row.iloc[0]['mid_price'], 
                            spot_price, 
                            far_atm_row.iloc[0]['strike'], 
                            far_T, 
                            self.rf_rate, 
                            'c'
                        )
                        if far_iv and atm_iv:
                            term_structure_spread = far_iv - atm_iv
                            if term_structure_spread > 0.01:
                                term_structure = "CONTANGO"  # Normal, favorable
                            elif term_structure_spread < -0.01:
                                term_structure = "BACKWARDATION"  # Event risk
                            else:
                                term_structure = "FLAT"
            
            if verbose:
                print(f"  [Detail] Greeks Filters:")
                print(f"    Net Delta: {net_delta:.3f} (Min: {min_delta}) -> {'PASS' if pass_delta else 'FAIL'}")
                print(f"    Theta %: {theta_pct:.4f} (Max: {max_theta_pct}) -> {'PASS' if pass_theta else 'FAIL'}")
                print(f"  [Detail] Term Structure: {term_structure} (Spread: {term_structure_spread:.4f})")
                print(f"  [Detail] Position Sizing:")
                print(f"    Account: ${account_size:,.0f}, Max Risk: {max_risk_pct:.1%}")
                print(f"    Risk/Contract: ${risk_per_contract:,.2f}")
                print(f"    Suggested Contracts: {suggested_contracts}")
            
            # Only fail on Greeks if not in verbose mode
            if not pass_theta or not pass_delta:
                if not verbose:
                    # Store specific failure reason
                    if not pass_delta:
                        self._last_option_failure = f"Delta too low ({net_delta:.2f} < {min_delta})"
                    elif not pass_theta:
                        self._last_option_failure = f"Theta too high ({theta_pct:.4f} > {max_theta_pct})"
                    return None
            
            if verbose:
                print(f"  [Detail] OTM Put IV: {otm_put_iv:.4f}")
                print(f"  [Detail] OTM Call IV: {otm_call_iv:.4f}")
                print(f"  [Detail] Entry Cost: ${entry_cost:.2f}")
                print(f"  [Detail] Net Delta: {net_delta:.2f} (Target ~1.0)")
                print(f"  [Detail] Net Theta: ${net_theta:.2f}/day (Time Decay)")
                print(f"  [Detail] Net Gamma: {net_gamma:.4f} (Delta Acceleration)")
                print(f"  [Detail] Net Vega: ${net_vega:.2f} (per 1% IV change)")
                print(f"  [Detail] Put-Call Parity Error: {parity_error_pct:.2f}% (Should be < 1%)")
                print(f"  [Detail] Margin Estimate: ${margin_estimate:.2f}/share")
                print(f"  [Detail] Max Leg Spread: {max_spread_val:.2%}")
                print(f"  [Detail] Break-even: ${breakeven:.2f} ({breakeven_pct:+.2f}% from spot)")
                print(f"  [Detail] Capital Efficiency: {capital_efficiency:.1f}% less capital vs stock")
                print(f"  [Detail] Max Loss (20% drop): ${max_loss_scenario:.2f} (per contract)")
                print(f"  [Detail] Risk/Reward (10% gain): {risk_reward:.2f}:1")
                print(f"  [Detail] Prob. of Profit: {prob_profit:.1f}%")
                print(f"  [Detail] Days to Expiration: {dte}")

            # 6. Calculate Skew
            # Skew = Put IV - Call IV
            skew = abs(otm_put_iv - otm_call_iv)
            # Store for failed candidate tracking
            self._last_skew = skew
            
            if verbose: print(f"  [Detail] Calculated Skew: {skew:.4f} (Limit: {skew_tolerance})")
            
            pass_skew = skew < skew_tolerance
            pass_all_options = pass_iv and pass_iv_hv and pass_skew and pass_oi
            
            # Store specific failure reason for failed candidates table
            if not pass_all_options:
                failures = []
                if not pass_iv:
                    failures.append(f"IV high ({atm_iv:.1%}>{iv_max_threshold:.0%})")
                if not pass_iv_hv:
                    failures.append(f"IV/HV high ({iv_hv_ratio:.2f}>{max_iv_hv_ratio})")
                if not pass_skew:
                    failures.append(f"Skew wide ({skew:.3f}>{skew_tolerance})")
                if not pass_oi:
                    failures.append(f"Low OI (<{min_oi})")
                self._last_option_failure = "; ".join(failures) if failures else "Options check failed"

            if pass_all_options:
                # Get IV percentile from momentum analysis (stored earlier)
                iv_pctl = getattr(self, '_last_iv_percentile', 50.0)
                spy_corr = getattr(self, '_last_spy_corr', 0.0)
                sector = getattr(self, '_last_sector', 'Unknown')
                macd_bullish = getattr(self, '_last_macd_bullish', False)
                rel_vol = getattr(self, '_last_rel_volume', 1.0)
                
                # Calculate composite score (0-100) for ranking
                # Higher = Better candidate
                # Factors: Low IV (good), Low Skew (good), High Delta (good), Low Cost (good), High R/R (good), MACD (good)
                score_iv = max(0, (iv_max_threshold - atm_iv) / iv_max_threshold * 20)  # 0-20 pts
                score_skew = max(0, (skew_tolerance - skew) / skew_tolerance * 15)  # 0-15 pts
                score_delta = min(20, net_delta * 20)  # 0-20 pts (max at delta=1)
                score_rr = min(15, risk_reward * 10)  # 0-15 pts
                score_cap_eff = min(15, capital_efficiency / 100 * 15)  # 0-15 pts
                score_macd = 10 if macd_bullish else 0  # 0-10 pts for bullish MACD
                score_volume = min(5, rel_vol * 2.5)  # 0-5 pts for relative volume
                composite_score = score_iv + score_skew + score_delta + score_rr + score_cap_eff + score_macd + score_volume
                
                return {
                    'symbol': symbol,
                    'sector': sector,
                    'price': spot_price,
                    'expiration': best_expiry,
                    'dte': dte,
                    'strike': target_strike,  # The actual strike we'd trade
                    'moneyness': round((target_strike / spot_price - 1) * 100, 1),  # % from spot
                    'atm_iv': round(atm_iv, 4),
                    'hv20': round(hv20, 4),
                    'iv_hv_ratio': round(iv_hv_ratio, 2),
                    'iv_pctl': round(iv_pctl, 0),
                    'term_structure': term_structure,
                    'skew': round(skew, 4),
                    'syn_cost': round(entry_cost, 2),
                    'net_delta': round(net_delta, 3),
                    'theta': round(net_theta, 2),
                    'theta_pct': round(theta_pct * 100, 2),  # Theta as % of capital
                    'gamma': round(net_gamma, 4),
                    'vega': round(net_vega, 2),
                    'breakeven': round(breakeven, 2),
                    'margin_est': round(margin_estimate, 2),
                    'max_spread': round(max_spread_val * 100, 2),
                    'cap_eff_pct': round(capital_efficiency, 1),
                    'max_loss': round(max_loss_scenario, 2),
                    'risk_reward': round(risk_reward, 2),
                    'prob_profit': round(prob_profit, 1),
                    'suggested_qty': suggested_contracts,
                    'macd_bullish': macd_bullish,
                    'rel_volume': round(rel_vol, 2),
                    'spy_corr': round(spy_corr, 2),
                    'score': round(composite_score, 1),
                    'parity_err': round(parity_error_pct, 2)
                }
            else:
                 if verbose: 
                     print(f"    -> Option Checks Failed (IV: {pass_iv}, IV/HV: {pass_iv_hv}, OI: {pass_oi}, Skew: {pass_skew})")
                
        except Exception as e:
            if verbose: print(f"Chain analysis failed for {symbol}: {e}")
            return None
            
        return None

    def _show_recommendations(self, failed_candidates: List[Dict], count: int = 5):
        """
        Shows recommendations from failed candidates, prioritized by how close they were to passing.
        Priority order (closest to passing first):
        1. Options failures (IV/Skew/Greeks) - passed momentum, just missed on options
        2. Momentum failures - stock is liquid, just needs better trend
        3. Earnings/Dividend risks - calendar-based, may pass later
        """
        if not failed_candidates:
            return
        
        # Assign priority scores (lower = closer to passing)
        def get_priority(candidate):
            reason = candidate.get('failure_reason', '')
            
            # Options failures are closest to passing (passed all other checks)
            if 'IV' in reason or 'Skew' in reason or 'Delta' in reason or 'Theta' in reason or 'OI' in reason:
                priority = 1
            elif 'No valid contracts' in reason:
                priority = 2
            # Momentum failures are next
            elif 'Momentum' in reason or 'Trend' in reason:
                priority = 3
            # Event risks are calendar-dependent
            elif 'Earnings' in reason:
                priority = 4
            elif 'Dividend' in reason:
                priority = 5
            else:
                priority = 6
            
            # Secondary sort: by RSI if available (higher RSI = stronger momentum)
            try:
                rsi = float(candidate.get('rsi', '0').replace('-', '0'))
            except:
                rsi = 0
            
            return (priority, -rsi)  # Negative RSI for descending sort
        
        # Sort by priority
        sorted_candidates = sorted(failed_candidates, key=get_priority)
        
        # Take top N
        recommendations = sorted_candidates[:count]
        
        print(f"\n{'='*60}")
        print(f"📋 RECOMMENDATIONS (Top {len(recommendations)} Near-Miss Candidates)")
        print(f"{'='*60}")
        print("These stocks came closest to passing all filters:\n")
        
        for i, rec in enumerate(recommendations, 1):
            symbol = rec.get('symbol', 'N/A')
            price = rec.get('price', 0)
            reason = rec.get('failure_reason', 'Unknown')
            rsi = rec.get('rsi', '-')
            iv = rec.get('iv', '-')
            
            print(f"{i}. {symbol} @ ${price:.2f}")
            print(f"   Failed: {reason}")
            print(f"   RSI: {rsi}, IV: {iv}")
            
            # Provide actionable advice based on failure reason
            if 'IV high' in reason or 'IV/HV' in reason:
                print(f"   💡 Tip: Wait for IV to drop or sell premium strategies instead")
            elif 'Skew wide' in reason:
                print(f"   💡 Tip: High put premium suggests fear - consider put spreads")
            elif 'Delta' in reason:
                print(f"   💡 Tip: Try slightly ITM strikes for better delta")
            elif 'Momentum' in reason or 'Trend' in reason:
                print(f"   💡 Tip: Wait for trend confirmation (Price > SMA50 > SMA200)")
            elif 'Earnings' in reason:
                print(f"   💡 Tip: Re-scan after earnings announcement")
            elif 'Dividend' in reason:
                print(f"   💡 Tip: Re-scan after ex-dividend date")
            elif 'OI' in reason:
                print(f"   💡 Tip: Try different expiration with more liquidity")
            print()
        
        print(f"{'='*60}\n")

    async def run(self, strategy: Optional[str] = None, limit: Optional[int] = None, symbol: Optional[Union[str, List[str]]] = None, verbose: bool = False, max_spread: Optional[float] = None):
        if strategy is None:
            strategy = self.config.get('option_scan', {}).get('default_strategy', 'synthetic_long')
            
        if max_spread is not None:
            self.scan_opts['max_option_spread_pct'] = max_spread / 100.0
            print(f"Overriding Max Option Spread: {max_spread}%")

        print(f"--- Starting Scanner (Strategy: {strategy}) ---")
        
        # Handle symbol input (str or list)
        target_symbols = []
        if symbol:
            if isinstance(symbol, str):
                target_symbols = [symbol.upper()]
            else:
                target_symbols = [s.upper() for s in symbol]
        
        explicit_mode = len(target_symbols) > 0
        # Enable verbose logging if explicit mode OR explicit verbose flag
        is_verbose = explicit_mode or verbose
        
        if explicit_mode:
            print(f"--- Explicit Scan Mode: {', '.join(target_symbols)} ---")
            # For explicit mode, we bypass the bulk logic but still need checks
            # We assume it's optionable or let it fail downstream
            candidates = target_symbols
        else:
            # Step 1: Universe
            candidates = await self.get_universe(limit_override=limit)
        
        print(f"Universe Size: {len(candidates)} candidates")
        
        # Step 2: Liquidity Snapshot
        print("Fetching market data snapshot...")
        snapshot_df = self.get_market_data_snapshot(candidates)
        print(f"Snapshot complete. {len(snapshot_df)} assets meet minimum price ${self.scan_opts.get('min_price', 15.0)}")
        
        results = []
        failed_candidates = []  # Track failed candidates
        
        total = len(snapshot_df)
        for index, row in snapshot_df.iterrows():
            sym = row['symbol']
            price = row['price']
            
            print(f"[{index+1}/{total}] Scanning {sym} (${price:.2f})... ", end="", flush=True)
            if is_verbose: print("") # New line for detail

            if strategy in ["synthetic_long", "synthetic_short", "short_call"]:
                # Initialize failure tracking
                failure_reason = None
                rsi = 0.0
                hv20 = 0.0
                # Clear previous option failure reason
                self._last_option_failure = None
                self._last_atm_iv = None
                self._last_skew = None
                
                # Step 3: Dividend & Earnings Check
                pass_div = self.check_dividend_risk(sym, verbose=is_verbose)
                if not pass_div:
                    print("Skipped (Dividend/Yield Risk)")
                    failure_reason = "Dividend Risk"
                    if not is_verbose:
                        failed_candidates.append({
                            'symbol': sym, 'price': price, 'failure_reason': failure_reason,
                            'rsi': '-', 'adx': '-', 'macd': '-', 'iv': '-', 'skew': '-'
                        })
                        continue
                
                pass_earn = self.check_earnings_risk(sym, verbose=is_verbose)
                if not pass_earn:
                     print("Skipped (Earnings Risk)")
                     failure_reason = "Earnings Risk"
                     if not is_verbose:
                         failed_candidates.append({
                             'symbol': sym, 'price': price, 'failure_reason': failure_reason,
                             'rsi': '-', 'adx': '-', 'macd': '-', 'iv': '-', 'skew': '-'
                         })
                         continue
                    
                # Step 4: Momentum Check
                trend_view = "bullish"
                if strategy in ["synthetic_short", "short_call"]:
                    trend_view = "bearish"
                
                pass_mom, rsi, hv20 = self.analyze_momentum(sym, trend_view=trend_view, verbose=is_verbose)
                
                # Get momentum details for failed table
                adx_val = getattr(self, '_last_adx', 0.0) if hasattr(self, '_last_adx') else '-'
                macd_status = 'Bullish' if getattr(self, '_last_macd_bullish', False) else 'Bearish'
                
                if not pass_mom:
                    print(f"Skipped (Momentum/Trend RSI={rsi:.1f})")
                    failure_reason = "Momentum/Trend"
                    if not is_verbose:
                        failed_candidates.append({
                            'symbol': sym, 'price': price, 'failure_reason': failure_reason,
                            'rsi': f"{rsi:.1f}", 'adx': f"{adx_val}" if isinstance(adx_val, str) else f"{adx_val:.1f}",
                            'macd': macd_status, 'iv': '-', 'skew': '-'
                        })
                        continue
                
                if not is_verbose:
                    print(f"PASS. Checking Options (RSI: {rsi:.1f}, HV: {hv20:.1%})...")
                else:
                    print(f"  [Detail] Momentum PASS. Checking Options...")
                
                # Step 5: Volatility/Skew Analysis
                vol_metrics = self.analyze_volatility_skew(sym, price, hv20, verbose=is_verbose, strategy=strategy)
                
                passed_all = pass_div and pass_earn and pass_mom and (vol_metrics is not None)

                if passed_all:
                    vol_metrics['rsi'] = rsi
                    results.append(vol_metrics)
                    print(f"*** MATCH FOUND: {sym} ***")
                else:
                    # Get specific failure reason from options analysis
                    specific_failure = getattr(self, '_last_option_failure', None)
                    if specific_failure:
                        failure_reason = specific_failure
                    else:
                        failure_reason = "Options (No valid contracts)"
                    
                    # Try to get IV/Skew info if available
                    iv_val = getattr(self, '_last_atm_iv', '-') if hasattr(self, '_last_atm_iv') else '-'
                    skew_val = getattr(self, '_last_skew', '-') if hasattr(self, '_last_skew') else '-'
                    
                    failed_candidates.append({
                        'symbol': sym, 'price': price, 'failure_reason': failure_reason,
                        'rsi': f"{rsi:.1f}", 'adx': f"{adx_val}" if isinstance(adx_val, str) else f"{adx_val:.1f}",
                        'macd': macd_status,
                        'iv': f"{iv_val:.2%}" if isinstance(iv_val, float) else iv_val,
                        'skew': f"{skew_val:.4f}" if isinstance(skew_val, float) else skew_val
                    })
                    
                    if is_verbose:
                        print(f"    -> Match Failed (Div: {pass_div}, Earn: {pass_earn}, Mom: {pass_mom}, Options: {vol_metrics is not None})")
                    else:
                        print(f"    -> No suitable option structure found.")
            else:
                 print(f"Strategy '{strategy}' not implemented.")
                 continue
        
        # =====================================================================
        # OUTPUT SECTION (All tables at the end)
        # =====================================================================
        
        print(f"\n{'='*70}")
        print(f"                         SCAN RESULTS")
        print(f"{'='*70}")
        
        output_dir = Path("trading_logs/scanner")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Failed Candidates Table (show first for context)
        if failed_candidates:
            df_failed = pd.DataFrame(failed_candidates)
            
            # Group by failure reason for summary
            failure_summary = df_failed['failure_reason'].value_counts()
            
            print(f"\n--- Failed Candidates Summary ---")
            print(f"Total Failed: {len(df_failed)}")
            print(f"Failure Breakdown:")
            for reason, count in failure_summary.items():
                print(f"  • {reason}: {count}")
            
            print(f"\n--- Failed Candidates Detail ---")
            display_cols_failed = ['symbol', 'price', 'failure_reason', 'rsi', 'adx', 'macd', 'iv', 'skew']
            print(df_failed[display_cols_failed].to_markdown(index=False))
            
            # Save failed results to CSV (only for batch mode)
            if not explicit_mode:
                df_failed.to_csv(output_dir / "synthetic_long_failed.csv", index=False)
                print(f"\nFailed candidates saved to {output_dir / 'synthetic_long_failed.csv'}")
        
        # 2. Recommendations (if no successful candidates)
        if not results and failed_candidates:
            recommendation_count = self.scan_opts.get('recommendation_count', 5)
            self._show_recommendations(failed_candidates, recommendation_count)
        
        # 3. Final Candidates Table (show last - the main result)
        if results:
            df_results = pd.DataFrame(results)
            
            # Sort by composite score (best candidates first)
            df_results = df_results.sort_values(by='score', ascending=False)
            
            print(f"\n--- Final Candidates (Sorted by Score) ---")
            print(f"Market Regime: {self.market_regime}")
            
            # Show sector distribution
            if 'sector' in df_results.columns:
                sector_counts = df_results['sector'].value_counts()
                print(f"Sector Distribution: {dict(sector_counts)}")
            
            # Select columns for display (exclude some for readability)
            display_cols = ['symbol', 'sector', 'price', 'dte', 'atm_iv', 'skew', 
                           'syn_cost', 'net_delta', 'cap_eff_pct', 'risk_reward', 'max_spread', 'score']
            display_cols = [c for c in display_cols if c in df_results.columns]
            
            print(df_results[display_cols].to_markdown(index=False))
            
            # Save full results to CSV
            df_results.to_csv(output_dir / "synthetic_long_candidates.csv", index=False)
            print(f"\nFull results saved to {output_dir / 'synthetic_long_candidates.csv'}")
        else:
            print("\n--- Final Candidates ---")
            print("No candidates found matching all criteria.")
            if self.vix_level > 25:
                print("Note: High VIX environment - consider waiting for volatility to subside.")
        
        print(f"\n{'='*70}")

# -----------------------------------------------------------------------------
# EXECUTION ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Strategy Option Scanner")
    parser.add_argument("--strategy", type=str, 
                        choices=["synthetic_long", "synthetic_short"],
                        help="Scan strategy to execute (default: read from config)")
    parser.add_argument("--limit", type=int, help="Limit the number of stocks to scan (overrides config)")
    parser.add_argument("--symbol", type=str, nargs='+', help="Scan specific stock symbol(s) and show detailed metrics")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Max bid/ask spread percent (default: 5.0)")
    args = parser.parse_args()

    scanner = OptionStrategyScanner()
    asyncio.run(scanner.run(strategy=args.strategy, limit=args.limit, symbol=args.symbol, verbose=args.verbose, max_spread=args.max_spread))
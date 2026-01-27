#!/usr/bin/env python3
"""
Get next earnings date for a stock from Yahoo Finance using yfinance.

Usage:
    python util/get_yahoo_earnings.py AAPL
"""

import argparse
import sys
import pandas as pd
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance is not installed. Please run: pip install yfinance", file=sys.stderr)
    sys.exit(1)

def get_next_earnings(symbol):
    symbol = symbol.upper()
    print(f"Fetching earnings data for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Method 1: get_earnings_dates()
        try:
            earnings_dates = ticker.get_earnings_dates(limit=20)
            if earnings_dates is not None and not earnings_dates.empty:
                if not pd.api.types.is_datetime64_any_dtype(earnings_dates.index):
                     earnings_dates.index = pd.to_datetime(earnings_dates.index)
                
                now = pd.Timestamp.now().floor('D')
                future_earnings = earnings_dates[earnings_dates.index >= now].sort_index()
                
                if not future_earnings.empty:
                    next_date = future_earnings.index[0]
                    print(f"Next Earnings Date: {next_date.date()}")
                    return
        except Exception:
            pass

        # Method 2: calendar property
        calendar = ticker.calendar
        if calendar is not None:
            if isinstance(calendar, dict):
                if 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    if isinstance(dates, list):
                        print(f"Next Earnings Date: {dates[0]}")
                    else:
                        print(f"Next Earnings Date: {dates}")
                    return
            elif hasattr(calendar, 'index'):
                if 'Earnings Date' in calendar.index:
                    val = calendar.loc['Earnings Date']
                    print(f"Next Earnings Date: {val}")
                    return
        
        print(f"Could not find upcoming earnings date for {symbol}.")

    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Get next earnings date from Yahoo Finance")
    parser.add_argument("symbol", help="Stock symbol (e.g. AAPL)")
    args = parser.parse_args()

    get_next_earnings(args.symbol)

if __name__ == "__main__":
    main()
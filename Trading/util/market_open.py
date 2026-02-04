#!/usr/bin/env python3
"""
Checks if the market is open using Alpaca API.
Exits with 0 if open, 1 if closed.
"""
import sys
import os
from pathlib import Path

# Resolve paths
current_file = Path(__file__).resolve()
trading_root = current_file.parent.parent # Trading/

# Add Trading/Trading to sys.path to import alpaca_client
sys.path.insert(0, str(trading_root / "Trading"))

try:
    from alpaca_client import AlpacaClient
    from dotenv import load_dotenv
    
    # Load env vars
    project_root = trading_root.parent
    load_dotenv(trading_root / ".env")
    load_dotenv(project_root / ".env")
    
    client = AlpacaClient()
    clock = client.get_clock()
    
    if clock.get('is_open'):
        print("Market is OPEN")
        sys.exit(0)
    else:
        next_open = clock.get('next_open')
        print(f"Market is CLOSED. Next open: {next_open}")
        sys.exit(1)
        
except Exception as e:
    print(f"Error checking market status: {e}", file=sys.stderr)
    sys.exit(1)
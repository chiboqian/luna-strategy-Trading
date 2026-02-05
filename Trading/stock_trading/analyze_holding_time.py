#!/usr/bin/env python3
"""
Analyze the average holding time of closed positions.
Calculates duration based on FIFO (First-In, First-Out) matching of Buy and Sell orders.

Usage:
    python Trading/stock_trading/analyze_holding_time.py --days 365
    python Trading/stock_trading/analyze_holding_time.py --symbol AAPL --csv
"""

import argparse
import sys
import statistics
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path

# Resolve paths
script_path = Path(__file__).resolve()
trading_root = script_path.parent.parent # Trading/

# Add Trading/Trading to sys.path to import alpaca_client
sys.path.insert(0, str(trading_root / "Trading"))

try:
    from alpaca_client import AlpacaClient
except ImportError:
    print("Error: Could not import AlpacaClient.", file=sys.stderr)
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze average holding time of closed positions")
    parser.add_argument("--days", type=int, default=365, help="Number of days to look back (default: 365)")
    parser.add_argument("--symbol", help="Filter by specific symbol")
    parser.add_argument("--csv", action="store_true", help="Output detailed trade list to trade_holding_times.csv")
    return parser.parse_args()

def get_all_filled_orders(client, days, symbol=None):
    """Fetch all filled orders within the lookback period."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    
    print(f"Fetching orders since {start_dt.date()}...", file=sys.stderr)
    
    # Fetch closed orders (filled/canceled/expired)
    # Note: Alpaca default limit is 50. We request 500. 
    # For a robust production script with thousands of trades, pagination loop would be needed.
    params = {
        'status': 'closed',
        'limit': 500,
        'after': start_dt.isoformat()
    }
    if symbol:
        params['symbols'] = [symbol]

    orders = client.get_orders(**params)
    
    # Filter for actually filled orders
    filled_orders = []
    for o in orders:
        if o.get('filled_at') and float(o.get('filled_qty', 0)) > 0:
            filled_orders.append(o)
            
    # Sort by filled_at ascending for FIFO processing
    filled_orders.sort(key=lambda x: x['filled_at'])
    
    return filled_orders

def calculate_holding_times(orders):
    """
    Process orders to calculate holding times using FIFO.
    Returns a list of dictionaries: {'symbol', 'open_time', 'close_time', 'duration', 'qty', 'side'}
    """
    # Inventory: symbol -> deque of {'time': datetime, 'qty': float}
    inventory = defaultdict(deque)
    position_qty = defaultdict(float)
    trades = []
    
    for order in orders:
        symbol = order['symbol']
        side = order['side'] # 'buy' or 'sell'
        qty = float(order['filled_qty'])
        # Parse timestamp (handle Z)
        fill_time = datetime.fromisoformat(order['filled_at'].replace('Z', '+00:00'))
        
        curr_pos = position_qty[symbol]
        remaining_order_qty = qty
        
        while remaining_order_qty > 0:
            curr_pos = position_qty[symbol]
            
            # Determine if we are Opening (increasing pos) or Closing (decreasing pos)
            is_opening = False
            if curr_pos == 0:
                is_opening = True
            elif curr_pos > 0 and side == 'buy':
                is_opening = True
            elif curr_pos < 0 and side == 'sell':
                is_opening = True
            
            if is_opening:
                # Add to inventory
                inventory[symbol].append({'time': fill_time, 'qty': remaining_order_qty})
                if side == 'buy':
                    position_qty[symbol] += remaining_order_qty
                else:
                    position_qty[symbol] -= remaining_order_qty
                remaining_order_qty = 0
            else:
                # Closing / Reducing
                if not inventory[symbol]:
                    # History incomplete or data mismatch; treat as new open to avoid crash
                    inventory[symbol].append({'time': fill_time, 'qty': remaining_order_qty})
                    if side == 'buy':
                        position_qty[symbol] += remaining_order_qty
                    else:
                        position_qty[symbol] -= remaining_order_qty
                    remaining_order_qty = 0
                    continue

                match = inventory[symbol][0]
                match_qty = match['qty']
                
                qty_to_close = min(remaining_order_qty, match_qty)
                
                # Record trade
                duration = fill_time - match['time']
                trades.append({
                    'symbol': symbol,
                    'open_time': match['time'],
                    'close_time': fill_time,
                    'duration': duration,
                    'qty': qty_to_close,
                    'type': 'Long' if side == 'sell' else 'Short'
                })
                
                # Update state
                match['qty'] -= qty_to_close
                remaining_order_qty -= qty_to_close
                
                if side == 'buy':
                    position_qty[symbol] += qty_to_close
                else:
                    position_qty[symbol] -= qty_to_close
                
                if match['qty'] <= 0.000001: # Float tolerance
                    inventory[symbol].popleft()
                    
    return trades

def main():
    args = parse_args()
    
    client = AlpacaClient()
    
    orders = get_all_filled_orders(client, args.days, args.symbol)
    if not orders:
        print("No filled orders found in the specified period.")
        return

    print(f"Processing {len(orders)} orders...")
    trades = calculate_holding_times(orders)
    
    if not trades:
        print("No completed trades found (positions might still be open or history is incomplete).")
        return

    # Analysis
    durations = [t['duration'].total_seconds() / 86400.0 for t in trades] # Days
    
    avg_days = statistics.mean(durations)
    median_days = statistics.median(durations)
    max_days = max(durations)
    min_days = min(durations)
    
    print("\n" + "="*40)
    print("HOLDING TIME ANALYSIS")
    print("="*40)
    print(f"Total Closed Trades: {len(trades)}")
    print(f"Average Holding Time: {avg_days:.2f} days")
    print(f"Median Holding Time:  {median_days:.2f} days")
    print(f"Min Holding Time:     {min_days:.4f} days")
    print(f"Max Holding Time:     {max_days:.2f} days")
    print("="*40)

    if args.csv:
        filename = "trade_holding_times.csv"
        print(f"\nWriting detailed log to {filename}...")
        with open(filename, 'w') as f:
            f.write("symbol,type,open_time,close_time,qty,duration_days\n")
            for t in trades:
                days = t['duration'].total_seconds() / 86400.0
                f.write(f"{t['symbol']},{t['type']},{t['open_time']},{t['close_time']},{t['qty']},{days:.4f}\n")

if __name__ == "__main__":
    main()
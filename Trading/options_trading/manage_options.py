#!/usr/bin/env python3
"""
Manage Option Positions (Take Profit, Stop Loss, DTE).

Scans open option positions, groups them by Underlying and Expiration (to identify spreads),
and checks against management rules:
1. Take Profit: Close if profit >= X% of max profit (default 50%).
2. Stop Loss: Close if loss >= Y% of max profit (default 100% - i.e., loss equals credit received).
3. DTE: Close if days to expiration <= Z (default 2).

Usage:
    python util/manage_options.py --dry-run
    python util/manage_options.py --tp 0.5 --sl 1.0 --dte 7
    python util/manage_options.py --symbol SPY
"""

import sys
import argparse
import json
import logging
import yaml
import re
import math
from datetime import datetime
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ManageOptions")

# Path setup
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "Trading"))
try:
    from alpaca_client import AlpacaClient
    from logging_config import setup_logging
except ImportError:
    print("Error: Could not import AlpacaClient.", file=sys.stderr)
    sys.exit(1)

try:
    from options_trading.mock_client import MockOptionClient
    import pandas as pd
except ImportError:
    MockOptionClient = None
    pd = None

def parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parses OCC option symbol.
    Format: Root(6) + Date(6) + Type(1) + Strike(8)
    Example: AAPL  230616C00150000 (spaces might be trimmed by Alpaca)
    Alpaca often returns: AAPL230616C00150000
    """
    # Regex to handle variable length root (greedy) followed by fixed length suffix
    # Suffix is 15 chars: YYMMDD (6) + T (1) + Strike (8)
    if len(symbol) < 15:
        return None
        
    suffix = symbol[-15:]
    root = symbol[:-15].strip()
    
    date_str = suffix[:6]
    type_char = suffix[6]
    strike_str = suffix[7:]
    
    try:
        expiry = datetime.strptime(date_str, "%y%m%d")
        strike = float(strike_str) / 1000.0
        return {
            'root': root,
            'expiry': expiry,
            'type': 'call' if type_char.upper() == 'C' else 'put',
            'strike': strike,
            'symbol': symbol
        }
    except ValueError:
        return None

def group_positions(positions: List[Dict]) -> Dict[tuple, List[Dict]]:
    """Groups positions by (Root, Expiration)."""
    groups = defaultdict(list)
    for pos in positions:
        if pos.get('asset_class') != 'us_option':
            continue
            
        parsed = parse_option_symbol(pos['symbol'])
        if parsed:
            # Key by Root and Expiration Date
            key = (parsed['root'], parsed['expiry'])
            # Merge parsed info into position dict
            groups[key].append({**pos, **parsed})
            
    return groups

def calculate_group_metrics(legs: List[Dict]) -> Dict[str, Any]:
    """Calculates P/L and cost metrics for a group of option legs."""
    
    # Initialize
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    total_cost_basis = 0.0
    
    # Determine if it's a Credit or Debit strategy
    # Alpaca: 
    #   Short: qty < 0, market_value < 0 (liability), cost_basis < 0 (proceeds?) 
    #   Actually Alpaca cost_basis is usually positive absolute value for calculation, 
    #   but let's rely on: Initial = MarketValue - UnrealizedPL
    
    net_initial_value = 0.0
    
    for leg in legs:
        qty = float(leg.get('qty', 0))
        mv = float(leg.get('market_value', 0))
        pl = float(leg.get('unrealized_pl', 0))
        
        total_market_value += mv
        total_unrealized_pl += pl
        
        # Infer initial value (negative = credit received, positive = debit paid)
        initial = mv - pl
        net_initial_value += initial

    # Strategy Type
    is_credit = net_initial_value < -0.01
    
    # Max Profit / Risk Basis
    if is_credit:
        # Credit Strategy (e.g. Bull Put Spread)
        # Max Profit is the credit received (absolute value of initial)
        basis = abs(net_initial_value)
        # P/L % is Profit / Max Profit
        pl_pct = total_unrealized_pl / basis if basis > 0 else 0.0
    else:
        # Debit Strategy (e.g. Long Call, Debit Spread)
        # Basis is the debit paid
        basis = net_initial_value
        # P/L % is Profit / Cost
        pl_pct = total_unrealized_pl / basis if basis > 0 else 0.0
        
    return {
        'market_value': total_market_value,
        'unrealized_pl': total_unrealized_pl,
        'net_initial': net_initial_value,
        'is_credit': is_credit,
        'basis': basis,
        'pl_pct': pl_pct
    }

def run_mock_loop(args):
    """Runs the management logic against historical data."""
    if not MockOptionClient or not pd:
        print("Error: pandas and MockOptionClient required for mock mode.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Starting Mock Management Backtest ---")
    print(f"Position File: {args.mock_position}")
    print(f"Historical Data: {args.historical}")
    
    # 1. Load Position
    try:
        with open(args.mock_position, 'r') as f:
            position_data = json.load(f)
    except Exception as e:
        print(f"Error loading position file: {e}", file=sys.stderr)
        sys.exit(1)

    legs_config = position_data.get('legs', [])
    if not legs_config:
        print("No legs found in position file.", file=sys.stderr)
        sys.exit(1)

    # Determine symbols and expiration from the first leg
    # Assuming all legs have same expiration for the strategy (Vertical/Iron Condor)
    first_leg_parsed = parse_option_symbol(legs_config[0]['symbol'])
    if not first_leg_parsed:
        print(f"Could not parse symbol: {legs_config[0]['symbol']}", file=sys.stderr)
        sys.exit(1)
        
    expiry_date = first_leg_parsed['expiry']
    root_symbol = first_leg_parsed['root']
    
    # Determine Start Date
    if args.start_date:
        current_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        # Try to infer from position data
        filled_at = position_data.get('filled_at') or position_data.get('created_at')
        if filled_at:
            try:
                # Handle ISO format (e.g. 2023-01-01T09:30:00)
                date_str = filled_at.split('T')[0] if 'T' in filled_at else filled_at.split(' ')[0]
                current_date = datetime.strptime(date_str, "%Y-%m-%d")
                if not args.json:
                    print(f"Inferred start date from order: {current_date.date()}")
            except ValueError:
                print("Error: Could not parse date from order file. Please use --start-date.", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: --start-date is required for mock mode.", file=sys.stderr)
            sys.exit(1)

    if not args.json:
        print(f"Monitoring {root_symbol} position expiring {expiry_date.date()}")
        print(f"Start Date: {current_date.date()}")

    last_metrics = None

    # 2. Iterate Days
    while current_date <= expiry_date:
        date_str = current_date.strftime("%Y-%m-%d")
        if not args.json:
            print(f"Processing {date_str}...", end="", flush=True)
        else:
            print(f"Processing {date_str}...", file=sys.stderr)
        
        # Load Data for the day (No deduplication to get all minutes)
        try:
            # We use MockOptionClient to handle file resolution and loading
            # We pass deduplicate=False to get the full time series
            mock_client = MockOptionClient(args.historical, current_date, deduplicate=False)
            df = mock_client.df
        except FileNotFoundError:
            if not args.json:
                print(f" No data found. Skipping.")
            else:
                print(f" No data found for {date_str}. Skipping.", file=sys.stderr)
            current_date += timedelta(days=1)
            continue
        except Exception as e:
            if not args.json:
                print(f" Error loading data: {e}")
            else:
                print(f" Error loading data for {date_str}: {e}", file=sys.stderr)
            current_date += timedelta(days=1)
            continue
            
        if df is None or df.empty:
            if not args.json:
                print(f" Empty data.")
            else:
                print(f" Empty data for {date_str}.", file=sys.stderr)
            current_date += timedelta(days=1)
            continue

        # Identify time column
        date_col = next((c for c in ['ts_event', 'date', 'quote_date', 'timestamp', 'time'] if c in df.columns), None)
        if not date_col:
            if not args.json:
                print(" No date column found.")
            else:
                print(f" No date column found for {date_str}.", file=sys.stderr)
            current_date += timedelta(days=1)
            continue
            
        # Filter for our symbols to speed up iteration
        my_symbols = [l['symbol'] for l in legs_config]
        # Assuming 'symbol' column exists (MockOptionClient ensures mapping)
        sym_col = mock_client.col_map.get('symbol', 'symbol')
        
        if sym_col in df.columns:
            df = df[df[sym_col].isin(my_symbols)].copy()
        
        if df.empty:
            if not args.json:
                print(" No data for position symbols.")
            else:
                print(f" No data for position symbols on {date_str}.", file=sys.stderr)
            current_date += timedelta(days=1)
            continue
            
        # Sort by time
        df = df.sort_values(date_col)
        
        # Group by timestamp (minute bars)
        # We can iterate unique timestamps
        timestamps = df[date_col].unique()
        if not args.json:
            print(f" {len(timestamps)} time steps.")
        else:
            print(f" {len(timestamps)} time steps.", file=sys.stderr)
        
        for ts in timestamps:
            # Get snapshot for this minute
            # This is a bit slow for large files, but accurate
            minute_df = df[df[date_col] == ts]
            
            # Construct "Positions" list for calculate_group_metrics
            current_legs = []
            
            # We need prices for ALL legs to calculate P/L validly
            # If a leg is missing in this minute, we might need to use last known price?
            # For simplicity, we skip minutes where we don't have full data, 
            # OR we assume the previous price holds (forward fill logic would be better outside loop).
            
            # Simple check: do we have all symbols?
            available_syms = minute_df[sym_col].unique()
            if not all(s in available_syms for s in my_symbols):
                continue # Skip incomplete data points
                
            for leg_conf in legs_config:
                sym = leg_conf['symbol']
                row = minute_df[minute_df[sym_col] == sym].iloc[-1]
                
                # Determine Price (Mid)
                # MockOptionClient maps columns. Let's use the map.
                bid_col = mock_client.col_map.get('bid', 'bid')
                ask_col = mock_client.col_map.get('ask', 'ask')
                
                bid = float(row.get(bid_col, 0))
                ask = float(row.get(ask_col, 0))
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                
                # If price is 0, skip
                if mid == 0:
                    break
                
                # Calculate Market Value & Unrealized P/L
                # Entry Price is needed. 
                # The strategy JSON usually has 'estimated_price' for each leg or we can infer.
                # If 'estimated_price' is in leg_conf, use it as cost basis.
                entry_price = float(leg_conf.get('estimated_price', 0))
                qty = float(leg_conf.get('ratio_qty', 1)) # Assuming ratio 1 for simplicity or read from file
                
                # Side
                side = leg_conf.get('side', 'buy')
                
                if side == 'buy':
                    # Long: MV = Price * Qty
                    # PL = (Price - Entry) * Qty
                    mv = mid * qty
                    pl = (mid - entry_price) * qty
                else:
                    # Short: MV = -Price * Qty
                    # PL = (Entry - Price) * Qty
                    mv = -mid * qty
                    pl = (entry_price - mid) * qty
                
                # Scale by 100 for options
                mv *= 100
                pl *= 100
                
                current_legs.append({
                    'symbol': sym,
                    'qty': qty if side == 'buy' else -qty,
                    'market_value': mv,
                    'unrealized_pl': pl
                })
            
            if len(current_legs) != len(legs_config):
                continue
                
            # Calculate Metrics
            metrics = calculate_group_metrics(current_legs)
            last_metrics = metrics
            
            # Check Rules
            # DTE
            ts_dt = pd.to_datetime(ts)
            dte = (expiry_date - ts_dt).days
            
            action = None
            reason = None
            
            if dte <= args.dte:
                action = "CLOSE"
                reason = f"DTE {dte} <= {args.dte}"
            elif metrics['pl_pct'] >= args.tp:
                action = "CLOSE"
                reason = f"Take Profit: {metrics['pl_pct']:.1%} >= {args.tp:.1%}"
            elif metrics['pl_pct'] <= -args.sl:
                action = "CLOSE"
                reason = f"Stop Loss: {metrics['pl_pct']:.1%} <= -{args.sl:.1%}"
            
            if action == "CLOSE":
                if not args.json:
                    logger.info(f"*** TRIGGERED {reason} ***")
                    logger.info(f"Time: {ts_dt}")
                    logger.info(f"P/L: ${metrics['unrealized_pl']:.2f} ({metrics['pl_pct']:.1%})")
                    logger.info(f"Initial Basis: ${metrics['basis']:.2f}")
                
                # Save result
                result = {
                    "status": "closed",
                    "close_date": str(ts_dt),
                    "reason": reason,
                    "pnl": metrics['unrealized_pl'],
                    "pnl_pct": metrics['pl_pct']
                }
                if args.json:
                    print(json.dumps(result))
                
                return

        current_date += timedelta(days=1)
    
    if args.json:
        pnl = last_metrics['unrealized_pl'] if last_metrics else 0.0
        pnl_pct = last_metrics['pl_pct'] if last_metrics else 0.0
        result = {
            "status": "expired",
            "close_date": str(current_date),
            "reason": "Expiration/End of Data",
            "pnl": pnl,
            "pnl_pct": pnl_pct
        }
        print(json.dumps(result))
    else:
        print("\nReached end of data/expiration without triggering exit.")

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

    defaults = config.get("options", {}).get("management", {})

    parser = argparse.ArgumentParser(description="Manage Option Positions")
    parser.add_argument("--tp", type=float, default=defaults.get("take_profit_pct", 0.50), help="Take Profit %% (e.g. 0.50 for 50%%)")
    parser.add_argument("--sl", type=float, default=defaults.get("stop_loss_pct", 0.50), help="Stop Loss %% (e.g. 0.50 for 50%%)")
    parser.add_argument("--dte", type=int, default=defaults.get("close_dte", 5), help="Close if DTE <= this value")
    parser.add_argument("--symbol", type=str, help="Filter by underlying symbol")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--historical", type=str, help="Path to historical data (mock mode)")
    parser.add_argument("--mock-position", type=str, help="Path to position JSON file (mock mode)")
    parser.add_argument("--start-date", type=str, help="Start date for mock execution (YYYY-MM-DD)")
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/options)")
    parser.add_argument("--log-file", help="Log file name (default: manage_options.log)")
    
    args = parser.parse_args()

    if args.historical and args.mock_position:
        run_mock_loop(args)
        return

    setup_logging(args.log_dir, args.log_file, config, default_dir='trading_logs/options', default_file='manage_options.log')

    client = AlpacaClient()
    
    # Fetch open orders to avoid duplicates/conflicts
    open_orders_symbols = set()
    try:
        orders = client.get_orders(status='open')
        for o in orders:
            if o.get('legs'):
                for leg in o['legs']:
                    open_orders_symbols.add(leg.get('symbol'))
            else:
                open_orders_symbols.add(o.get('symbol'))
    except Exception as e:
        if not args.json:
            print(f"Warning: Could not fetch open orders: {e}", file=sys.stderr)

    try:
        positions = client.get_all_positions()
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error fetching positions: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not positions:
        if args.json:
            print(json.dumps({"message": "No open positions"}))
        else:
            print("No open positions.")
        return

    groups = group_positions(positions)
    actions = []
    
    now = datetime.now()
    
    for (root, expiry), legs in groups.items():
        if args.symbol and root != args.symbol.upper():
            continue
            
        # Check for pending orders (skip if any leg has an open order)
        if not args.dry_run:
            has_pending = any(leg['symbol'] in open_orders_symbols for leg in legs)
            if has_pending:
                if not args.json:
                    print(f"Skipping {root} {expiry} due to pending orders.")
                continue
            
        metrics = calculate_group_metrics(legs)
        
        # Fetch quotes for Bid/Ask context
        quote_info = ""
        try:
            leg_symbols = [leg['symbol'] for leg in legs]
            if leg_symbols:
                snaps = client.get_option_snapshot(','.join(leg_symbols))
                # Handle single result vs dict of dicts
                if 'latestQuote' in snaps: 
                    snaps = {leg_symbols[0]: snaps}
                
                agg_bid = 0.0
                agg_ask = 0.0
                
                for leg in legs:
                    qty = float(leg.get('qty', 0))
                    sym = leg['symbol']
                    snap = snaps.get(sym, {})
                    quote = snap.get('latestQuote', {})
                    l_bid = float(quote.get('bp', 0))
                    l_ask = float(quote.get('ap', 0))
                    
                    if qty > 0:
                        agg_bid += qty * l_bid * 100
                        agg_ask += qty * l_ask * 100
                    else:
                        agg_bid += qty * l_ask * 100
                        agg_ask += qty * l_bid * 100
                
                quote_info = f"Bid=${agg_bid:.2f}, Ask=${agg_ask:.2f}, "
        except Exception:
            pass
        
        if not args.json:
            logger.info(f"{root} {expiry.strftime('%Y-%m-%d')}: {quote_info}Price=${metrics['market_value']:.2f}, P/L=${metrics['unrealized_pl']:.2f} ({metrics['pl_pct']:.1%})")
        
        # DTE Check
        dte = (expiry - now).days
        
        # Logic
        action = None
        reason = None
        
        # 1. DTE Check
        if dte <= args.dte:
            action = "CLOSE"
            reason = f"DTE {dte} <= {args.dte}"
        
        # 2. Take Profit
        # For Credit: P/L >= 50% of Credit
        # For Debit: P/L >= 50% of Cost
        elif metrics['pl_pct'] >= args.tp:
            action = "CLOSE"
            reason = f"Take Profit: {metrics['pl_pct']:.1%} >= {args.tp:.1%}"
            
        # 3. Stop Loss
        # For Credit: P/L <= -100% of Credit (Loss amount > Credit)
        # For Debit: P/L <= -50%? (User defined).
        # Note: pl_pct is positive for profit, negative for loss.
        # If args.sl is 1.0 (100%), we trigger if pl_pct <= -1.0
        elif metrics['pl_pct'] <= -args.sl:
            action = "CLOSE"
            reason = f"Stop Loss: {metrics['pl_pct']:.1%} <= -{args.sl:.1%}"
            
        if action == "CLOSE":
            # Identify symbols to close
            symbols_to_close = [leg['symbol'] for leg in legs]
            
            actions.append({
                "root": root,
                "expiry": expiry.strftime("%Y-%m-%d"),
                "dte": dte,
                "metrics": metrics,
                "reason": reason,
                "symbols": symbols_to_close
            })

    # Execute Actions
    results = []
    for act in actions:
        desc = f"{act['root']} {act['expiry']} (P/L: {act['metrics']['pl_pct']:.1%})"
        
        if args.dry_run:
            results.append({"status": "dry_run", "description": desc, "reason": act['reason']})
            if not args.json:
                logger.info(f"[Dry Run] Would CLOSE {desc}")
                print(f"  Reason: {act['reason']}")
                print(f"  Metrics: Initial ${act['metrics']['net_initial']:.2f}, Current P/L ${act['metrics']['unrealized_pl']:.2f}")
        else:
            # Close each position individually to avoid complex order validation errors
            errors = []
            for symbol in act['symbols']:
                try:
                    client.close_position(symbol)
                    if not args.json:
                        logger.info(f"Submitted CLOSE for {symbol}")
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)}")
                    if not args.json:
                        print(f"Error closing {symbol}: {e}", file=sys.stderr)
            
            if errors:
                results.append({"status": "error", "description": desc, "error": "; ".join(errors)})
            else:
                results.append({"status": "submitted", "description": desc, "reason": act['reason']})
                if not args.json:
                    print(f"  Reason: {act['reason']}")

    if args.json:
        print(json.dumps(results, indent=2))
    elif not results:
        logger.info(f"Scanned {len(positions)} positions ({len(groups)} groups). No adjustments needed.")

if __name__ == "__main__":
    main()
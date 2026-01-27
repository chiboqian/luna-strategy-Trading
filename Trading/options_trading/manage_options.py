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
import yaml
import re
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent / "Trading"))
try:
    from alpaca_client import AlpacaClient
except ImportError:
    print("Error: Could not import AlpacaClient.", file=sys.stderr)
    sys.exit(1)

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
    parser.add_argument("--tp", type=float, default=defaults.get("take_profit_pct", 0.50), help="Take Profit %% (e.g. 0.50 for 50%% of max profit)")
    parser.add_argument("--sl", type=float, default=defaults.get("stop_loss_pct", 1.00), help="Stop Loss %% (e.g. 1.00 for 100%% loss of credit amount)")
    parser.add_argument("--dte", type=int, default=defaults.get("close_dte", 7), help="Close if DTE <= this value")
    parser.add_argument("--symbol", type=str, help="Filter by underlying symbol")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    client = AlpacaClient()
    
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
            
        metrics = calculate_group_metrics(legs)
        
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
            # Construct Closing Order
            close_legs = []
            for leg in legs:
                qty = float(leg['qty'])
                # To close:
                # If Long (qty > 0) -> Sell
                # If Short (qty < 0) -> Buy
                side = 'sell' if qty > 0 else 'buy'
                close_legs.append({
                    "symbol": leg['symbol'],
                    "side": side,
                    "position_intent": "sell_to_close" if side == 'sell' else "buy_to_close",
                    "ratio_qty": 1 # Assuming we close the whole structure 1:1
                })
                
            # Determine quantity (absolute value of leg qty)
            # Assuming balanced spread (all legs have same abs qty)
            # If not, this simple logic might fail for ratio spreads.
            # We use the qty of the first leg.
            close_qty = abs(int(float(legs[0]['qty'])))
            
            actions.append({
                "root": root,
                "expiry": expiry.strftime("%Y-%m-%d"),
                "dte": dte,
                "metrics": metrics,
                "reason": reason,
                "legs": close_legs,
                "qty": close_qty
            })

    # Execute Actions
    results = []
    for act in actions:
        desc = f"{act['root']} {act['expiry']} (P/L: {act['metrics']['pl_pct']:.1%})"
        
        if args.dry_run:
            results.append({"status": "dry_run", "description": desc, "reason": act['reason']})
            if not args.json:
                print(f"[Dry Run] Would CLOSE {desc}")
                print(f"  Reason: {act['reason']}")
                print(f"  Metrics: Initial ${act['metrics']['net_initial']:.2f}, Current P/L ${act['metrics']['unrealized_pl']:.2f}")
        else:
            try:
                # Use multi-leg order if > 1 leg, else simple close
                if len(act['legs']) > 1:
                    client.place_option_market_order(
                        legs=act['legs'],
                        quantity=act['qty'],
                        time_in_force='day',
                        order_class='mleg'
                    )
                else:
                    # Single leg close
                    client.close_position(act['legs'][0]['symbol'])
                    
                results.append({"status": "submitted", "description": desc, "reason": act['reason']})
                if not args.json:
                    print(f"Submitted CLOSE for {desc}")
                    print(f"  Reason: {act['reason']}")
            except Exception as e:
                results.append({"status": "error", "description": desc, "error": str(e)})
                if not args.json:
                    print(f"Error closing {desc}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(results, indent=2))
    elif not results:
        print(f"Scanned {len(positions)} positions ({len(groups)} groups). No adjustments needed.")

if __name__ == "__main__":
    main()
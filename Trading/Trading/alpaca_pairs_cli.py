#!/usr/bin/env python3
"""
CLI to manage pairs trading positions (Short A / Long B).
Implements a market-neutral pairs trade by executing two simultaneous orders.

Usage:
  python alpaca_pairs_cli.py --long QQQ --short SPY --capital 10000 --entry-z 2.0 --exit-z 0.0
"""

import argparse
import json
import os
import logging
import sys
import yaml
from alpaca_client import AlpacaClient
from logging_config import setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlpacaPairsCLI")

def load_config():
    """Load trading defaults from Trading.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'Trading.yaml')
    defaults = {
        'default_feed': 'iex',
    }
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            trading = config.get('trading', {})
            for key in defaults:
                if key in trading:
                    defaults[key] = trading[key]
    except Exception:
        pass
    return defaults

def parse_args():
    config = load_config()
    parser = argparse.ArgumentParser(description="Execute Pairs Trade (Long A / Short B)")
    parser.add_argument("--long", required=False, help="Symbol to LONG (Leg 1)")
    parser.add_argument("--short", required=False, help="Symbol to SHORT (Leg 1) - OR second LONG if type=long-long")
    parser.add_argument("--long2", required=False, help="Symbol to LONG (Leg 2, optional)")
    parser.add_argument("--short2", required=False, help="Symbol to SHORT (Leg 2, optional)")
    parser.add_argument("--capital", type=float, default=10000, help="Total capital to deploy (split 50/50)")
    parser.add_argument("--action", choices=['open', 'close', 'close-all'], default='open', help="Action to take")
    parser.add_argument("--type", choices=['standard', 'long-long'], default='standard', help="Standard (L/S) or Long-Long (L/L) for inverse ETFs")
    parser.add_argument("--feed", default=config['default_feed'], help="Data feed (iex/sip)")
    parser.add_argument("--check-position", action="store_true", help="Skip if any pair symbol already has open position")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    parser.add_argument("--log-dir", help="Log directory")
    parser.add_argument("--log-file", help="Log file")
    return parser.parse_args()

def get_price(client, symbol, feed):
    try:
        quote = client.get_stock_latest_quote(symbol, feed=feed)
        # Use Ask for Long, Bid for Short to be conservative, 
        # but for sizing we use Midpoint
        bid = float(quote.get('bp', 0) or 0)
        ask = float(quote.get('ap', 0) or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        # Fallback to trade
        trade = client.get_stock_latest_trade(symbol, feed=feed)
        return float(trade.get('p', 0) or 0)
    except Exception as e:
        logger.error(f"Failed to get price for {symbol}: {e}")
        return 0.0

def main():
    args = parse_args()
    setup_logging(args.log_dir, args.log_file, default_dir='trading_logs/cli', default_file='pairs_trade.log')
    
    client = AlpacaClient()
    
    if args.action == 'open':
        # Determine symbols based on type
        # In 'standard' mode: long=LONG, short=SHORT
        # In 'long-long' mode: long=LONG, short=LONG (Inverse ETF)
        
        sym1 = args.long.upper()
        sym2 = args.short.upper()
        
        # Check for existing positions if --check-position is set
        if args.check_position:
            # Collect all symbols that would be part of this pair trade
            pair_symbols = {sym1, sym2}
            if args.long2:
                pair_symbols.add(args.long2.upper())
            if args.short2:
                pair_symbols.add(args.short2.upper())
            # Also check inverse ETF counterparts for QQQ/SPY trades
            inverse_map = {'QQQ': 'PSQ', 'SPY': 'SH', 'PSQ': 'QQQ', 'SH': 'SPY'}
            for sym in list(pair_symbols):
                if sym in inverse_map:
                    pair_symbols.add(inverse_map[sym])
            
            try:
                positions = client.get_all_positions() or []
                open_syms = {p.get('symbol', '').upper() for p in positions}
                overlap = pair_symbols & open_syms
                if overlap:
                    logger.warning(f"⚠️ SKIPPING: Already have open position(s) in: {overlap}")
                    sys.exit(0)
            except Exception as e:
                logger.error(f"Failed to check positions: {e}")
                # Continue anyway - better to trade than miss opportunity
        
        type_desc = "Long/Short" if args.type == 'standard' else "Long/Long (Inverse ETF)"
        logger.info(f"🔵 OPENING PAIR [{type_desc}]: {sym1} / {sym2} | Capital: ${args.capital}")
        
        # 1. Get Prices
        price1 = get_price(client, sym1, args.feed)
        price2 = get_price(client, sym2, args.feed)
        
        if price1 <= 0 or price2 <= 0:
            logger.error("❌ Invalid prices. Aborting.")
            sys.exit(1)
            
        # 2. Calculate Size (Neutral Dollar Amount)
        allocation = args.capital / 2
        qty1 = int(allocation / price1)
        qty2 = int(allocation / price2)
        
        logger.info(f"   {sym1}: ${price1:.2f} x {qty1} shares = ${price1 * qty1:.2f}")
        logger.info(f"   {sym2}: ${price2:.2f} x {qty2} shares = ${price2 * qty2:.2f}")
        
        if args.dry_run:
            logger.info("🔍 Dry run completed.")
            sys.exit(0)
            
        # 3. Execute Orders
        try:
            # Leg 1 is always BUY
            client.place_stock_order(symbol=sym1, side='buy', quantity=qty1, order_type='market', time_in_force='day')
            logger.info(f"✅ Leg 1 Placed (BUY {sym1})")
            
            # Leg 2 depends on type
            side2 = 'sell' if args.type == 'standard' else 'buy'
            client.place_stock_order(symbol=sym2, side=side2, quantity=qty2, order_type='market', time_in_force='day')
            logger.info(f"✅ Leg 2 Placed ({side2.upper()} {sym2})")
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            sys.exit(1)

    elif args.action in ['close', 'close-all']:
        symbols_to_close = []
        if args.long: symbols_to_close.append(args.long.upper())
        if args.short: symbols_to_close.append(args.short.upper())
        if args.long2: symbols_to_close.append(args.long2.upper())
        if args.short2: symbols_to_close.append(args.short2.upper())
        
        logger.info(f"🔴 CLOSING POSITIONS: {symbols_to_close}")
        
        if args.dry_run:
            logger.info("🔍 Dry run closing.")
            sys.exit(0)
            
        for sym in symbols_to_close:
            try:
                client.close_position(sym)
                logger.info(f"✅ Closed {sym}")
            except Exception as e:
                logger.error(f"Failed to close {sym}: {e}")

if __name__ == "__main__":
    main()

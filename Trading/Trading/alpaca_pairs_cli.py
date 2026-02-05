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
    parser.add_argument("--long", required=True, help="Symbol to LONG")
    parser.add_argument("--short", required=True, help="Symbol to SHORT")
    parser.add_argument("--capital", type=float, default=10000, help="Total capital to deploy (split 50/50)")
    parser.add_argument("--action", choices=['open', 'close'], default='open', help="Action to take")
    parser.add_argument("--feed", default=config['default_feed'], help="Data feed (iex/sip)")
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
    
    long_sym = args.long.upper()
    short_sym = args.short.upper()
    
    if args.action == 'open':
        logger.info(f"🔵 OPENING PAIR: Long {long_sym} / Short {short_sym} | Capital: ${args.capital}")
        
        # 1. Get Prices
        price_long = get_price(client, long_sym, args.feed)
        price_short = get_price(client, short_sym, args.feed)
        
        if price_long <= 0 or price_short <= 0:
            logger.error("❌ Invalid prices. Aborting.")
            sys.exit(1)
            
        # 2. Calculate Size (Neutral Dollar Amount)
        allocation = args.capital / 2
        qty_long = int(allocation / price_long)
        qty_short = int(allocation / price_short)
        
        logger.info(f"   {long_sym}: ${price_long:.2f} x {qty_long} shares = ${price_long * qty_long:.2f}")
        logger.info(f"   {short_sym}: ${price_short:.2f} x {qty_short} shares = ${price_short * qty_short:.2f}")
        
        if args.dry_run:
            logger.info("🔍 Dry run completed.")
            sys.exit(0)
            
        # 3. Execute Orders (Market for speed in this context, or limit at ask/bid)
        # Using Market for guaranteed execution of both legs simultaneously
        try:
            # Check for existing positions first to avoid doubling up? 
            # (Skipped for speed, logic should handle this upstream)

            order_long = client.place_stock_order(
                symbol=long_sym, side='buy', quantity=qty_long, 
                order_type='market', time_in_force='day'
            )
            logger.info(f"✅ Long Leg Placed: {order_long.get('id')}")
            
            order_short = client.place_stock_order(
                symbol=short_sym, side='sell', quantity=qty_short, 
                order_type='market', time_in_force='day'
            )
            logger.info(f"✅ Short Leg Placed: {order_short.get('id')}")
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            sys.exit(1)

    elif args.action == 'close':
        logger.info(f"🔴 CLOSING PAIR: {long_sym}/{short_sym}")
        
        # Close all positions for these symbols
        if args.dry_run:
            logger.info("🔍 Dry run closing.")
            sys.exit(0)
            
        try:
            client.close_position(long_sym)
            logger.info(f"✅ Closed {long_sym}")
        except Exception as e:
            logger.error(f"Failed to close {long_sym}: {e}")

        try:
            client.close_position(short_sym)
            logger.info(f"✅ Closed {short_sym}")
        except Exception as e:
            logger.error(f"Failed to close {short_sym}: {e}")

if __name__ == "__main__":
    main()

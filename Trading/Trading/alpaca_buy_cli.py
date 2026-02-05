#!/usr/bin/env python3
"""
CLI to buy a notional dollar amount of a single stock via Alpaca.

Rules:
- Only buy if there is no existing position in the symbol.
- Only buy if account has enough cash to cover the notional.
- Default order is limit at current bid; support price offset (e.g. +$0.01).
- Option to use ask price instead with --use-ask.
- Option to use midpoint of bid and ask with --mid-price.
- Option to use market order instead with --market.
- Attach stop loss and take profit using bracket order.
- Default stop loss: 5%, default take profit: 10% (customizable).
"""

import argparse
import json
import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from alpaca_client import AlpacaClient
from logging_config import setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlpacaBuyCLI")


def load_config():
    """Load trading defaults from Trading.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'Trading.yaml')
    defaults = {
        'default_dollars': 10000,
        'default_stop_loss_pct': 5.0,
        'default_take_profit_pct': 10.0,
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
        pass  # Use hardcoded defaults if config not found
    return defaults


def parse_args():
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Buy a notional dollar amount of a stock (Alpaca) with stop-loss and take-profit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("symbol", help="Stock symbol to buy (e.g., AAPL)")
    parser.add_argument(
        "dollars",
        type=float,
        nargs='?',
        default=config['default_dollars'],
        help="Dollar amount to buy (notional)"
    )
    parser.add_argument(
        "--market",
        action="store_true",
        help="Use market order instead of limit"
    )
    parser.add_argument(
        "--use-ask",
        action="store_true",
        help="Use ask price instead of bid (default is bid)"
    )
    parser.add_argument(
        "--mid-price",
        action="store_true",
        help="Use midpoint of bid and ask price"
    )
    parser.add_argument(
        "--price-offset",
        type=float,
        default=0.0,
        help="Additive offset to ask price for limit orders (e.g., 0.01 means $0.01 above ask)"
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=config['default_stop_loss_pct'],
        help="Stop loss percent below entry price"
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=config['default_take_profit_pct'],
        help="Take profit percent above entry price"
    )
    parser.add_argument(
        "--feed",
        type=str,
        default=config['default_feed'],
        help="Market data feed to use for quotes (iex, sip, otc)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print diagnostic information"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Output human-readable text instead of JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate order without executing"
    )
    parser.add_argument(
        "--exclude-symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols to check - skip if any have open position (e.g., SQQQ,TQQQ)"
    )
    parser.add_argument("--log-dir", help="Directory for log files (default: trading_logs/cli)")
    parser.add_argument("--log-file", help="Log file name (default: alpaca_buy.log)")
    return parser.parse_args()


def get_best_price(client: AlpacaClient, symbol: str, feed: str, verbose: bool, use_ask: bool = False, use_mid: bool = False) -> tuple:
    """
    Acquire the best available price for order calculation.
    If use_mid=True: Try to use (bid + ask) / 2.
    If use_ask=True: Fallback sequence: ask -> snapshot ask -> last trade -> bid -> daily bar close.
    If use_ask=False (default): Fallback sequence: bid -> snapshot bid -> last trade -> ask -> daily bar close.
    
    Returns:
        tuple: (ref_price, bid, ask) - reference price used, current bid, current ask
    """
    ask = 0.0
    bid = 0.0
    last_trade_price = 0.0
    
    # Try latest quote
    try:
        raw_quote = client.get_stock_latest_quote(symbol, feed=feed)
        ask = float(raw_quote.get('ap', 0) or 0)
        bid = float(raw_quote.get('bp', 0) or 0)
        if verbose:
            print(f"📊 Raw quote: bid=${bid:.4f}, ask=${ask:.4f}")
    except Exception as e:
        if verbose:
            print(f"⚠️ Quote fetch failed: {e}")
    
    if use_mid and ask > 0 and bid > 0:
        mid = (ask + bid) / 2.0
        if verbose:
            print(f"📊 Using mid price: ${mid:.4f}")
        return mid, bid, ask

    # Primary price based on use_ask flag
    primary = ask if use_ask else bid
    secondary = bid if use_ask else ask
    primary_label = "ask" if use_ask else "bid"
    secondary_label = "bid" if use_ask else "ask"
    
    if primary > 0:
        return primary, bid, ask
    
    # Fallback to snapshot
    try:
        snapshot = client.get_stock_snapshot(symbol, feed=feed)
        latest_quote = snapshot.get('latestQuote', {}) if isinstance(snapshot, dict) else {}
        snap_primary = float(latest_quote.get('ap' if use_ask else 'bp', 0) or 0)
        snap_bid = float(latest_quote.get('bp', 0) or 0)
        snap_ask = float(latest_quote.get('ap', 0) or 0)
        if snap_primary > 0:
            if verbose:
                print(f"📊 Using snapshot {primary_label}: ${snap_primary:.4f}")
            return snap_primary, snap_bid or bid, snap_ask or ask
    except Exception as e:
        if verbose:
            print(f"⚠️ Snapshot fallback failed: {e}")
    
    # Fallback to last trade
    try:
        last_trade = client.get_stock_latest_trade(symbol, feed=feed)
        last_trade_price = float(last_trade.get('p', 0) or 0)
        if last_trade_price > 0:
            if verbose:
                print(f"📊 Using last trade: ${last_trade_price:.4f}")
            return last_trade_price, bid, ask
    except Exception as e:
        if verbose:
            print(f"⚠️ Last trade fetch failed: {e}")
    
    # Fallback to secondary price
    if secondary > 0:
        if verbose:
            print(f"📊 Using {secondary_label} as proxy: ${secondary:.4f}")
        return secondary, bid, ask
    
    # Fallback to daily bar close
    try:
        bar = client.get_stock_latest_bar(symbol, feed=feed)
        close_price = float(bar.get('c', 0) or 0)
        if close_price > 0:
            if verbose:
                print(f"📊 Using daily bar close: ${close_price:.4f}")
            return close_price, bid, ask
    except Exception as e:
        if verbose:
            print(f"⚠️ Daily bar fallback failed: {e}")
    
    return 0.0, bid, ask


def output_error(message: str, use_text: bool, code: int = 1):
    """Output error message in JSON or text format and exit."""
    if use_text:
        logger.error(f"❌ {message}")
    else:
        print(json.dumps({"success": False, "error": message}))
    sys.exit(code)


def output_info(message: str, use_text: bool):
    """Output info message (only in text mode)."""
    if use_text:
        logger.info(message)


def main():
    args = parse_args()

    symbol = args.symbol.upper()
    notional = float(args.dollars)
    use_market = bool(args.market)
    price_offset = float(args.price_offset)
    stop_loss_pct = float(args.stop_loss_pct)
    take_profit_pct = float(args.take_profit_pct)
    use_text = args.text

    setup_logging(args.log_dir, args.log_file, default_dir='trading_logs/cli', default_file='alpaca_buy.log')

    # Validate inputs
    if notional <= 0:
        output_error("Dollars must be positive.", use_text, 2)
    if stop_loss_pct <= 0 or stop_loss_pct >= 100:
        output_error("stop-loss-pct must be between 0 and 100.", use_text, 2)
    if take_profit_pct <= 0 or take_profit_pct >= 1000:
        output_error("take-profit-pct must be between 0 and 1000.", use_text, 2)

    # Initialize client
    try:
        client = AlpacaClient()
    except Exception as e:
        output_error(f"Failed to initialize Alpaca client: {e}", use_text)

    # Check existing position
    try:
        position = None
        try:
            position = client.get_open_position(symbol)
        except Exception:
            # 404 or no position - proceed
            position = None

        if position and position.get('symbol') == symbol:
            if use_text:
                logger.info(f"ℹ️ Position for {symbol} already exists; not placing buy order.")
            else:
                print(json.dumps({"success": False, "error": f"Position for {symbol} already exists"}))
            sys.exit(0)
    except Exception as e:
        output_error(f"Error checking existing position: {e}", use_text)

    # Check for conflicting positions (mutual exclusion)
    if args.exclude_symbols:
        exclude_list = [s.strip().upper() for s in args.exclude_symbols.split(',') if s.strip()]
        if exclude_list:
            try:
                positions = client.get_all_positions() or []
                open_syms = {p.get('symbol', '').upper() for p in positions}
                conflicts = set(exclude_list) & open_syms
                if conflicts:
                    if use_text:
                        logger.info(f"⚠️ Skipping {symbol}: conflicting position(s) in {conflicts}")
                    else:
                        print(json.dumps({"success": False, "error": f"Conflicting position in {conflicts}"}))
                    sys.exit(0)
            except Exception as e:
                output_info(f"⚠️ Error checking exclude symbols: {e}", use_text)
                # Continue anyway - better to trade than miss opportunity

    # Cancel any existing open orders for this symbol
    canceled_orders = []
    try:
        canceled = client.cancel_orders_for_symbol(symbol)
        if canceled:
            canceled_orders = canceled
            success_count = sum(1 for c in canceled if c.get('status') == 'canceled')
            output_info(f"🔄 Canceled {success_count} existing order(s) for {symbol}", use_text)
            if args.verbose and use_text:
                for c in canceled:
                    logger.info(f"   Order {c['id']}: {c['status']}")
    except Exception as e:
        output_info(f"⚠️ Error canceling existing orders: {e}", use_text)
        # Continue anyway - not fatal

    # Check account cash
    try:
        account = client.get_account_info()
        cash = float(account.get('cash', 0) or 0)
        if cash < notional:
            output_error(f"Insufficient cash. Available: ${cash:,.2f}, required: ${notional:,.2f}", use_text, 2)
    except Exception as e:
        output_error(f"Error fetching account info: {e}", use_text)

    # Get current price
    use_ask = args.use_ask
    use_mid = args.mid_price
    if use_mid:
        price_type = "mid"
    else:
        price_type = "ask" if use_ask else "bid"
    
    ref_price, bid, ask = get_best_price(client, symbol, args.feed, args.verbose and use_text, use_ask, use_mid)
    if ref_price <= 0:
        output_error("Unable to determine stock price (quote/trade/bar unavailable).", use_text)

    # Determine entry price and order type (round to nearest penny)
    if use_market:
        entry_price = round(ref_price, 2)  # Approximate for calculations
        order_type = "market"
        limit_price = None
    else:
        entry_price = round(ref_price + price_offset, 2)
        order_type = "limit"
        limit_price = entry_price

    # Calculate quantity (whole shares only, round down)
    qty = int(notional / entry_price)
    if qty < 1:
        output_error(f"Not enough dollars to buy at least 1 share at ${entry_price:.2f}.", use_text, 2)
    actual_cost = qty * entry_price

    # Calculate stop loss and take profit prices (round to nearest penny)
    stop_loss_price = round(entry_price * (1.0 - (stop_loss_pct / 100.0)), 2)
    take_profit_price = round(entry_price * (1.0 + (take_profit_pct / 100.0)), 2)

    # Display order summary (text mode only)
    if use_text:
        logger.info(f"📈 Preparing order: {symbol} for ${notional:,.2f}")
        logger.info(f"   Current quote: bid ${bid:.2f} / ask ${ask:.2f}")
        logger.info(f"   Order type: {order_type}")
        if limit_price is not None:
            logger.info(f"   Limit price: ${limit_price:.2f} ({price_type} ${ref_price:.2f} + offset ${price_offset:.2f})")
        else:
            logger.info(f"   Reference price: ${ref_price:.2f} ({price_type})")
        logger.info(f"   Quantity: {qty} shares (actual cost: ${actual_cost:.2f})")
        logger.info(f"   Stop loss: {stop_loss_pct:.2f}% -> ${stop_loss_price:.2f}")
        logger.info(f"   Take profit: {take_profit_pct:.2f}% -> ${take_profit_price:.2f}")

    # Place bracket order with stop loss and take profit
    try:
        if args.dry_run:
            if use_text:
                logger.info(f"🔍 Dry run: Order would be placed successfully!")
                logger.info(f"   Status: simulated")
            else:
                result = {
                    "success": True,
                    "status": "simulated",
                    "order": {
                        "id": "simulated-id",
                        "status": "simulated",
                        "symbol": symbol,
                        "side": "buy",
                        "order_type": order_type,
                        "quantity": qty,
                        "limit_price": limit_price,
                        "entry_price": entry_price,
                        "actual_cost": actual_cost,
                        "stop_loss_price": stop_loss_price,
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_price": take_profit_price,
                        "take_profit_pct": take_profit_pct,
                    },
                    "quote": {
                        "bid": bid,
                        "ask": ask,
                        "ref_price": ref_price,
                        "price_type": price_type,
                    },
                    "canceled_orders": canceled_orders,
                }
                print(json.dumps(result, indent=2))
            sys.exit(0)

        order = client.place_stock_order(
            symbol=symbol,
            side="buy",
            quantity=qty,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force="day",
            extended_hours=False,
            order_class="bracket",
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
        order_id = order.get('id')
        status = order.get('status')
        
        if use_text:
            logger.info(f"✅ Order placed successfully!")
            logger.info(f"   Order ID: {order_id}")
            logger.info(f"   Status: {status}")
            logger.info(f"   Purchased Share Price: ${entry_price:.2f}")
        else:
            result = {
                "success": True,
                "order": {
                    "id": order_id,
                    "status": status,
                    "symbol": symbol,
                    "side": "buy",
                    "order_type": order_type,
                    "quantity": qty,
                    "limit_price": limit_price,
                    "entry_price": entry_price,
                    "actual_cost": actual_cost,
                    "stop_loss_price": stop_loss_price,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_price": take_profit_price,
                    "take_profit_pct": take_profit_pct,
                },
                "quote": {
                    "bid": bid,
                    "ask": ask,
                    "ref_price": ref_price,
                    "price_type": price_type,
                },
                "canceled_orders": canceled_orders,
            }
            print(json.dumps(result, indent=2))
    except Exception as e:
        output_error(f"Failed to place order: {e}", use_text)


if __name__ == "__main__":
    main()

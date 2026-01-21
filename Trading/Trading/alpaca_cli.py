#!/usr/bin/env python3
"""
Command Line Interface for Alpaca Trading Client
Provides easy access to all Alpaca API functions
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from typing import Optional
from alpaca_client import AlpacaClient, Sort


def format_currency(value):
    """Format value as currency"""
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value):
    """Format value as percentage"""
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def print_table(data, headers):
    """Print data in a simple table format"""
    if not data:
        print("No data to display")
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in data:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def cmd_account(client, args):
    """Get account information"""
    account = client.get_account_info()
    
    print("\n=== ACCOUNT INFORMATION ===")
    print(f"Status: {account.get('status')}")
    print(f"Account Number: {account.get('account_number')}")
    print(f"Currency: {account.get('currency')}")
    print(f"\n--- Balances ---")
    print(f"Cash: {format_currency(account.get('cash'))}")
    print(f"Portfolio Value: {format_currency(account.get('portfolio_value'))}")
    print(f"Equity: {format_currency(account.get('equity'))}")
    print(f"Buying Power: {format_currency(account.get('buying_power'))}")
    print(f"\n--- Margin Info ---")
    print(f"Multiplier: {account.get('multiplier')}")
    print(f"Initial Margin: {format_currency(account.get('initial_margin'))}")
    print(f"Maintenance Margin: {format_currency(account.get('maintenance_margin'))}")
    print(f"\n--- Day Trading ---")
    print(f"Pattern Day Trader: {account.get('pattern_day_trader')}")
    print(f"Day Trade Count: {account.get('daytrade_count')}")
    print(f"Daytrading Buying Power: {format_currency(account.get('daytrading_buying_power'))}")


def cmd_positions(client, args):
    """List all positions or get specific position"""
    if args.symbol:
        # Get specific position
        try:
            pos = client.get_open_position(args.symbol)
            print(f"\n=== POSITION: {pos.get('symbol')} ===")
            print(f"Quantity: {pos.get('qty')}")
            print(f"Side: {pos.get('side')}")
            print(f"Avg Entry Price: {format_currency(pos.get('avg_entry_price'))}")
            print(f"Current Price: {format_currency(pos.get('current_price'))}")
            print(f"Market Value: {format_currency(pos.get('market_value'))}")
            print(f"Cost Basis: {format_currency(pos.get('cost_basis'))}")
            print(f"Unrealized P/L: {format_currency(pos.get('unrealized_pl'))} ({format_percentage(pos.get('unrealized_plpc'))})")
            print(f"Intraday P/L: {format_currency(pos.get('unrealized_intraday_pl'))}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Get all positions
        positions = client.get_all_positions()
        
        if not positions:
            print("No open positions")
            return
        
        print(f"\n=== ALL POSITIONS ({len(positions)}) ===\n")
        
        data = []
        for pos in positions:
            data.append([
                pos.get('symbol'),
                pos.get('qty'),
                format_currency(pos.get('avg_entry_price')),
                format_currency(pos.get('current_price')),
                format_currency(pos.get('market_value')),
                format_currency(pos.get('unrealized_pl')),
                format_percentage(pos.get('unrealized_plpc'))
            ])
        
        headers = ["Symbol", "Qty", "Avg Price", "Current", "Market Value", "P/L", "P/L %"]
        print_table(data, headers)


def cmd_orders(client, args):
    """List orders"""
    orders = client.get_orders(
        status=args.status,
        limit=args.limit,
        symbols=args.symbols.split(',') if args.symbols else None
    )
    
    if not orders:
        print("No orders found")
        return
    
    print(f"\n=== ORDERS ({len(orders)}) ===\n")
    
    # Get unique symbols to fetch quotes
    symbols = list(set(order.get('symbol') for order in orders if order.get('symbol')))
    quotes = {}
    if symbols:
        try:
            if len(symbols) == 1:
                quote = client.get_stock_latest_quote(symbols[0])
                quotes[symbols[0]] = quote
            else:
                quotes = client.get_stock_latest_quote(symbols)
        except Exception:
            pass  # Continue without quotes if fetch fails
    
    data = []
    for order in orders:
        symbol = order.get('symbol')
        quote = quotes.get(symbol, {})
        bid = format_currency(quote.get('bp')) if quote.get('bp') else '-'
        ask = format_currency(quote.get('ap')) if quote.get('ap') else '-'
        
        # Get order price (limit_price or stop_price depending on order type)
        order_price = order.get('limit_price') or order.get('stop_price')
        order_price_str = format_currency(order_price) if order_price else '-'
        
        data.append([
            symbol,
            order.get('side'),
            order.get('qty'),
            order.get('type'),
            order_price_str,
            bid,
            ask,
            order.get('status'),
            f"{order.get('filled_qty', 0)}/{order.get('qty')}",
            format_currency(order.get('filled_avg_price', 0)) if order.get('filled_avg_price') else '-'
        ])
    
    headers = ["Symbol", "Side", "Qty", "Type", "Order Price", "Bid", "Ask", "Status", "Filled", "Fill Price"]
    print_table(data, headers)


def cmd_place_order(client, args):
    """Place a stock order"""
    try:
        order = client.place_stock_order(
            symbol=args.symbol,
            side=args.side,
            quantity=args.quantity,
            order_type=args.type,
            limit_price=args.limit_price,
            stop_price=args.stop_price,
            time_in_force=args.tif,
            extended_hours=args.extended_hours
        )
        
        print(f"\n=== ORDER PLACED ===")
        print(f"Order ID: {order.get('id')}")
        print(f"Symbol: {order.get('symbol')}")
        print(f"Side: {order.get('side')}")
        print(f"Quantity: {order.get('qty')}")
        print(f"Type: {order.get('type')}")
        print(f"Status: {order.get('status')}")
        print(f"Time in Force: {order.get('time_in_force')}")
        
        if order.get('limit_price'):
            print(f"Limit Price: {format_currency(order.get('limit_price'))}")
        if order.get('stop_price'):
            print(f"Stop Price: {format_currency(order.get('stop_price'))}")
            
    except Exception as e:
        print(f"Error placing order: {e}")


def cmd_cancel_order(client, args):
    """Cancel order(s)"""
    if args.order_id:
        try:
            result = client.cancel_order_by_id(args.order_id)
            print(f"Order {args.order_id} canceled successfully")
        except Exception as e:
            print(f"Error canceling order: {e}")
    elif args.all:
        try:
            results = client.cancel_all_orders()
            print(f"Canceled {len(results)} orders")
        except Exception as e:
            print(f"Error canceling orders: {e}")
    else:
        print("Error: Must specify --order-id or --all")


def cmd_close_position(client, args):
    """Close position(s)"""
    if args.symbol:
        try:
            # Get current quote before closing
            try:
                quote = client.get_stock_latest_quote(args.symbol)
                bid = format_currency(quote.get('bp')) if quote.get('bp') else '-'
                ask = format_currency(quote.get('ap')) if quote.get('ap') else '-'
            except Exception:
                bid, ask = '-', '-'
            
            result = client.close_position(
                symbol=args.symbol,
                qty=args.quantity,
                percentage=args.percentage
            )
            
            print(f"\n=== POSITION CLOSED: {args.symbol} ===")
            print(f"Order ID: {result.get('id')}")
            print(f"Symbol: {result.get('symbol')}")
            print(f"Side: {result.get('side')}")
            print(f"Quantity: {result.get('qty')}")
            print(f"Type: {result.get('type')}")
            print(f"Status: {result.get('status')}")
            print(f"Time in Force: {result.get('time_in_force')}")
            if result.get('limit_price'):
                print(f"Limit Price: {format_currency(result.get('limit_price'))}")
            if result.get('filled_avg_price'):
                print(f"Fill Price: {format_currency(result.get('filled_avg_price'))}")
            print(f"Current Bid/Ask: {bid} / {ask}")
        except Exception as e:
            print(f"Error closing position: {e}")
    elif args.all:
        try:
            results = client.close_all_positions(cancel_orders=args.cancel_orders)
            print(f"Closed {len(results)} positions")
        except Exception as e:
            print(f"Error closing positions: {e}")
    else:
        print("Error: Must specify --symbol or --all")


def cmd_bars(client, args):
    """Get historical bars"""
    bars = client.get_stock_bars(
        symbol=args.symbol,
        days=args.days,
        timeframe=args.timeframe,
        limit=args.limit,
        start=args.start,
        end=args.end
    )
    
    if not bars:
        print("No bar data found")
        return
    
    print(f"\n=== BARS: {args.symbol} ({args.timeframe}) ===\n")
    
    data = []
    for bar in bars[-args.limit:]:
        timestamp = bar.get('t', '')
        data.append([
            timestamp,
            format_currency(bar.get('o')),
            format_currency(bar.get('h')),
            format_currency(bar.get('l')),
            format_currency(bar.get('c')),
            f"{bar.get('v', 0):,}"
        ])
    
    headers = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    print_table(data, headers)


def cmd_quote(client, args):
    """Get latest quote"""
    symbols = args.symbols.split(',')
    
    if len(symbols) == 1:
        quote = client.get_stock_latest_quote(symbols[0])
        print(f"\n=== QUOTE: {symbols[0]} ===")
        print(f"Bid: {format_currency(quote.get('bp'))} x {quote.get('bs')}")
        print(f"Ask: {format_currency(quote.get('ap'))} x {quote.get('as')}")
        spread = float(quote.get('ap', 0)) - float(quote.get('bp', 0))
        print(f"Spread: {format_currency(spread)}")
    else:
        quotes = client.get_stock_latest_quote(symbols)
        print(f"\n=== QUOTES ({len(quotes)}) ===\n")
        
        data = []
        for symbol, quote in quotes.items():
            data.append([
                symbol,
                format_currency(quote.get('bp')),
                quote.get('bs'),
                format_currency(quote.get('ap')),
                quote.get('as')
            ])
        
        headers = ["Symbol", "Bid", "Bid Size", "Ask", "Ask Size"]
        print_table(data, headers)


def cmd_snapshot(client, args):
    """Get stock snapshot"""
    symbols = args.symbols.split(',')
    
    if len(symbols) == 1:
        snapshot = client.get_stock_snapshot(symbols[0])
        print(f"\n=== SNAPSHOT: {symbols[0]} ===")
        
        latest = snapshot.get('latestTrade', {})
        print(f"\nLast Trade: {format_currency(latest.get('p'))} ({latest.get('s')} shares)")
        
        quote = snapshot.get('latestQuote', {})
        print(f"Bid/Ask: {format_currency(quote.get('bp'))} / {format_currency(quote.get('ap'))}")
        
        daily = snapshot.get('dailyBar', {})
        if daily:
            print(f"\nDaily Bar:")
            print(f"  Open: {format_currency(daily.get('o'))}")
            print(f"  High: {format_currency(daily.get('h'))}")
            print(f"  Low: {format_currency(daily.get('l'))}")
            print(f"  Close: {format_currency(daily.get('c'))}")
            print(f"  Volume: {daily.get('v', 0):,}")
        
        prev = snapshot.get('prevDailyBar', {})
        if prev and latest:
            prev_close = float(prev.get('c', 0))
            current = float(latest.get('p', 0))
            if prev_close > 0:
                change = ((current - prev_close) / prev_close) * 100
                print(f"\nChange: {format_currency(current - prev_close)} ({change:+.2f}%)")
    else:
        snapshots = client.get_stock_snapshot(symbols)
        print(f"\n=== SNAPSHOTS ({len(snapshots)}) ===\n")
        
        data = []
        for symbol, snapshot in snapshots.items():
            latest = snapshot.get('latestTrade', {})
            prev = snapshot.get('prevDailyBar', {})
            
            price = latest.get('p', 0)
            prev_close = prev.get('c', 0)
            change = 0
            if prev_close:
                change = ((float(price) - float(prev_close)) / float(prev_close)) * 100
            
            data.append([
                symbol,
                format_currency(price),
                format_currency(prev_close),
                f"{change:+.2f}%"
            ])
        
        headers = ["Symbol", "Last", "Prev Close", "Change %"]
        print_table(data, headers)


def cmd_assets(client, args):
    """Get asset information"""
    if args.symbol:
        asset = client.get_asset(args.symbol)
        print(f"\n=== ASSET: {asset.get('symbol')} ===")
        print(f"Name: {asset.get('name')}")
        print(f"Exchange: {asset.get('exchange')}")
        print(f"Class: {asset.get('class')}")
        print(f"Status: {asset.get('status')}")
        print(f"Tradable: {asset.get('tradable')}")
        print(f"Marginable: {asset.get('marginable')}")
        print(f"Shortable: {asset.get('shortable')}")
        print(f"Fractionable: {asset.get('fractionable')}")
    else:
        assets = client.get_all_assets(
            status=args.status,
            asset_class=args.asset_class,
            exchange=args.exchange
        )
        
        print(f"\n=== ASSETS ({len(assets)}) ===")
        
        if args.limit and len(assets) > args.limit:
            print(f"Showing first {args.limit} of {len(assets)} assets\n")
            assets = assets[:args.limit]
        else:
            print()
        
        data = []
        for asset in assets:
            data.append([
                asset.get('symbol'),
                asset.get('name', '')[:30],
                asset.get('exchange'),
                asset.get('class'),
                'Yes' if asset.get('tradable') else 'No'
            ])
        
        headers = ["Symbol", "Name", "Exchange", "Class", "Tradable"]
        print_table(data, headers)


def cmd_portfolio_history(client, args):
    """Get portfolio history"""
    history = client.get_portfolio_history(
        period=args.period,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end
    )
    
    print(f"\n=== PORTFOLIO HISTORY ===")
    print(f"Base Value: {format_currency(history.get('base_value'))}")
    print(f"Timeframe: {history.get('timeframe')}")
    print()
    
    timestamps = history.get('timestamp', [])
    equity_values = history.get('equity', [])
    pnl_pct = history.get('profit_loss_pct', [])
    
    # Show requested number of data points
    limit = args.limit if args.limit else len(timestamps)
    start_idx = max(0, len(timestamps) - limit)
    
    data = []
    for i in range(start_idx, len(timestamps)):
        date = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M')
        equity = equity_values[i]
        pnl = pnl_pct[i] * 100 if i < len(pnl_pct) and pnl_pct[i] else 0
        
        data.append([
            date,
            format_currency(equity),
            f"{pnl:+.2f}%"
        ])
    
    headers = ["Date", "Equity", "P/L %"]
    print_table(data, headers)


def cmd_corporate_actions(client, args):
    """Get corporate actions"""
    actions = client.get_corporate_actions(
        ca_types=args.types,
        symbols=args.symbols.split(',') if args.symbols else None,
        start=args.start,
        end=args.end,
        limit=args.limit
    )
    
    if not actions:
        print("No corporate actions found")
        return
    
    print(f"\n=== CORPORATE ACTIONS ({len(actions)}) ===\n")
    
    data = []
    for action in actions:
        ca_type = action.get('corporate_action_type', '')
        symbol = action.get('initiating_symbol', '')
        ex_date = action.get('ex_date', '')
        
        # Format type-specific info
        info = ''
        if ca_type == 'dividend':
            info = format_currency(action.get('cash', 0))
        elif ca_type == 'split':
            info = f"{action.get('old_rate')}:{action.get('new_rate')}"
        
        data.append([
            symbol,
            ca_type.title(),
            ex_date,
            info
        ])
    
    headers = ["Symbol", "Type", "Ex-Date", "Details"]
    print_table(data, headers)


def cmd_options(client, args):
    """Get option contracts"""
    contracts = client.get_option_contracts(
        underlying_symbol=args.symbol,
        expiration_date_gte=args.exp_gte,
        expiration_date_lte=args.exp_lte,
        strike_price_gte=args.strike_gte,
        strike_price_lte=args.strike_lte,
        type=args.option_type,
        status=args.status,
        limit=args.limit
    )
    
    if not contracts:
        print("No option contracts found")
        return
    
    print(f"\n=== OPTION CONTRACTS ({len(contracts)}) ===\n")
    
    data = []
    for contract in contracts:
        data.append([
            contract.get('symbol'),
            contract.get('type', '').upper(),
            format_currency(contract.get('strike_price')),
            contract.get('expiration_date'),
            'Yes' if contract.get('tradable') else 'No'
        ])
    
    headers = ["Symbol", "Type", "Strike", "Expiration", "Tradable"]
    print_table(data, headers)


def main():
    parser = argparse.ArgumentParser(
        description="Alpaca Trading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Account command
    parser_account = subparsers.add_parser('account', help='Get account information')
    parser_account.set_defaults(func=cmd_account)
    
    # Positions command
    parser_positions = subparsers.add_parser('positions', help='List positions')
    parser_positions.add_argument('-s', '--symbol', help='Specific symbol')
    parser_positions.set_defaults(func=cmd_positions)
    
    # Orders command
    parser_orders = subparsers.add_parser('orders', help='List orders')
    parser_orders.add_argument('--status', default='all', help='Order status (open, closed, all)')
    parser_orders.add_argument('--limit', type=int, default=50, help='Max results')
    parser_orders.add_argument('--symbols', help='Comma-separated symbols')
    parser_orders.set_defaults(func=cmd_orders)
    
    # Place order command
    parser_place = subparsers.add_parser('buy', help='Place buy order')
    parser_place.add_argument('symbol', help='Stock symbol')
    parser_place.add_argument('quantity', type=float, help='Quantity')
    parser_place.add_argument('--type', default='market', help='Order type (market, limit, stop)')
    parser_place.add_argument('--limit-price', type=float, help='Limit price')
    parser_place.add_argument('--stop-price', type=float, help='Stop price')
    parser_place.add_argument('--tif', default='day', help='Time in force (day, gtc, ioc, fok)')
    parser_place.add_argument('--extended-hours', action='store_true', help='Extended hours')
    parser_place.set_defaults(func=cmd_place_order, side='buy')
    
    parser_sell = subparsers.add_parser('sell', help='Place sell order')
    parser_sell.add_argument('symbol', help='Stock symbol')
    parser_sell.add_argument('quantity', type=float, help='Quantity')
    parser_sell.add_argument('--type', default='market', help='Order type (market, limit, stop)')
    parser_sell.add_argument('--limit-price', type=float, help='Limit price')
    parser_sell.add_argument('--stop-price', type=float, help='Stop price')
    parser_sell.add_argument('--tif', default='day', help='Time in force (day, gtc, ioc, fok)')
    parser_sell.add_argument('--extended-hours', action='store_true', help='Extended hours')
    parser_sell.set_defaults(func=cmd_place_order, side='sell')
    
    # Cancel order command
    parser_cancel = subparsers.add_parser('cancel', help='Cancel order(s)')
    parser_cancel.add_argument('--order-id', help='Specific order ID')
    parser_cancel.add_argument('--all', action='store_true', help='Cancel all orders')
    parser_cancel.set_defaults(func=cmd_cancel_order)
    
    # Close position command
    parser_close = subparsers.add_parser('close', help='Close position(s)')
    parser_close.add_argument('--symbol', help='Specific symbol')
    parser_close.add_argument('--quantity', type=float, help='Quantity to close')
    parser_close.add_argument('--percentage', type=float, help='Percentage to close')
    parser_close.add_argument('--all', action='store_true', help='Close all positions')
    parser_close.add_argument('--cancel-orders', action='store_true', help='Also cancel orders')
    parser_close.set_defaults(func=cmd_close_position)
    
    # Bars command
    parser_bars = subparsers.add_parser('bars', help='Get historical bars')
    parser_bars.add_argument('symbol', help='Stock symbol')
    parser_bars.add_argument('--days', type=int, default=5, help='Days to look back')
    parser_bars.add_argument('--timeframe', default='1Day', help='Timeframe (1Min, 5Min, 1Hour, 1Day)')
    parser_bars.add_argument('--limit', type=int, default=10, help='Max bars to show')
    parser_bars.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser_bars.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser_bars.set_defaults(func=cmd_bars)
    
    # Quote command
    parser_quote = subparsers.add_parser('quote', help='Get latest quote')
    parser_quote.add_argument('symbols', help='Symbol or comma-separated symbols')
    parser_quote.set_defaults(func=cmd_quote)
    
    # Snapshot command
    parser_snapshot = subparsers.add_parser('snapshot', help='Get stock snapshot')
    parser_snapshot.add_argument('symbols', help='Symbol or comma-separated symbols')
    parser_snapshot.set_defaults(func=cmd_snapshot)
    
    # Assets command
    parser_assets = subparsers.add_parser('assets', help='Get asset information')
    parser_assets.add_argument('-s', '--symbol', help='Specific symbol')
    parser_assets.add_argument('--status', help='Status filter (active, inactive)')
    parser_assets.add_argument('--asset-class', help='Asset class (us_equity, crypto)')
    parser_assets.add_argument('--exchange', help='Exchange filter')
    parser_assets.add_argument('--limit', type=int, help='Max results to show')
    parser_assets.set_defaults(func=cmd_assets)
    
    # Portfolio history command
    parser_history = subparsers.add_parser('history', help='Get portfolio history')
    parser_history.add_argument('--period', default='1W', help='Period (1D, 1W, 1M, 3M, 1A, all)')
    parser_history.add_argument('--timeframe', help='Timeframe (1Min, 5Min, 15Min, 1H, 1D)')
    parser_history.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser_history.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser_history.add_argument('--limit', type=int, help='Max data points to show')
    parser_history.set_defaults(func=cmd_portfolio_history)
    
    # Corporate actions command
    parser_ca = subparsers.add_parser('corporate-actions', help='Get corporate actions')
    parser_ca.add_argument('--types', help='Action types (dividend, split, merger)')
    parser_ca.add_argument('--symbols', help='Comma-separated symbols')
    parser_ca.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser_ca.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser_ca.add_argument('--limit', type=int, default=50, help='Max results')
    parser_ca.set_defaults(func=cmd_corporate_actions)
    
    # Options command
    parser_options = subparsers.add_parser('options', help='Get option contracts')
    parser_options.add_argument('symbol', help='Underlying symbol')
    parser_options.add_argument('--exp-gte', help='Min expiration date (YYYY-MM-DD)')
    parser_options.add_argument('--exp-lte', help='Max expiration date (YYYY-MM-DD)')
    parser_options.add_argument('--strike-gte', type=float, help='Min strike price')
    parser_options.add_argument('--strike-lte', type=float, help='Max strike price')
    parser_options.add_argument('--option-type', choices=['call', 'put'], help='Option type')
    parser_options.add_argument('--status', default='active', help='Status (active, inactive)')
    parser_options.add_argument('--limit', type=int, default=20, help='Max results')
    parser_options.set_defaults(func=cmd_options)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Initialize client
        client = AlpacaClient()
        
        # Execute command
        args.func(client, args)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
CLI to buy, sell, and manage orders for IBKR using the TWS API.
Requires 'ibapi' package installed (pip install ibapi).
"""

import argparse
import sys
import time
import threading
from typing import Optional, List, Dict

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import OrderId
except ImportError:
    print("❌ Error: 'ibapi' module not found. Please install it using 'pip install ibapi'.", file=sys.stderr)
    sys.exit(1)


class IBKRWrapper(EWrapper, EClient):
    """
    Wrapper class to handle IBKR TWS API callbacks.
    Inherits from both EWrapper (callbacks) and EClient (requests).
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.next_order_id = None
        self.open_orders = []
        self.positions = []
        self.account_summary = {}
        self.market_data = {}
        self.order_event = threading.Event()
        self.connection_event = threading.Event()
        self.position_event = threading.Event()
        self.account_event = threading.Event()
        self.market_data_event = threading.Event()
        self.is_connected = False

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Filter out connectivity info messages (2104, 2106, 2158)
        if errorCode in [2104, 2106, 2158]:
            return
        print(f"⚠️ IBKR Error {errorCode}: {errorString}", file=sys.stderr)

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_order_id = orderId
        self.connection_event.set()

    def connectAck(self):
        super().connectAck()
        self.is_connected = True

    def connectionClosed(self):
        super().connectionClosed()
        self.is_connected = False

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        self.open_orders.append({
            "orderId": orderId,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "action": order.action,
            "qty": order.totalQuantity,
            "type": order.orderType,
            "lmtPrice": order.lmtPrice,
            "status": orderState.status
        })

    def openOrderEnd(self):
        super().openOrderEnd()
        self.order_event.set()
        
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
        # Optional: Print status updates if verbose mode was implemented
        # print(f"Order Status: Id: {orderId}, Status: {status}, Filled: {filled}, Remaining: {remaining}")

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        super().position(account, contract, position, avgCost)
        self.positions.append({
            "account": account,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "currency": contract.currency,
            "position": position,
            "avgCost": avgCost
        })

    def positionEnd(self):
        super().positionEnd()
        self.position_event.set()

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        super().accountSummary(reqId, account, tag, value, currency)
        if account not in self.account_summary:
            self.account_summary[account] = {}
        self.account_summary[account][tag] = (value, currency)

    def accountSummaryEnd(self, reqId: int):
        super().accountSummaryEnd(reqId)
        self.account_event.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        super().tickPrice(reqId, tickType, price, attrib)
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        self.market_data[reqId][tickType] = price
        
        # 4=Last, 68=DelayedLast, 9=Close. If we get any of these, we have a "price".
        if tickType in [4, 9, 68] and price > 0:
            self.market_data_event.set()


class IBKRClient:
    """
    Client class to manage connection and high-level operations.
    """
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.app = IBKRWrapper()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.thread = None

    def connect(self):
        """Connect to TWS/Gateway and start the message loop thread."""
        print(f"Connecting to IBKR TWS at {self.host}:{self.port} (Client ID: {self.client_id})...")
        self.app.connect(self.host, self.port, self.client_id)
        
        # Start the socket thread
        self.thread = threading.Thread(target=self.app.run, daemon=True)
        self.thread.start()
        
        # Wait for nextValidId which confirms connection and readiness
        if not self.app.connection_event.wait(timeout=5):
            print("❌ Failed to connect to TWS/Gateway. Make sure it is running and API is enabled.")
            self.disconnect()
            sys.exit(1)
        print("✅ Connected.")

    def disconnect(self):
        """Disconnect from TWS."""
        if self.app.isConnected():
            self.app.disconnect()
            print("Disconnected.")

    def place_order(self, symbol, action, qty, order_type="MKT", price=None, exchange="SMART", currency="USD"):
        """Place a stock order."""
        if not self.app.next_order_id:
            print("❌ No valid Order ID received.")
            return

        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency

        order = Order()
        order.action = action.upper()
        order.totalQuantity = float(qty)
        order.orderType = order_type.upper()
        if price:
            order.lmtPrice = float(price)
        
        # TWS expects TIF (Time in Force). Default DAY.
        order.tif = "DAY"

        oid = self.app.next_order_id
        self.app.placeOrder(oid, contract, order)
        print(f"✅ Placed {action} order for {qty} {symbol} (Order ID: {oid})")
        
        # Increment local order ID for subsequent calls
        self.app.next_order_id += 1
        
        # Give it a moment to propagate
        time.sleep(1)

    def get_open_orders(self):
        """Request and return list of open orders."""
        self.app.open_orders = []
        self.app.order_event.clear()
        
        print("Requesting open orders...")
        self.app.reqOpenOrders()
        
        # Wait for openOrderEnd
        if self.app.order_event.wait(timeout=5):
            return self.app.open_orders
        else:
            print("⚠️ Timeout waiting for open orders response.")
            return self.app.open_orders

    def cancel_order(self, order_id):
        """Cancel a specific order by ID."""
        self.app.cancelOrder(order_id, "")
        print(f"✅ Requested cancellation for Order ID: {order_id}")
        time.sleep(1)

    def get_positions(self):
        """Request and return list of positions."""
        self.app.positions = []
        self.app.position_event.clear()
        
        print("Requesting positions...")
        self.app.reqPositions()
        
        if self.app.position_event.wait(timeout=10):
            self.app.cancelPositions()
            return self.app.positions
        else:
            print("⚠️ Timeout waiting for positions response.")
            self.app.cancelPositions()
            return self.app.positions

    def get_account_summary(self):
        """Request and return account summary."""
        self.app.account_summary = {}
        self.app.account_event.clear()
        
        print("Requesting account summary...")
        # Requesting specific tags for summary
        tags = "NetLiquidation,TotalCashValue,BuyingPower,EquityWithLoanValue,AvailableFunds,GrossPositionValue"
        self.app.reqAccountSummary(9001, "All", tags)
        
        if self.app.account_event.wait(timeout=10):
            self.app.cancelAccountSummary(9001)
            return self.app.account_summary
        else:
            print("⚠️ Timeout waiting for account summary response.")
            self.app.cancelAccountSummary(9001)
            return self.app.account_summary

    def get_price(self, symbol, exchange="SMART", currency="USD"):
        """Get the current price (Last/Close) for a stock."""
        if not self.app.next_order_id:
            return None

        req_id = self.app.next_order_id
        self.app.next_order_id += 1
        
        self.app.market_data = {}
        self.app.market_data_event.clear()
        
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        
        print(f"Requesting market data for {symbol}...")
        # snapshot=True (4th arg) to get a one-time snapshot
        self.app.reqMktData(req_id, contract, "", True, False, [])
        
        if self.app.market_data_event.wait(timeout=5):
            data = self.app.market_data.get(req_id, {})
            # Prioritize Last (4) or Delayed Last (68), then Close (9)
            price = data.get(4) or data.get(68) or data.get(9)
            return price
        else:
            print("⚠️ Timeout waiting for market data.")
            return None


def cmd_buy(client, args):
    client.place_order(args.symbol, "BUY", args.quantity, args.type, args.price)

def cmd_sell(client, args):
    client.place_order(args.symbol, "SELL", args.quantity, args.type, args.price)

def cmd_orders(client, args):
    orders = client.get_open_orders()
    if not orders:
        print("No open orders found.")
        return

    print(f"\n=== OPEN ORDERS ({len(orders)}) ===")
    # Header
    print(f"{'ID':<10} {'Symbol':<8} {'Action':<6} {'Qty':<8} {'Type':<6} {'Price':<10} {'Status':<10}")
    print("-" * 65)
    
    for o in orders:
        price_display = f"{o['lmtPrice']:.2f}" if o['lmtPrice'] and o['lmtPrice'] < 1e9 else "MKT"
        print(f"{o['orderId']:<10} {o['symbol']:<8} {o['action']:<6} {o['qty']:<8} {o['type']:<6} {price_display:<10} {o['status']:<10}")

def cmd_cancel(client, args):
    client.cancel_order(args.order_id)

def cmd_positions(client, args):
    positions = client.get_positions()
    if not positions:
        print("No open positions found.")
        return

    print(f"\n=== POSITIONS ({len(positions)}) ===")
    print(f"{'Account':<12} {'Symbol':<8} {'SecType':<8} {'Qty':<10} {'AvgCost':<10}")
    print("-" * 60)
    for p in positions:
        print(f"{p['account']:<12} {p['symbol']:<8} {p['secType']:<8} {p['position']:<10} {p['avgCost']:.2f}")

def cmd_account(client, args):
    summary = client.get_account_summary()
    if not summary:
        print("No account summary received.")
        return

    print(f"\n=== ACCOUNT SUMMARY ===")
    for account, tags in summary.items():
        print(f"Account: {account}")
        for tag, (val, currency) in tags.items():
            curr_str = f" {currency}" if currency else ""
            print(f"  {tag:<25}: {val}{curr_str}")

def cmd_price(client, args):
    price = client.get_price(args.symbol)
    if price:
        print(f"Price for {args.symbol.upper()}: {price:.2f}")
    else:
        print(f"Could not retrieve price for {args.symbol.upper()}")


def main():
    parser = argparse.ArgumentParser(
        description="IBKR TWS API CLI - Buy, Sell, and Manage Orders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1", help="TWS Host IP")
    parser.add_argument("--port", type=int, default=7497, help="TWS Port (7497=Paper, 7496=Live)")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID (must be unique)")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Buy Command
    p_buy = subparsers.add_parser("buy", help="Place a buy order")
    p_buy.add_argument("symbol", help="Stock symbol (e.g., AAPL)")
    p_buy.add_argument("quantity", type=float, help="Number of shares")
    p_buy.add_argument("--type", default="MKT", choices=["MKT", "LMT"], help="Order type: Market or Limit")
    p_buy.add_argument("--price", type=float, help="Limit price (required for LMT)")
    p_buy.set_defaults(func=cmd_buy)

    # Sell Command
    p_sell = subparsers.add_parser("sell", help="Place a sell order")
    p_sell.add_argument("symbol", help="Stock symbol (e.g., AAPL)")
    p_sell.add_argument("quantity", type=float, help="Number of shares")
    p_sell.add_argument("--type", default="MKT", choices=["MKT", "LMT"], help="Order type: Market or Limit")
    p_sell.add_argument("--price", type=float, help="Limit price (required for LMT)")
    p_sell.set_defaults(func=cmd_sell)

    # Orders Command
    p_orders = subparsers.add_parser("orders", help="List all open orders")
    p_orders.set_defaults(func=cmd_orders)

    # Cancel Command
    p_cancel = subparsers.add_parser("cancel", help="Cancel an order by ID")
    p_cancel.add_argument("order_id", type=int, help="Order ID to cancel")
    p_cancel.set_defaults(func=cmd_cancel)

    # Positions Command
    p_positions = subparsers.add_parser("positions", help="List all positions")
    p_positions.set_defaults(func=cmd_positions)

    # Account Command
    p_account = subparsers.add_parser("account", help="Get account summary")
    p_account.set_defaults(func=cmd_account)

    # Price Command
    p_price = subparsers.add_parser("price", help="Get current price for a symbol")
    p_price.add_argument("symbol", help="Stock symbol")
    p_price.set_defaults(func=cmd_price)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validation
    if (args.command in ["buy", "sell"]) and args.type == "LMT" and not args.price:
        print("❌ Error: --price is required for LMT orders.", file=sys.stderr)
        sys.exit(1)

    # Execution
    client = IBKRClient(args.host, args.port, args.client_id)
    try:
        client.connect()
        args.func(client, args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
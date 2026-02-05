"""
Alpaca Markets API Client
Provides functions to interact with Alpaca trading platform
"""

import os
import sys
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union
from enum import Enum
from dotenv import load_dotenv


class Sort(Enum):
    """Sort order for historical data queries"""
    ASC = "asc"
    DESC = "desc"


class AlpacaClient:
    """Client for interacting with Alpaca Markets API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Alpaca client
        
        Args:
            api_key: Alpaca API key (if not provided, loads from .env)
            api_secret: Alpaca API secret (if not provided, loads from .env)
            base_url: Base URL for API (defaults to paper trading)
        """
        # Load environment variables
        load_dotenv()
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        self.base_url = base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.base_url = self.base_url.rstrip('/')
        
        # Market data uses a different URL
        self.data_url = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')
        self.data_url = self.data_url.rstrip('/')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be provided or set in .env file")
        
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, use_data_api: bool = False, **kwargs) -> Dict:
        """
        Make an API request
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            use_data_api: If True, use data API URL instead of trading API URL
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response as dictionary (or empty dict for 204 No Content)
        """
        base = self.data_url if use_data_api else self.base_url
        
        if not use_data_api and '/v2' not in base and not endpoint.startswith('/v'):
            endpoint = f"/v2{endpoint}"
            
        url = f"{base}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            # Handle 204 No Content (e.g., DELETE requests)
            if response.status_code == 204 or not response.content:
                return {}
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}", file=sys.stderr)
            print(f"Response: {response.text}", file=sys.stderr)
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}", file=sys.stderr)
            raise
    
    def get_clock(self) -> Dict:
        """
        Get current market clock status
        
        Returns:
            Dictionary containing:
                - timestamp: Current timestamp (RFC3339)
                - is_open: Whether the market is currently open
                - next_open: Next market open time (RFC3339)
                - next_close: Next market close time (RFC3339)
        """
        return self._make_request('GET', '/clock')
    
    def get_account_info(self) -> Dict:
        """
        Get account information including balance, margin, and status
        
        Returns:
            Dictionary containing:
                - account_number: Account number
                - status: Account status (ACTIVE, etc.)
                - currency: Account currency
                - cash: Available cash
                - portfolio_value: Total portfolio value
                - buying_power: Current buying power
                - equity: Total equity
                - last_equity: Previous day's equity
                - multiplier: Margin multiplier
                - initial_margin: Initial margin requirement
                - maintenance_margin: Maintenance margin requirement
                - daytrade_count: Number of day trades in last 5 days
                - daytrading_buying_power: Buying power for day trading
                - pattern_day_trader: Whether account is flagged as PDT
        """
        return self._make_request('GET', '/account')
    
    def get_all_positions(self) -> List[Dict]:
        """
        Get all current positions
        
        Returns:
            List of dictionaries, each containing:
                - asset_id: Unique asset ID
                - symbol: Stock symbol
                - exchange: Exchange where asset is traded
                - asset_class: Asset class (us_equity, etc.)
                - qty: Number of shares held
                - avg_entry_price: Average entry price
                - side: Position side (long/short)
                - market_value: Current market value
                - cost_basis: Total cost basis
                - unrealized_pl: Unrealized profit/loss
                - unrealized_plpc: Unrealized profit/loss percentage
                - unrealized_intraday_pl: Intraday unrealized P/L
                - unrealized_intraday_plpc: Intraday unrealized P/L percentage
                - current_price: Current asset price
                - lastday_price: Previous day's closing price
                - change_today: Today's price change percentage
        """
        return self._make_request('GET', '/positions')
    
    def get_open_position(self, symbol: str) -> Dict:
        """
        Get detailed information on a specific open position
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            Dictionary containing detailed position information:
                - asset_id: Unique asset ID
                - symbol: Stock symbol
                - exchange: Exchange where asset is traded
                - asset_class: Asset class
                - qty: Number of shares held
                - avg_entry_price: Average entry price
                - side: Position side (long/short)
                - market_value: Current market value
                - cost_basis: Total cost basis
                - unrealized_pl: Unrealized profit/loss
                - unrealized_plpc: Unrealized profit/loss percentage
                - unrealized_intraday_pl: Intraday unrealized P/L
                - unrealized_intraday_plpc: Intraday unrealized P/L percentage
                - current_price: Current asset price
                - lastday_price: Previous day's closing price
                - change_today: Today's price change percentage
        """
        return self._make_request('GET', f'/positions/{symbol.upper()}')
    
    def close_position(self, symbol: str, qty: Optional[int] = None, percentage: Optional[float] = None) -> Dict:
        """
        Close a position (fully or partially)
        
        Args:
            symbol: Stock symbol
            qty: Number of shares to close (optional)
            percentage: Percentage of position to close (optional)
            
        Returns:
            Order information
        """
        params = {}
        if qty:
            params['qty'] = qty
        if percentage:
            params['percentage'] = percentage
            
        return self._make_request('DELETE', f'/positions/{symbol.upper()}', params=params)
    
    def close_all_positions(self, cancel_orders: bool = True) -> List[Dict]:
        """
        Close all open positions
        
        Args:
            cancel_orders: Whether to cancel all open orders as well
            
        Returns:
            List of order information for each closed position
        """
        params = {'cancel_orders': cancel_orders}
        return self._make_request('DELETE', '/positions', params=params)
    
    def get_orders(self, status: Optional[str] = None, limit: Optional[int] = None, 
                   after: Optional[str] = None, until: Optional[str] = None, 
                   direction: Optional[str] = None, nested: Optional[bool] = None,
                   side: Optional[str] = None, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve all or filtered orders
        
        Args:
            status: Order status to filter by (open, closed, all). Default: open
            limit: Max number of orders to retrieve (default: 50, max: 500)
            after: Filter to orders after this timestamp (RFC3339 format or date YYYY-MM-DD)
            until: Filter to orders until this timestamp (RFC3339 format or date YYYY-MM-DD)
            direction: Order direction (asc or desc). Default: desc
            nested: If true, include nested multi-leg orders
            side: Filter by side (buy or sell)
            symbols: List of symbols to filter by
            
        Returns:
            List of order dictionaries containing:
                - id: Order ID
                - client_order_id: Client-specified order ID
                - created_at: Timestamp when order was created
                - updated_at: Timestamp when order was last updated
                - submitted_at: Timestamp when order was submitted
                - filled_at: Timestamp when order was filled (if filled)
                - expired_at: Timestamp when order expired (if expired)
                - canceled_at: Timestamp when order was canceled (if canceled)
                - failed_at: Timestamp when order failed (if failed)
                - replaced_at: Timestamp when order was replaced (if replaced)
                - replaced_by: Order ID that replaced this order
                - replaces: Order ID that this order replaces
                - asset_id: Asset ID
                - symbol: Symbol
                - asset_class: Asset class
                - notional: Dollar amount to trade (for fractional shares)
                - qty: Quantity
                - filled_qty: Filled quantity
                - filled_avg_price: Average fill price
                - order_class: Order class (simple, bracket, oco, oto)
                - order_type: Order type (market, limit, stop, stop_limit, trailing_stop)
                - type: Order type (same as order_type)
                - side: Side (buy or sell)
                - time_in_force: Time in force (day, gtc, opg, cls, ioc, fok)
                - limit_price: Limit price (if applicable)
                - stop_price: Stop price (if applicable)
                - status: Order status
                - extended_hours: Whether order can execute in extended hours
                - legs: Legs for multi-leg orders
                - trail_percent: Trailing stop percent
                - trail_price: Trailing stop price
                - hwm: High water mark (for trailing stops)
        """
        params = {}
        if status:
            params['status'] = status
        if limit:
            params['limit'] = limit
        if after:
            params['after'] = after
        if until:
            params['until'] = until
        if direction:
            params['direction'] = direction
        if nested is not None:
            params['nested'] = nested
        if side:
            params['side'] = side
        if symbols:
            params['symbols'] = ','.join(symbols)
            
        return self._make_request('GET', '/orders', params=params)
    
    def place_stock_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = "market", limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None, trail_price: Optional[float] = None,
                         trail_percent: Optional[float] = None, time_in_force: str = "day",
                         extended_hours: bool = False, client_order_id: Optional[str] = None,
                         order_class: Optional[str] = None, stop_loss_price: Optional[float] = None,
                         take_profit_price: Optional[float] = None) -> Dict:
        """
        Place a stock order of any type
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            side: 'buy' or 'sell'
            quantity: Number of shares (can be fractional)
            order_type: Order type - 'market', 'limit', 'stop', 'stop_limit', 'trailing_stop'
            limit_price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop and stop_limit orders)
            trail_price: Trail price in dollars (for trailing_stop)
            trail_percent: Trail percent (for trailing_stop)
            time_in_force: 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            extended_hours: Allow execution during extended hours
            client_order_id: Client-specified unique order ID
            order_class: Order class ('simple', 'bracket', 'oco', 'oto'). Use 'bracket' to attach stop loss/take profit
            stop_loss_price: Stop loss price (used when order_class='bracket')
            take_profit_price: Take profit price (optional, used when order_class='bracket')
            
        Returns:
            Dictionary containing order information including order ID and status
        """
        order_data = {
            'symbol': symbol.upper(),
            'side': side.lower(),
            'type': order_type.lower(),
            'time_in_force': time_in_force.lower(),
            'extended_hours': extended_hours
        }
        
        # Add quantity
        order_data['qty'] = str(quantity)
        
        # Add prices based on order type
        if order_type.lower() in ['limit', 'stop_limit'] and limit_price is not None:
            order_data['limit_price'] = str(limit_price)
        
        if order_type.lower() in ['stop', 'stop_limit'] and stop_price is not None:
            order_data['stop_price'] = str(stop_price)
        
        if order_type.lower() == 'trailing_stop':
            if trail_price is not None:
                order_data['trail_price'] = str(trail_price)
            elif trail_percent is not None:
                order_data['trail_percent'] = str(trail_percent)
            else:
                raise ValueError("Either trail_price or trail_percent must be specified for trailing_stop orders")
        
        if client_order_id:
            order_data['client_order_id'] = client_order_id

        # Optional bracket/advanced orders
        if order_class:
            order_data['order_class'] = order_class.lower()
            if order_class.lower() == 'bracket':
                # Attach stop loss and optional take profit
                if stop_loss_price is not None:
                    order_data['stop_loss'] = {
                        'stop_price': str(stop_loss_price)
                    }
                if take_profit_price is not None:
                    order_data['take_profit'] = {
                        'limit_price': str(take_profit_price)
                    }
        
        return self._make_request('POST', '/orders', json=order_data)
    
    def place_option_order(self, symbol: str, side: str, quantity: float,
                          order_type: str = "market", limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None, time_in_force: str = "day",
                          extended_hours: bool = False, client_order_id: Optional[str] = None,
                          order_class: Optional[str] = None,
                          stop_loss_price: Optional[float] = None,
                          take_profit_price: Optional[float] = None) -> Dict:
        """
        Place a single-leg option order (similar to stock order structure)
        
        Args:
            symbol: Option symbol (OCC format, e.g. 'AAPL230120C00150000')
            side: 'buy' or 'sell'
            quantity: Number of contracts
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: 'day', 'gtc', etc.
            extended_hours: Allow extended hours
            client_order_id: Client order ID
            order_class: 'simple' (bracket not supported for single options yet)
            stop_loss_price: Stop loss price (used when order_class='bracket')
            take_profit_price: Take profit price (optional, used when order_class='bracket')
            
        Returns:
            Order dictionary
        """
        return self.place_stock_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=order_class,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )

    def place_crypto_order(self, symbol: str, side: str, order_type: str = "market",
                          time_in_force: str = "gtc", qty: Optional[float] = None,
                          notional: Optional[float] = None, limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None, client_order_id: Optional[str] = None) -> Dict:
        """
        Place a crypto order supporting market, limit, and stop_limit types
        
        Args:
            symbol: Crypto symbol (e.g., 'BTCUSD', 'ETHUSD')
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', or 'stop_limit'
            time_in_force: 'gtc' (good til canceled) or 'ioc' (immediate or cancel)
            qty: Quantity of crypto (either qty or notional required, not both)
            notional: Dollar amount to trade (either qty or notional required, not both)
            limit_price: Limit price (required for limit and stop_limit orders)
            stop_price: Stop price (required for stop_limit orders)
            client_order_id: Client-specified unique order ID
            
        Returns:
            Dictionary containing order information including order ID and status
        """
        if qty is None and notional is None:
            raise ValueError("Either qty or notional must be specified")
        
        if qty is not None and notional is not None:
            raise ValueError("Cannot specify both qty and notional")
        
        order_data = {
            'symbol': symbol.upper(),
            'side': side.lower(),
            'type': order_type.lower(),
            'time_in_force': time_in_force.lower()
        }
        
        # Add quantity or notional
        if qty is not None:
            order_data['qty'] = str(qty)
        else:
            order_data['notional'] = str(notional)
        
        # Add prices based on order type
        if order_type.lower() in ['limit', 'stop_limit']:
            if limit_price is None:
                raise ValueError(f"limit_price is required for {order_type} orders")
            order_data['limit_price'] = str(limit_price)
        
        if order_type.lower() == 'stop_limit':
            if stop_price is None:
                raise ValueError("stop_price is required for stop_limit orders")
            order_data['stop_price'] = str(stop_price)
        
        if client_order_id:
            order_data['client_order_id'] = client_order_id
        
        return self._make_request('POST', '/orders', json=order_data)
    
    def place_option_market_order(self, legs: List[Dict], order_class: Optional[str] = None,
                                  quantity: int = 1, time_in_force: str = "day",
                                  extended_hours: bool = False) -> Dict:
        """
        Execute option strategy (single or multi-leg)
        
        Args:
            legs: List of option leg dictionaries, each containing:
                - symbol: Option contract symbol (OCC format)
                - side: 'buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close'
                - ratio_qty: Number of contracts for this leg (for multi-leg strategies)
            order_class: Order class - 'simple', 'bracket', 'oco', 'oto'
            quantity: Number of option contracts (for simple orders)
            time_in_force: 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            extended_hours: Allow execution during extended hours
            
        Returns:
            Dictionary containing order information including order ID and status
            
        Examples:
            # Single leg - Buy to open 1 call option
            legs = [{'symbol': 'AAPL230120C00150000', 'side': 'buy_to_open'}]
            
            # Vertical spread - Buy call at lower strike, sell call at higher strike
            legs = [
                {'symbol': 'AAPL230120C00150000', 'side': 'buy_to_open', 'ratio_qty': 1},
                {'symbol': 'AAPL230120C00160000', 'side': 'sell_to_open', 'ratio_qty': 1}
            ]
            
            # Iron condor - 4 legs
            legs = [
                {'symbol': 'SPY230120P00380000', 'side': 'buy_to_open', 'ratio_qty': 1},
                {'symbol': 'SPY230120P00390000', 'side': 'sell_to_open', 'ratio_qty': 1},
                {'symbol': 'SPY230120C00410000', 'side': 'sell_to_open', 'ratio_qty': 1},
                {'symbol': 'SPY230120C00420000', 'side': 'buy_to_open', 'ratio_qty': 1}
            ]
        """
        order_data = {
            'type': 'market',
            'time_in_force': time_in_force.lower(),
            'extended_hours': extended_hours,
            'legs': legs
        }
        
        # Always include quantity in the order data
        order_data['qty'] = str(quantity)
        
        if order_class:
            order_data['order_class'] = order_class.lower()
        
        return self._make_request('POST', '/orders', json=order_data)
    
    def place_option_limit_order(self, legs: List[Dict], limit_price: float,
                                 order_class: Optional[str] = None,
                                 quantity: int = 1, time_in_force: str = "day",
                                 extended_hours: bool = False) -> Dict:
        """
        Execute option strategy with limit price (single or multi-leg)
        
        Args:
            legs: List of option leg dictionaries, each containing:
                - symbol: Option contract symbol (OCC format)
                - side: 'buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close'
                - ratio_qty: Number of contracts for this leg (for multi-leg strategies)
            limit_price: Limit price for the spread (positive = credit, negative = debit)
            order_class: Order class - 'simple', 'bracket', 'oco', 'oto'
            quantity: Number of spread contracts
            time_in_force: 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            extended_hours: Allow execution during extended hours
            
        Returns:
            Dictionary containing order information including order ID and status
        """
        order_data = {
            'type': 'limit',
            'limit_price': str(abs(limit_price)),  # Alpaca expects positive value
            'time_in_force': time_in_force.lower(),
            'extended_hours': extended_hours,
            'legs': legs,
            'qty': str(quantity)
        }
        
        if order_class:
            order_data['order_class'] = order_class.lower()
        
        return self._make_request('POST', '/orders', json=order_data)
    
    def cancel_all_orders(self) -> List[Dict]:
        """
        Cancel all open orders
        
        Returns:
            List of dictionaries containing status information for each canceled order:
                - id: Order ID
                - status: HTTP status code (200, 404, 500)
                - body: Response body (order object if successful, error if not)
        """
        return self._make_request('DELETE', '/orders')
    
    def cancel_order_by_id(self, order_id: str) -> Dict:
        """
        Cancel a specific order by its ID
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            Dictionary containing the canceled order information
            
        Raises:
            HTTPError: If order doesn't exist or cannot be canceled
        """
        return self._make_request('DELETE', f'/orders/{order_id}')
    
    def cancel_orders_for_symbol(self, symbol: str) -> List[Dict]:
        """
        Cancel all open orders for a specific symbol
        
        Args:
            symbol: Stock symbol to cancel orders for (e.g., 'AAPL')
            
        Returns:
            List of dictionaries containing canceled order info:
                - id: Order ID that was canceled
                - status: 'canceled' if successful
        """
        symbol = symbol.upper()
        open_orders = self.get_orders(status='open', symbols=[symbol])
        # Filter client-side to ensure only this symbol's orders
        symbol_orders = [o for o in open_orders if o.get('symbol', '').upper() == symbol]
        
        canceled = []
        for order in symbol_orders:
            order_id = order.get('id')
            try:
                self.cancel_order_by_id(order_id)
                canceled.append({'id': order_id, 'status': 'canceled'})
            except Exception as e:
                canceled.append({'id': order_id, 'status': 'failed', 'error': str(e)})
        
        return canceled

    def exercise_options_position(self, symbol_or_contract_id: str) -> Dict:
        """
        Exercise a held option contract, converting it into the underlying asset
        
        Args:
            symbol_or_contract_id: Either the option symbol (OCC format) or contract ID
            
        Returns:
            Dictionary containing exercise request information:
                - id: Exercise request ID
                - symbol_or_contract_id: The identifier used
                - status: Status of the exercise request
                
        Note:
            - Only works for American-style options
            - Must be done before market close on expiration day
            - Account must have sufficient buying power for assignment
            - For calls: Need cash to buy shares at strike price
            - For puts: Need shares to sell at strike price
            
        Examples:
            # Exercise by option symbol
            client.exercise_options_position('AAPL230120C00150000')
            
            # Exercise by contract ID
            client.exercise_options_position('b0b6dd9d-8b9b-48a9-ba46-b9d54906e415')
        """
        # Determine if it's a symbol or contract ID (UUIDs contain hyphens)
        if '-' in symbol_or_contract_id:
            # It's a contract ID
            endpoint = f'/options/contracts/{symbol_or_contract_id}/exercise'
        else:
            # It's a symbol
            endpoint = f'/positions/{symbol_or_contract_id.upper()}/exercise'
        
        return self._make_request('POST', endpoint, json={})
    
    def get_option_contracts(self, underlying_symbol: str, expiration_date: Optional[str] = None,
                            expiration_date_gte: Optional[str] = None, expiration_date_lte: Optional[str] = None,
                            expiration_expression: Optional[str] = None, strike_price_gte: Optional[float] = None,
                            strike_price_lte: Optional[float] = None, type: Optional[str] = None,
                            status: Optional[str] = None, root_symbol: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict]:
        """
        Get option contracts with flexible filtering
        
        Args:
            underlying_symbol: The underlying symbol (e.g., 'AAPL', 'SPY')
            expiration_date: Filter by exact expiration date (YYYY-MM-DD)
            expiration_date_gte: Filter expirations greater than or equal to this date (YYYY-MM-DD)
            expiration_date_lte: Filter expirations less than or equal to this date (YYYY-MM-DD)
            expiration_expression: Filter by expiration pattern (e.g., '2024-01-*' for all Jan 2024)
            strike_price_gte: Filter strikes greater than or equal to this price
            strike_price_lte: Filter strikes less than or equal to this price
            type: Option type - 'call' or 'put'
            status: Contract status - 'active' or 'inactive'
            root_symbol: Root symbol for the option (usually same as underlying)
            limit: Maximum number of contracts to return (default: 100, max: 10000)
            
        Returns:
            List of option contract dictionaries containing:
                - id: Contract ID (UUID)
                - symbol: Option symbol in OCC format (e.g., 'AAPL230120C00150000')
                - name: Human-readable name
                - status: Contract status (active/inactive)
                - tradable: Whether contract is tradable
                - expiration_date: Expiration date (YYYY-MM-DD)
                - root_symbol: Root symbol
                - underlying_symbol: Underlying asset symbol
                - underlying_asset_id: Underlying asset ID
                - type: Option type (call/put)
                - style: Exercise style (american/european)
                - strike_price: Strike price
                - multiplier: Contract multiplier (usually 100)
                - size: Number of shares per contract
                - open_interest: Open interest
                - open_interest_date: Date of open interest data
                - close_price: Last close price
                - close_price_date: Date of close price
                
        Examples:
            # Get all active AAPL options expiring in January 2024
            contracts = client.get_option_contracts(
                underlying_symbol='AAPL',
                expiration_date_gte='2024-01-01',
                expiration_date_lte='2024-01-31',
                status='active'
            )
            
            # Get AAPL calls with strike between 150-160
            contracts = client.get_option_contracts(
                underlying_symbol='AAPL',
                type='call',
                strike_price_gte=150,
                strike_price_lte=160
            )
        """
        params = {'underlying_symbols': underlying_symbol.upper()}
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        if expiration_date_gte:
            params['expiration_date_gte'] = expiration_date_gte
        if expiration_date_lte:
            params['expiration_date_lte'] = expiration_date_lte
        if expiration_expression:
            params['expiration_expression'] = expiration_expression
        if strike_price_gte is not None:
            params['strike_price_gte'] = str(strike_price_gte)
        if strike_price_lte is not None:
            params['strike_price_lte'] = str(strike_price_lte)
        if type:
            params['type'] = type.lower()
        if status:
            params['status'] = status.lower()
        if root_symbol:
            params['root_symbol'] = root_symbol.upper()
        if limit:
            params['limit'] = limit
        
        response = self._make_request('GET', '/options/contracts', params=params)
        
        # API returns paginated results with 'option_contracts' key
        if isinstance(response, dict) and 'option_contracts' in response:
            return response['option_contracts']
        return response if isinstance(response, list) else []
    
    def get_option_latest_quote(self, option_symbol: str, feed: Optional[str] = None) -> Dict:
        """
        Get latest bid/ask quote for an option contract
        
        Args:
            option_symbol: Option symbol in OCC format (e.g., 'AAPL230120C00150000')
            feed: Data feed - 'opra' (default) or 'indicative'
            
        Returns:
            Dictionary containing latest quote:
                - t: Timestamp
                - ax: Ask exchange
                - ap: Ask price
                - as: Ask size
                - bx: Bid exchange
                - bp: Bid price
                - bs: Bid size
                - c: Condition
                
        Examples:
            quote = client.get_option_latest_quote('AAPL230120C00150000')
            print(f"Bid: ${quote['bp']} x {quote['bs']}")
            print(f"Ask: ${quote['ap']} x {quote['as']}")
            spread = quote['ap'] - quote['bp']
        """
        params = {}
        if feed:
            params['feed'] = feed.lower()
        
        response = self._make_request('GET', f'/v1beta1/options/quotes/{option_symbol.upper()}/latest', use_data_api=True, params=params)
        
        # API returns data wrapped in 'quote' key
        if isinstance(response, dict) and 'quote' in response:
            return response['quote']
        return response
    
    def get_option_snapshot(self, symbol_or_symbols: str, feed: Optional[str] = None) -> Dict:
        """
        Get comprehensive option snapshot including Greeks and underlying data
        
        Args:
            symbol_or_symbols: Single option symbol or comma-separated list of symbols
            feed: Data feed - 'opra' (default) or 'indicative'
            
        Returns:
            Dictionary (single symbol) or dict of dicts (multiple symbols) containing:
                - symbol: Option symbol
                - implied_volatility: Implied volatility
                - greeks: Dictionary of Greeks:
                    - delta: Delta
                    - gamma: Gamma
                    - theta: Theta
                    - vega: Vega
                    - rho: Rho
                - latestQuote: Latest quote data:
                    - t: Timestamp
                    - ap: Ask price
                    - as: Ask size
                    - bp: Bid price
                    - bs: Bid size
                - latestTrade: Latest trade data:
                    - t: Timestamp
                    - p: Price
                    - s: Size
                    - x: Exchange
                    - c: Conditions
                
        Examples:
            # Single option
            snapshot = client.get_option_snapshot('AAPL230120C00150000')
            print(f"Delta: {snapshot['greeks']['delta']}")
            print(f"IV: {snapshot['implied_volatility']}")
            
            # Multiple options
            snapshots = client.get_option_snapshot('AAPL230120C00150000,AAPL230120C00160000')
            for symbol, data in snapshots.items():
                print(f"{symbol}: Delta={data['greeks']['delta']}")
        """
        params = {}
        if feed:
            params['feed'] = feed.lower()
        
        # Always use the bulk endpoint to avoid issues with single symbol path interpretation
        params['symbols'] = symbol_or_symbols.upper()
        response = self._make_request('GET', '/v1beta1/options/snapshots', use_data_api=True, params=params)
        
        # API returns data wrapped in 'snapshots' key as dict
        if isinstance(response, dict) and 'snapshots' in response:
            snapshots = response['snapshots']
            # If user requested a single symbol without commas, return just that snapshot for backward compatibility
            if ',' not in symbol_or_symbols and symbol_or_symbols.upper() in snapshots:
                return snapshots[symbol_or_symbols.upper()]
            return snapshots
        return response
    
    def get_option_bars(self, symbol_or_symbols: Union[str, List[str]], timeframe: str = "1Min",
                       start: Optional[str] = None, end: Optional[str] = None,
                       limit: int = 1000, page_token: Optional[str] = None) -> Dict:
        """
        Get historical option bars.
        Returns full response dict including 'bars' and 'next_page_token'.
        """
        params = {
            'timeframe': timeframe,
            'limit': limit
        }
        if start: params['start'] = start
        if end: params['end'] = end
        if page_token: params['page_token'] = page_token
        
        if isinstance(symbol_or_symbols, list):
            params['symbols'] = ','.join(symbol_or_symbols)
        else:
            params['symbols'] = symbol_or_symbols
            
        return self._make_request('GET', '/v1beta1/options/bars', use_data_api=True, params=params)

    def get_option_trades(self, symbol_or_symbols: Union[str, List[str]],
                         start: Optional[str] = None, end: Optional[str] = None,
                         limit: int = 1000, page_token: Optional[str] = None) -> Dict:
        """
        Get historical option trades.
        Returns full response dict including 'trades' and 'next_page_token'.
        """
        params = {'limit': limit}
        if start: params['start'] = start
        if end: params['end'] = end
        if page_token: params['page_token'] = page_token
        
        if isinstance(symbol_or_symbols, list):
            params['symbols'] = ','.join(symbol_or_symbols)
        else:
            params['symbols'] = symbol_or_symbols
            
        return self._make_request('GET', '/v1beta1/options/trades', use_data_api=True, params=params)
        
    def get_option_quotes(self, symbol_or_symbols: Union[str, List[str]],
                         start: Optional[str] = None, end: Optional[str] = None,
                         limit: int = 1000, page_token: Optional[str] = None) -> Dict:
        """
        Get historical option quotes.
        Returns full response dict including 'quotes' and 'next_page_token'.
        """
        params = {'limit': limit}
        if start: params['start'] = start
        if end: params['end'] = end
        if page_token: params['page_token'] = page_token
        
        if isinstance(symbol_or_symbols, list):
            params['symbols'] = ','.join(symbol_or_symbols)
        else:
            params['symbols'] = symbol_or_symbols
            
        return self._make_request('GET', '/v1beta1/options/quotes', use_data_api=True, params=params)

    def _calculate_start_time(self, days: int = 0, hours: int = 0, minutes: int = 0, start: Optional[str] = None) -> str:
        """Helper to calculate start time for historical data queries"""
        if start:
            return start
        
        delta = timedelta(days=days, hours=hours, minutes=minutes)
        start_time = datetime.now(timezone.utc) - delta
        return start_time.isoformat() + 'Z'
    
    def get_stock_bars(self, symbol: str, days: int = 5, hours: int = 0, minutes: int = 15,
                      timeframe: str = "1Day", limit: int = 1000, start: Optional[str] = None,
                      end: Optional[str] = None, sort: Sort = Sort.ASC, feed: Optional[str] = None,
                      currency: Optional[str] = None, asof: Optional[str] = None) -> List[Dict]:
        """
        Get OHLCV historical bars with flexible timeframes
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            days: Number of days to look back (default: 5)
            hours: Additional hours to look back (default: 0)
            minutes: Additional minutes to look back (default: 15)
            timeframe: Bar timeframe - '1Min', '5Min', '15Min', '1Hour', '1Day', etc.
            limit: Maximum number of bars to return (max: 10000)
            start: Start time (RFC3339 or YYYY-MM-DD). Overrides days/hours/minutes
            end: End time (RFC3339 or YYYY-MM-DD)
            sort: Sort order (Sort.ASC or Sort.DESC)
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            asof: As-of date for point-in-time queries (YYYY-MM-DD)
            
        Returns:
            List of bar dictionaries containing:
                - t: Timestamp
                - o: Open price
                - h: High price
                - l: Low price
                - c: Close price
                - v: Volume
                - n: Number of trades (if available)
                - vw: Volume-weighted average price (if available)
                
        Examples:
            # Get daily bars for last 30 days
            bars = client.get_stock_bars('AAPL', days=30, timeframe='1Day')
            
            # Get 5-minute bars for last 2 hours
            bars = client.get_stock_bars('AAPL', hours=2, timeframe='5Min')
            
            # Get specific date range
            bars = client.get_stock_bars('AAPL', start='2024-01-01', end='2024-01-31', timeframe='1Hour')
        """
        params = {
            'timeframe': timeframe,
            'limit': limit,
            'sort': sort.value if isinstance(sort, Sort) else sort
        }
        
        if not start:
            start = self._calculate_start_time(days, hours, minutes)
        params['start'] = start
        
        if end:
            params['end'] = end
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        if asof:
            params['asof'] = asof
        
        response = self._make_request('GET', f'/v2/stocks/{symbol.upper()}/bars', use_data_api=True, params=params)
        
        # API returns data wrapped in 'bars' key
        if isinstance(response, dict) and 'bars' in response:
            return response['bars']
        return response if isinstance(response, list) else []
    
    def get_stock_quotes(self, symbol: str, days: int = 1, hours: int = 0, minutes: int = 15,
                        limit: int = 1000, sort: Sort = Sort.ASC, feed: Optional[str] = None,
                        currency: Optional[str] = None, asof: Optional[str] = None) -> List[Dict]:
        """
        Get historical quote data (level 1 bid/ask) for a stock
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            days: Number of days to look back (default: 1)
            hours: Additional hours to look back (default: 0)
            minutes: Additional minutes to look back (default: 15)
            limit: Maximum number of quotes to return (max: 10000)
            sort: Sort order (Sort.ASC or Sort.DESC)
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            asof: As-of date for point-in-time queries (YYYY-MM-DD)
            
        Returns:
            List of quote dictionaries containing:
                - t: Timestamp
                - ax: Ask exchange
                - ap: Ask price
                - as: Ask size
                - bx: Bid exchange
                - bp: Bid price
                - bs: Bid size
                - c: Condition codes
                - z: Tape
                
        Examples:
            # Get last hour of quotes
            quotes = client.get_stock_quotes('AAPL', hours=1)
            
            # Get quotes for specific timeframe
            quotes = client.get_stock_quotes('AAPL', days=1, limit=5000)
        """
        params = {
            'start': self._calculate_start_time(days, hours, minutes),
            'limit': limit,
            'sort': sort.value if isinstance(sort, Sort) else sort
        }
        
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        if asof:
            params['asof'] = asof
        
        response = self._make_request('GET', f'/v2/stocks/{symbol.upper()}/quotes', use_data_api=True, params=params)
        
        # API returns data wrapped in 'quotes' key
        if isinstance(response, dict) and 'quotes' in response:
            return response['quotes']
        return response if isinstance(response, list) else []
    
    def get_stock_trades(self, symbol: str, days: int = 1, minutes: int = 15, hours: int = 0,
                        limit: int = 1000, sort: Sort = Sort.ASC, feed: Optional[str] = None,
                        currency: Optional[str] = None, asof: Optional[str] = None) -> List[Dict]:
        """
        Get trade-level history for a stock
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            days: Number of days to look back (default: 1)
            minutes: Additional minutes to look back (default: 15)
            hours: Additional hours to look back (default: 0)
            limit: Maximum number of trades to return (max: 10000)
            sort: Sort order (Sort.ASC or Sort.DESC)
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            asof: As-of date for point-in-time queries (YYYY-MM-DD)
            
        Returns:
            List of trade dictionaries containing:
                - t: Timestamp
                - x: Exchange
                - p: Price
                - s: Size
                - c: Condition codes
                - i: Trade ID
                - z: Tape
                
        Examples:
            # Get last 15 minutes of trades
            trades = client.get_stock_trades('AAPL', minutes=15)
            
            # Get trades from last hour
            trades = client.get_stock_trades('AAPL', hours=1, limit=5000)
        """
        params = {
            'start': self._calculate_start_time(days, hours, minutes),
            'limit': limit,
            'sort': sort.value if isinstance(sort, Sort) else sort
        }
        
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        if asof:
            params['asof'] = asof
        
        response = self._make_request('GET', f'/v2/stocks/{symbol.upper()}/trades', use_data_api=True, params=params)
        
        # API returns data wrapped in 'trades' key
        if isinstance(response, dict) and 'trades' in response:
            return response['trades']
        return response if isinstance(response, list) else []
    
    def get_stock_latest_bar(self, symbol: str, feed: Optional[str] = None, 
                            currency: Optional[str] = None) -> Dict:
        """
        Get the most recent OHLC bar for a stock
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            
        Returns:
            Dictionary containing:
                - t: Timestamp
                - o: Open price
                - h: High price
                - l: Low price
                - c: Close price
                - v: Volume
                - n: Number of trades
                - vw: Volume-weighted average price
                
        Examples:
            bar = client.get_stock_latest_bar('AAPL')
            print(f"Last close: ${bar['c']}")
        """
        params = {}
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        
        response = self._make_request('GET', f'/v2/stocks/{symbol.upper()}/bars/latest', use_data_api=True, params=params)
        
        # API returns data wrapped in 'bar' key
        if isinstance(response, dict) and 'bar' in response:
            return response['bar']
        return response
    
    def get_stock_latest_quote(self, symbol_or_symbols: Union[str, List[str]], 
                              feed: Optional[str] = None, currency: Optional[str] = None) -> Dict:
        """
        Get real-time bid/ask quote for one or more symbols
        
        Args:
            symbol_or_symbols: Single symbol string or list of symbols
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            
        Returns:
            Dictionary (single symbol) or dict of dicts (multiple symbols) containing:
                - t: Timestamp
                - ax: Ask exchange
                - ap: Ask price
                - as: Ask size
                - bx: Bid exchange
                - bp: Bid price
                - bs: Bid size
                - c: Condition codes
                - z: Tape
                
        Examples:
            # Single symbol
            quote = client.get_stock_latest_quote('AAPL')
            print(f"Bid: ${quote['bp']} x {quote['bs']}")
            print(f"Ask: ${quote['ap']} x {quote['as']}")
            
            # Multiple symbols
            quotes = client.get_stock_latest_quote(['AAPL', 'TSLA', 'MSFT'])
            for symbol, quote in quotes.items():
                print(f"{symbol}: ${quote['bp']} / ${quote['ap']}")
        """
        params = {}
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        
        # Handle single vs multiple symbols
        if isinstance(symbol_or_symbols, list):
            params['symbols'] = ','.join(s.upper() for s in symbol_or_symbols)
            response = self._make_request('GET', '/v2/stocks/quotes/latest', use_data_api=True, params=params)
            
            # API returns data wrapped in 'quotes' key as dict
            if isinstance(response, dict) and 'quotes' in response:
                return response['quotes']
            return response
        else:
            response = self._make_request('GET', f'/v2/stocks/{symbol_or_symbols.upper()}/quotes/latest', use_data_api=True, params=params)
            
            # API returns data wrapped in 'quote' key
            if isinstance(response, dict) and 'quote' in response:
                return response['quote']
            return response
    
    def get_stock_latest_trade(self, symbol: str, feed: Optional[str] = None,
                              currency: Optional[str] = None) -> Dict:
        """
        Get the latest market trade price for a stock
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            
        Returns:
            Dictionary containing:
                - t: Timestamp
                - x: Exchange
                - p: Price
                - s: Size
                - c: Condition codes
                - i: Trade ID
                - z: Tape
                
        Examples:
            trade = client.get_stock_latest_trade('AAPL')
            print(f"Last trade: ${trade['p']} ({trade['s']} shares)")
        """
        params = {}
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        
        response = self._make_request('GET', f'/v2/stocks/{symbol.upper()}/trades/latest', use_data_api=True, params=params)
        
        # API returns data wrapped in 'trade' key
        if isinstance(response, dict) and 'trade' in response:
            return response['trade']
        return response
    
    def get_stock_snapshot(self, symbol_or_symbols: Union[str, List[str]], 
                          feed: Optional[str] = None, currency: Optional[str] = None) -> Dict:
        """
        Get comprehensive snapshot with latest quote, trade, minute bar, daily bar, and previous daily bar
        
        Args:
            symbol_or_symbols: Single symbol string or list of symbols
            feed: Data feed - 'iex', 'sip', or 'otc'
            currency: Currency for pricing (default: USD)
            
        Returns:
            Dictionary (single symbol) or dict of dicts (multiple symbols) containing:
                - latestTrade: Latest trade data
                - latestQuote: Latest quote data
                - minuteBar: Current minute bar
                - dailyBar: Current daily bar
                - prevDailyBar: Previous day's bar
                
        Examples:
            # Single symbol
            snapshot = client.get_stock_snapshot('AAPL')
            print(f"Last price: ${snapshot['latestTrade']['p']}")
            print(f"Daily high: ${snapshot['dailyBar']['h']}")
            print(f"Daily volume: {snapshot['dailyBar']['v']:,}")
            
            # Multiple symbols
            snapshots = client.get_stock_snapshot(['AAPL', 'TSLA', 'MSFT'])
            for symbol, data in snapshots.items():
                price = data['latestTrade']['p']
                prev_close = data['prevDailyBar']['c']
                change = ((price - prev_close) / prev_close) * 100
                print(f"{symbol}: ${price:.2f} ({change:+.2f}%)")
        """
        params = {}
        if feed:
            params['feed'] = feed
        if currency:
            params['currency'] = currency
        
        # Handle single vs multiple symbols
        if isinstance(symbol_or_symbols, list):
            params['symbols'] = ','.join(s.upper() for s in symbol_or_symbols)
            response = self._make_request('GET', '/v2/stocks/snapshots', use_data_api=True, params=params)
            
            # API returns data as dict of snapshots
            return response
        else:
            response = self._make_request('GET', f'/v2/stocks/{symbol_or_symbols.upper()}/snapshot', use_data_api=True, params=params)
            
            # API returns single snapshot
            return response
    
    def get_portfolio_history(self, timeframe: Optional[str] = None, period: Optional[str] = None,
                             start: Optional[str] = None, end: Optional[str] = None,
                             date_end: Optional[str] = None, intraday_reporting: Optional[str] = None,
                             pnl_reset: Optional[str] = None, extended_hours: Optional[bool] = None,
                             cashflow_types: Optional[str] = None) -> Dict:
        """
        Retrieve account portfolio history with equity and P/L over time
        
        Args:
            timeframe: Timeframe for data aggregation: '1Min', '5Min', '15Min', '1H', '1D'
                      Default: '1Min' for intraday, '1D' for longer periods
            period: Time period to return: '1D', '1W', '1M', '3M', '1A', 'all'
                   If provided, start and end are ignored
            start: Start date for custom range (RFC3339 or YYYY-MM-DD)
            end: End date for custom range (RFC3339 or YYYY-MM-DD)
            date_end: Deprecated, use 'end' instead
            intraday_reporting: How to report intraday data:
                               'continuous' - report at every bar
                               'market_hours' - only during market hours
            pnl_reset: P/L reset option:
                      'per_day' - reset P/L daily
                      'beginning' - calculate from account start
            extended_hours: Include extended hours data (pre/post-market)
            cashflow_types: Comma-separated list of cashflow types to include in P/L:
                           'dividends', 'interest', 'fee', 'csd'
            
        Returns:
            Dictionary containing:
                - timestamp: List of Unix timestamps
                - equity: List of portfolio equity values at each timestamp
                - profit_loss: List of cumulative P/L values
                - profit_loss_pct: List of cumulative P/L percentages
                - base_value: Starting portfolio value
                - timeframe: Timeframe used for aggregation
                
        Examples:
            # Get last week of portfolio history
            history = client.get_portfolio_history(period='1W', timeframe='1D')
            
            # Get intraday history for today
            history = client.get_portfolio_history(period='1D', timeframe='15Min')
            
            # Get custom date range
            history = client.get_portfolio_history(
                start='2024-01-01',
                end='2024-01-31',
                timeframe='1D'
            )
            
            # Plot portfolio value over time
            import matplotlib.pyplot as plt
            history = client.get_portfolio_history(period='1M', timeframe='1D')
            plt.plot(history['timestamp'], history['equity'])
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.title('Portfolio History')
            plt.show()
        """
        params = {}
        
        if timeframe:
            params['timeframe'] = timeframe
        if period:
            params['period'] = period
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        if date_end:
            params['date_end'] = date_end
        if intraday_reporting:
            params['intraday_reporting'] = intraday_reporting
        if pnl_reset:
            params['pnl_reset'] = pnl_reset
        if extended_hours is not None:
            params['extended_hours'] = extended_hours
        if cashflow_types:
            params['cashflow_types'] = cashflow_types
        
        return self._make_request('GET', '/account/portfolio/history', params=params)
    
    def get_asset(self, symbol: str) -> Dict:
        """
        Get asset metadata for a specific symbol
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTCUSD', 'SPY')
            
        Returns:
            Dictionary containing:
                - id: Asset ID (UUID)
                - class: Asset class (us_equity, crypto, etc.)
                - exchange: Exchange (NASDAQ, NYSE, CRYPTO, etc.)
                - symbol: Asset symbol
                - name: Asset name
                - status: Trading status (active, inactive)
                - tradable: Whether asset is tradable
                - marginable: Whether asset can be traded on margin
                - maintenance_margin_requirement: Margin requirement percentage
                - shortable: Whether asset can be sold short
                - easy_to_borrow: Whether asset is easy to borrow for shorting
                - fractionable: Whether fractional shares are supported
                - min_order_size: Minimum order size
                - min_trade_increment: Minimum trade increment
                - price_increment: Minimum price increment
                - attributes: List of special attributes
                
        Examples:
            asset = client.get_asset('AAPL')
            print(f"Name: {asset['name']}")
            print(f"Exchange: {asset['exchange']}")
            print(f"Tradable: {asset['tradable']}")
            print(f"Marginable: {asset['marginable']}")
            print(f"Fractionable: {asset['fractionable']}")
        """
        return self._make_request('GET', f'/assets/{symbol.upper()}')
    
    def get_all_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None,
                      exchange: Optional[str] = None, attributes: Optional[str] = None) -> List[Dict]:
        """
        List all tradable instruments with filtering options
        
        Args:
            status: Filter by status:
                   'active' - currently tradable
                   'inactive' - not currently tradable
            asset_class: Filter by asset class:
                        'us_equity' - US stocks
                        'crypto' - cryptocurrencies
                        'us_option' - US options
            exchange: Filter by exchange:
                     For stocks: 'AMEX', 'ARCA', 'BATS', 'NYSE', 'NASDAQ', 'NYSEARCA', 'OTC'
                     For crypto: 'CRYPTO', 'FTXU', 'CBSE', 'GNSS', 'ERSX'
            attributes: Comma-separated list of attributes to filter by:
                       'ptp_no_exception' - Doesn't require PTP (pattern day trader) exception
                       'ptp_with_exception' - Requires PTP exception
                       'fractional_eh_enabled' - Fractional trading in extended hours
                       'ipo' - IPO stock
                       'ipo_delayed' - IPO with delayed trading
                       
        Returns:
            List of asset dictionaries, each containing:
                - id: Asset ID (UUID)
                - class: Asset class
                - exchange: Exchange
                - symbol: Asset symbol
                - name: Asset name
                - status: Trading status
                - tradable: Whether asset is tradable
                - marginable: Whether asset can be traded on margin
                - maintenance_margin_requirement: Margin requirement percentage
                - shortable: Whether asset can be sold short
                - easy_to_borrow: Whether asset is easy to borrow
                - fractionable: Whether fractional shares are supported
                - min_order_size: Minimum order size
                - min_trade_increment: Minimum trade increment
                - price_increment: Minimum price increment
                - attributes: List of special attributes
                
        Examples:
            # Get all active US stocks
            assets = client.get_all_assets(status='active', asset_class='us_equity')
            print(f"Found {len(assets)} tradable stocks")
            
            # Get all cryptocurrencies
            crypto_assets = client.get_all_assets(asset_class='crypto')
            for asset in crypto_assets:
                print(f"{asset['symbol']}: {asset['name']}")
            
            # Get all NASDAQ stocks
            nasdaq_stocks = client.get_all_assets(
                status='active',
                asset_class='us_equity',
                exchange='NASDAQ'
            )
            
            # Get fractionable stocks
            fractional_stocks = client.get_all_assets(
                status='active',
                asset_class='us_equity',
                attributes='fractionable'
            )
            
            # Filter by multiple attributes
            assets = client.get_all_assets(
                status='active',
                attributes='ptp_no_exception,fractional_eh_enabled'
            )
        """
        params = {}
        
        if status:
            params['status'] = status.lower()
        if asset_class:
            params['asset_class'] = asset_class.lower()
        if exchange:
            params['exchange'] = exchange.upper()
        if attributes:
            params['attributes'] = attributes.lower()
        
        return self._make_request('GET', '/assets', params=params)
    
    def get_corporate_actions(self, ca_types: Optional[str] = None, start: Optional[str] = None,
                             end: Optional[str] = None, symbols: Optional[Union[str, List[str]]] = None,
                             cusips: Optional[Union[str, List[str]]] = None, ids: Optional[Union[str, List[str]]] = None,
                             limit: int = 1000, sort: str = "asc") -> List[Dict]:
        """
        Get historical and future corporate actions (e.g., earnings, dividends, splits)
        
        Args:
            ca_types: Comma-separated list of corporate action types to filter by:
                     'dividend' - Cash dividend distributions
                     'merger' - Merger or acquisition
                     'spinoff' - Company spinoff
                     'split' - Stock split (forward or reverse)
                     'name_change' - Symbol or company name change
                     'rights_distribution' - Rights offering
                     'unit_split' - Unit split
                     'recapitalization' - Recapitalization event
                     'redemption' - Security redemption
            start: Start date for the query (YYYY-MM-DD)
            end: End date for the query (YYYY-MM-DD)
            symbols: Single symbol string or list of symbols to filter by
            cusips: Single CUSIP or list of CUSIPs to filter by
            ids: Single corporate action ID or list of IDs
            limit: Maximum number of results to return (default: 1000, max: 10000)
            sort: Sort order - 'asc' or 'desc' (default: 'asc')
            
        Returns:
            List of corporate action dictionaries containing:
                - id: Corporate action ID
                - corporate_action_type: Type of action
                - ca_sub_type: Sub-type of the action
                - initiating_symbol: Symbol initiating the action
                - initiating_original_cusip: Original CUSIP
                - target_symbol: Target symbol (for mergers, etc.)
                - target_original_cusip: Target CUSIP
                - declaration_date: Date action was declared
                - ex_date: Ex-dividend/ex-date
                - record_date: Record date
                - payable_date: Payment date (for dividends)
                - cash: Cash amount (for dividends)
                - old_rate: Old rate (for splits)
                - new_rate: New rate (for splits)
                
        Examples:
            # Get all dividends for AAPL in 2024
            dividends = client.get_corporate_actions(
                ca_types='dividend',
                symbols='AAPL',
                start='2024-01-01',
                end='2024-12-31'
            )
            for div in dividends:
                print(f"Ex-date: {div['ex_date']}, Amount: ${div['cash']}")
            
            # Get all stock splits
            splits = client.get_corporate_actions(
                ca_types='split',
                start='2024-01-01'
            )
            for split in splits:
                print(f"{split['initiating_symbol']}: {split['old_rate']}:{split['new_rate']}")
            
            # Get multiple types of actions
            actions = client.get_corporate_actions(
                ca_types='dividend,split,merger',
                symbols=['AAPL', 'TSLA', 'MSFT'],
                start='2024-01-01'
            )
            
            # Get upcoming corporate actions
            from datetime import datetime, timedelta
            today = datetime.now().strftime('%Y-%m-%d')
            future = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            upcoming = client.get_corporate_actions(
                start=today,
                end=future,
                sort='asc'
            )
            
            # Get all actions for a specific symbol
            all_actions = client.get_corporate_actions(
                symbols='AAPL',
                start='2020-01-01',
                limit=100
            )
        """
        params = {
            'limit': limit,
            'sort': sort.lower()
        }
        
        if ca_types:
            params['ca_types'] = ca_types.lower()
        if start:
            params['since'] = start
        if end:
            params['until'] = end
        
        # Handle symbols - can be string or list
        if symbols:
            if isinstance(symbols, list):
                params['symbols'] = ','.join(s.upper() for s in symbols)
            else:
                params['symbols'] = symbols.upper()
        
        # Handle CUSIPs - can be string or list
        if cusips:
            if isinstance(cusips, list):
                params['cusips'] = ','.join(cusips)
            else:
                params['cusips'] = cusips
        
        # Handle IDs - can be string or list
        if ids:
            if isinstance(ids, list):
                params['ids'] = ','.join(ids)
            else:
                params['ids'] = ids
        
        return self._make_request('GET', '/corporate_actions/announcements', params=params)


def main():
    """Example usage of AlpacaClient"""
    try:
        # Initialize client
        client = AlpacaClient()
        
        print("=" * 60)
        print("ACCOUNT INFORMATION")
        print("=" * 60)
        account = client.get_account_info()
        print(f"Account Status: {account.get('status')}")
        print(f"Currency: {account.get('currency')}")
        print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"Equity: ${float(account.get('equity', 0)):,.2f}")
        print(f"Pattern Day Trader: {account.get('pattern_day_trader')}")
        print(f"Day Trade Count: {account.get('daytrade_count')}")
        print()
        
        print("=" * 60)
        print("PORTFOLIO HISTORY")
        print("=" * 60)
        try:
            history = client.get_portfolio_history(period='1W', timeframe='1D')
            print(f"Base Value: ${float(history.get('base_value', 0)):,.2f}")
            print(f"Timeframe: {history.get('timeframe')}")
            print(f"\nRecent equity values:")
            
            timestamps = history.get('timestamp', [])
            equity_values = history.get('equity', [])
            pnl_pct = history.get('profit_loss_pct', [])
            
            # Show last 5 data points
            for i in range(max(0, len(timestamps) - 5), len(timestamps)):
                from datetime import datetime
                date = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
                equity = equity_values[i]
                pnl = pnl_pct[i] * 100 if i < len(pnl_pct) and pnl_pct[i] else 0
                print(f"  {date}: ${equity:,.2f} ({pnl:+.2f}%)")
        except Exception as e:
            print(f"Error fetching portfolio history: {e}")
        print()
        
        print("=" * 60)
        print("ASSET INFORMATION")
        print("=" * 60)
        try:
            # Get specific asset
            asset = client.get_asset('AAPL')
            print(f"Symbol: {asset.get('symbol')}")
            print(f"Name: {asset.get('name')}")
            print(f"Exchange: {asset.get('exchange')}")
            print(f"Asset Class: {asset.get('class')}")
            print(f"Tradable: {asset.get('tradable')}")
            print(f"Marginable: {asset.get('marginable')}")
            print(f"Shortable: {asset.get('shortable')}")
            print(f"Fractionable: {asset.get('fractionable')}")
            
            # Get count of active stocks
            print(f"\nQuerying all active US equities...")
            all_stocks = client.get_all_assets(status='active', asset_class='us_equity')
            print(f"Total active US stocks: {len(all_stocks)}")
            
            # Get crypto assets
            crypto_assets = client.get_all_assets(asset_class='crypto', status='active')
            print(f"Total active crypto assets: {len(crypto_assets)}")
            if crypto_assets:
                print("Available cryptocurrencies:")
                for crypto in crypto_assets[:5]:
                    print(f"  - {crypto.get('symbol')}: {crypto.get('name')}")
        except Exception as e:
            print(f"Error fetching asset info: {e}")
        print()
        
        print("=" * 60)
        print("CORPORATE ACTIONS")
        print("=" * 60)
        try:
            # Get recent dividends for AAPL
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            dividends = client.get_corporate_actions(
                ca_types='dividend',
                symbols='AAPL',
                start=start_date,
                end=end_date,
                limit=5
            )
            
            if dividends:
                print(f"Recent AAPL dividends (last year):")
                for div in dividends[:5]:
                    ex_date = div.get('ex_date', 'N/A')
                    cash = div.get('cash', 0)
                    print(f"  Ex-date: {ex_date}, Amount: ${cash}")
            else:
                print("No recent dividends found for AAPL")
            
            # Get recent stock splits
            print(f"\nRecent stock splits (last year):")
            splits = client.get_corporate_actions(
                ca_types='split',
                start=start_date,
                limit=10
            )
            
            if splits:
                for split in splits[:5]:
                    symbol = split.get('initiating_symbol', 'N/A')
                    old_rate = split.get('old_rate', 1)
                    new_rate = split.get('new_rate', 1)
                    ex_date = split.get('ex_date', 'N/A')
                    print(f"  {symbol}: {old_rate}:{new_rate} split on {ex_date}")
            else:
                print("  No recent stock splits found")
                
        except Exception as e:
            print(f"Error fetching corporate actions: {e}")
        print()
        
        print("=" * 60)
        print("ALL POSITIONS")
        print("=" * 60)
        positions = client.get_all_positions()
        
        if positions:
            for pos in positions:
                print(f"\nSymbol: {pos.get('symbol')}")
                print(f"  Quantity: {pos.get('qty')}")
                print(f"  Avg Entry Price: ${float(pos.get('avg_entry_price', 0)):,.2f}")
                print(f"  Current Price: ${float(pos.get('current_price', 0)):,.2f}")
                print(f"  Market Value: ${float(pos.get('market_value', 0)):,.2f}")
                print(f"  Unrealized P/L: ${float(pos.get('unrealized_pl', 0)):,.2f} ({float(pos.get('unrealized_plpc', 0)) * 100:.2f}%)")
                print(f"  Side: {pos.get('side')}")
        else:
            print("No open positions")
        print()
        
        print("=" * 60)
        print("RECENT ORDERS")
        print("=" * 60)
        orders = client.get_orders(status='all', limit=10)
        
        if orders:
            for order in orders:
                print(f"\nOrder ID: {order.get('id')}")
                print(f"  Symbol: {order.get('symbol')}")
                print(f"  Side: {order.get('side')}")
                print(f"  Quantity: {order.get('qty')}")
                print(f"  Type: {order.get('type')}")
                print(f"  Status: {order.get('status')}")
                print(f"  Filled: {order.get('filled_qty', 0)}/{order.get('qty')}")
                if order.get('filled_avg_price'):
                    print(f"  Avg Fill Price: ${float(order.get('filled_avg_price')):,.2f}")
        else:
            print("No orders found")
        print()
        
        # Example: Cancel all orders (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("CANCEL ALL ORDERS")
        # print("=" * 60)
        # try:
        #     canceled = client.cancel_all_orders()
        #     print(f"Canceled {len(canceled)} orders")
        # except Exception as e:
        #     print(f"Error canceling orders: {e}")
        # print()
        
        # Example: Place a market order (UNCOMMENT TO TEST - BE CAREFUL!)
        # print("=" * 60)
        # print("PLACE STOCK ORDER EXAMPLE")
        # print("=" * 60)
        # try:
        #     order = client.place_stock_order(
        #         symbol='AAPL',
        #         side='buy',
        #         quantity=1,
        #         order_type='market',
        #         time_in_force='day'
        #     )
        #     print(f"Order placed: {order.get('id')}")
        #     print(f"Status: {order.get('status')}")
        # except Exception as e:
        #     print(f"Error placing order: {e}")
        
        # Example: Get option contracts (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("OPTION CONTRACTS")
        # print("=" * 60)
        # try:
        #     contracts = client.get_option_contracts(
        #         underlying_symbol='AAPL',
        #         type='call',
        #         status='active',
        #         limit=5
        #     )
        #     for contract in contracts[:5]:
        #         print(f"\nSymbol: {contract['symbol']}")
        #         print(f"  Strike: ${contract['strike_price']}")
        #         print(f"  Expiration: {contract['expiration_date']}")
        #         print(f"  Type: {contract['type']}")
        # except Exception as e:
        #     print(f"Error fetching contracts: {e}")
        # print()
        
        # Example: Get option quote (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("OPTION QUOTE")
        # print("=" * 60)
        # try:
        #     quote = client.get_option_latest_quote('AAPL250117C00200000')
        #     print(f"Bid: ${quote.get('bp')} x {quote.get('bs')}")
        #     print(f"Ask: ${quote.get('ap')} x {quote.get('as')}")
        #     spread = float(quote.get('ap', 0)) - float(quote.get('bp', 0))
        #     print(f"Spread: ${spread:.2f}")
        # except Exception as e:
        #     print(f"Error fetching quote: {e}")
        # print()
        
        # Example: Get option snapshot with Greeks (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("OPTION SNAPSHOT")
        # print("=" * 60)
        # try:
        #     snapshot = client.get_option_snapshot('AAPL250117C00200000')
        #     print(f"Implied Volatility: {snapshot.get('implied_volatility')}")
        #     greeks = snapshot.get('greeks', {})
        #     print(f"Delta: {greeks.get('delta')}")
        #     print(f"Gamma: {greeks.get('gamma')}")
        #     print(f"Theta: {greeks.get('theta')}")
        #     print(f"Vega: {greeks.get('vega')}")
        # except Exception as e:
        #     print(f"Error fetching snapshot: {e}")
        # print()
        
        # Example: Get stock bars (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("STOCK BARS (OHLCV)")
        # print("=" * 60)
        # try:
        #     bars = client.get_stock_bars('AAPL', days=5, timeframe='1Day', limit=5)
        #     for bar in bars[-5:]:
        #         print(f"\nDate: {bar['t']}")
        #         print(f"  Open: ${bar['o']:.2f}")
        #         print(f"  High: ${bar['h']:.2f}")
        #         print(f"  Low: ${bar['l']:.2f}")
        #         print(f"  Close: ${bar['c']:.2f}")
        #         print(f"  Volume: {bar['v']:,}")
        # except Exception as e:
        #     print(f"Error fetching bars: {e}")
        # print()
        
        # Example: Get latest stock quote (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("LATEST STOCK QUOTE")
        # print("=" * 60)
        # try:
        #     quote = client.get_stock_latest_quote('AAPL')
        #     print(f"Bid: ${quote['bp']:.2f} x {quote['bs']}")
        #     print(f"Ask: ${quote['ap']:.2f} x {quote['as']}")
        #     spread = quote['ap'] - quote['bp']
        #     print(f"Spread: ${spread:.2f}")
        # except Exception as e:
        #     print(f"Error fetching quote: {e}")
        # print()
        
        # Example: Get stock snapshot (UNCOMMENT TO TEST)
        # print("=" * 60)
        # print("STOCK SNAPSHOT")
        # print("=" * 60)
        # try:
        #     snapshot = client.get_stock_snapshot('AAPL')
        #     latest = snapshot['latestTrade']
        #     daily = snapshot['dailyBar']
        #     prev = snapshot['prevDailyBar']
        #     
        #     print(f"Last Trade: ${latest['p']:.2f} ({latest['s']} shares)")
        #     print(f"Daily High: ${daily['h']:.2f}")
        #     print(f"Daily Low: ${daily['l']:.2f}")
        #     print(f"Daily Volume: {daily['v']:,}")
        #     
        #     change = ((latest['p'] - prev['c']) / prev['c']) * 100
        #     print(f"Change from prev close: {change:+.2f}%")
        # except Exception as e:
        #     print(f"Error fetching snapshot: {e}")
        # print()
        
        # Example: Get specific position (uncomment and replace SYMBOL with actual ticker)
        # print("=" * 60)
        # print("SPECIFIC POSITION")
        # print("=" * 60)
        # try:
        #     position = client.get_open_position('AAPL')
        #     print(f"Symbol: {position.get('symbol')}")
        #     print(f"Quantity: {position.get('qty')}")
        #     print(f"Market Value: ${float(position.get('market_value', 0)):,.2f}")
        #     print(f"Unrealized P/L: ${float(position.get('unrealized_pl', 0)):,.2f}")
        # except Exception as e:
        #     print(f"Error getting position: {e}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

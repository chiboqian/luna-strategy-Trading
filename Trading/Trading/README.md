# Trading Module

Python module for interacting with trading platforms.

## Alpaca Markets Client

### Setup

The Alpaca client requires API credentials in your `.env` file:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Optional, defaults to paper trading
```

### Usage

```python
from Trading.alpaca_client import AlpacaClient

# Initialize client (reads from .env)
client = AlpacaClient()

# Get account information
account = client.get_account_info()
print(f"Cash: ${account['cash']}")
print(f"Portfolio Value: ${account['portfolio_value']}")

# Get all positions
positions = client.get_all_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']}")

# Get specific position
position = client.get_open_position('AAPL')
print(f"AAPL Position: {position['qty']} shares")
print(f"Unrealized P/L: ${position['unrealized_pl']}")
```

### Available Functions

- **get_account_info()** – View balance, margin, and account status
- **get_all_positions()** – List all held assets
- **get_open_position(symbol)** – Detailed info on a specific position
- **close_position(symbol, qty, percentage)** – Close position fully or partially
- **close_all_positions(cancel_orders)** – Close all open positions

### Running the Example

```bash
source venv/bin/activate
python Trading/alpaca_client.py
```

### Dependencies

Add to `requirements.txt`:
```
requests>=2.31.0
python-dotenv>=1.0.0
```

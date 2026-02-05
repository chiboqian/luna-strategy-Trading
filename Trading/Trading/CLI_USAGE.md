# Alpaca Trading CLI

Command-line interface for interacting with Alpaca Markets API.

## Installation

```bash
cd Trading
chmod +x alpaca_cli.py
```

## Usage

```bash
python alpaca_cli.py <command> [options]
```

Or if executable:
```bash
./alpaca_cli.py <command> [options]
```

## Available Commands

### Account Information
```bash
# View account balance, buying power, and status
python alpaca_cli.py account
```

### Positions
```bash
# List all positions
python alpaca_cli.py positions

# Get specific position
python alpaca_cli.py positions -s AAPL
```

### Orders
```bash
# List all orders
python alpaca_cli.py orders

# List open orders only
python alpaca_cli.py orders --status open

# Filter by symbols
python alpaca_cli.py orders --symbols AAPL,TSLA
```

### Place Orders
```bash
# Market buy order
python alpaca_cli.py buy AAPL 10

# Limit buy order
python alpaca_cli.py buy AAPL 10 --type limit --limit-price 150.00

# Market sell order
python alpaca_cli.py sell AAPL 5

# Limit sell with extended hours
python alpaca_cli.py sell TSLA 2 --type limit --limit-price 250.00 --extended-hours

# Stop order
python alpaca_cli.py sell AAPL 10 --type stop --stop-price 145.00

# GTC order
python alpaca_cli.py buy MSFT 5 --type limit --limit-price 350.00 --tif gtc
```

### Cancel Orders
```bash
# Cancel specific order
python alpaca_cli.py cancel --order-id <order-id>

# Cancel all open orders
python alpaca_cli.py cancel --all
```

### Close Positions
```bash
# Close specific position
python alpaca_cli.py close --symbol AAPL

# Close partial position
python alpaca_cli.py close --symbol AAPL --quantity 5

# Close percentage of position
python alpaca_cli.py close --symbol AAPL --percentage 50

# Close all positions
python alpaca_cli.py close --all

# Close all positions and cancel orders
python alpaca_cli.py close --all --cancel-orders
```

### Market Data

#### Historical Bars (OHLCV)
```bash
# Get daily bars for last 5 days
python alpaca_cli.py bars AAPL

# Get hourly bars
python alpaca_cli.py bars AAPL --timeframe 1Hour --days 2

# Get 5-minute bars
python alpaca_cli.py bars AAPL --timeframe 5Min --days 1 --limit 20

# Custom date range
python alpaca_cli.py bars AAPL --start 2024-01-01 --end 2024-01-31 --timeframe 1Day
```

#### Real-time Quotes
```bash
# Single symbol
python alpaca_cli.py quote AAPL

# Multiple symbols
python alpaca_cli.py quote AAPL,TSLA,MSFT
```

#### Stock Snapshots
```bash
# Single symbol snapshot
python alpaca_cli.py snapshot AAPL

# Multiple symbols
python alpaca_cli.py snapshot AAPL,TSLA,MSFT,GOOGL
```

### Asset Information
```bash
# Get specific asset
python alpaca_cli.py assets -s AAPL

# List all active stocks
python alpaca_cli.py assets --status active --asset-class us_equity --limit 50

# List cryptocurrencies
python alpaca_cli.py assets --asset-class crypto

# Filter by exchange
python alpaca_cli.py assets --exchange NASDAQ --status active --limit 20
```

### Portfolio History
```bash
# Last week
python alpaca_cli.py history --period 1W

# Last month with daily data
python alpaca_cli.py history --period 1M --timeframe 1D

# Intraday today
python alpaca_cli.py history --period 1D --timeframe 15Min

# Custom date range
python alpaca_cli.py history --start 2024-01-01 --end 2024-01-31 --timeframe 1D

# Show last 10 data points
python alpaca_cli.py history --period 1M --limit 10
```

### Corporate Actions
```bash
# Get dividends for AAPL
python alpaca_cli.py corporate-actions --types dividend --symbols AAPL

# Get stock splits
python alpaca_cli.py corporate-actions --types split --start 2024-01-01

# Get all actions for multiple symbols
python alpaca_cli.py corporate-actions --symbols AAPL,TSLA,MSFT --start 2024-01-01

# Get multiple types
python alpaca_cli.py corporate-actions --types dividend,split,merger --start 2024-01-01 --limit 100
```

### Options
```bash
# Get option contracts for AAPL
python alpaca_cli.py options AAPL

# Filter by expiration date
python alpaca_cli.py options AAPL --exp-gte 2024-01-01 --exp-lte 2024-12-31

# Filter by strike price
python alpaca_cli.py options AAPL --strike-gte 150 --strike-lte 160

# Get calls only
python alpaca_cli.py options AAPL --option-type call --limit 50

# Combine filters
python alpaca_cli.py options SPY --option-type put --strike-gte 400 --strike-lte 450 --exp-gte 2024-01-01
```

## Examples

### Check Account and Positions
```bash
# View account summary
python alpaca_cli.py account

# Check all positions
python alpaca_cli.py positions

# Check specific position
python alpaca_cli.py positions -s AAPL
```

### Simple Trading Workflow
```bash
# 1. Check current price
python alpaca_cli.py quote AAPL

# 2. Place market buy order
python alpaca_cli.py buy AAPL 10

# 3. Check order status
python alpaca_cli.py orders --status open

# 4. Check position
python alpaca_cli.py positions -s AAPL

# 5. Place limit sell order
python alpaca_cli.py sell AAPL 10 --type limit --limit-price 160.00

# 6. Cancel order if needed
python alpaca_cli.py cancel --order-id <order-id>
```

### Market Analysis
```bash
# Get historical data
python alpaca_cli.py bars AAPL --days 30 --timeframe 1Day

# Get latest snapshot
python alpaca_cli.py snapshot AAPL

# Check corporate actions
python alpaca_cli.py corporate-actions --symbols AAPL --types dividend,split
```

## Help

For help on any command:
```bash
python alpaca_cli.py <command> --help
```

For general help:
```bash
python alpaca_cli.py --help
```

## Environment Setup

Make sure your `.env` file contains:
```
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

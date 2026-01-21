# Utility Scripts

This directory contains scripts for executing trading workflows, analyzing stocks, and managing the system.

> **Note:** The recommendation engine (fetching buy/sell signals) is currently disconnected from the database. Scripts like `pick_recommendations.py`, `get_buy_list.py`, and `get_sell_list.py` will return empty lists until a new data source is configured.

## Trading Execution

### Orchestrators
These scripts run the full workflow: fetching recommendations and executing orders.

*   **`execute_buys.py`**: Executes buy orders for top recommended stocks.
    *   Usage: `python util/execute_buys.py --mid-price --market`
*   **`execute_sells.py`**: Executes short sell orders for top sell recommendations.
    *   Usage: `python util/execute_sells.py --mid-price`

### Granular Scripts
These scripts allow for individual steps of the workflow.

*   **`get_buy_list.py`**: Returns a JSON list of top buy recommendations.
    *   Usage: `python util/get_buy_list.py --limit 2`
*   **`get_sell_list.py`**: Returns a JSON list of top sell recommendations.
    *   Usage: `python util/get_sell_list.py --limit 1`
*   **`execute_single_buy.py`**: Executes a buy order for a single symbol.
    *   Usage: `python util/execute_single_buy.py AAPL --dollars 1000 --mid-price`
*   **`execute_single_sell.py`**: Executes a short sell order for a single symbol.
    *   Usage: `python util/execute_single_sell.py TSLA --dollars 1000 --mid-price`

### Position Management & Monitoring
*   **`close_old_positions.py`**: Closes positions that have been held longer than a specified number of days.
    *   Usage: `python util/close_old_positions.py --days 4 --json`
*   **`monitor_orders.py`**: Monitors a list of orders for completion.
    *   Usage: `python util/monitor_orders.py --orders '["id1", "id2"]' --json`
*   **`account_summary.py`**: Retrieves account summary and open positions.
    *   Usage: `python util/account_summary.py --json`
*   **`manage_options.py`**: Monitors and manages option positions (Take Profit, Stop Loss, DTE).
    *   Reads defaults from `config/Options.yaml` (section `options.management`).
    *   Usage: `python util/manage_options.py --dry-run` (uses config defaults)
    *   Usage: `python util/manage_options.py --tp 0.5 --sl 1.0 --dte 7` (overrides config)

## Analysis & Research
*   **`pick_recommendations.py`**: Analyzes recommendations to pick the best stocks to buy/sell.

### Option Strategy Scanner
Detailed documentation for the Option Strategy Scanner logic, configuration, and filtering pipeline.
---

## Options Strategies

### Synthetic Long (`synthetic_long.py`)

Creates a Synthetic Long position with protective put for defined-risk stock exposure.

**Structure:**
- Buy ATM Call + Sell ATM Put + Buy OTM Put (protection)

```bash
# Basic usage
python util/synthetic_long.py SPY --quantity 1 --dry-run

# With protection level
python util/synthetic_long.py AAPL --protection-pct 5 --days 30

# Dollar amount (auto-calculates quantity)
python util/synthetic_long.py QQQ --amount 5000
```

---

### Bull Put Spread Strategy
Detailed documentation for the Bull Put Spread strategy, including delta selection, quality filters, and position management.

**Configuration (`config/Options.yaml`):**
```yaml
options:
  management:
    take_profit_pct: 0.50
    stop_loss_pct: 1.00
    close_dte: 7
```
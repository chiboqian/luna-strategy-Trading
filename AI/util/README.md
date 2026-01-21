# Utility Scripts

This directory contains scripts for executing trading workflows, analyzing stocks, and managing the system.

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

## Analysis & Research
*   **`pick_recommendations.py`**: Analyzes the session database to pick the best stocks to buy/sell.
*   **`research.py`**: Performs research on stocks.
*   **`deep_dive.py`**: Performs deep dive analysis.
*   **`review_stock.py`**: Reviews a specific stock.

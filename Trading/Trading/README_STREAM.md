# Alpaca Stream Framework

A real-time market data monitoring and automation framework built on the Alpaca API. It listens to streaming quotes and bars for stocks and options, evaluates user-defined rules, and executes shell commands (e.g., trading scripts) when conditions are met.

## Features

*   **Real-time Streaming**: Subscribes to Alpaca's `sip` (stocks) and `opra` (options) feeds via `alpaca-py`.
*   **Dynamic Rule Engine**: Supports complex conditions using live price data and calculated technical indicators.
*   **Technical Indicators**: Real-time calculation of SMA, VWAP, StdDev, Z-Score, and RSI on streaming 1-minute bars.
*   **Hot-Reloading**: Automatically detects changes in configuration files (or directories) and updates rules without restarting.
*   **Action Execution**: Triggers external scripts or commands when rules are matched.
*   **Data Recording**: Logs incoming quotes and bars to CSV for future analysis (`trading_logs/market_data/`).
*   **Persistence**: Saves/loads bar history to maintain indicator continuity across restarts.

## Prerequisites

*   Python 3.8+
*   Alpaca Account (Live or Paper)
*   Environment Variables set in `.env`:
    ```bash
    ALPACA_API_KEY=your_api_key
    ALPACA_API_SECRET=your_api_secret
    ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets
    ```

## Installation

Ensure dependencies are installed:

```bash
pip install alpaca-py pyyaml python-dotenv
```

## Usage

Run the framework pointing to your configuration file or directory:

```bash
# Run with default config (Trading/config/streaming_rules.yaml)
python Trading/Trading/stream_framework.py

# Run with specific config file
python Trading/Trading/stream_framework.py --config my_rules.yaml

# Run with config directory (loads all .yaml files in directory)
python Trading/Trading/stream_framework.py --config Trading/config/streaming_rules/
```

### CLI Arguments

*   `--config`: Path to YAML config file or directory (default: `Trading/config/streaming_rules.yaml`).
*   `--log-dir`: Directory for logs (default: `trading_logs/streaming`).
*   `--log-file`: Log filename (default: `stream_framework.log`).

## Configuration

Rules are defined in YAML format. You can use a single file or a directory of files.

### Rule Structure

```yaml
rules:
  - name: "SPY Uptrend Buy"
    symbol: "SPY"
    asset_class: "stock"      # 'stock' or 'option'
    trigger: "bar"            # 'quote' (real-time bid/ask) or 'bar' (1-min candle close)
    conditions:
      - field: "close"
        operator: ">"
        value: "sma_50"       # Can compare against indicators
      - field: "rsi_14"
        operator: "<"
        value: 70.0           # Or static values
    action:
      command: "python Trading/Trading/alpaca_buy_cli.py {symbol} 1000"
    cooldown: 900             # Seconds to wait before re-triggering
    one_off: false            # If true, removes rule after firing once
```

### Supported Fields & Indicators

*   **Raw Data**: `ask_price`, `bid_price`, `close` (bar only), `volume` (bar only).
*   **Indicators** (Bar trigger only, calculated on 1-min bars):
    *   `sma_N`: Simple Moving Average (e.g., `sma_50`).
    *   `vwap_N`: Volume Weighted Average Price.
    *   `stddev_N`: Standard Deviation.
    *   `z_score_sma_N`: Z-Score of price relative to SMA `((Price - SMA) / StdDev)`.
    *   `z_score_vwap_N`: Z-Score of price relative to VWAP.
    *   `rsi_N`: Relative Strength Index (e.g., `rsi_14`).

### Operators

*   `>`, `>=`, `<`, `<=`, `==`
*   `abs>`, `abs<` (Absolute value comparison)

## Helper Scripts

*   **Add Option Watch**: Dynamically add a stop-loss rule for an option.
    ```bash
    python Trading/options_trading/add_option_watch.py AAPL230616C00150000 2.50
    ```
*   **Remove Watch**: Remove a rule by symbol or name.
    ```bash
    python Trading/options_trading/remove_watch.py --symbol AAPL
    ```
*   **Migrate Config**: Split a single `streaming_rules.yaml` into a directory structure.
    ```bash
    python Trading/config/migrate_rules.py
    ```
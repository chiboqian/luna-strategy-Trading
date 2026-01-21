# AI Trading System

An automated trading system that uses AI to research, analyze, and execute trades on Alpaca Markets.

## Project Structure

*   **`AI/`**: AI models and CLI tools for generating trading signals.
*   **`Trading/`**: Core trading logic and Alpaca API client wrappers.
*   **`util/`**: Utility scripts for executing workflows, managing positions, and research.
*   **`api/`**: FastAPI interface for triggering system actions.
*   **`config/`**: Configuration files.

## Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    *   Set up your `.env` file with Alpaca API credentials (`ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`).
    *   Review `config/Trading.yaml` for trading parameters.

## Usage

### API
The system provides a REST API to interact with the trading scripts.
See [api/README.md](api/README.md) for details.

To start the API:
```bash
./run_api.sh
```

### CLI
You can run individual scripts from the command line.
See [util/README.md](util/README.md) for details on execution scripts.
See [Trading/CLI_USAGE.md](Trading/CLI_USAGE.md) for Alpaca CLI usage.

### Key Scripts
*   `util/execute_buys.py`: Run the buy workflow.
*   `util/execute_sells.py`: Run the sell workflow.
*   `util/close_old_positions.py`: Close stale positions.

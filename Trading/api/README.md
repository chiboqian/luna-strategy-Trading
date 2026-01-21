# AI Trading API

A FastAPI-based interface for the AI Trading system. This API allows you to trigger trading actions, retrieve recommendations, and manage positions programmatically.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Ensure your `.env` file is configured with Alpaca credentials (optional if passing via API).

## Running the API

You can start the API server using the provided helper script:

```bash
./run_api.sh
```

Or manually using `uvicorn`:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Documentation

Interactive API documentation (Swagger UI) is available at:
`http://localhost:8000/docs`

## Endpoints

### Execution
*   `POST /execute/all`: Run the full trading cycle (buys and sells), merges results, and returns a comprehensive report including email text.
*   `POST /execute/buys`: Run the full buy workflow (fetch recommendations + execute orders).
*   `POST /execute/sells`: Run the full sell workflow.
*   `POST /positions/close-old`: Close positions older than X days.
*   `POST /monitor`: Monitor a list of order IDs until they are filled or max retries are reached.

### Account & Information
*   `POST /account/summary`: Get account summary and list of open positions (includes email-ready text).
*   `GET /trading/keys/{label}`: Retrieve API keys for a specific label (e.g., 'paper', 'live').

### Recommendations
*   `POST /recommendations/buys`: Get a list of top buy recommendations.
*   `POST /recommendations/sells`: Get a list of top sell recommendations.

### Orders
*   `POST /orders/buy`: Execute a single buy order for a specific symbol.
*   `POST /orders/sell`: Execute a single sell order for a specific symbol.

### Market & Logs
*   `POST /market/status`: Check if the market is open. Supports checking if market will be open for at least X minutes.
*   `POST /process/logs`: Process and aggregate logs and recommendation lists.

## Authentication

You can pass Alpaca credentials in the request body for any endpoint to override environment defaults:

```json
{
  "alpaca_api_key": "YOUR_KEY",
  "alpaca_api_secret": "YOUR_SECRET",
  ...other_params
}
```

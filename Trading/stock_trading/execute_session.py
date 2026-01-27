#!/usr/bin/env python3
"""
Execute trades for a trading session.

Can execute based on a Session ID (fetching from D1/R2) or raw recommendations.
Runs execute_buys.py and execute_sells.py and merges the results.
"""

import argparse
import json
import os
import sys
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Try imports that might be missing in basic envs
try:
    import requests
    import boto3
    from dotenv import load_dotenv
except ImportError:
    pass

# Path setup
UTIL_DIR = Path(__file__).parent
BASE_DIR = UTIL_DIR.parent

def load_environment():
    """Load environment variables from .env files."""
    if 'load_dotenv' in globals():
        load_dotenv(BASE_DIR / ".env")
        load_dotenv(BASE_DIR.parent / ".env")

def load_min_score_from_config() -> float:
    cfg_path = BASE_DIR / 'config' / 'Trading.yaml'
    try:
        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            rec = (data.get('recommendations') or {})
            val = rec.get('min_score')
            if isinstance(val, (int, float)):
                return float(val)
    except Exception:
        pass
    return 22.0

class R2Storage:
    """Handles object storage in Cloudflare R2."""

    def __init__(self, account_id: str, access_key: str, secret_key: str, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto"
        )

    def save_json(self, key: str, data: Dict[str, Any]):
        """Saves a JSON object to a specific key."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(data, indent=2),
                ContentType="application/json"
            )
            print(f"💾 Saved results to R2: s3://{self.bucket_name}/{key}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to save to R2: {e}", file=sys.stderr)

def get_session_info(session_id: str) -> Dict[str, Any]:
    """Query Cloudflare D1 for session info."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        return {}

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    sql = "SELECT * FROM sessions WHERE session_id = ? LIMIT 1"
    payload = {
        "sql": sql,
        "params": [session_id]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success") and data.get("result"):
            results = data["result"][0]["results"]
            if results:
                return results[0]
            else:
                return {}
    except Exception as e:
        print(f"Error querying sessions: {e}", file=sys.stderr)
        return {}

def create_orders_table_if_not_exists():
    """Ensures the orders table exists in D1."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        return

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    sql = """
    CREATE TABLE IF NOT EXISTS orders (
        session_id TEXT,
        order_id TEXT PRIMARY KEY,
        client_order_id TEXT,
        symbol TEXT,
        side TEXT,
        qty REAL,
        type TEXT,
        limit_price REAL,
        filled_avg_price REAL,
        status TEXT,
        created_at TEXT,
        updated_at TEXT,
        raw_json TEXT
    );
    """
    
    try:
        response = requests.post(api_url, headers=headers, json={"sql": sql, "params": []})
        response.raise_for_status()
    except Exception as e:
        print(f"Warning: Failed to create orders table: {e}", file=sys.stderr)

def log_order_to_d1(session_id: str, order_data: Dict[str, Any]):
    """Logs a single order to D1."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        print("Missing Cloudflare D1 credentials, skipping order log", file=sys.stderr)
        return

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Extract fields from Alpaca order object
    # Handle both field naming conventions (Alpaca API vs our CLI output)
    order_id = order_data.get('id')
    if not order_id:
        print("⚠️ Order has no ID, skipping D1 log", file=sys.stderr)
        return

    # qty can be 'qty' (Alpaca API) or 'quantity' (our CLI)
    qty = order_data.get('qty') or order_data.get('quantity') or 0
    # type can be 'type' (Alpaca API) or 'order_type' (our CLI)
    order_type = order_data.get('type') or order_data.get('order_type')
    
    created_at = order_data.get('created_at') or datetime.utcnow().isoformat()
    updated_at = order_data.get('updated_at') or datetime.utcnow().isoformat()

    sql = """
    INSERT OR REPLACE INTO orders 
    (session_id, order_id, client_order_id, symbol, side, qty, type, limit_price, filled_avg_price, status, created_at, updated_at, raw_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    params = [
        session_id,
        order_id,
        order_data.get('client_order_id'),
        order_data.get('symbol'),
        order_data.get('side'),
        float(qty) if qty else 0,
        order_type,
        float(order_data.get('limit_price') or 0) if order_data.get('limit_price') else None,
        float(order_data.get('filled_avg_price') or 0) if order_data.get('filled_avg_price') else None,
        order_data.get('status'),
        created_at,
        updated_at,
        json.dumps(order_data)
    ]

    try:
        response = requests.post(api_url, headers=headers, json={"sql": sql, "params": params})
        response.raise_for_status()
        
        result = response.json()
        if not result.get("success"):
            print(f"❌ Failed to log order to D1 (API Error): {result.get('errors')}", file=sys.stderr)
        else:
            print(f"📝 Logged order {order_id} to D1", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to log order to D1: {e}", file=sys.stderr)

def get_session_analysis(session_id: str) -> List[Dict[str, Any]]:
    """Query Cloudflare D1 for symbol analysis."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        return []

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    sql = "SELECT symbol, normalized_rating, conviction_level FROM symbol_analysis WHERE session_id = ?"
    payload = {
        "sql": sql,
        "params": [session_id]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and data.get("result"):
            results = data["result"][0]["results"]
            return results
    except Exception as e:
        print(f"Error querying symbol_analysis: {e}", file=sys.stderr)
        
    return []

def save_execution_results_to_r2(session_id: str, data: Dict[str, Any]):
    """Save execution results to R2 bucket following session structure."""
    if 'boto3' not in globals():
        print("boto3 not installed, skipping R2 upload", file=sys.stderr)
        return

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")

    if not all([account_id, access_key, secret_key, bucket_name]):
        print("⚠️  Missing R2 credentials. Skipping execution log upload.", file=sys.stderr)
        return

    # Determine path based on session info or session_id
    # Try to fetch session info to get the exact folder path used by run_session.py
    session_info = get_session_info(session_id)
    consolidated_path = session_info.get("consolidated_list_path")
    
    if consolidated_path:
        # consolidated_path ex: 2025-01-16/SESSION_ID/consolidated-list.json
        base_path = str(Path(consolidated_path).parent)
    else:
        # Fallback: Parse date from session_id
        # Session ID format: YYYYMMDD-HHMMSS-UUID (e.g., 20260116-112838-80cf75)
        try:
            date_part = session_id.split('-')[0] # 20260116
            date_formatted = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}" # 2026-01-16
            base_path = f"{date_formatted}/{session_id}"
        except Exception:
             # Ultimate fallback
             base_path = f"sessions/{session_id}"

    # Use timestamp to create unique filename to avoid overwriting
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    key = f"{base_path}/execution_results_{timestamp}.json"

    storage = R2Storage(account_id, access_key, secret_key, bucket_name)
    storage.save_json(key, data)

def run_script(script_path: Path, args: list, env_vars: dict = None) -> Dict[str, Any]:
    if not script_path.exists():
        return {"success": False, "error": f"Script not found: {script_path}"}
    
    cmd = ["python3", str(script_path)] + args
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            env=env
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def parse_args():
    parser = argparse.ArgumentParser(description="Execute trading session")
    
    # Session / Input
    parser.add_argument("--session-id", help="AI Session ID to fetch recommendations from")
    parser.add_argument("--recommendations-file", help="JSON file containing recommendations")
    parser.add_argument("--recommendations-json", help="JSON string containing recommendations")
    
    # Trading execution flags
    parser.add_argument("--dollars", type=float, help="Dollar amount per trade")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Buy specific
    parser.add_argument("--market", action="store_true", default=True, help="Use market orders (default)")
    parser.add_argument("--limit", action="store_true", help="Use limit orders instead of market")
    parser.add_argument("--mid-price", action="store_true", help="Use mid price")
    
    # Sell specific
    parser.add_argument("--use-bid", action="store_true", help="Use bid price for sells")
    parser.add_argument("--price-offset", type=float, default=0.0, help="Price offset for sells")
    
    # API credentials
    parser.add_argument('--alpaca-api-key', help='Alpaca API Key')
    parser.add_argument('--alpaca-api-secret', help='Alpaca API Secret')

    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    
    return parser.parse_args()

def generate_email_text(combined_result):
    lines = []
    lines.append("Trading Execution Report")
    lines.append("========================")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Orders: {combined_result['summary']['total']}")
    lines.append(f"Successful:   {combined_result['summary']['successful']}")
    lines.append("")
    
    if combined_result['executions']:
        lines.append("Execution Details:")
        lines.append("------------------")
        for exc in combined_result['executions']:
            status_icon = "✅" if exc.get('success') else "❌"
            action = exc.get('action', 'UNKNOWN')
            symbol = exc.get('symbol', 'UNKNOWN')
            lines.append(f"{status_icon} {action} {symbol}")
            
            if exc.get('success'):
                # Handle nested output from wrapper scripts
                output_data = exc.get('output', {}) if isinstance(exc.get('output'), dict) else exc
                order = output_data.get('order', {}) or exc.get('order', {})
                
                if order:
                    qty = order.get('quantity', 'N/A')
                    price = order.get('limit_price') or order.get('entry_price')
                    cost = order.get('actual_cost') or order.get('notional')
                    
                    details = []
                    if qty != 'N/A': details.append(f"Qty: {qty}")
                    if price: details.append(f"Price: ${float(price):.2f}")
                    if cost: details.append(f"Total: ${float(cost):.2f}")
                    
                    if details:
                        lines.append(f"   {', '.join(details)}")

                if exc.get('status') == 'simulated' or output_data.get('status') == 'simulated':
                    lines.append(f"   (Dry Run)")
            else:
                error = exc.get('error')
                if not error:
                    output_data = exc.get('output')
                    if isinstance(output_data, dict):
                        error = output_data.get('error')
                    elif isinstance(output_data, str) and output_data.strip():
                        error = output_data.strip()
                
                if not error:
                     if exc.get('stderr'):
                         error = exc.get('stderr')
                     elif exc.get('raw_output'):
                         error = exc.get('raw_output')
                         
                if not error:
                    error = 'Unknown error'
                    
                lines.append(f"   Error: {error}")
        lines.append("")

    if combined_result['recommendations']:
        lines.append("Top Recommendations (Actioned):")
        lines.append("-------------------------------")
        for rec in combined_result['recommendations']:
            symbol = rec.get('symbol')
            action = "BUY" if "BUY" in rec.get('recommendation', '').upper() else "SELL"
            score = rec.get('score', 0)
            lines.append(f"- {action} {symbol} (Score: {score})")
            
    # List other analyzed symbols that were not actioned (e.g. low score)
    if combined_result.get('all_recommendations'):
        
        # Identify actioned (symbol, recommendation) pairs to exclude
        actioned_keys = set()
        for rec in combined_result.get('recommendations', []):
            actioned_keys.add((rec.get('symbol'), rec.get('recommendation')))
            
        others = []
        for rec in combined_result['all_recommendations']:
            key = (rec.get('symbol'), rec.get('recommendation'))
            if key not in actioned_keys:
                others.append(rec)
                
        if others:
            lines.append("")
            lines.append("Other Analyzed Symbols (Not Actioned):")
            lines.append("--------------------------------------")
            # Sort by score descending
            others.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            for rec in others:
                symbol = rec.get('symbol')
                action = "BUY" if "BUY" in rec.get('recommendation', '').upper() else "SELL"
                score = rec.get('score', 0)
                lines.append(f"- {action} {symbol} (Score: {score})")
    
    return "\n".join(lines)

def main():
    args = parse_args()
    load_environment()

    # Setup Env Vars for subprocesses
    env_vars = {}
    if args.alpaca_api_key:
        env_vars["ALPACA_API_KEY"] = args.alpaca_api_key
    if args.alpaca_api_secret:
        env_vars["ALPACA_API_SECRET"] = args.alpaca_api_secret

    recommendations_payload = None

    # 1. Handle Session ID
    if args.session_id:
        if 'requests' not in globals():
             print(json.dumps({"error": "Missing dependencies (requests) for session handling"}))
             sys.exit(1)
        
        min_score = load_min_score_from_config()
        analysis_results = get_session_analysis(args.session_id)
        
        if not analysis_results:
             print(json.dumps({"error": "No analysis found for session or session not found"}))
             sys.exit(0) # Not necessarily an error, just nothing to trade
             
        # Normalize and aggregate
        # Key: (symbol, rating), Value: sum(conviction)
        aggregated = {}
        
        for row in analysis_results:
            symbol = row.get("symbol")
            rating = row.get("normalized_rating")
            conviction = row.get("conviction_level")
            
            if not symbol or not rating:
                continue
                
            try:
                conviction = float(conviction)
            except (ValueError, TypeError):
                conviction = 0.0
                
            key = (symbol, rating)
            if key not in aggregated:
                aggregated[key] = 0.0
            aggregated[key] += conviction
            
        recommendations_payload = []
        for (symbol, rating), total_conviction in aggregated.items():
            # Pass ALL normalized findings to the subprocesses. 
            # execute_buys.py/execute_sells.py will handle filtering by min_score for execution,
            # but preserve the full list for reporting.
            recommendations_payload.append({
                "symbol": symbol,
                "ticker": symbol,
                "analysis_rating": rating,
                "conviction_level": total_conviction,
                "action": rating # For consistency
            })

    # 2. Handle direct recommendations overrides
    if args.recommendations_json:
        recommendations_payload = json.loads(args.recommendations_json)
    elif args.recommendations_file:
        with open(args.recommendations_file, 'r') as f:
            recommendations_payload = json.load(f)

    # 3. Execute Buys
    buy_script = UTIL_DIR / "execute_buys.py"
    buy_args = []
    if args.dollars: buy_args.extend(["--dollars", str(args.dollars)])
    # Use market orders by default, unless --limit is specified
    if args.market and not args.limit: buy_args.append("--market")
    if args.mid_price: buy_args.append("--mid-price")
    if args.dry_run: buy_args.append("--dry-run")
    if args.verbose: buy_args.append("--verbose")
    if recommendations_payload:
        buy_args.extend(["--json-payload", json.dumps(recommendations_payload)])
    buy_args.append("--json")
    
    buy_result = run_script(buy_script, buy_args, env_vars)

    # 4. Execute Sells
    sell_script = UTIL_DIR / "execute_sells.py"
    sell_args = []
    if args.dollars: sell_args.extend(["--dollars", str(args.dollars)])
    # Use market orders by default, unless --limit is specified
    if args.market and not args.limit: sell_args.append("--market")
    if args.mid_price: sell_args.append("--mid-price")
    if args.use_bid: sell_args.append("--use-bid")
    if args.price_offset != 0.0: sell_args.extend(["--price-offset", str(args.price_offset)])
    if args.dry_run: sell_args.append("--dry-run")
    if args.verbose: sell_args.append("--verbose")
    if recommendations_payload:
        sell_args.extend(["--json-payload", json.dumps(recommendations_payload)])
    sell_args.append("--json")
    
    sell_result = run_script(sell_script, sell_args, env_vars)

    # 5. Merge Results
    combined_result = {
        "recommendations": [],
        "all_recommendations": [],
        "executions": [],
        "orders": [],
        "summary": {"total": 0, "successful": 0}
    }
    
    for res in [buy_result, sell_result]:
        if res.get("stdout"):
            try:
                data = json.loads(res["stdout"])
                combined_result["recommendations"].extend(data.get("recommendations", []))
                combined_result["all_recommendations"].extend(data.get("all_recommendations", []))
                combined_result["executions"].extend(data.get("executions", []))
                combined_result["summary"]["total"] += data.get("summary", {}).get("total", 0)
                combined_result["summary"]["successful"] += data.get("summary", {}).get("successful", 0)
                
                for exc in data.get("executions", []):
                    if exc.get("success"):
                        output_data = exc.get('output', {}) if isinstance(exc.get('output'), dict) else exc
                        order = output_data.get('order', {}) or exc.get('order', {})
                        if order.get('id'):
                            combined_result["orders"].append(order.get('id'))
            except json.JSONDecodeError:
                pass

    # 5b. Log Orders to D1 if Session ID is present
    if args.session_id and combined_result['executions']:
        # Ensure requests is available
        if 'requests' in globals():
            create_orders_table_if_not_exists()
            for exc in combined_result['executions']:
                if exc.get("success"):
                    output_data = exc.get('output', {}) if isinstance(exc.get('output'), dict) else exc
                    # Handle both structures (direct order or nested in output)
                    order = output_data.get('order', {}) or exc.get('order', {})
                    # Only log if it has a valid ID (implies a real/paper order, not just simulation text)
                    if isinstance(order, dict) and order.get('id'):
                        log_order_to_d1(args.session_id, order)

    # 6. Generate Email Text
    combined_result["email_text"] = generate_email_text(combined_result)

    # 7. Upload to R2 if session ID is present
    if args.session_id:
        save_execution_results_to_r2(args.session_id, combined_result)

    if args.json:
        print(json.dumps(combined_result))
    else:
        # If not JSON mode, print the report text
        print(combined_result["email_text"])

if __name__ == "__main__":
    main()

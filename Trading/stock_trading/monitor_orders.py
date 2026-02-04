#!/usr/bin/env python3
"""
Order Monitoring Program

Monitors orders provided via JSON list or session ID and updates status
by polling Alpaca every interval seconds. Cancels orders that remain
unfilled after max retries.

Configuration sources (precedence high→low):
1) CLI flags
2) config/Trading.yaml (monitoring section)
3) Hardcoded defaults

Usage:
  python util/monitor_orders.py --orders '["id1", "id2"]'
  python util/monitor_orders.py --session-id 20260116-112838-80cf75
  python util/monitor_orders.py --orders '["id1"]' --interval 30 --max-retries 5

Behavior:
- Polls order status via Alpaca API
- Updates status and metadata
- Updates D1 database if session-id is provided
- Tracks retry count per order; cancel via Alpaca once retries exceed max
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import yaml

# Try imports that might be missing
try:
    import requests
    import boto3
    from dotenv import load_dotenv
except ImportError:
    pass

# Ensure Trading module import path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Trading'))
from alpaca_client import AlpacaClient

# Path setup
UTIL_DIR = Path(__file__).parent
BASE_DIR = UTIL_DIR.parent

DEFAULTS = {
    'interval_seconds': 60,
    'max_retries': 10,
}


def load_environment():
    """Load environment variables from .env files."""
    if 'load_dotenv' in globals():
        load_dotenv(BASE_DIR / ".env")
        load_dotenv(BASE_DIR.parent / ".env")


def load_monitoring_defaults() -> Dict[str, Any]:
    """Load defaults from config/Trading.yaml if available."""
    config_path = Path(__file__).parent.parent / 'config' / 'Trading.yaml'
    values = DEFAULTS.copy()
    try:
        with open(config_path, 'r') as f:
            doc = yaml.safe_load(f) or {}
            mon = (doc.get('monitoring') or {})
            if 'interval_seconds' in mon:
                values['interval_seconds'] = int(mon['interval_seconds'])
            if 'max_retries' in mon:
                values['max_retries'] = int(mon['max_retries'])
    except Exception:
        pass
    return values


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


def get_orders_by_session_id(session_id: str) -> List[Dict[str, Any]]:
    """Fetch orders from D1 database by session ID."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        print("Missing Cloudflare D1 credentials", file=sys.stderr)
        return []

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    sql = "SELECT order_id, symbol, side, qty, type, limit_price, status, raw_json FROM orders WHERE session_id = ?"
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
        print(f"Error fetching orders from D1: {e}", file=sys.stderr)

    return []


def update_order_in_d1(session_id: str, order_id: str, status: str, filled_qty: Any = None, filled_avg_price: Any = None, raw_json: Dict = None):
    """Update order status in D1 database."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    database_id = os.getenv("CLOUDFLARE_D1_DATABASE_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not all([account_id, database_id, api_token]):
        return False

    api_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    updated_at = datetime.now(timezone.utc).isoformat()

    sql = """
    UPDATE orders 
    SET status = ?, filled_avg_price = ?, updated_at = ?, raw_json = ?
    WHERE session_id = ? AND order_id = ?
    """

    params = [
        status,
        float(filled_avg_price) if filled_avg_price else None,
        updated_at,
        json.dumps(raw_json) if raw_json else None,
        session_id,
        order_id
    ]

    try:
        response = requests.post(api_url, headers=headers, json={"sql": sql, "params": params})
        response.raise_for_status()
        result = response.json()
        return result.get("success", False)
    except Exception as e:
        print(f"Error updating order in D1: {e}", file=sys.stderr)
        return False


def save_monitoring_results_to_r2(session_id: str, data: Dict[str, Any]):
    """Save monitoring results to R2 bucket following session structure."""
    if 'boto3' not in globals():
        print("boto3 not installed, skipping R2 upload", file=sys.stderr)
        return

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")

    if not all([account_id, access_key, secret_key, bucket_name]):
        print("⚠️  Missing R2 credentials. Skipping monitoring log upload.", file=sys.stderr)
        return

    # Determine path based on session info or session_id
    session_info = get_session_info(session_id)
    consolidated_path = session_info.get("consolidated_list_path")
    
    if consolidated_path:
        # consolidated_path ex: 2025-01-16/SESSION_ID/consolidated-list.json
        base_path = str(Path(consolidated_path).parent)
    else:
        # Fallback: Parse date from session_id
        try:
            date_part = session_id.split('-')[0]
            date_formatted = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
            base_path = f"{date_formatted}/{session_id}"
        except Exception:
            base_path = f"sessions/{session_id}"

    # Use timestamp to create unique filename to avoid overwriting
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    key = f"{base_path}/monitoring_results_{timestamp}.json"

    storage = R2Storage(account_id, access_key, secret_key, bucket_name)
    storage.save_json(key, data)


def parse_args() -> argparse.Namespace:
    defaults = load_monitoring_defaults()
    p = argparse.ArgumentParser(description='Monitor and update Alpaca order status')
    
    # Input source - either orders or session-id
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--orders', help='JSON list of order IDs to monitor')
    input_group.add_argument('--session-id', help='Session ID to fetch orders from D1')
    
    p.add_argument('--interval', type=int, default=defaults['interval_seconds'], help='Polling interval in seconds')
    p.add_argument('--max-retries', type=int, default=defaults['max_retries'], help='Max retries before canceling order')
    p.add_argument('-q', '--quiet', action='store_true', help='Suppress non-error output')
    p.add_argument('--json', action='store_true', help='Output final status as JSON')
    return p.parse_args()


def main():
    args = parse_args()
    load_environment()
    
    # Track session_id for D1 updates
    session_id = args.session_id if hasattr(args, 'session_id') else None
    
    # Initialize Alpaca client
    try:
        client = AlpacaClient()
    except Exception as e:
        print(f"Error: Failed to initialize Alpaca client: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize orders - either from --orders or --session-id
    orders = []
    
    if args.session_id:
        if 'requests' not in globals():
            print("Error: Missing requests library for session-id lookup", file=sys.stderr)
            sys.exit(1)
        
        d1_orders = get_orders_by_session_id(args.session_id)
        if not d1_orders:
            print(f"No orders found for session {args.session_id}", file=sys.stderr)
            sys.exit(0)
        
        for d1_order in d1_orders:
            orders.append({
                'order_id': d1_order.get('order_id'),
                'status': d1_order.get('status', 'new'),
                'symbol': d1_order.get('symbol'),
                'side': d1_order.get('side'),
                'qty': d1_order.get('qty'),
                'metadata': {}
            })
        
        if not args.quiet:
            print(f"Found {len(orders)} order(s) for session {args.session_id}", file=sys.stderr)
    else:
        try:
            order_ids = json.loads(args.orders)
            if not isinstance(order_ids, list):
                raise ValueError("orders must be a list")
            for oid in order_ids:
                orders.append({
                    'order_id': oid,
                    'status': 'new',
                    'symbol': None,
                    'metadata': {}
                })
        except Exception as e:
            print(f"Error parsing orders JSON: {e}", file=sys.stderr)
            sys.exit(1)

    poll_statuses = {'accepted', 'new', 'pending', 'open', 'partially_filled'}
    terminal_statuses = {'filled', 'canceled', 'expired', 'rejected'}

    if not args.quiet:
        print(f"Monitoring orders every {args.interval}s (max-retries={args.max_retries})", file=sys.stderr)

    # First, fetch current status from Alpaca for all orders to get filled details
    if not args.quiet:
        print("Fetching initial order status from Alpaca...", file=sys.stderr)
    for order in orders:
        order_id = order['order_id']
        metadata = order['metadata']
        try:
            api_order = client._make_request('GET', f'/v2/orders/{order_id}')
            order['status'] = (api_order.get('status') or order['status']).lower()
            if not order.get('symbol'):
                order['symbol'] = api_order.get('symbol')
            metadata['api_order'] = {
                'updated_at': api_order.get('updated_at'),
                'filled_qty': api_order.get('filled_qty'),
                'filled_avg_price': api_order.get('filled_avg_price'),
                'status': api_order.get('status'),
            }
            if not args.quiet:
                print(f"  {order['symbol'] or order_id}: {order['status']}", file=sys.stderr)
        except Exception as e:
            if not args.quiet:
                print(f"  ⚠️ Failed to fetch {order_id}: {e}", file=sys.stderr)

    try:
        attempt = 0
        exited_due_to_no_open = False
        while attempt < args.max_retries:
            if not args.quiet:
                print(f"Attempt {attempt+1}/{args.max_retries}", file=sys.stderr)
            
            open_orders_exist = False
            
            for order in orders:
                order_id = order['order_id']
                status = (order['status'] or '').lower()
                metadata = order['metadata']
                
                if status in terminal_statuses:
                    continue
                
                open_orders_exist = True
                
                # Poll order status from Alpaca
                api_order = None
                try:
                    api_order = client._make_request('GET', f'/v2/orders/{order_id}')
                except Exception as e:
                    if not args.quiet:
                        print(f"⚠️ Poll failed for {order_id}: {e}", file=sys.stderr)
                    metadata['monitoring_retries'] = int(metadata.get('monitoring_retries', 0)) + 1
                    metadata['last_error'] = str(e)
                    continue

                # Extract latest status
                latest_status = (api_order.get('status') or status).lower()
                order['status'] = latest_status
                if not order['symbol']:
                    order['symbol'] = api_order.get('symbol')
                
                metadata['monitoring_retries'] = int(metadata.get('monitoring_retries', 0))
                metadata['last_checked'] = int(time.time())
                metadata['api_order'] = {
                    'updated_at': api_order.get('updated_at'),
                    'filled_qty': api_order.get('filled_qty'),
                    'filled_avg_price': api_order.get('filled_avg_price'),
                    'status': api_order.get('status'),
                }

                # Update D1 if session_id is present
                if session_id:
                    update_order_in_d1(
                        session_id=session_id,
                        order_id=order_id,
                        status=latest_status,
                        filled_qty=api_order.get('filled_qty'),
                        filled_avg_price=api_order.get('filled_avg_price'),
                        raw_json=api_order
                    )

                if latest_status in terminal_statuses:
                    if not args.quiet:
                        print(f"✓ {order_id} -> {latest_status}", file=sys.stderr)
                    continue

                if latest_status in poll_statuses:
                    metadata['monitoring_retries'] += 1
                    continue

            # Re-check if any orders are still open after processing
            open_orders_exist = any(
                order['status'] not in terminal_statuses 
                for order in orders
            )

            if not open_orders_exist:
                if not args.quiet:
                    print("No open orders remaining. Exiting monitor.", file=sys.stderr)
                exited_due_to_no_open = True
                break

            time.sleep(args.interval)
            attempt += 1
            
        if not exited_due_to_no_open:
            # Cancel all remaining open/pending orders
            for order in orders:
                if order['status'] in poll_statuses:
                    oid = order['order_id']
                    try:
                        client.cancel_order_by_id(oid)
                        order['status'] = 'canceled'
                        order['metadata']['canceled_by_monitor'] = True
                        
                        # Update D1 with canceled status
                        if session_id:
                            update_order_in_d1(
                                session_id=session_id,
                                order_id=oid,
                                status='canceled',
                                raw_json={'status': 'canceled', 'canceled_by_monitor': True}
                            )
                        
                        if not args.quiet:
                            print(f"⛔ Canceled {oid} due to max retries")
                    except Exception as e:
                        if not args.quiet:
                            print(f"⚠️ Failed to cancel {oid}: {e}")
            if not args.quiet:
                print(f"Max retries reached ({args.max_retries}). Exiting monitor.")
        
        # Build result data with comprehensive email text
        filled_count = sum(1 for o in orders if o.get('status') == 'filled')
        canceled_count = sum(1 for o in orders if o.get('status') == 'canceled')
        
        email_lines = []
        email_lines.append("Order Monitoring Report")
        email_lines.append("=" * 50)
    email_lines.append(f"Monitored At: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        if session_id:
            email_lines.append(f"Session ID: {session_id}")
        email_lines.append(f"Total Orders: {len(orders)}")
        email_lines.append(f"Filled: {filled_count} | Canceled: {canceled_count}")
        email_lines.append("")
        email_lines.append("Order Details:")
        email_lines.append("-" * 50)
        
        total_value = 0.0
        for order in orders:
            symbol = order.get('symbol') or 'N/A'
            status = order.get('status') or 'unknown'
            side = order.get('side') or 'N/A'
            metadata = order.get('metadata', {})
            api_order = metadata.get('api_order', {})
            filled_qty = api_order.get('filled_qty', '0')
            filled_price = api_order.get('filled_avg_price')
            
            status_icon = "✅" if status == 'filled' else "❌" if status in ('canceled', 'rejected', 'expired') else "⏳"
            
            line = f"{status_icon} {symbol} ({side.upper()}): {status}"
            if filled_qty and filled_price:
                try:
                    qty = float(filled_qty)
                    price = float(filled_price)
                    value = qty * price
                    total_value += value
                    line += f" | {filled_qty} shares @ ${price:.2f} = ${value:.2f}"
                except (ValueError, TypeError):
                    line += f" | {filled_qty} @ {filled_price}"
            email_lines.append(line)
        
        if total_value > 0:
            email_lines.append("")
            email_lines.append(f"Total Filled Value: ${total_value:,.2f}")
        
        email_lines.append("")
        email_lines.append("=" * 50)
        
        result_data = {
            "orders": orders,
            "email_text": "\n".join(email_lines),
            "summary": {
                "total": len(orders),
                "filled": filled_count,
                "canceled": canceled_count,
                "total_value": total_value
            },
            "session_id": session_id,
            "monitored_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Save to R2 if session_id is present
        if session_id:
            save_monitoring_results_to_r2(session_id, result_data)
        
        # Print all order information before exiting
        if args.json:
            print(json.dumps(result_data, indent=2))
        elif not args.quiet:
            if orders:
                print("\n" + "="*80)
                print("FINAL ORDER STATUS SUMMARY")
                print("="*80)
                for order in orders:
                    order_id = order['order_id']
                    status = order['status'] or 'unknown'
                    symbol = order['symbol'] or 'N/A'
                    metadata = order['metadata']
                    
                    retries = metadata.get('monitoring_retries', 0)
                    filled_qty = metadata.get('api_order', {}).get('filled_qty', 'N/A')
                    filled_price = metadata.get('api_order', {}).get('filled_avg_price', 'N/A')
                    
                    print(f"\nOrder ID: {order_id}")
                    print(f"  Symbol: {symbol}")
                    print(f"  Status: {status}")
                    print(f"  Monitoring Retries: {retries}")
                    print(f"  Filled Qty: {filled_qty}")
                    print(f"  Filled Avg Price: {filled_price}")
                print("="*80 + "\n")

    except KeyboardInterrupt:
        if not args.quiet:
            print("Stopping monitor...")
    finally:
        pass


if __name__ == '__main__':
    main()

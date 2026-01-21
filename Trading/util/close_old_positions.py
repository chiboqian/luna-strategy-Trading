#!/usr/bin/env python3
"""
Close positions older than X business days (Mon–Fri).

- Default X is 4 business days (configurable via Trading.yaml and CLI)
- Dry-run mode lists positions that would be closed

Usage:
    python util/close_old_positions.py --days 4
  python util/close_old_positions.py --dry-run
"""

import argparse
import sys
import json
import os
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yaml
from typing import Optional, Tuple, List
from dotenv import load_dotenv

# Load environment variables
# Note: python-dotenv does not override existing env vars by default.
# We load the most specific (.env in Trading/) first, then the parent (.env in AI_Trading/).
script_path = Path(__file__).resolve()
trading_root = script_path.parent.parent # Trading/
workspace_root = trading_root.parent     # AI_Trading/

load_dotenv(trading_root / ".env")
load_dotenv(workspace_root / ".env")

# Use AlpacaClient directly
sys.path.insert(0, str(trading_root / "Trading"))
from alpaca_client import AlpacaClient


def get_d1_credentials():
    return (
        os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        os.getenv("CLOUDFLARE_D1_DATABASE_ID"),
        os.getenv("CLOUDFLARE_API_TOKEN")
    )

def d1_query(sql: str, params: list = None):
    account_id, database_id, api_token = get_d1_credentials()
    if not all([account_id, database_id, api_token]):
        print("Missing D1 credentials", file=sys.stderr)
        return None
        
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(url, headers=headers, json={"sql": sql, "params": params or []})
        data = resp.json()
        if not data.get("success"):
            # Only print error if it's not a "duplicate column" error which is expected during migration
            is_dup_col = any("duplicate column" in str(e).lower() for e in data.get("errors", []))
            if not is_dup_col:
                print(f"D1 Query Error: {data.get('errors')}", file=sys.stderr)
        return data
    except Exception as e:
        print(f"D1 Request Error: {e}", file=sys.stderr)
        return None

def ensure_d1_columns():
    """Ensure order_closed fields exist."""
    # SQLite ALTER TABLE does not support IF NOT EXISTS for columns standardly
    # We try to add them one by one
    cols = [
        "order_closed BOOLEAN DEFAULT 0",
        "order_closed_date TEXT",
        "order_closed_by TEXT"
    ]
    for col in cols:
        d1_query(f"ALTER TABLE orders ADD COLUMN {col}")

def update_d1_order_closed(order_id: str):
    sql = """
    UPDATE orders 
    SET order_closed = 1,
        order_closed_date = ?,
        order_closed_by = 'close_old_positions.py'
    WHERE order_id = ?
    """
    now_str = datetime.now(timezone.utc).isoformat()
    d1_query(sql, [now_str, order_id])

def get_overdue_d1_orders(days: int) -> List[dict]:
    # Fetch filled buy orders not yet closed
    sql = """
    SELECT * FROM orders 
    WHERE side = 'buy' 
      AND status = 'filled' 
      AND (order_closed IS NOT 1 AND order_closed IS NOT TRUE)
    """
    resp = d1_query(sql)
    if not resp or not resp.get("success"):
        return []
    
    try:
        rows = resp["result"][0]["results"]
    except (KeyError, IndexError, TypeError):
        return []
        
    overdue = []
    now = datetime.now(timezone.utc)
    for row in rows:
        created_at_str = row.get("created_at")
        if not created_at_str:
            continue
        try:
            # Flexible ISO parsing
            dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            if (now - dt).days >= days:
                overdue.append(row)
        except ValueError:
            continue
            
    return overdue


def load_default_days() -> int:
    cfg_path = Path(__file__).parent.parent / "config" / "Trading.yaml"
    try:
        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f) or {}
            trading = data.get('trading', {})
            val = trading.get('close_positions_days')
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass
    return 4


def parse_dt(dt_str: str):
    if not dt_str:
        return None
    try:
        # Alpaca dates are ISO8601 with Z or offset
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception:
        return None


def business_days_between(start_dt: datetime, end_dt: datetime) -> int:
    """Compute elapsed business days (Mon–Fri) between two datetimes.

    Returns the number of business days that have passed since start_dt up to end_dt.
    """
    if end_dt < start_dt:
        return 0
    start = start_dt.date()
    end = end_dt.date()
    total_days = (end - start).days
    if total_days <= 0:
        return 0
    weeks, rem = divmod(total_days, 7)
    biz = weeks * 5
    for i in range(rem):
        wd = (start.weekday() + i) % 7  # 0=Mon .. 6=Sun
        if wd < 5:
            biz += 1
    return biz


def find_position_start_from_orders(client: AlpacaClient, symbol: str) -> Optional[datetime]:
    """Infer the start time of the current open position by scanning filled orders.

    Algorithm:
    - Fetch closed orders for the symbol in ascending time order
    - Accumulate signed filled quantities (+buy, -sell)
    - Track the last index where cumulative quantity returns to 0 (flat)
    - The next filled order time after the last flat point is the position start
    - If cumulative never hits 0, use the first filled order time
    Returns a timezone-aware datetime or None if not determinable
    """
    # Pull up to 500 orders over the last year to keep it bounded
    one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    try:
        orders = client.get_orders(
            status='closed',
            direction='asc',
            symbols=[symbol],
            after=one_year_ago,
            limit=500,
            nested=None,
        ) or []
    except Exception:
        return None

    # Filter to filled-only and map to (filled_at, side, filled_qty)
    items: List[Tuple[datetime, str, float]] = []
    for o in orders:
        filled_at = parse_dt(o.get('filled_at'))
        if not filled_at:
            continue
        try:
            qty = float(o.get('filled_qty') or 0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        side = (o.get('side') or '').lower()
        if side not in ('buy', 'sell'):
            continue
        items.append((filled_at, side, qty))

    if not items:
        return None

    # Accumulate signed quantities to find last flat point
    cum = 0.0
    last_flat_idx = -1
    for idx, (_, side, qty) in enumerate(items):
        signed = qty if side == 'buy' else -qty
        cum += signed
        if abs(cum) < 1e-6:
            last_flat_idx = idx

    # Determine start index (next after last flat)
    start_idx = max(0, last_flat_idx + 1)
    start_time = items[start_idx][0]
    return start_time


def main():
    parser = argparse.ArgumentParser(
        description='Close positions older than X business days (Mon–Fri)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--days', type=int, default=load_default_days(), help='Age threshold in business days (Mon–Fri)')
    parser.add_argument('--use-d1', action='store_true', help='Use D1 database to track order age instead of Alpaca history')
    parser.add_argument('--dry-run', action='store_true', help='Only list positions to close')
    parser.add_argument('--cancel-orders', action='store_true', help='Also cancel open orders when closing all')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    if args.days < 0:
        if args.json:
            print(json.dumps({"error": "--days must be >= 0"}))
            sys.exit(1)
        print('Error: --days must be >= 0', file=sys.stderr)
        sys.exit(1)

    client = AlpacaClient()
    positions = client.get_all_positions() or []
    
    # Check D1 Mode
    if args.use_d1:
        ensure_d1_columns()
        overdue_orders = get_overdue_d1_orders(args.days)
        pos_map = {p['symbol']: p for p in positions}
        
        to_close = []
        
        # Reconciliation and collection
        for order in overdue_orders:
            symbol = order.get('symbol')
            # Assuming 'close the order' implies closing the position if it exists
            if symbol in pos_map:
                # Add to closing list
                # Construct a pseudo-position object that fits existing logic
                p = pos_map[symbol]
                
                # Check duplication in to_close (if multiple old orders for same symbol)
                if not any(x[0]['symbol'] == symbol for x in to_close):
                    # We pass the triggering order's age (approx) for display
                    # Calculate rough age in days for display
                    try:
                        dt = datetime.fromisoformat(order['created_at'].replace('Z', '+00:00'))
                        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                        age_days = (datetime.now(timezone.utc) - dt).days
                    except:
                        age_days = args.days
                        
                    p_copy = dict(p)
                    p_copy['_holding_start'] = order['created_at'] # Use order creation as start
                    p_copy['_d1_order_id'] = order['order_id']
                    
                    to_close.append((p_copy, age_days))
            else:
                # Order exists in D1 as "open" but no position in Alpaca.
                # Just mark it closed in D1 to clean up.
                if not args.dry_run:
                    update_d1_order_closed(order['order_id'])
                    if not args.json:
                        print(f"Reconciled D1: Marked order {order['order_id']} for {symbol} as closed (no open position).")

        if not to_close:
            if args.json:
                print(json.dumps({
                    "positions": [],
                    "summary": f"No overdue D1 orders match active positions",
                    "email_text": "No overdue orders found in the database needing action."
                }))
                return
            print("No overdue orders found in the database needing action.")
            return

    else:
        # Original Logic
        if not positions:
            if args.json:
                print(json.dumps({
                    "positions": [], 
                    "summary": "No open positions",
                    "email_text": "Old Positions Closure Report\n============================\nNo open positions found."
                }))
                return
            print('No open positions')
            return

        now = datetime.now(timezone.utc)

        to_close = []
        for pos in positions:
            symbol = pos.get('symbol')
            if not symbol:
                continue
            # Determine holding start from order history
            start_dt = find_position_start_from_orders(client, symbol)
            if not start_dt:
                # Could not determine age reliably; skip
                continue
            biz_age = business_days_between(start_dt, now)
            if biz_age >= args.days:
                # Attach derived start timestamp to pos for printing
                pos = dict(pos)
                pos['_holding_start'] = start_dt.isoformat()
                pos['_business_days_age'] = biz_age
                to_close.append((pos, biz_age))

    if not to_close:
        if args.json:
            print(json.dumps({
                "positions": [], 
                "summary": f"No positions older than {args.days} business days",
                "email_text": f"Old Positions Closure Report\n============================\nNo positions found older than {args.days} business days."
            }))
            return
        print(f'No positions older than {args.days} business days')
        return

    if not args.json:
        print(f'Found {len(to_close)} positions older than {args.days} business days:')
        for pos, biz_age in to_close:
            symbol = pos.get('symbol')
            qty = pos.get('qty')
            start_ts = pos.get('_holding_start')
            print(f"- {symbol} qty={qty} holding_start={start_ts} age_business_days={biz_age}")

    if args.dry_run:
        if args.json:
            output_list = []
            email_lines = ["Old Positions Closure Report (Dry Run)", "="*40]
            email_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            email_lines.append(f"Threshold: {args.days} business days")
            email_lines.append("")
            
            for pos, biz_age in to_close:
                symbol = pos.get('symbol')
                qty = pos.get('qty')
                start = pos.get('_holding_start')
                email_lines.append(f"- {symbol}: Qty {qty}, Age {biz_age} days (Start: {start}) -> Will Close")
                
                output_list.append({
                    "symbol": symbol,
                    "qty": qty,
                    "holding_start": start,
                    "age_business_days": biz_age,
                    "status": "dry_run"
                })
            
            email_lines.append("")
            email_lines.append("Summary: Dry run, no actions taken.")
            
            print(json.dumps({
                "positions": output_list, 
                "summary": "Dry run: no positions closed",
                "email_text": "\n".join(email_lines)
            }))
            return
        print('\nDry run: no positions will be closed')
        return

    # Close individually to preserve per-symbol control
    closed = 0
    output_list = []
    email_lines = ["Old Positions Closure Report", "="*40]
    email_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    email_lines.append(f"Threshold: {args.days} business days")
    email_lines.append("")

    for pos, biz_age in to_close:
        symbol = pos.get('symbol')
        result_entry = {
            "symbol": symbol,
            "qty": pos.get('qty'),
            "holding_start": pos.get('_holding_start'),
            "age_business_days": biz_age
        }
        try:
            res = client.close_position(symbol=symbol)
            status = res.get('status')
            if not args.json:
                print(f"Closed {symbol}: status={status}")
            
            # If D1 mode, mark the triggering order as closed
            if status in ('filled', 'accepted', 'new', 'pending'):
                d1_oid = pos.get('_d1_order_id')
                if d1_oid:
                    update_d1_order_closed(d1_oid)
                    if not args.json:
                        print(f"Marked D1 order {d1_oid} as closed.")

            result_entry["status"] = status
            
            status_icon = "✅" if status in ('filled', 'accepted', 'new', 'pending') else "⚠️"
            email_lines.append(f"{status_icon} {symbol}: Qty {pos.get('qty')}, Age {biz_age} days -> {status}")
            
            closed += 1
        except Exception as e:
            if not args.json:
                print(f"Error closing {symbol}: {e}")
            result_entry["status"] = "error"
            result_entry["error"] = str(e)
            email_lines.append(f"❌ {symbol}: Error closing -> {e}")
            
        output_list.append(result_entry)

    email_lines.append("")
    email_lines.append(f"Summary: {closed}/{len(to_close)} positions closed")

    if args.json:
        print(json.dumps({
            "positions": output_list,
            "summary": f"{closed}/{len(to_close)} positions closed",
            "email_text": "\n".join(email_lines)
        }))
    else:
        print(f"\nSummary: {closed}/{len(to_close)} positions closed")


if __name__ == '__main__':
    main()

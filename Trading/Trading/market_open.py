#!/usr/bin/env python3
"""
Check if the market is open with sufficient time remaining before close.

Returns exit code 0 (success) if market is open with enough time remaining.
Returns exit code 1 if market is closed or not enough time remaining.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import yaml

from alpaca_client import AlpacaClient


def load_config():
    """Load market timing defaults from Trading.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'Trading.yaml')
    defaults = {
        'min_minutes_remaining': 30,
    }
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            market_timing = config.get('market_timing', {})
            for key in defaults:
                if key in market_timing:
                    defaults[key] = market_timing[key]
    except Exception:
        pass  # Use hardcoded defaults if config not found
    return defaults


def parse_args():
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Check if market is open with sufficient time remaining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=config['min_minutes_remaining'],
        help="Minimum minutes remaining before market close"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output (only use exit code)"
    )
    return parser.parse_args()


def is_market_open_with_time(min_minutes: int = 30) -> dict:
    """
    Check if the market is open with at least min_minutes remaining.
    
    Args:
        min_minutes: Minimum minutes required before market close
        
    Returns:
        dict with keys:
            - is_open: bool - True if market is open
            - has_time: bool - True if enough time remaining
            - can_trade: bool - True if is_open AND has_time
            - minutes_remaining: float - Minutes until close (if market is open)
            - next_open: str - Next market open time
            - next_close: str - Next market close time
            - timestamp: str - Current server time
    """
    client = AlpacaClient()
    clock = client.get_clock()
    
    is_open = clock.get('is_open', False)
    next_close = clock.get('next_close', '')
    next_open = clock.get('next_open', '')
    timestamp = clock.get('timestamp', '')
    
    minutes_remaining = None
    has_time = False
    
    if is_open and next_close:
        # Parse timestamps
        # Alpaca returns RFC3339 format: 2024-01-15T16:00:00-05:00
        try:
            # Parse current time and close time
            current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            close_time = datetime.fromisoformat(next_close.replace('Z', '+00:00'))
            
            # Calculate minutes remaining
            delta = close_time - current_time
            minutes_remaining = delta.total_seconds() / 60.0
            has_time = minutes_remaining >= min_minutes
        except Exception:
            # If parsing fails, assume not enough time
            has_time = False
    
    return {
        'is_open': is_open,
        'has_time': has_time,
        'can_trade': is_open and has_time,
        'minutes_remaining': round(minutes_remaining, 1) if minutes_remaining is not None else None,
        'min_minutes_required': min_minutes,
        'next_open': next_open,
        'next_close': next_close,
        'timestamp': timestamp,
    }


def main():
    args = parse_args()
    
    try:
        result = is_market_open_with_time(args.min_minutes)
        
        if args.json:
            print(json.dumps(result, indent=2))
        elif not args.quiet:
            if result['can_trade']:
                print(f"✅ Market is OPEN with {result['minutes_remaining']:.1f} minutes remaining")
                print(f"   (minimum required: {args.min_minutes} minutes)")
            elif result['is_open']:
                print(f"⚠️ Market is OPEN but only {result['minutes_remaining']:.1f} minutes remaining")
                print(f"   (minimum required: {args.min_minutes} minutes)")
            else:
                print(f"❌ Market is CLOSED")
                print(f"   Next open: {result['next_open']}")
        
        # Exit code: 0 if can trade, 1 otherwise
        sys.exit(0 if result['can_trade'] else 1)
        
    except Exception as e:
        if args.json:
            print(json.dumps({'error': str(e), 'can_trade': False}))
        elif not args.quiet:
            print(f"❌ Error checking market status: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

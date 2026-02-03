#!/usr/bin/env python3
"""
Helper script to dynamically add an option stop-loss watch to streaming_rules.yaml.
This allows the StreamFramework (which watches the config file) to pick up the new rule immediately.

Usage:
    python Trading/options_trading/add_option_watch.py AAPL230616C00150000 2.50
"""
import argparse
import yaml
import sys
from pathlib import Path

def add_option_watch(symbol, stop_price, config_path):
    # Construct the rule
    new_rule = {
        "name": f"Stop Loss {symbol}",
        "symbol": symbol,
        "asset_class": "option",
        "trigger": "quote",
        "condition": {
            "field": "bid_price",
            "operator": "<",
            "value": float(stop_price)
        },
        "action": {
            "command": f"python Trading/Trading/alpaca_close_cli.py {symbol}"
        },
        "one_off": True
    }
    
    path = Path(config_path)
    if not path.exists():
        # Fallback check for default location if running from root
        alt_path = Path("Trading/config/streaming_rules.yaml")
        if alt_path.exists():
            path = alt_path
        else:
            print(f"Error: Config file not found at {path}", file=sys.stderr)
            sys.exit(1)

    # Handle Directory Configuration
    if path.is_dir():
        safe_symbol = "".join(c for c in symbol if c.isalnum() or c in ('_','-'))
        rule_file = path / f"watch_{safe_symbol}.yaml"
        
        config = {'rules': []}
        if rule_file.exists():
            try:
                with open(rule_file, "r") as f:
                    config = yaml.safe_load(f) or {'rules': []}
                    if 'rules' not in config: config = {'rules': []}
            except Exception as e:
                print(f"Error reading {rule_file}: {e}", file=sys.stderr)
        
        # Update or Append
        updated = False
        for i, rule in enumerate(config['rules']):
            if rule.get('name') == new_rule['name']:
                config['rules'][i] = new_rule
                updated = True
                break
        if not updated:
            config['rules'].append(new_rule)
            
        with open(rule_file, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"Added/Updated watch for {symbol} in {rule_file}")
        return

    # Handle Single File Configuration
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}
            
        if 'rules' not in config:
            config['rules'] = []
            
        # Check if rule already exists to avoid duplicates
        updated = False
        for rule in config['rules']:
            if rule.get('symbol') == symbol and rule.get('name') == new_rule['name']:
                print(f"Watch for {symbol} already exists. Updating stop price.")
                rule['condition']['value'] = float(stop_price)
                updated = True
                break
        
        if not updated:
            config['rules'].append(new_rule)
        
        with open(path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
            
        print(f"Added/Updated watch for {symbol} at ${stop_price} in {path}")
        
    except Exception as e:
        print(f"Error updating config: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Add option stop-loss watch to streaming rules")
    parser.add_argument("symbol", help="Option symbol (e.g. AAPL230616C00150000)")
    parser.add_argument("stop_price", type=float, help="Stop price (bid)")
    parser.add_argument("--config", default="Trading/config/streaming_rules.yaml", help="Path to streaming rules config (file or directory)")
    
    args = parser.parse_args()
    add_option_watch(args.symbol, args.stop_price, args.config)

if __name__ == "__main__":
    main()
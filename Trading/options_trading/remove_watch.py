#!/usr/bin/env python3
"""
Helper script to remove a watch rule from streaming configuration.
Supports both single file config and directory config.

Usage:
    python Trading/options_trading/remove_watch.py --symbol AAPL
    python Trading/options_trading/remove_watch.py --name "Stop Loss AAPL..."
"""
import argparse
import yaml
import sys
import os
from pathlib import Path

def remove_watch(symbol, rule_name, config_path):
    path = Path(config_path)
    
    # Smart path detection
    if not path.exists():
        # Try default directory if file not found
        dir_path = Path("Trading/config/streaming_rules")
        if dir_path.exists() and dir_path.is_dir():
            path = dir_path
        else:
            # Try default file
            file_path = Path("Trading/config/streaming_rules.yaml")
            if file_path.exists():
                path = file_path
            else:
                print(f"Error: Config path {config_path} not found.", file=sys.stderr)
                sys.exit(1)

    print(f"Using config path: {path}")

    # Handle Directory Configuration
    if path.is_dir():
        files_to_check = list(path.glob("*.yaml"))
        removed_count = 0
        
        for file_path in files_to_check:
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                if not data: continue
                
                # Normalize data structure to list of rules
                rules = []
                is_dict_wrapper = False
                if isinstance(data, dict):
                    if 'rules' in data:
                        rules = data['rules']
                        is_dict_wrapper = True
                    else:
                        # Single rule object or settings
                        if 'symbol' in data:
                            rules = [data]
                        else:
                            # Likely settings file, skip
                            continue
                elif isinstance(data, list):
                    rules = data
                
                # Filter rules
                new_rules = []
                file_modified = False
                
                for rule in rules:
                    match = True
                    if symbol and rule.get('symbol') != symbol:
                        match = False
                    if rule_name and rule.get('name') != rule_name:
                        match = False
                    
                    if match:
                        print(f"Removing rule '{rule.get('name')}' for {rule.get('symbol')} from {file_path.name}")
                        file_modified = True
                        removed_count += 1
                    else:
                        new_rules.append(rule)
                
                if file_modified:
                    if not new_rules:
                        # If file is empty (and it was a rule file), delete it
                        os.remove(file_path)
                        print(f"Deleted empty file: {file_path}")
                    else:
                        # Write back
                        if is_dict_wrapper:
                            data['rules'] = new_rules
                            output = data
                        else:
                            output = new_rules
                            
                        with open(file_path, 'w') as f:
                            yaml.dump(output, f, sort_keys=False)
                            
            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)
        
        if removed_count == 0:
            print(f"No matching rules found.")
        else:
            print(f"Removed {removed_count} rule(s).")

    # Handle Single File Configuration
    else:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
            
            if 'rules' not in config:
                print("No 'rules' list found in config file.")
                return

            original_count = len(config['rules'])
            new_rules = []
            
            for rule in config['rules']:
                match = True
                if symbol and rule.get('symbol') != symbol:
                    match = False
                if rule_name and rule.get('name') != rule_name:
                    match = False
                
                if match:
                    print(f"Removing rule '{rule.get('name')}' for {rule.get('symbol')}")
                else:
                    new_rules.append(rule)
            
            if len(new_rules) < original_count:
                config['rules'] = new_rules
                with open(path, "w") as f:
                    yaml.dump(config, f, sort_keys=False)
                print(f"Removed {original_count - len(new_rules)} rule(s).")
            else:
                print(f"No matching rules found.")
                
        except Exception as e:
            print(f"Error updating config: {e}", file=sys.stderr)
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Remove watch rule from streaming configuration")
    parser.add_argument("--symbol", help="Symbol to remove (e.g. AAPL)")
    parser.add_argument("--name", help="Specific rule name to remove")
    parser.add_argument("--config", default="Trading/config/streaming_rules.yaml", help="Path to streaming rules config (file or directory)")
    
    args = parser.parse_args()
    
    if not args.symbol and not args.name:
        parser.error("Must specify either --symbol or --name")
        
    remove_watch(args.symbol, args.name, args.config)

if __name__ == "__main__":
    main()
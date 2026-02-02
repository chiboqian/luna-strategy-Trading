#!/usr/bin/env python3
import yaml
import os
import re
from pathlib import Path
import sys

def main():
    # Default paths
    src_file = Path("./config/streaming_rules.yaml")
    dest_dir = Path("./config/streaming_rules")

    if not src_file.exists():
        print(f"Error: Source file {src_file} not found.")
        sys.exit(1)

    print(f"Migrating {src_file} to {dest_dir}...")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Load source
    with open(src_file, 'r') as f:
        config = yaml.safe_load(f)

    if not config:
        print("Source file is empty.")
        sys.exit(0)

    # Migrate Rules
    rules = config.get('rules', [])
    if not rules and isinstance(config, list):
        # Handle case where root is a list
        rules = config
        config = {} # No global settings in this case

    count = 0
    for i, rule in enumerate(rules):
        name = rule.get('name', f"rule_{i}")
        # Create safe filename
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        if not safe_name:
            safe_name = f"rule_{i}"
            
        filename = dest_dir / f"{safe_name}.yaml"
        
        with open(filename, 'w') as f:
            # Wrap in 'rules' list for consistency
            yaml.dump({'rules': [rule]}, f, sort_keys=False)
        
        print(f"  - Created {filename}")
        count += 1

    print(f"\nSuccessfully migrated {count} rules.")
    print(f"To use the new directory, run:")
    print(f"  python Trading/Trading/stream_framework.py --config {dest_dir}")

if __name__ == "__main__":
    main()
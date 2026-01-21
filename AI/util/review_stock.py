#!/usr/bin/env python3
"""
Helper script to run vertex_ai_cli.py with review_stock system instruction.

Usage:
  python util/review_stock.py "AAPL earnings strong, guidance raised"
  python util/review_stock.py --input "Market analysis for TSLA"
  python util/review_stock.py --system-instruction-file config/SI/custom.txt "Analyze stock"
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def get_system_instruction_path(cli_path=None):
    """Get system instruction path from CLI arg or config file."""
    if cli_path:
        return Path(cli_path)
    
    # Try to load from config
    config_path = Path(__file__).parent.parent / "config" / "AI.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'review_stock' in config and 'system_instruction_path' in config['review_stock']:
                return Path(config['review_stock']['system_instruction_path'])
    except Exception:
        pass
    
    # Fallback to default
    return Path("config") / "SI" / "review_stock.txt"


def main():
    parser = argparse.ArgumentParser(
        description='Run Vertex AI with review_stock system instruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', nargs='?', help='Input text for the AI model')
    parser.add_argument('--input', dest='input_flag', help='Input text (alternative to positional arg)')
    parser.add_argument(
        '--system-instruction-file',
        help='Path to system instruction file (overrides config/AI.yaml)'
    )
    parser.add_argument(
        '--config',
        help='Path to AI config file (default: config/AI.yaml)'
    )
    args = parser.parse_args()
    
    # Get input from either positional or flag
    user_input = args.input or args.input_flag
    if not user_input:
        print("Error: Input text required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Resolve system instruction file path
    si_file = get_system_instruction_path(args.system_instruction_file)
    
    # Make path absolute if relative
    if not si_file.is_absolute():
        si_file = Path(__file__).parent.parent / si_file
    
    # Validate file exists
    if not si_file.exists():
        print(f"Error: System instruction file not found: {si_file}", file=sys.stderr)
        sys.exit(1)
    
    # Build command
    script_path = Path(__file__).parent / "vertex_ai_cli.py"
    
    cmd = [
        str(script_path),
        "--system-instruction-file",
        str(si_file),
        "generate",
        user_input
    ]
    
    # Execute
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

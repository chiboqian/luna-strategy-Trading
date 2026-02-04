#!/usr/bin/env python3
"""
Orchestrator script to run an AI session and then execute trades based on the results.

Usage:
    python Trading/util/run_full_cycle.py --command-config AI/config/commands.v1.yaml --dollars 5000 --dry-run
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run AI Session and then Execute Trades")
    
    # run_session args
    parser.add_argument("--command-config", help="Path to command config for run_session")
    parser.add_argument("--prompts", nargs="+", help="Prompts for run_session")
    parser.add_argument("--ai-config", help="Path to AI config file")
    
    # execute_session args
    parser.add_argument("--dollars", type=float, help="Dollar amount per trade")
    parser.add_argument("--dry-run", action="store_true", help="Dry run execution")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--market", action="store_true", help="Use market orders")
    parser.add_argument("--limit", action="store_true", help="Use limit orders")
    parser.add_argument("--mid-price", action="store_true", help="Use mid price")
    parser.add_argument("--check-market", action="store_true", help="Check if market is open before execution")
    
    args = parser.parse_args()

    # Resolve paths
    current_file = Path(__file__).resolve()
    
    # Detect if running from Trading/util/ or common/
    if (current_file.parent.parent / "AI").exists():
        project_root = current_file.parent.parent
    else:
        project_root = current_file.parent.parent.parent
        
    trading_root = project_root / "Trading"
    
    run_session_script = project_root / "AI" / "util" / "run_session.py"
    ai_dir = project_root / "AI"
    execute_session_script = trading_root / "stock_trading" / "execute_session.py"
    market_open_script = trading_root / "market_open.py"
    
    if not run_session_script.exists():
        print(f"Error: Could not find {run_session_script}", file=sys.stderr)
        sys.exit(1)
        
    if not execute_session_script.exists():
        print(f"Error: Could not find {execute_session_script}", file=sys.stderr)
        sys.exit(1)

    if args.check_market:
        if not market_open_script.exists():
            print(f"Error: Could not find {market_open_script}", file=sys.stderr)
            sys.exit(1)
            
        print(f"Checking market status using {market_open_script.name}...")
        # market_open.py returns 0 if open, 1 if closed
        market_status = subprocess.call([sys.executable, str(market_open_script)])
        if market_status != 0:
            print("Market is closed or insufficient time remaining. Aborting cycle.")
            sys.exit(0)

    # 1. Run Session
    print("="*60)
    print("STEP 1: Running AI Session")
    print("="*60)
    
    cmd_session = [sys.executable, str(run_session_script)]
    if args.command_config:
        cmd_session.extend(["--command-config", str(Path(args.command_config).resolve())])
    if args.prompts:
        cmd_session.extend(["--prompts"] + args.prompts)
    if args.ai_config:
        cmd_session.extend(["--config", str(Path(args.ai_config).resolve())])
        
    print(f"Executing: {' '.join(cmd_session)}")
    
    # We capture output to extract session_id, but also print it
    session_id = None
    
    try:
        # Popen to stream output and capture it
        process = subprocess.Popen(
            cmd_session,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ai_dir)
        )
        
        full_output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='') # Stream to console
                full_output.append(line.strip())
                
                # Check for session_id JSON on the fly
                try:
                    stripped = line.strip()
                    if stripped.startswith('{') and 'session_id' in stripped:
                        data = json.loads(stripped)
                        if "session_id" in data:
                            session_id = data["session_id"]
                except json.JSONDecodeError:
                    pass
        
        if process.returncode != 0:
            print(f"Session script failed with return code {process.returncode}", file=sys.stderr)
            sys.exit(process.returncode)
            
        if not session_id:
            # Try parsing from full output if missed during streaming
            for line in reversed(full_output):
                try:
                    data = json.loads(line)
                    if "session_id" in data:
                        session_id = data["session_id"]
                        break
                except json.JSONDecodeError:
                    continue
        
        if not session_id:
            print("Error: Could not extract session_id from run_session output.", file=sys.stderr)
            sys.exit(1)
            
        print(f"\n✅ Captured Session ID: {session_id}")
        
    except Exception as e:
        print(f"Error running session script: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Execute Session
    print("\n" + "="*60)
    print("STEP 2: Executing Trades")
    print("="*60)
    
    cmd_exec = [sys.executable, str(execute_session_script), "--session-id", session_id]
    if args.dollars:
        cmd_exec.extend(["--dollars", str(args.dollars)])
    if args.dry_run:
        cmd_exec.append("--dry-run")
    if args.verbose:
        cmd_exec.append("--verbose")
    if args.market:
        cmd_exec.append("--market")
    if args.limit:
        cmd_exec.append("--limit")
    if args.mid_price:
        cmd_exec.append("--mid-price")
        
    print(f"Executing: {' '.join(cmd_exec)}")
    try:
        subprocess.run(cmd_exec, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing trades: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Deep dive analysis runner.

- Runs `util/review_stock.py` N times per symbol (default 3)
- Extracts JSON from output using regex; parses `symbol`, `analysis_rating`, `conviction_level`
- N is configurable via CLI `--runs` or from `config/commands.yaml` under `deep_dive.runs`

Usage:
  python util/deep_dive.py --symbols AAPL MSFT --runs 5
"""

import argparse
import subprocess
import sys
import re
import json
import time
from pathlib import Path
import yaml

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def load_default_runs():
    """Load default runs from config/commands.yaml if present."""
    cfg = Path(__file__).parent.parent / "config" / "commands.yaml"
    try:
        with open(cfg, 'r') as f:
            data = yaml.safe_load(f) or {}
            dd = data.get('deep_dive', {})
            runs = dd.get('runs')
            if isinstance(runs, int) and runs > 0:
                return runs
    except Exception:
        pass
    return 3


def run_review(symbol: str) -> str:
    """Run review_stock.py for a symbol and return stdout text."""
    script = Path(__file__).parent / "review_stock.py"
    cmd = [str(script), symbol]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"review_stock.py failed for {symbol}: {result.stderr.strip()}" )
    return result.stdout


def extract_json(text: str) -> dict:
    """Extract the last JSON object from text using regex and parse it."""
    matches = list(JSON_BLOCK_RE.finditer(text))
    if not matches:
        raise ValueError("No JSON block found in output")
    raw = matches[-1].group(0)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw: {raw[:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Run deep dive analysis for all symbols',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--runs', type=int, default=load_default_runs(), help='Number of runs per symbol')
    parser.add_argument('--symbols', nargs='+', required=True, help='List of symbols to process')
    parser.add_argument('--json-output', action='store_true', help='Output results as JSON to stdout')
    args = parser.parse_args()

    if args.runs <= 0:
        print("Error: --runs must be >= 1", file=sys.stderr)
        sys.exit(1)

    symbols = args.symbols

    if not args.json_output:
        print(f"Running deep dive for {len(symbols)} symbols, {args.runs} runs each...")
    
    results = []
    for sym in symbols:
        for i in range(1, args.runs + 1):
            for attempt in range(3):
                try:
                    out = run_review(sym)
                    parsed, raw = extract_json(out)
                    # Normalize keys: lowercase and convert spaces/hyphens to underscores
                    norm = {}
                    for k, v in parsed.items():
                        if isinstance(k, str):
                            nk = k.strip().lower().replace(" ", "_").replace("-", "_")
                            norm[nk] = v
                        else:
                            norm[k] = v
                    # Accept keys with various casing or synonyms
                    symbol_val = norm.get('symbol') or norm.get('ticker') or sym
                    analysis_rating = (
                        norm.get('analysis_rating')
                        or norm.get('analyst_rating')
                        or norm.get('analysis')
                        or norm.get('rating')
                    )
                    conviction_level = (
                        norm.get('conviction_level')
                        or norm.get('conviction')
                        or norm.get('confidence')
                        or norm.get('conviction_score') 
                        or norm.get('confidence_score')
                    )
                    
                    # Print progress to stderr so it doesn't interfere with JSON stdout
                    print(f"⏳ {sym} run {i}/{args.runs}: processed", file=sys.stderr)
                    
                    if args.json_output:
                        results.append({
                            "symbol": symbol_val,
                            "analysis_rating": analysis_rating,
                            "conviction_level": conviction_level,
                            "raw_json": parsed,
                            "run_index": i
                        })
                    
                    # If successful, break the retry loop
                    break

                except Exception as e:
                    err_msg = str(e)
                    # If it's a resource exhausted error (429), retry
                    if attempt < 2 and ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg):
                        print(f"⚠️ {sym} run {i}: caught 429/Resource Exhausted (attempt {attempt+1}/3). Retrying in 5s...", file=sys.stderr)
                        time.sleep(5)
                    else:
                        # Log error and give up on this run index (or retry logic exhausted)
                        if attempt == 2: # Last attempt
                             print(f"{sym} run {i}: error -> {e}", file=sys.stderr)
                        elif "429" not in err_msg and "RESOURCE_EXHAUSTED" not in err_msg:
                             # If it's another error, we might not want to retry, or we process it as failure immediately
                             # User asked specifically for "this error" (429). 
                             print(f"{sym} run {i}: error -> {e}", file=sys.stderr)
                             break
    
    if args.json_output:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

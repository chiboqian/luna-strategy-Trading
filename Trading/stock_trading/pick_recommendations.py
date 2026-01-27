#!/usr/bin/env python3
"""
Pick top stock recommendations.

Output: prints top 2 buys and top 1 sell to stdout.

Usage:
  python util/pick_recommendations.py
  
Optional:
  --json   Print output as JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml

RECOMMENDATION_ORDER_BUY = {
    'STRONG BUY': 2,
    'BUY': 1,
}
RECOMMENDATION_ORDER_SELL = {
    'STRONG SELL': 2,
    'SELL': 1,
}


def _load_min_score_from_config() -> float:
    cfg_path = Path(__file__).parent.parent / 'config' / 'Trading.yaml'
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


def fetch_recommendations(min_score: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
    """Fetch buy and sell recommendations.

    Applies a minimum total conviction score threshold for BUY/SELL picks.
    Precedence: CLI arg -> Trading.yaml (recommendations.min_score) -> 22.0
    """
    if min_score is None:
        min_score = _load_min_score_from_config()
    
    # TODO: Implement fetching recommendations from a non-DB source
    buys: List[Dict] = []
    sells: List[Dict] = []
    
    return buys, sells


def main():
    parser = argparse.ArgumentParser(description='Pick top stock recommendations')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--min-score', type=float, help='Minimum total conviction score to include')
    args = parser.parse_args()
    
    try:
        buys, sells = fetch_recommendations(min_score=args.min_score)
        top_buys = buys[:2]
        top_sells = sells[:1]
        
        if args.json:
            print(json.dumps({
                'buys': [{'symbol': b['symbol'], 'recommendation': b['recommendation'], 'score': b['score']} for b in top_buys],
                'sells': [{'symbol': s['symbol'], 'recommendation': s['recommendation'], 'score': s['score']} for s in top_sells]
            }, indent=2))
        else:
            print('Top Buys:')
            for b in top_buys:
                print(f"- {b['symbol']} ({b['recommendation']}) [score: {b['score']:.1f}]")
            print('Top Sells:')
            for s in top_sells:
                print(f"- {s['symbol']} ({s['recommendation']}) [score: {s['score']:.1f}]")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

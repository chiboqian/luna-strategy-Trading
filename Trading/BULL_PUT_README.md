# Bull Put Spread Strategy

## Overview
**Script:** `bull_put_spread.py`

Creates a Bull Put Spread (credit spread) for income generation with defined risk.

**Structure:**
- Sell OTM Put (higher strike) - collect premium
- Buy further OTM Put (lower strike) - protection

## Key Features
| Feature | Description |
|---------|-------------|
| Delta-based selection | Find short put by target delta (e.g., 30 delta) |
| Trend confirmation | SMA20/SMA50 crossover + RSI(14) analysis |
| IV Rank check | Only sells premium when IV > 30% |
| IV/HV Ratio | Identifies rich vs cheap premium |
| Vertical Skew | Analyzes skew between short/long strikes |
| Earnings protection | Blocks trades within 7 days of earnings |
| Dividend protection | Warns if ex-dividend falls before expiration |
| Support level analysis | Warns if strike near 20-day low |
| VIX context | Regime-aware warnings (low/normal/high VIX) |
| Black-Scholes probability | Accurate P(profit) and P(touch) calculation |
| Expected Move | Shows 1 std dev price range at expiry |
| Kelly criterion | Optimal position sizing guidance |
| Theta efficiency | Warns if daily theta/risk ratio is too low |
| Buying power check | Validates account has sufficient capital |
| Existing position check | Warns if you already have exposure |
| ASCII payoff diagram | Visual profit/loss at expiration |
| Position management | Take profit, stop loss, rolling rules |
| Optimization tips | Context-aware suggestions when trade fails filters |

## Usage

```bash
# Delta-based (25 delta short put, $5 wide, dry run)
python util/bull_put_spread.py SPY --short-delta 0.25 --width 5 --dry-run

# Higher delta for more premium (riskier)
python util/bull_put_spread.py SPY --short-delta 0.30 --width 10 --dry-run

# Percentage-based (5% OTM short put)
python util/bull_put_spread.py AAPL --short-pct 5 --width 10 --quantity 2

# Auto-calculate quantity from max risk budget
python util/bull_put_spread.py QQQ --amount 2000 --dry-run

# Skip quality checks (force execution)
python util/bull_put_spread.py TSLA --skip-checks --skip-earnings

# Limit order at mid-price (better fill)
python util/bull_put_spread.py NVDA --limit-order --dry-run
```

## CLI Options
| Option | Default | Description |
|--------|---------|-------------|
| `--days` | 30 | Minimum DTE |
| `--window` | 14 | Search window after min DTE |
| `--short-delta` | 0.30 | Target delta for short put |
| `--short-pct` | 5.0 | Short put % below spot (overrides delta) |
| `--width` | 5.0 | Spread width in dollars |
| `--quantity` | 1 | Number of spreads |
| `--amount` | - | Max risk budget (auto-calculates quantity) |
| `--min-iv-rank` | 0.30 | Minimum IV rank to enter |
| `--skip-checks` | false | Skip all quality filters |
| `--skip-earnings` | false | Skip earnings date check |
| `--limit-order` | false | Use limit order at mid-price |
| `--dry-run` | false | Analyze only, don't execute |
| `--json` | false | JSON output |

## Quality Filters (auto-checked)
- ✅ IV Rank ≥ 30% (elevated volatility)
- ✅ Credit ≥ 20% of spread width
- ✅ Open Interest ≥ 100 on both legs
- ✅ Bid-Ask spread < 20%
- ✅ No earnings within 7 days of expiration
- ✅ No ex-dividend date before expiration
- ✅ Uptrend confirmed (SMA20 > SMA50)
- ✅ Short strike above support level
- ✅ Sufficient buying power in account
- ✅ Theta efficiency > 0.1% daily return on risk

## Output Includes
- Market context (VIX level, sector, beta, trend, RSI, support levels)
- Financial analysis (credit, max loss, break-even, risk/reward, annualized ROC)
- Probability metrics (P(profit), P(touch), expected value)
- Volatility analysis (IV rank, IV/HV ratio, vertical skew)
- Greeks (net delta, gamma, theta, vega per leg)
- Capital requirements (margin, buying power check)
- Expected move range (1 std dev bounds)
- Position management rules (50% take profit, stop loss, rolling guidance)
- Theta burn estimate (days to 50% profit)
- ASCII payoff diagram
- Optimization tips (when trade fails filters)

## Sample Output
```
--- Market Context ---
Market Status:    Market Open
VIX Level:        18.5 (NORMAL)
20-Day Support:   $580.00 (price 3.2% above)
Trend:            Strong Uptrend ✓
                  RSI(14)=55.2 (Neutral)

--- Financial Analysis (Per Share) ---
Net Credit:       $1.25 (25% of spread width)
Max Profit:       $1.25 (stock ≥ $585.00)
Max Loss:         $3.75 (stock ≤ $580.00)
Break Even:       $583.75
Risk/Reward:      0.33 (33% return on risk)
Est. P(Profit):   72.5% (Black-Scholes)
Est. P(Touch):    55.0% (Risk of touching short strike)

--- Greeks (Net Position) ---
Delta:            0.045 (Bullish)
Theta:            $0.008/day (Time Decay Profit)

--- Payoff Diagram (at Expiration) ---
       Risk | Profit
$ 600.00           |### +$1.25  
$ 595.00           |### +$1.25 ← NOW
$ 590.00           |### +$1.25  
$ 585.00           |### +$1.25 ← SHORT
$ 583.75        ###|           $0.00 ← BE
$ 580.00 ##########|           -$3.75 ← LONG
```

## Configuration (`config/Options.yaml`)
```yaml
options:
  bull_put_spread:
    min_days_to_expiration: 30
    search_window_days: 14
    default_short_delta: 0.30
    default_width: 5.0
    min_iv_rank: 0.30
    min_credit_pct: 0.20
    min_open_interest: 100
    max_bid_ask_spread_pct: 0.20
```
# Option Strategy Scanner

## What Is This?

A tool that scans the market to find the **best stocks for Synthetic Long positions** — a strategy that mimics owning 100 shares using options with less capital.

### Why Use a Synthetic Long?
```
┌─────────────────────────────────────────────────────────────┐
│  SYNTHETIC LONG = Long Call + Short Put (Same Strike)       │
│                                                             │
│  ✓ Same profit/loss as owning 100 shares                    │
│  ✓ Uses 70-80% LESS capital than buying stock               │
│  ✓ Built-in leverage without margin interest                │
│  ✓ Configurable strike (ATM, ITM, or OTM)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Scan Top 50 Stocks
```bash
./util/optionStrategyScanner.py
```

### Analyze One Stock (Detailed)
```bash
./util/optionStrategyScanner.py --symbol AAPL --verbose
```

### Scan More Stocks
```bash
./util/optionStrategyScanner.py --limit 100
```

---

## How It Works (5-Stage Filter)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   STAGE 1    │    │   STAGE 2    │    │   STAGE 3    │    │   STAGE 4    │    │   STAGE 5    │
│   Universe   │ -> │   Liquidity  │ -> │  Event Risk  │ -> │   Momentum   │ -> │   Options    │
│   (S&P 500)  │    │   (Price)    │    │ (Earn/Div)   │    │ (RSI/MACD)   │    │ (IV/Greeks)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     500              ~400               ~300                ~100                 MATCHES
```

### Stage 1: Universe
- Starts with **S&P 500** stocks (cached 7 days)
- Keeps only top N by market cap
- Must be tradable + have options on Alpaca

### Stage 2: Liquidity
- Stock price > $15 (avoid penny stocks)
- Must have recent trading activity

### Stage 3: Event Risk (Avoid Surprises)
| Risk | Why It's Bad | Check |
|:-----|:-------------|:------|
| **Earnings** | Gap risk + IV crush | No earnings in next 60 days |
| **Dividends** | Early assignment risk | No ex-div in next 90 days |
| **ETFs** | No earnings/dividends | Auto-pass ✓ |

### Stage 4: Momentum (Is It Trending Up?)
| Check | Rule | Why |
|:------|:-----|:----|
| **Trend** | Price > SMA50 > SMA200 | Confirms uptrend |
| **RSI** | Between 50-75 | Strong but not overbought |
| **ADX** | > 20 | Trend has strength |
| **MACD** | Above signal line | Bullish momentum confirmed |
| **Volume** | > 80% of 20-day avg | Sufficient trading activity |

### Stage 5: Options (Is It Fairly Priced?)
| Check | Rule | Why |
|:------|:-----|:----|
| **IV** | < 45% | Options not overpriced |
| **IV/HV Ratio** | < 1.25 | Fair vs historical vol |
| **Skew** | < 4% | No fear premium in puts |
| **Open Interest** | > 50 | Liquid options |
| **Spread** | < 10% | Tight bid-ask |
| **Net Delta** | > 0.90 | Near-stock exposure |
| **Theta %** | < 0.5% | Acceptable time decay |

---

## Market Context (Displayed at Startup)

```
Market Regime: NORMAL (VIX: 18.5)
```

| VIX Level | Regime | Implication |
|:----------|:-------|:------------|
| < 15 | LOW_VOL (Complacent) | Options cheap, good entry |
| 15-20 | NORMAL | Standard conditions |
| 20-30 | ELEVATED | Caution, higher premiums |
| > 30 | HIGH_VOL (Fear) | Consider waiting |

---

## Output Metrics Explained

When a stock **PASSES**, you'll see these metrics:

### Position Metrics
| Metric | What It Means | Good Value |
|:-------|:--------------|:-----------|
| `strike` | Strike price for position | Configurable |
| `moneyness` | % from spot price | 0% = ATM |
| `syn_cost` | Cost to enter (Call Ask - Put Bid) | Lower = better |
| `net_delta` | Position behaves like X shares | ~1.0 (100 shares) |
| `breakeven` | Stock price needed to profit | Close to current |
| `margin_est` | Capital required for short put | Lower = better |
| `cap_eff_pct` | % less capital vs buying stock | Higher = better |
| `suggested_qty` | Recommended contracts (risk-based) | Based on account size |

### Risk Metrics
| Metric | What It Means | Good Value |
|:-------|:--------------|:-----------|
| `theta` | Daily time decay ($/day) | Small negative |
| `theta_pct` | Theta as % of capital | < 0.5% |
| `gamma` | How fast delta changes | Higher = more responsive |
| `vega` | Sensitivity to IV change | Depends on outlook |
| `max_loss` | Loss if stock drops 20% | Risk management |
| `risk_reward` | Gain (10% up) vs Loss (20% down) | Higher = better |
| `prob_profit` | Estimated chance of profit | Higher = better |

### Volatility Metrics
| Metric | What It Means | Good Value |
|:-------|:--------------|:-----------|
| `atm_iv` | Current option volatility | < 45% |
| `hv20` | 20-day historical volatility | Baseline |
| `iv_hv_ratio` | IV vs HV comparison | < 1.25 |
| `iv_pctl` | Where IV ranks vs past year | Lower = cheaper |
| `skew` | Put IV - Call IV difference | < 0.04 |
| `term_structure` | IV curve shape | CONTANGO = favorable |

### Momentum Indicators
| Metric | What It Means | Good Value |
|:-------|:--------------|:-----------|
| `macd_bullish` | MACD above signal line | True |
| `rel_volume` | Volume vs 20-day average | > 0.8 |
| `spy_corr` | Correlation with SPY (20-day) | Diversification info |
| `sector` | Stock's sector | Diversification info |

### Scoring
| Metric | What It Means | Range |
|:-------|:--------------|:------|
| `score` | Composite ranking score | 0-100 |

**Score Components:**
- IV Quality: 0-20 pts (lower IV = better)
- Skew: 0-15 pts (flatter = better)
- Delta: 0-20 pts (closer to 1.0 = better)
- Risk/Reward: 0-15 pts (higher ratio = better)
- Capital Efficiency: 0-15 pts (less capital = better)
- MACD Bullish: 0-10 pts (bullish = better)
- Relative Volume: 0-5 pts (higher = better)

---

## Configuration

Edit `Trading/config/Options.yaml`:

```yaml
option_scan:
  # What to scan
  max_scan_count: 50        # How many stocks
  min_price: 15.0           # Min stock price
  
  # Momentum filters
  rsi_min: 50.0             # Min RSI
  rsi_max: 75.0             # Max RSI  
  min_adx: 20.0             # Min trend strength
  min_rel_volume: 0.8       # Min volume vs 20-day avg
  require_macd_bullish: true # MACD must be bullish
  
  # Option filters
  iv_max_threshold: 0.45    # Max IV (45%)
  iv_hv_ratio_threshold: 1.25
  skew_tolerance: 0.04
  max_option_spread_pct: 0.10
  min_open_interest: 50
  
  # Strike selection
  target_moneyness: 1.0     # 1.0=ATM, 0.98=2% ITM, 1.02=2% OTM
  
  # Greeks filters
  min_net_delta: 0.90       # Min delta for position
  max_theta_pct: 0.005      # Max daily decay (0.5%)
  
  # Expiration window
  target_dte_min: 10        # Min days out
  target_dte_max: 21        # Max days out
  
  # Position sizing
  account_size: 100000      # Account size ($)
  max_risk_per_trade: 0.02  # Max risk per trade (2%)
```

---

## Example Output

```
Initialized Scanner. Risk-Free Rate: 4.25%
Market Regime: NORMAL (VIX: 18.5)

[1/50] Scanning NVDA ($148.50)... 
  [Detail] Momentum Metrics for NVDA:
    Price: $148.50
    SMA50: $142.30 (Price > SMA50: True)
    SMA200: $128.75 (SMA50 > SMA200: True)
    RSI: 62.3 (Allowed: 50-75)
    ADX: 35.2 (Min: 20, Trend Strength: Strong)
    MACD: 2.150 (Signal: 1.820, Histogram: 0.330)
    MACD Status: BULLISH (Recent Crossover: No)
    Rel Volume: 1.25x (Min: 0.8x, Today: 45,230,000)
    HV20: 28.50% (Historical Volatility)
    IV Percentile: 35% (Current vol rank vs 1yr)
    SPY Correlation: 0.72 (Diversification: High)
    Sector: Technology
    
  [Detail] Greeks Filters:
    Net Delta: 0.998 (Min: 0.90) -> PASS
    Theta %: 0.0023 (Max: 0.005) -> PASS
  [Detail] Term Structure: CONTANGO (Spread: 0.0180)
  [Detail] Position Sizing:
    Account: $100,000, Max Risk: 2.0%
    Risk/Contract: $3,195.00
    Suggested Contracts: 1
  
*** MATCH FOUND: NVDA ***

--- Final Candidates (Sorted by Score) ---
Market Regime: NORMAL (VIX: 18.5)
Sector Distribution: {'Technology': 3, 'Healthcare': 2, 'Finance': 1}

| symbol | sector     | price  | dte | atm_iv | skew | syn_cost | delta | cap_eff | r/r  | score |
|--------|------------|--------|-----|--------|------|----------|-------|---------|------|-------|
| NVDA   | Technology | 148.50 | 14  | 0.32   | 0.02 | $1.25    | 1.00  | 78.5%   | 0.46 | 82.3  |
| AAPL   | Technology | 185.20 | 14  | 0.28   | 0.01 | $0.85    | 0.99  | 81.2%   | 0.52 | 79.1  |
```

### No Candidates Found - Recommendations

When no stocks pass all filters, the scanner provides actionable recommendations from the failed candidates:

```
============================================================
📋 RECOMMENDATIONS (Top 5 Near-Miss Candidates)
============================================================
These stocks came closest to passing all filters:

1. META @ $485.20
   Failed: IV high (48%>45%)
   RSI: 58.2, IV: 48%
   💡 Tip: Wait for IV to drop or sell premium strategies instead

2. TSLA @ $275.30
   Failed: Skew wide (0.055>0.04)
   RSI: 61.5, IV: 42%
   💡 Tip: High put premium suggests fear - consider put spreads

3. GOOGL @ $142.80
   Failed: Momentum (Price < SMA50)
   RSI: 48.3, IV: 35%
   💡 Tip: Wait for trend confirmation (Price > SMA50 > SMA200)
============================================================
```

**Recommendation Priority** (closest to passing first):
1. **Options failures** - IV/Skew/Greeks issues (passed momentum checks)
2. **Contract issues** - No valid contracts found at target strikes
3. **Momentum failures** - Trend not confirmed yet
4. **Earnings risk** - Calendar-based, will clear after announcement
5. **Dividend risk** - Calendar-based, will clear after ex-date

Configure recommendation count:
```yaml
option_scan:
  recommendation_count: 5    # Top N recommendations when no matches
```

---

## Term Structure Analysis

The scanner analyzes IV term structure to identify favorable conditions:

```
CONTANGO (Normal)         FLAT                    BACKWARDATION (Risky)
     ▲                      ▲                           ▲
  IV │    ____/             │  ─────────                │  \____
     │   /                  │                           │       \
     └────────────►         └────────────►              └────────────►
       Near    Far            Near    Far                 Near    Far
       DTE                    DTE                         DTE
       
   ✓ Favorable              ○ Neutral                  ✗ Event Risk
```

- **CONTANGO**: Far-dated options have higher IV → normal market, favorable
- **FLAT**: Similar IV across expirations → neutral
- **BACKWARDATION**: Near-dated IV higher → event risk (earnings, etc.)

---

## Technical Notes

- **Black-Scholes Solver**: Calculates IV, Delta, Gamma, Theta, Vega locally
- **Risk-Free Rate**: Fetches live 3-month T-Bill rate from `^IRX`
- **VIX Level**: Fetched from `^VIX` for market regime classification
- **Caching**: S&P 500 list cached for 7 days
- **ETF Detection**: Automatically handles SPY, QQQ, etc. (no earnings/dividends)
- **Sector Data**: Fetched from Yahoo Finance for diversification tracking
- **SPY Correlation**: 20-day rolling correlation for portfolio context
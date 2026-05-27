# Quality + Timing Two-Stage Model Design

## Problem

Current approaches try to predict stock returns from 74 technical features in a single model across all 3000+ stocks. This fails because:

1. **60% of stocks are noise** — ST, micro-cap, loss-making stocks dominate the universe
2. **Frequency mismatch** — fundamental data (quarterly PE/ROE) mixed with daily technical data
3. **Wasted capacity** — model must learn "good vs bad" before it can learn "good vs timing"

The fundamental prediction limit across our 10 experiments is consistent: **~1-3% above random on individual stock return prediction.**

## Solution: Two-Stage Decomposition

### Stage 1: Quality Filter (fundamental)
**When to buy**: always (hold a quality portfolio)
**Which stocks**: fundamentally sound companies
**Output**: investable universe (~1000 out of 3000)

### Stage 2: Timing Model (technical)  
**When to buy**: technical signals indicate entry
**Which stocks**: among the quality universe, which will outperform
**Output**: daily scores → trading signals

---

## Stage 1: Quality Filter

### Data Sources

| Source | Columns | Frequency |
|--------|---------|-----------|
| `daily_basic/YYYY_MM_DD.parquet` | `pe_ttm`, `pb`, `total_mv`, `circ_mv`, `turnover_rate` | Daily |
| `stock_basic/stock_basic.parquet` | `industry`, `market`, `list_date` | Static |
| `stock_st/YYYY_MM_DD.parquet` | `ts_code` (ST list) | Daily |

### Filter Rules

Applied per date, per stock:

| Rule | Condition | Rationale |
|------|-----------|-----------|
| Listing age | `list_date` >= 1 year ago | Exclude 新股 / speculative recent IPOs |
| ST | Not in `stock_st` for this date | Exclude special treatment |
| Market | `market == "主板"` | Main board only |
| Market cap | `circ_mv >= 50亿` (5 billion CNY) | Exclude micro-caps |
| PE | `5 < pe_ttm < 40` | Value + not loss-making (excludes PE <0) |
| PB | `0 < pb < 10` | Not distressed, not bubbles |
| ROE | From alpha158 `ROE_RAW > 0` | Positive return on equity |
| Turnover | `turnover_rate > 0.1%` | Minimum liquidity |

### Implementation

```python
def build_quality_universe(data_dir, trade_date):
    """Return set of ts_codes passing quality filter."""
    db = pd.read_parquet(f"{data_dir}/daily_basic/{trade_date:Y_m_d}.parquet")
    sb = pd.read_parquet(f"{data_dir}/stock_basic.parquet")
    st = pd.read_parquet(f"{data_dir}/stock_st/{trade_date:Y_m_d}.parquet")
    
    # Exclude ST
    codes = set(db["ts_code"]) - set(st["ts_code"])
    
    # Market, listing age filters
    sb_filtered = sb[sb["market"] == "主板"]
    sb_filtered = sb_filtered[pd.to_datetime(trade_date) - pd.to_datetime(sb_filtered["list_date"]) >= timedelta(days=365)]
    codes &= set(sb_filtered["ts_code"])
    
    # Merge with daily_basic for fundamental filters
    merged = db[db["ts_code"].isin(codes)].merge(sb_filtered[["ts_code"]], on="ts_code")
    
    # Apply thresholds
    merged = merged[
        (merged["circ_mv"] >= 5e9) &
        (merged["pe_ttm"] > 5) & (merged["pe_ttm"] < 40) &
        (merged["pb"] > 0) & (merged["pb"] < 10) &
        (merged["turnover_rate"] > 0.1)
    ]
    
    return set(merged["ts_code"])
```

**Expected size**: ~600-1200 stocks per day (from ~3000 main board)

---

## Stage 2: Timing Model

### Features (34 dimensions)

**Alpha360 (14):**
```
open, high, low, close, volume, vwap,
volume_ratio, turnover_rate, hl_ratio, ret_5d, close_ma20,
atr_ratio, vol_change, amt_change
```

**Alpha158 unique signals (20):**
```
SKEWP10, SKEWP20, SLOPE5, SLOPE20, CHP30,
GAP, DRP, O2C, HLC,
EMA12C, EMA26C, RSI14, MACD,
MV, pe_ttm, pb, ROE, ROE_RAW,
TURN, VOLRATIO
```

**Key change**: NO KMID/KLEN variants (96 each are redundant with alpha360's hl_ratio + ret_5d). NO per-sequence z-score on fundamentals (keep PE/PB/ROE as raw values or cross-sectional z-scores).

### Sequence

- 20-day or 60-day sequence
- For each stock on date t, features span [t-seq_len+1, t]
- Labels: forward returns at t+5, t+20, t+60

### Labels

Percentile rank WITHIN the quality universe, NOT the full market:

```
For each date t:
  1. Get all quality-universe stocks on date t
  2. For each stock, compute ret = close[t+h] / close[t] - 1
  3. Rank stocks by ret → assign percentile [0, 99]
```

This ensures the model learns timing among quality stocks, not "good vs bad."

### Model

Reuse `RankScoreTransformer`:
- Input: (B, seq_len, 34) 
- Output: (B, 3) — scores for 5d/20d/60d
- Loss: paired_loss (same as current)
- Evaluation: Spearman correlation within quality universe

### Training Procedure

1. Build flat cache with quality-universe filter: only include stocks that pass quality filter on the target date
2. For each stock, extract sequences normally
3. Labels: percentile ranks computed within each date's quality universe
4. Train with pairwise ranking loss
5. Evaluate on held-out validation period

---

## Data Pipeline Changes

### New File: `quality_filter.py`

```python
def load_quality_flat(data_dir, start, end) -> Path:
    """Build flat parquet, filtering to quality universe per date."""
    # For each date in range:
    #   1. Load daily_basic + stock_basic + stock_st
    #   2. Apply quality filter → set of good codes
    #   3. Load alpha158 + alpha360 for good codes only
    #   4. Merge and append
```

### Modified Cache Builder

The existing `build_numpy` is reused but loads from quality-filtered flat parquet.

### Timeline

1. **Implement quality_filter.py** — 1 hour
2. **Build flat cache with filter** — 5 minutes
3. **Retrain timing model** — ~20 minutes
4. **Evaluate** — compare Spearman with full-universe results

---

## Expected Outcomes

| Metric | Full Universe | Quality Universe |
|--------|--------------|-----------------|
| Stocks per date | ~3000 | ~800-1200 |
| Noise stocks | High | Low |
| Timing signal | 0.01 Spearman | Expected 0.03-0.05 |
| Feature effectiveness | Diffuse | Focused |

## Future Extensions

- **Industry-neutral scores**: normalize scores within industry (long best-in-sector, short worst-in-sector)
- **Composite quality score**: instead of binary filter, use weighted fundamental score (e.g., Piotroski F-Score)
- **Position sizing**: allocate proportional to timing score, adjusted by volatility

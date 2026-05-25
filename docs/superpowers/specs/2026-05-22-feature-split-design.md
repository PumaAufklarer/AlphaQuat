# Feature Module Split Design

## Overview

Split the monolithic `alpha158.py` (207 factors) into three separate files organized by data
source: core OHLCV factors (Alpha158), extended price-pattern factors, and fundamental factors.
Add a combined registry for training with all factors.

## Motivation

- `alpha158.py` has grown to ~430 lines mixing Qlib's original factors with custom additions
- Adding new factors requires touching the same file, risking regressions
- The model can now select which factor set to use for training (e.g. test with only
  core Alpha158, or only fundamental)
- Clear separation makes it obvious which factors come from which data source

## File Structure

```
src/alpha_quat/features/alphasets/
├── __init__.py
├── alpha158.py          # Pure Qlib Alpha158 (OHLCV core, ~170 factors)
├── alpha_ext.py         # Extended price-pattern factors (CHP, SKEWP, GAP, etc.)
└── alpha_fund.py        # Fundamental factors (PE_TTM, PB, ROE, MV, TURN, VOLRATIO)
```

## alpha158.py — Core OHLCV Factors

Extract from current `alpha158.py`:
- `$open/$high/$low/$close` — REF, MEAN, STD, MAX, MIN at windows 5/10/20/30/60
- `$vwap` — REF, MEAN, STD at windows 5/10/20/30/60
- `$volume/$amount` — REF, MEAN, STD at windows 5/10/20/30/60, MAX at 5/10/20/30
- `CORR($close, $volume)` and `CORR($close, $amount)` at windows 5/10/20/30/60
- `RANK` (cross-sectional) of close-derived expressions at windows 5/10/20/30/60
- `QUANTILE` (cross-sectional) of close-derived expressions at windows 5/10/20/30/60
- `DELTA($close/$volume/$amount)` at windows 5/10/20/30/60
- `SUM($close)` at windows 5/10/20/30/60

**Remove:** SUM($volume/$amount), CHP*, SKEWP*, GAP, DRP, O2C, HLC, PE_TTM, PB, ROE*, MV, TURN, VOLRATIO, ROE_RAW

**Builder:** `build_alpha158() -> FactorRegistry` (same name, unchanged API)

## alpha_ext.py — Extended Price-Pattern Factors

Extract from current `alpha158.py`:
- CHP30 (keep sole effective window; CHP5/10/20/60 pruned)
- SKEWP10, SKEWP20 (keep two effective windows; SKEWP5/30/60 pruned)
- GAP, DRP, O2C, HLC

**Builder:** `build_alpha_ext() -> FactorRegistry`

## alpha_fund.py — Fundamental Factors

Extract from current `alpha158.py`:
- PE_TTM, PB, MV (3-year time-series position versions)
- ROE_RAW (intermediate), ROE (3-year time-series position)
- TURN, VOLRATIO

**Builder:** `build_alpha_fund() -> FactorRegistry`

## Combined Registry

```python
def build_alpha_combined() -> FactorRegistry:
    """Merge all factor sets into a single registry for training."""
    reg = build_alpha158()
    for f in build_alpha_ext().factors.values():
        reg.register(f)
    for f in build_alpha_fund().factors.values():
        reg.register(f)
    return reg
```

Located in `alphasets/__init__.py` or `alphasets/combined.py`.

## CLI Changes

```python
ALL_FEATURE_SETS = {
    "alpha158": "alpha_quat.features.alphasets.alpha158:build_alpha158",
    "alpha_ext": "alpha_quat.features.alphasets.alpha_ext:build_alpha_ext",
    "alpha_fund": "alpha_quat.features.alphasets.alpha_fund:build_alpha_fund",
    "alpha_combined": "alpha_quat.features.alphasets:build_alpha_combined",
}
```

Default used by `alpha-quat feature` and `alpha-quat model`: `alpha_combined`.
Users can choose specific subsets:
```bash
uv run alpha-quat feature -f alpha158         # core only
uv run alpha-quat feature -f alpha_fund       # fundamentals only
uv run alpha-quat feature -f alpha_combined   # all (default)
```

## Data Flow

```
alpha_combined (or any subset)
    │
    ▼
FeatureEngine.compute_batch(registry, dates)   # same DuckDB CTE engine
    │
    ▼
features/{YYYMMDD}.parquet
    │
    ▼
DatasetBuilder  →  model training
```

The FeatureEngine reads from the same `daily/` and `daily_basic/` parquet
files regardless of which factor set is used. Missing `daily_basic` data is
handled with NULL defaults (backward compatible). No engine changes needed.

## Testing

Existing alpha158 tests already validate compilation, deps, cycles, and lookback.
These move to `test_alpha158.py` (unchanged).

New tests:
- `test_alpha_ext.py` — 6 factors register, compile, no cycles, deps exist
- `test_alpha_fund.py` — 7 factors register, compile, no cycles, deps exist
- `test_alpha_combined.py` — combined registry has correct total count

## Non-goals

- Engine changes (FeatureEngine, base CTE, DSL compiler)
- Changes to the Factor dataclass or FactorRegistry
- Backward compatibility breaks (old `alpha158` feature files can still be loaded)

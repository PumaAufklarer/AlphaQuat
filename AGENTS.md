# AGENTS.md — alpha-quat

## Toolchain

Always use `uv` — pip/poetry won't work here.

```
uv run pytest --cov=src          # test
uv run ruff format .              # format
uv run ruff check --fix .         # lint
uv run pyright                    # typecheck (46 pre-existing errors, don't fix)
```

## Verification order

```
uv run ruff format . && uv run ruff check --fix . && uv run pytest --cov=src
```

## Architecture: ML models (`src/alpha_quat/model/`)

### LightGBM variants (`model/lightgbm/variants/`)

Each variant is a separate file inheriting from `LightGBMBasePipeline(ABC)`, registered via `@register`:

| File | mode | Trains |
|------|------|--------|
| `regression.py` | `"regression"` | 3 models (5d/20d/60d), MSE loss |
| `quantile.py` | `"quantile"` | 9 models (3 horizons × 3 alphas), pinball loss |
| `lambdarank.py` | `"lambdarank"` | 3 models with ranking objective |

`pipeline.py` is a factory — `run_variant(data_dir, ExperimentConfig)` dispatches to the right variant. Never instantiate `LightGBMPipeline` directly.

**Model output:** `data/models/experiments/<name>/` with `experiment.yaml` config snapshot + model files.

### Neural network (`model/nn/`)

Current variant: `sr_transformer` — predicts support/resistance probability distributions.

| Module | Purpose |
|--------|---------|
| `transformer/labels.py` | SR label generation (local peak/trough detection) |
| `transformer/models/transformer.py` | `StockTransformer` — 4-layer TransformerEncoder |
| `transformer/models/dataset.py` | `SRSequenceDataset` — per-sequence normalization (price ratios + log volume) |
| `transformer/train.py` | AdamW + warmup-cosine + grad clipping + label smoothing |
| `transformer/evaluate.py` | Per-horizon loss, top-3 bin accuracy |
| `transformer/inference.py` | `SRInference` — `predict_batch()` for GPU-batched inference |

**Per-sequence normalization** — each (60, 6) sequence normalized independently:
- Price features (O/H/L/C/VWAP): `value / close[-1] - 1` → ratio relative to last close
- Volume: `log(1+vol) → z-score` (global stats from training set)

**Labels:** class indices (int64, not one-hot). `CrossEntropyLoss` with `label_smoothing=0.1`. Invalid horizons masked via float weight `w = 1/(1 + dist/5)` where dist = days to nearest peak.

### Architecture: experiment system (`src/alpha_quat/experiment/`)

| Module | Purpose |
|--------|---------|
| `config.py` | `ExperimentConfig` — name, mode, date ranges, hyperparams. Save/load via yaml |
| `registry.py` | `ExperimentRegistry` — `data/models/registry.json` tracks all experiments |

Every model run requires `--name <experiment_name>`. Models saved to `data/models/experiments/<name>/`.

## Architecture: SR cache (`src/alpha_quat/data/sr_cache.py`)

Pre-computes Alpha360 features (6 raw OHLCV fields) + SR labels (nearest support/resistance levels with distance in days). Output: `data/alpha360/YYYYMMDD.parquet`.

Algorithm per stock:
1. Vectorized local peak/trough detection via pandas rolling (C-level)
2. Verify rejection/bounce within 3 days (0.5% threshold)
3. Reverse-fill nearest verified peak via deque (O(n) per horizon)
4. Per-horizon neighborhoods: 5d=±2, 20d=±10, 60d=±30

## Architecture: backtesting

### LightGBM ranking backtest (`backtest/engine.py`)

`BacktestEngine` — day-by-day, T+1 open execution, top-K rebalance. Supports experiment-based signal selection via `strategy/signals/variants/`.

### SR Transformer backtest (`backtest/engine_sr.py`)

`run_sr_backtest()` — separate entry for SR-based strategies. Price-triggered execution (checks daily low/high vs target price). Position sizing: score-weighted allocation, max 25% per stock, max 8 holdings, 10% min cash.

Performance optimization: pre-loads all alpha360 data once, builds daily batches from in-memory numpy arrays.

## Signal variants (`strategy/signals/variants/`)

Each variant registered via `@register`. `BaseMLSignal(ABC)` with `_load_models()` + `generate()`.

| Variant | mode | Loads |
|---------|------|-------|
| `regression_signal.py` | `"regression"` | 3 models (5d/20d/60d) |
| `quantile_signal.py` | `"quantile"` | 9 models (3 horizons × 3 alphas) |
| `lambdarank_signal.py` | `"lambdarank"` | 3 lambdarank models |
| `transformer_sr_signal.py` | `"transformer_sr"` | `SRInference` — batch GPU inference |

## CLI

```
uv run alpha-quat                                    # same as "fetch"
uv run alpha-quat fetch -s daily,trade_cal
uv run alpha-quat feature

# LightGBM — variant is positional
uv run alpha-quat model lightgbm quantile --name exp_quantile_v1
uv run alpha-quat model lightgbm regression --name exp_reg_v1
uv run alpha-quat model lightgbm lambdarank --name exp_lr_v1

# Transformer SR
uv run alpha-quat sr-cache                                            # precompute alpha360 + SR labels
uv run alpha-quat model nn sr_transformer --name exp_sr_v1            # train
uv run alpha-quat model nn sr_transformer --name tune --tune          # grid search

# Backtest
uv run alpha-quat backtest --experiment exp_quantile_v1               # LightGBM ranking bt
uv run alpha-quat backtest-sr --experiment exp_sr_v1 --start 20240101 # SR entry/exit bt

# Manage
uv run alpha-quat experiment list
uv run alpha-quat experiment show exp_name
uv run alpha-quat --summary
```

## Data output

```
data/
├── registry.db                    # metadata: api_name + last date
├── daily/YYYY_MM_DD.parquet      # raw tushare data
├── daily_basic/YYYY_MM_DD.parquet
├── stock_basic.parquet
├── trade_cal.parquet
├── stock_st/YYYY_MM_DD.parquet
├── features/YYYYMMDD.parquet     # alpha158 factor files (no dashes)
├── alpha360/YYYYMMDD.parquet     # SR cache: 6 price+volume + 12 SR columns
└── models/
    ├── experiments/<name>/       # named experiment directory
    │   ├── experiment.yaml       # full config snapshot
    │   ├── model.pt              # or lightgbm_model_*.txt
    │   ├── metrics.json
    │   └── norm_params.json      # log_vol stats for inference
    └── registry.json             # index of all experiments
```

## Gotchas

### Data
- **stock_st schema drift** — some files VARCHAR ts_code, others INTEGER. Always `read_parquet(..., union_by_name=true)` + `CAST(ts_code AS VARCHAR)`.
- **`daily/` uses YYYY_MM_DD** underscores; **`features/` and `alpha360/` use YYYYMMDD** no dashes.
- **vwap is computed** — `amount / vol` in base CTE. Not a raw tushare field.
- **`stock_st` API name** — tushare pro uses `stock_st` not `stk_st`.

### Model training
- **`--name` is required** for all training commands. Old `data/models/` path abandoned.
- **`meta` stacking removed** — was ineffective, code cleaned up.
- **`stride` default = 10** — stride=30 was too aggressive, not enough near-term samples.
- **per-sequence normalization** — divide by last close, not global z-score. This removed stock-specific shortcuts.
- **batch inference** — `SRInference.predict_batch()` runs one forward pass for all stocks, not per-stock loops.

### Backtest
- **SR backtest (`backtest-sr`) is separate** from `engine.py`. Do not add SR if/else to `engine.py`.
- **pre-loads all alpha360 data** once before the day loop for performance.
- **Price-triggered execution** — checks daily high/low vs target price. `unfilled_signals` tracks price not met.
- **Holding.stop_price** — support-level stop loss stored per position. Portfolio.buy() accepts `stop_price=` parameter.
- **Support predictions are weaker than resistance** — 25% win rate strategy with high-reward tail.

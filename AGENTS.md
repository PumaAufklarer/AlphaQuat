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

Three variants: `sr_transformer` (legacy production), `keltner` (regime classification, experimental), `rl_agent` (RL position control, experimental).

#### SR Transformer (`model/nn/transformer/`)

Predicts support/resistance probability distributions from 60-day sequences.

**Input:** 17 features [6 OHLCV + 8 derived + 3 Keltner], per-sequence normalized.

| Module | Purpose |
|--------|---------|
| `models/transformer.py` | `StockTransformer` — 4-layer TransformerEncoder |
| `models/dataset.py` | `SRSequenceDataset` — per-sequence normalization (price ratios + log volume + per-seq z-score for derived features) |
| `train.py` | AdamW + warmup-cosine + grad clipping + label smoothing |
| `evaluate.py` | Per-horizon loss, top-3 bin accuracy, peak detection, price-top3-given-peak |
| `inference.py` | `SRInference` — batch GPU inference + entry/exit signals |

**Labels:** 100 price bins + 1 "no peak" bin (= 101 bins). `CrossEntropyLoss` with `label_smoothing=0.1`. `w = 1/(1 + dist/5)` for price bins, `w = 1.0` for "no peak" bin.

**Status:** Not effective. Support/resistance are inherently historical, not predictive.

#### Keltner Regime Model (`model/nn/keltner/`)

Predicts market regime (ranging/support_test/resistance_test/breakout_up/breakout_down) from 14 OHLCV-derived features (Keltner channel NOT in input).

**Status:** Abandoned. Model learned to read Keltner position from features (circular), not market structure. 55% accuracy was auto-correlation, not prediction.

#### RL Agent (`model/nn/rl_agent/`)

Per-stock continuous position control [-1, 1] via REINFORCE.

| Module | Purpose |
|--------|---------|
| `models/position_agent.py` | `PositionAgent` — same encoder + Gaussian policy head |
| `pretrain.py` | Supervised direction-prediction pre-training (accuracy 55% val, overfits) |
| `train.py` | REINFORCE with cross-sectional normalization, per-stock baseline, informative-day filtering |
| `evaluate.py` | Multi-horizon Sharpe (5d+20d) on held-out stocks |
| `rank_model.py` | Cross-sectional pairwise ranking loss (Spearman=0) |
| `variants/rl_agent_variant.py` | Two-phase pipeline: pretrain → RL |

**Key limiting factor:** 14 OHLCV-derived features don't have enough signal for per-stock RL. LightGBM lambdarank succeeds because it uses 158+ cross-sectional features (Alpha158).

**Experiments:**
| Experiment | Method | Result |
|-----------|--------|--------|
| exp_agent_v1 | REINFORCE raw reward, global baseline | Sharpe stuck at ±0.464 |
| exp_agent_v2 | + per-stock baseline + informative filter | Same |
| exp_agent_v3 | + cross-sectional normalization | Same |
| exp_agent_v4 | + sign(reward) advantage | Same |
| exp_agent_v5 | + supervised pretrain | Val acc 55%, RL same |
| exp_rank_v1 | Cross-sectional ranking (pairwise loss) | Spearman ≈ 0 |

**Lesson learned:** Per-stock OHLCV features are insufficient for RL trading agents. The cross-sectional Alpha158 factor set (158+ factors with rank/quantile/industry) is what makes LightGBM effective. Transformer-based approaches need either more informative features or a cross-sectional training setup.

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

**Added features (2026-05):**
- 8 derived features: `volume_ratio`, `turnover_rate`, `hl_ratio`, `ret_5d`, `close_ma20`, `atr_ratio`, `vol_change`, `amt_change`
- 3 Keltner Channel features: `keltner_pos`, `keltner_width`, `keltner_above_ema`
- Total: 6 raw + 8 derived + 3 Keltner = **17 features** per stock per day

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

# Keltner Regime (experimental)
uv run alpha-quat model nn keltner --name exp_keltner_v1

# RL Agent (experimental — does not learn effectively)
uv run alpha-quat model nn rl_agent --name exp_agent_v5

# Rank Scorer (experimental — Spearman ~0)
uv run alpha-quat build-rank-cache                                         # pre-compute sequence cache
uv run alpha-quat model nn rank_scorer --name exp_ar_v5                    # train pairwise rank model

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
├── alpha360/YYYYMMDD.parquet     # SR cache: 17 features + 12 SR columns
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

### Rank Scorer Transformer (`model/nn/alpha_rank/`)

Cross-sectional pairwise ranking model: Transformer → score per stock, trained with margin ranking loss.

| Module | Purpose |
|--------|---------|
| `build_cache.py` | Flat parquet (alpha360 + alpha158 + daily_basic) → numpy sequence cache |
| `model.py` | `RankScoreTransformer` — 60-day encoder + scoring head |
| `train.py` | Pairwise ranking loss, per-date batching, Spearman eval |
| `quality_filter.py` | Industry-relative PE/PB quality filter (look-back-free) |

**Experiments:**

| Experiment | Features | Seq | Universe | Loss↓ | Spearman |
|-----------|----------|-----|----------|-------|----------|
| exp_ar_v1 | 42 feat (A360+A158) | 20d | All 主板 | 4.60→4.60 | ~0 |
| exp_ar_v2 | 10-bin soft CE + edge compensation | 20d | All 主板 | 6.85→6.65 | top1=13% |
| exp_ar_v3 | 74 feat pairwise rank | 60d | All 主板 | 0.48→0.42 | 0.01 |
| exp_ar_v4 | 74 feat + quality filter | 60d | Quality pool | 0.48→0.42 | 0.01 |
| exp_ar_v5 | 21 feat (A360 raw + temporal + industry) | 60d | Quality pool | 0.48→0.46 | ~0 |

**Root cause:** Transformer cannot learn cross-sectional stock ranking from OHLCV sequences alone. The loss decreases by exploiting correlated features (temporal features, date co-linearity) without learning actual ranking. LightGBM succeeds because its 158+ Alpha158 features include cross-sectional rank/quantile statistics, which are the actual signal source.

### Key Learnings & Industry Context

Gemini 3.5 analysis of the 10+ failed experiments identified three structural issues:

1. **Structural prior mismatch** — GBDT is designed for tabular data (non-linear cuts, robust to outliers). Transformer assumes strong sequential semantics (language, images). Stock OHLCV has no such semantics — it's a near-random walk.

2. **Information density** — LightGBM's 158+ features already include cross-sectional operators (cs_rank, cs_zscore, cs_quantile). The Transformer experiments used only per-stock features with no knowledge of the broader market cross-section.

3. **Loss decrease without rank improvement** — The model learns to exploit correlated features (date co-linearity, temporal structure) to reduce pairwise margin loss without actually learning ranking. This is structural, not fixable by tuning.

### Industry Approaches (Top Chinese Quant Funds)

| Approach | Description | Example |
|----------|-------------|---------|
| **Two-stage (时序→横截面)** | Stage 1: share-weighted temporal encoder → stock embedding. Stage 2: cross-stock attention on embeddings | Spatio-Temporal Graph, Dual-Asymmetric Transformer |
| **Foundation pretraining** | MAE-style: mask 30% of price/factor tokens, reconstruct them. Self-supervised on factors, not returns. | Creates market manifold embeddings |
| **Transformer as factor synthesizer** | Transformer outputs 32-64 dim embeddings → appended to LightGBM 158-factor pool | Linear boost to OOS Sharpe |

### Next Exploration Paths (Priority Order)

| Pri | Path | Method | Expected Delta |
|-----|------|--------|---------------|
| 1 | **Hybrid: Transformer → LGBM** | Extract penultimate layer embeddings from rank_scorer → append to LightGBM 158 factors → retrain lambdarank | Small positive (Sharpe 1.4→1.45?) |
| 2 | **Conditional LN** | Remove time features. Add Conditional LayerNorm conditioned on industry/market-cap. Retrain. | Validate if v5 improvement was real |
| 3 | **Two-stage temporal+CS** | Full two-stage: temporal encoder (60d×per-stock features) → CS attention (cross-stock). | High cost, uncertain gain |

### RL / Neural models
- **14 OHLCV features insufficient** — RL agents and ranking models fail to learn from 14 features alone.
  LightGBM lambdarank succeeds because of its 158+ cross-sectional Alpha158 features (rank/quantile/industry).
- **Per-stock RL doesn't work** — REINFORCE with continuous positions (-1, 1) produces constant-bias policies
  (Sharpe stuck at ±0.464). The signal-to-noise ratio in individual stock returns is too low.
- **"No peak" bin (bin 100)** — improves SR label quality but doesn't fix the fundamental issue.
- **Keltner features are leaky** — including keltner features as model inputs creates circular learning
  (model predicts channel position from channel position). Use for dataset curation only.

### Backtest
- **SR backtest (`backtest-sr`) is separate** from `engine.py`. Do not add SR if/else to `engine.py`.
- **pre-loads all alpha360 data** once before the day loop for performance.
- **Price-triggered execution** — checks daily high/low vs target price. `unfilled_signals` tracks price not met.
- **Holding.stop_price** — support-level stop loss stored per position. Portfolio.buy() accepts `stop_price=` parameter.
- **Support predictions are weaker than resistance** — 25% win rate strategy with high-reward tail.
- **25% win rate is misleading** — counts partial stop-loss fills as separate losing trades. Position-level win rate is ~58% (712 matched buy→sell pairs). The reported win rate penalizes stop-loss strategies that partially exit and then re-enter.
- **Best trades are in 0-5 yuan bucket**: 266 trades, 67% win rate, +621 avg PnL, +165K total PnL. 10-15 yuan bucket underperforms (36% win rate, negative total PnL).
- **`--min-price` filter is harmful** — excludes the best-performing low-price bucket. Model's strength is picking cheap stocks with high upside skew.
- **CLI flags**: `--min-price <float>` (default 0.0), `--quality-filter` (bool), both passed through `BacktestConfig` → `build_universe()` in `filters.py`.

### Strategy characteristics
- **High skew strategy**: 57.9% of positions win, but winners are 4-5× larger than losers (75th pct=+591 vs 25th pct=-280). Right tail extends to +21,683 max win.
- **Hold duration matters**: Held ≤5d = 0% win rate (stop-loss kills), held >20d = 68.8% win rate (let trades run).
- **Model prefers low-price stocks**: All buys in 3-8 yuan range. This is not a bug — KLEN/KMID volatility signals are strongest on cheap stocks, and MV adjustment isn't enough to overcome the KLEN/KMID dominance.
- **Fundamentals help but don't change character**: MV became #1 feature (gain=1525.8) with fundamentals enabled, but top-k selection still picks cheap volatile stocks due to KLEN/KMID group dominance (~86% of total gain).

# AlphaQuat

A quantitative stock trading framework for A-shares. Pulls Chinese market data via Tushare, computes 200+ technical factors (Qlib Alpha158), trains LightGBM quantile regression / lambdarank models, and backtests strategies. Also includes experimental SR Transformer, Keltner regime, and RL agent models.

## Workflow

```
data fetch → feature engineering → model training → backtest → daily predict
```

## Quick Start

```bash
# Install
uv sync

# Pull data (one-time, ~8k daily files from 1990)
uv run alpha-quat fetch -s all

# Compute factors (200+ from alpha158 + alpha_ext + alpha_fund)
uv run alpha-quat feature -f alpha_combined --rebuild --since 20180101

# Train quantile regression ensemble (9 models: 3 horizons × 3 quantiles)
uv run alpha-quat model lightgbm \
  --train-start 20180401 --train-end 20220330 \
  --val-start 20220401 --val-end 20260330 \
  --no-tune --quantile
```

## Backtest (Best Config — Lambdarank + Fundamentals)

```bash
uv run alpha-quat backtest \
  --experiment exp_lr_v4 \
  --start 20220501 --end 20260501 \
  --capital 50000 --monthly 8000 \
  --top-k 15 --rebalance-interval 5 \
  --stop-loss 0.10 --weighting score_momentum
```

Key parameters:
| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | — | Experiment name for ML signal |
| `--rebalance-interval` | `5` | Trading days between rebalances |
| `--stop-loss` | `0.10` | Dynamic stop-loss from peak price |
| `--top-k` | `15` | Max holdings |
| `--weighting` | `score_momentum` | Position sizing: equal, kelly, vol_parity, score_momentum |
| `--quality-filter` | — | Apply PE/PB industry-relative quality screen |
| `--min-price` | `0.0` | Minimum stock price (exclude pennies — harmful, don't use) |

## Model Training

```bash
# Point-estimate regression (faster)
uv run alpha-quat model lightgbm --no-tune

# Quantile regression (9 models: 10%/50%/90% × 5d/20d/60d) — best
uv run alpha-quat model lightgbm --no-tune --quantile

# With Optuna tuning  (50 trials × TimeSeriesSplit CV)
uv run alpha-quat model lightgbm --trials 50

# With meta stacking layer (optional enhancement)
uv run alpha-quat model lightgbm --no-tune --quantile \
  --meta-start 20220401 --meta-end 20230330
```

## Daily Scoring

```bash
uv run alpha-quat predict --top-k 10 --holdings data/holdings.yaml
```

Auto-pulls latest data → computes factors → scores 3000+ stocks → prints Top-K, Bottom-K, and current holdings with rankings (confidence intervals when quantile mode detected).

Holdings file: `data/holdings.yaml`
```yaml
holdings:
  - ts_code: "603288.SH"
    shares: 100
    avg_cost: 35.10
```

## Factor DSL

Built-in operators: `REF`, `MEAN`, `STD`, `SUM`, `MAX`, `MIN`, `CORR`, `DELTA`, `RANK`, `QUANTILE`, `EMA`, `RSI`, `REG_SLOPE`

Raw fields: `$open`, `$high`, `$low`, `$close`, `$volume`, `$amount`, `$vwap` (daily) + `$pe_ttm`, `$pb`, `$total_mv`, `$turnover_rate`, `$volume_ratio` (daily_basic)

## Dependencies

- `duckdb` — data processing
- `lightgbm` — model training
- `optuna` — hyperparameter tuning
- `tushare` — China market data
- `matplotlib` — backtest report charts

## Project Structure

```
src/alpha_quat/
├── cli.py              # CLI entry point
├── data/               # Data fetching (tushare → parquet)
├── features/           # Factor computation engine (DuckDB)
│   └── alphasets/      # Factor set definitions (alpha158/ext/fund)
├── model/              # ML model (dataset, train, evaluate, meta, predict, rolling)
│   ├── lightgbm/       # LightGBM — regression, quantile, lambdarank (production)
│   └── nn/             # Neural networks (all experimental — none effective)
│       ├── transformer/    # StockTransformer — SR price prediction
│       ├── keltner/        # Keltner Channel regime prediction
│       ├── rl_agent/       # REINFORCE position control agent
│       └── alpha_rank/     # Cross-sectional ranking transformer
├── strategy/           # Signal generators + position managers
└── backtest/           # Backtesting engine + metrics + report
```

## OOS Results

### Lambdarank + Fundamentals (2026-05)

```
Period:      2022-05 ~ 2026-05 (4 year out-of-sample)
Capital:     ¥50,000 + ¥8,000/month
Strategy:    LightGBM lambdarank, 206 features incl. fundamentals, score_momentum weighting
Return:      +59.27% (cumulative), +12.87% (annualized)
Sharpe:      1.09
Max DD:      -18.24%
Trades:      1431
Win Rate:    28.8% (per-event) / 57.9% (per-position)
```

Key findings:
- **Fundamentals (MV, PE, PB, ROE, MACD, RSI) added +0.28 Sharpe** vs Alpha158-only baseline
- **Model picks 3-8 yuan cheap stocks** despite MV preference for large caps — KLEN/KMID volatility signals dominate
- **0-5 yuan bucket**: 67% win rate, +165K total PnL — model's sweet spot
- **25-29% win rate is misleading** — counts partial stop-loss fills. Real position-level win rate is ~58%
- **Quality post-filter hurt Sharpe** (1.09 → 0.93) — model already learns fundamental preference internally

### Quantile Regression (2025 reference)

```
Period:      2022-04 ~ 2026-03 (4 year out-of-sample)
Capital:     ¥50,000 + ¥8,000/month
Strategy:    Quantile median ensemble, bi-weekly rebalance, graded sell
Return:      +168.98% (cumulative), +29.45% (annualized)
Sharpe:      1.88
Max DD:      -20.08%
Trades:      347
```

## Rolling Backtest (8-fold)

```
Fold 1: +17.5% Sharpe 1.75    Fold 5: +6.6%  Sharpe 0.30
Fold 2: +13.8% Sharpe 2.15    Fold 6: +8.5%  Sharpe 0.69
Fold 3: +7.2%  Sharpe 0.82    Fold 7: +18.9% Sharpe 1.98
Fold 4: +14.3% Sharpe 1.54    Fold 8: +11.8% Sharpe 2.05
Average: +12.3%/half-year, Sharpe 1.41, all 8 folds profitable
```

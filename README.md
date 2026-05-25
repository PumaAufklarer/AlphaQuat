# AlphaQuat

A quantitative stock selection framework for A-shares. Pulls Chinese market data via Tushare, computes 200+ technical factors (Qlib Alpha158), trains LightGBM quantile regression models to predict future price channel position, backtests strategies, and provides daily scoring.

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

## Backtest (Best Config)

```bash
uv run alpha-quat backtest \
  --start 20220401 --end 20260330 \
  --capital 50000 --monthly 8000 \
  --top-k 5 --rebalance-interval 10 \
  --sell-threshold 0.40
```

Key parameters:
| Flag | Default | Description |
|------|---------|-------------|
| `--rebalance-interval` | `5` | Trading days between rebalances (5=weekly, 10=bi-weekly) |
| `--sell-threshold` | `0.40` | Sell stocks outside Top-K only if score < threshold |
| `--daily-monitor` | — | Continuous daily mode (sell weak, buy best) |
| `--sell-score-percentile` | — | Sell holdings scoring below this percentile |
| `--stop-loss` | `0.15` | Dynamic stop-loss from peak price |
| `--top-k` | `5` | Max holdings |
| `--monthly` | `8000` | Monthly capital addition |
| `--weighting` | `equal` | Position sizing: equal, kelly, vol_parity, score_momentum |
| `--quantile` | — | Train quantile regression (9 models) instead of point estimates |

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
│   └── lightgbm/       # LightGBM implementation
├── strategy/           # Signal generators + position managers
└── backtest/           # Backtesting engine + metrics + report
```

## OOS Best Result

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

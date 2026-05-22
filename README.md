# AlphaQuat

A quantitative stock selection framework for A-shares. Pulls Chinese market data via Tushare, computes 200+ technical factors (Qlib Alpha158), trains LightGBM models to predict future price channel position, backtests strategies, and provides daily scoring.

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

# Compute factors (158+ built-in factors)
uv run alpha-quat feature -f alpha158 --rebuild --since 20180101

# Train ensemble model: 5d / 20d / 60d channel position labels
uv run alpha-quat model lightgbm \
  --train-start 20180401 --train-end 20220430 \
  --val-start 20220501 --val-end 20260430 --trials 50
```

## Backtest

```bash
# MA crossover strategy
uv run alpha-quat backtest

# ML ensemble strategy (best config)
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
| `--model-dir` | `data/models/` | ML model directory |

## Daily Scoring

```bash
uv run alpha-quat predict --top-k 10 --holdings data/holdings.yaml
```

Auto-pulls latest data → computes factors → scores 3000+ stocks → prints Top-K, Bottom-K, and current holdings with rankings.

Holdings file: `data/holdings.yaml`
```yaml
holdings:
  - ts_code: "603288.SH"
    shares: 100
    avg_cost: 35.10
  - ts_code: "000529.SZ"
    shares: 800
    avg_cost: 12.30
```

## Factor DSL

Built-in operator support: `REF`, `MEAN`, `STD`, `SUM`, `MAX`, `MIN`, `CORR`, `DELTA`, `RANK`, `QUANTILE`

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
│   └── alphasets/      # Factor set definitions
├── model/              # ML model (dataset, train, evaluate)
│   └── lightgbm/       # LightGBM implementation
├── strategy/           # Signal generators + position managers
└── backtest/           # Backtesting engine + metrics + report
```

## OOS Best Config

```
Period:      2022-04 ~ 2026-03
Capital:     ¥50,000 + ¥8,000/month
Strategy:    Bi-weekly rebalance, graded sell (< 0.40)
Return:      +97.54% (cumulative), +19.43% (annualized)
Sharpe:      1.20
Max DD:      -16.49%
Trades:      447
```

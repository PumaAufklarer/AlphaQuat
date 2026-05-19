# Backtest Engine Design

## Overview

Build a backtesting module that simulates a daily trading strategy over a date range,
tracking portfolio value, generating trades, and producing performance metrics and an
HTML report.

## Requirements

- **Strategy**: Dual MA crossover (golden cross buy, dead cross sell)
  using alpha158 `KLEN35` (MA5/close) and `KLEN36` (MA20/close) factors.
  Golden cross when `KLEN35 > KLEN36` and previous day `KLEN35 <= KLEN36`;
  dead cross is the reverse.
- **Position**: Top-5 equal weight, 15% stop loss from entry cost.
- **Capital**: 20k initial, 8k monthly addition on the first trading day of each month.
- **Filters**: Exclude ST stocks (from `stock_st` data), only main board (stock_basic.market == "主板").
- **Costs**: 0.05% commission rate (万0.5), no minimum (不免五).
- **Timing**: Signal at close of day T → execute at open of day T+1.
  Stop loss: if T-1 close < cost * 0.85 → sell at T's open.
- **Period**: 2022-05 to 2026-05 (default, configurable).
- **Output**: CLI subcommand `alpha-quat backtest` + HTML report (equity curve,
  drawdown chart, Sharpe, trade details).

## Architecture

New module `src/alpha_quat/backtest/` sits on top of the existing `strategy/` layer.
It orchestrates the day-by-day loop, manages portfolio state, loads data, and
produces outputs.

```
src/alpha_quat/backtest/
├── __init__.py
├── config.py        # BacktestConfig dataclass
├── portfolio.py     # Portfolio — cash, holdings, snapshots, trade log
├── engine.py        # BacktestEngine — day loop orchestration
├── metrics.py       # return, drawdown, Sharpe, win rate
├── report.py        # HTML report generation
└── filters.py       # universe filters (ST exclusion, main board)

src/alpha_quat/strategy/           # extensions to existing module
├── signals/
│   ├── __init__.py
│   └── ma_cross.py      # MACrossSignal(ISignalGenerator)
└── positions/
    ├── __init__.py
    └── equal_weight.py  # EqualWeightTopKPosition(IPositionManager) with stop loss
```

The existing `Strategy`, `ISignalGenerator`, and `IPositionManager` remain unchanged
in their contracts. The backtest engine wraps `Strategy.run()` per trading day.

### Data layer relationship

```
data/
├── daily/YYYY_MM_DD.parquet        ← read by BacktestEngine for price/volume
├── stock_basic.parquet             ← read for market field (universe filter)
├── stock_st/YYYY_MM_DD.parquet     ← read for ST exclusion per date
├── features/YYYYMMDD.parquet       ← read for KLEN35/KLEN36 (signal input)
└── trade_cal.parquet               ← read for trading date list
```

## BacktestConfig

```python
@dataclass
class BacktestConfig:
    start_date: str = "20220501"
    end_date: str = "20260501"
    initial_capital: float = 20000
    monthly_addition: float = 8000
    commission_rate: float = 0.0005   # 万0.5
    min_commission: float = 0.0       # 不免五
    stop_loss_pct: float = 0.15
    short_factor: str = "KLEN35"      # MA5/close
    long_factor: str = "KLEN36"       # MA20/close
    top_k: int = 5
    benchmark: str | None = None
```

## Day-by-day Loop

For each trading date `d` in the range:

```
1. If d is the first trading day of the month: cash += monthly_addition
2. Execute pending orders from d-1 at d's open price (deduct commission)
3. Stop loss: for each holding where d-1 close < avg_cost * (1 - stop_loss_pct),
   generate an immediate sell order at d's open price
4. Load d's features parquet → filter universe (exclude ST, non-主板)
5. MACrossSignal: compare KLEN35 vs KLEN36 on d vs d-1 to detect crosses
6. Convert buy/sell signals into pending orders for d+1's open
7. Mark-to-market all holdings at d's close price
8. Record portfolio snapshot (date, cash, market_value, total_value)
```

### First day (no previous)

- No pending orders to execute on day 1
- First day of range, but may not be first day of month — check calendar month
- Add initial capital as starting cash
- Generate signals from day 1's features (no cross possible without d-1 data,
  so day 1 typically produces no signals)

### Last day

- Execute pending orders from day before
- No new signals needed (no next day to execute)
- Record final snapshot

## Portfolio

```python
@dataclass
class Holding:
    ts_code: str
    shares: int           # lots of 100, rounded down
    avg_cost: float       # volume-weighted average buy price
    buy_date: str         # most recent buy date (for stop loss reference)

class Portfolio:
    cash: float
    holdings: dict[str, Holding]
    snapshots: list[dict]   # {date, cash, market_value, total_value}
    trades: list[dict]      # {date, ts_code, action, shares, price, commission, pnl}

    def buy(self, ts_code, price, target_value, date, commission_rate) -> int: ...
    def sell(self, ts_code, price, shares, date, commission_rate) -> float: ...
    def market_value(self, prices: dict[str, float]) -> float: ...
    def total_value(self, prices) -> float: ...
```

- `buy()`: computes affordable shares (round down to 100-share lots), deducts cost + commission.
- `sell()`: liquidates given shares at price, deducts commission, adds cash.
- Shares are always rounded down to multiples of 100 (A-share lot size).
- Stop-loss sells liquidate the entire holding.

## Filters

```python
def build_universe(date: str, data_dir: Path) -> set[str]:
    """
    Returns set of ts_code eligible for trading on the given date.
    Excludes:
    - ST stocks (check stock_st/YYYY_MM_DD.parquet for the date)
    - Non-main-board (stock_basic.market != "主板")
    """
```

- Reads `stock_basic.parquet` once, caches the market mapping.
- Reads `stock_st/` hive-partitioned parquet, builds a set of ST ts_codes per date.
- A stock is excluded if it appears in stock_st on or overlapping the date.

## Signal: MACrossSignal

Implements `ISignalGenerator`. Tracks previous-day factor ratios internally.

```python
class MACrossSignal(ISignalGenerator):
    def __init__(self, short_factor="KLEN35", long_factor="KLEN36"):
        self.short = short_factor
        self.long = long_factor
        self._prev: pd.DataFrame | None = None  # previous day's ts_code + factors

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        cur = features[["ts_code", self.short, self.long]].copy()
        cur["cross_above"] = cur[self.short] > cur[self.long]

        buy_codes = []
        sell_codes = []
        if self._prev is not None:
            merged = cur.merge(
                self._prev, on="ts_code", suffixes=("", "_prev")
            )
            golden = merged.loc[
                merged["cross_above"] & ~merged["cross_above_prev"], "ts_code"
            ]
            dead = merged.loc[
                ~merged["cross_above"] & merged["cross_above_prev"], "ts_code"
            ]
            buy_codes = golden.tolist()
            sell_codes = dead.tolist()

        self._prev = cur[["ts_code", "cross_above"]]
        # ...
        return SignalResult(signals=df)
```

- `SignalResult.signals` DataFrame schema: `ts_code`, `action` ("buy"/"sell"), `score`.
- Buy signals come from golden cross; sell signals come from dead cross.
- Score is set to 1.0 for buys, 0 for sells (equal weight in Top-K).

## Position: EqualWeightTopKPosition

Implements `IPositionManager`. Provides equal-weight allocation, universe filtering,
and stop-loss logic.

```python
class EqualWeightTopKPosition(IPositionManager):
    def __init__(self, top_k=5):
        self.top_k = top_k

    def allocate(self, signals, ctx) -> pd.DataFrame:
        # Pick top_k buy signals by score, assign weight 1/top_k each
        # Sells remove the stock from allocation (weight = 0)

    def constrain(self, positions, ctx) -> pd.DataFrame:
        # Filter to universe, clamp weights 0..1/top_k, normalize

    def execute(self, target, prev, ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Compute shares = target_weight * capital / price, round to lots
        # Generate buy/sell orders for delta vs prev holdings
```

- `target` DataFrame schema: `ts_code`, `weight`.
- `orders` DataFrame schema: `ts_code`, `action`, `shares`.
- Stop loss is handled by `BacktestEngine` before calling `Strategy.run()`, not inside position manager,
  because it depends on portfolio-level cost basis tracking.

## Metrics

```python
def compute_metrics(
    snapshots: list[dict],
    trades: list[dict],
    total_invested: float,
    risk_free_rate: float = 0.025,
) -> dict:
```

| Metric | Formula |
|--------|---------|
| Cumulative return | `(final_value - total_invested) / total_invested` |
| Annualized return | `(1 + cum_return) ^ (1 / years) - 1` |
| Max drawdown | max peak-to-trough decline, with date |
| Sharpe ratio | `(ann_return - risk_free) / ann_volatility` |
| Win rate | profitable trades / total closed trades |

## HTML Report

Generated as a single self-contained HTML file using matplotlib charts embedded as
base64 images. No external CDN dependencies.

Sections:
1. **Summary cards**: cumulative return, annualized return, Sharpe, max drawdown,
   win rate, total trades.
2. **Equity curve chart**: portfolio NAV over time.
3. **Drawdown chart**: drawdown percentage over time.
4. **Monthly returns table**: P&L per month.
5. **Trade log table**: date, ts_code, action, price, shares, commission, P&L.
6. **Config summary**: backtest parameters.

## CLI Integration

```
uv run alpha-quat backtest [options]

Options:
  --start YYYYMMDD         Start date (default: 20220501)
  --end YYYYMMDD           End date (default: 20260501)
  --capital FLOAT          Initial capital (default: 20000)
  --monthly FLOAT          Monthly addition (default: 8000)
  --commission FLOAT       Commission rate (default: 0.0005)
  --stop-loss FLOAT        Stop loss percentage (default: 0.15)
  --top-k INT              Max holdings (default: 5)
  --ma-short INT           Short MA window (default: 5)
  --ma-long INT            Long MA window (default: 20)
  --output PATH            HTML report output path (default: data/backtest_report.html)
```

## Testing

Unit tests per module:
- `tests/test_backtest/test_config.py` — default values, validation
- `tests/test_backtest/test_portfolio.py` — buy/sell rounding, commission, P&L
- `tests/test_backtest/test_filters.py` — ST exclusion, main board filter
- `tests/test_backtest/test_metrics.py` — return/drawdown/sharpe arithmetic
- `tests/test_strategy/test_ma_cross.py` — golden/dead cross detection
- `tests/test_strategy/test_equal_weight.py` — allocation, constraints, share rounding
- `tests/test_backtest/test_engine.py` — integration: full mini backtest with synthetic data

## Dependencies

- `matplotlib` — charts for HTML report (new dependency)

## Non-goals

- Benchmark comparison (e.g., CSI 300) — deferred to future iteration
- Non-daily frequency (weekly/monthly)
- Short selling or margin
- Slippage modeling beyond fixed commission
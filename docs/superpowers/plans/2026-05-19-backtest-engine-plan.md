# Backtest Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily backtesting engine that simulates a dual-MA crossover strategy with portfolio accounting, performance metrics, and HTML reporting.

**Architecture:** New `src/alpha_quat/backtest/` module. The engine calls `ISignalGenerator` at EOD and `IPositionManager` at next-day open for T日→T+1 open execution timing. `Strategy.run()` is bypassed — the engine directly composes signal/position calls for timing control. New `strategy/signals/ma_cross.py` and `strategy/positions/equal_weight.py` implement the strategy components.

**Tech Stack:** Python, pandas, duckdb, matplotlib (new dep), existing alpha_quat infrastructure.

---

## File Structure

```
Create:
  src/alpha_quat/backtest/__init__.py
  src/alpha_quat/backtest/config.py
  src/alpha_quat/backtest/filters.py
  src/alpha_quat/backtest/portfolio.py
  src/alpha_quat/backtest/metrics.py
  src/alpha_quat/backtest/engine.py
  src/alpha_quat/backtest/report.py
  src/alpha_quat/strategy/signals/__init__.py
  src/alpha_quat/strategy/signals/ma_cross.py
  src/alpha_quat/strategy/positions/__init__.py
  src/alpha_quat/strategy/positions/equal_weight.py
  tests/test_backtest/__init__.py
  tests/test_backtest/test_config.py
  tests/test_backtest/test_filters.py
  tests/test_backtest/test_portfolio.py
  tests/test_backtest/test_metrics.py
  tests/test_backtest/test_engine.py
  tests/test_strategy/test_ma_cross.py
  tests/test_strategy/test_equal_weight.py
Modify:
  pyproject.toml
  src/alpha_quat/cli.py
```

See plan file for detailed tasks 1-11 with exact code.

Task 1: matplotlib dep (pyproject.toml)
Task 2: BacktestConfig (config.py + tests)
Task 3: Universe filters (filters.py + tests)  
Task 4: Portfolio (portfolio.py + tests)
Task 5: Metrics (metrics.py + tests)
Task 6: MACrossSignal (strategy/signals/ma_cross.py + tests)
Task 7: EqualWeightTopKPosition (strategy/positions/equal_weight.py + tests)
Task 8: BacktestEngine (engine.py + tests)
Task 9: HTML Report (report.py)
Task 10: CLI backtest subcommand (cli.py)
Task 11: Integration verification + typecheck

### Task 1: Add matplotlib dependency

**Files:** Modify: `pyproject.toml`

- [ ] **Step 1: Add matplotlib to dependencies**

Edit `pyproject.toml`, add `"matplotlib>=3.8.0"` after `"duckdb>=1.5.2"` in the dependencies list:

```toml
dependencies = [
    "duckdb>=1.5.2",
    "matplotlib>=3.8.0",
    "pyarrow>=24.0.0",
    "pyright>=1.1.409",
    "pytest>=9.0.3",
    "pyyaml>=6.0.3",
    "ruff>=0.15.12",
    "tushare>=1.4.29",
]
```

- [ ] **Step 2: Install**

```bash
uv sync
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add matplotlib dependency"
```

---


### Task 2: BacktestConfig

**Files:**
- Create: `src/alpha_quat/backtest/__init__.py`
- Create: `src/alpha_quat/backtest/config.py`
- Create: `tests/test_backtest/__init__.py`
- Create: `tests/test_backtest/test_config.py`

- [ ] **Step 1: Write failing test** — `tests/test_backtest/test_config.py`

```python
from alpha_quat.backtest.config import BacktestConfig


class TestBacktestConfig:
    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.start_date == "20220501"
        assert cfg.end_date == "20260501"
        assert cfg.initial_capital == 20000
        assert cfg.monthly_addition == 8000
        assert cfg.commission_rate == 0.0005
        assert cfg.min_commission == 0.0
        assert cfg.stop_loss_pct == 0.15
        assert cfg.short_factor == "KLEN35"
        assert cfg.long_factor == "KLEN36"
        assert cfg.top_k == 5
        assert cfg.benchmark is None

    def test_custom_values(self):
        cfg = BacktestConfig(
            start_date="20230101", end_date="20240101",
            initial_capital=100000, commission_rate=0.0003, top_k=10,
        )
        assert cfg.start_date == "20230101"
        assert cfg.initial_capital == 100000
        assert cfg.commission_rate == 0.0003
        assert cfg.top_k == 10
        assert cfg.monthly_addition == 8000
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_backtest/test_config.py -v
```

- [ ] **Step 3: Create source files**

`src/alpha_quat/backtest/__init__.py` (empty)

`src/alpha_quat/backtest/config.py`:
```python
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    start_date: str = "20220501"
    end_date: str = "20260501"
    initial_capital: float = 20000
    monthly_addition: float = 8000
    commission_rate: float = 0.0005
    min_commission: float = 0.0
    stop_loss_pct: float = 0.15
    short_factor: str = "KLEN35"
    long_factor: str = "KLEN36"
    top_k: int = 5
    benchmark: str | None = None
```

- [ ] **Step 4: Run test, expect 2 PASS**

```bash
uv run pytest tests/test_backtest/test_config.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_backtest/test_config.py src/alpha_quat/backtest/ && uv run ruff check --fix tests/test_backtest/test_config.py src/alpha_quat/backtest/
git add tests/test_backtest/ src/alpha_quat/backtest/
git commit -m "feat: add BacktestConfig dataclass"
```

---


### Task 3: Universe filters

**Files:**
- Create: `src/alpha_quat/backtest/filters.py`
- Create: `tests/test_backtest/test_filters.py`

- [ ] **Step 1: Write failing test** — `tests/test_backtest/test_filters.py`

```python
import tempfile
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.filters import build_universe


class TestBuildUniverse:
    def test_excludes_st_stocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            sb = pd.DataFrame({
                "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "market": ["主板", "主板", "主板"],
                "list_status": ["L", "L", "L"],
            })
            sb.to_parquet(data_dir / "stock_basic.parquet")
            st_dir = data_dir / "stock_st"
            st_dir.mkdir()
            st = pd.DataFrame({"ts_code": ["000002.SZ"], "trade_date": ["20240115"]})
            st.to_parquet(st_dir / "2024_01_15.parquet")
            universe = build_universe("20240115", data_dir)
            assert "000001.SZ" in universe
            assert "000002.SZ" not in universe
            assert "000003.SZ" in universe

    def test_excludes_non_main_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            sb = pd.DataFrame({
                "ts_code": ["000001.SZ", "300001.SZ", "688001.SH"],
                "market": ["主板", "创业板", "科创板"],
                "list_status": ["L", "L", "L"],
            })
            sb.to_parquet(data_dir / "stock_basic.parquet")
            (data_dir / "stock_st").mkdir()
            universe = build_universe("20240115", data_dir)
            assert universe == {"000001.SZ"}

    def test_no_st_data_returns_main_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            sb = pd.DataFrame({
                "ts_code": ["000001.SZ", "000002.SZ"],
                "market": ["主板", "主板"],
                "list_status": ["L", "L"],
            })
            sb.to_parquet(data_dir / "stock_basic.parquet")
            (data_dir / "stock_st").mkdir()
            universe = build_universe("20240115", data_dir)
            assert universe == {"000001.SZ", "000002.SZ"}
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_backtest/test_filters.py -v
```

- [ ] **Step 3: Create `src/alpha_quat/backtest/filters.py`**

```python
from pathlib import Path

import pandas as pd


def _date_to_path(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}_{yyyymmdd[4:6]}_{yyyymmdd[6:8]}"


def build_universe(trade_date: str, data_dir: Path) -> set[str]:
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    main_board = set(sb.loc[sb["market"] == "主板", "ts_code"])
    st_path = data_dir / "stock_st" / f"{_date_to_path(trade_date)}.parquet"
    if st_path.exists():
        st = pd.read_parquet(st_path)
        return main_board - set(st["ts_code"])
    return main_board
```

- [ ] **Step 4: Run test, expect 3 PASS**

```bash
uv run pytest tests/test_backtest/test_filters.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_backtest/test_filters.py src/alpha_quat/backtest/filters.py && uv run ruff check --fix tests/test_backtest/test_filters.py src/alpha_quat/backtest/filters.py
git add tests/test_backtest/test_filters.py src/alpha_quat/backtest/filters.py
git commit -m "feat: add universe filter (ST exclusion, main board)"
```

---


### Task 4: Portfolio (Holding + Portfolio)

**Files:**
- Create: `src/alpha_quat/backtest/portfolio.py`
- Create: `tests/test_backtest/test_portfolio.py`

- [ ] **Step 1: Write failing test** — `tests/test_backtest/test_portfolio.py`

```python
import pytest
from alpha_quat.backtest.portfolio import Holding, Portfolio


class TestHolding:
    def test_creation(self):
        h = Holding(ts_code="000001.SZ", shares=500, avg_cost=10.0, buy_date="20240115")
        assert h.ts_code == "000001.SZ"
        assert h.shares == 500
        assert h.avg_cost == 10.0


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(cash=20000)
        assert p.cash == 20000
        assert p.holdings == {}
        assert p.snapshots == []
        assert p.trades == []

    def test_buy_rounds_down_to_lots(self):
        p = Portfolio(cash=20000)
        shares = p.buy("000001.SZ", price=10.0, target_amount=1050.0,
                       trade_date="20240115", commission_rate=0.0005)
        assert shares == 100
        cost = 100 * 10.0
        commission = cost * 0.0005
        assert p.cash == pytest.approx(20000 - cost - commission)
        assert p.holdings["000001.SZ"].shares == 100
        assert p.holdings["000001.SZ"].avg_cost == 10.0

    def test_buy_cannot_afford(self):
        p = Portfolio(cash=50)
        shares = p.buy("000001.SZ", price=100.0, target_amount=1000.0,
                       trade_date="20240115", commission_rate=0.0005)
        assert shares == 0
        assert p.holdings == {}

    def test_buy_min_commission(self):
        p = Portfolio(cash=20000)
        shares = p.buy("000001.SZ", price=10.0, target_amount=1000.0,
                       trade_date="20240115", commission_rate=0.0005,
                       min_commission=5.0)
        assert shares == 100
        assert p.cash == pytest.approx(20000 - 1000 - 5.0)

    def test_sell_full_position(self):
        p = Portfolio(cash=10000)
        p.buy("000001.SZ", price=10.0, target_amount=5000.0,
              trade_date="20240115", commission_rate=0.0005)
        pnl = p.sell("000001.SZ", price=12.0, shares=500,
                     trade_date="20240116", commission_rate=0.0005)
        proceeds = 500 * 12.0
        commission = proceeds * 0.0005
        assert pnl == pytest.approx((12.0 - 10.0) * 500 - commission)
        assert "000001.SZ" not in p.holdings

    def test_sell_partial_position(self):
        p = Portfolio(cash=50000)
        p.buy("000001.SZ", price=10.0, target_amount=10000.0,
              trade_date="20240115", commission_rate=0.0005)
        pnl = p.sell("000001.SZ", price=11.0, shares=300,
                     trade_date="20240116", commission_rate=0.0005)
        assert p.holdings["000001.SZ"].shares == 700
        assert p.holdings["000001.SZ"].avg_cost == 10.0
        assert pnl > 0

    def test_weighted_avg_cost(self):
        p = Portfolio(cash=50000)
        p.buy("000001.SZ", price=10.0, target_amount=10000.0,
              trade_date="20240115", commission_rate=0.0005)
        p.buy("000001.SZ", price=12.0, target_amount=12000.0,
              trade_date="20240116", commission_rate=0.0005)
        assert p.holdings["000001.SZ"].shares == 2000
        assert p.holdings["000001.SZ"].avg_cost == 11.0
        assert p.holdings["000001.SZ"].buy_date == "20240116"

    def test_market_value(self):
        p = Portfolio(cash=10000)
        p.buy("000001.SZ", price=10.0, target_amount=5000.0,
              trade_date="20240115", commission_rate=0.0005)
        mv = p.market_value({"000001.SZ": 11.0})
        assert mv == 500 * 11.0

    def test_record_snapshot(self):
        p = Portfolio(cash=20000)
        p.buy("000001.SZ", price=10.0, target_amount=5000.0,
              trade_date="20240115", commission_rate=0.0005)
        p.record_snapshot("20240115", {"000001.SZ": 11.0})
        assert len(p.snapshots) == 1
        assert p.snapshots[0]["date"] == "20240115"
        assert "cash" in p.snapshots[0]
        assert "market_value" in p.snapshots[0]
        assert "total_value" in p.snapshots[0]
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_backtest/test_portfolio.py -v
```

- [ ] **Step 3: Create `src/alpha_quat/backtest/portfolio.py`**

```python
from dataclasses import dataclass

LOT_SIZE = 100


@dataclass
class Holding:
    ts_code: str
    shares: int
    avg_cost: float
    buy_date: str


class Portfolio:
    def __init__(self, cash: float = 0.0):
        self.cash = cash
        self.holdings: dict[str, Holding] = {}
        self.snapshots: list[dict] = []
        self.trades: list[dict] = []

    def buy(self, ts_code, price, target_amount, trade_date,
            commission_rate, min_commission=0.0):
        if price <= 0:
            return 0
        desired_shares = int(target_amount / price)
        lots = (desired_shares // LOT_SIZE) * LOT_SIZE
        if lots == 0:
            return 0
        while lots > 0:
            trade_cost = lots * price
            commission = max(trade_cost * commission_rate, min_commission)
            if trade_cost + commission <= self.cash:
                break
            lots -= LOT_SIZE
        if lots == 0:
            return 0
        trade_cost = lots * price
        commission = max(trade_cost * commission_rate, min_commission)
        self.cash -= trade_cost + commission
        if ts_code in self.holdings:
            old = self.holdings[ts_code]
            total_shares = old.shares + lots
            new_avg = (old.avg_cost * old.shares + price * lots) / total_shares
            self.holdings[ts_code] = Holding(
                ts_code=ts_code, shares=total_shares,
                avg_cost=new_avg, buy_date=trade_date)
        else:
            self.holdings[ts_code] = Holding(
                ts_code=ts_code, shares=lots,
                avg_cost=price, buy_date=trade_date)
        self.trades.append({
            "date": trade_date, "ts_code": ts_code, "action": "buy",
            "shares": lots, "price": price, "commission": commission, "pnl": 0.0})
        return lots

    def sell(self, ts_code, price, shares, trade_date,
             commission_rate, min_commission=0.0):
        if ts_code not in self.holdings:
            return 0.0
        holding = self.holdings[ts_code]
        actual = min(shares, holding.shares)
        if actual == 0:
            return 0.0
        proceeds = actual * price
        commission = max(proceeds * commission_rate, min_commission)
        realized_pnl = (price - holding.avg_cost) * actual - commission
        self.cash += proceeds - commission
        remaining = holding.shares - actual
        if remaining == 0:
            del self.holdings[ts_code]
        else:
            self.holdings[ts_code] = Holding(
                ts_code=ts_code, shares=remaining,
                avg_cost=holding.avg_cost, buy_date=holding.buy_date)
        self.trades.append({
            "date": trade_date, "ts_code": ts_code, "action": "sell",
            "shares": actual, "price": price, "commission": commission,
            "pnl": realized_pnl})
        return realized_pnl

    def market_value(self, prices):
        total = 0.0
        for code, h in self.holdings.items():
            px = prices.get(code)
            if px is not None:
                total += h.shares * px
        return total

    def total_value(self, prices):
        return self.cash + self.market_value(prices)

    def record_snapshot(self, date, prices):
        mv = self.market_value(prices)
        self.snapshots.append({
            "date": date, "cash": self.cash,
            "market_value": mv, "total_value": self.cash + mv})
```

- [ ] **Step 4: Run test, expect 10 PASS**

```bash
uv run pytest tests/test_backtest/test_portfolio.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_backtest/test_portfolio.py src/alpha_quat/backtest/portfolio.py && uv run ruff check --fix tests/test_backtest/test_portfolio.py src/alpha_quat/backtest/portfolio.py
git add tests/test_backtest/test_portfolio.py src/alpha_quat/backtest/portfolio.py
git commit -m "feat: add Portfolio with buy/sell/valuation"
```

---


### Task 5: Metrics

**Files:**
- Create: `src/alpha_quat/backtest/metrics.py`
- Create: `tests/test_backtest/test_metrics.py`

- [ ] **Step 1: Write failing test** — `tests/test_backtest/test_metrics.py`

```python
import pytest
from alpha_quat.backtest.metrics import compute_metrics


class TestComputeMetrics:
    def test_zero_return(self):
        snapshots = [
            {"date": "d1", "cash": 100000, "market_value": 0, "total_value": 100000},
            {"date": "d2", "cash": 100000, "market_value": 0, "total_value": 100000},
        ]
        result = compute_metrics(snapshots, [], total_invested=100000)
        assert result["cumulative_return"] == pytest.approx(0.0)
        assert result["final_value"] == 100000

    def test_positive_return(self):
        snapshots = [
            {"date": "d1", "cash": 100000, "market_value": 0, "total_value": 100000},
            {"date": "d365", "cash": 120000, "market_value": 0, "total_value": 120000},
        ]
        result = compute_metrics(snapshots, [], total_invested=100000)
        assert result["cumulative_return"] == pytest.approx(0.20)
        assert result["annualized_return"] == pytest.approx(0.095, abs=0.01)

    def test_drawdown(self):
        snapshots = [
            {"date": "d1", "cash": 0, "market_value": 100, "total_value": 100},
            {"date": "d2", "cash": 0, "market_value": 90, "total_value": 90},
            {"date": "d3", "cash": 0, "market_value": 95, "total_value": 95},
            {"date": "d4", "cash": 0, "market_value": 80, "total_value": 80},
            {"date": "d5", "cash": 0, "market_value": 85, "total_value": 85},
        ]
        result = compute_metrics(snapshots, [], total_invested=100)
        assert result["max_drawdown"] == pytest.approx(-0.20)
        assert result["max_drawdown_date"] == "d4"

    def test_win_rate(self):
        trades = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 200.0}]
        result = compute_metrics(
            [{"date": "d1", "cash": 0, "market_value": 100, "total_value": 100}],
            trades, total_invested=100)
        assert result["win_rate"] == pytest.approx(2 / 3)

    def test_no_trades(self):
        result = compute_metrics(
            [{"date": "d1", "cash": 0, "market_value": 100, "total_value": 100}],
            [], total_invested=100)
        assert result["win_rate"] == 0.0
        assert result["total_trades"] == 0

    def test_sharpe_zero_vol(self):
        snapshots = [
            {"date": "d1", "cash": 0, "market_value": 100, "total_value": 100},
            {"date": "d2", "cash": 0, "market_value": 100, "total_value": 100},
        ]
        result = compute_metrics(snapshots, [], total_invested=100)
        assert result["sharpe_ratio"] == 0.0
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_backtest/test_metrics.py -v
```

- [ ] **Step 3: Create `src/alpha_quat/backtest/metrics.py`**

```python
import math


def compute_metrics(snapshots, trades, total_invested, risk_free_rate=0.025):
    if not snapshots:
        return {
            "cumulative_return": 0.0, "annualized_return": 0.0,
            "max_drawdown": 0.0, "max_drawdown_date": None,
            "sharpe_ratio": 0.0, "win_rate": 0.0, "total_trades": 0,
            "final_value": 0.0, "total_invested": total_invested}

    final_value = snapshots[-1]["total_value"]
    cumulative_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0.0

    n_dates = len(snapshots)
    if n_dates >= 2 and total_invested > 0 and final_value > 0:
        years = n_dates / 252.0
        annualized_return = (final_value / total_invested) ** (1.0 / years) - 1.0
    else:
        annualized_return = 0.0

    max_drawdown = 0.0
    max_drawdown_date = None
    peak = snapshots[0]["total_value"]
    for s in snapshots:
        tv = s["total_value"]
        if tv > peak:
            peak = tv
        dd = (tv - peak) / peak if peak > 0 else 0.0
        if dd < max_drawdown:
            max_drawdown = dd
            max_drawdown_date = s["date"]

    daily_returns = []
    for i in range(1, len(snapshots)):
        prev_tv = snapshots[i - 1]["total_value"]
        curr_tv = snapshots[i]["total_value"]
        if prev_tv > 0:
            daily_returns.append(curr_tv / prev_tv - 1.0)

    if len(daily_returns) >= 2:
        mean_ret = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean_ret * 252 - risk_free_rate) / (std * math.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    if trades:
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    return {
        "cumulative_return": cumulative_return, "annualized_return": annualized_return,
        "max_drawdown": max_drawdown, "max_drawdown_date": max_drawdown_date,
        "sharpe_ratio": sharpe, "win_rate": win_rate,
        "total_trades": len(trades), "final_value": final_value,
        "total_invested": total_invested}
```

- [ ] **Step 4: Run test, expect 6 PASS**

```bash
uv run pytest tests/test_backtest/test_metrics.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_backtest/test_metrics.py src/alpha_quat/backtest/metrics.py && uv run ruff check --fix tests/test_backtest/test_metrics.py src/alpha_quat/backtest/metrics.py
git add tests/test_backtest/test_metrics.py src/alpha_quat/backtest/metrics.py
git commit -m "feat: add performance metrics (return, drawdown, sharpe, win rate)"
```

---


### Task 6: MACrossSignal

**Files:**
- Create: `src/alpha_quat/strategy/signals/__init__.py`
- Create: `src/alpha_quat/strategy/signals/ma_cross.py`
- Create: `tests/test_strategy/test_ma_cross.py`

- [ ] **Step 1: Write failing test** — `tests/test_strategy/test_ma_cross.py`

```python
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.signals.ma_cross import MACrossSignal


class TestMACrossSignal:
    def test_no_prev_returns_empty(self):
        signal = MACrossSignal()
        features = pd.DataFrame({
            "ts_code": ["A", "B"], "KLEN35": [1.05, 0.95], "KLEN36": [1.00, 1.00]})
        ctx = StrategyContext(trade_date="d", capital=1e5)
        result = signal.generate(features, ctx)
        assert len(result.signals) == 0

    def test_golden_cross_buy(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [0.98], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.02], "KLEN36": [1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 1
        row = result.signals.iloc[0]
        assert row["ts_code"] == "A"
        assert row["action"] == "buy"
        assert row["score"] == 1.0

    def test_dead_cross_sell(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.02], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({"ts_code": ["A"], "KLEN35": [0.98], "KLEN36": [1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 1
        assert result.signals.iloc[0]["action"] == "sell"
        assert result.signals.iloc[0]["score"] == 0.0

    def test_no_cross_no_signal(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.05], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.06], "KLEN36": [1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 0

    def test_new_stock_no_signal(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.02], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({
            "ts_code": ["A", "B"], "KLEN35": [1.02, 1.05], "KLEN36": [1.00, 1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 0

    def test_metadata(self):
        signal = MACrossSignal(short_factor="A", long_factor="B")
        features = pd.DataFrame({"ts_code": ["X"], "A": [1.05], "B": [1.00]})
        result = signal.generate(features, StrategyContext(trade_date="d1", capital=1e5))
        assert result.metadata["signal_name"] == "ma_cross"
        assert result.metadata["short_factor"] == "A"
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_strategy/test_ma_cross.py -v
```

- [ ] **Step 3: Create source files**

`src/alpha_quat/strategy/signals/__init__.py` (empty)

`src/alpha_quat/strategy/signals/ma_cross.py`:
```python
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.signal import ISignalGenerator


class MACrossSignal(ISignalGenerator):
    def __init__(self, short_factor="KLEN35", long_factor="KLEN36"):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self._prev: pd.DataFrame | None = None

    def generate(self, features, ctx):
        cur = features[["ts_code", self.short_factor, self.long_factor]].copy()
        cur["cross_above"] = cur[self.short_factor] > cur[self.long_factor]

        buy_codes = []
        sell_codes = []
        if self._prev is not None and not self._prev.empty:
            merged = cur.merge(
                self._prev[["ts_code", "cross_above"]],
                on="ts_code", how="inner", suffixes=("", "_prev"))
            golden = merged.loc[
                merged["cross_above"] & ~merged["cross_above_prev"], "ts_code"]
            dead = merged.loc[
                ~merged["cross_above"] & merged["cross_above_prev"], "ts_code"]
            buy_codes = golden.tolist()
            sell_codes = dead.tolist()

        self._prev = cur[["ts_code", "cross_above"]]

        rows = []
        for c in buy_codes:
            rows.append({"ts_code": c, "action": "buy", "score": 1.0})
        for c in sell_codes:
            rows.append({"ts_code": c, "action": "sell", "score": 0.0})

        if rows:
            signals_df = pd.DataFrame(rows, columns=["ts_code", "action", "score"])
        else:
            signals_df = pd.DataFrame(columns=["ts_code", "action", "score"])

        return SignalResult(
            signals=signals_df,
            metadata={"signal_name": "ma_cross",
                      "short_factor": self.short_factor,
                      "long_factor": self.long_factor})
```

- [ ] **Step 4: Run test, expect 6 PASS**

```bash
uv run pytest tests/test_strategy/test_ma_cross.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_strategy/test_ma_cross.py src/alpha_quat/strategy/signals/ && uv run ruff check --fix tests/test_strategy/test_ma_cross.py src/alpha_quat/strategy/signals/
git add tests/test_strategy/test_ma_cross.py src/alpha_quat/strategy/signals/
git commit -m "feat: add MACrossSignal (golden/dead cross detection)"
```

---


### Task 7: EqualWeightTopKPosition

**Files:**
- Create: `src/alpha_quat/strategy/positions/__init__.py`
- Create: `src/alpha_quat/strategy/positions/equal_weight.py`
- Create: `tests/test_strategy/test_equal_weight.py`

- [ ] **Step 1: Write failing test** — `tests/test_strategy/test_equal_weight.py`

```python
import pytest
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.positions.equal_weight import EqualWeightTopKPosition


class TestEqualWeightTopKPosition:
    def test_allocate_top_k(self):
        pm = EqualWeightTopKPosition(top_k=3)
        signals = SignalResult(signals=pd.DataFrame({
            "ts_code": ["A", "B", "C", "D"],
            "action": ["buy"] * 4, "score": [0.9, 0.8, 0.7, 0.6]}))
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 3
        assert pos["target_weight"].sum() == pytest.approx(1.0)

    def test_allocate_fewer_than_k(self):
        pm = EqualWeightTopKPosition(top_k=5)
        signals = SignalResult(signals=pd.DataFrame({
            "ts_code": ["A", "B"], "action": ["buy", "buy"], "score": [0.9, 0.8]}))
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 2
        assert pos["target_weight"].iloc[0] == 0.5
        assert pos["target_weight"].sum() == pytest.approx(1.0)

    def test_allocate_sells_ignored(self):
        pm = EqualWeightTopKPosition(top_k=3)
        signals = SignalResult(signals=pd.DataFrame({
            "ts_code": ["A", "B"], "action": ["buy", "sell"], "score": [0.9, 0.0]}))
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 1
        assert pos["ts_code"].iloc[0] == "A"

    def test_allocate_no_buys(self):
        pm = EqualWeightTopKPosition(top_k=3)
        signals = SignalResult(signals=pd.DataFrame({
            "ts_code": ["A"], "action": ["sell"], "score": [0.0]}))
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 0

    def test_constrain_universe(self):
        pm = EqualWeightTopKPosition(top_k=3)
        pos = pd.DataFrame({"ts_code": ["A", "B", "C"], "target_weight": [0.4, 0.3, 0.3]})
        ctx = StrategyContext(trade_date="d", capital=1e5, universe=["A", "C"])
        result = pm.constrain(pos, ctx)
        assert set(result["ts_code"]) == {"A", "C"}
        assert result["target_weight"].sum() == pytest.approx(1.0)

    def test_execute_shares(self):
        pm = EqualWeightTopKPosition(top_k=3)
        target = pd.DataFrame({"ts_code": ["A", "B"], "target_weight": [0.6, 0.4]})
        ctx = StrategyContext(
            trade_date="d", capital=100000.0,
            prices=pd.DataFrame({"ts_code": ["A", "B"], "open": [10.0, 20.0]}))
        pos, orders = pm.execute(target, None, ctx)
        assert pos.loc[pos["ts_code"] == "A", "target_shares"].iloc[0] == 6000
        assert pos.loc[pos["ts_code"] == "B", "target_shares"].iloc[0] == 2000
        assert len(orders) == 2

    def test_execute_rounding(self):
        pm = EqualWeightTopKPosition(top_k=3)
        target = pd.DataFrame({"ts_code": ["A"], "target_weight": [1.0]})
        ctx = StrategyContext(
            trade_date="d", capital=1500.0,
            prices=pd.DataFrame({"ts_code": ["A"], "open": [10.0]}))
        pos, _ = pm.execute(target, None, ctx)
        assert pos["target_shares"].iloc[0] == 100

    def test_empty_target(self):
        pm = EqualWeightTopKPosition(top_k=3)
        target = pd.DataFrame(columns=["ts_code", "target_weight"])
        pos, orders = pm.execute(target, None, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 0
        assert len(orders) == 0
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_strategy/test_equal_weight.py -v
```

- [ ] **Step 3: Create source files**

`src/alpha_quat/strategy/positions/__init__.py` (empty)

`src/alpha_quat/strategy/positions/equal_weight.py`:
```python
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.position import IPositionManager

LOT_SIZE = 100


class EqualWeightTopKPosition(IPositionManager):
    def __init__(self, top_k=5):
        self.top_k = top_k

    def allocate(self, signals, ctx):
        df = signals.signals
        buys = df.loc[df["action"] == "buy"].copy()
        if buys.empty:
            return pd.DataFrame(columns=["ts_code", "target_weight"])
        buys = buys.sort_values("score", ascending=False).head(self.top_k)
        weight = 1.0 / len(buys)
        return pd.DataFrame({
            "ts_code": buys["ts_code"].values,
            "target_weight": [weight] * len(buys)})

    def constrain(self, positions, ctx):
        if positions.empty:
            return positions
        result = positions.copy()
        if ctx.universe is not None:
            result = result.loc[result["ts_code"].isin(ctx.universe)]
        if not result.empty:
            total = result["target_weight"].sum()
            if total > 0:
                result["target_weight"] = result["target_weight"] / total
        return result.reset_index(drop=True)

    def execute(self, target, prev, ctx):
        if target.empty:
            return (
                pd.DataFrame(columns=["ts_code", "target_weight", "target_shares", "target_amount"]),
                pd.DataFrame(columns=["ts_code", "action", "delta_shares", "delta_amount"]))
        prices = ctx.prices
        if prices is None:
            raise ValueError("ctx.prices is required")
        price_map = dict(zip(prices["ts_code"], prices["open"]))
        rows = []
        for _, row in target.iterrows():
            code = row["ts_code"]
            px = price_map.get(code)
            if px is None or px <= 0:
                continue
            shares = int(row["target_weight"] * ctx.capital / px)
            shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares == 0:
                continue
            rows.append({"ts_code": code, "target_weight": row["target_weight"],
                        "target_shares": shares, "target_amount": shares * px})
        pos_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["ts_code", "target_weight", "target_shares", "target_amount"])
        orders = [{"ts_code": r["ts_code"], "action": "buy",
                   "delta_shares": r["target_shares"], "delta_amount": r["target_amount"]}
                  for r in rows]
        orders_df = pd.DataFrame(orders) if orders else pd.DataFrame(
            columns=["ts_code", "action", "delta_shares", "delta_amount"])
        return pos_df, orders_df
```

- [ ] **Step 4: Run test, expect 8 PASS**

```bash
uv run pytest tests/test_strategy/test_equal_weight.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_strategy/test_equal_weight.py src/alpha_quat/strategy/positions/ && uv run ruff check --fix tests/test_strategy/test_equal_weight.py src/alpha_quat/strategy/positions/
git add tests/test_strategy/test_equal_weight.py src/alpha_quat/strategy/positions/
git commit -m "feat: add EqualWeightTopKPosition with top-K allocation"
```

---


### Task 8: BacktestEngine

**Files:**
- Create: `src/alpha_quat/backtest/engine.py`
- Create: `tests/test_backtest/test_engine.py`

- [ ] **Step 1: Write failing test** — `tests/test_backtest/test_engine.py`

```python
import pytest
import tempfile
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine


def _make_minimal_data(data_dir, dates):
    daily_dir = data_dir / "daily"
    daily_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir()
    st_dir = data_dir / "stock_st"
    st_dir.mkdir(parents=True)
    for d in dates:
        d_int = int(d)
        d_path = f"{d[:4]}_{d[4:6]}_{d[6:8]}"
        daily = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [d_int, d_int],
            "open": [10.0, 20.0], "high": [10.5, 20.5],
            "low": [9.5, 19.5], "close": [10.2, 20.2],
            "pre_close": [10.0, 20.0], "change": [0.2, 0.2],
            "pct_chg": [2.0, 1.0], "vol": [1e5, 2e5],
            "amount": [1.02e6, 4.04e6]})
        daily.to_parquet(daily_dir / f"{d_path}.parquet")
        features = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [d, d],
            "KLEN35": [1.00, 1.00], "KLEN36": [1.00, 1.00]})
        features.to_parquet(features_dir / f"{d}.parquet")
        pd.DataFrame(columns=["ts_code"]).to_parquet(st_dir / f"{d_path}.parquet")
    sb = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"],
        "market": ["主板", "主板"], "list_status": ["L", "L"]})
    sb.to_parquet(data_dir / "stock_basic.parquet")
    cal = pd.DataFrame({"cal_date": dates, "is_open": [1] * len(dates)})
    cal.to_parquet(data_dir / "trade_cal.parquet")


class TestBacktestEngine:
    def test_runs_and_produces_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _make_minimal_data(data_dir, ["20240115", "20240116", "20240117"])
            config = BacktestConfig(
                start_date="20240115", end_date="20240117",
                initial_capital=100000, monthly_addition=0)
            engine = BacktestEngine(config, data_dir)
            result = engine.run()
            assert len(result["snapshots"]) == 3
            assert result["metrics"]["total_invested"] == 100000
            assert result["metrics"]["final_value"] > 0

    def test_monthly_addition(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _make_minimal_data(data_dir, ["20240115", "20240201"])
            config = BacktestConfig(
                start_date="20240115", end_date="20240201",
                initial_capital=100000, monthly_addition=5000)
            engine = BacktestEngine(config, data_dir)
            result = engine.run()
            assert result["metrics"]["total_invested"] == 105000
            assert len(result["snapshots"]) == 2

    def test_missing_trade_cal_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            engine = BacktestEngine(BacktestConfig(), data_dir)
            with pytest.raises(FileNotFoundError, match="trade_cal"):
                engine.run()
```

- [ ] **Step 2: Run test, expect ModuleNotFoundError**

```bash
uv run pytest tests/test_backtest/test_engine.py -v
```

- [ ] **Step 3: Create `src/alpha_quat/backtest/engine.py`**

```python
import logging
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.filters import build_universe
from alpha_quat.backtest.portfolio import Portfolio
from alpha_quat.backtest.metrics import compute_metrics
from alpha_quat.strategy.types import StrategyContext
from alpha_quat.strategy.signals.ma_cross import MACrossSignal
from alpha_quat.strategy.positions.equal_weight import EqualWeightTopKPosition

logger = logging.getLogger(__name__)


def _ymd_to_path(ymd: str) -> str:
    return f"{ymd[:4]}_{ymd[4:6]}_{ymd[6:8]}"


class BacktestEngine:
    def __init__(self, config: BacktestConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.portfolio = Portfolio(cash=config.initial_capital)
        self.signal_gen = MACrossSignal(
            short_factor=config.short_factor, long_factor=config.long_factor)
        self.position_mgr = EqualWeightTopKPosition(top_k=config.top_k)
        self._pending_signals = None
        self._total_invested = config.initial_capital

    def run(self):
        cal_path = self.data_dir / "trade_cal.parquet"
        if not cal_path.exists():
            raise FileNotFoundError(
                "trade_cal.parquet not found. Run 'alpha-quat fetch' first.")

        cal = pd.read_parquet(cal_path)
        all_dates = sorted(
            cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())
        dates = [d for d in all_dates
                 if self.config.start_date <= d <= self.config.end_date]

        if not dates:
            logger.warning("No trading dates in range")
            return self._result()

        tracked_months: set[str] = set()
        self.signal_gen._prev = None
        self._pending_signals = None

        for idx, td in enumerate(dates):
            month_key = td[:6]
            if month_key not in tracked_months:
                tracked_months.add(month_key)
                if idx > 0:
                    self.portfolio.cash += self.config.monthly_addition
                    self._total_invested += self.config.monthly_addition

            daily_path = self.data_dir / "daily" / f"{_ymd_to_path(td)}.parquet"
            if not daily_path.exists():
                logger.warning("No daily data for %s, skipping", td)
                continue

            daily = pd.read_parquet(daily_path)
            open_px = dict(zip(daily["ts_code"], daily["open"]))
            close_px = dict(zip(daily["ts_code"], daily["close"]))
            universe = build_universe(td, self.data_dir)

            # Stop loss at today's open
            for code, h in list(self.portfolio.holdings.items()):
                prev_close = close_px.get(code)
                if prev_close and prev_close < h.avg_cost * (1.0 - self.config.stop_loss_pct):
                    px = open_px.get(code)
                    if px and px > 0 and code in universe:
                        self.portfolio.sell(
                            ts_code=code, price=px, shares=h.shares,
                            trade_date=td, commission_rate=self.config.commission_rate,
                            min_commission=self.config.min_commission)

            # Execute pending signals from yesterday at today's open
            if self._pending_signals is not None and not self._pending_signals.signals.empty:
                sig_df = self._pending_signals.signals
                sig_df = sig_df.loc[sig_df["ts_code"].isin(universe)]

                prices_df = daily[["ts_code", "open"]].copy()
                ctx = StrategyContext(
                    trade_date=td, capital=self.portfolio.total_value(open_px),
                    universe=list(universe), prices=prices_df)

                # Buys
                buy_sigs = sig_df.loc[sig_df["action"] == "buy"]
                if not buy_sigs.empty:
                    fake_result = type(self._pending_signals)(
                        signals=buy_sigs.reset_index(drop=True),
                        metadata=self._pending_signals.metadata)
                    alloc = self.position_mgr.allocate(fake_result, ctx)
                    alloc = self.position_mgr.constrain(alloc, ctx)
                    for _, row in alloc.iterrows():
                        code = row["ts_code"]
                        px = open_px.get(code)
                        if px and px > 0:
                            target_amt = row["target_weight"] * ctx.capital
                            self.portfolio.buy(
                                ts_code=code, price=px, target_amount=target_amt,
                                trade_date=td, commission_rate=self.config.commission_rate,
                                min_commission=self.config.min_commission)

                # Sells
                sell_sigs = sig_df.loc[sig_df["action"] == "sell"]
                for _, row in sell_sigs.iterrows():
                    code = row["ts_code"]
                    if code in self.portfolio.holdings:
                        h = self.portfolio.holdings[code]
                        px = open_px.get(code)
                        if px and px > 0:
                            self.portfolio.sell(
                                ts_code=code, price=px, shares=h.shares,
                                trade_date=td, commission_rate=self.config.commission_rate,
                                min_commission=self.config.min_commission)

            # Generate today's signals (EOD, execute tomorrow)
            feat_path = self.data_dir / "features" / f"{td}.parquet"
            if feat_path.exists():
                features = pd.read_parquet(feat_path)
                features = features.loc[features["ts_code"].isin(universe)]
                ctx_sig = StrategyContext(
                    trade_date=td, capital=self.portfolio.total_value(close_px))
                self._pending_signals = self.signal_gen.generate(features, ctx_sig)
            else:
                self._pending_signals = None

            self.portfolio.record_snapshot(td, close_px)

        return self._result()

    def _result(self):
        metrics = compute_metrics(
            snapshots=self.portfolio.snapshots,
            trades=self.portfolio.trades,
            total_invested=self._total_invested)
        return {"snapshots": self.portfolio.snapshots,
                "trades": self.portfolio.trades, "metrics": metrics}
```

- [ ] **Step 4: Run test, expect 3 PASS**

```bash
uv run pytest tests/test_backtest/test_engine.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format tests/test_backtest/test_engine.py src/alpha_quat/backtest/engine.py && uv run ruff check --fix tests/test_backtest/test_engine.py src/alpha_quat/backtest/engine.py
git add tests/test_backtest/test_engine.py src/alpha_quat/backtest/engine.py
git commit -m "feat: add BacktestEngine with day-loop orchestration"
```

---


### Task 9: HTML Report

**Files:**
- Create: `src/alpha_quat/backtest/report.py`

Note: No separate test file -- tested via CLI integration in Task 10. See `src/alpha_quat/backtest/report.py` for the full implementation (generates self-contained HTML with matplotlib equity curve and drawdown charts embedded as base64 images, summary cards, trade log table, and config display).

The report module provides one function:

```python
def generate_html_report(result: dict, config: BacktestConfig, output_path: Path) -> Path:
```

It creates a single self-contained HTML file with:
- Summary cards (cumulative return, annualized return, Sharpe, max drawdown, win rate, total trades, final value)
- Equity curve chart (matplotlib, embedded as base64 PNG)
- Drawdown chart (matplotlib, embedded as base64 PNG)
- Trade log table (up to 200 rows)
- Configuration summary table

- [ ] **Step 1: Create `src/alpha_quat/backtest/report.py`**

```python
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def generate_html_report(result, config, output_path):
    snapshots = result["snapshots"]
    trades = result["trades"]
    metrics = result["metrics"]

    dates = [s["date"] for s in snapshots]
    nav = [s["total_value"] for s in snapshots]
    invested_line = [metrics["total_invested"]] * len(nav)

    equity_img = _make_equity_chart(dates, nav, invested_line)
    dd_img = _make_drawdown_chart(nav, dates)
    trade_html = _build_trade_table(trades)
    config_html = _build_config_table(config)

    cum_ret = metrics["cumulative_return"] * 100
    ann_ret = metrics["annualized_return"] * 100
    max_dd = metrics["max_drawdown"] * 100
    ret_color = "green" if cum_ret >= 0 else "red"

    html = _HTML_TEMPLATE.format(
        cum_ret=cum_ret, ann_ret=ann_ret,
        max_dd=max_dd,
        sharpe=metrics["sharpe_ratio"],
        win_rate=metrics["win_rate"] * 100,
        total_trades=metrics["total_trades"],
        final_value=metrics["final_value"],
        ret_color=ret_color,
        equity_img=equity_img, dd_img=dd_img,
        trade_html=trade_html, config_html=config_html)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _make_equity_chart(dates, nav, invested_line):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, nav, label="Portfolio NAV", color="#2c7fb8", linewidth=1.2)
    ax.plot(dates, invested_line, label="Total Invested",
            color="#999999", linestyle="--", linewidth=1.0)
    above = [n >= i for n, i in zip(nav, invested_line)]
    below = [n < i for n, i in zip(nav, invested_line)]
    if any(above):
        ax.fill_between(range(len(dates)), nav, invested_line,
                        where=above, color="#2c7fb8", alpha=0.1)
    if any(below):
        ax.fill_between(range(len(dates)), nav, invested_line,
                        where=below, color="#d7191c", alpha=0.1)
    ax.set_title("Equity Curve", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/10000:.1f}w"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                       rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def _make_drawdown_chart(nav, dates):
    peak = nav[0]
    dd_vals = []
    for v in nav:
        if v > peak:
            peak = v
        dd_vals.append((v - peak) / peak * 100 if peak > 0 else 0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(dates)), dd_vals, 0, color="#d7191c", alpha=0.3)
    ax.plot(dd_vals, color="#d7191c", linewidth=1)
    ax.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(True, alpha=0.3)
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                       rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def _build_trade_table(trades):
    rows = []
    for t in trades[:200]:
        pnl_str = f'{t["pnl"]:+.2f}'
        color = "green" if t["pnl"] > 0 else ("red" if t["pnl"] < 0 else "gray")
        rows.append(
            f'<tr><td>{t["date"]}</td><td>{t["ts_code"]}</td>'
            f'<td>{t["action"]}</td><td>{t["price"]:.2f}</td>'
            f'<td>{t["shares"]}</td><td>{t["commission"]:.2f}</td>'
            f'<td style="color:{color}">{pnl_str}</td></tr>')
    return "".join(rows)


def _build_config_table(config):
    return (
        f'<tr><td>Period</td><td>{config.start_date} ~ {config.end_date}</td></tr>'
        f'<tr><td>Initial Capital</td><td>{config.initial_capital:,.0f}</td></tr>'
        f'<tr><td>Monthly Addition</td><td>{config.monthly_addition:,.0f}</td></tr>'
        f'<tr><td>Commission</td><td>{config.commission_rate*10000:.1f} bp</td></tr>'
        f'<tr><td>Stop Loss</td><td>{config.stop_loss_pct*100:.0f}%</td></tr>'
        f'<tr><td>MA Factors</td><td>{config.short_factor} / {config.long_factor}</td></tr>'
        f'<tr><td>Max Holdings</td><td>{config.top_k}</td></tr>')


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f5f5; color: #333; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 20px; color: #1a1a2e; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;
            margin-bottom: 24px; }}
  .card {{ background: white; border-radius: 8px; padding: 16px 20px;
           min-width: 130px; text-align: center;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
  .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 16px;
                      margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                      text-align: center; }}
  .chart-container img {{ max-width: 100%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  tr:hover {{ background: #f8f9fa; }}
  h2 {{ font-size: 16px; margin: 24px 0 12px; color: #1a1a2e; }}
  .section {{ background: white; border-radius: 8px; padding: 16px 20px;
              margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>Backtest Report</h1>

<div class="cards">
  <div class="card"><div class="label">Cumulative Return</div>
    <div class="value" style="color:{ret_color}">{cum_ret:.2f}%</div></div>
  <div class="card"><div class="label">Annualized Return</div>
    <div class="value" style="color:{ret_color}">{ann_ret:.2f}%</div></div>
  <div class="card"><div class="label">Sharpe Ratio</div>
    <div class="value">{sharpe:.2f}</div></div>
  <div class="card"><div class="label">Max Drawdown</div>
    <div class="value" style="color:red">{max_dd:.2f}%</div></div>
  <div class="card"><div class="label">Win Rate</div>
    <div class="value">{win_rate:.1f}%</div></div>
  <div class="card"><div class="label">Total Trades</div>
    <div class="value">{total_trades}</div></div>
  <div class="card"><div class="label">Final Value</div>
    <div class="value">{final_value:,.0f}</div></div>
</div>

<div class="chart-container"><h2>Equity Curve</h2>
  <img src="data:image/png;base64,{equity_img}" alt="Equity Curve"></div>

<div class="chart-container"><h2>Drawdown</h2>
  <img src="data:image/png;base64,{dd_img}" alt="Drawdown"></div>

<div class="section"><h2>Trade Log</h2><table>
  <thead><tr><th>Date</th><th>Code</th><th>Action</th><th>Price</th>
    <th>Shares</th><th>Commission</th><th>P&amp;L</th></tr></thead>
  <tbody>{trade_html}</tbody></table></div>

<div class="section"><h2>Configuration</h2><table>
  {config_html}</table></div>
</body></html>"""
```

- [ ] **Step 2: Lint and commit**

```bash
uv run ruff format src/alpha_quat/backtest/report.py && uv run ruff check --fix src/alpha_quat/backtest/report.py
git add src/alpha_quat/backtest/report.py
git commit -m "feat: add HTML report generation with equity/drawdown charts"
```

---


### Task 10: CLI backtest subcommand

**Files:**
- Modify: `src/alpha_quat/cli.py`

- [ ] **Step 1: Add backtest support to `cli.py`**

Add import at top (after existing imports):
```python
from pathlib import Path
from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine
from alpha_quat.backtest.report import generate_html_report
```

Add parser builder:
```python
def _build_backtest_parser(subparsers):
    parser = subparsers.add_parser("backtest", help="Run strategy backtest")
    parser.add_argument("--start", default="20220501", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20260501", help="End date YYYYMMDD")
    parser.add_argument("--capital", type=float, default=20000, help="Initial capital")
    parser.add_argument("--monthly", type=float, default=8000, help="Monthly addition")
    parser.add_argument("--commission", type=float, default=0.0005, help="Commission rate")
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop loss pct")
    parser.add_argument("--top-k", type=int, default=5, help="Max holdings")
    parser.add_argument("--output", default=None, help="HTML report output path")
    return parser
```

Add command handler:
```python
def _cmd_backtest(args, config):
    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        monthly_addition=args.monthly,
        commission_rate=args.commission,
        stop_loss_pct=args.stop_loss,
        top_k=args.top_k,
    )
    engine = BacktestEngine(cfg, config.data_dir)
    result = engine.run()

    metrics = result["metrics"]
    print()
    print("=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Period:          {cfg.start_date} ~ {cfg.end_date}")
    print(f"  Total Invested:  {metrics['total_invested']:,.0f}")
    print(f"  Final Value:     {metrics['final_value']:,.2f}")
    print(f"  Cumulative Ret:  {metrics['cumulative_return']*100:+.2f}%")
    print(f"  Annualized Ret:  {metrics['annualized_return']*100:+.2f}%")
    print(f"  Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate:        {metrics['win_rate']*100:.1f}%")
    print(f"  Total Trades:    {metrics['total_trades']}")
    print("=" * 50)
    print()

    output_path = (
        Path(args.output) if args.output
        else config.data_dir / "backtest_report.html")
    generate_html_report(result, cfg, output_path)
    print(f"Report saved to: {output_path}")
```

In `main()`, register the parser (after `_build_feature_parser(subparsers)`):
```python
    _build_backtest_parser(subparsers)
```

In `main()`, add dispatch (after `elif args.command == "feature":`):
```python
    elif args.command == "backtest":
        _cmd_backtest(args, config)
```

- [ ] **Step 2: Verify CLI help**

```bash
uv run alpha-quat backtest --help
```

Expected: Shows backtest options.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff format src/alpha_quat/cli.py && uv run ruff check --fix src/alpha_quat/cli.py
git add src/alpha_quat/cli.py
git commit -m "feat: add backtest CLI subcommand with HTML report"
```

---


### Task 11: Full verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest --cov=src -v
```

Expected: All tests pass. Check for unexpected failures in existing tests.

- [ ] **Step 2: Run typecheck**

```bash
uv run pyright
```

Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Full verification chain**

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright && uv run pytest --cov=src
```

Expected: All clean, all tests pass.

- [ ] **Step 4: Final commit**

```bash
git status
git add -A
git commit -m "chore: final verification - all tests pass, typecheck clean"
```

---


# Strategy Base Class Design

## Overview

Design an abstract quantitative strategy framework that separates signal generation from position management, composed via a Pipeline pattern. The strategy receives feature engineering results directly from the external feature engine — no file I/O, no data fetching inside strategy code.

## Architecture

```
feature DataFrame (ts_code, trade_date, factor1, factor2, ...)
    │
    ▼
ISignalGenerator.generate(df, ctx)
    │
    ▼
SignalResult (ts_code, score, side, ...)
    │
    ▼
IPositionManager.allocate(signals, ctx)
    │
    ▼
positions DataFrame (ts_code, target_weight, ...)
    │
    ▼
IPositionManager.constrain(positions, ctx)
    │
    ▼
positions DataFrame (constrained, same schema)
    │
    ▼
IPositionManager.execute(positions, prev_holdings, ctx)
    │
    ▼
(positions_with_shares, orders)   # execute computes target_shares + delta_shares
    │
    ▼
StrategyResult(target_positions, orders, metadata)
```

## File Layout

```
src/alpha_quat/strategy/
├── __init__.py
├── types.py          # StrategyContext, SignalResult, StrategyResult
├── signal.py         # ISignalGenerator (ABC)
├── position.py       # IPositionManager (ABC)
└── strategy.py       # Strategy (concrete pipeline orchestrator)
```

## Data Types (`types.py`)

### StrategyContext

Runtime context injected by the caller (backtest engine / live trader) for each trading day.

```python
@dataclass
class StrategyContext:
    trade_date: str                     # YYYYMMDD
    capital: float                      # total capital
    universe: list[str] | None = None   # stock pool; None = all stocks
    prices: pd.DataFrame | None = None  # reference price for share calc (ts_code, close/vwap)
    prev_holdings: pd.DataFrame | None = None  # previous holdings (ts_code, shares, cost)
    constraints: dict | None = None     # e.g. {max_single_weight, max_industry_weight, max_turnover}
```

### SignalResult

Output of the signal generation stage.

```python
@dataclass
class SignalResult:
    signals: pd.DataFrame   # minimum columns: ts_code, score; optional: side (1=long, -1=short)
    metadata: dict          # strategy name, parameters, etc.
```

### StrategyResult

Output of the full strategy pipeline. Contains both target positions and trade instructions.

```python
@dataclass
class StrategyResult:
    target_positions: pd.DataFrame  # ts_code, target_weight, target_shares, target_amount
    orders: pd.DataFrame            # ts_code, action(buy/sell), delta_shares, delta_amount
    metadata: dict
```

## Interfaces

### ISignalGenerator (`signal.py`)

Single responsibility: produce trading signals from feature data.

```python
class ISignalGenerator(ABC):
    @abstractmethod
    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        ...
```

- Input `features`: wide-format DataFrame with at least `ts_code`, `trade_date`, and factor columns
- Output `SignalResult.signals`: at minimum `ts_code` and `score` columns
- `score`: continuous value; positive = long bias, negative = short bias
- Must not access filesystem, databases, or network — purely data-in, data-out

### IPositionManager (`position.py`)

Three-step pipeline to convert signals into constrained positions and trade orders.

```python
class IPositionManager(ABC):
    @abstractmethod
    def allocate(self, signals: SignalResult, ctx: StrategyContext) -> pd.DataFrame:
        """Weight allocation + position sizing.
        Input:  SignalResult
        Output: DataFrame[ts_code, target_weight]  -- target_weight in [0, 1], sum <= 1.0
        Note:   target_shares is NOT computed here -- price data is needed for share calculation.
        """
        ...

    @abstractmethod
    def constrain(self, positions: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """Risk constraints: single-stock caps, industry caps, drawdown controls.
        Input/Output: same schema [ts_code, target_weight], weights are clamped/adjusted.
        """
        ...

    @abstractmethod
    def execute(
        self, target: pd.DataFrame, prev: pd.DataFrame | None, ctx: StrategyContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Turnover execution: compute target_shares, diff vs previous, generate orders.
        Input:  target (ts_code, target_weight), prev (ts_code, shares, cost) or None
        Output: (positions, orders)
                - positions: ts_code, target_weight, target_shares, target_amount
                - orders:    ts_code, action(buy/sell), delta_shares, delta_amount
        Note:   prev=None on first trading day or when no prior positions exist.
                target_shares computed via: target_weight * capital / price.
        """
        ...
```

| Step | Responsibility | Input | Output |
|------|---------------|-------|--------|
| `allocate` | Position sizing + weight allocation | SignalResult | `[ts_code, target_weight]` |
| `constrain` | Risk constraint clamping | Same schema as alloc output | Same schema (clamped) |
| `execute` | Compute target_shares, diff vs prev, generate orders | target_weight + prev + ctx | `(positions [ts_code, weight, shares, amount], orders [ts_code, action, delta_shares, delta_amount])` |

The three methods are called in fixed order by `Strategy.run()`. Subclasses must implement all three — there is no way to bypass a step.

### Strategy (`strategy.py`)

Concrete pipeline orchestrator. Composes `ISignalGenerator` + `IPositionManager`.

```python
class Strategy:
    def __init__(self, signal: ISignalGenerator, position: IPositionManager):
        self.signal = signal
        self.position = position

    def run(self, features: pd.DataFrame, ctx: StrategyContext) -> StrategyResult:
        sig = self.signal.generate(features, ctx)
        pos = self.position.allocate(sig, ctx)
        pos = self.position.constrain(pos, ctx)
        pos, orders = self.position.execute(pos, ctx.prev_holdings, ctx)
        return StrategyResult(
            target_positions=pos,
            orders=orders,
            metadata={"signal": sig.metadata},
        )
```

Key design decisions:

- `Strategy` is NOT abstract — it's a concrete orchestrator. It can be instantiated directly: `Strategy(MySignal(), MyPosition())`. Subclassing is optional (e.g., for strategies with custom lifecycle hooks).
- `signal` and `position` are injected via constructor — any combination of `ISignalGenerator` and `IPositionManager` works.
- `run()` is the single entry point. Pipeline order is immutable.

## Conventions

- **No I/O in strategy**: Strategy code never reads files, writes to disk, or calls APIs. All data flows in through method parameters.
- **DataFrame schema as contract**: Each stage defines its expected input/output columns. No additional DTO classes between stages — the DataFrame is the transfer object.
- **Single responsibility**: `ISignalGenerator` answers "what to trade". `IPositionManager` answers "how much and how".
- **Dependency injection**: All dependencies are constructor-injected. No global state, no singletons, no hidden side effects.

## Example Usage

```python
from alpha_quat.strategy.types import StrategyContext, StrategyResult
from alpha_quat.strategy.strategy import Strategy

# Concrete implementations would be in alpha_quat/strategy/signals/ and alpha_quat/strategy/positions/
strategy = Strategy(
    signal=FactorWeightedSignal(weights={"f_001": 0.5, "f_002": 0.3, "f_003": 0.2}),
    position=EqualWeightPosition(max_stocks=30, max_single_weight=0.05),
)

ctx = StrategyContext(
    trade_date="20240115",
    capital=1_000_000,
    prev_holdings=prev,
    constraints={"max_single_weight": 0.05},
)

result: StrategyResult = strategy.run(features_df, ctx)
# result.target_positions  -> target portfolio
# result.orders             -> trade instructions
```

## Future Extensions

- Concrete signal strategies in `strategy/signals/` (factor-weighted, IC-weighted, ML-prediction)
- Concrete position managers in `strategy/positions/` (equal-weight, risk-parity, inverse-volatility)
- `Strategy` subclass with `pre_run` / `post_run` hooks for lifecycle events
- CLI integration: `uv run alpha-quat strategy run --signal factor_weighted --position equal_weight`
- Backtest integration: iterating over dates, accumulating `StrategyResult` per day

# Strategy Base Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the strategy framework base classes — data types, ISignalGenerator ABC, IPositionManager ABC, and Strategy pipeline orchestrator.

**Architecture:** Five source files in `src/alpha_quat/strategy/` and five test files in `tests/test_strategy/`. All external data flows in via method parameters (DataFrame, StrategyContext). Pipeline order is immutable: generate → allocate → constrain → execute.

**Tech Stack:** Python 3.11+, dataclasses, ABC, pandas, pytest

---

## File Structure

```
src/alpha_quat/strategy/
├── __init__.py          # Re-exports: StrategyContext, SignalResult, StrategyResult,
│                        #   ISignalGenerator, IPositionManager, Strategy
├── types.py             # StrategyContext, SignalResult, StrategyResult (dataclasses)
├── signal.py            # ISignalGenerator (ABC with generate())
├── position.py          # IPositionManager (ABC with allocate(), constrain(), execute())
└── strategy.py          # Strategy (concrete pipeline orchestrator, holds signal + position)

tests/test_strategy/
├── __init__.py          # (empty)
├── test_types.py        # Test dataclass creation and defaults
├── test_signal.py       # Test ABC enforcement, concrete subclass
├── test_position.py     # Test ABC enforcement, concrete subclass, prev=None
└── test_strategy.py     # Integration: mock signal + position, pipeline order
```

---

### Task 1: Data Types (`types.py`)

**Files:**
- Create: `src/alpha_quat/strategy/__init__.py` (empty)
- Create: `tests/test_strategy/__init__.py` (empty)
- Create: `src/alpha_quat/strategy/types.py`
- Create: `tests/test_strategy/test_types.py`

- [ ] **Step 1: Write failing tests for all three dataclasses**

Write `tests/test_strategy/test_types.py`:

```python
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult, StrategyResult


class TestStrategyContext:
    def test_create_minimal(self):
        ctx = StrategyContext(trade_date="20240115", capital=1000000.0)
        assert ctx.trade_date == "20240115"
        assert ctx.capital == 1000000.0
        assert ctx.universe is None
        assert ctx.prices is None
        assert ctx.prev_holdings is None
        assert ctx.constraints is None

    def test_create_full(self):
        prev = pd.DataFrame({"ts_code": ["000001.SZ"], "shares": [1000], "cost": [10.5]})
        prices = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]})
        ctx = StrategyContext(
            trade_date="20240115",
            capital=1000000.0,
            universe=["000001.SZ", "000002.SZ"],
            prices=prices,
            prev_holdings=prev,
            constraints={"max_single_weight": 0.05},
        )
        assert ctx.universe == ["000001.SZ", "000002.SZ"]
        assert len(ctx.prices) == 1
        assert len(ctx.prev_holdings) == 1
        assert ctx.constraints["max_single_weight"] == 0.05


class TestSignalResult:
    def test_create(self):
        signals = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "score": [0.8, 0.3]})
        result = SignalResult(signals=signals, metadata={"model": "factor_weighted"})
        assert len(result.signals) == 2
        assert list(result.signals.columns) == ["ts_code", "score"]
        assert result.metadata["model"] == "factor_weighted"


class TestStrategyResult:
    def test_create(self):
        positions = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "target_weight": [0.05],
            "target_shares": [500],
            "target_amount": [50000.0],
        })
        orders = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "action": ["buy"],
            "delta_shares": [500],
            "delta_amount": [50000.0],
        })
        result = StrategyResult(
            target_positions=positions,
            orders=orders,
            metadata={"signal": {"model": "test"}},
        )
        assert result.target_positions is positions
        assert result.orders is orders
        assert result.metadata["signal"]["model"] == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategy/test_types.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha_quat.strategy.types'`

- [ ] **Step 3: Implement `src/alpha_quat/strategy/types.py`**

```python
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StrategyContext:
    trade_date: str
    capital: float
    universe: list[str] | None = None
    prices: pd.DataFrame | None = None
    prev_holdings: pd.DataFrame | None = None
    constraints: dict | None = None


@dataclass
class SignalResult:
    signals: pd.DataFrame
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyResult:
    target_positions: pd.DataFrame
    orders: pd.DataFrame
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategy/test_types.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/strategy/__init__.py tests/test_strategy/__init__.py src/alpha_quat/strategy/types.py tests/test_strategy/test_types.py
git commit -m "feat: add strategy data types (StrategyContext, SignalResult, StrategyResult)"
```

---

### Task 2: ISignalGenerator ABC (`signal.py`)

**Files:**
- Create: `src/alpha_quat/strategy/signal.py`
- Create: `tests/test_strategy/test_signal.py`

- [ ] **Step 1: Write failing tests for ISignalGenerator**

Write `tests/test_strategy/test_signal.py`:

```python
import pytest
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.signal import ISignalGenerator


class TestISignalGenerator:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ISignalGenerator()

    def test_concrete_subclass_works(self):
        class MySignal(ISignalGenerator):
            def generate(self, features, ctx):
                return SignalResult(
                    signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]}),
                    metadata={"name": "my_signal"},
                )

        signal = MySignal()
        features = pd.DataFrame({
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20240115", "20240115"],
            "factor_001": [0.1, 0.2],
        })
        ctx = StrategyContext(trade_date="20240115", capital=1000000.0)
        result = signal.generate(features, ctx)
        assert isinstance(result, SignalResult)
        assert len(result.signals) == 1
        assert result.metadata["name"] == "my_signal"

    def test_missing_generate_raises(self):
        with pytest.raises(TypeError):
            class BadSignal(ISignalGenerator):
                pass
            BadSignal()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategy/test_signal.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha_quat.strategy.signal'`

- [ ] **Step 3: Implement `src/alpha_quat/strategy/signal.py`**

```python
from abc import ABC, abstractmethod

import pandas as pd

from alpha_quat.strategy.types import StrategyContext, SignalResult


class ISignalGenerator(ABC):
    @abstractmethod
    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategy/test_signal.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/strategy/signal.py tests/test_strategy/test_signal.py
git commit -m "feat: add ISignalGenerator ABC"
```

---

### Task 3: IPositionManager ABC (`position.py`)

**Files:**
- Create: `src/alpha_quat/strategy/position.py`
- Create: `tests/test_strategy/test_position.py`

- [ ] **Step 1: Write failing tests for IPositionManager**

Write `tests/test_strategy/test_position.py`:

```python
import pytest
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.position import IPositionManager


class TestIPositionManager:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            IPositionManager()

    def test_concrete_subclass_works(self):
        class MyPosition(IPositionManager):
            def allocate(self, signals, ctx):
                return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

            def constrain(self, positions, ctx):
                return positions

            def execute(self, target, prev, ctx):
                positions = pd.DataFrame({
                    "ts_code": ["000001.SZ"],
                    "target_weight": [0.1],
                    "target_shares": [100],
                    "target_amount": [10000.0],
                })
                orders = pd.DataFrame({
                    "ts_code": ["000001.SZ"],
                    "action": ["buy"],
                    "delta_shares": [100],
                    "delta_amount": [10000.0],
                })
                return positions, orders

        pm = MyPosition()
        signals = SignalResult(
            signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]})
        )
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        pos = pm.allocate(signals, ctx)
        assert list(pos.columns) == ["ts_code", "target_weight"]

        constrained = pm.constrain(pos, ctx)
        assert list(constrained.columns) == ["ts_code", "target_weight"]

        pos, orders = pm.execute(constrained, None, ctx)
        assert isinstance(pos, pd.DataFrame)
        assert isinstance(orders, pd.DataFrame)
        assert "target_shares" in pos.columns
        assert "action" in orders.columns

    def test_execute_with_prev_none(self):
        class MyPosition(IPositionManager):
            def allocate(self, signals, ctx):
                return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

            def constrain(self, positions, ctx):
                return positions

            def execute(self, target, prev, ctx):
                assert prev is None
                positions = target.assign(target_shares=100, target_amount=10000.0)
                orders = pd.DataFrame({
                    "ts_code": ["000001.SZ"],
                    "action": ["buy"],
                    "delta_shares": [100],
                    "delta_amount": [10000.0],
                })
                return positions, orders

        pm = MyPosition()
        signals = SignalResult(
            signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]})
        )
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        pos = pm.allocate(signals, ctx)
        pos = pm.constrain(pos, ctx)
        pos, orders = pm.execute(pos, None, ctx)
        assert orders["action"].iloc[0] == "buy"

    def test_missing_method_raises(self):
        with pytest.raises(TypeError):
            class BadPosition1(IPositionManager):
                def allocate(self, signals, ctx):
                    pass

                def constrain(self, positions, ctx):
                    pass
            BadPosition1()

        with pytest.raises(TypeError):
            class BadPosition2(IPositionManager):
                def allocate(self, signals, ctx):
                    pass

                def execute(self, target, prev, ctx):
                    pass
            BadPosition2()

        with pytest.raises(TypeError):
            class BadPosition3(IPositionManager):
                def constrain(self, positions, ctx):
                    pass

                def execute(self, target, prev, ctx):
                    pass
            BadPosition3()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategy/test_position.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha_quat.strategy.position'`

- [ ] **Step 3: Implement `src/alpha_quat/strategy/position.py`**

```python
from abc import ABC, abstractmethod

import pandas as pd

from alpha_quat.strategy.types import StrategyContext, SignalResult


class IPositionManager(ABC):
    @abstractmethod
    def allocate(self, signals: SignalResult, ctx: StrategyContext) -> pd.DataFrame:
        ...

    @abstractmethod
    def constrain(self, positions: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        ...

    @abstractmethod
    def execute(
        self, target: pd.DataFrame, prev: pd.DataFrame | None, ctx: StrategyContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategy/test_position.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/strategy/position.py tests/test_strategy/test_position.py
git commit -m "feat: add IPositionManager ABC with allocate/constrain/execute"
```

---

### Task 4: Strategy Orchestrator (`strategy.py`)

**Files:**
- Create: `src/alpha_quat/strategy/strategy.py`
- Create: `tests/test_strategy/test_strategy.py`

- [ ] **Step 1: Write failing tests for Strategy**

Write `tests/test_strategy/test_strategy.py`:

```python
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult, StrategyResult
from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.position import IPositionManager
from alpha_quat.strategy.strategy import Strategy


class MockSignal(ISignalGenerator):
    def generate(self, features, ctx):
        return SignalResult(
            signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]}),
            metadata={"signal_name": "mock"},
        )


class MockPosition(IPositionManager):
    def allocate(self, signals, ctx):
        return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

    def constrain(self, positions, ctx):
        return positions

    def execute(self, target, prev, ctx):
        positions = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "target_weight": [0.1],
            "target_shares": [100],
            "target_amount": [10000.0],
        })
        orders = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "action": ["buy"],
            "delta_shares": [100],
            "delta_amount": [10000.0],
        })
        return positions, orders


class TestStrategy:
    def test_run_returns_strategy_result(self):
        strategy = Strategy(signal=MockSignal(), position=MockPosition())
        features = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "trade_date": ["20240115"],
            "factor_001": [0.5],
        })
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        result = strategy.run(features, ctx)
        assert isinstance(result, StrategyResult)
        assert len(result.target_positions) == 1
        assert len(result.orders) == 1
        assert result.metadata["signal"] == {"signal_name": "mock"}

    def test_pipeline_order_is_correct(self):
        call_order = []

        class OrderedSignal(ISignalGenerator):
            def generate(self, features, ctx):
                call_order.append("signal")
                return SignalResult(
                    signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]})
                )

        class OrderedPosition(IPositionManager):
            def allocate(self, signals, ctx):
                call_order.append("allocate")
                return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

            def constrain(self, positions, ctx):
                call_order.append("constrain")
                return positions

            def execute(self, target, prev, ctx):
                call_order.append("execute")
                positions = target.assign(target_shares=100, target_amount=10000.0)
                orders = pd.DataFrame({
                    "ts_code": ["000001.SZ"],
                    "action": ["buy"],
                    "delta_shares": [100],
                    "delta_amount": [10000.0],
                })
                return positions, orders

        strategy = Strategy(signal=OrderedSignal(), position=OrderedPosition())
        features = pd.DataFrame({
            "ts_code": ["000001.SZ"], "trade_date": ["20240115"], "factor_001": [0.5]
        })
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        strategy.run(features, ctx)
        assert call_order == ["signal", "allocate", "constrain", "execute"]

    def test_dependency_injection(self):
        sig_a = MockSignal()
        pos_a = MockPosition()
        sig_b = MockSignal()
        strategy = Strategy(signal=sig_a, position=pos_a)
        assert strategy.signal is sig_a
        assert strategy.position is pos_a
        strategy.signal = sig_b
        assert strategy.signal is sig_b
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategy/test_strategy.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'alpha_quat.strategy.strategy'`

- [ ] **Step 3: Implement `src/alpha_quat/strategy/strategy.py`**

```python
import pandas as pd

from alpha_quat.strategy.types import StrategyContext, StrategyResult
from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.position import IPositionManager


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

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategy/test_strategy.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/strategy/strategy.py tests/test_strategy/test_strategy.py
git commit -m "feat: add Strategy pipeline orchestrator"
```

---

### Task 5: Package Exports (`__init__.py`)

**Files:**
- Modify: `src/alpha_quat/strategy/__init__.py`

- [ ] **Step 1: Write a failing test for imports**

Write `tests/test_strategy/test_imports.py`:

```python
def test_all_exports_available():
    from alpha_quat.strategy import (
        StrategyContext,
        SignalResult,
        StrategyResult,
        ISignalGenerator,
        IPositionManager,
        Strategy,
    )
    assert StrategyContext is not None
    assert SignalResult is not None
    assert StrategyResult is not None
    assert ISignalGenerator is not None
    assert IPositionManager is not None
    assert Strategy is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_strategy/test_imports.py -v
```
Expected: FAIL — `ImportError: cannot import name 'StrategyContext' from 'alpha_quat.strategy'`

- [ ] **Step 3: Populate `src/alpha_quat/strategy/__init__.py`**

```python
from alpha_quat.strategy.types import StrategyContext, SignalResult, StrategyResult
from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.position import IPositionManager
from alpha_quat.strategy.strategy import Strategy
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_strategy/test_imports.py -v
```
Expected: 1 PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/test_strategy/ -v
```
Expected: 15 PASS (4 types + 3 signal + 4 position + 3 strategy + 1 imports)

- [ ] **Step 6: Commit**

```bash
git add src/alpha_quat/strategy/__init__.py tests/test_strategy/test_imports.py
git commit -m "feat: add strategy package exports"
```

---

### Task 6: Final Verification

- [ ] **Step 1: Run typecheck**

```bash
uv run pyright
```
Expected: PASS (no new errors)

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest --cov=src -v
```
Expected: all tests pass, coverage includes `src/alpha_quat/strategy/`

- [ ] **Step 3: Run lint + format**

```bash
uv run ruff format src/alpha_quat/strategy/ tests/test_strategy/ && uv run ruff check --fix src/alpha_quat/strategy/ tests/test_strategy/
```
Expected: no errors

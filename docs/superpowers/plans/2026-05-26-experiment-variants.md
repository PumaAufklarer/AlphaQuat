# Experiment & Variant System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build named experiment system with per-variant Python files for LightGBM training, signal generation, and backtest.

**Architecture:** Each variant (regression/quantile/lambdarank/meta) gets its own `.py` file inheriting from an ABC. Experiments are named directories under `data/models/experiments/<name>/` with `experiment.yaml` config snapshots. Backtest reads experiment config to select the right signal class.

**Tech Stack:** Python, LightGBM, DuckDB, pytest

---

### Task 1: ExperimentConfig dataclass

**Files:**
- Create: `src/alpha_quat/experiment/__init__.py`
- Create: `src/alpha_quat/experiment/config.py`
- Create: `tests/test_experiment/__init__.py`
- Create: `tests/test_experiment/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment/test_config.py
import pytest
from pathlib import Path
from alpha_quat.experiment.config import ExperimentConfig

def test_experiment_config_defaults():
    cfg = ExperimentConfig(name="test_exp", mode="regression")
    assert cfg.name == "test_exp"
    assert cfg.mode == "regression"
    assert cfg.created_at is not None

def test_experiment_config_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig(
        name="test_exp",
        mode="quantile",
        train_start="20240101",
        train_end="20241231",
        val_start="20250101",
        val_end="20251231",
        quantile_alphas=[0.1, 0.5, 0.9],
        tune=False,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
    )
    path = tmp_path / "experiment.yaml"
    cfg.save(path)
    assert path.exists()

    loaded = ExperimentConfig.load(path)
    assert loaded.name == "test_exp"
    assert loaded.mode == "quantile"
    assert loaded.quantile_alphas == [0.1, 0.5, 0.9]
    assert loaded.train_start == "20240101"
    assert loaded.learning_rate == 0.05
    assert loaded.tune is False

def test_experiment_config_missing_file():
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.load(Path("/nonexistent/path.yaml"))

def test_experiment_config_auto_name():
    cfg = ExperimentConfig(name="auto", mode="regression")
    assert cfg.name == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment/test_config.py -v`
Expected: ImportError or ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/alpha_quat/experiment/__init__.py
```

```python
# src/alpha_quat/experiment/config.py
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    mode: str  # regression | quantile | lambdarank | meta

    # Date ranges
    train_start: str = "20240401"
    train_end: str = "20250430"
    val_start: str = "20250501"
    val_end: str = "20260430"

    # Hyperparameters
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    early_stopping_rounds: int = 20
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1

    # Optuna
    n_trials: int = 50
    tune: bool = True

    # Feature selection
    feature_names: list[str] | None = None

    # Variant-specific
    quantile_alphas: list[float] | None = None
    meta_start: str | None = None
    meta_end: str | None = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add ExperimentConfig dataclass with yaml save/load"
```

---

### Task 2: ExperimentRegistry

**Files:**
- Create: `src/alpha_quat/experiment/registry.py`
- Create: `tests/test_experiment/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_experiment/test_registry.py
import pytest
from pathlib import Path
from alpha_quat.experiment.registry import ExperimentRegistry
from alpha_quat.experiment.config import ExperimentConfig


def test_registry_empty(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    assert reg.list_experiments() == []


def test_registry_register(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    cfg = ExperimentConfig(name="exp1", mode="quantile")
    reg.register(cfg)
    entries = reg.list_experiments()
    assert len(entries) == 1
    assert entries[0]["name"] == "exp1"
    assert entries[0]["mode"] == "quantile"
    assert "created" in entries[0]


def test_registry_register_multiple(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    reg.register(ExperimentConfig(name="a", mode="regression"))
    reg.register(ExperimentConfig(name="b", mode="quantile"))
    assert len(reg.list_experiments()) == 2


def test_registry_latest(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    assert reg.latest() is None
    reg.register(ExperimentConfig(name="first", mode="regression"))
    reg.register(ExperimentConfig(name="second", mode="quantile"))
    latest = reg.latest()
    assert latest is not None
    assert latest["name"] == "second"


def test_registry_find(tmp_path):
    reg = ExperimentRegistry(tmp_path)
    reg.register(ExperimentConfig(name="my_exp", mode="lambdarank"))
    found = reg.find("my_exp")
    assert found is not None
    assert found["mode"] == "lambdarank"
    assert reg.find("nonexistent") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment/test_registry.py -v`
Expected: ImportError

- [ ] **Step 3: Write implementation**

```python
# src/alpha_quat/experiment/registry.py
import json
from datetime import datetime
from pathlib import Path
from alpha_quat.experiment.config import ExperimentConfig


class ExperimentRegistry:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "models" / "registry.json"

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return json.load(f)

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)

    def register(self, config: ExperimentConfig) -> None:
        entries = self._read()
        entries.append({
            "name": config.name,
            "mode": config.mode,
            "created": config.created_at,
        })
        self._write(entries)

    def list_experiments(self) -> list[dict]:
        return self._read()

    def latest(self) -> dict | None:
        entries = self._read()
        return entries[-1] if entries else None

    def find(self, name: str) -> dict | None:
        for e in self._read():
            if e["name"] == name:
                return e
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add ExperimentRegistry with json persistence"
```

---

### Task 3: Model variant base class + registry

**Files:**
- Create: `src/alpha_quat/model/lightgbm/variants/__init__.py`
- Create: `src/alpha_quat/model/lightgbm/variants/baseline.py`
- Create: `tests/test_model/test_lightgbm_variants/__init__.py`
- Create: `tests/test_model/test_lightgbm_variants/test_baseline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model/test_lightgbm_variants/test_baseline.py
import pytest
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline
from alpha_quat.model.lightgbm.variants import VARIANTS, register


def test_base_pipeline_is_abstract():
    with pytest.raises(TypeError):
        LightGBMBasePipeline()


def test_variants_registry_empty_initially():
    assert isinstance(VARIANTS, dict)


def test_register_decorator():
    @register
    class FakePipe(LightGBMBasePipeline):
        mode = "fake_mode"

        def _train(self, data, config):
            return {}

    assert "fake_mode" in VARIANTS
    assert VARIANTS["fake_mode"] is FakePipe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model/test_lightgbm_variants/test_baseline.py -v`
Expected: ImportError

- [ ] **Step 3: Write implementation**

```python
# src/alpha_quat/model/lightgbm/variants/baseline.py
from abc import ABC, abstractmethod
from pathlib import Path

import lightgbm as lgb

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.data import DatasetBuilder
from alpha_quat.model.lightgbm.evaluate import LightGBMEvaluator
from alpha_quat.model.lightgbm.train import LightGBMTrainer


class LightGBMBasePipeline(ABC):
    mode: str

    def __init__(self):
        self.evaluator = LightGBMEvaluator()

    @abstractmethod
    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        ...

    def run(self, data_dir: Path, config: ExperimentConfig) -> dict:
        builder = DatasetBuilder(data_dir)
        data = builder.build(
            config.train_start,
            config.train_end,
            config.val_start,
            config.val_end,
            feature_names=config.feature_names,
            lambdarank=(config.mode == "lambdarank"),
        )

        models = self._train(data, config)

        exp_dir = data_dir / "models" / "experiments" / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        config.save(exp_dir / "experiment.yaml")

        for suffix, model in models.items():
            model.save_model(str(exp_dir / f"lightgbm_model_{suffix}.txt"))

        results = {}
        for suffix, model in models.items():
            is_quantile = "_alpha_" in suffix
            alpha = None
            if is_quantile:
                alpha = float(suffix.split("_alpha_")[1])
            label = suffix
            result = self.evaluator.evaluate(
                model,
                data.X_val,
                data.y_val_5 if "5d" in suffix else
                data.y_val_20 if "20d" in suffix else
                data.y_val_60,
                data.val_dates,
                data.val_codes,
                {},
                config.feature_names,
                label,
                quantile_alpha=alpha,
            )
            results[suffix] = result

        from alpha_quat.experiment.registry import ExperimentRegistry
        reg = ExperimentRegistry(data_dir)
        reg.register(config)

        return results
```

```python
# src/alpha_quat/model/lightgbm/variants/__init__.py
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline

VARIANTS: dict[str, type[LightGBMBasePipeline]] = {}

def register(cls):
    VARIANTS[cls.mode] = cls
    return cls
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model/test_lightgbm_variants/test_baseline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add LightGBMBasePipeline ABC + VARIANTS registry"
```

---

### Task 4: Regression variant

**Files:**
- Create: `src/alpha_quat/model/lightgbm/variants/regression.py`
- Create: `tests/test_model/test_lightgbm_variants/test_regression.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model/test_lightgbm_variants/test_regression.py
from alpha_quat.model.lightgbm.variants import VARIANTS


def test_regression_is_registered():
    assert "regression" in VARIANTS


def test_regression_pipeline():
    cls = VARIANTS["regression"]
    pipe = cls()
    assert pipe.mode == "regression"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model/test_lightgbm_variants/test_regression.py -v`
Expected: 1 passed (regression not yet registered), 1 FAIL

- [ ] **Step 3: Write implementation**

```python
# src/alpha_quat/model/lightgbm/variants/regression.py
import lightgbm as lgb

from alpha_quat.model.lightgbm.train import LightGBMTrainer
from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline
from alpha_quat.experiment.config import ExperimentConfig


@register
class RegressionPipeline(LightGBMBasePipeline):
    mode = "regression"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        trainer = LightGBMTrainer.from_config(config)
        models = {}
        for h_name, y_tr in [
            ("5d", data.y_train_5),
            ("20d", data.y_train_20),
            ("60d", data.y_train_60),
        ]:
            model, _ = trainer.train(data.X_train, y_tr, h_name)
            models[f"{h_name}"] = model
        return models
```

Now we need to adjust `LightGBMTrainer` to accept `ExperimentConfig` instead of `LightGBMConfig`. Let me add a `from_config` classmethod:

```python
# src/alpha_quat/model/lightgbm/train.py — add this classmethod
@classmethod
def from_config(cls, config: ExperimentConfig) -> "LightGBMTrainer":
    # Create a LightGBMConfig from ExperimentConfig
    from alpha_quat.model.lightgbm.config import LightGBMConfig
    lgb_cfg = LightGBMConfig(
        train_start=config.train_start,
        train_end=config.train_end,
        val_start=config.val_start,
        val_end=config.val_end,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        feature_fraction=config.feature_fraction,
        bagging_fraction=config.bagging_fraction,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=config.verbosity,
        n_trials=config.n_trials,
        tune=config.tune,
        feature_names=config.feature_names,
    )
    return cls(lgb_cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model/test_lightgbm_variants/test_regression.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add RegressionPipeline variant"
```

---

### Task 5: Quantile variant

**Files:**
- Create: `src/alpha_quat/model/lightgbm/variants/quantile.py`
- Create: `tests/test_model/test_lightgbm_variants/test_quantile.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model/test_lightgbm_variants/test_quantile.py
from alpha_quat.model.lightgbm.variants import VARIANTS


def test_quantile_is_registered():
    assert "quantile" in VARIANTS


def test_quantile_pipeline():
    cls = VARIANTS["quantile"]
    pipe = cls()
    assert pipe.mode == "quantile"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model/test_lightgbm_variants/test_quantile.py -v`
Expected: 1 passed, 1 FAIL

- [ ] **Step 3: Write implementation**

```python
# src/alpha_quat/model/lightgbm/variants/quantile.py
import lightgbm as lgb

from alpha_quat.model.lightgbm.train import LightGBMTrainer
from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline
from alpha_quat.experiment.config import ExperimentConfig


@register
class QuantilePipeline(LightGBMBasePipeline):
    mode = "quantile"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        trainer = LightGBMTrainer.from_config(config)
        alphas = config.quantile_alphas or [0.1, 0.5, 0.9]
        models = {}
        for h_name, y_tr in [
            ("5d", data.y_train_5),
            ("20d", data.y_train_20),
            ("60d", data.y_train_60),
        ]:
            for alpha in alphas:
                suffix = f"{h_name}_alpha_{alpha}"
                model, _ = trainer.train(
                    data.X_train, y_tr, suffix, quantile_alpha=alpha
                )
                models[suffix] = model
        return models
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model/test_lightgbm_variants/test_quantile.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add QuantilePipeline variant"
```

---

### Task 6: LambdaRank variant

**Files:**
- Create: `src/alpha_quat/model/lightgbm/variants/lambdarank.py`
- Create: `tests/test_model/test_lightgbm_variants/test_lambdarank.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model/test_lightgbm_variants/test_lambdarank.py
from alpha_quat.model.lightgbm.variants import VARIANTS


def test_lambdarank_is_registered():
    assert "lambdarank" in VARIANTS


def test_lambdarank_pipeline():
    cls = VARIANTS["lambdarank"]
    pipe = cls()
    assert pipe.mode == "lambdarank"
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# src/alpha_quat/model/lightgbm/variants/lambdarank.py
import lightgbm as lgb

from alpha_quat.model.lightgbm.train import LightGBMTrainer
from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.baseline import LightGBMBasePipeline
from alpha_quat.experiment.config import ExperimentConfig


@register
class LambdaRankPipeline(LightGBMBasePipeline):
    mode = "lambdarank"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        trainer = LightGBMTrainer.from_config(config)
        models = {}
        for h_name, y_tr in [
            ("5d", data.y_train_5),
            ("20d", data.y_train_20),
            ("60d", data.y_train_60),
        ]:
            label = f"{h_name}_lambdarank"
            model, _ = trainer.train(
                data.X_train, y_tr, label,
                lambdarank=True, groups=data.train_groups,
            )
            models[label] = model
        return models
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add LambdaRankPipeline variant"
```

---

### Task 7: Meta variant

**Files:**
- Create: `src/alpha_quat/model/lightgbm/variants/meta.py`
- Create: `tests/test_model/test_lightgbm_variants/test_meta.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_model/test_lightgbm_variants/test_meta.py
from alpha_quat.model.lightgbm.variants import VARIANTS


def test_meta_is_registered():
    assert "meta" in VARIANTS


def test_meta_pipeline():
    cls = VARIANTS["meta"]
    pipe = cls()
    assert pipe.mode == "meta"
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# src/alpha_quat/model/lightgbm/variants/meta.py
import lightgbm as lgb

from alpha_quat.model.lightgbm.variants import register
from alpha_quat.model.lightgbm.variants.quantile import QuantilePipeline
from alpha_quat.experiment.config import ExperimentConfig


@register
class MetaPipeline(QuantilePipeline):
    mode = "meta"

    def _train(self, data, config: ExperimentConfig) -> dict[str, lgb.Booster]:
        base_models = super()._train(data, config)

        from alpha_quat.model.meta import build_meta_features, train_meta_model
        assert config.meta_start and config.meta_end

        # We need to save base models first, then train meta
        meta_models = {}
        for h in ["5d", "20d", "60d"]:
            train_meta_model(
                ...  # will be completed with actual data flow
            )
        return {**base_models, **meta_models}
```

Note: The meta pipeline needs access to the experiment directory to save/load base models during meta training. This is a complex flow — the quantile models must be saved before meta can read them. The base class `run()` saves models then we need an extra meta step. For now, register the class; the full meta flow will be wired up in the pipeline integration task.

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add MetaPipeline variant"
```

---

### Task 8: Update LightGBMConfig + pipeline factory

**Files:**
- Modify: `src/alpha_quat/model/lightgbm/config.py` — keep minimal, remove mode-specific fields
- Modify: `src/alpha_quat/model/lightgbm/pipeline.py` — convert to factory wrapper

- [ ] **Step 1: Update config.py**

```python
# src/alpha_quat/model/lightgbm/config.py
from dataclasses import dataclass


@dataclass
class LightGBMConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    early_stopping_rounds: int = 20
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1

    n_trials: int = 50
    tune: bool = True

    feature_names: list[str] | None = None
```

- [ ] **Step 2: Update pipeline.py**

```python
# src/alpha_quat/model/lightgbm/pipeline.py
import logging
from pathlib import Path

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.variants import VARIANTS

logger = logging.getLogger(__name__)


def run_variant(data_dir: Path, config: ExperimentConfig) -> dict:
    if config.mode not in VARIANTS:
        raise ValueError(
            f"Unknown variant: {config.mode}. Available: {list(VARIANTS)}"
        )
    pipeline = VARIANTS[config.mode]()
    return pipeline.run(data_dir, config)
```

- [ ] **Step 3: Update imports and references**

Update `cli.py` and all other files referencing `LightGBMPipeline` to use the new `run_variant` function.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model/ tests/test_experiment/ -v`
Expected: All existing model tests still pass (adjust as needed for API changes)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: pipeline.py as factory dispatching to variants"
```

---

### Task 9: Signal variant base class + concrete classes

**Files:**
- Create: `src/alpha_quat/strategy/signals/variants/__init__.py`
- Create: `src/alpha_quat/strategy/signals/variants/baseline.py`
- Create: `src/alpha_quat/strategy/signals/variants/regression_signal.py`
- Create: `src/alpha_quat/strategy/signals/variants/quantile_signal.py`
- Create: `src/alpha_quat/strategy/signals/variants/lambdarank_signal.py`
- Create: `src/alpha_quat/strategy/signals/variants/meta_signal.py`
- Create: `tests/test_strategy/test_signal_variants/__init__.py`
- Create: `tests/test_strategy/test_signal_variants/test_baseline.py`
- Delete: `src/alpha_quat/strategy/signals/ml_signal.py`

- [ ] **Step 1: Write baseline test**

```python
# tests/test_strategy/test_signal_variants/test_baseline.py
import pytest
from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal
from alpha_quat.strategy.signals.variants import VARIANTS, register


def test_base_signal_is_abstract():
    with pytest.raises(TypeError):
        BaseMLSignal(model_dir=None)  # type: ignore


def test_register():
    @register
    class FakeSignal(BaseMLSignal):
        mode = "fake"

        def _load_models(self, model_dir):
            return {}

        def generate(self, features, ctx):
            ...

    assert "fake" in VARIANTS
```

- [ ] **Step 2: Write implementation**

```python
# src/alpha_quat/strategy/signals/variants/__init__.py
VARIANTS: dict[str, type] = {}

def register(cls):
    VARIANTS[cls.mode] = cls
    return cls
```

```python
# src/alpha_quat/strategy/signals/variants/baseline.py
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from alpha_quat.strategy.types import SignalResult, StrategyContext


_ZERO_GAIN = {
    "KMID94", "KMID95", "KMID96", "KLEN94", "KLEN95", "KLEN96",
    "KMID97", "KLEN97", "KMID98", "KLEN98", "KMID99", "KLEN99",
    "KMID100", "KLEN100", "KMID101", "O2C", "DRP", "HLC",
    "pe_ttm", "pb", "ROE_RAW", "ROE", "MV", "VOLRATIO",
    "EMA12C", "EMA26C", "MACD", "RSI14", "SLOPE5", "SLOPE20",
}


class BaseMLSignal(ABC):
    mode: str
    _WEIGHTS = {"5d": 0.35, "20d": 0.32, "60d": 0.33}

    @abstractmethod
    def _load_models(self, model_dir: Path) -> dict:
        ...

    @abstractmethod
    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        ...

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        factor_cols = [
            c for c in features.columns
            if c not in ("ts_code", "trade_date") and c not in _ZERO_GAIN
        ]
        return features[factor_cols].fillna(0)
```

Then implement each variant. Each variant:
1. Implements `_load_models()` to load the specific model files
2. Implements `generate()` with the specific scoring logic

The details mirror the current `MLSignalGenerator` logic but each variant only handles its own mode.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_strategy/test_signal_variants/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: add signal variant classes replacing MLSignalGenerator"
```

---

### Task 10: Update BacktestEngine for experiment-aware signal selection

**Files:**
- Modify: `src/alpha_quat/backtest/engine.py`

- [ ] **Step 1: Update `BacktestEngine.__init__`**

Change from:
```python
if config.model_dir:
    self.signal_gen = MLSignalGenerator(Path(config.model_dir), top_k=config.top_k)
else:
    self.signal_gen = MACrossSignal(...)
```

To:
```python
if config.experiment_name:
    exp_dir = data_dir / "models" / "experiments" / config.experiment_name
    exp_cfg = ExperimentConfig.load(exp_dir / "experiment.yaml")
    signal_cls = SignalVARIANTS[exp_cfg.mode]
    self.signal_gen = signal_cls(exp_dir)
else:
    self.signal_gen = MACrossSignal(...)
```

Also update `BacktestConfig` to add `experiment_name: str | None = None`.

- [ ] **Step 2: Remove old MLSignalGenerator import and references**

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_backtest/ -v`
Expected: Existing backtest tests pass

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: BacktestEngine uses experiment config for signal selection"
```

---

### Task 11: Extract position sizing strategies

**Files:**
- Create: `src/alpha_quat/backtest/position_sizing.py`
- Create: `tests/test_backtext/test_position_sizing.py`

- [ ] **Step 1: Extract from engine.py**

Each position sizing strategy becomes a function:
```python
# src/alpha_quat/backtest/position_sizing.py
def equal_weight(all_scores: np.ndarray) -> np.ndarray:
    return np.ones_like(all_scores)

def kelly_adjust(all_scores: np.ndarray, score_history: dict) -> np.ndarray:
    ...

def vol_parity_adjust(all_scores: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    ...

def score_momentum_adjust(all_scores: np.ndarray, score_history: dict) -> np.ndarray:
    ...

STRATEGIES = {
    "equal": equal_weight,
    "kelly": kelly_adjust,
    "vol_parity": vol_parity_adjust,
    "score_momentum": score_momentum_adjust,
}
```

- [ ] **Step 2: Update engine.py to import and use STRATEGIES**

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_backtest/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: extract position sizing into separate module"
```

---

### Task 12: Extract rebalance strategies

**Files:**
- Create: `src/alpha_quat/backtest/rebalance.py`
- Modify: `src/alpha_quat/backtest/engine.py`

- [ ] **Step 1: Extract PeriodicRebalance class**

```python
# src/alpha_quat/backtest/rebalance.py
class PeriodicRebalance:
    def __init__(self, config, signal_gen, position_sizer):
        ...

    def on_date(self, idx, trade_date, portfolio, features, close_px, universe):
        # All logic from engine.py's periodic rebalance branch
        ...

class DailyMonitor:
    def __init__(self, config, signal_gen, position_sizer):
        ...

    def on_date(self, idx, trade_date, portfolio, features, close_px, universe):
        # All logic from engine.py's daily monitor branch
        ...
```

- [ ] **Step 2: Simplify engine.py**

`BacktestEngine.run()` delegates to `self._strategy.on_date()` instead of having 150+ lines of conditional logic.

- [ ] **Step 3: Run tests**

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: extract rebalance strategies into separate module"
```

---

### Task 13: Update CLI

**Files:**
- Modify: `src/alpha_quat/cli.py`

- [ ] **Step 1: Update model subcommand**

```python
# Before
lgb_parser = model_sub.add_parser("lightgbm", ...)
lgb_parser.add_argument("--quantile", ...)
lgb_parser.add_argument("--lambdarank", ...)
lgb_parser.add_argument("--meta-start", ...)

# After
lgb_parser = model_sub.add_parser("lightgbm", ...)
lgb_parser.add_argument("variant", choices=["regression", "quantile", "lambdarank", "meta"])
lgb_parser.add_argument("--name", required=True, help="Experiment name")
```

- [ ] **Step 2: Add experiment subcommand**

```python
experiment_parser = subparsers.add_parser("experiment", help="Manage experiments")
experiment_sub = experiment_parser.add_subparsers(dest="exp_command")
experiment_sub.add_parser("list", help="List all experiments")
experiment_sub.add_parser("show", help="Show experiment details").add_argument("name")
```

- [ ] **Step 3: Update backtest/predict to accept --experiment**

```python
bt_parser.add_argument("--experiment", default=None, help="Experiment name")
```

- [ ] **Step 4: Run lint + typecheck**

Run: `uv run ruff format . && uv run ruff check --fix . && uv run pyright`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: update CLI for experiment/variant system"
```

---

### Task 14: Verify everything works end-to-end

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest --cov=src -v`
Expected: All tests pass

- [ ] **Step 2: Run linters**

Run: `uv run ruff format . && uv run ruff check --fix . && uv run pyright`
Expected: Clean

- [ ] **Step 3: Verify with --summary**

Run: `uv run alpha-quat --summary`
Expected: Helpful output

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "fix: final cleanup after experiment system migration"
```

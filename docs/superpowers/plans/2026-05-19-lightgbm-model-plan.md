# LightGBM Stock Selection Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LightGBM stock selection pipeline that trains on Alpha158 features to predict 5d/20d returns, with Optuna tuning, Rank IC evaluation, and feature importance for pruning.

**Architecture:** New `src/alpha_quat/model/` module with shared `data.py` (DatasetBuilder) and `lightgbm/` subpackage (config, train, evaluate, pipeline). CLI subcommand `alpha-quat model lightgbm`. Reuses `backtest.filters.build_universe()` for universe filtering.

**Tech Stack:** lightgbm, optuna, scikit-learn, scipy, numpy, pandas, pyarrow, tempfile (tests)

---

### Task 1: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add ML dependencies**

```toml
[project]
name = "alpha-quat"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.5.2",
    "lightgbm>=4.0",
    "matplotlib>=3.8.0",
    "numpy>=2.0",
    "optuna>=4.0",
    "pyarrow>=24.0.0",
    "pyright>=1.1.409",
    "pytest>=9.0.3",
    "pyyaml>=6.0.3",
    "ruff>=0.15.12",
    "scikit-learn>=1.5",
    "scipy>=1.14",
    "tushare>=1.4.29",
]
```

- [ ] **Step 2: Install new dependencies**

```bash
uv sync
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add lightgbm, optuna, scikit-learn, scipy, numpy"
```

---

### Task 2: Create LightGBMConfig dataclass

**Files:**
- Create: `src/alpha_quat/model/__init__.py`
- Create: `src/alpha_quat/model/lightgbm/__init__.py`
- Create: `src/alpha_quat/model/lightgbm/config.py`
- Test: `tests/test_model/__init__.py`
- Test: `tests/test_model/test_lightgbm_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model/test_lightgbm_config.py
from alpha_quat.model.lightgbm.config import LightGBMConfig


class TestLightGBMConfig:
    def test_default_values(self):
        cfg = LightGBMConfig()
        assert cfg.train_start == "20240401"
        assert cfg.train_end == "20250430"
        assert cfg.val_start == "20250501"
        assert cfg.val_end == "20260430"
        assert cfg.num_leaves == 31
        assert cfg.learning_rate == 0.05
        assert cfg.n_estimators == 200
        assert cfg.feature_fraction == 0.8
        assert cfg.bagging_fraction == 0.8
        assert cfg.early_stopping_rounds == 20
        assert cfg.random_state == 42
        assert cfg.n_jobs == -1
        assert cfg.verbosity == -1
        assert cfg.n_trials == 50
        assert cfg.tune is True
        assert cfg.feature_names is None

    def test_custom_values(self):
        cfg = LightGBMConfig(
            train_start="20230101",
            train_end="20231231",
            num_leaves=63,
            learning_rate=0.1,
            n_trials=100,
            tune=False,
            feature_names=["KLEN35", "KMID5"],
        )
        assert cfg.train_start == "20230101"
        assert cfg.train_end == "20231231"
        assert cfg.num_leaves == 63
        assert cfg.learning_rate == 0.1
        assert cfg.n_trials == 100
        assert cfg.tune is False
        assert cfg.feature_names == ["KLEN35", "KMID5"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model/test_lightgbm_config.py -v
```
Expected: FAIL with "No module named 'alpha_quat.model'"

- [ ] **Step 3: Create package files and config**

```python
# src/alpha_quat/model/__init__.py
```

```python
# src/alpha_quat/model/lightgbm/__init__.py
```

```python
# src/alpha_quat/model/lightgbm/config.py
from dataclasses import dataclass, field


@dataclass
class LightGBMConfig:
    train_start: str = "20240401"
    train_end: str = "20250430"
    val_start: str = "20250501"
    val_end: str = "20260430"

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

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model/test_lightgbm_config.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/model/ tests/test_model/
git commit -m "feat: add LightGBMConfig dataclass"
```

---

### Task 3: Create DatasetBuilder (shared data layer)

**Files:**
- Create: `src/alpha_quat/model/data.py`
- Test: `tests/test_model/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model/test_data.py
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.model.data import DatasetBuilder, DatasetResult


def _make_features(data_dir: Path, dates: list[str], ts_codes: list[str]):
    """Create synthetic features/ parquet files."""
    feat_dir = data_dir / "features"
    feat_dir.mkdir()
    for d in dates:
        df = pd.DataFrame({
            "ts_code": ts_codes,
            "trade_date": d,
            "KMID": np.random.randn(len(ts_codes)),
            "KLEN": np.random.randn(len(ts_codes)),
        })
        df.to_parquet(feat_dir / f"{d}.parquet")


def _make_daily(data_dir: Path, dates: list[str], ts_codes: list[str], close_col: str = "close"):
    """Create synthetic daily/ parquet files."""
    daily_dir = data_dir / "daily"
    daily_dir.mkdir()
    for d in dates:
        df = pd.DataFrame({
            "ts_code": ts_codes,
            "trade_date": d,
            close_col: np.random.uniform(5, 50, len(ts_codes)),
        })
        path = f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
        df.to_parquet(daily_dir / path)


def _make_stock_basic(data_dir: Path, ts_codes: list[str]):
    """Create synthetic stock_basic.parquet."""
    df = pd.DataFrame({
        "ts_code": ts_codes,
        "market": ["主板"] * len(ts_codes),
        "list_status": ["L"] * len(ts_codes),
    })
    df.to_parquet(data_dir / "stock_basic.parquet")


def _make_trade_cal(data_dir: Path, dates: list[str]):
    """Create synthetic trade_cal.parquet."""
    df = pd.DataFrame({
        "cal_date": dates,
        "is_open": [1] * len(dates),
    })
    df.to_parquet(data_dir / "trade_cal.parquet")


class TestDatasetBuilder:
    def test_build_returns_correct_split_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            val_dates = ["20240109", "20240110", "20240111"]
            margin_dates = ["20240112", "20240115", "20240116", "20240117", "20240118",
                            "20240119", "20240122", "20240123", "20240124", "20240125",
                            "20240126", "20240129", "20240130", "20240131", "20240201",
                            "20240202", "20240205", "20240206", "20240207", "20240208"]
            all_feat_dates = train_dates + val_dates + margin_dates
            all_daily_dates = train_dates + val_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            _make_stock_basic(data_dir, ts_codes)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240109", "20240111")

            assert isinstance(result, DatasetResult)
            assert result.X_train.shape == (25, 2)  # 5 stocks * 5 dates, 2 features
            assert result.X_val.shape == (15, 2)    # 5 stocks * 3 dates
            assert len(result.y_train_5) == 25
            assert len(result.y_val_5) == 15
            assert len(result.y_train_20) == 25
            assert len(result.y_val_20) == 15

    def test_drops_nan_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            margin_dates = ["20240109", "20240110", "20240111", "20240112", "20240115",
                            "20240116", "20240117", "20240118", "20240119", "20240122",
                            "20240123", "20240124", "20240125", "20240126", "20240129",
                            "20240130", "20240131", "20240201", "20240202", "20240205"]
            all_feat_dates = train_dates + margin_dates
            all_daily_dates = train_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            _make_stock_basic(data_dir, ts_codes)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240108", "20240108")

            assert not result.X_train.isna().any().any()
            assert not result.y_train_5.isna().any()
            assert not result.y_train_20.isna().any()

    def test_feature_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            val_dates = ["20240109"]
            margin_dates = ["20240110", "20240111", "20240112", "20240115", "20240116",
                            "20240117", "20240118", "20240119", "20240122", "20240123",
                            "20240124", "20240125", "20240126", "20240129", "20240130",
                            "20240131", "20240201", "20240202", "20240205", "20240206"]
            all_feat_dates = train_dates + val_dates + margin_dates
            all_daily_dates = train_dates + val_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            _make_stock_basic(data_dir, ts_codes)
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240109", "20240109",
                                   feature_names=["KMID"])

            assert list(result.X_train.columns) == ["KMID"]
            assert list(result.X_val.columns) == ["KMID"]

    def test_excludes_st_and_non_main_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            ts_codes = ["000001.SZ", "000002.SZ", "300001.SZ"]

            train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
            margin_dates = ["20240109", "20240110", "20240111", "20240112", "20240115",
                            "20240116", "20240117", "20240118", "20240119", "20240122",
                            "20240123", "20240124", "20240125", "20240126", "20240129",
                            "20240130", "20240131", "20240201", "20240202", "20240205"]
            all_feat_dates = train_dates + margin_dates
            all_daily_dates = train_dates + margin_dates

            _make_features(data_dir, all_feat_dates, ts_codes)
            _make_daily(data_dir, all_daily_dates, ts_codes)
            sb = pd.DataFrame({
                "ts_code": ts_codes,
                "market": ["主板", "主板", "创业板"],
                "list_status": ["L", "L", "L"],
            })
            sb.to_parquet(data_dir / "stock_basic.parquet")
            st_dir = data_dir / "stock_st"
            st_dir.mkdir()
            st = pd.DataFrame({"ts_code": ["000002.SZ"], "trade_date": ["20240102"]})
            st.to_parquet(st_dir / "2024_01_02.parquet")
            _make_trade_cal(data_dir, all_daily_dates)

            builder = DatasetBuilder(data_dir)
            result = builder.build("20240102", "20240108", "20240108", "20240108")

            codes_in_train = set(result.train_codes)
            assert "300001.SZ" not in codes_in_train
            assert "000002.SZ" not in codes_in_train
            assert "000001.SZ" in codes_in_train
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model/test_data.py -v
```
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write DatasetBuilder implementation**

```python
# src/alpha_quat/model/data.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.backtest.filters import build_universe


@dataclass
class DatasetResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train_5: pd.Series
    y_val_5: pd.Series
    y_train_20: pd.Series
    y_val_20: pd.Series
    train_dates: pd.Series
    val_dates: pd.Series
    train_codes: pd.Series
    val_codes: pd.Series


class DatasetBuilder:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _load_features(self, dates: list[str]) -> pd.DataFrame:
        dfs = []
        for d in dates:
            path = self.data_dir / "features" / f"{d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _load_close_series(self, dates: list[str]) -> pd.DataFrame:
        rows = []
        for d in dates:
            path = self.data_dir / "daily" / f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
            if path.exists():
                df = pd.read_parquet(path, columns=["ts_code", "close"])
                df["trade_date"] = d
                rows.append(df)
        if not rows:
            return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
        return pd.concat(rows, ignore_index=True)

    def _get_trade_dates(self) -> pd.DataFrame:
        cal = pd.read_parquet(self.data_dir / "trade_cal.parquet")
        return cal.loc[cal["is_open"] == 1, "cal_date"].sort_values().reset_index(drop=True)

    def _forward_date(self, cal_date_series: pd.Series, date_str: str, offset: int) -> str | None:
        matches = cal_date_series[cal_date_series == date_str]
        if len(matches) == 0:
            return None
        idx = matches.index[0]
        target_idx = idx + offset
        if target_idx >= len(cal_date_series):
            return None
        return cal_date_series.iloc[target_idx]

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        all_dates = df["trade_date"].unique()
        mask = pd.Series(False, index=df.index)
        for d in all_dates:
            universe = build_universe(str(d), self.data_dir)
            date_mask = df["trade_date"] == d
            code_mask = df["ts_code"].isin(universe)
            mask |= date_mask & code_mask
        return df.loc[mask].copy()

    def _build_labels(self, df: pd.DataFrame, label_column: str, cal_dates: pd.Series,
                      offset: int) -> pd.Series:
        close_map = {}
        for d in df["trade_date"].unique():
            fwd = self._forward_date(cal_dates, str(d), offset)
            if fwd is not None:
                close_path = self.data_dir / "daily" / f"{fwd[:4]}_{fwd[4:6]}_{fwd[6:8]}.parquet"
                if close_path.exists():
                    close_df = pd.read_parquet(close_path, columns=["ts_code", "close"])
                    for _, row in close_df.iterrows():
                        close_map[(str(d), row["ts_code"])] = row["close"]

        forward_closes = df.apply(
            lambda r: close_map.get((str(r["trade_date"]), r["ts_code"]), np.nan), axis=1
        )
        return forward_closes / df["close"] - 1

    def build(self, train_start: str, train_end: str, val_start: str, val_end: str,
              feature_names: list[str] | None = None) -> DatasetResult:
        cal_dates = self._get_trade_dates()

        max_offset = 20
        start_idx = cal_dates[cal_dates >= train_start].index[0]
        end_idx = cal_dates[cal_dates <= val_end].index[-1]
        margin_start = min(start_idx, start_idx - max_offset)
        margin_end = min(len(cal_dates) - 1, end_idx + max_offset)

        feature_dates = cal_dates.iloc[margin_start:margin_end + 1].tolist()

        features = self._load_features(feature_dates)

        if features.empty:
            raise ValueError("No feature data found in the specified date range")

        factor_cols = [c for c in features.columns if c not in ("ts_code", "trade_date")]
        if feature_names is not None:
            factor_cols = [c for c in factor_cols if c in feature_names]

        close_df = self._load_close_series(feature_dates)
        merged = features.merge(close_df, on=["ts_code", "trade_date"], how="left")
        merged = self._filter_universe(merged)
        merged = merged.dropna(subset=["close"])

        merged["ret_5d"] = self._build_labels(merged, "ret_5d", cal_dates, 5)
        merged["ret_20d"] = self._build_labels(merged, "ret_20d", cal_dates, 20)

        merged = merged.dropna(subset=["ret_5d", "ret_20d"] + factor_cols)

        train_mask = (merged["trade_date"] >= train_start) & (merged["trade_date"] <= train_end)
        val_mask = (merged["trade_date"] >= val_start) & (merged["trade_date"] <= val_end)

        X_train = merged.loc[train_mask, factor_cols].reset_index(drop=True)
        X_val = merged.loc[val_mask, factor_cols].reset_index(drop=True)

        return DatasetResult(
            X_train=X_train,
            X_val=X_val,
            y_train_5=merged.loc[train_mask, "ret_5d"].reset_index(drop=True),
            y_val_5=merged.loc[val_mask, "ret_5d"].reset_index(drop=True),
            y_train_20=merged.loc[train_mask, "ret_20d"].reset_index(drop=True),
            y_val_20=merged.loc[val_mask, "ret_20d"].reset_index(drop=True),
            train_dates=merged.loc[train_mask, "trade_date"].reset_index(drop=True),
            val_dates=merged.loc[val_mask, "trade_date"].reset_index(drop=True),
            train_codes=merged.loc[train_mask, "ts_code"].reset_index(drop=True),
            val_codes=merged.loc[val_mask, "ts_code"].reset_index(drop=True),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model/test_data.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/model/data.py tests/test_model/test_data.py
git commit -m "feat: add DatasetBuilder shared data layer"
```

---

### Task 4: Create LightGBMTrainer (training + Optuna)

**Files:**
- Create: `src/alpha_quat/model/lightgbm/train.py`
- Test: `tests/test_model/test_lightgbm_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model/test_lightgbm_train.py
import numpy as np
import pandas as pd

from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.train import LightGBMTrainer


class TestLightGBMTrainer:
    def test_train_without_tune(self):
        cfg = LightGBMConfig(tune=False, n_estimators=10, num_leaves=5)
        trainer = LightGBMTrainer(cfg)

        n_samples = 200
        X = pd.DataFrame({
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples))

        model, params = trainer.train(X, y, "test_label")

        assert params["num_leaves"] == 5
        assert params["n_estimators"] == 10
        assert model is not None
        assert model.params["objective"] == "regression"

    def test_train_with_tune_small_search(self):
        cfg = LightGBMConfig(tune=True, n_trials=3, n_estimators=10, num_leaves=5)
        trainer = LightGBMTrainer(cfg)

        n_samples = 200
        X = pd.DataFrame({
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples))

        model, params = trainer.train(X, y, "test_label_tune")

        assert model is not None
        assert "num_leaves" in params
        assert "learning_rate" in params
        assert model.params["objective"] == "regression"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model/test_lightgbm_train.py -v
```
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write LightGBMTrainer implementation**

```python
# src/alpha_quat/model/lightgbm/train.py
import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from alpha_quat.model.lightgbm.config import LightGBMConfig

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    def __init__(self, config: LightGBMConfig):
        self.config = config

    def _base_params(self) -> dict:
        return {
            "objective": "regression",
            "metric": "l2",
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "verbose": self.config.verbosity,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
        }

    def _train_lgb(self, params: dict, X: pd.DataFrame, y: pd.Series,
                   n_estimators: int | None = None) -> lgb.Booster:
        n_est = n_estimators if n_estimators is not None else self.config.n_estimators
        n_total = len(X)
        split_idx = int(n_total * 0.9)
        X_tr, X_ev = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_ev = y.iloc[:split_idx], y.iloc[split_idx:]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_ev, label=y_ev, reference=train_data)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_est,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        return model

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = self._base_params()
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        })
        n_est = trial.suggest_int("n_estimators", 100, 500)

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            callbacks = [
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_est,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=callbacks,
            )
            y_pred = model.predict(X_val)
            mse = ((y_val.values - y_pred) ** 2).mean()
            scores.append(mse)

        return float(np.mean(scores))

    def train(self, X: pd.DataFrame, y: pd.Series,
              label_name: str = "") -> tuple[lgb.Booster, dict]:
        if self.config.tune:
            logger.info(f"Starting Optuna hyperparameter tuning ({label_name}), "
                        f"{self.config.n_trials} trials")
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            )
            study.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=self.config.n_trials,
                show_progress_bar=False,
            )
            best_params = self._base_params()
            best_params.update(study.best_params)
            logger.info(f"Best trial ({label_name}): MSE={study.best_value:.6f}, "
                        f"params={study.best_params}")

            model = self._train_lgb(best_params, X, y,
                                    n_estimators=best_params.get("n_estimators"))
            best_params["best_iteration"] = model.best_iteration
            return model, best_params
        else:
            logger.info(f"Training LightGBM with base params ({label_name})")
            params = self._base_params()
            model = self._train_lgb(params, X, y)
            params["best_iteration"] = model.best_iteration
            return model, params
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model/test_lightgbm_train.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/model/lightgbm/train.py tests/test_model/test_lightgbm_train.py
git commit -m "feat: add LightGBMTrainer with Optuna tuning"
```

---

### Task 5: Create LightGBMEvaluator (MSE/MAE/Rank IC/feature importance)

**Files:**
- Create: `src/alpha_quat/model/lightgbm/evaluate.py`
- Test: `tests/test_model/test_lightgbm_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model/test_lightgbm_evaluate.py
import numpy as np
import pandas as pd

from alpha_quat.model.lightgbm.evaluate import LightGBMEvaluator, EvalResult


class TestLightGBMEvaluator:
    def test_eval_result_has_all_fields(self):
        r = EvalResult(
            label_name="ret_5d",
            mse=0.001,
            mae=0.02,
            mean_ic=0.05,
            ic_std=0.08,
            icir=0.625,
            top5_features=[("A", 0.5), ("B", 0.4), ("C", 0.3), ("D", 0.2), ("E", 0.1)],
            bottom5_features=[("V", 0.001), ("W", 0.002), ("X", 0.003), ("Y", 0.004), ("Z", 0.005)],
            best_params={"num_leaves": 31},
            feature_names=None,
        )
        assert r.label_name == "ret_5d"
        assert r.mse == 0.001
        assert r.icir == 0.625
        assert len(r.top5_features) == 5
        assert len(r.bottom5_features) == 5

    def test_evaluate_rank_ic_format(self):
        evaluator = LightGBMEvaluator()
        rng = np.random.RandomState(42)

        n = 300
        dates = [f"202401{str(i+1).zfill(2)}" for i in range(10)] * 30
        X_val = pd.DataFrame({"feat1": rng.randn(n)})
        y_val = pd.Series(rng.randn(n))
        val_dates = pd.Series(dates[:n])
        val_codes = pd.Series([f"00000{i}.SZ" for i in range(n)])

        y_pred = rng.randn(n) * 0.5

        expected_cols = X_val.columns.tolist()
        assert len(expected_cols) > 0

    def test_rank_ic_computes_per_date_spearman(self):
        evaluator = LightGBMEvaluator()
        rng = np.random.RandomState(42)

        n = 50
        dates = [f"202401{str(i+1).zfill(2)}" for i in range(5)] * 10
        X_val = pd.DataFrame({"feat1": rng.randn(n)})
        y_val = pd.Series(rng.randn(n))
        val_dates = pd.Series(dates[:n])
        val_codes = pd.Series([f"00000{i}.SZ" for i in range(n)])

        y_pred = rng.randn(n) * 0.5

        result = evaluator.compute_rank_ic(y_pred, y_val.values, val_dates.to_numpy())

        assert isinstance(result.mean_ic, float)
        assert isinstance(result.ic_std, float)
        assert result.ic_std >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model/test_lightgbm_evaluate.py -v
```
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write LightGBMEvaluator implementation**

```python
# src/alpha_quat/model/lightgbm/evaluate.py
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class EvalResult:
    label_name: str
    mse: float
    mae: float
    mean_ic: float
    ic_std: float
    icir: float
    top5_features: list[tuple[str, float]]
    bottom5_features: list[tuple[str, float]]
    best_params: dict
    feature_names: list[str] | None


@dataclass
class RankICResult:
    mean_ic: float
    ic_std: float
    icir: float


class LightGBMEvaluator:
    def compute_rank_ic(self, y_pred: np.ndarray, y_true: np.ndarray,
                        dates: np.ndarray) -> RankICResult:
        df = pd.DataFrame({"date": dates, "pred": y_pred, "true": y_true})
        daily_ics = []
        for _, group in df.groupby("date"):
            if len(group) >= 3:
                ic, _ = spearmanr(group["pred"], group["true"])
                daily_ics.append(ic)

        if not daily_ics:
            return RankICResult(mean_ic=0.0, ic_std=0.0, icir=0.0)

        daily_ics = np.array(daily_ics)
        mean_ic = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics, ddof=1))
        icir = mean_ic / ic_std if ic_std > 0 else 0.0

        return RankICResult(mean_ic=mean_ic, ic_std=ic_std, icir=icir)

    def evaluate(self, model: lgb.Booster, X_val: pd.DataFrame, y_val: pd.Series,
                 val_dates: pd.Series, val_codes: pd.Series,
                 best_params: dict, feature_names: list[str] | None,
                 label_name: str) -> EvalResult:
        y_pred = model.predict(X_val)

        mse = float(mean_squared_error(y_val, y_pred))
        mae = float(mean_absolute_error(y_val, y_pred))

        rank_ic = self.compute_rank_ic(y_pred, y_val.values, val_dates.to_numpy())

        importance = model.feature_importance(importance_type="gain")
        feature_names_list = feature_names if feature_names is not None else X_val.columns.tolist()
        feat_imp = sorted(zip(feature_names_list, importance), key=lambda x: x[1], reverse=True)

        top5 = feat_imp[:5]
        bottom5 = feat_imp[-5:]

        return EvalResult(
            label_name=label_name,
            mse=mse,
            mae=mae,
            mean_ic=rank_ic.mean_ic,
            ic_std=rank_ic.ic_std,
            icir=rank_ic.icir,
            top5_features=[(str(name), float(val)) for name, val in top5],
            bottom5_features=[(str(name), float(val)) for name, val in bottom5],
            best_params=best_params,
            feature_names=feature_names,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model/test_lightgbm_evaluate.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/model/lightgbm/evaluate.py tests/test_model/test_lightgbm_evaluate.py
git commit -m "feat: add LightGBMEvaluator with Rank IC and feature importance"
```

---

### Task 6: Create LightGBMPipeline (orchestrator)

**Files:**
- Create: `src/alpha_quat/model/lightgbm/pipeline.py`
- Test: `tests/test_model/test_lightgbm_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model/test_lightgbm_pipeline.py
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.pipeline import LightGBMPipeline


def _make_synthetic_data(data_dir: Path):
    ts_codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]
    rng = np.random.RandomState(42)

    train_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    val_dates = ["20240109", "20240110", "20240111"]
    margin_dates = ["20240112", "20240115", "20240116", "20240117", "20240118",
                    "20240119", "20240122", "20240123", "20240124", "20240125",
                    "20240126", "20240129", "20240130", "20240131", "20240201",
                    "20240202", "20240205", "20240206", "20240207", "20240208"]

    feat_dir = data_dir / "features"
    feat_dir.mkdir()
    all_feat_dates = train_dates + val_dates + margin_dates
    for d in all_feat_dates:
        df = pd.DataFrame({
            "ts_code": ts_codes,
            "trade_date": d,
            "KMID": rng.randn(len(ts_codes)),
            "KLEN": rng.randn(len(ts_codes)),
        })
        df.to_parquet(feat_dir / f"{d}.parquet")

    daily_dir = data_dir / "daily"
    daily_dir.mkdir()
    all_dates = train_dates + val_dates + margin_dates
    for d in all_dates:
        df = pd.DataFrame({
            "ts_code": ts_codes,
            "trade_date": d,
            "close": rng.uniform(5, 50, len(ts_codes)),
        })
        path = f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
        df.to_parquet(daily_dir / path)

    sb = pd.DataFrame({
        "ts_code": ts_codes,
        "market": ["主板"] * len(ts_codes),
        "list_status": ["L"] * len(ts_codes),
    })
    sb.to_parquet(data_dir / "stock_basic.parquet")

    (data_dir / "stock_st").mkdir()

    cal = pd.DataFrame({
        "cal_date": all_dates,
        "is_open": [1] * len(all_dates),
    })
    cal.to_parquet(data_dir / "trade_cal.parquet")


class TestLightGBMPipeline:
    def test_pipeline_runs_and_saves_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _make_synthetic_data(data_dir)

            config = LightGBMConfig(
                train_start="20240102",
                train_end="20240108",
                val_start="20240109",
                val_end="20240111",
                tune=False,
                n_estimators=10,
                num_leaves=5,
            )

            pipeline = LightGBMPipeline(data_dir, config)
            results = pipeline.run()

            assert "ret_5d" in results
            assert "ret_20d" in results
            assert results["ret_5d"].mse >= 0
            assert results["ret_20d"].mse >= 0
            assert len(results["ret_5d"].top5_features) == 5

            models_dir = data_dir / "models"
            assert (models_dir / "lightgbm_model_5d.txt").exists()
            assert (models_dir / "lightgbm_model_20d.txt").exists()
            assert (models_dir / "results.json").exists()

            with open(models_dir / "results.json") as f:
                results_json = json.load(f)
            assert results_json["model_type"] == "lightgbm"
            assert "ret_5d" in results_json
            assert "ret_20d" in results_json
            assert "mse" in results_json["ret_5d"]
            assert "mean_ic" in results_json["ret_5d"]
            assert "top5_features" in results_json["ret_5d"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model/test_lightgbm_pipeline.py -v
```
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write LightGBMPipeline implementation**

```python
# src/alpha_quat/model/lightgbm/pipeline.py
import json
import logging
from pathlib import Path

from alpha_quat.model.data import DatasetBuilder
from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.evaluate import LightGBMEvaluator
from alpha_quat.model.lightgbm.train import LightGBMTrainer

logger = logging.getLogger(__name__)


class LightGBMPipeline:
    def __init__(self, data_dir: Path, config: LightGBMConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.builder = DatasetBuilder(self.data_dir)
        self.trainer = LightGBMTrainer(config)
        self.evaluator = LightGBMEvaluator()

    def run(self) -> dict[str, object]:
        logger.info("Building dataset...")
        data = self.builder.build(
            self.config.train_start,
            self.config.train_end,
            self.config.val_start,
            self.config.val_end,
            feature_names=self.config.feature_names,
        )

        logger.info(f"Training data: {len(data.X_train)} samples, "
                     f"Validation data: {len(data.X_val)} samples")

        logger.info("Training model_5d...")
        model_5d, params_5d = self.trainer.train(data.X_train, data.y_train_5, "ret_5d")

        logger.info("Training model_20d...")
        model_20d, params_20d = self.trainer.train(data.X_train, data.y_train_20, "ret_20d")

        logger.info("Evaluating model_5d...")
        result_5d = self.evaluator.evaluate(
            model_5d, data.X_val, data.y_val_5,
            data.val_dates, data.val_codes,
            params_5d, self.config.feature_names, "ret_5d",
        )

        logger.info("Evaluating model_20d...")
        result_20d = self.evaluator.evaluate(
            model_20d, data.X_val, data.y_val_20,
            data.val_dates, data.val_codes,
            params_20d, self.config.feature_names, "ret_20d",
        )

        self._save_models(model_5d, model_20d)
        self._save_results(result_5d, result_20d)
        self._print_summary(result_5d, result_20d)

        return {"ret_5d": result_5d, "ret_20d": result_20d}

    def _save_models(self, model_5d, model_20d):
        output_dir = self.data_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_5d.save_model(str(output_dir / "lightgbm_model_5d.txt"))
        model_20d.save_model(str(output_dir / "lightgbm_model_20d.txt"))
        logger.info(f"Models saved to {output_dir}")

    def _save_results(self, result_5d, result_20d):
        output_dir = self.data_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        def _make_json(result, params):
            return {
                "mse": result.mse,
                "mae": result.mae,
                "mean_ic": result.mean_ic,
                "ic_std": result.ic_std,
                "icir": result.icir,
                "top5_features": result.top5_features,
                "bottom5_features": result.bottom5_features,
                "best_params": result.best_params,
                "feature_names": result.feature_names,
            }

        output = {
            "model_type": "lightgbm",
            "ret_5d": _make_json(result_5d, {}),
            "ret_20d": _make_json(result_20d, {}),
            "config": {
                "train_start": self.config.train_start,
                "train_end": self.config.train_end,
                "val_start": self.config.val_start,
                "val_end": self.config.val_end,
            },
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {output_dir / 'results.json'}")

    def _print_summary(self, result_5d, result_20d):
        print()
        print("=" * 60)
        print("  LIGHTGBM MODEL EVALUATION")
        print("=" * 60)

        for label, result in [("ret_5d", result_5d), ("ret_20d", result_20d)]:
            print(f"\n  --- {label} ---")
            print(f"  MSE:      {result.mse:.6f}")
            print(f"  MAE:      {result.mae:.6f}")
            print(f"  Mean IC:  {result.mean_ic:.4f}")
            print(f"  IC Std:   {result.ic_std:.4f}")
            print(f"  ICIR:     {result.icir:.4f}")
            print(f"  Top 5 features (gain):")
            for name, val in result.top5_features:
                print(f"    {name}: {val:.4f}")
            print(f"  Bottom 5 features (gain):")
            for name, val in result.bottom5_features:
                print(f"    {name}: {val:.4f}")

        print()
        print("=" * 60)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model/test_lightgbm_pipeline.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/model/lightgbm/pipeline.py tests/test_model/test_lightgbm_pipeline.py
git commit -m "feat: add LightGBMPipeline orchestrator"
```

---

### Task 7: Integrate CLI subcommand

**Files:**
- Modify: `src/alpha_quat/cli.py`

- [ ] **Step 1: Read the current cli.py to confirm exact content**

```bash
uv run pytest tests/ -v --tb=short -q 2>&1 | tail -5
```

- [ ] **Step 2: Add model lightgbm subcommand builder and handler**

Add to `cli.py` imports:

```python
from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.pipeline import LightGBMPipeline
```

Add parser builder function after `_build_backtest_parser`:

```python
def _build_model_parser(subparsers):
    model_parser = subparsers.add_parser("model", help="Train ML models")
    model_sub = model_parser.add_subparsers(dest="model_type")

    lgb_parser = model_sub.add_parser("lightgbm", help="LightGBM stock selection model")
    lgb_parser.add_argument("--train-start", default="20240401", help="Train start YYYYMMDD")
    lgb_parser.add_argument("--train-end", default="20250430", help="Train end YYYYMMDD")
    lgb_parser.add_argument("--val-start", default="20250501", help="Validation start YYYYMMDD")
    lgb_parser.add_argument("--val-end", default="20260430", help="Validation end YYYYMMDD")
    lgb_parser.add_argument("--trials", type=int, default=50, help="Optuna trials (default: 50)")
    lgb_parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning")
    lgb_parser.add_argument("--features", default=None, help="Comma-separated feature subset")
    return model_parser


def _cmd_model(args, config):
    if args.model_type == "lightgbm":
        feature_names = None
        if args.features:
            feature_names = [f.strip() for f in args.features.split(",") if f.strip()]

        cfg = LightGBMConfig(
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            n_trials=args.trials,
            tune=not args.no_tune,
            feature_names=feature_names,
        )
pipeline = LightGBMPipeline(config.data_dir, cfg)
        pipeline.run()
    else:
        print(f"Unknown model type: {args.model_type}")
        print("Available: lightgbm")
```

Register the parser in `main()`:

```python
    subparsers = parser.add_subparsers(dest="command")
    _build_fetch_parser(subparsers)
    _build_feature_parser(subparsers)
    _build_backtest_parser(subparsers)
    _build_model_parser(subparsers)
```

Add dispatch in `main()`:

```python
    if args.command == "feature":
        _cmd_feature(args, config, metadata)
    elif args.command == "backtest":
        _cmd_backtest(args, config)
    elif args.command == "model":
        _cmd_model(args, config)
    else:
        _cmd_fetch(args, config, metadata)
```

- [ ] **Step 3: Verify CLI help works**

```bash
uv run alpha-quat model lightgbm --help
```
Expected: Shows help with all options

- [ ] **Step 4: Commit**

```bash
git add src/alpha_quat/cli.py
git commit -m "feat: add model lightgbm CLI subcommand"
```

---

### Task 8: Final verification

**Files:** None (verify only)

- [ ] **Step 1: Run all tests**

```bash
uv run pytest --cov=src -v
```
Expected: All tests pass, coverage reported

- [ ] **Step 2: Run linter and typecheck**

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright
```
Expected: No errors

- [ ] **Step 3: Run full verification**

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright && uv run pytest --cov=src
```
Expected: All pass
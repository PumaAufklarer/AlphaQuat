# LightGBM Stock Selection Model Design

## Overview

Build a LightGBM-based stock selection module that trains regression models on Alpha158
features to predict future returns (5-day and 20-day), with Optuna hyperparameter tuning,
cross-sectional Rank IC evaluation, and feature importance analysis for pruning.

## Requirements

- **Features**: Alpha158 factor set (158 daily-frequency factors from `features/YYYYMMDD.parquet`).
  Future factor sets can be added or pruned via `--features` flag.
- **Labels**: `ret_5d = close_{t+5} / close_t - 1` and `ret_20d = close_{t+20} / close_t - 1`.
  Future close prices read from `daily/YYYY_MM_DD.parquet`.
- **Data split**: Train 2024-04 to 2025-04, validation 2025-04 to 2026-04 (configurable via CLI).
- **Filters**: Main board only (`stock_basic.market == "主板"`), exclude ST stocks per date
  (reuse existing `build_universe()` from `backtest/filters.py`).
- **Missing values**: Drop rows with any NaN features.
- **Model**: LightGBM regression (`objective="regression"`). Two independent models: `model_5d` and `model_20d`.
- **Hyperparameters**: Base config given, optional Optuna tuning (50 trials with `TimeSeriesSplit`).
- **Early stopping**: LightGBM native `early_stopping_rounds=20` with 10% training holdout as eval set.
- **Evaluation**: MSE, MAE, per-date cross-sectional Spearman Rank IC (mean/std/ICIR),
  feature importance (gain) — top 5 and bottom 5.
- **Output**: Saved model files (`data/models/lightgbm_model_{5d,20d}.txt`),
  JSON results (`data/models/results.json`), terminal printout.
- **Pruning support**: `feature_names` config field + `--features` CLI flag to select subset of factors.
  `results.json` records which features were used.

## Architecture

New module `src/alpha_quat/model/` with a shared data layer and a `lightgbm/` subpackage
for LightGBM-specific logic. Future models (XGBoost, etc.) add parallel subpackages
reusing the shared `data.py`.

```
src/alpha_quat/model/
├── __init__.py
├── data.py              # DatasetBuilder — shared: load features + labels + universe filter
└── lightgbm/
    ├── __init__.py
    ├── config.py        # LightGBMConfig dataclass
    ├── train.py         # LightGBMTrainer — training with optional Optuna tuning
    ├── evaluate.py      # LightGBMEvaluator — MSE/MAE/Rank IC/ICIR/feature importance
    └── pipeline.py      # LightGBMPipeline — orchestrator: data → train → eval → save
```

### Data layer relationship

```
data/
├── daily/YYYY_MM_DD.parquet        ← read for close prices (label construction)
├── features/YYYYMMDD.parquet       ← read for 158 factor values (model input)
├── stock_basic.parquet             ← read for market field (universe filter)
├── stock_st/YYYY_MM_DD.parquet     ← read for ST exclusion per date
├── trade_cal.parquet               ← read for forward-date lookup (t+5, t+20)
└── models/
    ├── lightgbm_model_5d.txt       ← saved model
    ├── lightgbm_model_20d.txt      ← saved model
    └── results.json                ← evaluation results
```

## LightGBMConfig

```python
@dataclass
class LightGBMConfig:
    # Date ranges
    train_start: str = "20240401"
    train_end: str = "20250430"
    val_start: str = "20250501"
    val_end: str = "20260430"

    # Base hyperparameters
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

    # Feature selection (pruning)
    feature_names: list[str] | None = None  # None = use all 158
```

## DatasetBuilder (shared)

```python
@dataclass
class DatasetResult:
    X_train: pd.DataFrame    # (n_train_samples, n_features)
    X_val: pd.DataFrame      # (n_val_samples, n_features)
    y_train_5: pd.Series     # ret_5d labels
    y_val_5: pd.Series
    y_train_20: pd.Series    # ret_20d labels
    y_val_20: pd.Series
    train_dates: pd.Series   # trade_date per sample (for Rank IC)
    val_dates: pd.Series
    train_codes: pd.Series   # ts_code per sample
    val_codes: pd.Series

class DatasetBuilder:
    def __init__(self, data_dir: Path): ...

    def build(self, train_start, train_end, val_start, val_end,
              feature_names=None) -> DatasetResult: ...
```

### Build pipeline

```
1. Read features/{YYYYMMDD}.parquet for all dates in [train_start, val_end + margin]
2. Read daily/{YYYY_MM_DD}.parquet for close prices
3. For each trade_date t, find t+5 and t+20 trading days via trade_cal
4. Join forward close → compute ret_5d, ret_20d
5. Filter universe per date: build_universe(date, data_dir) → main board minus ST
6. Split by trade_date: train_start..train_end → train, val_start..val_end → val
7. If feature_names is not None, select only those columns
8. Drop rows with any NaN
```

### Forward date lookup

For a trading date `t` at index `i` in the sorted `trade_cal` list:
- `t+5` is `trade_cal[i+5]`, `t+20` is `trade_cal[i+20]`
- Load close from `daily/{t+5}.parquet` and `daily/{t+20}.parquet`
- If `t+5` or `t+20` don't exist (no future data), label is NaN → dropped

## LightGBMTrainer

```python
class LightGBMTrainer:
    def __init__(self, config: LightGBMConfig): ...

    def train(self, X, y, label_name: str) -> tuple[lgb.Booster, dict]:
        """
        Returns: (trained_model, best_params_dict)
        """
```

### Training flow

**With tuning (`config.tune=True`, default):**
1. `TimeSeriesSplit(n_splits=5)` on training data
2. Optuna study with `direction="minimize"`, metric = validation MSE
3. Search space:
   - `num_leaves`: int 15~63
   - `learning_rate`: float 0.01~0.2 (log scale)
   - `n_estimators`: int 100~500
   - `feature_fraction`: float 0.5~1.0
   - `bagging_fraction`: float 0.5~1.0
   - `min_child_samples`: int 10~100
   - `reg_alpha`: float 1e-8~10 (log scale)
   - `reg_lambda`: float 1e-8~10 (log scale)
4. Per trial: train on 4 folds, validate on 1 fold (last fold for early stopping)
5. After 50 trials: retrain final model on full training data with best params
6. Native early stopping: 10% training data held out as eval_set, `early_stopping_rounds=20`
7. Record `best_iteration` from early stopping

**Without tuning (`config.tune=False`):**
1. Train directly with base config params
2. Same early stopping setup, record `best_iteration`

**Uses LightGBM native API** (`lgb.train`) for better performance and memory efficiency.

## LightGBMEvaluator

```python
@dataclass
class EvalResult:
    label_name: str                      # "ret_5d" | "ret_20d"
    mse: float
    mae: float
    mean_ic: float                       # average daily cross-sectional Spearman
    ic_std: float                        # std of daily ICs
    icir: float                          # mean_ic / ic_std
    top5_features: list[tuple[str, float]]    # (name, gain)
    bottom5_features: list[tuple[str, float]]
    best_params: dict
    feature_names: list[str] | None

class LightGBMEvaluator:
    def evaluate(self, model: lgb.Booster, X_val, y_val,
                 val_dates, val_codes) -> EvalResult: ...
```

### Evaluation details

**MSE / MAE:**
- Overall `mean_squared_error` and `mean_absolute_error` on full validation set.

**Rank IC:**
1. Predict on validation set → `y_pred`
2. Group by `trade_date`
3. Per date: `scipy.stats.spearmanr(y_pred, y_true)` → daily IC
4. Aggregate: `mean_ic = mean(daily_ics)`, `ic_std = std(daily_ics)`, `icir = mean_ic / ic_std`

**Feature importance:**
- `model.feature_importance(importance_type="gain")`
- Sort descending by gain → top 5 (highest gain), bottom 5 (lowest gain)
- Record as `[(feature_name, gain_value), ...]`

## LightGBMPipeline

```python
class LightGBMPipeline:
    def __init__(self, data_dir: Path, config: LightGBMConfig): ...

    def run(self) -> dict[str, EvalResult]:
        # 1. Build dataset
        data = self.builder.build(...)
        # 2. Train two models
        model_5d, params_5d = self.trainer.train(data.X_train, data.y_train_5, "ret_5d")
        model_20d, params_20d = self.trainer.train(data.X_train, data.y_train_20, "ret_20d")
        # 3. Evaluate
        result_5d = self.evaluator.evaluate(model_5d, data.X_val, data.y_val_5, ...)
        result_20d = self.evaluator.evaluate(model_20d, data.X_val, data.y_val_20, ...)
        # 4. Save
        self._save_model(model_5d, "lightgbm_model_5d.txt")
        self._save_model(model_20d, "lightgbm_model_20d.txt")
        self._save_results(...)
        # 5. Print summary to terminal
        return {"ret_5d": result_5d, "ret_20d": result_20d}
```

### `data/models/results.json` format

```json
{
  "model_type": "lightgbm",
  "ret_5d": {
    "mse": 0.0012,
    "mae": 0.025,
    "mean_ic": 0.035,
    "ic_std": 0.08,
    "icir": 0.44,
    "top5_features": [["KLEN35", 0.12], ["KMID5", 0.09], ...],
    "bottom5_features": [["KMID2", 0.001], ...],
    "best_params": {"num_leaves": 27, "learning_rate": 0.03, ...},
    "feature_names": null
  },
  "ret_20d": {
    "mse": 0.0028,
    "mae": 0.041,
    "mean_ic": 0.042,
    "ic_std": 0.09,
    "icir": 0.47,
    "top5_features": [["KLEN60", 0.15], ...],
    "bottom5_features": [["KMID8", 0.002], ...],
    "best_params": {"num_leaves": 31, "learning_rate": 0.05, ...},
    "feature_names": null
  },
  "config": {
    "train_start": "20240401",
    "train_end": "20250430",
    "val_start": "20250501",
    "val_end": "20260430"
  }
}
```

## CLI Integration

```
uv run alpha-quat model lightgbm [options]

Options:
  --train-start YYYYMMDD    Training start date (default: 20240401)
  --train-end YYYYMMDD      Training end date (default: 20250430)
  --val-start YYYYMMDD      Validation start date (default: 20250501)
  --val-end YYYYMMDD        Validation end date (default: 20260430)
  --trials INT              Optuna trials (default: 50)
  --no-tune                 Skip Optuna, use base hyperparameters
  --features F1,F2,...      Comma-separated feature subset (default: all 158)
```

- `model` is the top-level subcommand (reserved for future `model xgboost` etc.)
- `lightgbm` is the model-type subcommand
- Terminal prints MSE, MAE, Rank IC stats, and feature importance for both models

## Pruning workflow (future)

1. First run with all features: `alpha-quat model lightgbm`
2. Review `results.json` → identify `bottom5_features` with lowest gain
3. Remove them: `alpha-quat model lightgbm --features KLEN35,KMID5,...` (excluding bottom 5)
4. Compare ICIR/MSE between runs to validate pruning effect
5. Iterate

## Dependencies to add

```toml
"lightgbm>=4.0",
"optuna>=4.0",
"scikit-learn>=1.5",
"scipy>=1.14",
"numpy>=2.0",
```

- `lightgbm` — model training
- `optuna` — hyperparameter tuning
- `scikit-learn` — `TimeSeriesSplit`, `mean_squared_error`, `mean_absolute_error`
- `scipy` — `spearmanr` for Rank IC
- `numpy` — array operations

## Testing

Unit tests per module:
- `tests/test_model/test_data.py` — label construction, universe filtering, NaN dropping, feature subset selection
- `tests/test_model/test_lightgbm_config.py` — default values
- `tests/test_model/test_lightgbm_train.py` — model trains without crashing, early stopping works, optuna/no-tune paths
- `tests/test_model/test_lightgbm_evaluate.py` — MSE/MAE calculation, Rank IC computation, feature importance extraction
- `tests/test_model/test_lightgbm_pipeline.py` — integration: end-to-end with synthetic data, model file + results.json saved

## Non-goals

- Inference/prediction for new dates (model saved for future use, but no `predict` CLI)
- Rolling window retraining
- Feature engineering beyond Alpha158
- GPU training
- Non-daily frequency
- Ensemble with other model types (deferred until other models exist)
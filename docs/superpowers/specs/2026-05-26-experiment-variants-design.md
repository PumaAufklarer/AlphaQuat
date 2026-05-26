# Experiment & Variant System Design

## Problem

LightGBM has multiple training variants (regression, quantile, lambdarank, meta,
tune/no-tune) and backtest variants (periodic rebalance, daily monitor, multiple
position sizing strategies). Currently all variants:

1. Are mixed via `if/else` conditionals in the same files
2. Write model files to the same directory (`data/models/`), clobbering each other
3. Have no way to preserve, name, or compare experiments
4. Require modifying core files to add a new variant

## Goals

- Each variant is a named experiment with its own model directory
- Each variant is a separate `.py` file inheriting from a base class
- CLI distinguishes variants via subcommands
- `experiment list/show` for visibility
- Backtest/predict can target a specific experiment

## Storage Layout

```
data/models/
├── registry.json                 ← index of all experiments
└── experiments/
    ├── exp_quantile_baseline/    ← named experiment directory
    │   ├── experiment.yaml       ← full config snapshot (human + machine readable)
    │   ├── lightgbm_model_5d_alpha_0.1.txt
    │   ├── lightgbm_model_5d_alpha_0.5.txt
    │   ├── ...
    │   ├── lightgbm_model_60d_alpha_0.9.txt
    │   └── results.json
    ├── exp_lambdarank_v1/
    │   ├── experiment.yaml
    │   └── ...
    └── exp_rolling_2026/
        ├── experiment.yaml
        ├── fold_1/
        │   ├── ...
        │   └── backtest_results.json
        └── fold_2/
            └── ...
```

Old `data/models/lightgbm_model_*.txt` files are abandoned (no migration needed).

### experiment.yaml

```yaml
name: exp_quantile_baseline
created_at: "2026-05-26T12:00:00"
mode: quantile
config:
  train_start: "20240401"
  train_end: "20250430"
  val_start: "20250501"
  val_end: "20260430"
  quantile_alphas:
    - 0.1
    - 0.5
    - 0.9
  tune: false
  n_trials: 50
  feature_names: null
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 200
  feature_fraction: 0.8
  bagging_fraction: 0.8
```

### registry.json

```json
[
  {"name": "exp_quantile_baseline", "mode": "quantile", "created": "2026-05-26T12:00:00"},
  {"name": "exp_lambdarank_v1", "mode": "lambdarank", "created": "2026-05-26T13:00:00"}
]
```

## Code Architecture

### Model training — variant per file

```
src/alpha_quat/model/lightgbm/
├── __init__.py
├── config.py              ← LightGBMConfig (shared base hyperparams)
├── evaluate.py            ← LightGBMEvaluator (shared)
├── train.py               ← LightGBMTrainer (shared)
├── pipeline.py            ← factory: selects variant by mode, orchestrates run
└── variants/
    ├── __init__.py         ← VARIANTS registry dict + register() function
    ├── baseline.py         ← LightGBMBasePipeline(ABC)
    ├── regression.py       ← RegressionPipeline(BasePipeline)
    ├── quantile.py         ← QuantilePipeline(BasePipeline)
    ├── lambdarank.py       ← LambdaRankPipeline(BasePipeline)
    └── meta.py             ← MetaPipeline(QuantilePipeline) — stacking layer
```

**LightGBMBasePipeline:**

```python
class LightGBMBasePipeline(ABC):
    @property
    @abstractmethod
    def mode(self) -> str: ...

    def run(self, data_dir: Path, config: LightGBMConfig, name: str) -> dict:
        data = self._build_dataset(data_dir, config)
        models = self._train(data, config)
        exp_dir = data_dir / "experiments" / name
        self._save_experiment_config(exp_dir, config, name)
        self._save_models(exp_dir, models)
        results = self._evaluate(models, data, config)
        self._save_results(exp_dir, results)
        self._register_experiment(data_dir, name, self.mode)
        return results

    @abstractmethod
    def _train(self, data, config) -> dict[str, lgb.Booster]: ...
```

**VARIANTS registry:**

```python
# variants/__init__.py
VARIANTS: dict[str, type[LightGBMBasePipeline]] = {}

def register(cls):
    VARIANTS[cls.mode] = cls
    return cls

# variants/quantile.py
@register
class QuantilePipeline(LightGBMBasePipeline):
    mode = "quantile"

    def _train(self, data, config) -> dict[str, lgb.Booster]:
        # trains 3 alphas × 3 horizons = 9 models
        ...
```

**pipeline.py factory:**

```python
def create_pipeline(mode: str) -> LightGBMBasePipeline:
    if mode not in VARIANTS:
        raise ValueError(f"Unknown variant: {mode}. Available: {list(VARIANTS)}")
    return VARIANTS[mode]()
```

### Signal generation — variant per file

MLSignalGenerator's runtime detection cascade is replaced by explicit variant classes.

```
src/alpha_quat/strategy/signals/variants/
├── __init__.py             ← VARIANTS registry + register()
├── baseline.py             ← BaseMLSignal(ABC)
├── regression_signal.py    ← RegressionSignal
├── quantile_signal.py      ← QuantileSignal
├── lambdarank_signal.py    ← LambdaRankSignal
└── meta_signal.py          ← MetaSignal
```

**BaseMLSignal:**

```python
class BaseMLSignal(ABC):
    @property
    @abstractmethod
    def mode(self) -> str: ...

    def __init__(self, model_dir: Path):
        self.models = self._load_models(model_dir)

    @abstractmethod
    def _load_models(self, model_dir: Path) -> dict: ...

    @abstractmethod
    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult: ...
```

The backtest's `MLSignalGenerator` class is replaced by `BaseMLSignal` subclasses.
The experiment's `experiment.yaml` `mode` field determines which class to instantiate.

### Backtest — rebalance strategy extraction

The `BacktestEngine.run()` 395-line method is split:

```
src/alpha_quat/backtest/
├── rebalance.py            ← PeriodicRebalance, DailyMonitor strategies
├── position_sizing.py      ← equal_weight, kelly, vol_parity, score_momentum
└── engine.py               ← BacktestEngine (strategy → day loop only)
```

**BacktestEngine:**

```python
class BacktestEngine:
    def run(self):
        for date in dates:
            # Monthly addition
            # Read daily data
            # Update peak prices
            # Stop-loss check (shared)
            # Delegate to strategy.on_date()
            self._strategy.on_date(date, portfolio, ...)
            # Snapshot
```

**Rebalance strategies:**

```python
class PeriodicRebalance:
    def __init__(self, config):
        self.position_sizer = PositionSizer.create(config.weighting_strategy)

    def on_date(self, date, portfolio, ...):
        # Existing rebalance logic, extracted from engine.py
        ...

class DailyMonitor:
    def __init__(self, config):
        self.position_sizer = PositionSizer.create(config.weighting_strategy)

    def on_date(self, date, portfolio, ...):
        # Existing daily monitor logic, extracted from engine.py
        ...
```

## CLI

```bash
# Train — --name is required for non-rolling
uv run alpha-quat model lightgbm quantile --name exp_quantile_v2 --no-tune
uv run alpha-quat model lightgbm lambdarank --name exp_lr_v1
uv run alpha-quat model lightgbm regression --name exp_reg_base --no-tune

# Experimental management
uv run alpha-quat experiment list
uv run alpha-quat experiment show exp_quantile_v2

# Backtest — --experiment points to a trained experiment
uv run alpha-quat backtest --experiment exp_quantile_v2

# Predict — --experiment points to a trained experiment
uv run alpha-quat predict --experiment exp_quantile_v2

# Rolling — creates sub-experiments under a shared name
uv run alpha-quat rolling --experiment exp_rolling_2026
```

**Backward compatibility:** If `--experiment` is omitted, backtest reads from the
latest experiment in `registry.json`. If no experiments exist, error with
instructions.

## Implementation Plan

### Phase 1: ExperimentConfig + Registry (data classes only)

1. Create `ExperimentConfig` dataclass with save/load to/from `experiment.yaml`
2. Create `ExperimentRegistry` for `registry.json` read/write/append

### Phase 2: Variant Files (Model Training)

3. Create `variants/` package with `baseline.py` (ABC) + `register()` decorator
4. Extract `_run_regression()` from current pipeline → `regression.py`
5. Extract `_run_lambdarank()` → `lambdarank.py`
6. Extract `_run_quantile()` → `quantile.py`
7. Extract `_run_meta()` → `meta.py` (extends QuantilePipeline)
8. Refactor `pipeline.py` as factory (`create_pipeline(mode)`)
9. Update CLI: `model lightgbm <variant> --name <name> [options]`
10. Pipeline run() saves models to `experiments/<name>/` + writes experiment.yaml + registry entry

### Phase 3: Variant Files (Signal Generation)

11. Create `strategy/signals/variants/` with `baseline.py` + `register()`
12. Extract each scoring mode from current `MLSignalGenerator` into its own class
13. Old `MLSignalGenerator` is deprecated (deleted after migration)
14. Backtest reads `mode` from experiment.yaml → selects signal class from registry

### Phase 4: Backtest Extraction

15. Extract `position_sizing.py` (kelly, vol_parity, score_momentum, equal strategies)
16. Extract `rebalance.py` (PeriodicRebalance, DailyMonitor strategy classes)
17. Simplify `BacktestEngine.run()` to delegate to rebalance strategy

### Phase 5: CLI + Rolling

18. Add `experiment list/show` subcommands
19. Update rolling: `--experiment <name>` → per-fold subdirs at `experiments/<name>/fold_N/` with own experiment.yaml
20. Update predict to read from experiment directory

## Non-goals

- MLflow / wandb integration
- Web UI for experiments
- Cross-variant model ensemble (e.g., blending quantile + lambdarank)
- Automatic experiment comparison reports (manual for now)
- Database-backed experiment store (file-based registry is sufficient)

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

    n_tile: int = 10
    label_gain: list[int] | None = None  # auto-exponential if None

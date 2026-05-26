from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str
    mode: str

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

    quantile_alphas: list[float] | None = None
    meta_start: str | None = None
    meta_end: str | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

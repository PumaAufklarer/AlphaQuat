from pathlib import Path

from alpha_quat.strategy.signal import ISignalGenerator


def create_signal_from_experiment(
    data_dir: str | Path, experiment_name: str
) -> ISignalGenerator:
    from alpha_quat.experiment.config import ExperimentConfig
    from alpha_quat.strategy.signals.variants import VARIANTS

    exp_dir = Path(data_dir) / "models" / "experiments" / experiment_name
    exp_cfg = ExperimentConfig.load(exp_dir / "experiment.yaml")
    cls = VARIANTS.get(exp_cfg.mode)
    if cls is None:
        raise ValueError(f"Unknown signal mode: {exp_cfg.mode}")
    return cls(exp_dir)

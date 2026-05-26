import logging
from pathlib import Path

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.variants import VARIANTS

logger = logging.getLogger(__name__)


def run_variant(data_dir: Path, config: ExperimentConfig) -> dict:
    if config.mode not in VARIANTS:
        raise ValueError(f"Unknown variant: {config.mode}. Available: {list(VARIANTS)}")
    pipeline = VARIANTS[config.mode]()
    return pipeline.run(data_dir, config)

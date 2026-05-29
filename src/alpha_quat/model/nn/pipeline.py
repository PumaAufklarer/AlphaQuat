import logging
from pathlib import Path

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.nn.variants import VARIANTS

logger = logging.getLogger(__name__)


def run_variant_nn(data_dir: Path, config: ExperimentConfig) -> dict:
    if config.mode not in VARIANTS:
        raise ValueError(
            f"Unknown NN variant: {config.mode}. Available: {list(VARIANTS)}"
        )
    pipeline = VARIANTS[config.mode]()
    return pipeline.run(data_dir, config)

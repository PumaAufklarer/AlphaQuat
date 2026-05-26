import logging
from pathlib import Path

from alpha_quat.experiment.config import ExperimentConfig

logger = logging.getLogger(__name__)


def run_variant_nn(data_dir: Path, config: ExperimentConfig) -> dict:
    """Dispatch NN training based on config.mode."""
    if config.mode == "sr_transformer":
        from alpha_quat.model.nn.transformer.variants.sr_transformer import (
            run_sr_transformer,
        )

        return run_sr_transformer(data_dir, config)
    raise ValueError(f"Unknown NN variant: {config.mode}")

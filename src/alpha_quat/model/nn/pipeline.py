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
    if config.mode == "keltner":
        from alpha_quat.model.nn.keltner.variants.keltner_regime import (
            run_keltner_regime,
        )

        return run_keltner_regime(data_dir, config)
    if config.mode == "rl_agent":
        from alpha_quat.model.nn.rl_agent.variants.rl_agent_variant import (
            run_rl_agent,
        )

        return run_rl_agent(data_dir, config)
    raise ValueError(f"Unknown NN variant: {config.mode}")

import json
import logging
from pathlib import Path

import torch

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.experiment.registry import ExperimentRegistry
from alpha_quat.model.nn.rl_agent.evaluate import evaluate
from alpha_quat.model.nn.rl_agent.models.position_agent import PositionAgent
from alpha_quat.model.nn.rl_agent.pretrain import pretrain
from alpha_quat.model.nn.rl_agent.train import train_rl
from alpha_quat.model.nn.variants import register
from alpha_quat.model.nn.variants import NNBasePipeline

logger = logging.getLogger(__name__)


@register
class RLAgentPipeline(NNBasePipeline):
    mode = "rl_agent"

    def run(self, data_dir: Path, config: ExperimentConfig) -> dict:
        """Two-phase training: supervised pre-train → RL fine-tune."""
        exp_dir = data_dir / "models" / "experiments" / config.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        config.save(exp_dir / "experiment.yaml")

        # Phase 1: supervised pre-training (direction classification)
        logger.info("=" * 50)
        logger.info("Phase 1: Supervised pre-training (direction prediction)")
        logger.info("=" * 50)
        encoder = pretrain(
            data_dir,
            output_dir=exp_dir,
            train_start=config.train_start,
            train_end=config.train_end,
            val_start=config.val_start,
            val_end=config.val_end,
            n_epochs=20,
            max_samples=500_000,
        )

        # Phase 2: RL fine-tuning
        logger.info("=" * 50)
        logger.info("Phase 2: RL fine-tuning (position control)")
        logger.info("=" * 50)
        model = PositionAgent(n_market_features=14)
        model.load_pretrained_encoder(encoder)

        model = train_rl(
            data_dir,
            output_dir=exp_dir,
            model=model,
            train_start=config.train_start,
            train_end=config.train_end,
            val_start=config.val_start,
            val_end=config.val_end,
        )

        torch.save(model.state_dict(), exp_dir / "model.pt")

        agent_cfg = {
            "n_market_features": 14,
            "seq_length": 60,
            "d_model": 128,
            "nhead": 4,
            "n_layers": 4,
            "dim_feed": 512,
            "dropout": 0.1,
        }
        with open(exp_dir / "agent_config.json", "w") as f:
            json.dump(agent_cfg, f, indent=2)

        metrics = evaluate(model, data_dir, config.val_start, config.val_end)
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        ExperimentRegistry(data_dir).register(config)
        logger.info("RL Agent experiment '%s' complete.", config.name)
        return metrics

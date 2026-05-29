from abc import ABC, abstractmethod
from pathlib import Path

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.variant_registry import VariantRegistry


class NNBasePipeline(ABC):
    mode: str = ""

    @abstractmethod
    def run(self, data_dir: Path, config: ExperimentConfig) -> dict: ...


_REGISTRY = VariantRegistry[NNBasePipeline]("NN")
VARIANTS = _REGISTRY.as_dict()
register = _REGISTRY.register


# Import concrete variant submodules to populate VARIANTS via @register
from alpha_quat.model.nn.transformer.variants import sr_transformer  # noqa: F401, E402
from alpha_quat.model.nn.keltner.variants import keltner_regime  # noqa: F401, E402
from alpha_quat.model.nn.rl_agent.variants import rl_agent_variant  # noqa: F401, E402

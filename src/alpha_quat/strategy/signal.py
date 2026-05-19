from abc import ABC, abstractmethod

import pandas as pd

from alpha_quat.strategy.types import StrategyContext, SignalResult


class ISignalGenerator(ABC):
    @abstractmethod
    def generate(
        self, features: pd.DataFrame, ctx: StrategyContext
    ) -> SignalResult: ...

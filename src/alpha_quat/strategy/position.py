from abc import ABC, abstractmethod

import pandas as pd

from alpha_quat.strategy.types import StrategyContext, SignalResult


class IPositionManager(ABC):
    @abstractmethod
    def allocate(self, signals: SignalResult, ctx: StrategyContext) -> pd.DataFrame: ...

    @abstractmethod
    def constrain(
        self, positions: pd.DataFrame, ctx: StrategyContext
    ) -> pd.DataFrame: ...

    @abstractmethod
    def execute(
        self, target: pd.DataFrame, prev: pd.DataFrame | None, ctx: StrategyContext
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

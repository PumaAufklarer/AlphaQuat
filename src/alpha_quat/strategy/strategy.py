import pandas as pd

from alpha_quat.strategy.types import StrategyContext, StrategyResult
from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.position import IPositionManager


class Strategy:
    def __init__(self, signal: ISignalGenerator, position: IPositionManager):
        self.signal = signal
        self.position = position

    def run(self, features: pd.DataFrame, ctx: StrategyContext) -> StrategyResult:
        sig = self.signal.generate(features, ctx)
        pos = self.position.allocate(sig, ctx)
        pos = self.position.constrain(pos, ctx)
        pos, orders = self.position.execute(pos, ctx.prev_holdings, ctx)
        return StrategyResult(
            target_positions=pos,
            orders=orders,
            metadata={"signal": sig.metadata},
        )

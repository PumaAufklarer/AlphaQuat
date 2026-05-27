import numpy as np
import pandas as pd

from alpha_quat.strategy.position import IPositionManager
from alpha_quat.strategy.types import StrategyContext, SignalResult


class VolParityPositionManager(IPositionManager):
    """Inverse-variance position sizing: weight ∝ 1/σ² per buy candidate.

    Uses rolling close history (passed via StrategyContext.close_history) to
    compute 20-day annualized return volatility. Caps individual position at
    max_weight and renormalizes.
    """

    def __init__(
        self,
        top_k: int = 5,
        vol_window: int = 20,
        max_weight: float = 0.25,
        min_vol: float = 0.10,
    ):
        self.top_k = top_k
        self.vol_window = vol_window
        self.max_weight = max_weight
        self.min_vol = min_vol

    def allocate(self, signals: SignalResult, ctx: StrategyContext) -> pd.DataFrame:
        df = signals.signals
        buys = df.loc[df["action"] == "buy"].copy()
        if buys.empty:
            return pd.DataFrame(columns=["ts_code", "target_weight"])

        buys = buys.sort_values("score", ascending=False).head(self.top_k)

        close_history: dict[str, list[float]] = (
            ctx.close_history if ctx.close_history is not None else {}
        )

        vols = {}
        for code in buys["ts_code"]:
            closes = np.array(close_history.get(code, []), dtype=float)
            if len(closes) >= 6:
                n = min(len(closes) - 1, self.vol_window)
                rets = np.diff(closes[-n - 1 :]) / closes[-n - 1 : -1]
                vol = float(np.std(rets, ddof=1) * np.sqrt(252))
            else:
                vol = self.min_vol * 2.0
            vols[code] = max(vol, self.min_vol)

        inv_var = np.array([1.0 / vols[c] ** 2 for c in buys["ts_code"]])
        raw_weights = inv_var / inv_var.sum()
        weights = np.minimum(raw_weights, self.max_weight)
        weights = weights / weights.sum()

        return pd.DataFrame(
            {"ts_code": buys["ts_code"].values, "target_weight": weights}
        )

    def constrain(self, positions: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        if positions.empty:
            return positions
        result = positions.copy()
        if ctx.universe is not None:
            result = result.loc[result["ts_code"].isin(ctx.universe)]
        if not result.empty:
            total = result["target_weight"].sum()
            if total > 0:
                result["target_weight"] = result["target_weight"] / total
        return result.reset_index(drop=True)

    def execute(
        self,
        target: pd.DataFrame,
        prev: pd.DataFrame | None,
        ctx: StrategyContext,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Engine handles execution directly — this is unused but required by interface.
        return target, pd.DataFrame()

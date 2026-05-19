import pandas as pd
from alpha_quat.strategy.types import SignalResult
from alpha_quat.strategy.signal import ISignalGenerator


class MACrossSignal(ISignalGenerator):
    def __init__(self, short_factor="KLEN35", long_factor="KLEN36"):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self._prev: pd.DataFrame | None = None

    def generate(self, features, ctx):
        cur = features[["ts_code", self.short_factor, self.long_factor]].copy()
        cur["cross_above"] = cur[self.short_factor] > cur[self.long_factor]

        buy_codes = []
        sell_codes = []
        if self._prev is not None and not self._prev.empty:
            merged = cur.merge(
                self._prev[["ts_code", "cross_above"]],
                on="ts_code",
                how="inner",
                suffixes=("", "_prev"),
            )
            golden = merged.loc[
                merged["cross_above"] & ~merged["cross_above_prev"], "ts_code"
            ]
            dead = merged.loc[
                ~merged["cross_above"] & merged["cross_above_prev"], "ts_code"
            ]
            buy_codes = golden.tolist()
            sell_codes = dead.tolist()

        self._prev = cur[["ts_code", "cross_above"]]

        rows = []
        for c in buy_codes:
            rows.append({"ts_code": c, "action": "buy", "score": 1.0})
        for c in sell_codes:
            rows.append({"ts_code": c, "action": "sell", "score": 0.0})

        if rows:
            signals_df = pd.DataFrame(rows, columns=["ts_code", "action", "score"])
        else:
            signals_df = pd.DataFrame(columns=["ts_code", "action", "score"])

        return SignalResult(
            signals=signals_df,
            metadata={
                "signal_name": "ma_cross",
                "short_factor": self.short_factor,
                "long_factor": self.long_factor,
            },
        )

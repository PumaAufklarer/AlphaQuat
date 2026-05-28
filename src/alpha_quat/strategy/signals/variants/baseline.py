from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.strategy.types import SignalResult, StrategyContext

_ZERO_GAIN = {
    "KMID94",
    "KMID95",
    "KMID96",
    "KLEN94",
    "KLEN95",
    "KLEN96",
    "KMID97",
    "KLEN97",
    "KMID98",
    "KLEN98",
    "KMID99",
    "KLEN99",
    "KMID100",
    "KLEN100",
    "KMID101",
}

_IND_FACTORS = ["PE_TTM", "PB", "MV", "TURN", "ROE"]
_INDUSTRY_MAP: dict[str, str] | None = None
_HOLDER_CACHE: dict | None = (
    None  # {ts_code: [(ann_date_int, holder_num, holder_num_prev), ...]}
)


def _load_industry_map(data_dir: Path) -> dict[str, str]:
    global _INDUSTRY_MAP
    if _INDUSTRY_MAP is None:
        sb = pd.read_parquet(data_dir / "stock_basic.parquet")
        _INDUSTRY_MAP = dict(zip(sb["ts_code"], sb["industry"]))
    return _INDUSTRY_MAP


class BaseMLSignal(ABC):
    mode: str
    _WEIGHTS = {"5d": 0.35, "20d": 0.32, "60d": 0.33}

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.models = self._load_models(self.model_dir)

    @abstractmethod
    def _load_models(self, model_dir: Path) -> dict: ...

    @abstractmethod
    def generate(
        self, features: pd.DataFrame, ctx: StrategyContext
    ) -> SignalResult: ...

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        factor_cols = [
            c
            for c in features.columns
            if c not in ("ts_code", "trade_date") and c not in _ZERO_GAIN
        ]
        result = features[factor_cols].copy()

        # Industry-relative ratios — must match training preprocessing.
        data_dir = self.model_dir.parent.parent.parent
        imap = _load_industry_map(data_dir)
        ind_series = (
            result.index.to_series()
            .map(lambda i: imap.get(features.loc[i, "ts_code"], "Unknown"))
            .values
            if "ts_code" in features.columns
            else ["Unknown"] * len(result)
        )

        for f in _IND_FACTORS:
            if f in result.columns:
                ind_median = (
                    pd.DataFrame(
                        {
                            "val": result[f].values,
                            "ind": ind_series,
                            "td": features["trade_date"].values
                            if "trade_date" in features.columns
                            else ["20240101"] * len(result),
                        }
                    )
                    .groupby(["td", "ind"])["val"]
                    .transform("median")
                    .values
                )
                result[f"{f}_ind"] = result[f].values / (ind_median + 1e-8)

        # Holder number features — per-stock latest quarter (ann_date ≤ trade_date).
        global _HOLDER_CACHE
        holder_dir = data_dir / "holdernumber"
        if holder_dir.exists():
            if _HOLDER_CACHE is None:
                lookup: dict[str, list[tuple[int, float, float]]] = {}
                for hf in holder_dir.glob("*.parquet"):
                    code = hf.stem
                    hdf = pd.read_parquet(hf, columns=["ann_date", "holder_num"])
                    hdf = hdf.sort_values("ann_date")
                    entries = []
                    prev = float("nan")
                    for _, row in hdf.iterrows():
                        hn = float(row["holder_num"])
                        entries.append((int(row["ann_date"]), hn, prev))
                        prev = hn
                    if entries:
                        lookup[code] = entries
                _HOLDER_CACHE = lookup

            holder_lookup = _HOLDER_CACHE
            td_col = (
                features["trade_date"]
                if "trade_date" in features.columns
                else pd.Series(["20240101"] * len(result))
            )
            codes = features["ts_code"].values
            td_ints = td_col.astype(int).values
            hnums = np.full(len(result), np.nan)
            hnums_prev = np.full(len(result), np.nan)

            for i in range(len(result)):
                entries = holder_lookup.get(str(codes[i]))
                if entries:
                    td = td_ints[i]
                    best = None
                    for ann, hn, hp in entries:
                        if ann <= td:
                            best = (hn, hp)
                        else:
                            break
                    if best:
                        hnums[i] = best[0]
                        hnums_prev[i] = best[1]

            result["holder_num"] = hnums
            result["holder_num_qoq"] = np.where(
                ~np.isnan(hnums_prev) & (hnums_prev != 0),
                (hnums - hnums_prev) / hnums_prev,
                0.0,
            )
            result["holder_num_qoq"] = np.clip(result["holder_num_qoq"], -0.5, 0.5)

        # Cross-sectional rank within each trade_date.
        if "trade_date" in features.columns:
            result = result.groupby(features["trade_date"], group_keys=False).rank(
                pct=True
            )

        result = result.fillna(0)
        assert isinstance(result, pd.DataFrame)
        return result

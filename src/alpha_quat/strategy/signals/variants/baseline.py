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

    def _prepare_features(
        self, features: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        # Industry momentum: median backward return per industry+date.
        if "ret_back_5" in result.columns:
            for w in [5, 20]:
                col = f"ret_back_{w}"
                result[f"ind_mom_{w}d"] = (
                    pd.DataFrame({"v": result[col].values, "ind": ind_series})
                    .groupby("ind")["v"]
                    .transform("median")
                    .values
                )
            result.drop(
                columns=["ret_back_5", "ret_back_20"], errors="ignore", inplace=True
            )

        # Cross-sectional rank within each trade_date.
        if "trade_date" in features.columns:
            result = result.groupby(features["trade_date"], group_keys=False).rank(
                pct=True
            )

        result = result.fillna(0)

        # --- Risk-free rate baseline: 0.5 on all features, model ranks it ---
        rfr_row = {c: 0.5 for c in result.columns}
        rfr_df = pd.DataFrame([rfr_row])
        result = pd.concat([result, rfr_df], ignore_index=True)
        # Extend features DataFrame's ts_code for the signal generator
        if "ts_code" in features.columns:
            features = features.copy()
            features = pd.concat(
                [features, pd.DataFrame({"ts_code": ["__RFR__"]})], ignore_index=True
            )

        assert isinstance(result, pd.DataFrame)
        return result, features

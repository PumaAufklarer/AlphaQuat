from abc import ABC, abstractmethod
from pathlib import Path

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
    "O2C",
    "DRP",
    "HLC",
    "pe_ttm",
    "pb",
    "ROE_RAW",
    "ROE",
    "MV",
    "VOLRATIO",
    "EMA12C",
    "EMA26C",
    "MACD",
    "RSI14",
    "SLOPE5",
    "SLOPE20",
}


class BaseMLSignal(ABC):
    mode: str
    _WEIGHTS = {"5d": 0.35, "20d": 0.32, "60d": 0.33}

    def __init__(self, model_dir: str | Path):
        self.models = self._load_models(Path(model_dir))

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
        result = features[factor_cols].fillna(0)
        assert isinstance(result, pd.DataFrame)
        return result

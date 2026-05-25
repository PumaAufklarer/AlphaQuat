"""MLSignalGenerator — scores stocks using ensemble of LightGBM models.

Detection order: meta → lambdarank → quantile → regression.
"""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.types import SignalResult, StrategyContext

logger = logging.getLogger(__name__)

_WEIGHTS = {"5d": 0.35, "20d": 0.32, "60d": 0.33}
_HORIZONS = ["5d", "20d", "60d"]
_ALPHAS = [0.1, 0.5, 0.9]
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


class MLSignalGenerator(ISignalGenerator):
    def __init__(self, model_dir: Path, top_k: int = 5):
        self.top_k = top_k
        self.model_dir = Path(model_dir)
        self.models: dict[str, lgb.Booster] = {}
        self.extra: dict[tuple[str, float], lgb.Booster] = {}
        self.meta_models: dict[str, lgb.Booster] = {}
        self.quantile_mode = False
        self.meta_mode = False
        self.lambdarank_mode = False

        # 1. Meta mode (stacking: needs quantile base models)
        has_meta = all((model_dir / f"meta_model_{h}.txt").exists() for h in _HORIZONS)
        has_all_quantile = all(
            (model_dir / f"lightgbm_model_{h}_alpha_{a}.txt").exists()
            for h in _HORIZONS
            for a in _ALPHAS
        )
        if has_meta and has_all_quantile:
            logger.info("Meta (stacking) mode detected")
            self.meta_mode = True
            self.quantile_mode = True
            for h in _HORIZONS:
                self.models[h] = lgb.Booster(
                    model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.5.txt")
                )
                for a in _ALPHAS:
                    if a != 0.5:
                        self.extra[(h, a)] = lgb.Booster(
                            model_file=str(
                                model_dir / f"lightgbm_model_{h}_alpha_{a}.txt"
                            )
                        )
                self.meta_models[h] = lgb.Booster(
                    model_file=str(model_dir / f"meta_model_{h}.txt")
                )
            logger.info("Loaded 9 quantile + 3 meta models")

        # 2. Lambdarank mode
        if not self.meta_mode:
            has_lr = all(
                (model_dir / f"lightgbm_model_{h}_lambdarank.txt").exists()
                for h in _HORIZONS
            )
            if has_lr:
                logger.info("Lambdarank mode detected")
                self.lambdarank_mode = True
                for h in _HORIZONS:
                    self.models[h] = lgb.Booster(
                        model_file=str(model_dir / f"lightgbm_model_{h}_lambdarank.txt")
                    )
                logger.info("Loaded 3 lambdarank models")

        # 3. Quantile mode
        if not self.meta_mode and not self.lambdarank_mode:
            has_q = all(
                (model_dir / f"lightgbm_model_{h}_alpha_0.5.txt").exists()
                for h in _HORIZONS
            )
            if has_q:
                logger.info("Quantile mode")
                self.quantile_mode = True
                for h in _HORIZONS:
                    self.models[h] = lgb.Booster(
                        model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.5.txt")
                    )
                    p10 = model_dir / f"lightgbm_model_{h}_alpha_0.1.txt"
                    p90 = model_dir / f"lightgbm_model_{h}_alpha_0.9.txt"
                    if p10.exists() and p90.exists():
                        self.extra[(h, 0.1)] = lgb.Booster(model_file=str(p10))
                        self.extra[(h, 0.9)] = lgb.Booster(model_file=str(p90))

        # 4. Regression mode
        if not self.models:
            for h in _HORIZONS:
                path = model_dir / f"lightgbm_model_{h}.txt"
                if path.exists():
                    self.models[h] = lgb.Booster(model_file=str(path))
            if self.models:
                logger.info("Regression mode")

        if not self.models:
            raise FileNotFoundError(f"No models found in {model_dir}")

    def _predict_all(self, X: pd.DataFrame) -> dict[str, dict[float, np.ndarray]]:
        preds: dict[str, dict[float, np.ndarray]] = {}
        for h in _HORIZONS:
            preds[h] = {}
            if h in self.models:
                preds[h][0.5] = np.asarray(self.models[h].predict(X), dtype=float)
            for a in _ALPHAS:
                if (h, a) in self.extra:
                    preds[h][a] = np.asarray(self.extra[(h, a)].predict(X), dtype=float)
        return preds

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        factor_cols = [
            c
            for c in features.columns
            if c not in ("ts_code", "trade_date") and c not in _ZERO_GAIN
        ]
        X = features[factor_cols].fillna(0)

        n = len(X)
        score = np.zeros(n)
        meta: dict = {"model": "ml_ensemble"}

        if self.meta_mode:
            base = self._predict_all(X)
            extra_low = np.zeros(n)
            extra_high = np.zeros(n)
            for h in _HORIZONS:
                feats = np.column_stack(
                    [base[hor][a] for hor in _HORIZONS for a in _ALPHAS]
                )
                h_score = np.asarray(self.meta_models[h].predict(feats), dtype=float)
                score += _WEIGHTS.get(h, 0) * h_score
                if 0.1 in base[h] and 0.9 in base[h]:
                    extra_low += _WEIGHTS.get(h, 0) * base[h][0.1]
                    extra_high += _WEIGHTS.get(h, 0) * base[h][0.9]
            ci = (extra_high - extra_low) / 2
            meta["ci_width"] = ci
            meta["confidence"] = 1.0 - ci / max(np.max(ci), 1e-8)
            meta["model"] = "meta_stacking"

        elif self.lambdarank_mode:
            # Lambdarank: predict ranking scores, weight by ICIR weights
            for h, model in self.models.items():
                pred = np.asarray(model.predict(X), dtype=float)
                score += _WEIGHTS.get(h, 0) * pred

        elif self.quantile_mode:
            extra_low = np.zeros(n)
            extra_high = np.zeros(n)
            has_extra = bool(self.extra)
            for h, model in self.models.items():
                w = _WEIGHTS.get(h, 0)
                pred = np.asarray(model.predict(X), dtype=float)
                score += w * pred
                if has_extra and (h, 0.1) in self.extra:
                    low = np.asarray(self.extra[(h, 0.1)].predict(X), dtype=float)
                    hi = np.asarray(self.extra[(h, 0.9)].predict(X), dtype=float)
                    extra_low += w * low
                    extra_high += w * hi
            ci = (extra_high - extra_low) / 2
            meta["ci_width"] = ci
            meta["confidence"] = 1.0 - ci / max(np.max(ci), 1e-8)

        else:
            # Regression mode
            for h, model in self.models.items():
                pred = np.asarray(model.predict(X), dtype=float)
                score += _WEIGHTS.get(h, 0) * pred

        df = features[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"

        return SignalResult(signals=df, metadata=meta)

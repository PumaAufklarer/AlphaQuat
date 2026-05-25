"""MLSignalGenerator — scores stocks using ensemble of LightGBM models.

Supports regression, quantile, and meta (stacking) modes.
Meta mode loads 9 base quantile models + meta model for final scoring.
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


class MLSignalGenerator(ISignalGenerator):
    """Scores universe using ensemble of LightGBM models."""

    def __init__(self, model_dir: Path, top_k: int = 5):
        self.top_k = top_k
        self.model_dir = Path(model_dir)
        self.models: dict[str, lgb.Booster] = {}
        self.extra: dict[tuple[str, float], lgb.Booster] = {}
        self.meta_models: dict[str, lgb.Booster] = {}
        self.quantile_mode = False
        self.meta_mode = False

        # Detect meta mode
        if all((model_dir / f"meta_model_{h}.txt").exists() for h in _HORIZONS):
            logger.info("Meta (stacking) mode detected")
            self.meta_mode = True
            # Need base quantile models for meta predictions
            has_all = all(
                (model_dir / f"lightgbm_model_{h}_alpha_{a}.txt").exists()
                for h in _HORIZONS
                for a in _ALPHAS
            )
            if has_all:
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
                for h in _HORIZONS:
                    self.meta_models[h] = lgb.Booster(
                        model_file=str(model_dir / f"meta_model_{h}.txt")
                    )
                logger.info("Loaded 9 base quantile models + 3 meta models")
            else:
                logger.warning("Meta mode requires quantile models, falling back")
                self.meta_mode = False

        if not self.meta_mode:
            # Detect quantile mode
            has_q = all(
                (model_dir / f"lightgbm_model_{h}_alpha_0.5.txt").exists()
                for h in _HORIZONS
            )
            has_all = has_q and all(
                (model_dir / f"lightgbm_model_{h}_alpha_{a}.txt").exists()
                for h in _HORIZONS
                for a in [0.1, 0.9]
            )
            if has_all:
                logger.info("Quantile mode: loading median + 10%/90% models")
                self.quantile_mode = True
                for h in _HORIZONS:
                    self.models[h] = lgb.Booster(
                        model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.5.txt")
                    )
                    self.extra[(h, 0.1)] = lgb.Booster(
                        model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.1.txt")
                    )
                    self.extra[(h, 0.9)] = lgb.Booster(
                        model_file=str(model_dir / f"lightgbm_model_{h}_alpha_0.9.txt")
                    )
            elif has_q:
                logger.info("Quantile mode (median only)")
                self.quantile_mode = True
                for h in _HORIZONS:
                    path = model_dir / f"lightgbm_model_{h}_alpha_0.5.txt"
                    if path.exists():
                        self.models[h] = lgb.Booster(model_file=str(path))
            else:
                logger.info("Regression mode")
                for h in _HORIZONS:
                    path = model_dir / f"lightgbm_model_{h}.txt"
                    if path.exists():
                        self.models[h] = lgb.Booster(model_file=str(path))

        if not self.models:
            raise FileNotFoundError(f"No models found in {model_dir}")

    def _predict_all(self, X: pd.DataFrame) -> dict[str, dict[float, np.ndarray]]:
        """Predict with all 9 base quantile models."""
        preds: dict[str, dict[float, np.ndarray]] = {}
        for h in _HORIZONS:
            preds[h] = {}
        for h, model in self.models.items():
            preds[h][0.5] = np.asarray(model.predict(X), dtype=float)
        for (h, a), model in self.extra.items():
            preds.setdefault(h, {})[a] = np.asarray(model.predict(X), dtype=float)
        return preds

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        factor_cols = [
            c for c in features.columns if c not in ("ts_code", "trade_date")
        ]
        X = features[factor_cols].fillna(0)

        if self.meta_mode:
            # Meta mode: predict 9 base → meta model predicts final score
            base_preds = self._predict_all(X)
            n = len(X)
            score = np.zeros(n)
            extra_low = np.zeros(n)
            extra_high = np.zeros(n)

            for h in _HORIZONS:
                meta_feats = np.column_stack(
                    [base_preds[hor][a] for hor in _HORIZONS for a in _ALPHAS]
                )
                h_score = np.asarray(
                    self.meta_models[h].predict(meta_feats), dtype=float
                )
                w = _WEIGHTS.get(h, 0)
                score += w * h_score

                # Also collect quantile extremes for CI
                if 0.1 in base_preds[h] and 0.9 in base_preds[h]:
                    extra_low += w * base_preds[h][0.1]
                    extra_high += w * base_preds[h][0.9]

            df = features[["ts_code"]].copy()
            df["score"] = score
            df["action"] = "buy"

            meta: dict = {"model": "meta_stacking"}
            ci = (extra_high - extra_low) / 2
            meta["ci_width"] = ci
            max_ci = max(np.max(ci), 1e-8)
            meta["confidence"] = 1.0 - ci / max_ci

            return SignalResult(signals=df, metadata=meta)

        # Original quantile/regression path
        n = len(X)
        score = np.zeros(n)
        extra_low = np.zeros(n)
        extra_high = np.zeros(n)
        has_extra = False

        for label, model in self.models.items():
            w = _WEIGHTS.get(label, 0)
            pred = np.asarray(model.predict(X), dtype=float)
            score += w * pred
            if self.quantile_mode and (label, 0.1) in self.extra:
                has_extra = True
                low = np.asarray(self.extra[(label, 0.1)].predict(X), dtype=float)
                hi = np.asarray(self.extra[(label, 0.9)].predict(X), dtype=float)
                extra_low += w * low
                extra_high += w * hi

        df = features[["ts_code"]].copy()
        df["score"] = score
        df["action"] = "buy"

        meta: dict = {"model": "ml_ensemble"}
        if has_extra:
            meta["score_low"] = extra_low
            meta["score_high"] = extra_high
            ci = (extra_high - extra_low) / 2
            meta["ci_width"] = ci
            max_ci = max(np.max(ci), 1e-8)
            meta["confidence"] = 1.0 - ci / max_ci

        return SignalResult(signals=df, metadata=meta)

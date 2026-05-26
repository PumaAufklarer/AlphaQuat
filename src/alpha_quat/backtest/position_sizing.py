from collections.abc import Callable

import numpy as np
import pandas as pd

STRATEGIES: dict[str, Callable] = {}


def register(name):
    def decorator(fn):
        STRATEGIES[name] = fn
        return fn

    return decorator


@register("equal")
def equal_weight(all_scores: np.ndarray) -> np.ndarray:
    return np.ones_like(all_scores)


@register("kelly")
def kelly_adjust(
    all_scores: np.ndarray,
    score_history: dict[str, list[float]],
    codes: list[str],
) -> np.ndarray:
    adjusted = all_scores.copy()
    for i, code in enumerate(codes):
        hist = score_history.get(code, [all_scores[i]])
        if len(hist) >= 3:
            mean_s = float(np.mean(hist))
            var_s = max(float(np.var(hist, ddof=1)) + 1e-8, 1e-8)
            f = mean_s / var_s
            adjusted[i] = all_scores[i] * min(f, 3.0)
    return adjusted


@register("vol_parity")
def vol_parity_adjust(
    all_scores: np.ndarray,
    features: pd.DataFrame,
    vol_col: str | None = None,
) -> np.ndarray:
    if vol_col is None:
        vol_col = next((c for c in features.columns if "KLEN38" in c), None)
    if vol_col:
        vol = np.asarray(features[vol_col].fillna(0), dtype=float) + 1e-8
        adjusted = all_scores / vol
        return adjusted
    return all_scores.copy()


@register("score_momentum")
def score_momentum_adjust(
    all_scores: np.ndarray,
    score_history: dict[str, list[float]],
    codes: list[str],
) -> np.ndarray:
    adjusted = all_scores.copy()
    for i, code in enumerate(codes):
        hist = score_history.get(code, [])
        bonus = min(len(hist) / 5, 1.0)
        adjusted[i] = all_scores[i] * (1 + bonus)
    return adjusted

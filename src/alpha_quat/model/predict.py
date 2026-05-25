"""Predict stocks with ensemble of 5d/20d/60d LightGBM models.

Supports both point-estimate (regression) and quantile regression models.
When quantile models are available, displays 80% confidence intervals.
"""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.backtest.filters import _date_to_path

logger = logging.getLogger(__name__)

_WEIGHTS = {"5d": 0.35, "20d": 0.32, "60d": 0.33}
_ALPHAS = [0.1, 0.5, 0.9]


def predict(data_dir: Path, holdings: list[dict] | None = None, top_k: int = 10):
    """Score all stocks and print recommendations."""
    model_dir = data_dir / "models"

    # Detect mode: quantile models or regression mode
    quantile_mode = all(
        (model_dir / f"lightgbm_model_{h}_alpha_0.1.txt").exists()
        for h in ["5d", "20d", "60d"]
    )

    if quantile_mode:
        logger.info("Quantile regression mode (10%/50%/90%)")
        models = {}
        for h in ["5d", "20d", "60d"]:
            models[h] = {}
            for a in _ALPHAS:
                path = model_dir / f"lightgbm_model_{h}_alpha_{a}.txt"
                if path.exists():
                    models[h][a] = lgb.Booster(model_file=str(path))
                    logger.info("Loaded %s alpha=%.1f", h, a)

        # Check we have all required models
        if not all(a in models[h] for h in ["5d", "20d", "60d"] for a in _ALPHAS):
            logger.warning("Incomplete quantile models, falling back to regression")
            quantile_mode = False

    if not quantile_mode:
        models = {}
        for h in ["5d", "20d", "60d"]:
            path = model_dir / f"lightgbm_model_{h}.txt"
            if path.exists():
                models[h] = lgb.Booster(model_file=str(path))
                logger.info("Loaded %s", h)

    if not models:
        raise FileNotFoundError(f"No models found in {model_dir}")

    # Find latest feature date
    feat_dir = data_dir / "features"
    feat_files = sorted(feat_dir.glob("*.parquet"))
    if not feat_files:
        raise FileNotFoundError("No feature files found")
    latest = feat_files[-1].stem
    feat_path = feat_dir / f"{latest}.parquet"
    logger.info("Using features from %s", latest)

    features = pd.read_parquet(feat_path)

    # Filter universe
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    main_board = set(sb.loc[sb["market"] == "主板", "ts_code"])
    features = features.loc[features["ts_code"].isin(list(main_board))]

    st_path = data_dir / "stock_st" / f"{_date_to_path(latest)}.parquet"
    if st_path.exists():
        st_codes = set(pd.read_parquet(st_path)["ts_code"])
        features = features.loc[~features["ts_code"].isin(list(st_codes))]

    factor_cols = [c for c in features.columns if c not in ("ts_code", "trade_date")]
    X = features[factor_cols].fillna(0)

    if quantile_mode:
        # Predict with quantile models
        preds = {}
        for h in ["5d", "20d", "60d"]:
            preds[h] = {
                a: np.asarray(models[h][a].predict(X), dtype=float) for a in _ALPHAS
            }

        features["score"] = sum(
            _WEIGHTS[h] * preds[h][0.5] for h in ["5d", "20d", "60d"]
        )
        features["s5"] = preds["5d"][0.5]
        features["s20"] = preds["20d"][0.5]
        features["s60"] = preds["60d"][0.5]
        features["ci5"] = abs(preds["5d"][0.9] - preds["5d"][0.1]) / 2
        features["ci20"] = abs(preds["20d"][0.9] - preds["20d"][0.1]) / 2
        features["ci60"] = abs(preds["60d"][0.9] - preds["60d"][0.1]) / 2
    else:
        # Predict with point models
        preds = {}
        for h in ["5d", "20d", "60d"]:
            preds[h] = np.asarray(models[h].predict(X), dtype=float)

        features["score"] = sum(_WEIGHTS[h] * preds[h] for h in ["5d", "20d", "60d"])
        features["s5"] = preds["5d"]
        features["s20"] = preds["20d"]
        features["s60"] = preds["60d"]

    # Sort
    features = features.sort_values("score", ascending=False).reset_index(drop=True)

    # Stock name mapping
    name_map = dict(zip(sb["ts_code"], sb.get("name", sb["ts_code"])))
    features["name"] = features["ts_code"].map(name_map)

    print()
    print(f"===== PREDICT: {latest} =====")
    print()

    # Top-K
    has_ci = quantile_mode
    header = (
        f"{'':>3} {'代码':>10} {'名称':>8}  {'5d':>6} {'20d':>6} {'60d':>6} {'综合':>6}"
    )
    if has_ci:
        header += f"  {'CI':>6}"
    print(header)
    print("-" * (54 if not has_ci else 63))
    for i in range(min(top_k, len(features))):
        r = features.iloc[i]
        line = (
            f"  {i + 1:>2} {r['ts_code']:>10} {str(r.get('name', ''))[:8]:>8}  "
            f"{r['s5']:.3f} {r['s20']:.3f} {r['s60']:.3f} {r['score']:.3f}"
        )
        if has_ci:
            ci = (r.get("ci5", 0) + r.get("ci20", 0) + r.get("ci60", 0)) / 3
            line += f"  {ci:.3f}"
        print(line)

    # Bottom-K
    print()
    print(f"=== BOTTOM {top_k} ===")
    print(header)
    print("-" * (54 if not has_ci else 63))
    n = len(features)
    for i in range(min(top_k, n)):
        r = features.iloc[n - 1 - i]
        line = (
            f"  {i + 1:>2} {r['ts_code']:>10} {str(r.get('name', ''))[:8]:>8}  "
            f"{r['s5']:.3f} {r['s20']:.3f} {r['s60']:.3f} {r['score']:.3f}"
        )
        if has_ci:
            ci = (r.get("ci5", 0) + r.get("ci20", 0) + r.get("ci60", 0)) / 3
            line += f"  {ci:.3f}"
        print(line)

    # Current holdings
    if holdings:
        print()
        print("=== 当前持仓 ===")
        ch = f"{'代码':>10} {'名称':>8}   {'评分':>6} {'排名':>6}"
        ci_header = f" {'CI':>6}" if has_ci else f""
        ch += ci_header + f"   {'持仓':>6} {'成本':>8}"
        print(ch)
        print("-" * (68 if has_ci else 60))
        code_to_idx = dict(zip(features["ts_code"], features.index))
        for h in holdings:
            code = h.get("ts_code", "")
            idx = code_to_idx.get(code)
            if idx is not None:
                r = features.iloc[idx]
                shares = h.get("shares", 0)
                cost = h.get("avg_cost", 0)
                line = (
                    f"  {code:>10} {str(r.get('name', ''))[:8]:>8}  "
                    f"{r['score']:.3f} {idx + 1:>6}"
                )
                if has_ci:
                    ci = (r.get("ci5", 0) + r.get("ci20", 0) + r.get("ci60", 0)) / 3
                    line += f" {ci:.3f}"
                line += f"   {shares:>6} {cost:>8.2f}"
                print(line)
            else:
                print(f"  {code:>10} {'(不在评分范围内)':<20}")

    # Stats
    scores = features["score"]
    print()
    print("=== 当日统计 ===")
    print(f"  {'分位回归' if quantile_mode else '点估计'}模式")
    print(f"  评分股票数: {len(features)}")
    print(f"  综合评分均值: {scores.mean():.4f}")
    print(f"  综合评分中位: {scores.median():.4f}")
    print(f"  综合评分范围: {scores.min():.4f} ~ {scores.max():.4f}")
    if has_ci:
        all_cis = []
        for h in ["5d", "20d", "60d"]:
            ci = abs(preds[h][0.9] - preds[h][0.1]) / 2
            all_cis.append(np.mean(ci))
        print(f"  平均置信区间半宽: {np.mean(all_cis):.4f}")

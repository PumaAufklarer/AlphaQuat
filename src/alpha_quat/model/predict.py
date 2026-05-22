"""Predict stocks with ensemble of 5d/20d/60d LightGBM models."""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.backtest.filters import _date_to_path

logger = logging.getLogger(__name__)

_WEIGHTS = {"ret_5d": 0.35, "ret_20d": 0.32, "ret_60d": 0.33}


def predict(data_dir: Path, holdings: list[dict] | None = None, top_k: int = 10):
    """Score all stocks and print recommendations."""
    model_dir = data_dir / "models"

    models = {}
    for label in ["ret_5d", "ret_20d", "ret_60d"]:
        suffix = label.replace("ret_", "")
        path = model_dir / f"lightgbm_model_{suffix}.txt"
        if path.exists():
            models[label] = lgb.Booster(model_file=str(path))
            logger.info("Loaded %s", label)

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

    # Predict
    preds = {}
    for label, model in models.items():
        preds[label] = np.asarray(model.predict(X), dtype=float)

    features["score"] = sum(_WEIGHTS[l] * preds[l] for l in models)
    features["s5"] = preds["ret_5d"]
    features["s20"] = preds["ret_20d"]
    features["s60"] = preds["ret_60d"]

    # Sort
    features = features.sort_values("score", ascending=False).reset_index(drop=True)

    # Stock name mapping
    name_map = dict(zip(sb["ts_code"], sb.get("name", sb["ts_code"])))
    features["name"] = features["ts_code"].map(name_map)

    print()
    print(f"===== PREDICT: {latest} =====")
    print()

    # Top-K
    print(f"=== TOP {top_k} ===")
    print(
        f"{'':>3} {'代码':>10} {'名称':>8}  {'5d':>6} {'20d':>6} {'60d':>6} {'综合':>6}"
    )
    print("-" * 54)
    for i in range(min(top_k, len(features))):
        r = features.iloc[i]
        print(
            f"  {i + 1:>2} {r['ts_code']:>10} {str(r.get('name', ''))[:8]:>8}  "
            f"{r['s5']:.3f} {r['s20']:.3f} {r['s60']:.3f} {r['score']:.3f}"
        )

    # Bottom-K
    print()
    print(f"=== BOTTOM {top_k} ===")
    print(
        f"{'':>3} {'代码':>10} {'名称':>8}  {'5d':>6} {'20d':>6} {'60d':>6} {'综合':>6}"
    )
    print("-" * 54)
    n = len(features)
    for i in range(min(top_k, n)):
        r = features.iloc[n - 1 - i]
        print(
            f"  {i + 1:>2} {r['ts_code']:>10} {str(r.get('name', ''))[:8]:>8}  "
            f"{r['s5']:.3f} {r['s20']:.3f} {r['s60']:.3f} {r['score']:.3f}"
        )

    # Current holdings
    if holdings:
        print()
        print("=== 当前持仓 ===")
        print(
            f"{'代码':>10} {'名称':>8}   {'评分':>6} {'排名':>6} {'5d':>6} {'20d':>6} {'60d':>6}   {'持仓':>6} {'成本':>8}"
        )
        print("-" * 75)
        code_to_idx = dict(zip(features["ts_code"], features.index))
        for h in holdings:
            code = h.get("ts_code", "")
            idx = code_to_idx.get(code)
            if idx is not None:
                r = features.iloc[idx]
                shares = h.get("shares", 0)
                cost = h.get("avg_cost", 0)
                print(
                    f"  {code:>10} {str(r.get('name', ''))[:8]:>8}  "
                    f"{r['score']:.3f} {idx + 1:>6} {r['s5']:.3f} {r['s20']:.3f} {r['s60']:.3f}  "
                    f"{shares:>6} {cost:>8.2f}"
                )
            else:
                print(f"  {code:>10} {'(不在评分范围内)':<20}")

    # Stats
    scores = features["score"]
    print()
    print("=== 当日统计 ===")
    print(f"  评分股票数: {len(features)}")
    print(f"  综合评分均值: {scores.mean():.4f}")
    print(f"  综合评分中位: {scores.median():.4f}")
    print(f"  综合评分范围: {scores.min():.4f} ~ {scores.max():.4f}")

"""Meta model — Stacking Layer that learns to combine 9 quantile predictions.

Train after the 9 base quantile models. Input is the 9 predictions
(3 horizons × 3 quantiles), output is a single optimal score.
"""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from alpha_quat.model.data import DatasetBuilder

logger = logging.getLogger(__name__)

HORIZONS = ["5d", "20d", "60d"]
ALPHAS = [0.1, 0.5, 0.9]


def _load_base_models(model_dir: Path) -> dict[str, dict[float, lgb.Booster]]:
    """Load all 9 quantile base models."""
    models: dict[str, dict[float, lgb.Booster]] = {}
    for h in HORIZONS:
        models[h] = {}
        for a in ALPHAS:
            path = model_dir / f"lightgbm_model_{h}_alpha_{a}.txt"
            if path.exists():
                models[h][a] = lgb.Booster(model_file=str(path))
    return models


def _predict_all_models(
    models: dict[str, dict[float, lgb.Booster]],
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict with all 9 models and return DataFrame with prediction columns."""
    factor_cols = [c for c in features.columns if c not in ("ts_code", "trade_date")]
    X = features[factor_cols].fillna(0)

    result = features[["ts_code", "trade_date"]].copy()
    for h in HORIZONS:
        for a in ALPHAS:
            if h in models and a in models[h]:
                pred = np.asarray(models[h][a].predict(X), dtype=float)
                result[f"pred_{h}_{a}"] = pred
    return result


def build_meta_features(
    data_dir: Path,
    model_dir: Path,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build meta model training features from base model predictions.

    Returns:
        meta_train_df: features + labels for meta training period
        meta_val_df: features + labels for meta validation period
    """
    models = _load_base_models(model_dir)
    if not models:
        raise FileNotFoundError(f"No base models found in {model_dir}")

    builder = DatasetBuilder(data_dir)

    # Load data for both train and val periods combined
    logger.info("Loading dataset for meta features...")
    data = builder.build(train_start, train_end, val_start, val_end)

    # Combine train + val into one DataFrame with predictions
    all_X = pd.concat([data.X_train, data.X_val], ignore_index=True)
    all_y_5 = pd.concat([data.y_train_5, data.y_val_5], ignore_index=True)
    all_y_20 = pd.concat([data.y_train_20, data.y_val_20], ignore_index=True)
    all_y_60 = pd.concat([data.y_train_60, data.y_val_60], ignore_index=True)
    all_dates = pd.concat([data.train_dates, data.val_dates], ignore_index=True)
    all_codes = pd.concat([data.train_codes, data.val_codes], ignore_index=True)

    # Reconstruct features DataFrame for model prediction
    all_features = all_X.copy()
    all_features["ts_code"] = all_codes
    all_features["trade_date"] = all_dates

    logger.info("Predicting with 9 base models on meta training range...")
    pred_df = _predict_all_models(models, all_features)

    # Add labels
    pred_df["ret_5d"] = all_y_5.values
    pred_df["ret_20d"] = all_y_20.values
    pred_df["ret_60d"] = all_y_60.values

    # Split back into train/val
    train_mask = pred_df["trade_date"].between(train_start, train_end)
    val_mask = pred_df["trade_date"].between(val_start, val_end)

    # Drop NaN in predictions (dates where some model couldn't predict)
    pred_cols = [f"pred_{h}_{a}" for h in HORIZONS for a in ALPHAS]
    pred_df = pred_df.dropna(subset=pred_cols)

    train_df = pred_df[train_mask].reset_index(drop=True)
    val_df = pred_df[val_mask].reset_index(drop=True)

    logger.info(
        "Meta train: %d samples, meta val: %d samples", len(train_df), len(val_df)
    )
    return train_df, val_df


def train_meta_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    horizon: str,
    output_dir: Path,
    tune: bool = True,
    n_trials: int = 30,
):
    """Train a meta LightGBM model for one horizon.

    Input features: 9 predictions from base models.
    Target: actual ret_{horizon} channel position [0, 1].
    """
    pred_cols = [f"pred_{h}_{a}" for h in HORIZONS for a in ALPHAS]
    label_col = f"ret_{horizon}"

    X_train = train_df[pred_cols]
    y_train = train_df[label_col]
    X_val = val_df[pred_cols]
    y_val = val_df[label_col]

    meta_model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.04,
        num_leaves=12,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_gain_to_split=0.001,
        verbose=-1,
        random_state=42,
    )
    meta_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=0)],
    )

    val_pred = meta_model.predict(X_val)
    y_val_arr = np.asarray(y_val, dtype=float)
    mse = float(((y_val_arr - val_pred) ** 2).mean())

    logger.info("Meta model %s — val MSE: %.6f", horizon, mse)

    # Save
    path = output_dir / f"meta_model_{horizon}.txt"
    meta_model.booster_.save_model(str(path))
    logger.info("Saved meta model to %s", path)

    # Feature importance
    imp = sorted(zip(pred_cols, meta_model.feature_importances_), key=lambda x: -x[1])
    logger.info("Top 5 meta features for %s:", horizon)
    for name, val in imp[:5]:
        logger.info("  %s: %.1f", name, val)

    return meta_model


def meta_predict(
    base_preds: dict[str, dict[float, np.ndarray]],
    meta_models: dict[str, lgb.LGBMRegressor],
) -> dict[str, np.ndarray]:
    """Predict final scores using meta models on top of base predictions.

    Args:
        base_preds: {horizon: {alpha: predictions_array}}
        meta_models: {horizon: trained_meta_model}

    Returns:
        {horizon: final_score_array}
    """
    n = len(next(iter(next(iter(base_preds.values())).values())))
    results = {}
    for h in HORIZONS:
        if h not in meta_models:
            # Fallback: use median prediction
            results[h] = base_preds[h][0.5]
            continue
        # Build feature matrix: 9 predictions
        X = np.column_stack(
            [
                base_preds[hor][a]
                for hor in HORIZONS
                for a in ALPHAS
                if hor in base_preds and a in base_preds[hor]
            ]
        )
        results[h] = meta_models[h].predict(X)
    return results

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class EvalResult:
    label_name: str
    mse: float
    mae: float
    mean_ic: float
    ic_std: float
    icir: float
    top5_features: list[tuple[str, float]]
    bottom5_features: list[tuple[str, float]]
    best_params: dict
    feature_names: list[str] | None


@dataclass
class RankICResult:
    mean_ic: float
    ic_std: float
    icir: float


class LightGBMEvaluator:
    def compute_rank_ic(
        self, y_pred: np.ndarray, y_true: np.ndarray, dates: np.ndarray
    ) -> RankICResult:
        df = pd.DataFrame({"date": dates, "pred": y_pred, "true": y_true})
        daily_ics = []
        for _, group in df.groupby("date"):
            if len(group) >= 3:
                ic, _ = spearmanr(group["pred"], group["true"])
                daily_ics.append(ic)

        if not daily_ics:
            return RankICResult(mean_ic=0.0, ic_std=0.0, icir=0.0)

        daily_ics = np.array(daily_ics)
        mean_ic = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics, ddof=1))
        icir = mean_ic / ic_std if ic_std > 0 else 0.0

        return RankICResult(mean_ic=mean_ic, ic_std=ic_std, icir=icir)

    def evaluate(
        self,
        model: lgb.Booster,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_dates: pd.Series,
        val_codes: pd.Series,
        best_params: dict,
        feature_names: list[str] | None,
        label_name: str,
    ) -> EvalResult:
        y_pred_raw = model.predict(X_val)
        y_pred = np.asarray(y_pred_raw, dtype=float)
        y_true = np.asarray(y_val, dtype=float)

        mse = float(mean_squared_error(y_val, y_pred))
        mae = float(mean_absolute_error(y_val, y_pred))

        rank_ic = self.compute_rank_ic(y_pred, y_true, val_dates.to_numpy())

        importance = model.feature_importance(importance_type="gain")
        feature_names_list = (
            feature_names if feature_names is not None else X_val.columns.tolist()
        )
        feat_imp = sorted(
            zip(feature_names_list, importance), key=lambda x: x[1], reverse=True
        )

        top5 = feat_imp[:5]
        bottom5 = feat_imp[-5:]

        return EvalResult(
            label_name=label_name,
            mse=mse,
            mae=mae,
            mean_ic=rank_ic.mean_ic,
            ic_std=rank_ic.ic_std,
            icir=rank_ic.icir,
            top5_features=[(str(name), float(val)) for name, val in top5],
            bottom5_features=[(str(name), float(val)) for name, val in bottom5],
            best_params=best_params,
            feature_names=feature_names,
        )

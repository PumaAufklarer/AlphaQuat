import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr

from alpha_quat.model.lightgbm.config import LightGBMConfig

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    def __init__(self, config: LightGBMConfig):
        self.config = config

    def _base_params(self) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "n_estimators": self.config.n_estimators,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "verbose": self.config.verbosity,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
        }

    def _fit(
        self,
        params: dict,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_estimators: int,
        tr_groups: list[int] | None = None,
        val_groups: list[int] | None = None,
    ) -> lgb.Booster:
        train_data = lgb.Dataset(X_tr, label=y_tr)
        if tr_groups:
            train_data.set_group(tr_groups)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        if val_groups:
            valid_data.set_group(val_groups)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        return lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=callbacks,
        )

    def _split_groups(
        self, groups: list[int], split_ratio: float = 0.9
    ) -> tuple[int, list[int], list[int]]:
        n_groups = len(groups)
        split_group_idx = max(1, int(n_groups * split_ratio))
        split_group_idx = min(split_group_idx, n_groups - 1)
        split_row = sum(groups[:split_group_idx])
        return split_row, groups[:split_group_idx], groups[split_group_idx:]

    def _train_lgb(
        self,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        groups: list[int] | None = None,
        n_estimators: int | None = None,
    ) -> lgb.Booster:
        n_est = n_estimators if n_estimators is not None else self.config.n_estimators

        if groups and len(groups) > 1:
            split_row, tr_g, val_g = self._split_groups(groups)
            X_tr, X_ev = X.iloc[:split_row], X.iloc[split_row:]
            y_tr, y_ev = y.iloc[:split_row], y.iloc[split_row:]
            return self._fit(params, X_tr, y_tr, X_ev, y_ev, n_est, tr_g, val_g)
        else:
            n_total = len(X)
            split_idx = int(n_total * 0.9)
            X_tr, X_ev = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_ev = y.iloc[:split_idx], y.iloc[split_idx:]
            return self._fit(params, X_tr, y_tr, X_ev, y_ev, n_est)

    def _cv_folds(
        self, groups: list[int], n_splits: int = 5
    ) -> list[tuple[list[int], list[int], int, int]]:
        n_groups = len(groups)
        folds = []
        for i in range(n_splits):
            train_end = int(n_groups * (i + 1) / n_splits)
            if train_end < 2:
                continue
            val_start = train_end
            val_size = max(1, int(n_groups / n_splits))
            val_end = min(n_groups, val_start + val_size)
            if val_end <= val_start:
                val_end = val_start + 1

            tr_groups = groups[:val_start]
            val_groups = groups[val_start:val_end]
            train_rows = sum(tr_groups)
            val_rows = sum(val_groups)
            folds.append((tr_groups, val_groups, train_rows, val_rows))
        return folds

    def _rank_ic_score(
        self, y_pred: np.ndarray, y_true: np.ndarray, val_groups: list[int]
    ) -> float:
        start = 0
        day_ics = []
        for g in val_groups:
            end = start + g
            if g >= 3:
                ic, _ = spearmanr(y_pred[start:end], y_true[start:end])
                if not np.isnan(ic):
                    day_ics.append(ic)
            start = end
        if not day_ics:
            return 0.0
        return float(np.mean(day_ics))

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        groups: list[int],
    ) -> float:
        params = self._base_params()
        params.update(
            {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        )
        n_est = trial.suggest_int("n_estimators", 100, 500)

        folds = self._cv_folds(groups, n_splits=5)
        scores = []
        for tr_groups, val_groups, train_rows, val_rows in folds:
            if val_rows == 0:
                continue
            X_tr, X_val = (
                X.iloc[:train_rows],
                X.iloc[train_rows : train_rows + val_rows],
            )
            y_tr, y_val = (
                y.iloc[:train_rows],
                y.iloc[train_rows : train_rows + val_rows],
            )

            model = self._fit(
                params, X_tr, y_tr, X_val, y_val, n_est, tr_groups, val_groups
            )
            y_pred_raw = model.predict(X_val)
            y_pred_arr = np.asarray(y_pred_raw, dtype=float)
            ic = self._rank_ic_score(y_pred_arr, y_val.values, val_groups)
            scores.append(ic)

        return -float(np.mean(scores)) if scores else 0.0

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        label_name: str = "",
        groups: list[int] | None = None,
    ) -> tuple[lgb.Booster, dict]:
        if self.config.tune:
            logger.info(
                "Starting Optuna hyperparameter tuning (%s), %d trials",
                label_name,
                self.config.n_trials,
            )
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            )
            study.optimize(
                lambda trial: self._objective(trial, X, y, groups or []),
                n_trials=self.config.n_trials,
                show_progress_bar=False,
            )
            best_params = self._base_params()
            best_params.update(study.best_params)
            logger.info(
                "Best trial (%s): IC=%.4f, params=%s",
                label_name,
                -study.best_value if study.best_value is not None else 0,
                study.best_params,
            )

            model = self._train_lgb(
                best_params,
                X,
                y,
                groups,
                n_estimators=best_params.get("n_estimators"),
            )
            best_params["best_iteration"] = model.best_iteration
            return model, best_params
        else:
            logger.info("Training LightGBM with base params (%s)", label_name)
            params = self._base_params()
            model = self._train_lgb(params, X, y, groups)
            params["best_iteration"] = model.best_iteration
            return model, params

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from alpha_quat.model.lightgbm.config import LightGBMConfig

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    def __init__(self, config: LightGBMConfig):
        self.config = config

    def _base_params(self) -> dict:
        return {
            "objective": "regression",
            "metric": "l2",
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
    ) -> lgb.Booster:
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

    def _train_lgb(
        self,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int | None = None,
    ) -> lgb.Booster:
        n_est = n_estimators if n_estimators is not None else self.config.n_estimators
        n_total = len(X)
        split_idx = int(n_total * 0.9)
        X_tr, X_ev = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_ev = y.iloc[:split_idx], y.iloc[split_idx:]

        return self._fit(params, X_tr, y_tr, X_ev, y_ev, n_est)

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
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

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self._fit(params, X_tr, y_tr, X_val, y_val, n_est)
            y_pred = model.predict(X_val)
            mse = float(((y_val.values - y_pred) ** 2).mean())
            scores.append(mse)

        return float(np.mean(scores))

    def train(
        self, X: pd.DataFrame, y: pd.Series, label_name: str = ""
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
                lambda trial: self._objective(trial, X, y),
                n_trials=self.config.n_trials,
                show_progress_bar=False,
            )
            best_params = self._base_params()
            best_params.update(study.best_params)
            logger.info(
                "Best trial (%s): MSE=%.6f, params=%s",
                label_name,
                study.best_value,
                study.best_params,
            )

            model = self._train_lgb(
                best_params, X, y, n_estimators=best_params.get("n_estimators")
            )
            best_params["best_iteration"] = model.best_iteration
            return model, best_params
        else:
            logger.info("Training LightGBM with base params (%s)", label_name)
            params = self._base_params()
            model = self._train_lgb(params, X, y)
            params["best_iteration"] = model.best_iteration
            return model, params

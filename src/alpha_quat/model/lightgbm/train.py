import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.config import LightGBMConfig

logger = logging.getLogger(__name__)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


class LightGBMTrainer:
    def __init__(self, config: LightGBMConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "LightGBMTrainer":
        lgb_cfg = LightGBMConfig(
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            feature_fraction=config.feature_fraction,
            bagging_fraction=config.bagging_fraction,
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            verbosity=config.verbosity,
            n_trials=config.n_trials,
            tune=config.tune,
            feature_names=config.feature_names,
            n_tile=config.n_tile,
            label_gain=config.label_gain,
        )
        return cls(lgb_cfg)

    def _base_params(
        self, quantile_alpha: float | None = None, lambdarank: bool = False
    ) -> dict:
        if lambdarank:
            if self.config.label_gain is not None:
                gain = self.config.label_gain
            else:
                gain = list(range(self.config.n_tile))
            return {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "label_gain": gain,
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "feature_fraction": self.config.feature_fraction,
                "bagging_fraction": self.config.bagging_fraction,
                "verbose": self.config.verbosity,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
            }
        if quantile_alpha is not None:
            return {
                "objective": "quantile",
                "metric": "quantile",
                "alpha": quantile_alpha,
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "n_estimators": self.config.n_estimators,
                "feature_fraction": self.config.feature_fraction,
                "bagging_fraction": self.config.bagging_fraction,
                "verbose": self.config.verbosity,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
            }
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

    def _split_by_groups(
        self, groups: list[int], ratio: float = 0.9
    ) -> tuple[list[int], list[int], int]:
        split_n = max(1, int(len(groups) * ratio))
        if split_n >= len(groups):
            split_n = len(groups) - 1
        split_row = sum(groups[:split_n])
        return groups[:split_n], groups[split_n:], split_row

    def _objective_lambdarank(
        self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, groups: list[int]
    ) -> float:
        params = self._base_params(lambdarank=True)
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

        train_g, val_g, split = self._split_by_groups(groups)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        model = self._fit(params, X_tr, y_tr, X_val, y_val, n_est, train_g, val_g)
        y_pred = np.asarray(model.predict(X_val), dtype=float)
        return float(np.mean((y_val.values - y_pred) ** 2))

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        quantile_alpha: float | None = None,
    ) -> float:
        params = self._base_params(quantile_alpha)
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
            y_pred = np.asarray(model.predict(X_val), dtype=float)
            if quantile_alpha is not None:
                scores.append(pinball_loss(y_val.values, y_pred, quantile_alpha))
            else:
                scores.append(float(((y_val.values - y_pred) ** 2).mean()))
        return float(np.mean(scores))

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        label_name: str = "",
        quantile_alpha: float | None = None,
        lambdarank: bool = False,
        groups: list[int] | None = None,
    ) -> tuple[lgb.Booster, dict]:
        if lambdarank:
            params = self._base_params(lambdarank=True)
            if self.config.tune:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
                )
                study.optimize(
                    lambda trial: self._objective_lambdarank(trial, X, y, groups or []),
                    n_trials=self.config.n_trials,
                    show_progress_bar=False,
                )
                best_params = self._base_params(lambdarank=True)
                best_params.update(study.best_params)
                logger.info(
                    "Best lambdarank (%s): MSE=%.6f", label_name, study.best_value
                )

                tr_g, val_g, split = self._split_by_groups(groups or [])
                model = self._fit(
                    best_params,
                    X.iloc[:split],
                    y.iloc[:split],
                    X.iloc[split:],
                    y.iloc[split:],
                    best_params.get("n_estimators", 200),
                    tr_g,
                    val_g,
                )
                best_params["best_iteration"] = model.best_iteration
                return model, best_params
            else:
                tr_g, val_g, split = self._split_by_groups(groups or [])
                logger.info(
                    "Training lambdarank (%s): %d groups, %d samples",
                    label_name,
                    len(tr_g),
                    split,
                )
                model = self._fit(
                    params,
                    X.iloc[:split],
                    y.iloc[:split],
                    X.iloc[split:],
                    y.iloc[split:],
                    self.config.n_estimators,
                    tr_g,
                    val_g,
                )
                params["best_iteration"] = model.best_iteration
                return model, params

        # Non-lambdarank path
        params = self._base_params(quantile_alpha)
        if self.config.tune:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            )
            study.optimize(
                lambda trial: self._objective(trial, X, y, quantile_alpha),
                n_trials=self.config.n_trials,
                show_progress_bar=False,
            )
            best_params = self._base_params(quantile_alpha)
            best_params.update(study.best_params)
            model = self._train_lgb(best_params, X, y, best_params.get("n_estimators"))
            best_params["best_iteration"] = model.best_iteration
            return model, best_params
        else:
            model = self._train_lgb(params, X, y)
            params["best_iteration"] = model.best_iteration
            return model, params

    def _train_lgb(
        self,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int | None = None,
    ) -> lgb.Booster:
        n_est = n_estimators or self.config.n_estimators
        n_total = len(X)
        split_idx = int(n_total * 0.9)
        return self._fit(
            params,
            X.iloc[:split_idx],
            y.iloc[:split_idx],
            X.iloc[split_idx:],
            y.iloc[split_idx:],
            n_est,
        )

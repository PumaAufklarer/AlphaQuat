"""Rolling retrain + backtest — train models on expanding windows, backtest each 6-month fold."""

import json
import logging
import shutil
from pathlib import Path

from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine
from alpha_quat.model.lightgbm.config import LightGBMConfig
from alpha_quat.model.lightgbm.pipeline import LightGBMPipeline

logger = logging.getLogger(__name__)

# 8 folds: each trains on expanding window, backtests on next 6 months
_FOLDS = [
    ("20180401", "20220330", "20220401", "20220930"),
    ("20180401", "20220930", "20221001", "20230331"),
    ("20180401", "20230331", "20230401", "20230930"),
    ("20180401", "20230930", "20231001", "20240331"),
    ("20180401", "20240331", "20240401", "20240930"),
    ("20180401", "20240930", "20241001", "20250331"),
    ("20180401", "20250331", "20250401", "20250930"),
    ("20180401", "20250930", "20251001", "20260330"),
]

_BACKTEST_ARGS = {
    "capital": 50000,
    "monthly": 8000,
    "top_k": 5,
    "rebalance_interval": 10,
    "sell_threshold": 0.40,
    "stop_loss": 0.15,
}


def run_rolling(data_dir: Path) -> list[dict]:
    """Run 8-fold rolling retrain + backtest.

    Each fold:
      1. Train quantile models on expanding window (--no-tune)
      2. Save models to data/models/fold_N/
      3. Run backtest on validation window
      4. Collect metrics
    """
    results = []
    for i, (tr_s, tr_e, val_s, val_e) in enumerate(_FOLDS):
        fold_dir = data_dir / "models" / f"fold_{i + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(
            "FOLD %d/%d: train %s-%s, validate %s-%s",
            i + 1,
            len(_FOLDS),
            tr_s,
            tr_e,
            val_s,
            val_e,
        )
        logger.info("=" * 60)

        # Train quantile models
        cfg = LightGBMConfig(
            train_start=tr_s,
            train_end=tr_e,
            val_start=tr_s,
            val_end=val_e,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            tune=False,
            quantile_alphas=[0.1, 0.5, 0.9],
        )
        pipeline = LightGBMPipeline(data_dir, cfg)
        pipeline.run()

        # Copy models to fold directory
        for f in data_dir.glob("models/lightgbm_model_*.txt"):
            shutil.copy2(f, fold_dir / f.name)

        # Run backtest
        bt_cfg = BacktestConfig(
            start_date=val_s,
            end_date=val_e,
            initial_capital=_BACKTEST_ARGS["capital"],
            monthly_addition=_BACKTEST_ARGS["monthly"],
            commission_rate=0.0005,
            stop_loss_pct=_BACKTEST_ARGS["stop_loss"],
            top_k=_BACKTEST_ARGS["top_k"],
            model_dir=str(fold_dir),
            rebalance_interval=_BACKTEST_ARGS["rebalance_interval"],
            sell_threshold=_BACKTEST_ARGS["sell_threshold"],
        )
        engine = BacktestEngine(bt_cfg, data_dir)
        result = engine.run()

        metrics = result["metrics"]
        logger.info(
            "Fold %d results: cum_ret=%.2f%% sharpe=%.2f mdd=%.2f%% trades=%d",
            i + 1,
            metrics["cumulative_return"] * 100,
            metrics["sharpe_ratio"],
            metrics["max_drawdown"] * 100,
            metrics["total_trades"],
        )
        results.append(
            {
                "fold": i + 1,
                "train": f"{tr_s}-{tr_e}",
                "test": f"{val_s}-{val_e}",
                "cum_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "total_trades": metrics["total_trades"],
                "final_value": metrics["final_value"],
                "total_invested": metrics["total_invested"],
            }
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ROLLING BACKTEST SUMMARY")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "Fold %d (%s): cum=%.1f%% sharpe=%.2f mdd=%.1f%% trades=%d",
            r["fold"],
            r["test"],
            r["cum_return"] * 100,
            r["sharpe"],
            r["max_drawdown"] * 100,
            r["total_trades"],
        )

    avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
    avg_return = sum(r["cum_return"] for r in results) / len(results)
    max_mdd = max(r["max_drawdown"] for r in results)
    total_trades = sum(r["total_trades"] for r in results)
    logger.info(
        "AVERAGE: cum=%.1f%% sharpe=%.2f max_mdd=%.1f%% total_trades=%d",
        avg_return * 100,
        avg_sharpe,
        max_mdd * 100,
        total_trades,
    )

    # Save results
    with open(data_dir / "models" / "rolling_results.json", "w") as f:
        json.dump(
            {
                "folds": results,
                "average_sharpe": avg_sharpe,
                "average_return": avg_return,
                "max_drawdown": max_mdd,
                "total_trades": total_trades,
            },
            f,
            indent=2,
        )

    return results

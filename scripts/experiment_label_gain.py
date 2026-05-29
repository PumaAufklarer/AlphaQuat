#!/usr/bin/env python3
"""Train LambdaRank variants with tuning (30 trials), backtest, document.

Usage: uv run python3 scripts/experiment_label_gain.py
"""

# ruff: noqa: E402

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine
from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.pipeline import run_variant

DATA_DIR = Path("data")
TRAIN_START, TRAIN_END = "20230101", "20240630"
VAL_START, VAL_END = "20240701", "20250430"
BT_START, BT_END = "20240701", "20260501"
SEED, TRIALS = 42, 30

VARIANTS = {
    "baseline": {"n_tile": 10, "label_gain": None, "desc": "NTILE=10, linear[0..9]"},
    "expgain": {
        "n_tile": 10,
        "label_gain": [0, 1, 2, 4, 7, 11, 16, 23, 31, 42],
        "desc": "NTILE=10, exp gain ~42:1",
    },
    "ntile5": {
        "n_tile": 5,
        "label_gain": None,
        "desc": "NTILE=5, linear[0..4]",
    },
}

RESULTS = []

for vname, vcfg in VARIANTS.items():
    exp_name = f"exp_lbl_{vname}"
    print(f"\n{'=' * 60}")
    print(f"  {vcfg['desc']}")
    print(f"  {exp_name}  seed={SEED}  trials={TRIALS}")
    print(f"{'=' * 60}")

    config = ExperimentConfig(
        name=exp_name,
        mode="lambdarank",
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        val_start=VAL_START,
        val_end=VAL_END,
        tune=True,
        n_trials=TRIALS,
        random_state=SEED,
        n_tile=vcfg["n_tile"],
        label_gain=vcfg["label_gain"],
    )

    try:
        train_result = run_variant(DATA_DIR, config)

        ics = {}
        for suffix, res in train_result.items():
            ics[suffix] = {"mean_ic": round(res.mean_ic, 4), "icir": round(res.icir, 4)}
            print(
                f"  {suffix}: IC={res.mean_ic:.4f}  ICIR={res.icir:.4f}  "
                f"top3={[f[0] for f in res.top5_features[:3]]}"
            )

        exp_dir = DATA_DIR / "models" / "experiments" / exp_name
        saved_cfg = ExperimentConfig.load(exp_dir / "experiment.yaml")

        bt_cfg = BacktestConfig(
            start_date=BT_START,
            end_date=BT_END,
            initial_capital=50000,
            monthly_addition=8000,
            top_k=15,
            rebalance_interval=5,
            weighting_strategy="score_momentum",
            experiment_name=exp_name,
        )
        engine = BacktestEngine(bt_cfg, DATA_DIR)
        bt_result = engine.run()
        m = bt_result["metrics"]

        entry = {
            "experiment": exp_name,
            "variant": vname,
            "desc": vcfg["desc"],
            "n_tile": vcfg["n_tile"],
            "label_gain": vcfg["label_gain"],
            "seed": SEED,
            "trials": TRIALS,
            "ics": ics,
            "sharpe": round(m["sharpe_ratio"], 4),
            "cum_ret": round(m["cumulative_return"], 4),
            "annual_ret": round(m["annualized_return"], 4),
            "max_dd": round(m["max_drawdown"], 4),
            "win_rate": round(m["win_rate"], 4),
            "trades": m["total_trades"],
            "best_params": {
                "num_leaves": saved_cfg.num_leaves,
                "learning_rate": saved_cfg.learning_rate,
                "n_estimators": saved_cfg.n_estimators,
                "feature_fraction": saved_cfg.feature_fraction,
                "bagging_fraction": saved_cfg.bagging_fraction,
            },
        }
        RESULTS.append(entry)
        print(
            f"  BT: Sharpe={entry['sharpe']:.4f}  CumRet={entry['cum_ret']:.2%}  "
            f"MaxDD={entry['max_dd']:.2%}  Trades={entry['trades']}  WR={entry['win_rate']:.1%}"
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        RESULTS.append({"experiment": exp_name, "variant": vname, "error": str(e)})

# Save
output_path = Path("experiments_label_gain.json")
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)

# Print table
print(f"\n{'=' * 95}")
print(
    f"{'Variant':<12} {'5d IC':>7} {'20d IC':>7} {'60d IC':>7} "
    f"{'Sharpe':>8} {'CumRet':>8} {'MaxDD':>7} {'Trades':>7} {'WR':>6}"
)
print("-" * 95)
for r in RESULTS:
    if "error" in r:
        print(f"{r['variant']:<12} ERROR: {r['error'][:60]}")
        continue
    i = r["ics"]
    print(
        f"{r['variant']:<12} "
        f"{i.get('5d', {}).get('mean_ic', 0):>7.4f} "
        f"{i.get('20d', {}).get('mean_ic', 0):>7.4f} "
        f"{i.get('60d', {}).get('mean_ic', 0):>7.4f} "
        f"{r['sharpe']:>8.4f} {r['cum_ret']:>7.2%} {r['max_dd']:>7.2%} "
        f"{r['trades']:>7d} {r['win_rate']:>6.1%}"
    )
print(f"\nSaved: {output_path}")

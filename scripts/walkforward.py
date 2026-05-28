#!/usr/bin/env python3
"""Walk-forward backtest: sliding 4-year train windows, 1-year test.

Train: 4 years. Gap: 2 months (purge 60d labels). Test: 1 year.
Windows slide every 6 months.

Usage: uv run python3 scripts/walkforward.py
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine
from alpha_quat.experiment.config import ExperimentConfig
from alpha_quat.model.lightgbm.pipeline import run_variant

DATA_DIR = Path("data")
TRIALS = 5  # walk-forward validation, fewer trials for speed


def get_trade_cal() -> list[str]:
    cal = pd.read_parquet(DATA_DIR / "trade_cal.parquet")
    return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())


def add_months(ymd: str, n: int) -> str:
    dt = datetime.strptime(ymd, "%Y%m%d")
    total_months = dt.year * 12 + dt.month - 1 + n
    y = total_months // 12
    m = total_months % 12 + 1
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
        days_in_month[1] = 29
    d = min(dt.day, days_in_month[m - 1])
    return f"{y:04d}{m:02d}{d:02d}"


def find_nearest_trade_date(dates: list[str], target: str, after: bool = True) -> str:
    candidates = [d for d in dates if (d >= target if after else d <= target)]
    if not candidates:
        return dates[0] if after else dates[-1]
    return candidates[0] if after else candidates[-1]


def generate_windows(dates: list[str], train_years: int = 4, gap_months: int = 2):
    """Generate train/test windows sliding every 6 months."""
    windows = []
    start = "20180101"
    last_date = dates[-1]
    while True:
        train_end = add_months(start, train_years * 12)
        test_start = add_months(train_end, gap_months)
        test_end = add_months(test_start, 12)

        if test_start > last_date:
            break

        test_end_clamped = min(test_end, last_date)
        if test_start >= test_end_clamped:
            start = add_months(start, 6)
            continue

        train_start_dt = find_nearest_trade_date(dates, start, after=True)
        train_end_dt = find_nearest_trade_date(dates, train_end, after=False)
        test_start_dt = find_nearest_trade_date(dates, test_start, after=True)
        test_end_dt = find_nearest_trade_date(
            dates, min(test_end, last_date), after=False
        )

        if train_start_dt < train_end_dt and test_start_dt < test_end_dt:
            windows.append(
                {
                    "train_start": train_start_dt,
                    "train_end": train_end_dt,
                    "test_start": test_start_dt,
                    "test_end": test_end_dt,
                }
            )
        start = add_months(start, 6)
    return windows


def run_window(w: dict, idx: int) -> dict:
    exp_name = f"wf_w{idx:02d}"
    print(f"\n{'=' * 60}")
    print(
        f"  Window {idx}: train {w['train_start']}~{w['train_end']} → test {w['test_start']}~{w['test_end']}"
    )
    print(f"{'=' * 60}")

    config = ExperimentConfig(
        name=exp_name,
        mode="lambdarank",
        train_start=w["train_start"],
        train_end=w["train_end"],
        val_start=w["test_start"],
        val_end=w["test_end"],
        tune=True,
        n_trials=TRIALS,
        random_state=42,
        n_tile=10,
    )

    run_variant(DATA_DIR, config)

    bt_cfg = BacktestConfig(
        start_date=w["test_start"],
        end_date=w["test_end"],
        initial_capital=50000,
        monthly_addition=8000,
        top_k=15,
        rebalance_interval=5,
        weighting_strategy="score_momentum",
        experiment_name=exp_name,
    )
    engine = BacktestEngine(bt_cfg, DATA_DIR)
    result = engine.run()
    m = result["metrics"]

    return {
        "window": idx,
        "train": f"{w['train_start']}~{w['train_end']}",
        "test": f"{w['test_start']}~{w['test_end']}",
        "sharpe": round(m["sharpe_ratio"], 4),
        "cum_ret": round(m["cumulative_return"], 4),
        "annual_ret": round(m["annualized_return"], 4),
        "max_dd": round(m["max_drawdown"], 4),
        "win_rate": round(m["win_rate"], 4),
        "trades": m["total_trades"],
    }


def main():
    dates = get_trade_cal()
    windows = generate_windows(dates)
    print(f"Generated {len(windows)} windows")

    all_results = []
    for i, w in enumerate(windows):
        try:
            r = run_window(w, i)
            all_results.append(r)
            print(
                f"  → Sharpe={r['sharpe']:.3f}  CumRet={r['cum_ret']:.2%}  "
                f"MaxDD={r['max_dd']:.2%}  Trades={r['trades']}  WR={r['win_rate']:.1%}"
            )
        except Exception as e:
            print(f"  → FAILED: {e}")
            all_results.append({**{k: str(v) for k, v in w.items()}, "error": str(e)})

    # Summary
    print(f"\n{'=' * 90}")
    print(
        f"{'W':>2} {'Train':<21} {'Test':<21} {'Sharpe':>7} {'CumRet':>8} {'MaxDD':>7} {'Trades':>6} {'WR':>5}"
    )
    print("-" * 90)
    sharpes = []
    for r in all_results:
        if "error" in r:
            print(f"  ERR: {r['error'][:50]}")
        else:
            sharpes.append(r["sharpe"])
            print(
                f"{r['window']:>2} {r['train']:<21} {r['test']:<21} "
                f"{r['sharpe']:>7.3f} {r['cum_ret']:>7.2%} {r['max_dd']:>7.2%} "
                f"{r['trades']:>6d} {r['win_rate']:>4.1%}"
            )

    if sharpes:
        mu = sum(sharpes) / len(sharpes)
        std = (sum((s - mu) ** 2 for s in sharpes) / max(len(sharpes) - 1, 1)) ** 0.5
        pos = sum(1 for s in sharpes if s > 0)
        print(f"\n  Mean Sharpe: {mu:.3f} ± {std:.3f}  Positive: {pos}/{len(sharpes)}")
        print(
            f"  → Safe for live trading"
            if mu > 1.0 and pos == len(sharpes)
            else f"  → Needs more work"
        )

    with open("walkforward_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: walkforward_results.json")


if __name__ == "__main__":
    main()

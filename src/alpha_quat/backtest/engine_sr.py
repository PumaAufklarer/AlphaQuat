"""SR backtest engine — pre-loaded data, daily batch inference."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from alpha_quat.backtest.filters import build_universe
from alpha_quat.backtest.metrics import compute_metrics
from alpha_quat.backtest.portfolio import Portfolio
from alpha_quat.strategy.signals.variants.transformer_sr_signal import (
    TransformerSRSignal,
)
from alpha_quat.strategy.types import SignalResult

logger = logging.getLogger(__name__)

_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]


MAX_POS_PCT = 0.25
MIN_CASH_PCT = 0.10
MAX_HOLDINGS = 8


def _load_cache_range(
    data_dir: Path, start: str, end: str, lookback: int
) -> pd.DataFrame:
    """Load alpha360 cache for a date range, adding trade_date column."""
    from datetime import datetime, timedelta

    cache_dir = data_dir / "alpha360"
    td = datetime.strptime(start, "%Y%m%d")
    start_dt = td - timedelta(days=lookback * 2 + 10)

    chunks = []
    for f in sorted(cache_dir.glob("*.parquet")):
        ds = f.stem
        if ds >= start_dt.strftime("%Y%m%d") and ds <= end:
            df = pd.read_parquet(f)
            df["trade_date"] = ds
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _build_daily_batch(
    stock_data: dict,
    stock_dates: dict,
    inference,
    trade_date: str,
    universe: set,
    seq_length: int,
):
    """Build (sequences, codes, close_prices) for all universe stocks on trade_date."""
    sequences, codes, close_prices = [], [], []
    for code in universe:
        if code not in stock_data:
            continue
        arr = stock_data[code]
        dates = stock_dates[code]
        try:
            day_idx = dates.index(trade_date)
        except ValueError:
            continue
        if day_idx < seq_length - 1:
            continue

        seq = arr[day_idx - seq_length + 1 : day_idx + 1].copy()
        if np.isnan(seq).any():
            continue

        close_last = seq[-1, 3]
        if close_last <= 0:
            continue

        normed = inference._normalize(seq)
        sequences.append(normed)
        codes.append(code)
        close_prices.append(close_last)

    return np.stack(sequences) if sequences else None, codes, close_prices


def run_sr_backtest(
    data_dir: Path,
    experiment_name: str,
    start_date: str = "20220101",
    end_date: str = "20241231",
    initial_capital: float = 50000,
    commission_rate: float = 0.0005,
    stop_loss_pct: float = 0.15,
) -> dict:
    cal = pd.read_parquet(data_dir / "trade_cal.parquet")
    all_dates = sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())
    dates = [d for d in all_dates if start_date <= d <= end_date]
    if not dates:
        raise ValueError(f"No trading dates in range {start_date}~{end_date}")

    exp_dir = data_dir / "models" / "experiments" / experiment_name
    signal_gen = TransformerSRSignal(exp_dir, data_dir=data_dir)
    inference = signal_gen.inference
    seq_length = inference.config.seq_length

    # ── Pre-load all alpha360 data once ──
    logger.info("Pre-loading alpha360 cache for %s ~ %s ...", start_date, end_date)
    all_cache = _load_cache_range(data_dir, dates[0], dates[-1], seq_length)
    logger.info("Loaded %d rows", len(all_cache))

    # Build per-stock arrays (sorted by trade_date)
    stock_data: dict[str, np.ndarray] = {}
    stock_dates: dict[str, list[str]] = {}
    for code, df in all_cache.groupby("ts_code"):
        df = df.sort_values("trade_date")
        stock_data[code] = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        stock_dates[code] = df["trade_date"].tolist()
    logger.info("Pre-loaded %d stocks", len(stock_data))

    # ── Backtest loop ──
    portfolio = Portfolio(cash=initial_capital)
    total_invested = initial_capital
    additions: dict[str, float] = {}
    tracked_months: set[str] = set()
    unfilled_signals: list[dict] = []
    pending_signals: SignalResult | None = None

    for idx, td in enumerate(tqdm(dates, desc="Backtesting")):
        # Monthly addition
        month_key = td[:6]
        if month_key not in tracked_months:
            tracked_months.add(month_key)
            if idx > 0:
                portfolio.cash += 8000
                total_invested += 8000
                additions[td] = additions.get(td, 0) + 8000

        daily_path = data_dir / "daily" / f"{td[:4]}_{td[4:6]}_{td[6:8]}.parquet"
        if not daily_path.exists():
            continue

        daily = pd.read_parquet(daily_path)
        open_px = dict(zip(daily["ts_code"], daily["open"]))
        close_px = dict(zip(daily["ts_code"], daily["close"]))
        low_px = dict(zip(daily["ts_code"], daily["low"]))
        high_px = dict(zip(daily["ts_code"], daily["high"]))
        universe = build_universe(td, data_dir)

        # Stop-loss: trailing stop + support-level stop
        portfolio.update_peak_prices(close_px)
        for code, h in list(portfolio.holdings.items()):
            cp = close_px.get(code)
            should_sell = (cp and cp < h.peak_price * (1 - stop_loss_pct)) or (
                cp and h.stop_price > 0 and cp < h.stop_price
            )
            if should_sell:
                px = open_px.get(code)
                if px and px > 0 and code in universe:
                    portfolio.sell(
                        code,
                        price=px,
                        shares=h.shares,
                        trade_date=td,
                        commission_rate=commission_rate,
                    )

        # Execute pending signals (from T-1, price-triggered)
        if pending_signals is not None and not pending_signals.signals.empty:
            sigs = pending_signals.signals

            sell_sigs = sigs[sigs["action"] == "sell"].sort_values(
                "score", ascending=False
            )
            for _, row in sell_sigs.iterrows():
                code = row["ts_code"]
                if code not in portfolio.holdings or code not in universe:
                    continue
                target = row["target_price"]
                hh = high_px.get(code)
                if hh and target > 0 and hh >= target:
                    h = portfolio.holdings[code]
                    fill = min(target, hh * 0.98)
                    portfolio.sell(
                        code,
                        price=fill,
                        shares=h.shares,
                        trade_date=td,
                        commission_rate=commission_rate,
                    )
                else:
                    unfilled_signals.append(
                        {
                            "date": td,
                            "code": code,
                            "action": "sell",
                            "target": target,
                            "reason": "not_triggered",
                        }
                    )

            buy_sigs = sigs[sigs["action"] == "buy"].sort_values(
                "score", ascending=False
            )
            if not buy_sigs.empty and len(portfolio.holdings) < MAX_HOLDINGS:
                slots = MAX_HOLDINGS - len(portfolio.holdings)
                buy_sigs = buy_sigs.head(slots)
                total_score = buy_sigs["score"].sum()
                available = portfolio.cash * (1 - MIN_CASH_PCT)

                for _, row in buy_sigs.iterrows():
                    code = row["ts_code"]
                    if code in portfolio.holdings or code not in universe:
                        continue
                    target = row["target_price"]
                    ll = low_px.get(code)
                    if ll and target > 0 and ll <= target:
                        score = row["score"]
                        alloc = (
                            available * score / total_score if total_score > 0 else 0
                        )
                        alloc = min(
                            alloc, portfolio.total_value(close_px) * MAX_POS_PCT
                        )
                        fill = max(target, ll * 1.002)
                        stop = row.get("stop_price", fill * 0.93)
                        portfolio.buy(
                            code,
                            price=fill,
                            target_amount=alloc,
                            trade_date=td,
                            commission_rate=commission_rate,
                            stop_price=stop,
                        )
                    else:
                        unfilled_signals.append(
                            {
                                "date": td,
                                "code": code,
                                "action": "buy",
                                "target": target,
                                "reason": "not_triggered",
                            }
                        )

        # Generate next day's signals
        X_batch, codes, close_prices = _build_daily_batch(
            stock_data, stock_dates, inference, td, universe, seq_length
        )
        if X_batch is not None:
            batch_results = inference.compute_entry_exit_batch(
                X_batch, np.array(close_prices)
            )
            records = []
            for code, sr, cl in zip(codes, batch_results, close_prices):
                if sr["entry"]:
                    ep = cl * (1 + sr["expected_down"])
                    records.append(
                        {
                            "ts_code": code,
                            "action": "buy",
                            "score": min(
                                sr["rr_ratio"] * sr["support_confidence"], 1.0
                            ),
                            "target_price": round(ep, 2),
                            "stop_price": round(ep * 0.93, 2),
                            "rr_ratio": round(sr["rr_ratio"], 2),
                        }
                    )
                if sr["exit"]:
                    ep = cl * (1 + sr["expected_up"])
                    records.append(
                        {
                            "ts_code": code,
                            "action": "sell",
                            "score": round(sr["resistance_confidence"], 2),
                            "target_price": round(ep, 2),
                            "stop_price": 0,
                            "rr_ratio": round(sr["rr_ratio"], 2),
                        }
                    )
            pending_signals = SignalResult(
                signals=pd.DataFrame(records)
                if records
                else pd.DataFrame(
                    columns=[
                        "ts_code",
                        "action",
                        "score",
                        "target_price",
                        "stop_price",
                        "rr_ratio",
                    ]
                ),
                metadata={"model": "transformer_sr", "trade_date": td},
            )
        else:
            pending_signals = None

        portfolio.record_snapshot(td, close_px)

    metrics = compute_metrics(
        snapshots=portfolio.snapshots,
        trades=portfolio.trades,
        total_invested=total_invested,
        additions=additions,
    )
    logger.info(
        "Backtest complete: cum=%.2f%% sharpe=%.2f mdd=%.2f%% trades=%d unfilled=%d",
        metrics["cumulative_return"] * 100,
        metrics["sharpe_ratio"],
        metrics["max_drawdown"] * 100,
        metrics["total_trades"],
        len(unfilled_signals),
    )
    return {
        "snapshots": portfolio.snapshots,
        "trades": portfolio.trades,
        "metrics": metrics,
        "unfilled_signals": unfilled_signals,
    }

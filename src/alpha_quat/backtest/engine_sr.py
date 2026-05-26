"""SR backtest engine — price-triggered entry/exit with position sizing."""

import logging
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.filters import build_universe
from alpha_quat.backtest.metrics import compute_metrics
from alpha_quat.backtest.portfolio import Portfolio
from alpha_quat.strategy.signals.variants.transformer_sr_signal import (
    TransformerSRSignal,
)
from alpha_quat.strategy.types import SignalResult, StrategyContext

logger = logging.getLogger(__name__)


def _ymd_to_path(ymd: str) -> str:
    return f"{ymd[:4]}_{ymd[4:6]}_{ymd[6:8]}"


# Position sizing constants
MAX_POS_PCT = 0.25  # max 25% per stock
MIN_CASH_PCT = 0.10  # min 10% cash reserve
MAX_HOLDINGS = 8


def run_sr_backtest(
    data_dir: Path,
    experiment_name: str,
    start_date: str = "20220101",
    end_date: str = "20241231",
    initial_capital: float = 50000,
    commission_rate: float = 0.0005,
    stop_loss_pct: float = 0.15,
) -> dict:
    cal_path = data_dir / "trade_cal.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(
            "trade_cal.parquet not found. Run 'alpha-quat fetch' first."
        )

    cal = pd.read_parquet(cal_path)
    all_dates = sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist())
    dates = [d for d in all_dates if start_date <= d <= end_date]
    if not dates:
        raise ValueError(f"No trading dates in range {start_date}~{end_date}")

    exp_dir = data_dir / "models" / "experiments" / experiment_name
    signal_gen = TransformerSRSignal(exp_dir, data_dir=data_dir)
    portfolio = Portfolio(cash=initial_capital)

    total_invested = initial_capital
    additions: dict[str, float] = {}
    tracked_months: set[str] = set()
    unfilled_signals: list[dict] = []
    pending_signals: SignalResult | None = None

    for idx, td in enumerate(dates):
        month_key = td[:6]
        if month_key not in tracked_months:
            tracked_months.add(month_key)
            if idx > 0:
                portfolio.cash += 8000
                total_invested += 8000
                additions[td] = additions.get(td, 0) + 8000

        daily_path = data_dir / "daily" / f"{_ymd_to_path(td)}.parquet"
        if not daily_path.exists():
            continue

        daily = pd.read_parquet(daily_path)
        open_px = dict(zip(daily["ts_code"], daily["open"]))
        close_px = dict(zip(daily["ts_code"], daily["close"]))
        low_px = dict(zip(daily["ts_code"], daily["low"]))
        high_px = dict(zip(daily["ts_code"], daily["high"]))
        universe = build_universe(td, data_dir)

        # --- Stop-loss check (close-triggered) ---
        portfolio.update_peak_prices(close_px)
        for code, h in list(portfolio.holdings.items()):
            cp = close_px.get(code)
            if cp and cp < h.peak_price * (1 - stop_loss_pct):
                px = open_px.get(code)
                if px and px > 0 and code in universe:
                    portfolio.sell(
                        code,
                        price=px,
                        shares=h.shares,
                        trade_date=td,
                        commission_rate=commission_rate,
                    )
                    logger.debug("Stop-loss: sold %s at %.2f", code, px)

        # --- Execute signals from T-1 (price-triggered) ---
        if pending_signals is not None and not pending_signals.signals.empty:
            sigs = pending_signals.signals

            # Execute sells first (free up cash for buys)
            sell_sigs = sigs[sigs["action"] == "sell"].sort_values(
                "score", ascending=False
            )
            for _, row in sell_sigs.iterrows():
                code = row["ts_code"]
                if code not in portfolio.holdings or code not in universe:
                    continue
                target = row["target_price"]
                todays_high = high_px.get(code)
                if todays_high and target > 0 and todays_high >= target:
                    h = portfolio.holdings[code]
                    fill = min(target, max(target, todays_high * 0.98))
                    portfolio.sell(
                        code,
                        price=fill,
                        shares=h.shares,
                        trade_date=td,
                        commission_rate=commission_rate,
                    )
                    logger.debug(
                        "Exit: sold %s at %.2f (target %.2f)", code, fill, target
                    )
                elif todays_high and target > 0 and todays_high < target:
                    unfilled_signals.append(
                        {
                            "date": td,
                            "code": code,
                            "action": "sell",
                            "target": target,
                            "reason": "not_triggered",
                        }
                    )

            # Execute buys — sorted by score, allocate cash proportionally
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
                    todays_low = low_px.get(code)
                    if todays_low and target > 0 and todays_low <= target:
                        score = row["score"]
                        alloc = (
                            available * score / total_score if total_score > 0 else 0
                        )
                        alloc = min(
                            alloc, portfolio.total_value(close_px) * MAX_POS_PCT
                        )
                        fill = max(target, todays_low * 1.002)
                        portfolio.buy(
                            code,
                            price=fill,
                            target_amount=alloc,
                            trade_date=td,
                            commission_rate=commission_rate,
                        )
                        logger.debug(
                            "Entry: bought %s at %.2f (target %.2f)", code, fill, target
                        )
                    elif todays_low and target > 0 and todays_low > target:
                        unfilled_signals.append(
                            {
                                "date": td,
                                "code": code,
                                "action": "buy",
                                "target": target,
                                "reason": "not_triggered",
                            }
                        )

        # --- Generate signals for tomorrow ---
        ctx = StrategyContext(
            trade_date=td,
            capital=portfolio.total_value(close_px),
            universe=list(universe),
            prices=daily[["ts_code", "open"]],
        )
        signal_result = signal_gen.generate(pd.DataFrame(), ctx)
        pending_signals = signal_result

        portfolio.record_snapshot(td, close_px)

    metrics = compute_metrics(
        snapshots=portfolio.snapshots,
        trades=portfolio.trades,
        total_invested=total_invested,
        additions=additions,
    )

    logger.info(
        "SR backtest complete: cum_ret=%.2f%% sharpe=%.2f mdd=%.2f%% trades=%d",
        metrics["cumulative_return"] * 100,
        metrics["sharpe_ratio"],
        metrics["max_drawdown"] * 100,
        metrics["total_trades"],
    )
    logger.info(
        "Unfilled signals: %d (%d sell, %d buy)",
        len(unfilled_signals),
        sum(1 for u in unfilled_signals if u["action"] == "sell"),
        sum(1 for u in unfilled_signals if u["action"] == "buy"),
    )

    return {
        "snapshots": portfolio.snapshots,
        "trades": portfolio.trades,
        "metrics": metrics,
        "unfilled_signals": unfilled_signals,
    }

import logging
from pathlib import Path

import pandas as pd

from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.filters import build_universe
from alpha_quat.backtest.portfolio import Portfolio
from alpha_quat.backtest.metrics import compute_metrics
from alpha_quat.strategy.types import StrategyContext
from alpha_quat.strategy.signals.ma_cross import MACrossSignal
from alpha_quat.strategy.positions.equal_weight import EqualWeightTopKPosition

logger = logging.getLogger(__name__)


def _ymd_to_path(ymd: str) -> str:
    return f"{ymd[:4]}_{ymd[4:6]}_{ymd[6:8]}"


class BacktestEngine:
    def __init__(self, config: BacktestConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.portfolio = Portfolio(cash=config.initial_capital)
        self.signal_gen = MACrossSignal(
            short_factor=config.short_factor, long_factor=config.long_factor
        )
        self.position_mgr = EqualWeightTopKPosition(top_k=config.top_k)
        self._pending_signals = None
        self._total_invested = config.initial_capital

    def run(self):
        cal_path = self.data_dir / "trade_cal.parquet"
        if not cal_path.exists():
            raise FileNotFoundError(
                "trade_cal.parquet not found. Run 'alpha-quat fetch' first."
            )

        cal = pd.read_parquet(cal_path)
        all_dates = sorted(
            cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist()
        )
        dates = [
            d for d in all_dates if self.config.start_date <= d <= self.config.end_date
        ]

        if not dates:
            logger.warning("No trading dates in range")
            return self._result()

        tracked_months: set[str] = set()
        self.signal_gen._prev = None
        self._pending_signals = None

        for idx, td in enumerate(dates):
            month_key = td[:6]
            if month_key not in tracked_months:
                tracked_months.add(month_key)
                if idx > 0:
                    self.portfolio.cash += self.config.monthly_addition
                    self._total_invested += self.config.monthly_addition

            daily_path = self.data_dir / "daily" / f"{_ymd_to_path(td)}.parquet"
            if not daily_path.exists():
                logger.warning("No daily data for %s, skipping", td)
                continue

            daily = pd.read_parquet(daily_path)
            open_px = dict(zip(daily["ts_code"], daily["open"]))
            close_px = dict(zip(daily["ts_code"], daily["close"]))
            universe = build_universe(td, self.data_dir)

            for code, h in list(self.portfolio.holdings.items()):
                prev_close = close_px.get(code)
                if prev_close and prev_close < h.avg_cost * (
                    1.0 - self.config.stop_loss_pct
                ):
                    px = open_px.get(code)
                    if px and px > 0 and code in universe:
                        self.portfolio.sell(
                            ts_code=code,
                            price=px,
                            shares=h.shares,
                            trade_date=td,
                            commission_rate=self.config.commission_rate,
                            min_commission=self.config.min_commission,
                        )

            if (
                self._pending_signals is not None
                and not self._pending_signals.signals.empty
            ):
                sig_df = self._pending_signals.signals
                sig_df = sig_df.loc[sig_df["ts_code"].isin(universe)]

                prices_df = daily[["ts_code", "open"]].copy()
                ctx = StrategyContext(
                    trade_date=td,
                    capital=self.portfolio.total_value(open_px),
                    universe=list(universe),
                    prices=prices_df,
                )

                buy_sigs = sig_df.loc[sig_df["action"] == "buy"]
                if not buy_sigs.empty:
                    fake_result = type(self._pending_signals)(
                        signals=buy_sigs.reset_index(drop=True),
                        metadata=self._pending_signals.metadata,
                    )
                    alloc = self.position_mgr.allocate(fake_result, ctx)
                    alloc = self.position_mgr.constrain(alloc, ctx)
                    for _, row in alloc.iterrows():
                        code = row["ts_code"]
                        px = open_px.get(code)
                        if px and px > 0:
                            target_amt = row["target_weight"] * ctx.capital
                            self.portfolio.buy(
                                ts_code=code,
                                price=px,
                                target_amount=target_amt,
                                trade_date=td,
                                commission_rate=self.config.commission_rate,
                                min_commission=self.config.min_commission,
                            )

                sell_sigs = sig_df.loc[sig_df["action"] == "sell"]
                for _, row in sell_sigs.iterrows():
                    code = row["ts_code"]
                    if code in self.portfolio.holdings:
                        h = self.portfolio.holdings[code]
                        px = open_px.get(code)
                        if px and px > 0:
                            self.portfolio.sell(
                                ts_code=code,
                                price=px,
                                shares=h.shares,
                                trade_date=td,
                                commission_rate=self.config.commission_rate,
                                min_commission=self.config.min_commission,
                            )

            feat_path = self.data_dir / "features" / f"{td}.parquet"
            if feat_path.exists():
                features = pd.read_parquet(feat_path)
                features = features.loc[features["ts_code"].isin(universe)]
                ctx_sig = StrategyContext(
                    trade_date=td, capital=self.portfolio.total_value(close_px)
                )
                self._pending_signals = self.signal_gen.generate(features, ctx_sig)
            else:
                self._pending_signals = None

            self.portfolio.record_snapshot(td, close_px)

        return self._result()

    def _result(self):
        metrics = compute_metrics(
            snapshots=self.portfolio.snapshots,
            trades=self.portfolio.trades,
            total_invested=self._total_invested,
        )
        return {
            "snapshots": self.portfolio.snapshots,
            "trades": self.portfolio.trades,
            "metrics": metrics,
        }

import pandas as pd

from alpha_quat.strategy.types import SignalResult, StrategyContext


class PeriodicRebalance:
    def __init__(self, config, signal_gen) -> None:
        self.config = config
        self.signal_gen = signal_gen
        self._last_rebalance_idx: int | None = None
        self._score_history: dict[str, list[float]] = {}
        self._pending_signals = None


class DailyMonitor:
    def __init__(self, config, signal_gen) -> None:
        self.config = config
        self.signal_gen = signal_gen
        self._last_rebalance_idx: int | None = None
        self._score_history: dict[str, list[float]] = {}
        self._pending_signals = None

    def on_date(
        self,
        idx: int,
        trade_date: str,
        portfolio,
        features: pd.DataFrame,
        close_px: dict,
        open_px: dict,
        universe: set,
    ) -> SignalResult | None:
        ctx = StrategyContext(
            trade_date=trade_date, capital=portfolio.total_value(close_px)
        )
        signal_result = self.signal_gen.generate(features, ctx)
        scores_map = dict(
            zip(
                signal_result.signals["ts_code"],
                signal_result.signals["score"],
            )
        )
        meta = signal_result.metadata
        has_ci = "ci_width" in meta and "confidence" in meta

        ci_map: dict[str, float] = {}
        conf_map: dict[str, float] = {}
        if has_ci:
            ci_map = dict(zip(signal_result.signals["ts_code"], meta["ci_width"]))
            conf_map = dict(zip(signal_result.signals["ts_code"], meta["confidence"]))

        sell_codes = []
        buy_codes = []
        for code, h in list(portfolio.holdings.items()):
            px = close_px.get(code)
            # Dynamic stop-loss
            if px and px < h.peak_price * (1 - self.config.stop_loss_pct):
                sell_codes.append(code)
                continue
            # CI-based sell
            if (
                has_ci
                and self.config.confidence_threshold is not None
                and conf_map.get(code, 0) >= self.config.confidence_threshold
            ):
                score = scores_map.get(code, 0)
                ci = ci_map.get(code, 0)
                if score + ci < self.config.sell_upper_threshold:
                    sell_codes.append(code)
                    continue
            # Fallback score percentile sell
            if not has_ci and self.config.sell_score_percentile is not None:
                all_scores = signal_result.signals["score"]
                score_cut = all_scores.quantile(self.config.sell_score_percentile)
                if isinstance(score_cut, pd.Series):
                    score_cut = float(score_cut.iloc[0])
                else:
                    score_cut = float(score_cut)
                if scores_map.get(code, 0) < score_cut:
                    sell_codes.append(code)

        # Buy pool
        current_set = set(portfolio.holdings.keys()) - set(sell_codes)
        buy_pool = signal_result.signals[
            ~signal_result.signals["ts_code"].isin(current_set | set(sell_codes))
        ].copy()
        if has_ci:
            buy_pool["buy_score"] = buy_pool["score"] * buy_pool["ts_code"].map(
                lambda c: conf_map.get(c, 0)
            )
            buy_pool = buy_pool.sort_values("buy_score", ascending=False)
        else:
            buy_pool = buy_pool.sort_values("score", ascending=False)
        buy_codes = (
            buy_pool["ts_code"].head(self.config.top_k - len(current_set)).tolist()
        )

        # Ensure top_k initially
        if self._last_rebalance_idx is None and len(current_set) < self.config.top_k:
            needed = self.config.top_k - len(current_set)
            for code in signal_result.signals["ts_code"]:
                if (
                    code not in current_set
                    and code not in buy_codes
                    and code not in sell_codes
                ):
                    buy_codes.append(code)
                    if len(buy_codes) >= needed:
                        break
            self._last_rebalance_idx = idx

        if sell_codes or buy_codes:
            sig_df = pd.DataFrame(
                {
                    "ts_code": buy_codes + sell_codes,
                    "score": [1.0] * len(buy_codes) + [0.0] * len(sell_codes),
                    "action": ["buy"] * len(buy_codes) + ["sell"] * len(sell_codes),
                }
            )
            self._pending_signals = SignalResult(signals=sig_df)
            return self._pending_signals
        else:
            self._pending_signals = None
            return None

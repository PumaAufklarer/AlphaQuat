"""TransformerSRSignal — entry/exit signals from SR model."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.model.nn.transformer.inference import SRInference
from alpha_quat.strategy.signals.variants import register
from alpha_quat.strategy.signals.variants.baseline import BaseMLSignal
from alpha_quat.strategy.types import SignalResult, StrategyContext

logger = logging.getLogger(__name__)

_FEATURE_COLS = ["open", "high", "low", "close", "volume", "vwap"]


def _load_recent_alpha360(
    data_dir: Path, trade_date: str, lookback: int
) -> pd.DataFrame:
    """Load alpha360 cache for lookback days before trade_date."""
    from datetime import datetime, timedelta

    cache_dir = data_dir / "alpha360"
    td = datetime.strptime(trade_date, "%Y%m%d")
    dates = []
    for d in range(lookback + 5):
        dt = td - timedelta(days=d)
        ds = dt.strftime("%Y%m%d")
        if (cache_dir / f"{ds}.parquet").exists():
            dates.append(ds)
        if len(dates) >= lookback:
            break
    dates = dates[:lookback]
    dates.reverse()

    if len(dates) < lookback:
        logger.warning(
            "Only found %d/%d trading days before %s", len(dates), lookback, trade_date
        )
        return pd.DataFrame()

    chunks = []
    for d in dates:
        df = pd.read_parquet(cache_dir / f"{d}.parquet")
        df["trade_date"] = d
        chunks.append(df)
    all_df = pd.concat(chunks, ignore_index=True)
    all_df = all_df.sort_values(["ts_code", "trade_date"])
    return all_df


@register
class TransformerSRSignal(BaseMLSignal):
    mode = "transformer_sr"

    def __init__(self, model_dir: Path, data_dir: Path | None = None):
        self.model_dir = Path(model_dir)
        self.inference = SRInference(self.model_dir)
        self.seq_length = self.inference.config.seq_length
        self.data_dir = data_dir

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        """Generate entry/exit signals for all stocks using SR model."""
        if self.data_dir is None:
            raise ValueError("data_dir required for SR signal")

        td = ctx.trade_date
        raw = _load_recent_alpha360(self.data_dir, td, self.seq_length)
        if raw.empty:
            return SignalResult(
                signals=pd.DataFrame(columns=["ts_code", "score", "action"])
            )

        buy_codes, sell_codes = [], []
        metadata = {"model": "transformer_sr"}

        for ts_code, stock_df in raw.groupby("ts_code"):
            stock_df = stock_df.sort_values("trade_date")
            if len(stock_df) < self.seq_length:
                continue

            vals = stock_df[_FEATURE_COLS].to_numpy(dtype=np.float32)[
                -self.seq_length :
            ]
            if np.isnan(vals).any():
                continue

            close_last = vals[-1, 3]  # close is col 3
            if close_last <= 0:
                continue

            try:
                sr = self.inference.compute_entry_exit(vals, close_last)
            except Exception:
                continue

            if sr["entry"]:
                buy_codes.append(ts_code)
            if sr["exit"]:
                sell_codes.append(ts_code)

        signals = pd.DataFrame(
            {
                "ts_code": buy_codes + sell_codes,
                "score": [1.0] * len(buy_codes) + [0.0] * len(sell_codes),
                "action": ["buy"] * len(buy_codes) + ["sell"] * len(sell_codes),
            }
        )
        return SignalResult(signals=signals, metadata=metadata)

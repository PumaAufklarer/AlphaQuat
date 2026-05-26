"""TransformerSRSignal — batch inference for speed."""

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
    from datetime import datetime, timedelta

    cache_dir = data_dir / "alpha360"
    td = datetime.strptime(trade_date, "%Y%m%d")
    dates = []
    for d in range(lookback * 2 + 10):
        dt = td - timedelta(days=d)
        ds = dt.strftime("%Y%m%d")
        if (cache_dir / f"{ds}.parquet").exists():
            dates.append(ds)
        if len(dates) >= lookback:
            break
    dates.reverse()
    if len(dates) < lookback:
        return pd.DataFrame()
    chunks = [pd.read_parquet(cache_dir / f"{d}.parquet") for d in dates]
    return pd.concat(chunks, ignore_index=True)


@register
class TransformerSRSignal(BaseMLSignal):
    mode = "transformer_sr"

    def __init__(self, model_dir: Path, data_dir: Path | None = None):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.inference = SRInference(self.model_dir)
        self.seq_length = self.inference.config.seq_length
        self.data_dir = data_dir

    def _load_models(self, model_dir: Path) -> dict:
        return {}

    def generate(self, features: pd.DataFrame, ctx: StrategyContext) -> SignalResult:
        if self.data_dir is None:
            raise ValueError("data_dir required for SR signal")

        raw = _load_recent_alpha360(self.data_dir, ctx.trade_date, self.seq_length)
        if raw.empty:
            return SignalResult(
                signals=pd.DataFrame(
                    columns=["ts_code", "action", "score", "target_price", "rr_ratio"]
                )
            )

        # Build batch from all stocks
        sequences, codes, close_prices = [], [], []
        for ts_code, stock_df in raw.groupby("ts_code"):
            stock_df = stock_df.sort_values("trade_date")
            if len(stock_df) < self.seq_length:
                continue
            vals = stock_df[_FEATURE_COLS].to_numpy(dtype=np.float32)[
                -self.seq_length :
            ]
            if np.isnan(vals).any():
                continue
            close_last = vals[-1, 3]
            if close_last <= 0:
                continue
            # Normalize per-sequence
            seq = self.inference._normalize(vals)
            sequences.append(seq)
            codes.append(ts_code)
            close_prices.append(close_last)

        if not sequences:
            return SignalResult(
                signals=pd.DataFrame(
                    columns=["ts_code", "action", "score", "target_price", "rr_ratio"]
                )
            )

        X = np.stack(sequences)
        close_arr = np.array(close_prices)
        batch_results = self.inference.compute_entry_exit_batch(X, close_arr)

        records = []
        for ts_code, sr, close_last in zip(codes, batch_results, close_prices):
            if sr["entry"]:
                entry_price = close_last * (1 + sr["expected_down"])
                records.append(
                    {
                        "ts_code": ts_code,
                        "action": "buy",
                        "score": min(sr["rr_ratio"] * sr["support_confidence"], 1.0),
                        "target_price": round(entry_price, 2),
                        "stop_price": round(entry_price * 0.93, 2),
                        "rr_ratio": round(sr["rr_ratio"], 2),
                        "expected_up": round(sr["expected_up"], 4),
                    }
                )
            if sr["exit"]:
                exit_price = close_last * (1 + sr["expected_up"])
                records.append(
                    {
                        "ts_code": ts_code,
                        "action": "sell",
                        "score": round(sr["resistance_confidence"], 2),
                        "target_price": round(exit_price, 2),
                        "stop_price": 0,
                        "rr_ratio": round(sr["rr_ratio"], 2),
                        "expected_up": 0,
                    }
                )

        return SignalResult(
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
                    "expected_up",
                ]
            ),
            metadata={"model": "transformer_sr", "trade_date": ctx.trade_date},
        )

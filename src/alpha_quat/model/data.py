from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.backtest.filters import build_universe


@dataclass
class DatasetResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train_5: pd.Series
    y_val_5: pd.Series
    y_train_20: pd.Series
    y_val_20: pd.Series
    train_dates: pd.Series
    val_dates: pd.Series
    train_codes: pd.Series
    val_codes: pd.Series


class DatasetBuilder:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _load_features(self, dates: list[str]) -> pd.DataFrame:
        dfs = []
        for d in dates:
            path = self.data_dir / "features" / f"{d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _load_close_series(self, dates: list[str]) -> pd.DataFrame:
        rows = []
        for d in dates:
            path = self.data_dir / "daily" / f"{d[:4]}_{d[4:6]}_{d[6:8]}.parquet"
            if path.exists():
                df = pd.read_parquet(path, columns=["ts_code", "close"])
                df["trade_date"] = d
                rows.append(df)
        if not rows:
            return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
        return pd.concat(rows, ignore_index=True)

    def _get_trade_dates(self) -> pd.Series:
        cal = pd.read_parquet(self.data_dir / "trade_cal.parquet")
        open_dates = cal.loc[cal["is_open"] == 1, "cal_date"]
        return open_dates.sort_values().reset_index(drop=True)

    def _forward_date(
        self, cal_date_series: pd.Series, date_str: str, offset: int
    ) -> str | None:
        positions = np.where(cal_date_series.to_numpy() == date_str)[0]
        if len(positions) == 0:
            return None
        idx = int(positions[0])
        target_idx = idx + offset
        if target_idx >= len(cal_date_series):
            return None
        return cal_date_series.iloc[target_idx]

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        all_dates = df["trade_date"].unique()
        mask = pd.Series(False, index=df.index)
        for d in all_dates:
            universe = build_universe(str(d), self.data_dir)
            date_mask = df["trade_date"] == d
            code_mask = df["ts_code"].isin(list(universe))
            mask |= date_mask & code_mask
        return df.loc[mask].copy()

    def _build_labels(
        self, df: pd.DataFrame, cal_dates: pd.Series, offset: int
    ) -> pd.Series:
        close_map = {}
        for d in df["trade_date"].unique():
            fwd = self._forward_date(cal_dates, str(d), offset)
            if fwd is not None:
                close_path = (
                    self.data_dir / "daily" / f"{fwd[:4]}_{fwd[4:6]}_{fwd[6:8]}.parquet"
                )
                if close_path.exists():
                    close_df = pd.read_parquet(close_path, columns=["ts_code", "close"])
                    for _, row in close_df.iterrows():
                        close_map[(str(d), row["ts_code"])] = row["close"]

        forward_closes = df.apply(
            lambda r: close_map.get((str(r["trade_date"]), r["ts_code"]), np.nan),
            axis=1,
        )
        return forward_closes / df["close"] - 1

    def build(
        self,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        feature_names: list[str] | None = None,
    ) -> DatasetResult:
        cal_dates = self._get_trade_dates()

        max_offset = 20
        cal_arr = cal_dates.to_numpy()
        start_idx = int(np.where(cal_arr >= train_start)[0][0])
        end_idx = int(np.where(cal_arr <= val_end)[0][-1])
        margin_start = max(0, start_idx - max_offset)
        margin_end = min(len(cal_dates) - 1, end_idx + max_offset)

        feature_dates = cal_dates.iloc[margin_start : margin_end + 1].tolist()

        features = self._load_features(feature_dates)

        if features.empty:
            raise ValueError("No feature data found in the specified date range")

        factor_cols = [
            c for c in features.columns if c not in ("ts_code", "trade_date")
        ]
        if feature_names is not None:
            factor_cols = [c for c in factor_cols if c in feature_names]

        close_df = self._load_close_series(feature_dates)
        merged = features.merge(close_df, on=["ts_code", "trade_date"], how="left")
        merged = self._filter_universe(merged)
        merged = merged.dropna(subset=["close"])

        merged["ret_5d"] = self._build_labels(merged, cal_dates, 5)
        merged["ret_20d"] = self._build_labels(merged, cal_dates, 20)

        merged = merged.dropna(subset=["ret_5d", "ret_20d"] + factor_cols)

        train_mask = (merged["trade_date"] >= train_start) & (
            merged["trade_date"] <= train_end
        )
        val_mask = (merged["trade_date"] >= val_start) & (
            merged["trade_date"] <= val_end
        )

        X_train = merged.loc[train_mask, factor_cols].reset_index(drop=True)
        X_val = merged.loc[val_mask, factor_cols].reset_index(drop=True)

        return DatasetResult(
            X_train=X_train,
            X_val=X_val,
            y_train_5=merged.loc[train_mask, "ret_5d"].reset_index(drop=True),
            y_val_5=merged.loc[val_mask, "ret_5d"].reset_index(drop=True),
            y_train_20=merged.loc[train_mask, "ret_20d"].reset_index(drop=True),
            y_val_20=merged.loc[val_mask, "ret_20d"].reset_index(drop=True),
            train_dates=merged.loc[train_mask, "trade_date"].reset_index(drop=True),
            val_dates=merged.loc[val_mask, "trade_date"].reset_index(drop=True),
            train_codes=merged.loc[train_mask, "ts_code"].reset_index(drop=True),
            val_codes=merged.loc[val_mask, "ts_code"].reset_index(drop=True),
        )

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_quat.backtest.filters import _date_to_path


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
        self._main_board_cache: set[str] | None = None

    def _get_main_board(self) -> set[str]:
        if self._main_board_cache is None:
            sb = pd.read_parquet(self.data_dir / "stock_basic.parquet")
            self._main_board_cache = set(sb.loc[sb["market"] == "主板", "ts_code"])
        return self._main_board_cache

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
            path = self.data_dir / "daily" / f"{_date_to_path(d)}.parquet"
            if path.exists():
                df = pd.read_parquet(path, columns=["ts_code", "close"])
                df["trade_date"] = d
                rows.append(df)
        if not rows:
            return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
        return pd.concat(rows, ignore_index=True)

    def _get_trade_dates(self) -> pd.Series:
        cal = pd.read_parquet(self.data_dir / "trade_cal.parquet")
        open_dates = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str)
        return open_dates.sort_values().reset_index(drop=True)

    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        main_board = self._get_main_board()
        all_dates = df["trade_date"].unique()
        mask = pd.Series(False, index=df.index)
        for d in all_dates:
            st_path = self.data_dir / "stock_st" / f"{_date_to_path(d)}.parquet"
            st_codes: set[str] = set()
            if st_path.exists():
                st = pd.read_parquet(st_path)
                st_codes = set(st["ts_code"])
            universe = list(main_board - st_codes)
            mask |= (df["trade_date"] == d) & df["ts_code"].isin(universe)
        return df.loc[mask].copy()

    def _build_labels(
        self,
        df: pd.DataFrame,
        cal_dates: pd.Series,
        offset: int,
        close_df: pd.DataFrame,
    ) -> pd.Series:
        cal_arr = cal_dates.to_numpy()
        date_index = {str(d): i for i, d in enumerate(cal_arr)}
        mapping_rows = []
        for d in df["trade_date"].unique():
            d_str = str(d)
            idx = date_index.get(d_str)
            if idx is not None and idx + offset < len(cal_arr):
                mapping_rows.append((d_str, str(cal_arr[idx + offset])))

        if not mapping_rows:
            return pd.Series([np.nan] * len(df), dtype=float)

        mapping_df = pd.DataFrame(mapping_rows, columns=["trade_date", "fwd_date"])
        fwd_prices = close_df.rename(
            columns={"trade_date": "fwd_date", "close": "fwd_close"}
        )
        label_df = df[["ts_code", "trade_date", "close"]].copy()
        label_df["trade_date"] = label_df["trade_date"].astype(str)
        label_df = label_df.merge(mapping_df, on="trade_date", how="left")
        label_df = label_df.merge(
            fwd_prices,
            left_on=["fwd_date", "ts_code"],
            right_on=["fwd_date", "ts_code"],
            how="left",
        )
        return (label_df["fwd_close"] / label_df["close"] - 1).astype(float)

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

        close_margin_end = min(len(cal_dates) - 1, end_idx + max_offset)
        close_dates = cal_dates.iloc[margin_start : close_margin_end + 1].tolist()

        features = self._load_features(feature_dates)

        if features.empty:
            raise ValueError("No feature data found in the specified date range")

        factor_cols = [
            c
            for c in features.columns
            if c != "ts_code" and not c.startswith("trade_date")
        ]
        if feature_names is not None:
            factor_cols = [c for c in factor_cols if c in feature_names]

        close_df = self._load_close_series(close_dates)
        merged = features.merge(close_df, on=["ts_code", "trade_date"], how="left")
        merged = self._filter_universe(merged)
        merged = merged.dropna(subset=["close"])

        merged["ret_5d"] = self._build_labels(merged, cal_dates, 5, close_df)
        merged["ret_20d"] = self._build_labels(merged, cal_dates, 20, close_df)

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

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_quat.backtest.filters import _date_to_path

logger = logging.getLogger(__name__)


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

    def _get_trade_dates(self) -> pd.Series:
        cal = pd.read_parquet(self.data_dir / "trade_cal.parquet")
        open_dates = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str)
        return open_dates.sort_values().reset_index(drop=True)

    def build(
        self,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        feature_names: list[str] | None = None,
    ) -> DatasetResult:
        cal_dates = self._get_trade_dates()
        cal_arr = cal_dates.to_numpy()

        max_offset = 20
        start_idx = int(np.where(cal_arr >= train_start)[0][0])
        end_idx = int(np.where(cal_arr <= val_end)[0][-1])
        margin_start = max(0, start_idx - max_offset)
        margin_end = min(len(cal_dates) - 1, end_idx + max_offset)

        feature_dates = cal_dates.iloc[margin_start : margin_end + 1].tolist()
        all_start = feature_dates[0]
        all_end = feature_dates[-1]

        # Build forward date mapping
        date_index = {str(d): i for i, d in enumerate(cal_arr)}
        fwd_rows = []
        for i in range(margin_start, margin_end + 1):
            d = str(cal_arr[i])
            idx_5 = i + 5
            idx_20 = i + 20
            fwd_5 = str(cal_arr[idx_5]) if idx_5 < len(cal_arr) else None
            fwd_20 = str(cal_arr[idx_20]) if idx_20 < len(cal_arr) else None
            fwd_rows.append((d, fwd_5, fwd_20))

        fwd_df = pd.DataFrame(fwd_rows, columns=["trade_date", "fwd_5", "fwd_20"])

        # Read stock_basic for main board filter
        sb = pd.read_parquet(self.data_dir / "stock_basic.parquet")
        main_board = sb.loc[sb["market"] == "主板", ["ts_code"]].copy()

        con = duckdb.connect(":memory:")

        features_path = str(self.data_dir / "features" / "*.parquet")
        daily_path = str(self.data_dir / "daily" / "*.parquet")

        stock_st_dir = self.data_dir / "stock_st"
        if stock_st_dir.exists() and list(stock_st_dir.glob("*.parquet")):
            stock_st_path = str(stock_st_dir / "*.parquet")
            st_clause = f"""st_codes AS (
            SELECT DISTINCT ts_code, trade_date
            FROM read_parquet('{stock_st_path}', hive_partitioning=false)
            WHERE trade_date >= '{all_start}'
        ),"""
            st_join = """LEFT JOIN st_codes s ON d.ts_code = s.ts_code AND d.trade_date = s.trade_date
            WHERE s.ts_code IS NULL"""
        else:
            st_clause = ""
            st_join = "WHERE 1=1"

        # Determine factor columns from one sample parquet file
        sample = pd.read_parquet(self.data_dir / "features" / f"{all_start}.parquet")
        all_factor_cols = [
            c
            for c in sample.columns
            if c != "ts_code" and not c.startswith("trade_date")
        ]
        if feature_names is not None:
            factor_cols = [c for c in all_factor_cols if c in feature_names]
        else:
            factor_cols = all_factor_cols

        factor_select = ", ".join(f'"{c}"' for c in factor_cols)
        factor_notnull = " AND ".join(f'"{c}" IS NOT NULL' for c in factor_cols)

        query = f"""
        WITH features AS (
            SELECT * FROM read_parquet('{features_path}', hive_partitioning=false)
            WHERE trade_date >= '{all_start}' AND trade_date <= '{all_end}'
        ),
        daily AS (
            SELECT ts_code, trade_date, close
            FROM read_parquet('{daily_path}', hive_partitioning=false)
            WHERE trade_date >= '{all_start}'
        ),
        fwd_map AS (SELECT * FROM fwd_df),
        main_board AS (SELECT * FROM main_board),
        {st_clause}
        universe AS (
            SELECT d.ts_code, d.trade_date, d.close
            FROM daily d
            INNER JOIN main_board m ON d.ts_code = m.ts_code
            {st_join}
        ),
        base AS (
            SELECT f.*, u.close
            FROM features f
            INNER JOIN universe u ON f.ts_code = u.ts_code AND f.trade_date = u.trade_date
        ),
        with_fwd AS (
            SELECT b.*,
                   fm.fwd_5, fm.fwd_20,
                   d5.close AS close_5,
                   d20.close AS close_20
            FROM base b
            LEFT JOIN fwd_map fm ON b.trade_date = fm.trade_date
            LEFT JOIN daily d5 ON b.ts_code = d5.ts_code AND fm.fwd_5 = d5.trade_date
            LEFT JOIN daily d20 ON b.ts_code = d20.ts_code AND fm.fwd_20 = d20.trade_date
        ),
        labeled AS (
            SELECT *,
                   close_5 / NULLIF(close, 0) - 1 AS ret_5d,
                   close_20 / NULLIF(close, 0) - 1 AS ret_20d
            FROM with_fwd
            WHERE close IS NOT NULL
              AND close_5 IS NOT NULL
              AND close_20 IS NOT NULL
              AND {factor_notnull}
        )
        SELECT ts_code, trade_date, close, ret_5d, ret_20d, {factor_select}
        FROM labeled
        ORDER BY trade_date
        """

        merged = con.execute(query).fetchdf()
        con.close()

        if merged.empty:
            raise ValueError("No feature data found in the specified date range")

        merged["trade_date"] = merged["trade_date"].astype(str)

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

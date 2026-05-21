import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train_5: pd.Series
    y_val_5: pd.Series
    y_train_20: pd.Series
    y_val_20: pd.Series
    y_train_60: pd.Series
    y_val_60: pd.Series
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

        max_offset = 60
        start_idx = int(np.where(cal_arr >= train_start)[0][0])
        end_idx = int(np.where(cal_arr <= val_end)[0][-1])
        margin_start = max(0, start_idx - max_offset)
        margin_end = min(len(cal_dates) - 1, end_idx + max_offset)

        feature_dates = cal_dates.iloc[margin_start : margin_end + 1].tolist()
        all_start = feature_dates[0]
        all_end = feature_dates[-1]

        # Build forward date mapping
        date_index = {str(d): i for i, d in enumerate(cal_arr)}  # noqa: F841
        fwd_rows = []
        for i in range(margin_start, margin_end + 1):
            d = str(cal_arr[i])
            idx_5 = i + 5
            idx_20 = i + 20
            idx_60 = i + 60
            fwd_5 = str(cal_arr[idx_5]) if idx_5 < len(cal_arr) else None
            fwd_20 = str(cal_arr[idx_20]) if idx_20 < len(cal_arr) else None
            fwd_60 = str(cal_arr[idx_60]) if idx_60 < len(cal_arr) else None
            if fwd_5 is not None and fwd_20 is not None and fwd_60 is not None:
                fwd_rows.append((d, fwd_5, fwd_20, fwd_60))

        fwd_df = pd.DataFrame(  # noqa: F841
            fwd_rows, columns=["trade_date", "fwd_5", "fwd_20", "fwd_60"]
        )

        # Read stock_basic for main board filter
        sb = pd.read_parquet(self.data_dir / "stock_basic.parquet")
        main_board = sb.loc[sb["market"] == "主板", ["ts_code"]].copy()  # noqa: F841

        con = duckdb.connect(":memory:")

        features_path = str(self.data_dir / "features" / "*.parquet")
        daily_path = str(self.data_dir / "daily" / "*.parquet")

        stock_st_dir = self.data_dir / "stock_st"
        if stock_st_dir.exists() and list(stock_st_dir.glob("*.parquet")):
            stock_st_path = str(stock_st_dir / "*.parquet")
            st_clause = f"""st_codes AS (
            SELECT DISTINCT CAST(ts_code AS VARCHAR) AS ts_code,
                   CAST(trade_date AS VARCHAR) AS trade_date
            FROM read_parquet('{stock_st_path}', hive_partitioning=false, union_by_name=true)
            WHERE CAST(trade_date AS VARCHAR) >= '{all_start}'
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
            SELECT CAST(ts_code AS VARCHAR) AS ts_code,
                   CAST(trade_date AS VARCHAR) AS trade_date,
                   CAST(open AS DOUBLE) AS open,
                   CAST(high AS DOUBLE) AS high,
                   CAST(low AS DOUBLE) AS low,
                   CAST(close AS DOUBLE) AS close
            FROM read_parquet('{daily_path}', hive_partitioning=false, union_by_name=true)
            WHERE CAST(trade_date AS VARCHAR) >= '{all_start}'
        ),
        fwd_map AS (SELECT * FROM fwd_df),
        main_board AS (SELECT * FROM main_board),
        {st_clause}
        universe AS (
            SELECT d.ts_code, d.trade_date
            FROM daily d
            INNER JOIN main_board m ON d.ts_code = m.ts_code
            {st_join}
        ),
        base AS (
            SELECT f.*
            FROM features f
            INNER JOIN universe u ON f.ts_code = u.ts_code AND f.trade_date = u.trade_date
        ),
        channel_5d AS (
            SELECT b.ts_code, b.trade_date,
                   MIN(d.low) AS min_low_5,
                   MAX(d.high) AS max_high_5,
                   MAX(CASE WHEN d.trade_date = fm.fwd_5 THEN d.close END) AS close_5
            FROM base b
            JOIN fwd_map fm ON b.trade_date = fm.trade_date
            JOIN daily d ON b.ts_code = d.ts_code
                AND d.trade_date >= b.trade_date
                AND d.trade_date <= fm.fwd_5
            GROUP BY b.ts_code, b.trade_date
        ),
        channel_20d AS (
            SELECT b.ts_code, b.trade_date,
                   MIN(d.low) AS min_low_20,
                   MAX(d.high) AS max_high_20,
                   MAX(CASE WHEN d.trade_date = fm.fwd_20 THEN d.close END) AS close_20
            FROM base b
            JOIN fwd_map fm ON b.trade_date = fm.trade_date
            JOIN daily d ON b.ts_code = d.ts_code
                AND d.trade_date >= b.trade_date
                AND d.trade_date <= fm.fwd_20
            GROUP BY b.ts_code, b.trade_date
        ),
        channel_60d AS (
            SELECT b.ts_code, b.trade_date,
                   MIN(d.low) AS min_low_60,
                   MAX(d.high) AS max_high_60,
                   MAX(CASE WHEN d.trade_date = fm.fwd_60 THEN d.close END) AS close_60
            FROM base b
            JOIN fwd_map fm ON b.trade_date = fm.trade_date
            JOIN daily d ON b.ts_code = d.ts_code
                AND d.trade_date >= b.trade_date
                AND d.trade_date <= fm.fwd_60
            GROUP BY b.ts_code, b.trade_date
        ),
        labeled AS (
            SELECT b.*,
                   (c5.close_5 - c5.min_low_5) / NULLIF(c5.max_high_5 - c5.min_low_5, 0) AS ret_5d,
                   (c20.close_20 - c20.min_low_20) / NULLIF(c20.max_high_20 - c20.min_low_20, 0) AS ret_20d,
                   (c60.close_60 - c60.min_low_60) / NULLIF(c60.max_high_60 - c60.min_low_60, 0) AS ret_60d
            FROM base b
            LEFT JOIN channel_5d c5 ON b.ts_code = c5.ts_code AND b.trade_date = c5.trade_date
            LEFT JOIN channel_20d c20 ON b.ts_code = c20.ts_code AND b.trade_date = c20.trade_date
            LEFT JOIN channel_60d c60 ON b.ts_code = c60.ts_code AND b.trade_date = c60.trade_date
            WHERE c5.close_5 IS NOT NULL
              AND c20.close_20 IS NOT NULL
              AND c60.close_60 IS NOT NULL
              AND c5.max_high_5 != c5.min_low_5
              AND c20.max_high_20 != c20.min_low_20
              AND c60.max_high_60 != c60.min_low_60
              AND {factor_notnull}
        )
        SELECT ts_code, trade_date, ret_5d, ret_20d, ret_60d, {factor_select}
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
            y_train_60=merged.loc[train_mask, "ret_60d"].reset_index(drop=True),
            y_val_60=merged.loc[val_mask, "ret_60d"].reset_index(drop=True),
            train_dates=merged.loc[train_mask, "trade_date"].reset_index(drop=True),
            val_dates=merged.loc[val_mask, "trade_date"].reset_index(drop=True),
            train_codes=merged.loc[train_mask, "ts_code"].reset_index(drop=True),
            val_codes=merged.loc[val_mask, "ts_code"].reset_index(drop=True),
        )

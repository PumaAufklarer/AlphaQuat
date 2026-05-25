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
    train_groups: list[int] | None = None
    val_groups: list[int] | None = None


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

        # Find first actual feature file for column detection
        sample = None
        for d in feature_dates:
            sample_path = self.data_dir / "features" / f"{d}.parquet"
            if sample_path.exists():
                sample = pd.read_parquet(sample_path)
                break
        if sample is None:
            raise ValueError("No feature data found in the specified date range")

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

        # Determine factor columns from sample (already loaded above)
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
                   CAST(high AS DOUBLE) AS high,
                   CAST(low AS DOUBLE) AS low,
                   CAST(close AS DOUBLE) AS close
            FROM read_parquet('{daily_path}', hive_partitioning=false, union_by_name=true)
            WHERE CAST(trade_date AS VARCHAR) >= '{all_start}'
        ),
        daily_win AS (
            SELECT *,
                   LEAD(close, 5) OVER w AS close_5,
                   LEAD(close, 20) OVER w AS close_20,
                   LEAD(close, 60) OVER w AS close_60,
                   MIN(low) OVER (w ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING) AS min_low_5,
                   MAX(high) OVER (w ROWS BETWEEN CURRENT ROW AND 5 FOLLOWING) AS max_high_5,
                   MIN(low) OVER (w ROWS BETWEEN CURRENT ROW AND 20 FOLLOWING) AS min_low_20,
                   MAX(high) OVER (w ROWS BETWEEN CURRENT ROW AND 20 FOLLOWING) AS max_high_20,
                   MIN(low) OVER (w ROWS BETWEEN CURRENT ROW AND 60 FOLLOWING) AS min_low_60,
                   MAX(high) OVER (w ROWS BETWEEN CURRENT ROW AND 60 FOLLOWING) AS max_high_60
            FROM daily
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        ),
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
        labeled AS (
            SELECT b.*,
                   (dw.close_5 - dw.min_low_5) / NULLIF(dw.max_high_5 - dw.min_low_5, 0) AS ret_5d,
                   (dw.close_20 - dw.min_low_20) / NULLIF(dw.max_high_20 - dw.min_low_20, 0) AS ret_20d,
                   (dw.close_60 - dw.min_low_60) / NULLIF(dw.max_high_60 - dw.min_low_60, 0) AS ret_60d
            FROM base b
            LEFT JOIN daily_win dw ON b.ts_code = dw.ts_code AND b.trade_date = dw.trade_date
            WHERE dw.close_5 IS NOT NULL
              AND dw.close_20 IS NOT NULL
              AND dw.close_60 IS NOT NULL
              AND dw.max_high_5 != dw.min_low_5
              AND dw.max_high_20 != dw.min_low_20
              AND dw.max_high_60 != dw.min_low_60
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

        # Discretize labels for lambdarank (10 quantile bins per trade_date)
        for col in ["ret_5d", "ret_20d", "ret_60d"]:
            merged[col] = merged.groupby("trade_date")[col].transform(
                lambda g: (
                    pd.qcut(g, 10, labels=False, duplicates="drop")
                    if len(g) >= 10
                    else pd.Categorical(
                        pd.qcut(g, min(len(g), 5), labels=False, duplicates="drop")
                    ).astype(int)
                )
            )
            merged = merged.dropna(subset=[col])
            merged[col] = merged[col].astype(int)

        train_mask = (merged["trade_date"] >= train_start) & (
            merged["trade_date"] <= train_end
        )
        val_mask = (merged["trade_date"] >= val_start) & (
            merged["trade_date"] <= val_end
        )

        X_train = merged.loc[train_mask, factor_cols].reset_index(drop=True)
        X_val = merged.loc[val_mask, factor_cols].reset_index(drop=True)

        train_groups = (
            merged.loc[train_mask].groupby("trade_date", sort=False).size().tolist()
        )
        val_groups = (
            merged.loc[val_mask].groupby("trade_date", sort=False).size().tolist()
        )

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
            train_groups=train_groups,
            val_groups=val_groups,
        )

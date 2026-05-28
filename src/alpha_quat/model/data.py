import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Redundant high-index KLEN/KMID variants — zero-gain from previous experiments
_ZERO_GAIN_FEATURES = {
    "KMID94",
    "KMID95",
    "KMID96",
    "KLEN94",
    "KLEN95",
    "KLEN96",
    "KMID97",
    "KLEN97",
    "KMID98",
    "KLEN98",
    "KMID99",
    "KLEN99",
    "KMID100",
    "KLEN100",
    "KMID101",
}


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
        lambdarank: bool = False,
        n_tile: int = 10,
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
            factor_cols = [c for c in all_factor_cols if c not in _ZERO_GAIN_FEATURES]

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
                   MAX(high) OVER (w ROWS BETWEEN CURRENT ROW AND 20 FOLLOWING) AS max_high_20
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
                   dw.close_60 / NULLIF(dw.close, 0) - 1 AS ret_60d
            FROM base b
            LEFT JOIN daily_win dw ON b.ts_code = dw.ts_code AND b.trade_date = dw.trade_date
            WHERE dw.close_5 IS NOT NULL
              AND dw.close_20 IS NOT NULL
              AND dw.close_60 IS NOT NULL
              AND dw.max_high_5 != dw.min_low_5
              AND dw.max_high_20 != dw.min_low_20
              AND {factor_notnull}
        )
        """
        if lambdarank:
            query += f"""
            , final AS (
              SELECT *,
                CAST(NTILE({n_tile}) OVER (PARTITION BY trade_date ORDER BY ret_5d) - 1 AS INTEGER) AS y5,
                CAST(NTILE({n_tile}) OVER (PARTITION BY trade_date ORDER BY ret_20d) - 1 AS INTEGER) AS y20,
                CAST(NTILE({n_tile}) OVER (PARTITION BY trade_date ORDER BY ret_60d) - 1 AS INTEGER) AS y60
              FROM labeled
            )
            SELECT ts_code, trade_date, y5, y20, y60, ret_5d, ret_20d, ret_60d, {factor_select}
            FROM final
            WHERE y5 >= 0 AND y20 >= 0 AND y60 >= 0
            ORDER BY trade_date
            """
        else:
            query += f"""
            SELECT ts_code, trade_date, ret_5d, ret_20d, ret_60d, {factor_select}
            FROM labeled
            ORDER BY trade_date
            """

        temp_path = self.data_dir / ".temp_build.parquet"
        con.execute(f"COPY ({query}) TO '{temp_path}' (FORMAT PARQUET)")
        con.close()

        merged = pd.read_parquet(temp_path)
        temp_path.unlink()

        if merged.empty:
            raise ValueError("No feature data found in the specified date range")

        merged["trade_date"] = merged["trade_date"].astype(str)

        # --- Industry-relative features: continuous ratios vs industry median ---
        sb = pd.read_parquet(self.data_dir / "stock_basic.parquet")
        industry_map = dict(zip(sb["ts_code"], sb["industry"]))
        merged["industry"] = merged["ts_code"].map(industry_map).fillna("Unknown")

        _IND_FACTORS = ["PE_TTM", "PB", "MV", "TURN", "ROE"]
        for f in _IND_FACTORS:
            if f in merged.columns:
                ind_median = merged.groupby(["trade_date", "industry"], observed=False)[
                    f
                ].transform("median")
                merged[f"{f}_ind"] = merged[f] / (ind_median + 1e-8)
                factor_cols.append(f"{f}_ind")

        # --- Holder number features (shareholder concentration) ---
        # Each stock uses its own latest available quarter (ann_date ≤ trade_date).
        holder_dir = self.data_dir / "holdernumber"
        if holder_dir.exists():
            holder_files = list(holder_dir.glob("*.parquet"))
            if holder_files:
                # Build per-stock: {ts_code: [(ann_date_int, holder_num, prev_holder_num), ...]}
                holder_lookup: dict[str, list[tuple[int, float, float]]] = {}
                for hf in holder_files:
                    code = hf.stem
                    hdf = pd.read_parquet(hf, columns=["ann_date", "holder_num"])
                    hdf = hdf.sort_values("ann_date")
                    entries = []
                    prev = float("nan")
                    for _, row in hdf.iterrows():
                        hn = float(row["holder_num"])
                        entries.append((int(row["ann_date"]), hn, prev))
                        prev = hn
                    if entries:
                        holder_lookup[code] = entries

                codes = merged["ts_code"].values
                td_ints = merged["trade_date"].astype(int).values
                hnums = np.full(len(merged), np.nan)
                hnums_prev = np.full(len(merged), np.nan)

                for i in range(len(merged)):
                    entries = holder_lookup.get(str(codes[i]))
                    if entries:
                        td = td_ints[i]
                        best = None
                        for ann, hn, hp in entries:
                            if ann <= td:
                                best = (hn, hp)
                            else:
                                break
                        if best:
                            hnums[i] = best[0]
                            hnums_prev[i] = best[1]

                merged["holder_num"] = hnums
                merged["holder_num_qoq"] = np.where(
                    ~np.isnan(hnums_prev) & (hnums_prev != 0),
                    (hnums - hnums_prev) / hnums_prev,
                    0.0,
                )
                merged["holder_num_qoq"] = merged["holder_num_qoq"].clip(-0.5, 0.5)

                for f in ["holder_num", "holder_num_qoq"]:
                    factor_cols.append(f)

        # --- Cross-sectional rank: transform all features to [0,1] percentile ---
        merged[factor_cols] = merged.groupby("trade_date")[factor_cols].rank(pct=True)

        # --- Risk-free rate as synthetic baseline asset ---
        # Annualized 2% (China 10Y govt bond proxy). Model learns to rank it
        # against stocks; if RFR ranks in top-K, it signals a bear market.
        if lambdarank and "ret_5d" in merged.columns:
            RFR_ANNUAL = 0.02
            rfr_5d = RFR_ANNUAL * 5 / 252
            rfr_20d = RFR_ANNUAL * 20 / 252
            rfr_60d = RFR_ANNUAL * 60 / 252

            rfr_rows = []
            for td in sorted(merged["trade_date"].unique()):
                td_mask = merged["trade_date"] == td
                td_rets = merged.loc[td_mask, ["ret_5d", "ret_20d", "ret_60d"]]

                # Where does RFR rank in this date's cross-section?
                y5 = int((td_rets["ret_5d"] <= rfr_5d).mean() * n_tile)
                y20 = int((td_rets["ret_20d"] <= rfr_20d).mean() * n_tile)
                y60 = int((td_rets["ret_60d"] <= rfr_60d).mean() * n_tile)

                row = {
                    c: 0.5 for c in factor_cols
                }  # neutral (median) on all dimensions
                row["ts_code"] = "__RFR__"
                row["trade_date"] = td
                row["y5"] = min(y5, n_tile - 1)
                row["y20"] = min(y20, n_tile - 1)
                row["y60"] = min(y60, n_tile - 1)
                rfr_rows.append(row)

            rfr_df = pd.DataFrame(rfr_rows)
            merged = pd.concat([merged, rfr_df], ignore_index=True)
            # Drop raw return columns (no longer needed after RFR label computation)
            merged.drop(
                columns=["ret_5d", "ret_20d", "ret_60d"], inplace=True, errors="ignore"
            )

        # --- Purge/embargo: prevent label leakage at train/val boundary ---
        # Labels look forward up to max_offset (60) trading days. Samples near
        # train_end have labels that overlap with validation data. Purge removes
        # these. Embargo adds a gap to break feature autocorrelation.
        cal_list = cal_dates.tolist()
        embargo_days = 5
        try:
            te_idx = cal_list.index(train_end)
            adjusted_train_end = cal_list[max(0, te_idx - max_offset)]
        except ValueError:
            adjusted_train_end = train_end
        try:
            vs_idx = cal_list.index(val_start)
            adjusted_val_start = cal_list[min(len(cal_list) - 1, vs_idx + embargo_days)]
        except ValueError:
            adjusted_val_start = val_start
        if adjusted_train_end != train_end or adjusted_val_start != val_start:
            logger.info(
                "Purged boundary: train_end %s→%s, val_start %s→%s",
                train_end,
                adjusted_train_end,
                val_start,
                adjusted_val_start,
            )

        train_mask = (merged["trade_date"] >= train_start) & (
            merged["trade_date"] <= adjusted_train_end
        )
        val_mask = (merged["trade_date"] >= adjusted_val_start) & (
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

        if lambdarank:
            return DatasetResult(
                X_train=X_train,
                X_val=X_val,
                y_train_5=merged.loc[train_mask, "y5"].reset_index(drop=True),
                y_val_5=merged.loc[val_mask, "y5"].reset_index(drop=True),
                y_train_20=merged.loc[train_mask, "y20"].reset_index(drop=True),
                y_val_20=merged.loc[val_mask, "y20"].reset_index(drop=True),
                y_train_60=merged.loc[train_mask, "y60"].reset_index(drop=True),
                y_val_60=merged.loc[val_mask, "y60"].reset_index(drop=True),
                train_dates=merged.loc[train_mask, "trade_date"].reset_index(drop=True),
                val_dates=merged.loc[val_mask, "trade_date"].reset_index(drop=True),
                train_codes=merged.loc[train_mask, "ts_code"].reset_index(drop=True),
                val_codes=merged.loc[val_mask, "ts_code"].reset_index(drop=True),
                train_groups=train_groups,
                val_groups=val_groups,
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

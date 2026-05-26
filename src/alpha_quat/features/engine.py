"""FeatureEngine — DuckDB CTE compiler and executor.

Uses a 2-CTE approach for performance:
  - _ts CTE: all time-series expressions + RANK/QUANTILE inner expressions (with WINDOW w_time)
  - _rank CTE: all RANK/QUANTILE outer expressions (no WINDOW, pure cross-sectional ranking)
"""

from pathlib import Path

import duckdb
import pandas as pd

from alpha_quat.features.registry import FactorRegistry
from alpha_quat.features.factor import (
    _unwrap_quantile,
    _unwrap_rank,
    compile,
)


class FeatureEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.conn = duckdb.connect()

    def compute(self, registry: FactorRegistry, trade_date: str) -> pd.DataFrame:
        """Compute factors for a single trade date."""
        result = self.compute_batch(registry, [trade_date])
        return result.get(trade_date, pd.DataFrame())

    def compute_batch(
        self, registry: FactorRegistry, trade_dates: list[str]
    ) -> dict[str, pd.DataFrame]:
        """Compute factors for multiple dates in a single DuckDB query."""
        if not trade_dates:
            return {}

        factors = registry.topological_order()
        lookback = registry.min_lookback()
        margin = int(
            (lookback + 5) * 365 / 252
        )  # convert trading days to calendar days

        min_date = min(trade_dates)
        max_date = max(trade_dates)

        ts_exprs = []  # Time-series: compiled expressions with w_time
        rank_exprs = []  # Cross-sectional: RANK()/NTILE() expressions

        for f in factors:
            inner = _unwrap_rank(f.expression)
            if inner is not None:
                compiled_inner = compile(inner)
                inner_name = f"__{f.name}_inner"
                ts_exprs.append(f"{compiled_inner} AS {inner_name}")
                rank_exprs.append(
                    f"RANK() OVER (PARTITION BY trade_date ORDER BY {inner_name}) AS {f.name}"
                )
                continue

            quantile = _unwrap_quantile(f.expression)
            if quantile is not None:
                inner_expr, n = quantile
                compiled_inner = compile(inner_expr)
                inner_name = f"__{f.name}_inner"
                ts_exprs.append(f"{compiled_inner} AS {inner_name}")
                rank_exprs.append(
                    f"NTILE({n}) OVER (PARTITION BY trade_date ORDER BY {inner_name}) AS {f.name}"
                )
                continue

            compiled = compile(f.expression)
            if "RANK() OVER" in compiled or "NTILE(" in compiled:
                rank_exprs.append(f"{compiled} AS {f.name}")
            else:
                ts_exprs.append(f"{compiled} AS {f.name}")

        sql = self._base_cte_range(min_date, max_date, margin)
        prev = "raw"

        # If any factor needs __rn/__pN/__diff (for EMA/REG_SLOPE/RSI), add helpers
        needs_rn = any(
            "__rn" in e or "__p" in e or "__diff" in e for e in ts_exprs + rank_exprs
        )
        if needs_rn:
            sql += ", _rn AS (\n  SELECT *, ROW_NUMBER() OVER w_time AS __rn,\n"
            sql += "    close - LAG(close, 1) OVER w_time AS __diff\n"
            sql += "  FROM raw\n"
            sql += "  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)\n)"
            # Pre-compute row positions within windows
            positions = [
                f"__rn - MIN(__rn) OVER (w_time ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS __p{w}"
                for w in [5, 10, 12, 14, 20, 26, 30, 60]
            ]
            sql += ", _rp AS (\n  SELECT *, " + ",\n    ".join(positions)
            sql += "\n  FROM _rn\n  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)\n)"
            prev = "_rp"

        if ts_exprs:
            sql += ", _ts AS (\n  SELECT *,\n    "
            sql += ",\n    ".join(ts_exprs)
            sql += f"\n  FROM {prev}\n  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)\n)"
            prev = "_ts"

        if rank_exprs:
            sql += ", _rank AS (\n  SELECT *,\n    "
            sql += ",\n    ".join(rank_exprs)
            sql += f"\n  FROM {prev}\n)"
            prev = "_rank"

        cols = ", ".join(f.name for f in factors)
        sql += f"""
SELECT ts_code, trade_date, {cols}
FROM {prev}
WHERE trade_date BETWEEN '{min_date}' AND '{max_date}'
ORDER BY ts_code, trade_date
"""
        df = self.conn.execute(sql).df()

        result: dict[str, pd.DataFrame] = {}
        for date in trade_dates:
            chunk: pd.DataFrame = df.loc[df["trade_date"] == date]
            if not chunk.empty:
                result[date] = chunk.reset_index(drop=True)

        return result

    def _base_cte_range(self, min_date: str, max_date: str, margin: int) -> str:
        daily_path = self.data_dir / "daily" / "**" / "*.parquet"
        basic_dir = self.data_dir / "daily_basic"

        has_basic = basic_dir.exists() and list(basic_dir.glob("**/*.parquet"))

        if has_basic:
            basic_path = basic_dir / "**" / "*.parquet"
            basic_join = f"""LEFT JOIN (
      SELECT ts_code, trade_date,
        COALESCE(pe, NULL::DOUBLE) AS pe,
        COALESCE(pe_ttm, NULL::DOUBLE) AS pe_ttm,
        COALESCE(pb, NULL::DOUBLE) AS pb,
        COALESCE(total_mv, NULL::DOUBLE) AS total_mv,
        COALESCE(circ_mv, NULL::DOUBLE) AS circ_mv,
        COALESCE(turnover_rate, NULL::DOUBLE) AS turnover_rate,
        COALESCE(volume_ratio, NULL::DOUBLE) AS volume_ratio
      FROM read_parquet('{basic_path}', hive_partitioning=true, union_by_name=true)
    ) db
    ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date"""
            basic_cols = """,
    db.pe,
    db.pe_ttm,
    db.pb,
    db.total_mv,
    db.circ_mv,
    db.turnover_rate,
    db.volume_ratio"""
        else:
            basic_join = ""
            basic_cols = """,
    NULL::DOUBLE AS pe,
    NULL::DOUBLE AS pe_ttm,
    NULL::DOUBLE AS pb,
    NULL::DOUBLE AS total_mv,
    NULL::DOUBLE AS circ_mv,
    NULL::DOUBLE AS turnover_rate,
    NULL::DOUBLE AS volume_ratio"""

        return f"""
WITH raw AS (
  SELECT
    d.ts_code,
    CAST(d.trade_date AS VARCHAR) AS trade_date,
    d.open,
    d.high,
    d.low,
    d.close,
    d.vol AS volume,
    d.amount,
    d.amount / NULLIF(d.vol, 0) AS vwap{basic_cols}
  FROM read_parquet('{daily_path}', hive_partitioning=true) d
  {basic_join}
  WHERE strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d') >= strptime('{min_date}', '%Y%m%d') - INTERVAL {margin} DAY
    AND strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d') <= strptime('{max_date}', '%Y%m%d')
)"""

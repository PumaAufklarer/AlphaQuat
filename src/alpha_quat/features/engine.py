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
        margin = lookback + 5

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
    d.amount / NULLIF(d.vol, 0) AS vwap
  FROM read_parquet('{daily_path}', hive_partitioning=true) d
  WHERE strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d') >= strptime('{min_date}', '%Y%m%d') - INTERVAL {margin} DAY
    AND strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d') <= strptime('{max_date}', '%Y%m%d')
)"""

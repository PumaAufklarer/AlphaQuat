"""FeatureEngine — DuckDB CTE compiler and executor."""

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
        factors = registry.topological_order()
        lookback = registry.min_lookback()
        margin = lookback + 5

        if not factors:
            return pd.DataFrame()

        sql = self._base_cte(trade_date, margin)
        prev = "raw"

        for f in factors:
            inner = _unwrap_rank(f.expression)
            if inner is not None:
                compiled_inner = compile(inner)
                inner_name = f"__{f.name}_inner"
                sql += f""",
cte_{inner_name} AS (
  SELECT *, {compiled_inner} AS {inner_name}
  FROM {prev}
  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)
)"""
                prev = f"cte_{inner_name}"
                sql += f""",
cte_{f.name} AS (
  SELECT *, RANK() OVER (PARTITION BY trade_date ORDER BY {inner_name}) AS {f.name}
  FROM {prev}
)"""
                prev = f"cte_{f.name}"
                continue

            quantile = _unwrap_quantile(f.expression)
            if quantile is not None:
                inner_expr, n = quantile
                compiled_inner = compile(inner_expr)
                inner_name = f"__{f.name}_inner"
                sql += f""",
cte_{inner_name} AS (
  SELECT *, {compiled_inner} AS {inner_name}
  FROM {prev}
  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)
)"""
                prev = f"cte_{inner_name}"
                sql += f""",
cte_{f.name} AS (
  SELECT *, NTILE({n}) OVER (PARTITION BY trade_date ORDER BY {inner_name}) AS {f.name}
  FROM {prev}
)"""
                prev = f"cte_{f.name}"
                continue

            compiled = compile(f.expression)
            sql += f""",
cte_{f.name} AS (
  SELECT *, {compiled} AS {f.name}
  FROM {prev}
  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)
)"""
            prev = f"cte_{f.name}"

        sql += self._final_select(factors, trade_date, prev)
        return self.conn.execute(sql).df()

    def _base_cte(self, trade_date: str, margin: int) -> str:
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
    d.amount
  FROM read_parquet('{daily_path}', hive_partitioning=true) d
  WHERE strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d') >= strptime('{trade_date}', '%Y%m%d') - INTERVAL {margin} DAY
    AND strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d') <= strptime('{trade_date}', '%Y%m%d')
)"""

    def _final_select(self, factors, trade_date: str, prev: str) -> str:
        cols = ", ".join(f.name for f in factors)
        return f"""
SELECT ts_code, trade_date, {cols}
FROM {prev}
WHERE trade_date = '{trade_date}'
ORDER BY ts_code
"""

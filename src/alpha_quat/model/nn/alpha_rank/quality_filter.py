"""Quality filter — industry-relative PE/PB + market cap + liquidity.

Per date: computes industry medians, filters to undervalued quality stocks.
"""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_quality_codes(
    data_dir: Path,
    start: str,
    end: str,
    min_mv: float = 500000,  # 50亿 (circ_mv is in 万 units)
) -> dict[str, set[str]]:
    """For each date in [start, end], return {ts_code} passing quality filter."""
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    ind_map = dict(zip(sb["ts_code"], sb["industry"]))

    # ST by date
    st_dir = data_dir / "stock_st"
    st_by_date: dict[str, set[str]] = defaultdict(set)
    for f in sorted(st_dir.glob("*.parquet")):
        ds = f.stem.replace("_", "")
        if start <= ds <= end:
            for _, r in pd.read_parquet(f).iterrows():
                st_by_date[ds].add(str(r["ts_code"]))

    daily_dir = data_dir / "daily_basic"
    result: dict[str, set[str]] = {}

    for f in sorted(daily_dir.glob("*.parquet")):
        ds = f.stem.replace("_", "")
        if ds < start or ds > end:
            continue

        db = pd.read_parquet(f)
        db["industry"] = db["ts_code"].map(ind_map)
        db = db.dropna(subset=["industry", "pe_ttm", "pb", "circ_mv"])

        if db.empty:
            continue

        # Industry medians
        ind_pe = db.groupby("industry")["pe_ttm"].transform("median")
        ind_pb = db.groupby("industry")["pb"].transform("median")

        # Quality rules:
        mask = (
            (db["circ_mv"] >= min_mv)  # market cap
            & (db["turnover_rate"] >= 0.1)  # liquidity
            & ~db["ts_code"].isin(st_by_date.get(ds, set()))  # non-ST
            & (db["pe_ttm"] > 0)
            & (db["pe_ttm"] < ind_pe * 1.5)  # PE < 150% industry median
            & (
                db["pe_ttm"] >= ind_pe * 0.1
            )  # PE >= 10% industry median (not distressed)
            & (db["pb"] > 0)
            & (db["pb"] < ind_pb * 1.5)  # PB < 150% industry median
        )
        result[ds] = set(db.loc[mask, "ts_code"])

    logger.info(
        "Quality filter: %d dates, avg %.0f stocks/date",
        len(result),
        np.mean([len(v) for v in result.values()]) if result else 0,
    )
    return result

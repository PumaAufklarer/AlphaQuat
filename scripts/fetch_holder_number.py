#!/usr/bin/env python3
"""Fetch all stk_holdernumber via Tushare, save per-stock to data/holdernumber/.

Usage: uv run python3 scripts/fetch_holder_number.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_quat.config import Config
from alpha_quat.data.fetcher import Fetcher

OUTPUT = Path("data/holdernumber")
BATCHES = [
    ("20180101", "20191231"),
    ("20200101", "20211231"),
    ("20220101", "20231231"),
    ("20240101", "20251231"),
]


def main():
    cfg = Config.from_yaml("config.yaml")
    f = Fetcher(token=cfg.token)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    total = 0
    for s, e in BATCHES:
        logger.info("Fetching %s ~ %s", s, e)
        try:
            df = f.query(
                "stk_holdernumber",
                start_date=s,
                end_date=e,
                fields="ts_code,end_date,ann_date,holder_num",
            )
        except Exception as ex:
            logger.warning("Failed %s~%s: %s", s, e, ex)
            continue
        if df is None or df.empty:
            continue
        logger.info("  %d rows", len(df))
        total += len(df)

        for code, grp in df.groupby("ts_code"):
            path = OUTPUT / f"{code}.parquet"
            grp = grp.sort_values("end_date").drop_duplicates(
                subset=["end_date"], keep="last"
            )
            if path.exists():
                old = pd.read_parquet(path)
                grp = pd.concat([old, grp], ignore_index=True)
                grp = grp.sort_values("end_date").drop_duplicates(
                    subset=["end_date"], keep="last"
                )
            grp.to_parquet(path, index=False)

    logger.info(
        "Done. %d rows, %d files in %s",
        total,
        len(list(OUTPUT.glob("*.parquet"))),
        OUTPUT,
    )


if __name__ == "__main__":
    main()

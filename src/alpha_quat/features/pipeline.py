"""FeaturePipeline — date scheduling with incremental/rebuild support."""

import logging
from datetime import date

import pandas as pd

from alpha_quat.features.registry import FactorRegistry

logger = logging.getLogger(__name__)

BATCH_SIZE = 20  # trading dates per DuckDB query


class FeaturePipeline:
    def __init__(self, data_dir, output_dir, engine, writer, metadata):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.engine = engine
        self.writer = writer
        self.metadata = metadata

    @staticmethod
    def _to_iso_date(yyyymmdd: str) -> str:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

    def run(self, registry: FactorRegistry, rebuild=False, since=None) -> dict:
        trade_cal_path = self.data_dir / "trade_cal.parquet"
        if not trade_cal_path.exists():
            msg = "trade_cal.parquet not found. Run 'alpha-quat -s trade_cal' first."
            logger.warning(msg)
            return {"success": 0, "failed": 0, "errors": [], "message": msg}

        cal = pd.read_parquet(trade_cal_path)
        open_dates = sorted(
            cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist()
        )
        today_str = date.today().strftime("%Y%m%d")
        open_dates = [d for d in open_dates if d <= today_str]

        if rebuild:
            self.metadata.delete_since(registry.name, None)
            pending = list(open_dates)
        elif since:
            self.metadata.delete_since(registry.name, self._to_iso_date(since))
            pending = [d for d in open_dates if d >= since]
        else:
            last = self.metadata.get_last_date(registry.name)
            if last:
                last_str = last.strftime("%Y%m%d")
                pending = [d for d in open_dates if d > last_str]
            else:
                pending = list(open_dates)

        lookback = registry.min_lookback()
        if lookback > 0 and not rebuild and not since:
            all_idx = {d: i for i, d in enumerate(open_dates)}
            pending = [d for d in pending if all_idx.get(d, 0) >= lookback]

        results = {"success": 0, "failed": 0, "errors": []}
        total = len(pending)

        for i in range(0, total, BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            logger.info(
                "Batch %d/%d (%s .. %s)",
                i // BATCH_SIZE + 1,
                (total + BATCH_SIZE - 1) // BATCH_SIZE,
                batch[0],
                batch[-1],
            )
            try:
                dfs = self.engine.compute_batch(registry, batch)
                for trade_date, df in dfs.items():
                    path = self.output_dir / f"{trade_date}.parquet"
                    self.writer.merge(df, path)
                    self.metadata.insert(
                        registry.name,
                        self._to_iso_date(trade_date),
                        str(path),
                        len(df),
                    )
                    results["success"] += 1
            except Exception as e:
                logger.error("Batch failed: %s", e)
                results["failed"] += len(batch)
                results["errors"].append(str(e))

        return results

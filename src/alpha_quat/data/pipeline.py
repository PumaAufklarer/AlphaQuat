"""Pipeline that orchestrates data fetching, writing, and metadata tracking."""

import logging
from pathlib import Path

import pandas as pd

from alpha_quat.data.source import DataSource

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        data_dir: Path,
        fetcher,
        metadata,
        writer,
    ):
        self.data_dir = data_dir
        self.fetcher = fetcher
        self.metadata = metadata
        self.writer = writer

    def run(self, sources: list[DataSource]) -> dict:
        results: dict = {
            "full": {"success": 0, "failed": 0},
            "incremental": {"success": 0, "failed": 0},
        }
        for source in sources:
            if source.partition_by == "none":
                try:
                    self.run_full_source(source)
                    results["full"]["success"] += 1
                except Exception as e:
                    results["full"]["failed"] += 1
                    logger.error(f"[{source.api_name}] full source failed: {e}")
            else:
                result = self.run_incremental_source(source)
                if result.get("message"):
                    logger.warning(result["message"])
                results["incremental"]["success"] += result.get("success", 0)
                results["incremental"]["failed"] += result.get("failed", 0)
                if result.get("errors"):
                    for err in result["errors"]:
                        logger.error(err)
        return results

    def run_full_source(self, source: DataSource):
        params = source.get_params()
        df = self.fetcher.query(source.api_name, fields=source.fields, **params)
        path = source.path_for(self.data_dir)
        self.writer.overwrite(df, path)
        self.metadata.insert(
            api_name=source.api_name,
            trade_date=None,
            file_path=str(path),
            row_count=len(df),
        )
        logger.info(f"[{source.api_name}] pulled {len(df)} rows -> {path}")

    def run_incremental_source(self, source: DataSource) -> dict:
        trade_cal_path = self.data_dir / "trade_cal.parquet"
        if not trade_cal_path.exists():
            return {
                "success": 0,
                "failed": 0,
                "message": "trade_cal.parquet not found",
                "errors": [],
            }

        cal_df = pd.read_parquet(trade_cal_path)
        open_dates = sorted(
            cal_df[cal_df["is_open"] == 1]["cal_date"].astype(str).tolist()
        )

        last_date = self.metadata.get_last_date(source.api_name)

        if last_date:
            last_str = last_date.strftime("%Y%m%d")
            pending = [d for d in open_dates if d > last_str]
        else:
            pending = open_dates

        success, failed = 0, 0
        errors = []

        for trade_date in pending:
            try:
                params = source.get_params(trade_date=trade_date)
                df = self.fetcher.query(source.api_name, fields=source.fields, **params)
                date_file = f"{trade_date[:4]}_{trade_date[4:6]}_{trade_date[6:8]}"
                base_dir = source.path_for(self.data_dir, trade_date=None)
                self.writer.write(df, base_dir, trade_date=date_file)
                trade_date_iso = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
                self.metadata.insert(
                    api_name=source.api_name,
                    trade_date=trade_date_iso,
                    file_path=str(base_dir / f"{date_file}.parquet"),
                    row_count=len(df),
                )
                success += 1
                logger.info(f"[{source.api_name}] {trade_date}: {len(df)} rows")
            except Exception as e:
                failed += 1
                errors.append(f"[{source.api_name}] {trade_date}: {e}")
                logger.error(f"[{source.api_name}] {trade_date}: {e}")

        return {"success": success, "failed": failed, "errors": errors}

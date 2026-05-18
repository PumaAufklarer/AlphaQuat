"""CLI entry point for alpha-quat data fetching."""

import argparse
import logging

from alpha_quat.config import Config
from alpha_quat.data.fetcher import Fetcher
from alpha_quat.data.metadata import MetadataManager
from alpha_quat.data.pipeline import Pipeline
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.sources.stock_basic import StockBasicSource
from alpha_quat.data.sources.trade_cal import TradeCalSource
from alpha_quat.data.sources.stk_st import StkStSource
from alpha_quat.data.sources.daily import DailySource
from alpha_quat.data.sources.daily_basic import DailyBasicSource

ALL_SOURCES = {
    "stock_basic": StockBasicSource,
    "trade_cal": TradeCalSource,
    "stk_st": StkStSource,
    "daily": DailySource,
    "daily_basic": DailyBasicSource,
}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch stock data from tushare and store as Parquet"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        choices=list(ALL_SOURCES.keys()) + ["all"],
        default=["all"],
        help="Data sources to pull (default: all)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show data registry summary and exit"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = Config.from_yaml(args.config)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    db_path = config.data_dir / "registry.db"
    metadata = MetadataManager(str(db_path))

    if args.summary:
        rows = metadata.summary()
        if rows:
            print(f"{'api_name':<15} {'count':<8} {'max_date'}")
            print("-" * 40)
            for row in rows:
                print(f"{row[0]:<15} {row[1]:<8} {row[2] or 'N/A'}")
        else:
            print("No data in registry. Run a pull first.")
        return

    fetcher = Fetcher(token=config.token)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=config.data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    names = (
        list(ALL_SOURCES.keys()) if "all" in args.sources else args.sources
    )
    sources = [ALL_SOURCES[name]() for name in names]
    result = pipeline.run(sources)

    # Print summary after run
    print()
    summary_rows = metadata.summary()
    if summary_rows:
        print(f"{'api_name':<15} {'count':<8} {'max_date'}")
        print("-" * 40)
        for row in summary_rows:
            print(f"{row[0]:<15} {row[1]:<8} {row[2] or 'N/A'}")

"""CLI entry point for alpha-quat data fetching and feature engineering."""

import argparse
import logging

from alpha_quat.config import Config
from alpha_quat.data.fetcher import Fetcher
from alpha_quat.data.metadata import MetadataManager
from alpha_quat.data.pipeline import Pipeline
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.sources.stock_basic import StockBasicSource
from alpha_quat.data.sources.trade_cal import TradeCalSource
from alpha_quat.data.sources.stock_st import StockStSource
from alpha_quat.data.sources.daily import DailySource
from alpha_quat.data.sources.daily_basic import DailyBasicSource

ALL_SOURCES = {
    "stock_basic": StockBasicSource,
    "trade_cal": TradeCalSource,
    "stock_st": StockStSource,
    "daily": DailySource,
    "daily_basic": DailyBasicSource,
}

ALL_FEATURE_SETS = {
    "alpha158": "alpha_quat.features.alphasets.alpha158:build_alpha158",
}


def _build_fetch_parser(subparsers):
    parser = subparsers.add_parser(
        "fetch", help="Fetch raw data from tushare"
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        choices=list(ALL_SOURCES.keys()) + ["all"],
        default=["all"],
        help="Data sources to pull (default: all)",
    )
    return parser


def _build_feature_parser(subparsers):
    parser = subparsers.add_parser(
        "feature", help="Compute feature factors from raw data"
    )
    parser.add_argument(
        "-f",
        "--factor-set",
        choices=list(ALL_FEATURE_SETS.keys()),
        default="alpha158",
        help="Factor set to compute (default: alpha158)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete all computed factors and recompute from scratch"
    )
    parser.add_argument(
        "--since",
        help="Recompute factors from this date (YYYYMMDD) onward"
    )
    return parser


def _cmd_fetch(args, config, metadata):
    fetcher = Fetcher(token=config.token)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=config.data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )
    names = list(ALL_SOURCES.keys()) if "all" in args.sources else args.sources
    sources = [ALL_SOURCES[name]() for name in names]
    pipeline.run(sources)

    print()
    summary_rows = metadata.summary()
    if summary_rows:
        print(f"{'api_name':<15} {'count':<8} {'max_date'}")
        print("-" * 40)
        for row in summary_rows:
            print(f"{row[0]:<15} {row[1]:<8} {row[2] or 'N/A'}")


def _cmd_feature(args, config, metadata):
    from importlib import import_module

    from alpha_quat.features.engine import FeatureEngine
    from alpha_quat.features.pipeline import FeaturePipeline

    module_path, func_name = ALL_FEATURE_SETS[args.factor_set].split(":")
    module = import_module(module_path)
    registry = getattr(module, func_name)()

    engine = FeatureEngine(data_dir=config.data_dir)
    writer = ParquetWriter()
    output_dir = config.data_dir / "features"

    pipeline = FeaturePipeline(
        data_dir=config.data_dir,
        output_dir=output_dir,
        engine=engine,
        writer=writer,
        metadata=metadata,
    )
    result = pipeline.run(registry, rebuild=args.rebuild, since=args.since)

    print()
    print(f"Factor set: {registry.name}")
    print(f"  Success: {result['success']}")
    print(f"  Failed:  {result['failed']}")
    if result["errors"]:
        for err in result["errors"]:
            print(f"  Error: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="alpha-quat: stock data fetching and feature engineering"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show data registry summary and exit"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command")
    _build_fetch_parser(subparsers)
    _build_feature_parser(subparsers)

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

    if args.command == "feature":
        _cmd_feature(args, config, metadata)
    else:
        _cmd_fetch(args, config, metadata)

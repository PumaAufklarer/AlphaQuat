"""CLI entry point for alpha-quat data fetching and feature engineering."""

import argparse
import logging
from pathlib import Path

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
from alpha_quat.backtest.config import BacktestConfig
from alpha_quat.backtest.engine import BacktestEngine
from alpha_quat.backtest.report import generate_html_report
from alpha_quat.experiment.config import ExperimentConfig as ExpConfig
from alpha_quat.experiment.registry import ExperimentRegistry
from alpha_quat.model.lightgbm.pipeline import run_variant

ALL_SOURCES = {
    "stock_basic": StockBasicSource,
    "trade_cal": TradeCalSource,
    "stock_st": StockStSource,
    "daily": DailySource,
    "daily_basic": DailyBasicSource,
}

ALL_FEATURE_SETS = {
    "alpha158": "alpha_quat.features.alphasets.alpha158:build_alpha158",
    "alpha_ext": "alpha_quat.features.alphasets.alpha_ext:build_alpha_ext",
    "alpha_fund": "alpha_quat.features.alphasets.alpha_fund:build_alpha_fund",
    "alpha_combined": "alpha_quat.features.alphasets:build_alpha_combined",
}


def _build_fetch_parser(subparsers):
    parser = subparsers.add_parser("fetch", help="Fetch raw data from tushare")
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
        "--rebuild",
        action="store_true",
        help="Delete all computed factors and recompute from scratch",
    )
    parser.add_argument(
        "--since", help="Recompute factors from this date (YYYYMMDD) onward"
    )
    return parser


def _build_backtest_parser(subparsers):
    parser = subparsers.add_parser("backtest", help="Run strategy backtest")
    parser.add_argument("--start", default="20220501", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20260501", help="End date YYYYMMDD")
    parser.add_argument("--capital", type=float, default=20000, help="Initial capital")
    parser.add_argument("--monthly", type=float, default=8000, help="Monthly addition")
    parser.add_argument(
        "--commission", type=float, default=0.0005, help="Commission rate"
    )
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop loss pct")
    parser.add_argument("--top-k", type=int, default=5, help="Max holdings")
    parser.add_argument("--output", default=None, help="HTML report output path")
    parser.add_argument(
        "--model-dir", default=None, help="Path to model directory for ML backtest"
    )
    parser.add_argument(
        "--rebalance-interval",
        type=int,
        default=5,
        help="Trading days between rebalances (5=weekly, 10=bi-weekly)",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=0.40,
        help="Sell stocks outside Top-K only if score < threshold (default 0.40)",
    )
    parser.add_argument(
        "--daily-monitor",
        action="store_true",
        help="Enable daily monitoring mode (sell weak, buy best)",
    )
    parser.add_argument(
        "--sell-score-percentile",
        type=float,
        default=None,
        help="Sell holdings scoring below this percentile (e.g. 0.50=below median)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Min confidence (0-1) for CI-based sell decisions (requires quantile models)",
    )
    parser.add_argument(
        "--sell-upper",
        type=float,
        default=0.35,
        help="Sell if confident AND upper bound < this threshold (default 0.35)",
    )
    parser.add_argument(
        "--weighting",
        default="equal",
        choices=["equal", "vol_parity", "score_momentum", "kelly"],
        help="Position sizing strategy (default: equal)",
    )
    parser.add_argument(
        "--experiment", default=None, help="Experiment name for ML signal"
    )
    return parser


def _build_predict_parser(subparsers):
    parser = subparsers.add_parser(
        "predict", help="Pull data, score stocks, show top picks"
    )
    parser.add_argument("--holdings", default=None, help="Path to holdings YAML file")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top picks to show"
    )
    parser.add_argument(
        "--experiment", default=None, help="Experiment name for model loading"
    )
    return parser


def _cmd_predict(args, config):
    from alpha_quat.model.predict import predict as run_predict

    logger = logging.getLogger(__name__)

    # Fetch latest data
    from alpha_quat.data.writer import ParquetWriter
    from alpha_quat.data.sources.daily import DailySource
    from alpha_quat.data.sources.daily_basic import DailyBasicSource

    fetcher = Fetcher(token=config.token)
    writer = ParquetWriter()

    db_path = config.data_dir / "registry.db"
    from alpha_quat.data.metadata import MetadataManager

    metadata = MetadataManager(str(db_path))

    pipeline = Pipeline(
        data_dir=config.data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )
    logger.info("Pulling daily and daily_basic data...")
    pipeline.run([DailySource(), DailyBasicSource()])

    # Compute features (incremental)
    from alpha_quat.features.engine import FeatureEngine
    from alpha_quat.features.pipeline import FeaturePipeline
    from alpha_quat.features.alphasets.alpha158 import build_alpha158

    engine = FeatureEngine(data_dir=config.data_dir)
    output_dir = config.data_dir / "features"
    feat_pipeline = FeaturePipeline(
        data_dir=config.data_dir,
        output_dir=output_dir,
        engine=engine,
        writer=writer,
        metadata=metadata,
    )
    registry = build_alpha158()
    logger.info("Computing features...")
    feat_pipeline.run(registry)

    # Load holdings
    holdings = None
    if args.holdings:
        import yaml

        with open(args.holdings) as f:
            data = yaml.safe_load(f)
        holdings = data.get("holdings", [])

    if args.experiment:
        logger.info("Using experiment: %s", args.experiment)

    # Predict
    run_predict(config.data_dir, holdings=holdings, top_k=args.top_k)


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


def _cmd_backtest(args, config):
    model_dir = args.model_dir or config.data_dir / "models"

    cfg = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        monthly_addition=args.monthly,
        commission_rate=args.commission,
        stop_loss_pct=args.stop_loss,
        top_k=args.top_k,
        model_dir=str(model_dir) if args.model_dir or model_dir.exists() else None,
        experiment_name=args.experiment,
        rebalance_interval=args.rebalance_interval,
        sell_threshold=args.sell_threshold,
        daily_monitor=args.daily_monitor,
        sell_score_percentile=args.sell_score_percentile,
        confidence_threshold=args.confidence_threshold,
        sell_upper_threshold=args.sell_upper,
        weighting_strategy=args.weighting,
    )
    engine = BacktestEngine(cfg, config.data_dir)
    result = engine.run()

    metrics = result["metrics"]
    print()
    print("=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Period:          {cfg.start_date} ~ {cfg.end_date}")
    print(f"  Total Invested:  {metrics['total_invested']:,.0f}")
    print(f"  Final Value:     {metrics['final_value']:,.2f}")
    print(f"  Cumulative Ret:  {metrics['cumulative_return'] * 100:+.2f}%")
    print(f"  Annualized Ret:  {metrics['annualized_return'] * 100:+.2f}%")
    print(f"  Max Drawdown:    {metrics['max_drawdown'] * 100:.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate:        {metrics['win_rate'] * 100:.1f}%")
    print(f"  Total Trades:    {metrics['total_trades']}")
    print("=" * 50)
    print()

    output_path = (
        Path(args.output) if args.output else config.data_dir / "backtest_report.html"
    )
    generate_html_report(result, cfg, output_path)
    print(f"Report saved to: {output_path}")


def _build_experiment_parser(subparsers):
    parser = subparsers.add_parser("experiment", help="Manage experiments")
    exp_sub = parser.add_subparsers(dest="exp_command")
    exp_sub.add_parser("list", help="List all experiments")
    show_parser = exp_sub.add_parser("show", help="Show experiment details")
    show_parser.add_argument("name", help="Experiment name")
    return parser


def _cmd_experiment(args, config):
    reg = ExperimentRegistry(config.data_dir)
    if args.exp_command == "list":
        experiments = reg.list_experiments()
        if not experiments:
            print("No experiments found.")
            return
        print(f"{'Name':<30} {'Mode':<15} {'Created':<20}")
        print("-" * 65)
        for exp in experiments:
            print(f"{exp['name']:<30} {exp['mode']:<15} {exp['created']:<20}")
    elif args.exp_command == "show":
        found = reg.find(args.name)
        if found is None:
            print(f"Experiment '{args.name}' not found.")
            return
        print(f"Name:    {found['name']}")
        print(f"Mode:    {found['mode']}")
        print(f"Created: {found['created']}")
        exp_dir = config.data_dir / "models" / "experiments" / args.name
        if exp_dir.exists():
            model_files = list(exp_dir.glob("*.txt"))
            print(f"Models:  {len(model_files)} files")
            results_path = exp_dir / "results.json"
            if results_path.exists():
                print(f"Results: {results_path}")


def _build_sr_cache_parser(subparsers):
    parser = subparsers.add_parser(
        "sr-cache", help="Pre-compute Alpha360 + support/resistance labels"
    )
    return parser


def _cmd_sr_cache(args, config):
    from alpha_quat.data.sr_cache import build_cache

    written = build_cache(config.data_dir)
    print(
        f"Alpha360 cache: {written} date files written to "
        f"{config.data_dir / 'alpha360'}"
    )


def _build_model_parser(subparsers):
    model_parser = subparsers.add_parser("model", help="Train ML models")
    model_sub = model_parser.add_subparsers(dest="model_type")

    lgb_parser = model_sub.add_parser("lightgbm", help="LightGBM stock selection model")
    lgb_parser.add_argument(
        "variant",
        choices=["regression", "quantile", "lambdarank"],
        help="Model variant to train",
    )
    lgb_parser.add_argument(
        "--name", required=True, help="Experiment name (e.g. exp_quantile_v2)"
    )
    lgb_parser.add_argument(
        "--train-start", default="20240401", help="Train start YYYYMMDD"
    )
    lgb_parser.add_argument(
        "--train-end", default="20250430", help="Train end YYYYMMDD"
    )
    lgb_parser.add_argument(
        "--val-start", default="20250501", help="Validation start YYYYMMDD"
    )
    lgb_parser.add_argument(
        "--val-end", default="20260430", help="Validation end YYYYMMDD"
    )
    lgb_parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    lgb_parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning")
    lgb_parser.add_argument(
        "--features", default=None, help="Comma-separated feature subset"
    )
    return_parser = lgb_parser  # noqa

    nn_parser = model_sub.add_parser("nn", help="Neural network models")
    nn_parser.add_argument(
        "nn_type",
        choices=["sr_transformer"],
        help="NN variant to train",
    )
    nn_parser.add_argument("--name", required=True, help="Experiment name")
    nn_parser.add_argument(
        "--train-start", default="20200101", help="Train start YYYYMMDD"
    )
    nn_parser.add_argument("--train-end", default="20231231", help="Train end YYYYMMDD")
    nn_parser.add_argument(
        "--val-start", default="20240101", help="Validation start YYYYMMDD"
    )
    nn_parser.add_argument(
        "--val-end", default="20240630", help="Validation end YYYYMMDD"
    )
    nn_parser.add_argument(
        "--tune", action="store_true", help="Run grid search over hyperparams"
    )

    return model_parser


def _cmd_model(args, config):
    if args.model_type == "lightgbm":
        feature_names = None
        if args.features:
            feature_names = [f.strip() for f in args.features.split(",") if f.strip()]

        quantile_alphas = [0.1, 0.5, 0.9] if args.variant == "quantile" else None

        exp_cfg = ExpConfig(
            name=args.name,
            mode=args.variant,
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            n_trials=args.trials,
            tune=not args.no_tune,
            feature_names=feature_names,
            quantile_alphas=quantile_alphas,
        )
        run_variant(config.data_dir, exp_cfg)
        print(f"\nExperiment '{args.name}' completed successfully.")
        print(
            f"Models saved to: {config.data_dir / 'models' / 'experiments' / args.name}"
        )
    elif args.model_type == "nn":
        exp_cfg = ExpConfig(
            name=args.name,
            mode=args.nn_type,
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
        )
        if args.tune:
            from alpha_quat.model.nn.tune import grid_search

            grid_search(config.data_dir, exp_cfg)
        else:
            from alpha_quat.model.nn.pipeline import run_variant_nn

            run_variant_nn(config.data_dir, exp_cfg)
        print(f"\nNN Experiment '{args.name}' completed successfully.")
        print(
            f"Models saved to: {config.data_dir / 'models' / 'experiments' / args.name}"
        )
    else:
        print(f"Unknown model type: {args.model_type}. Available: lightgbm, nn")


def _cmd_fetch(args, config, metadata):
    from alpha_quat.data.writer import ParquetWriter

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
    _build_backtest_parser(subparsers)
    _build_model_parser(subparsers)
    _build_predict_parser(subparsers)
    _build_experiment_parser(subparsers)
    _build_sr_cache_parser(subparsers)

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
    elif args.command == "backtest":
        _cmd_backtest(args, config)
    elif args.command == "model":
        _cmd_model(args, config)
    elif args.command == "predict":
        _cmd_predict(args, config)
    elif args.command == "experiment":
        _cmd_experiment(args, config)
    elif args.command == "sr-cache":
        _cmd_sr_cache(args, config)
    else:
        _cmd_fetch(args, config, metadata)

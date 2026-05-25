from dataclasses import dataclass


@dataclass
class BacktestConfig:
    start_date: str = "20220501"
    end_date: str = "20260501"
    initial_capital: float = 20000
    monthly_addition: float = 8000
    commission_rate: float = 0.0005
    min_commission: float = 0.0
    stop_loss_pct: float = 0.15
    short_factor: str = "KLEN35"
    long_factor: str = "KLEN36"
    top_k: int = 5
    benchmark: str | None = None
    model_dir: str | None = None
    rebalance_interval: int = 5  # trading days between rebalances
    sell_threshold: float | None = (
        0.40  # None=sell all out-of-top-K; 0.40=only if score<0.40
    )
    daily_monitor: bool = False
    sell_score_percentile: float | None = None
    confidence_threshold: float | None = (
        None  # min confidence (0-1) for CI-based daily mode
    )
    sell_upper_threshold: float = 0.35
    weighting_strategy: str = "equal"  # equal|vol_parity|score_momentum|kelly

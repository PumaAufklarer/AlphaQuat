from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StrategyContext:
    trade_date: str
    capital: float
    universe: list[str] | None = None
    prices: pd.DataFrame | None = None
    prev_holdings: pd.DataFrame | None = None
    constraints: dict | None = None


@dataclass
class SignalResult:
    signals: pd.DataFrame
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyResult:
    target_positions: pd.DataFrame
    orders: pd.DataFrame
    metadata: dict = field(default_factory=dict)

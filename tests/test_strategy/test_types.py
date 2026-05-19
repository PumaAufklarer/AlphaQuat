import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult, StrategyResult


class TestStrategyContext:
    def test_create_minimal(self):
        ctx = StrategyContext(trade_date="20240115", capital=1000000.0)
        assert ctx.trade_date == "20240115"
        assert ctx.capital == 1000000.0
        assert ctx.universe is None
        assert ctx.prices is None
        assert ctx.prev_holdings is None
        assert ctx.constraints is None

    def test_create_full(self):
        prev = pd.DataFrame(
            {"ts_code": ["000001.SZ"], "shares": [1000], "cost": [10.5]}
        )
        prices = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]})
        ctx = StrategyContext(
            trade_date="20240115",
            capital=1000000.0,
            universe=["000001.SZ", "000002.SZ"],
            prices=prices,
            prev_holdings=prev,
            constraints={"max_single_weight": 0.05},
        )
        assert ctx.universe == ["000001.SZ", "000002.SZ"]
        assert len(ctx.prices) == 1
        assert len(ctx.prev_holdings) == 1
        assert ctx.constraints["max_single_weight"] == 0.05


class TestSignalResult:
    def test_create(self):
        signals = pd.DataFrame(
            {"ts_code": ["000001.SZ", "000002.SZ"], "score": [0.8, 0.3]}
        )
        result = SignalResult(signals=signals, metadata={"model": "factor_weighted"})
        assert len(result.signals) == 2
        assert list(result.signals.columns) == ["ts_code", "score"]
        assert result.metadata["model"] == "factor_weighted"


class TestStrategyResult:
    def test_create(self):
        positions = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "target_weight": [0.05],
                "target_shares": [500],
                "target_amount": [50000.0],
            }
        )
        orders = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "action": ["buy"],
                "delta_shares": [500],
                "delta_amount": [50000.0],
            }
        )
        result = StrategyResult(
            target_positions=positions,
            orders=orders,
            metadata={"signal": {"model": "test"}},
        )
        assert result.target_positions is positions
        assert result.orders is orders
        assert result.metadata["signal"]["model"] == "test"

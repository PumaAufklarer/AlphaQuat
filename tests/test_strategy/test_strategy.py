import pandas as pd

from alpha_quat.strategy.types import StrategyContext, SignalResult, StrategyResult
from alpha_quat.strategy.signal import ISignalGenerator
from alpha_quat.strategy.position import IPositionManager
from alpha_quat.strategy.strategy import Strategy


class MockSignal(ISignalGenerator):
    def generate(self, features, ctx):
        return SignalResult(
            signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]}),
            metadata={"signal_name": "mock"},
        )


class MockPosition(IPositionManager):
    def allocate(self, signals, ctx):
        return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

    def constrain(self, positions, ctx):
        return positions

    def execute(self, target, prev, ctx):
        positions = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "target_weight": [0.1],
                "target_shares": [100],
                "target_amount": [10000.0],
            }
        )
        orders = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "action": ["buy"],
                "delta_shares": [100],
                "delta_amount": [10000.0],
            }
        )
        return positions, orders


class TestStrategy:
    def test_run_returns_strategy_result(self):
        strategy = Strategy(signal=MockSignal(), position=MockPosition())
        features = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240115"],
                "factor_001": [0.5],
            }
        )
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        result = strategy.run(features, ctx)
        assert isinstance(result, StrategyResult)
        assert len(result.target_positions) == 1
        assert len(result.orders) == 1
        assert result.metadata["signal"] == {"signal_name": "mock"}

    def test_pipeline_order_is_correct(self):
        call_order = []

        class OrderedSignal(ISignalGenerator):
            def generate(self, features, ctx):
                call_order.append("signal")
                return SignalResult(
                    signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]})
                )

        class OrderedPosition(IPositionManager):
            def allocate(self, signals, ctx):
                call_order.append("allocate")
                return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

            def constrain(self, positions, ctx):
                call_order.append("constrain")
                return positions

            def execute(self, target, prev, ctx):
                call_order.append("execute")
                positions = target.assign(target_shares=100, target_amount=10000.0)
                orders = pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "action": ["buy"],
                        "delta_shares": [100],
                        "delta_amount": [10000.0],
                    }
                )
                return positions, orders

        strategy = Strategy(signal=OrderedSignal(), position=OrderedPosition())
        features = pd.DataFrame(
            {"ts_code": ["000001.SZ"], "trade_date": ["20240115"], "factor_001": [0.5]}
        )
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        strategy.run(features, ctx)
        assert call_order == ["signal", "allocate", "constrain", "execute"]

    def test_dependency_injection(self):
        sig_a = MockSignal()
        pos_a = MockPosition()
        sig_b = MockSignal()
        strategy = Strategy(signal=sig_a, position=pos_a)
        assert strategy.signal is sig_a
        assert strategy.position is pos_a
        strategy.signal = sig_b
        assert strategy.signal is sig_b

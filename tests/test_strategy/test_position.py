import pytest
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.position import IPositionManager


class TestIPositionManager:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            IPositionManager()  # pyright: ignore[reportAbstractUsage]

    def test_concrete_subclass_works(self):
        class MyPosition(IPositionManager):
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

        pm = MyPosition()
        signals = SignalResult(
            signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]})
        )
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        pos = pm.allocate(signals, ctx)
        assert list(pos.columns) == ["ts_code", "target_weight"]

        constrained = pm.constrain(pos, ctx)
        assert list(constrained.columns) == ["ts_code", "target_weight"]

        pos, orders = pm.execute(constrained, None, ctx)
        assert isinstance(pos, pd.DataFrame)
        assert isinstance(orders, pd.DataFrame)
        assert "target_shares" in pos.columns
        assert "action" in orders.columns

    def test_execute_with_prev_none(self):
        class MyPosition(IPositionManager):
            def allocate(self, signals, ctx):
                return pd.DataFrame({"ts_code": ["000001.SZ"], "target_weight": [0.1]})

            def constrain(self, positions, ctx):
                return positions

            def execute(self, target, prev, ctx):
                assert prev is None
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

        pm = MyPosition()
        signals = SignalResult(
            signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]})
        )
        ctx = StrategyContext(trade_date="20240115", capital=100000.0)
        pos = pm.allocate(signals, ctx)
        pos = pm.constrain(pos, ctx)
        pos, orders = pm.execute(pos, None, ctx)
        assert orders["action"].iloc[0] == "buy"

    def test_missing_method_raises(self):
        with pytest.raises(TypeError):

            class BadPosition1(IPositionManager):
                def allocate(self, signals, ctx) -> pd.DataFrame:
                    return pd.DataFrame()

                def constrain(self, positions, ctx) -> pd.DataFrame:
                    return positions

            BadPosition1()  # pyright: ignore[reportAbstractUsage]

        with pytest.raises(TypeError):

            class BadPosition2(IPositionManager):
                def allocate(self, signals, ctx) -> pd.DataFrame:
                    return pd.DataFrame()

                def execute(
                    self, target, prev, ctx
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
                    return pd.DataFrame(), pd.DataFrame()

            BadPosition2()  # pyright: ignore[reportAbstractUsage]

        with pytest.raises(TypeError):

            class BadPosition3(IPositionManager):
                def constrain(self, positions, ctx) -> pd.DataFrame:
                    return positions

                def execute(
                    self, target, prev, ctx
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
                    return pd.DataFrame(), pd.DataFrame()

            BadPosition3()  # pyright: ignore[reportAbstractUsage]

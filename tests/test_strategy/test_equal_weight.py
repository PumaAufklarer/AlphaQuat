import pytest
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.positions.equal_weight import EqualWeightTopKPosition


class TestEqualWeightTopKPosition:
    def test_allocate_top_k(self):
        pm = EqualWeightTopKPosition(top_k=3)
        signals = SignalResult(
            signals=pd.DataFrame(
                {
                    "ts_code": ["A", "B", "C", "D"],
                    "action": ["buy"] * 4,
                    "score": [0.9, 0.8, 0.7, 0.6],
                }
            )
        )
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 3
        assert pos["target_weight"].sum() == pytest.approx(1.0)

    def test_allocate_fewer_than_k(self):
        pm = EqualWeightTopKPosition(top_k=5)
        signals = SignalResult(
            signals=pd.DataFrame(
                {"ts_code": ["A", "B"], "action": ["buy", "buy"], "score": [0.9, 0.8]}
            )
        )
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 2
        assert pos["target_weight"].iloc[0] == pytest.approx(0.529, abs=0.01)
        assert pos["target_weight"].sum() == pytest.approx(1.0)

    def test_allocate_sells_ignored(self):
        pm = EqualWeightTopKPosition(top_k=3)
        signals = SignalResult(
            signals=pd.DataFrame(
                {"ts_code": ["A", "B"], "action": ["buy", "sell"], "score": [0.9, 0.0]}
            )
        )
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 1
        assert pos["ts_code"].iloc[0] == "A"

    def test_allocate_no_buys(self):
        pm = EqualWeightTopKPosition(top_k=3)
        signals = SignalResult(
            signals=pd.DataFrame({"ts_code": ["A"], "action": ["sell"], "score": [0.0]})
        )
        pos = pm.allocate(signals, StrategyContext(trade_date="d", capital=1e5))
        assert len(pos) == 0

    def test_constrain_universe(self):
        pm = EqualWeightTopKPosition(top_k=3)
        pos = pd.DataFrame(
            {"ts_code": ["A", "B", "C"], "target_weight": [0.4, 0.3, 0.3]}
        )
        ctx = StrategyContext(trade_date="d", capital=1e5, universe=["A", "C"])
        result = pm.constrain(pos, ctx)
        assert set(result["ts_code"]) == {"A", "C"}
        assert result["target_weight"].sum() == pytest.approx(1.0)

    def test_execute_shares(self):
        pm = EqualWeightTopKPosition(top_k=3)
        target = pd.DataFrame({"ts_code": ["A", "B"], "target_weight": [0.6, 0.4]})
        ctx = StrategyContext(
            trade_date="d",
            capital=100000.0,
            prices=pd.DataFrame({"ts_code": ["A", "B"], "open": [10.0, 20.0]}),
        )
        pos, orders = pm.execute(target, None, ctx)
        assert pos.loc[pos["ts_code"] == "A", "target_shares"].iloc[0] == 6000
        assert pos.loc[pos["ts_code"] == "B", "target_shares"].iloc[0] == 2000
        assert len(orders) == 2

    def test_execute_rounding(self):
        pm = EqualWeightTopKPosition(top_k=3)
        target = pd.DataFrame({"ts_code": ["A"], "target_weight": [1.0]})
        ctx = StrategyContext(
            trade_date="d",
            capital=1500.0,
            prices=pd.DataFrame({"ts_code": ["A"], "open": [10.0]}),
        )
        pos, _ = pm.execute(target, None, ctx)
        assert pos["target_shares"].iloc[0] == 100

    def test_empty_target(self):
        pm = EqualWeightTopKPosition(top_k=3)
        target = pd.DataFrame(columns=["ts_code", "target_weight"])
        pos, orders = pm.execute(
            target, None, StrategyContext(trade_date="d", capital=1e5)
        )
        assert len(pos) == 0
        assert len(orders) == 0

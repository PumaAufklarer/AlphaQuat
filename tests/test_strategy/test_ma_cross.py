import pandas as pd
from alpha_quat.strategy.types import StrategyContext
from alpha_quat.strategy.signals.ma_cross import MACrossSignal


class TestMACrossSignal:
    def test_no_prev_returns_empty(self):
        signal = MACrossSignal()
        features = pd.DataFrame(
            {"ts_code": ["A", "B"], "KLEN35": [1.05, 0.95], "KLEN36": [1.00, 1.00]}
        )
        ctx = StrategyContext(trade_date="d", capital=1e5)
        result = signal.generate(features, ctx)
        assert len(result.signals) == 0

    def test_golden_cross_buy(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [0.98], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.02], "KLEN36": [1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 1
        row = result.signals.iloc[0]
        assert row["ts_code"] == "A"
        assert row["action"] == "buy"
        assert row["score"] == 1.0

    def test_dead_cross_sell(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.02], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({"ts_code": ["A"], "KLEN35": [0.98], "KLEN36": [1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 1
        assert result.signals.iloc[0]["action"] == "sell"
        assert result.signals.iloc[0]["score"] == 0.0

    def test_no_cross_no_signal(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.05], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.06], "KLEN36": [1.00]})
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 0

    def test_new_stock_no_signal(self):
        signal = MACrossSignal()
        prev = pd.DataFrame({"ts_code": ["A"], "KLEN35": [1.02], "KLEN36": [1.00]})
        signal.generate(prev, StrategyContext(trade_date="d1", capital=1e5))
        cur = pd.DataFrame(
            {"ts_code": ["A", "B"], "KLEN35": [1.02, 1.05], "KLEN36": [1.00, 1.00]}
        )
        result = signal.generate(cur, StrategyContext(trade_date="d2", capital=1e5))
        assert len(result.signals) == 0

    def test_metadata(self):
        signal = MACrossSignal(short_factor="A", long_factor="B")
        features = pd.DataFrame({"ts_code": ["X"], "A": [1.05], "B": [1.00]})
        result = signal.generate(
            features, StrategyContext(trade_date="d1", capital=1e5)
        )
        assert result.metadata["signal_name"] == "ma_cross"
        assert result.metadata["short_factor"] == "A"

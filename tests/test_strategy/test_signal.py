import pytest
import pandas as pd
from alpha_quat.strategy.types import StrategyContext, SignalResult
from alpha_quat.strategy.signal import ISignalGenerator


class TestISignalGenerator:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ISignalGenerator()  # pyright: ignore[reportAbstractUsage]

    def test_concrete_subclass_works(self):
        class MySignal(ISignalGenerator):
            def generate(self, features, ctx):
                return SignalResult(
                    signals=pd.DataFrame({"ts_code": ["000001.SZ"], "score": [0.5]}),
                    metadata={"name": "my_signal"},
                )

        signal = MySignal()
        features = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": ["20240115", "20240115"],
                "factor_001": [0.1, 0.2],
            }
        )
        ctx = StrategyContext(trade_date="20240115", capital=1000000.0)
        result = signal.generate(features, ctx)
        assert isinstance(result, SignalResult)
        assert len(result.signals) == 1
        assert result.metadata["name"] == "my_signal"

    def test_missing_generate_raises(self):
        with pytest.raises(TypeError):

            class BadSignal(ISignalGenerator):
                pass

            BadSignal()  # pyright: ignore[reportAbstractUsage]

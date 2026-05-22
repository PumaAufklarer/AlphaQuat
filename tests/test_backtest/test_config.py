from alpha_quat.backtest.config import BacktestConfig


class TestBacktestConfig:
    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.start_date == "20220501"
        assert cfg.end_date == "20260501"
        assert cfg.initial_capital == 20000
        assert cfg.monthly_addition == 8000
        assert cfg.commission_rate == 0.0005
        assert cfg.min_commission == 0.0
        assert cfg.stop_loss_pct == 0.15
        assert cfg.short_factor == "KLEN35"
        assert cfg.long_factor == "KLEN36"
        assert cfg.top_k == 5
        assert cfg.benchmark is None
        assert cfg.model_dir is None
        assert cfg.rebalance_interval == 5

    def test_custom_values(self):
        cfg = BacktestConfig(
            start_date="20230101",
            end_date="20240101",
            initial_capital=100000,
            commission_rate=0.0003,
            top_k=10,
        )
        assert cfg.start_date == "20230101"
        assert cfg.initial_capital == 100000
        assert cfg.commission_rate == 0.0003
        assert cfg.top_k == 10
        assert cfg.monthly_addition == 8000

from alpha_quat.backtest.rebalance import DailyMonitor, PeriodicRebalance


def test_periodic_rebalance_exists():
    assert hasattr(PeriodicRebalance, "__init__")
    assert hasattr(DailyMonitor, "on_date")

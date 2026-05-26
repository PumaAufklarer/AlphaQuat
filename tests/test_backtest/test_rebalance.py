from alpha_quat.backtest.rebalance import DailyMonitor, PeriodicRebalance


def test_periodic_rebalance_exists():
    assert hasattr(PeriodicRebalance, "on_date")
    assert hasattr(DailyMonitor, "on_date")

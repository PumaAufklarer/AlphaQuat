import pytest
from alpha_quat.backtest.metrics import compute_metrics


class TestComputeMetrics:
    def test_zero_return(self):
        snapshots = [
            {"date": "d1", "cash": 100000, "market_value": 0, "total_value": 100000},
            {"date": "d2", "cash": 100000, "market_value": 0, "total_value": 100000},
        ]
        result = compute_metrics(snapshots, [], total_invested=100000)
        assert result["cumulative_return"] == pytest.approx(0.0)
        assert result["final_value"] == 100000

    def test_positive_return(self):
        snapshots = [
            {"date": f"d{i}", "cash": 0, "market_value": v, "total_value": v}
            for i, v in enumerate(
                [100000] + [100000 + (i + 1) * 20000 / 504 for i in range(504)], start=1
            )
        ]
        result = compute_metrics(snapshots, [], total_invested=100000)
        assert result["cumulative_return"] == pytest.approx(0.20)
        assert result["annualized_return"] == pytest.approx(0.095, abs=0.01)

    def test_drawdown(self):
        snapshots = [
            {"date": "d1", "cash": 0, "market_value": 100, "total_value": 100},
            {"date": "d2", "cash": 0, "market_value": 90, "total_value": 90},
            {"date": "d3", "cash": 0, "market_value": 95, "total_value": 95},
            {"date": "d4", "cash": 0, "market_value": 80, "total_value": 80},
            {"date": "d5", "cash": 0, "market_value": 85, "total_value": 85},
        ]
        result = compute_metrics(snapshots, [], total_invested=100)
        assert result["max_drawdown"] == pytest.approx(-0.20)
        assert result["max_drawdown_date"] == "d4"

    def test_win_rate(self):
        trades = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 200.0}]
        result = compute_metrics(
            [{"date": "d1", "cash": 0, "market_value": 100, "total_value": 100}],
            trades,
            total_invested=100,
        )
        assert result["win_rate"] == pytest.approx(2 / 3)

    def test_no_trades(self):
        result = compute_metrics(
            [{"date": "d1", "cash": 0, "market_value": 100, "total_value": 100}],
            [],
            total_invested=100,
        )
        assert result["win_rate"] == 0.0
        assert result["total_trades"] == 0

    def test_sharpe_zero_vol(self):
        snapshots = [
            {"date": "d1", "cash": 0, "market_value": 100, "total_value": 100},
            {"date": "d2", "cash": 0, "market_value": 100, "total_value": 100},
        ]
        result = compute_metrics(snapshots, [], total_invested=100)
        assert result["sharpe_ratio"] == 0.0

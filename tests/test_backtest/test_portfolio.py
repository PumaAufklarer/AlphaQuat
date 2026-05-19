import pytest
from alpha_quat.backtest.portfolio import Holding, Portfolio


class TestHolding:
    def test_creation(self):
        h = Holding(ts_code="000001.SZ", shares=500, avg_cost=10.0, buy_date="20240115")
        assert h.ts_code == "000001.SZ"
        assert h.shares == 500
        assert h.avg_cost == 10.0


class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(cash=20000)
        assert p.cash == 20000
        assert p.holdings == {}
        assert p.snapshots == []
        assert p.trades == []

    def test_buy_rounds_down_to_lots(self):
        p = Portfolio(cash=20000)
        shares = p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=1050.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        assert shares == 100
        cost = 100 * 10.0
        commission = cost * 0.0005
        assert p.cash == pytest.approx(20000 - cost - commission)
        assert p.holdings["000001.SZ"].shares == 100
        assert p.holdings["000001.SZ"].avg_cost == 10.0

    def test_buy_cannot_afford(self):
        p = Portfolio(cash=50)
        shares = p.buy(
            "000001.SZ",
            price=100.0,
            target_amount=1000.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        assert shares == 0
        assert p.holdings == {}

    def test_buy_min_commission(self):
        p = Portfolio(cash=20000)
        shares = p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=1000.0,
            trade_date="20240115",
            commission_rate=0.0005,
            min_commission=5.0,
        )
        assert shares == 100
        assert p.cash == pytest.approx(20000 - 1000 - 5.0)

    def test_sell_full_position(self):
        p = Portfolio(cash=10000)
        p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=5000.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        pnl = p.sell(
            "000001.SZ",
            price=12.0,
            shares=500,
            trade_date="20240116",
            commission_rate=0.0005,
        )
        proceeds = 500 * 12.0
        commission = proceeds * 0.0005
        assert pnl == pytest.approx((12.0 - 10.0) * 500 - commission)
        assert "000001.SZ" not in p.holdings

    def test_sell_partial_position(self):
        p = Portfolio(cash=50000)
        p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=10000.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        pnl = p.sell(
            "000001.SZ",
            price=11.0,
            shares=300,
            trade_date="20240116",
            commission_rate=0.0005,
        )
        assert p.holdings["000001.SZ"].shares == 700
        assert p.holdings["000001.SZ"].avg_cost == 10.0
        assert pnl > 0

    def test_weighted_avg_cost(self):
        p = Portfolio(cash=50000)
        p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=10000.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        p.buy(
            "000001.SZ",
            price=12.0,
            target_amount=12000.0,
            trade_date="20240116",
            commission_rate=0.0005,
        )
        assert p.holdings["000001.SZ"].shares == 2000
        assert p.holdings["000001.SZ"].avg_cost == 11.0
        assert p.holdings["000001.SZ"].buy_date == "20240116"

    def test_market_value(self):
        p = Portfolio(cash=10000)
        p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=5000.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        mv = p.market_value({"000001.SZ": 11.0})
        assert mv == 500 * 11.0

    def test_record_snapshot(self):
        p = Portfolio(cash=20000)
        p.buy(
            "000001.SZ",
            price=10.0,
            target_amount=5000.0,
            trade_date="20240115",
            commission_rate=0.0005,
        )
        p.record_snapshot("20240115", {"000001.SZ": 11.0})
        assert len(p.snapshots) == 1
        assert p.snapshots[0]["date"] == "20240115"
        assert "cash" in p.snapshots[0]
        assert "market_value" in p.snapshots[0]
        assert "total_value" in p.snapshots[0]

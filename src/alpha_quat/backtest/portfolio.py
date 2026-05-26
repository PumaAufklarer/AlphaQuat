from dataclasses import dataclass

LOT_SIZE = 100


@dataclass
class Holding:
    ts_code: str
    shares: int
    avg_cost: float
    buy_date: str
    peak_price: float = 0.0
    stop_price: float = 0.0


class Portfolio:
    def __init__(self, cash: float = 0.0):
        self.cash = cash
        self.holdings: dict[str, Holding] = {}
        self.snapshots: list[dict] = []
        self.trades: list[dict] = []

    def buy(
        self,
        ts_code,
        price,
        target_amount,
        trade_date,
        commission_rate,
        min_commission=0.0,
        stop_price=0.0,
    ):
        if price <= 0:
            return 0
        desired_shares = int(target_amount / price)
        lots = (desired_shares // LOT_SIZE) * LOT_SIZE
        if lots == 0:
            return 0
        while lots > 0:
            trade_cost = lots * price
            commission = max(trade_cost * commission_rate, min_commission)
            if trade_cost + commission <= self.cash:
                break
            lots -= LOT_SIZE
        if lots == 0:
            return 0
        trade_cost = lots * price
        commission = max(trade_cost * commission_rate, min_commission)
        self.cash -= trade_cost + commission
        if ts_code in self.holdings:
            old = self.holdings[ts_code]
            total_shares = old.shares + lots
            new_avg = (old.avg_cost * old.shares + price * lots) / total_shares
            old_stop = old.stop_price if old.stop_price > 0 else stop_price
            self.holdings[ts_code] = Holding(
                ts_code=ts_code,
                shares=total_shares,
                avg_cost=new_avg,
                buy_date=trade_date,
                peak_price=max(old.peak_price, price),
                stop_price=old_stop,
            )
        else:
            self.holdings[ts_code] = Holding(
                ts_code=ts_code,
                shares=lots,
                avg_cost=price,
                buy_date=trade_date,
                peak_price=price,
                stop_price=stop_price,
            )
        self.trades.append(
            {
                "date": trade_date,
                "ts_code": ts_code,
                "action": "buy",
                "shares": lots,
                "price": price,
                "commission": commission,
                "pnl": 0.0,
            }
        )
        return lots

    def sell(
        self, ts_code, price, shares, trade_date, commission_rate, min_commission=0.0
    ):
        if ts_code not in self.holdings:
            return 0.0
        holding = self.holdings[ts_code]
        actual = min(shares, holding.shares)
        if actual == 0:
            return 0.0
        proceeds = actual * price
        commission = max(proceeds * commission_rate, min_commission)
        realized_pnl = (price - holding.avg_cost) * actual - commission
        self.cash += proceeds - commission
        remaining = holding.shares - actual
        if remaining == 0:
            del self.holdings[ts_code]
        else:
            self.holdings[ts_code] = Holding(
                ts_code=ts_code,
                shares=remaining,
                avg_cost=holding.avg_cost,
                buy_date=holding.buy_date,
            )
        self.trades.append(
            {
                "date": trade_date,
                "ts_code": ts_code,
                "action": "sell",
                "shares": actual,
                "price": price,
                "commission": commission,
                "pnl": realized_pnl,
            }
        )
        return realized_pnl

    def market_value(self, prices):
        total = 0.0
        for code, h in self.holdings.items():
            px = prices.get(code)
            if px is not None:
                total += h.shares * px
        return total

    def total_value(self, prices):
        return self.cash + self.market_value(prices)

    def update_peak_prices(self, prices: dict[str, float]):
        for code, h in self.holdings.items():
            px = prices.get(code)
            if px is not None and px > h.peak_price:
                h.peak_price = px

    def record_snapshot(self, date, prices):
        mv = self.market_value(prices)
        self.snapshots.append(
            {
                "date": date,
                "cash": self.cash,
                "market_value": mv,
                "total_value": self.cash + mv,
            }
        )

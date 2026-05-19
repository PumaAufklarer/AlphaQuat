import pandas as pd
from alpha_quat.strategy.position import IPositionManager

LOT_SIZE = 100


class EqualWeightTopKPosition(IPositionManager):
    def __init__(self, top_k=5):
        self.top_k = top_k

    def allocate(self, signals, ctx):
        df = signals.signals
        buys = df.loc[df["action"] == "buy"].copy()
        if buys.empty:
            return pd.DataFrame(columns=["ts_code", "target_weight"])
        buys = buys.sort_values("score", ascending=False).head(self.top_k)
        weight = 1.0 / len(buys)
        return pd.DataFrame(
            {"ts_code": buys["ts_code"].values, "target_weight": [weight] * len(buys)}
        )

    def constrain(self, positions, ctx):
        if positions.empty:
            return positions
        result = positions.copy()
        if ctx.universe is not None:
            result = result.loc[result["ts_code"].isin(ctx.universe)]
        if not result.empty:
            total = result["target_weight"].sum()
            if total > 0:
                result["target_weight"] = result["target_weight"] / total
        return result.reset_index(drop=True)

    def execute(self, target, prev, ctx):
        if target.empty:
            return (
                pd.DataFrame(
                    columns=[
                        "ts_code",
                        "target_weight",
                        "target_shares",
                        "target_amount",
                    ]
                ),
                pd.DataFrame(
                    columns=["ts_code", "action", "delta_shares", "delta_amount"]
                ),
            )
        prices = ctx.prices
        if prices is None:
            raise ValueError("ctx.prices is required")
        price_map = dict(zip(prices["ts_code"], prices["open"]))
        rows = []
        for _, row in target.iterrows():
            code = row["ts_code"]
            px = price_map.get(code)
            if px is None or px <= 0:
                continue
            shares = int(row["target_weight"] * ctx.capital / px)
            shares = (shares // LOT_SIZE) * LOT_SIZE
            if shares == 0:
                continue
            rows.append(
                {
                    "ts_code": code,
                    "target_weight": row["target_weight"],
                    "target_shares": shares,
                    "target_amount": shares * px,
                }
            )
        pos_df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=["ts_code", "target_weight", "target_shares", "target_amount"]
            )
        )
        orders = [
            {
                "ts_code": r["ts_code"],
                "action": "buy",
                "delta_shares": r["target_shares"],
                "delta_amount": r["target_amount"],
            }
            for r in rows
        ]
        orders_df = (
            pd.DataFrame(orders)
            if orders
            else pd.DataFrame(
                columns=["ts_code", "action", "delta_shares", "delta_amount"]
            )
        )
        return pos_df, orders_df

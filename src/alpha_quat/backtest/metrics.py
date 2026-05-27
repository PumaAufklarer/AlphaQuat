import math


def compute_metrics(
    snapshots,
    trades,
    total_invested,
    risk_free_rate=0.025,
    additions: dict[str, float] | None = None,
):
    if not snapshots:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_date": None,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "final_value": 0.0,
            "total_invested": total_invested,
        }

    if additions is None:
        additions = {}

    final_value = snapshots[-1]["total_value"]
    cumulative_return = (
        (final_value - total_invested) / total_invested if total_invested > 0 else 0.0
    )

    n_dates = len(snapshots)

    # Time-weighted return: compound adjusted daily returns
    daily_returns = []
    for i in range(1, len(snapshots)):
        prev_tv = snapshots[i - 1]["total_value"]
        curr_tv = snapshots[i]["total_value"]
        date = snapshots[i]["date"]
        addition = additions.get(date, 0)
        if prev_tv > 0:
            adjusted_curr = curr_tv - addition
            daily_returns.append(adjusted_curr / prev_tv - 1.0)

    if n_dates >= 2 and total_invested > 0 and final_value > 0:
        years = n_dates / 252.0
        annualized_return = (final_value / total_invested) ** (1.0 / years) - 1.0
    else:
        annualized_return = 0.0

    max_drawdown = 0.0
    max_drawdown_date = None
    peak = snapshots[0]["total_value"]
    for s in snapshots:
        tv = s["total_value"]
        if tv > peak:
            peak = tv
        dd = (tv - peak) / peak if peak > 0 else 0.0
        if dd < max_drawdown:
            max_drawdown = dd
            max_drawdown_date = s["date"]

    if len(daily_returns) >= 2:
        mean_ret = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (
            (mean_ret * 252 - risk_free_rate) / (std * math.sqrt(252))
            if std > 0
            else 0.0
        )
    else:
        sharpe = 0.0

    # Position-level win rate (group sells by position, not per-event)
    sell_trades = [t for t in trades if t.get("action") == "sell"]
    if sell_trades:
        positions: dict[tuple[str, str], float] = {}
        for t in sell_trades:
            key = (t["ts_code"], t.get("buy_date", ""))
            positions[key] = positions.get(key, 0.0) + t.get("pnl", 0.0)
        position_wins = sum(1 for pnl in positions.values() if pnl > 0)
        position_losses = sum(1 for pnl in positions.values() if pnl <= 0)
        win_rate = (
            position_wins / (position_wins + position_losses)
            if (position_wins + position_losses) > 0
            else 0.0
        )
    else:
        win_rate = 0.0

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_drawdown_date,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "final_value": final_value,
        "total_invested": total_invested,
    }

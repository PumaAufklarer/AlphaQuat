import logging

import numpy as np
import torch

from alpha_quat.model.nn.rl_agent.dataset import (
    _FEATURE_COLS,
    _load_stock_data,
    build_state,
    select_stocks,
)
from alpha_quat.model.nn.rl_agent.models.position_agent import PositionAgent

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEQ_LEN = 60
_HORIZON_SHORT = 5
_HORIZON_LONG = 20


@torch.no_grad()
def evaluate(
    model: PositionAgent,
    data_dir,
    start: str,
    end: str,
    max_stocks: int = 100,
    filter_universe: bool = True,
    circ_mv_percentile: float | None = None,
) -> dict:
    """Evaluate model with multi-horizon reward (5d + 20d). Uses raw reward (no baseline)."""
    stock_data = _load_stock_data(
        data_dir,
        start,
        end,
        filter_universe=filter_universe,
        circ_mv_percentile=circ_mv_percentile,
    )
    codes = select_stocks(stock_data, max_stocks=max_stocks)
    model.eval()

    all_rewards = []
    per_stock = {}

    for code in codes:
        df = stock_data.get(code)
        if df is None or len(df) < _SEQ_LEN + _HORIZON_LONG:
            continue
        vals = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        close_arr = df["close"].to_numpy(dtype=np.float64)
        atr_arr = df["atr_ratio"].to_numpy(dtype=np.float64)

        episode_len = len(df) - _SEQ_LEN - _HORIZON_LONG
        if episode_len <= 0:
            continue

        rewards = []
        position = 0.0
        for t in range(_SEQ_LEN, len(df) - _HORIZON_LONG):
            market_seq = vals[t - _SEQ_LEN : t].copy()
            state = build_state(market_seq, position, 0, _SEQ_LEN)
            state_t = torch.from_numpy(state).unsqueeze(0).to(_DEVICE)
            dist = model.get_dist(state_t)
            mu = dist.mean
            new_pos = float(torch.tanh(mu).item())

            close_now = close_arr[t]
            ret5 = close_arr[t + _HORIZON_SHORT] / close_now - 1.0
            ret20 = close_arr[t + _HORIZON_LONG] / close_now - 1.0
            vol = max(atr_arr[t], 1e-6)
            cost = abs(new_pos - position) * 0.0005
            profit5 = position * (ret5 / vol)
            profit20 = position * (ret20 / vol)
            reward = 0.5 * profit5 + 0.5 * profit20 - cost
            rewards.append(reward)
            position = new_pos

        if rewards:
            r = np.array(rewards)
            per_stock[code] = {
                "sharpe": float(r.mean() / (r.std() + 1e-8)),
                "mean_ret": float(r.mean()),
                "std_ret": float(r.std()),
                "n_days": len(rewards),
                "cum_return": float(np.prod(1 + r) - 1),
            }
            all_rewards.extend(rewards)

    r = np.array(all_rewards)
    combined_sharpe = float(r.mean() / (r.std() + 1e-8)) * np.sqrt(252 / _HORIZON_LONG)
    stock_sharpes = [v["sharpe"] for v in per_stock.values()]
    win_rates = [v["cum_return"] > 0 for v in per_stock.values()]

    metrics = {
        "combined_sharpe_annualized": combined_sharpe,
        "mean_reward": float(r.mean()),
        "std_reward": float(r.std()),
        "avg_stock_sharpe": float(np.mean(stock_sharpes)),
        "median_stock_sharpe": float(np.median(stock_sharpes)),
        "win_rate_stocks": float(np.mean(win_rates)),
        "n_stocks": len(codes),
        "n_days_total": len(all_rewards),
    }

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    return metrics

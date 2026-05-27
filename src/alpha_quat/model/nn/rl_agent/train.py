import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from alpha_quat.model.nn.rl_agent.dataset import (
    _FEATURE_COLS,
    _load_stock_data,
    _normalize_market,
    build_state,
    select_stocks,
)
from alpha_quat.model.nn.rl_agent.models.position_agent import PositionAgent

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEQ_LEN = 60
_BATCH_STOCKS = 8
_MAX_STOCKS = 200
_TOTAL_EPOCHS = 100
_CHUNK_SIZE = 200
_HORIZON_SHORT = 5
_HORIZON_LONG = 20
_SR_DIST_COLS = [
    "resistance_20d_dist",
    "resistance_60d_dist",
    "support_20d_dist",
    "support_60d_dist",
]


def _run_batch_episode(
    stock_dfs: list,
    model: PositionAgent,
    optimizer: optim.Optimizer,
    device: torch.device,
    seq_len: int = _SEQ_LEN,
    chunk_size: int = _CHUNK_SIZE,
) -> float:
    """Process a batch of stocks in parallel.

    Reward: 0.5 × 5d position_return + 0.5 × 20d position_return.
    Only backpropagates on steps where |keltner_pos| > 0.5 or SR distance ≤ 10.
    Uses per-stock running baseline for advantage.
    """
    B = len(stock_dfs)
    if B == 0:
        return 0.0

    min_T = min(len(df) for df in stock_dfs)
    episode_len = min_T - seq_len - _HORIZON_LONG
    if episode_len <= 0:
        return 0.0

    # Pre-compute all arrays: (B, episode_len, ...)
    all_market, all_close, all_close5, all_close20, all_atr, all_mask = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for df in stock_dfs:
        vals = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        close_arr = df["close"].to_numpy(dtype=np.float64)
        atr_arr = df["atr_ratio"].to_numpy(dtype=np.float64)
        kpos = df["keltner_pos"].to_numpy(dtype=np.float64)
        sr = df[_SR_DIST_COLS].fillna(999).to_numpy(dtype=np.float64)
        min_sr = sr.min(axis=-1)

        ms, cs, c5, c20, ats, inf = [], [], [], [], [], []
        for t in range(seq_len, seq_len + episode_len):
            ms.append(_normalize_market(vals[t - seq_len : t].copy()))
            cs.append(close_arr[t])
            c5.append(close_arr[t + _HORIZON_SHORT])
            c20.append(close_arr[t + _HORIZON_LONG])
            ats.append(atr_arr[t])
            inf.append(bool(abs(kpos[t]) > 0.5 or min_sr[t] <= 10))

        all_market.append(np.stack(ms))
        all_close.append(cs)
        all_close5.append(c5)
        all_close20.append(c20)
        all_atr.append(ats)
        all_mask.append(inf)

    market_t = torch.from_numpy(np.stack(all_market)).float().to(device)
    close_t = torch.tensor(all_close, device=device).float()
    close5_t = torch.tensor(all_close5, device=device).float()
    close20_t = torch.tensor(all_close20, device=device).float()
    atr_t = torch.tensor(all_atr, device=device).float()
    mask_t = torch.tensor(all_mask, device=device).bool()

    positions = torch.zeros(B, 1, device=device)
    total_loss = 0.0
    total_steps = 0

    for step in range(episode_len):
        active = mask_t[:, step]
        if not active.any():
            continue

        # State: market (B,60,14) + position + days_held
        market = market_t[:, step]
        pos = positions.unsqueeze(1).expand(-1, seq_len, -1)
        dh = positions.abs() > 0
        dh_arr = dh.float().expand(-1, seq_len).unsqueeze(-1)
        state_in = torch.cat([market, pos, dh_arr], dim=-1)

        dist = model.get_dist(state_in[active])
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        new_pos_i = torch.tanh(actions)

        # Rewards for active stocks only
        ret5 = close5_t[:, step] / close_t[:, step] - 1.0
        ret20 = close20_t[:, step] / close_t[:, step] - 1.0
        vol = atr_t[:, step].clamp(min=1e-6)
        p_old = positions[active].squeeze()
        p_new = new_pos_i.squeeze()

        ret5_a = ret5[active]
        ret20_a = ret20[active]
        vol_a = vol[active]

        profit5 = p_old * (ret5_a / vol_a)
        profit20 = p_old * (ret20_a / vol_a)
        cost_a = (p_new - p_old).abs() * 0.0005
        rewards = 0.5 * profit5 + 0.5 * profit20 - cost_a

        adv = torch.sign(rewards)

        step_loss = (-log_probs * adv.unsqueeze(1)).mean()
        step_loss.backward()

        total_loss += step_loss.item()
        total_steps += 1

        # Update ALL positions (active set their new pos, inactive hold)
        new_pos_full = positions.clone()
        new_pos_full[active] = new_pos_i
        positions = new_pos_full

        # Chunk: optimizer step every chunk_size steps
        if total_steps > 0 and total_steps % chunk_size == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Final optimizer step for remainder
    if total_steps > 0 and total_steps % chunk_size != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(total_steps, 1)


def train_rl(
    data_dir: Path,
    output_dir: Path,
    model: PositionAgent | None = None,
    train_start: str = "20200101",
    train_end: str = "20231231",
    val_start: str = "20240101",
    val_end: str = "20240630",
    n_epochs: int = _TOTAL_EPOCHS,
    batch_stocks: int = _BATCH_STOCKS,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    logger.info("Loading training data: %s ~ %s", train_start, train_end)
    stock_data = _load_stock_data(data_dir, train_start, train_end)
    train_codes = select_stocks(stock_data, max_stocks=_MAX_STOCKS)
    logger.info("Training on %d stocks", len(train_codes))

    logger.info("Loading validation data: %s ~ %s", val_start, val_end)
    val_stock_data = _load_stock_data(data_dir, val_start, val_end)
    val_codes = select_stocks(val_stock_data, max_stocks=500, min_days=60)
    logger.info("Validating on %d stocks", len(val_codes))

    if model is None:
        model = PositionAgent(n_market_features=14)
    model = model.to(_DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_sharpe = float("-inf")
    patience = 0

    for epoch in range(n_epochs):
        model.train()
        n_opt_steps = 0

        torch.manual_seed(epoch)
        perm = torch.randperm(len(train_codes))

        # Reset optimizer at epoch start
        optimizer.zero_grad()

        for i in range(0, len(train_codes), batch_stocks):
            batch = [train_codes[j] for j in perm[i : i + batch_stocks]]
            dfs = [stock_data.get(c) for c in batch]
            dfs = [
                d for d in dfs if d is not None and len(d) >= _SEQ_LEN + _HORIZON_LONG
            ]
            if len(dfs) < 2:
                continue

            # optimizer stepping happens inside _run_batch_episode (chunking)
            _run_batch_episode(dfs, model, optimizer, _DEVICE, _SEQ_LEN)
            n_opt_steps += 1

        sharpes = _evaluate(model, val_stock_data, val_codes)
        avg_sharpe = sum(sharpes) / max(len(sharpes), 1)

        logger.info(
            "Epoch %2d/%d: opt_steps=%d val_sharpe=%.3f",
            epoch + 1,
            n_epochs,
            n_opt_steps,
            avg_sharpe,
        )

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            patience = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info("  → new best model (sharpe=%.3f)", best_sharpe)
        else:
            patience += 1
            if patience >= 15:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pt", map_location=_DEVICE, weights_only=True
        )
    )
    logger.info("Training complete. Best val Sharpe: %.3f", best_sharpe)
    return model


@torch.no_grad()
def _evaluate(model, stock_data, codes):
    """Deterministic validation. Reward = 0.5×5d + 0.5×20d. No baseline adjustment."""
    model.eval()
    sharpes = []
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
            sharpe = float(r.mean() / (r.std() + 1e-8))
            sharpes.append(sharpe)

    return sharpes

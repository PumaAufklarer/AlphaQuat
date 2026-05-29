"""Cross-sectional ranking transformer — pairwise rank loss.

Efficient: pre-builds flat index → on-demand sequence construction per date.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from alpha_quat.model.nn.rl_agent.pretrain import DirectionEncoder

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SEQ_LEN = 60
_FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "volume_ratio",
    "turnover_rate",
    "hl_ratio",
    "ret_5d",
    "close_ma20",
    "atr_ratio",
    "vol_change",
    "amt_change",
]


def _normalize_market(x: np.ndarray) -> np.ndarray:
    out = x.copy().astype(np.float32)
    close_last = out[-1, 3]
    if close_last <= 0:
        close_last = 1.0
    out[:, 0] = out[:, 0] / close_last - 1
    out[:, 1] = out[:, 1] / close_last - 1
    out[:, 2] = out[:, 2] / close_last - 1
    out[:, 3] = out[:, 3] / close_last - 1
    out[:, 5] = out[:, 5] / close_last - 1
    out[:, 4] = np.log1p(out[:, 4])
    for j in range(6, x.shape[1]):
        col = out[:, j]
        m = col.mean()
        s = col.std()
        out[:, j] = (col - m) / s if s > 1e-8 else 0.0
        out[:, j] = np.clip(out[:, j], -5.0, 5.0)
    return out


class RankScorer(nn.Module):
    def __init__(self, n_features=14, d_model=128, **kwargs) -> None:
        super().__init__()
        self.encoder = DirectionEncoder(
            n_features=n_features, d_model=d_model, **kwargs
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.score_head(self.encoder(x)).squeeze(-1)


# ── Data pipeline ──────────────────────────────────────────────────────────


def _build_flat(data_dir: Path, start: str, end: str, cache_name: str = "rank_flat"):
    """Build flat parquet with (14 features + fwd_5d_ret) per (stock, date).

    Returns DataFrame with columns: ts_code, trade_date, 14 features, fwd_5d_ret
    """
    cache_path = data_dir / f"{cache_name}_{start}_{end}.parquet"
    if cache_path.exists():
        logger.info("Loading cached flat data: %s", cache_path)
        return pd.read_parquet(cache_path)

    cache_dir = data_dir / "alpha360"
    all_dates = sorted(
        d.stem for d in cache_dir.glob("*.parquet") if start <= d.stem <= end
    )
    if not all_dates:
        raise FileNotFoundError(f"No alpha360 cache for {start}~{end}")

    # Load universe filters
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    main_board = set(sb.loc[sb["market"] == "主板", "ts_code"])
    st_dir = data_dir / "stock_st"
    st_set: set[tuple[str, str]] = set()
    for f in st_dir.glob("*.parquet"):
        ds = f.stem.replace("_", "")
        if start <= ds <= end:
            st = pd.read_parquet(f)
            st["trade_date"] = ds
            for _, row in st.iterrows():
                st_set.add((str(row["ts_code"]), ds))

    chunks = []
    for d in all_dates:
        df = pd.read_parquet(cache_dir / f"{d}.parquet")
        df["trade_date"] = d
        df = df[df["ts_code"].isin(main_board)]
        if st_set:
            st_mask = df.apply(
                lambda r: (r["ts_code"], r["trade_date"]) in st_set, axis=1
            )
            df = df[~st_mask]
        chunks.append(df)
    full = pd.concat(chunks, ignore_index=True)
    logger.info(
        "Loaded %d rows from %s to %s (filtered: 主板 + non-ST)",
        len(full),
        start,
        end,
    )

    # Compute forward 5-day return per stock (vectorized)
    full = full.sort_values(["ts_code", "trade_date"])
    full["fwd_5d_ret"] = (
        full.groupby("ts_code")["close"].shift(-5) / full["close"] - 1.0
    )
    full["fwd_5d_ret"] = full["fwd_5d_ret"].fillna(-999.0)

    out = full[["ts_code", "trade_date", "fwd_5d_ret"] + _FEATURE_COLS]
    out.to_parquet(cache_path, index=False)
    logger.info("Cached flat data: %s (%d rows)", cache_path, len(out))
    return out


def _build_date_index(df: pd.DataFrame):
    """Build per-date stock list.

    Returns: {trade_date: [(stock_row_idx_in_df, stock_idx_in_per_stock_list), ...]}
    Also returns: per_stock_dfs (for sequence building)
    """
    per_stock = {}
    for code, grp in df.groupby("ts_code"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        per_stock[code] = grp

    # Build a fast lookup: per stock, map trade_date → row index within that stock
    date_index: dict[str, list[tuple[str, int]]] = {}
    for code, sdf in tqdm(per_stock.items(), desc="Indexing dates"):
        for idx, row in sdf.iterrows():
            d = row["trade_date"]
            if d not in date_index:
                date_index[d] = []
            date_index[d].append((code, idx))

    logger.info("Indexed %d dates", len(date_index))
    return date_index, per_stock


def _get_sequence(per_stock: dict, code: str, end_idx: int):
    """Extract (60, 14) normalized sequence for a stock ending at end_idx."""
    sdf = per_stock[code]
    if end_idx < _SEQ_LEN:
        return None
    raw = sdf.iloc[end_idx - _SEQ_LEN : end_idx][_FEATURE_COLS].to_numpy(
        dtype=np.float32
    )
    if np.isnan(raw).any():
        return None
    return _normalize_market(raw)


# ── Loss ───────────────────────────────────────────────────────────────────


def ranking_loss(
    scores: torch.Tensor, rets: torch.Tensor, margin: float = 0.5
) -> torch.Tensor:
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    perm = torch.randperm(n, device=scores.device)
    half = n // 2
    i, j = perm[:half], perm[half : half + half]
    r_diff = torch.sign(rets[i] - rets[j])
    loss = F.relu(margin - r_diff * (scores[i] - scores[j])).mean()
    return loss


# ── Training ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_epoch(model, per_stock, date_index, eval_dates, max_stocks=500):
    """Spearman correlation on held-out dates."""
    model.eval()
    rhos = []
    for date in eval_dates:
        entries = date_index.get(date, [])
        if len(entries) < 10:
            continue
        seqs, rets = [], []
        for code, idx in entries:
            seq = _get_sequence(per_stock, code, idx)
            if seq is None:
                continue
            sdf = per_stock[code]
            ret = sdf.iloc[idx]["fwd_5d_ret"]
            if ret <= -900:
                continue
            seqs.append(seq)
            rets.append(ret)
            if len(seqs) >= max_stocks:
                break
        if len(seqs) < 10:
            continue
        X = torch.from_numpy(np.stack(seqs)).float().to(_DEVICE)
        r = torch.tensor(rets, device=_DEVICE)
        s = model(X).cpu().numpy()
        r_np = r.cpu().numpy()
        if np.std(s) > 1e-8 and np.std(r_np) > 1e-8:
            rhos.append(float(np.corrcoef(np.argsort(s), np.argsort(r_np))[0, 1]))
    return float(np.mean(rhos)) if rhos else 0.0


def train_rank(
    data_dir: Path,
    output_dir: Path,
    train_start: str = "20200101",
    train_end: str = "20231231",
    val_start: str = "20240101",
    val_end: str = "20240630",
    n_epochs: int = 30,
    lr: float = 3e-4,
    max_stocks_per_date: int = 400,
    dates_per_epoch: int = 100,
):
    # Build flat data + index (one-time)
    train_df = _build_flat(data_dir, train_start, train_end, "train")
    val_df = _build_flat(data_dir, val_start, val_end, "val")

    logger.info("Building date index...")
    train_date_index, per_stock = _build_date_index(train_df)
    val_date_index, val_per_stock = _build_date_index(val_df)

    sorted_dates = sorted(
        d for d in train_date_index.keys() if len(train_date_index[d]) >= 10
    )
    val_sorted = sorted(
        d for d in val_date_index.keys() if len(val_date_index[d]) >= 10
    )
    logger.info("Train dates: %d, Val dates: %d", len(sorted_dates), len(val_sorted))

    model = RankScorer(n_features=14, d_model=128).to(_DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_corr = -1.0
    for epoch in range(n_epochs):
        model.train()
        total_loss, n_days = 0.0, 0

        rng = np.random.default_rng(epoch)
        chosen = rng.choice(
            sorted_dates, size=min(dates_per_epoch, len(sorted_dates)), replace=False
        )

        for date in tqdm(chosen, desc=f"Epoch {epoch + 1}"):
            entries = train_date_index[date]
            if len(entries) < 5:
                continue

            # Sample stocks for this date
            if len(entries) > max_stocks_per_date:
                selected = rng.choice(
                    len(entries), size=max_stocks_per_date, replace=False
                )
                entries = [entries[i] for i in selected]

            seqs, rets = [], []
            for code, idx in entries:
                ret = per_stock[code].iloc[idx]["fwd_5d_ret"]
                if ret <= -900:
                    continue
                if idx < _SEQ_LEN:
                    continue
                raw = (
                    per_stock[code]
                    .iloc[idx - _SEQ_LEN : idx][_FEATURE_COLS]
                    .to_numpy(dtype=np.float32)
                )
                if np.isnan(raw).any():
                    continue
                seqs.append(_normalize_market(raw))
                rets.append(ret)

            if len(seqs) < 5:
                continue

            X = torch.from_numpy(np.stack(seqs)).float().to(_DEVICE)
            r = torch.tensor(rets, dtype=torch.float, device=_DEVICE)

            optimizer.zero_grad()
            scores = model(X)
            loss = ranking_loss(scores, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_days += 1

        # Validation
        avg_spearman = _eval_epoch(model, val_per_stock, val_date_index, val_sorted)
        avg_loss = total_loss / max(n_days, 1)

        logger.info(
            "Epoch %2d/%d: loss=%.4f spearman=%.3f",
            epoch + 1,
            n_epochs,
            avg_loss,
            avg_spearman,
        )

        if avg_spearman > best_corr:
            best_corr = avg_spearman
            torch.save(model.state_dict(), output_dir / "best_rank_model.pt")
            logger.info("  → new best (spearman=%.3f)", best_corr)

    if (output_dir / "best_rank_model.pt").exists():
        model.load_state_dict(
            torch.load(
                output_dir / "best_rank_model.pt",
                map_location=_DEVICE,
                weights_only=True,
            )
        )
    logger.info("Training complete. Best Spearman: %.3f", best_corr)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_rank(
        Path("data"),
        Path("data/models/experiments/exp_rank_v1"),
    )

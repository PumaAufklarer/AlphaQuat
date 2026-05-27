"""
Hybrid: train transformer → extract 64-dim embeddings → merge into data/features/.
"""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from alpha_quat.model.nn.alpha_rank.model import RankScoreTransformer
from alpha_quat.model.nn.alpha_rank.train import paired_loss

logger = logging.getLogger(__name__)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EMBED_PREFIX = "emb_"


def train_and_extract(
    data_dir,
    train_start="20200101",
    train_end="20231231",
    n_epochs=5,
    lr=3e-4,
):
    base = data_dir / "rank_cache"

    # Load training data
    X_tr = np.load(base / "train" / "X.npy", mmap_mode="r")
    R_tr = np.load(base / "train" / "R.npy", mmap_mode="r")
    dates_tr = np.load(base / "train" / "dates.npy")
    codes_tr = np.load(base / "train" / "codes.npy")
    logger.info("Train: X%s, codes=%d", X_tr.shape, len(codes_tr))

    date_idx = defaultdict(list)
    for i, d in enumerate(dates_tr):
        date_idx[str(d)].append(i)
    date_groups = list(date_idx.items())
    logger.info("Dates: %d train", len(date_groups))

    model = RankScoreTransformer(n_features=X_tr.shape[-1]).to(_DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        model.train()
        rng = np.random.default_rng(epoch)
        rng.shuffle(date_groups)
        losses, steps = 0, 0
        for date, idx in date_groups:
            i_arr = np.array(idx, dtype=np.intp)
            rng.shuffle(i_arr)
            Xb = torch.from_numpy(X_tr[i_arr].copy()).float().to(_DEVICE)
            Rb = torch.from_numpy(R_tr[i_arr]).float().to(_DEVICE)
            optimizer.zero_grad()
            loss = paired_loss(model(Xb), Rb)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            steps += 1
        logger.info("Epoch %d: loss=%.4f", epoch + 1, losses / max(steps, 1))

    # Extract embeddings
    logger.info("Extracting 64-dim embeddings...")
    model.eval()
    rows = []
    with torch.no_grad():
        for date, indices in date_groups:
            idx = np.array(indices, dtype=np.intp)
            Xb = torch.from_numpy(X_tr[idx].copy()).float().to(_DEVICE)
            emb = model.embed(Xb).cpu().numpy()
            for j, i_pos in enumerate(range(len(idx))):
                rows.append(
                    {
                        "ts_code": codes_tr[idx[i_pos]],
                        "trade_date": date,
                        **{f"{_EMBED_PREFIX}{k}": float(emb[j, k]) for k in range(64)},
                    }
                )

    emb_df = pd.DataFrame(rows)
    merged_cnt = 0
    for date, grp in emb_df.groupby("trade_date"):
        src = data_dir / "features" / f"{date}.parquet"
        if not src.exists():
            continue
        existing = pd.read_parquet(src)
        new_cols = grp.set_index("ts_code").drop(columns="trade_date")
        merged = existing.merge(new_cols, on="ts_code", how="left")
        merged.to_parquet(src, index=False)
        merged_cnt += 1
    logger.info("Merged into %d feature files", merged_cnt)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_extract(Path("data"))

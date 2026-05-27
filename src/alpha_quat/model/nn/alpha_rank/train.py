"""Train RankScoreTransformer with pairwise ranking loss — date-grouped batches."""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from alpha_quat.model.nn.alpha_rank.model import RankScoreTransformer

logger = logging.getLogger(__name__)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HORIZONS = [5, 20, 60]


def _load(base):
    X = np.load(base / "X.npy", mmap_mode="r")
    R = np.load(base / "R.npy", mmap_mode="r")
    dates = np.load(base / "dates.npy")
    return X, R, dates


def paired_loss(scores, returns, margin=0.5):
    """Pairwise margin ranking loss within a batch (same date).

    scores: (B, 3), returns: (B, 3)
    """
    B = scores.size(0)
    if B < 2:
        return torch.tensor(0.0, device=scores.device)

    loss = 0.0
    half = B // 2
    perm = torch.randperm(B, device=scores.device)
    i_idx = perm[:half]
    j_idx = perm[half : 2 * half]
    if len(j_idx) < len(i_idx):
        j_idx = perm[half : half + len(i_idx)]

    for h in range(3):
        s_i, s_j = scores[i_idx, h], scores[j_idx, h]
        r_i, r_j = returns[i_idx, h], returns[j_idx, h]
        r_diff = torch.sign(r_i - r_j)
        valid = r_diff.abs() > 1e-8
        if valid.any():
            loss += F.relu(margin - r_diff[valid] * (s_i[valid] - s_j[valid])).mean()
    return loss / 3


@torch.no_grad()
def _eval(model, X_val, R_val):
    """Spearman correlation per horizon."""
    model.eval()
    rhos = [[] for _ in range(3)]
    for start in range(0, len(X_val), 4096):
        sl = slice(start, start + 4096)
        Xb = torch.from_numpy(X_val[sl].copy()).float().to(_DEVICE)
        Rb = torch.from_numpy(R_val[sl])
        scores = model(Xb).cpu().numpy()
        for h in range(3):
            s, r = scores[:, h], Rb[:, h]
            if s.std() > 1e-8 and r.std() > 1e-8:
                rhos[h].extend([float(np.corrcoef(np.argsort(s), np.argsort(r))[0, 1])])
    return {
        f"{_HORIZONS[h]}d": float(np.mean(v)) if v else 0.0 for h, v in enumerate(rhos)
    }


def train(
    data_dir,
    output_dir,
    train_start="20200101",
    train_end="20231231",
    val_start="20240101",
    val_end="20240630",
    n_epochs=30,
    lr=3e-4,
):
    base = data_dir / "rank_cache"
    X_tr, R_tr, dates_tr = _load(base / "train")
    X_val, R_val, _ = _load(base / "val")
    logger.info(
        "Train: %d samples (%d feat, 60d seq), Val: %d",
        len(X_tr),
        X_tr.shape[-1],
        len(X_val),
    )

    date_idx = defaultdict(list)
    for i, d in enumerate(dates_tr):
        date_idx[str(d)].append(i)
    date_groups = list(date_idx.items())
    logger.info("Dates: %d train", len(date_groups))

    model = RankScoreTransformer(n_features=X_tr.shape[-1]).to(_DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_s = 0.0

    for epoch in range(n_epochs):
        model.train()
        rng = np.random.default_rng(epoch)
        rng.shuffle(date_groups)
        total_loss, steps = 0, 0

        for date, indices in date_groups:
            idx = np.array(indices, dtype=np.intp)
            rng.shuffle(idx)
            Xb = torch.from_numpy(X_tr[idx].copy()).float().to(_DEVICE)
            Rb = torch.from_numpy(R_tr[idx]).float().to(_DEVICE)

            optimizer.zero_grad()
            scores = model(Xb)  # (B, 3)
            loss = paired_loss(scores, Rb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        sched.step()

        rhos = _eval(model, X_val, R_val)
        avg_rho = np.mean(list(rhos.values()))
        logger.info(
            "Epoch %2d/%d: loss=%.4f rho=%s",
            epoch + 1,
            n_epochs,
            total_loss / max(steps, 1),
            {k: f"{v:.4f}" for k, v in rhos.items()},
        )

        if avg_rho > best_s:
            best_s = avg_rho
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info("  → best (rho=%.4f)", best_s)

    logger.info("Done. Best rho: %.4f", best_s)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(Path("data"), Path("data/models/experiments/exp_ar_v3"))

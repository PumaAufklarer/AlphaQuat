"""Build rank cache: 60-day × 6 OHLCV + temporal + industry features."""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SEQ_LEN = 60
_HORIZONS = [5, 20, 60]

# Raw OHLCV + derived (5) + fundamentals (2) = 13 from flat parquet
_A360 = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "volume_ratio",
    "turnover_rate",
    "ret_5d",
    "close_ma20",
    "atr_ratio",
]
_A158 = ["pe_ttm", "pb"]

# Temporal feature names (computed, not from flat)
_TEMPORAL = ["year_sin", "year_cos", "er_sin", "er_cos", "quarter_sin", "quarter_cos"]
_EXTRA = ["vol_spike_3d", "pe_rel", "pb_rel", "ind_momentum"]
ALL_FEATURES = _A360 + _TEMPORAL + _EXTRA  # 11 + 6 + 4 = 21


def _temporal_features(date_str: str) -> dict:
    """Compute 6 temporal features from a YYYYMMDD date string."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    doy = dt.timetuple().tm_yday  # day of year [1, 365]
    q = (dt.month - 1) // 3 + 1  # quarter [1, 4]
    return {
        "year_sin": np.sin(2 * np.pi * doy / 365),
        "year_cos": np.cos(2 * np.pi * doy / 365),
        "er_sin": np.sin(2 * np.pi * doy / 90),
        "er_cos": np.cos(2 * np.pi * doy / 90),
        "quarter_sin": np.sin(2 * np.pi * q / 4),
        "quarter_cos": np.cos(2 * np.pi * q / 4),
    }


def _norm_seq(seqs: np.ndarray) -> np.ndarray:
    """Normalize alpha360 features (cols 0-5) per-sequence. Leave cols 6+ as-is."""
    out = seqs.copy()
    # 0-5 are alpha360: divide relative to last close
    cl = out[:, -1, 3:4].copy()
    cl[cl <= 0] = 1.0
    for i in [0, 1, 2, 3]:
        out[:, :, i] = out[:, :, i] / cl - 1  # OHLC price ratios
    out[:, :, 5] = out[:, :, 5] / cl - 1  # vwap
    out[:, :, 4] = np.log1p(np.maximum(out[:, :, 4], 0))  # log volume
    return out


def build_flat(
    data_dir: Path,
    start: str,
    end: str,
    quality_codes: dict[str, set[str]] | None = None,
) -> Path:
    cache = data_dir / f"ar_flat_{start}_{end}.parquet"
    if cache.exists():
        return cache

    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    main = set(sb.loc[sb["market"] == "主板", "ts_code"])
    ind_map = dict(zip(sb["ts_code"], sb["industry"]))

    st_dir = data_dir / "stock_st"
    st_set = set()
    for f in sorted(st_dir.glob("*.parquet")):
        ds = f.stem.replace("_", "")
        if start <= ds <= end:
            for _, r in pd.read_parquet(f).iterrows():
                st_set.add((str(r["ts_code"]), ds))

    dates = sorted(
        d.stem
        for d in (data_dir / "alpha360").glob("*.parquet")
        if start <= d.stem <= end
    )
    all_cols = _A360 + _A158
    rows = []

    for d in dates:
        fp = data_dir / "features" / f"{d}.parquet"
        if not fp.exists():
            continue
        a158 = pd.read_parquet(fp)
        dupe = set(a158.columns) & set(_A360)
        if dupe:
            a158 = a158.drop(columns=list(dupe), errors="ignore")
        a360 = pd.read_parquet(data_dir / "alpha360" / f"{d}.parquet")
        m = a360.merge(a158, on="ts_code", how="inner")
        m = m[m["ts_code"].isin(main)]
        if st_set:
            m = m[~m.apply(lambda r: (r["ts_code"], d) in st_set, axis=1)]
        if quality_codes:
            m = m[m["ts_code"].isin(quality_codes.get(d, set()))]
        if m.empty:
            continue
        m["industry"] = m["ts_code"].map(ind_map)
        # Drop stocks without industry (need for industry features)
        m = m.dropna(subset=["industry"])
        if m.empty:
            continue
        keep = [
            c
            for c in ["ts_code", "trade_date", "industry"] + all_cols
            if c in m.columns
        ]
        m["trade_date"] = d
        rows.append(m[keep])

    full = pd.concat(rows, ignore_index=True)
    full.to_parquet(cache, index=False)
    logger.info("Flat: %d rows, %d cols → %s", len(full), len(full.columns), cache)
    return cache


def build_numpy(flat_path: Path, out_dir: Path):
    out_x = out_dir / "X.npy"
    out_r = out_dir / "R.npy"
    out_d = out_dir / "dates.npy"
    if out_x.exists() and out_r.exists() and out_d.exists():
        return

    df = pd.read_parquet(flat_path)
    max_h = max(_HORIZONS)
    n_feat = len(_A360) + len(_TEMPORAL) + len(_EXTRA)  # 11 + 6 + 4 = 21

    # Pre-compute industry medians per date (for pe_rel, pb_rel, ind_momentum)
    logger.info("Computing industry medians...")
    ind_pe = (
        df.groupby(["trade_date", "industry"])["pe_ttm"].transform("median").fillna(0)
    )
    ind_pb = df.groupby(["trade_date", "industry"])["pb"].transform("median").fillna(0)
    ind_ret = (
        df.groupby(["trade_date", "industry"])["ret_5d"].transform("median").fillna(0)
    )

    X, R, D, D_codes = [], [], [], []
    total_stocks = len(df.groupby("ts_code", sort=False))
    logger.info("Building sequences: %d stocks, %d features/day", total_stocks, n_feat)

    for i, (code, grp) in enumerate(df.groupby("ts_code", sort=False)):
        if i % 500 == 0 and i > 0:
            logger.info("  %d/%d", i, total_stocks)
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        arr = grp[_A360].to_numpy(dtype=np.float32)
        close = grp["close"].to_numpy(dtype=np.float64)
        dates = grp["trade_date"].tolist()
        T = len(grp)
        T = len(grp)
        n = T - _SEQ_LEN - max_h
        if n <= 0:
            continue

        # Get industry-relative features
        pe_med = grp["pe_ttm"] - ind_pe.loc[ind_pe.index[grp.index]]
        pb_med = grp["pb"] - ind_pb.loc[ind_pb.index[grp.index]]
        ret_med = grp["ret_5d"] - ind_ret.loc[ind_ret.index[grp.index]]

        # Build sequences
        seqs = np.zeros((n, _SEQ_LEN, n_feat), dtype=np.float32)
        rets = np.zeros((n, 3), dtype=np.float32)
        out_dates = []

        for idx, pos in enumerate(range(_SEQ_LEN, T - max_h)):
            # Alpha360 block (cols 0-10): raw + derived
            seqs[idx, :, :11] = arr[pos - _SEQ_LEN : pos]

            # Temporal block (cols 11-16): computed from each date in sequence
            for t_offset, td in enumerate(dates[pos - _SEQ_LEN : pos]):
                tf = _temporal_features(td)
                seqs[idx, t_offset, 11] = tf["year_sin"]
                seqs[idx, t_offset, 12] = tf["year_cos"]
                seqs[idx, t_offset, 13] = tf["er_sin"]
                seqs[idx, t_offset, 14] = tf["er_cos"]
                seqs[idx, t_offset, 15] = tf["quarter_sin"]
                seqs[idx, t_offset, 16] = tf["quarter_cos"]

            # Extra features (cols 17-20): vol_spike_3d, pe_rel, pb_rel, ind_momentum
            if pos >= 3:
                vol3 = arr[pos - 3 : pos, 4].mean()  # volume last 3 days
                vol60 = arr[pos - _SEQ_LEN : pos, 4].mean()  # volume all 60 days
                vol_spike = vol3 / max(vol60, 1e-8)
            else:
                vol_spike = 1.0
            seqs[idx, -1, 17] = vol_spike

            # Industry-relative: last timestep only (these are point-in-time)
            seqs[idx, -1, 18] = pe_med.iloc[pos]
            seqs[idx, -1, 19] = pb_med.iloc[pos]
            seqs[idx, -1, 20] = ret_med.iloc[pos]

            # Forward returns
            for hi, h in enumerate(_HORIZONS):
                ret_val = close[pos + h] / close[pos] - 1.0
                rets[idx, hi] = np.nan_to_num(ret_val, nan=0.0, posinf=0.0, neginf=0.0)
            out_dates.append(dates[pos])

        # Nan/inf check
        bad = np.isnan(seqs).any(axis=(1, 2)) | np.isinf(seqs).any(axis=(1, 2))
        if bad.all():
            continue

        keep = ~bad
        # Normalize alpha360 (cols 0-10, but only 0-5 need per-seq norm)
        seqs_norm = seqs.copy()
        seqs_norm[keep] = _norm_seq(seqs[keep])

        n_keep = keep.sum()
        X.append(seqs_norm[keep])
        R.append(rets[keep])
        D.extend(np.array(out_dates)[keep])
        D_codes.extend([code] * n_keep)

    X_all = np.concatenate(X, axis=0).astype(np.float32)
    R_all = np.concatenate(R, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_x, X_all)
    np.save(out_r, R_all)
    np.save(out_d, np.array(D))
    np.save(out_dir / "codes.npy", np.array(D_codes))
    logger.info(
        "Saved X%s R%s dates(%d) codes(%d)",
        X_all.shape,
        R_all.shape,
        len(D),
        len(D_codes),
    )


def build(data_dir, out_dir, start, end, quality_filter=False):
    quality_codes = None
    if quality_filter:
        from alpha_quat.model.nn.alpha_rank.quality_filter import build_quality_codes

        quality_codes = build_quality_codes(data_dir, start, end)
    flat = build_flat(data_dir, start, end, quality_codes=quality_codes)
    build_numpy(flat, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build(Path("data"), Path("data/rank_cache/train"), "20200101", "20231231")

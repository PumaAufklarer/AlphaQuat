"""Shared constants for model training and inference."""

LOT_SIZE: int = 100

# Features with zero/negligible gain from prior experiments — excluded from training.
# Unified superset across all call sites (data.py, ml_signal.py, baseline.py).
ZERO_GAIN_FEATURES: set[str] = {
    "KMID94",
    "KMID95",
    "KMID96",
    "KLEN94",
    "KLEN95",
    "KLEN96",
    "KMID97",
    "KLEN97",
    "KMID98",
    "KLEN98",
    "KMID99",
    "KLEN99",
    "KMID100",
    "KLEN100",
    "KMID101",
    "O2C",
    "DRP",
    "HLC",
    "pe_ttm",
    "pb",
    "ROE_RAW",
    "ROE",
    "MV",
    "VOLRATIO",
    "EMA12C",
    "EMA26C",
    "MACD",
    "RSI14",
    "SLOPE5",
    "SLOPE20",
}


def date_to_path(yyyymmdd: str) -> str:
    """Convert YYYYMMDD to daily parquet path segment YYYY_MM_DD."""
    return f"{yyyymmdd[:4]}_{yyyymmdd[4:6]}_{yyyymmdd[6:8]}"

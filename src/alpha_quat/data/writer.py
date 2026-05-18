"""Parquet file writer with partitioning support."""

from pathlib import Path

import pandas as pd


class ParquetWriter:
    def overwrite(self, df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def write(self, df: pd.DataFrame, base_dir: Path, trade_date: str):
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / f"{trade_date}.parquet"
        df.to_parquet(file_path, index=False)

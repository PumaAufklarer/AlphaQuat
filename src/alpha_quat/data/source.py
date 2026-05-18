"""Abstract base class for tushare data sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal


class DataSource(ABC):
    api_name: str
    partition_by: Literal["none", "date"]
    fields: str
    start_date: str | None = None

    @abstractmethod
    def get_params(self, trade_date: str | None = None) -> dict: ...

    def path_for(self, data_dir: Path, trade_date: str | None = None) -> Path:
        if self.partition_by == "none":
            return data_dir / f"{self.api_name}.parquet"
        if trade_date is None:
            return data_dir / self.api_name
        return data_dir / self.api_name / f"{trade_date}.parquet"

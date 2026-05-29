"""Configuration from YAML file."""

from pathlib import Path

import yaml


class Config:
    token: str
    data_dir: Path

    def __init__(self, token: str, data_dir: Path | str = "data") -> None:
        self.token = token
        self.data_dir = Path(data_dir)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            token=raw["tushare"]["token"],
            data_dir=raw.get("data", {}).get("dir", "data"),
        )

import json
from pathlib import Path

from alpha_quat.experiment.config import ExperimentConfig


class ExperimentRegistry:
    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "models" / "registry.json"

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return json.load(f)

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)

    def register(self, config: ExperimentConfig) -> None:
        entries = self._read()
        entries.append(
            {"name": config.name, "mode": config.mode, "created": config.created_at}
        )
        self._write(entries)

    def list_experiments(self) -> list[dict]:
        return self._read()

    def latest(self) -> dict | None:
        entries = self._read()
        return entries[-1] if entries else None

    def find(self, name: str) -> dict | None:
        entries = self._read()
        for entry in entries:
            if entry["name"] == name:
                return entry
        return None

"""Experiment configuration (a thin wrapper over the YAML file)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["data_dir"]

    @property
    def reports_dir(self) -> Path:
        return self.root / self.raw["reports_dir"]

    def path(self, *parts: str) -> Path:
        p = self.data_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f))

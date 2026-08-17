from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    required = {"data", "model", "loss", "training", "output"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    return config


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(base or Path.cwd()) / path


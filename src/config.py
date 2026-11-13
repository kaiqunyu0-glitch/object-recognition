"""Load config.yaml and apply small command-line overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML file. The top level must be a mapping of settings."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain settings in `key: value` form.")
    return loaded


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Set dotted keys such as ``train.head_epochs=1``.

    The value is parsed as YAML, so ``true``, ``1``, and ``0.1`` keep the
    right type. Unknown keys are rejected so a typo fails immediately.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key=value.")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f"Unknown config key {key!r}.")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ValueError(f"Unknown config key {key!r}.")
        node[parts[-1]] = value
    return cfg


def dump_config(path: Path, cfg: dict[str, Any]) -> None:
    """Save the settings that were actually used, next to the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)

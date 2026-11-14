"""Save and load the training checkpoint.

The file holds the weights plus the class names and image size, so
evaluate.py and predict.py can run without guessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by this project.

    ``weights_only=False`` is required because the file also stores class
    names and the config, not only tensors. Only load checkpoints you trained.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. Train first:\n"
            "  python train.py --data /path/to/photos"
        )
    try:
        loaded = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(loaded, dict) or "model_state" not in loaded:
        raise RuntimeError(f"{checkpoint_path} is not a checkpoint from this project.")
    return loaded


def default_checkpoint(config_path: str | Path) -> Path:
    """best.pt inside the output_dir written in the YAML file."""
    from src.config import load_config

    cfg = load_config(config_path)
    return Path(cfg["output_dir"]) / "best.pt"

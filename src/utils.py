"""Small helpers shared by the training scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix the random numbers we control.

    Data augmentation still picks random crops on purpose. The seed makes
    those choices repeatable, and it decides the train/val/test split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON that can contain Chinese class names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


_WARNED: set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Print a warning the first time `key` is seen."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(message, flush=True)

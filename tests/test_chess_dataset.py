"""Chess pilot config and notebook. No download and no training."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "train_chess_yolo11.ipynb"

EXPECTED_NAMES = [
    "bishop",
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]


def _download_module():
    path = ROOT / "scripts" / "download_chess.py"
    spec = importlib.util.spec_from_file_location("download_chess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chess_config_lists_the_source_classes_in_order():
    loaded = yaml.safe_load((ROOT / "config.chess.yaml").read_text(encoding="utf-8"))
    names = loaded["names"]
    ordered = [names[key] for key in sorted(names, key=int)]
    assert ordered == EXPECTED_NAMES
    text = (ROOT / "config.chess.yaml").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text
    assert "LibreYOLO/chess-pieces-mjzgj" in text
    assert "train 202" in text
    assert "valid 58" in text
    assert "test 29" in text
    module = _download_module()
    assert module.HUB_REPO == "LibreYOLO/chess-pieces-mjzgj"
    assert module.EXPECTED_IMAGES == {"train": 202, "valid": 58, "test": 29}
    assert module.LICENSE == "CC BY 4.0"
    assert module.class_names() == EXPECTED_NAMES


def test_chess_notebook_defaults_to_yolo11x_finetune():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    chunks = []
    for cell in notebook["cells"]:
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        chunks.append(text)
        if cell["cell_type"] == "code":
            ast.parse(text)
            assert cell["execution_count"] is None
    source = "\n".join(chunks)
    assert 'MODE = "finetune"' in source
    assert 'MODEL_SIZE = "x"' in source
    assert "yolo11{MODEL_SIZE}.pt" in source or "yolo11x.pt" in source
    assert "LibreYOLO/chess-pieces-mjzgj" in source
    assert "AcceptableResultStop" in source
    assert "on_fit_epoch_end" in source
    assert "resume=True" in source
    assert "/kaggle/working" in source
    assert "CC BY 4.0" in source

"""The Kaggle notebook is valid notebook JSON and keeps the documented defaults."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "train_coco_yolo11x.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        source = cell["source"]
        chunks.append("".join(source) if isinstance(source, list) else source)
    return "\n".join(chunks)


def test_kaggle_notebook_is_nbformat_json():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code", "raw"}
        assert "source" in cell
        if cell["cell_type"] == "code":
            assert "outputs" in cell
            assert cell["execution_count"] is None
            source = cell["source"]
            text = "".join(source) if isinstance(source, list) else source
            ast.parse(text)

    try:
        import nbformat
        from nbformat.validator import validate
    except ImportError:
        return
    validate(nbformat.reads(NOTEBOOK.read_text(encoding="utf-8"), as_version=4))


def test_kaggle_notebook_defaults_to_pretrained_finetune():
    source = _source(_notebook())
    assert 'MODE = "finetune"' in source
    assert "yolo11x.pt" in source
    assert "yolo11x.yaml" in source
    assert "resume=True" in source
    assert "/kaggle/working" in source
    assert "patience" in source
    assert "on_fit_epoch_end" in source
    # From-scratch stays a switch. The notes have to say how long it is.
    assert "scratch" in source
    assert "30" in source

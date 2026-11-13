"""Tests for the written report and the unknown decision."""

from __future__ import annotations

from pathlib import Path

from src.inference import format_prediction
from src.reporting import write_report


def test_report_counts_mistakes_and_writes_files(tmp_path: Path):
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    probabilities = [
        [0.9, 0.1],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.3, 0.7],
    ]
    paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    metrics = write_report(
        tmp_path,
        ["mug", "keys"],
        labels,
        predictions,
        probabilities,
        paths,
        tta=False,
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["correct"] == 3
    assert metrics["total"] == 4
    assert metrics["misclassified"] == 1
    assert "held-out" in metrics["note"]
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 0
    text = (tmp_path / "misclassified.csv").read_text(encoding="utf-8-sig")
    body = "\n".join(text.splitlines()[1:])
    assert "b.jpg" in body
    assert "a.jpg" not in body


def test_low_confidence_is_unknown():
    text = format_prediction(
        Path("photo.jpg"),
        ["mug", "keys"],
        [0.42, 0.58],
        top_k=2,
        threshold=0.6,
    )
    assert "decision: unknown" in text
    assert "mug" in text


def test_high_confidence_names_the_class():
    text = format_prediction(
        Path("photo.jpg"),
        ["mug", "keys"],
        [0.1, 0.9],
        top_k=2,
        threshold=0.6,
    )
    assert "decision: keys" in text

"""End-to-end test on synthetic shapes. No personal photos and no download."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "command failed: "
            + " ".join(command)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def test_help_mentions_usage():
    for script in ("train.py", "evaluate.py", "predict.py", "export_onnx.py"):
        result = run([sys.executable, str(ROOT / script), "--help"])
        assert "usage" in result.stdout.lower()


def test_pipeline_on_synthetic_shapes(tmp_path: Path):
    data = tmp_path / "data"
    out = tmp_path / "run"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_synthetic_data.py"),
            "--out",
            str(data),
            "--per-class",
            "20",
            "--image-size",
            "64",
            "--seed",
            "0",
        ]
    )
    trained = run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--data",
            str(data),
            "--config",
            str(ROOT / "config.smoke.yaml"),
            "--output-dir",
            str(out),
        ]
    )
    assert "Device:" in trained.stdout
    if "Device: cuda" not in trained.stdout:
        assert "full precision" in trained.stdout
    assert "[head]" in trained.stdout
    assert "[full]" in trained.stdout
    assert "Near-duplicate check:" in trained.stdout
    assert "Held-out test accuracy:" in trained.stdout
    assert (out / "best.pt").is_file()
    assert (out / "last.pt").is_file()

    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert {row["stage"] for row in history} == {"head", "full"}

    rows = list(csv.DictReader((out / "split_manifest.csv").open(encoding="utf-8-sig")))
    grouped = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        grouped[row["split"]].add(row["path"])
    assert grouped["train"].isdisjoint(grouped["val"])
    assert grouped["train"].isdisjoint(grouped["test"])
    assert grouped["val"].isdisjoint(grouped["test"])

    metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["total"] == len(grouped["test"])
    assert metrics["total"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] >= 0.8
    assert "held-out" in metrics["note"]
    assert (out / "confusion_matrix.png").stat().st_size > 0

    evaluated = run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--checkpoint",
            str(out / "best.pt"),
            "--config",
            str(ROOT / "config.smoke.yaml"),
        ]
    )
    assert "Held-out test accuracy:" in evaluated.stdout

    test_image = sorted(grouped["test"])[0]
    predicted = run(
        [
            sys.executable,
            str(ROOT / "predict.py"),
            test_image,
            "--checkpoint",
            str(out / "best.pt"),
            "--config",
            str(ROOT / "config.smoke.yaml"),
        ]
    )
    assert "decision:" in predicted.stdout

    onnx_path = out / "model.onnx"
    run(
        [
            sys.executable,
            str(ROOT / "export_onnx.py"),
            "--checkpoint",
            str(out / "best.pt"),
            "--output",
            str(onnx_path),
        ]
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    labels = json.loads((out / "model.labels.json").read_text(encoding="utf-8"))
    assert labels == ["circle", "square", "triangle"]

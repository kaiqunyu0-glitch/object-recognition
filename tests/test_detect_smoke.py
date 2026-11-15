"""End-to-end detection test on synthetic shapes. CPU only, no weight download."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            "command failed: "
            + " ".join(command)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def test_detection_help():
    for script in ("detect_train.py", "detect_eval.py", "detect_predict.py"):
        result = run([sys.executable, str(ROOT / script), "--help"])
        assert "usage" in result.stdout.lower()
        assert "yolo11m.pt" in result.stdout
    train_help = run([sys.executable, str(ROOT / "detect_train.py"), "--help"]).stdout
    for name in ("yolo11n.pt", "yolo11s.pt", "yolo11l.pt", "yolo11x.pt"):
        assert name in train_help


def test_detection_pipeline(tmp_path: Path):
    data = tmp_path / "data"
    out = tmp_path / "run"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_synthetic_detection.py"),
            "--out",
            str(data),
            "--images",
            "12",
            "--image-size",
            "128",
            "--seed",
            "0",
        ]
    )
    trained = run(
        [
            sys.executable,
            str(ROOT / "detect_train.py"),
            "--data",
            str(data),
            "--config",
            str(ROOT / "config.detect.smoke.yaml"),
            "--output-dir",
            str(out),
        ]
    )
    assert "Device:" in trained.stdout
    assert "Near-duplicate check:" in trained.stdout
    assert "Best checkpoint:" in trained.stdout
    if "Device: cuda" not in trained.stdout:
        assert "full precision" in trained.stdout
    best = out / "train" / "weights" / "best.pt"
    assert best.is_file()

    evaluated = run(
        [
            sys.executable,
            str(ROOT / "detect_eval.py"),
            "--output-dir",
            str(out),
            "--config",
            str(ROOT / "config.detect.smoke.yaml"),
        ]
    )
    assert "mAP50:" in evaluated.stdout
    assert "Held-out test set" in evaluated.stdout
    metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert 0.0 <= metrics["map50"] <= 1.0
    assert 0.0 <= metrics["map50_95"] <= 1.0
    assert metrics["images"] > 0
    assert "held-out" in metrics["note"].lower() or "Held-out" in metrics["note"]
    assert (out / "confusion_matrix.png").stat().st_size > 0
    assert (out / "errors.csv").is_file()

    test_image = sorted((out / "dataset" / "images" / "test").glob("*.jpg"))[0]
    predicted = run(
        [
            sys.executable,
            str(ROOT / "detect_predict.py"),
            str(test_image),
            "--output-dir",
            str(out),
            "--config",
            str(ROOT / "config.detect.smoke.yaml"),
        ]
    )
    assert "box=" in predicted.stdout or "no objects" in predicted.stdout
    assert "Saved annotated output" in predicted.stdout
    saved = list((out / "predictions").rglob("*.jpg"))
    assert saved
    assert saved[0].stat().st_size > 0

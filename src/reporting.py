"""Write the held-out test report: numbers, a confusion matrix, and mistakes.

Accuracy alone hides a model that is perfect on mugs and useless on keys.
The confusion matrix and the misclassified list show which objects to
photograph again.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils import write_json

NO_TEST_MESSAGE = (
    "No held-out test photos.\n"
    "Refusing to report a test accuracy.\n"
    "Validation accuracy was already used to pick the best checkpoint, "
    "so it is not an honest final score.\n"
    "Add a test folder, or put all photos in class folders and let the program split them."
)


def _configure_cjk_font() -> None:
    """Use a Traditional Chinese font when the machine has one.

    Class folders may be named in Chinese. Without a matching font the
    confusion matrix would draw empty boxes for those names.
    """
    from matplotlib import font_manager

    candidates = [
        "PingFang TC",
        "Heiti TC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Microsoft JhengHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def clear_test_reports(report_dir: Path) -> None:
    """Remove a previous test report so an old score cannot be mistaken for a new one."""
    for name in (
        "test_metrics.json",
        "confusion_matrix.png",
        "misclassified.csv",
        "classification_report.txt",
    ):
        stale = report_dir / name
        if stale.is_file():
            stale.unlink()


def write_report(
    report_dir: Path,
    class_names: list[str],
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    paths: list[str],
    tta: bool,
) -> dict:
    """Write every test artifact and return the metrics dictionary."""
    if not labels:
        raise RuntimeError("There are no test predictions to score.")
    report_dir.mkdir(parents=True, exist_ok=True)
    _configure_cjk_font()

    label_ids = list(range(len(class_names)))
    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    accuracy = float(accuracy_score(y_true, y_pred))
    correct = int((y_true == y_pred).sum())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    mistakes = _misclassified(class_names, labels, predictions, probabilities, paths)

    metrics = {
        "split": "test",
        "accuracy": accuracy,
        "correct": correct,
        "total": int(len(labels)),
        "macro_f1": float(np.mean([item["f1"] for item in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "misclassified": len(mistakes),
        "tta": tta,
        "note": (
            "This score uses only the held-out test photos. "
            "They were not used for training or for choosing the best checkpoint."
        ),
    }
    write_json(report_dir / "test_metrics.json", metrics)

    text = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    (report_dir / "classification_report.txt").write_text(text, encoding="utf-8")
    _write_confusion_matrix(report_dir / "confusion_matrix.png", matrix, class_names)
    _write_mistakes(report_dir / "misclassified.csv", mistakes)
    return metrics


def _misclassified(
    class_names: list[str],
    labels: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    paths: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, true_index, pred_index, probs in zip(paths, labels, predictions, probabilities):
        if true_index == pred_index:
            continue
        # Confidence of the wrong guess. A confident mistake is the first
        # photo to look at: the model was sure, and it was wrong.
        rows.append(
            {
                "path": path,
                "true_label": class_names[true_index],
                "predicted_label": class_names[pred_index],
                "confidence": float(probs[pred_index]),
            }
        )
    rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return rows


def _write_mistakes(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "true_label", "predicted_label", "confidence"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence": f"{float(row['confidence']):.6f}",
                }
            )


def _write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    count = len(class_names)
    figure, axis = plt.subplots(figsize=(max(6.0, count * 0.55), max(5.0, count * 0.5)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(count), class_names, rotation=45, ha="right")
    axis.set_yticks(range(count), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix (rows are the true class)")
    peak = matrix.max() if matrix.size else 0
    for row in range(count):
        for col in range(count):
            value = int(matrix[row, col])
            color = "white" if peak and value > peak / 2 else "black"
            axis.text(col, row, str(value), ha="center", va="center", color=color)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def print_test_report(metrics: dict, report_dir: Path) -> None:
    """Print the score a person should trust, plus where the files went."""
    print("", flush=True)
    print("=" * 64, flush=True)
    print("Held-out test set", flush=True)
    print("These photos were not used for training, and they were not used", flush=True)
    print("to choose the best checkpoint. That choice used validation only.", flush=True)
    print("=" * 64, flush=True)
    print(
        f"Held-out test accuracy: {metrics['accuracy']:.1%}  "
        f"({metrics['correct']}/{metrics['total']})",
        flush=True,
    )
    if metrics.get("tta"):
        print("Test-time augmentation was on: each photo was also scored flipped.", flush=True)
    for name, scores in metrics["per_class"].items():
        print(
            f"  {name:<16} precision={scores['precision']:.1%}  "
            f"recall={scores['recall']:.1%}  support={scores['support']}",
            flush=True,
        )
    print(f"Confusion matrix: {report_dir / 'confusion_matrix.png'}", flush=True)
    print(
        f"Misclassified: {metrics['misclassified']}. "
        f"List: {report_dir / 'misclassified.csv'}",
        flush=True,
    )
    print(f"Full numbers: {report_dir / 'test_metrics.json'}", flush=True)

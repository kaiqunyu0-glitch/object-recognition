"""Write the held-out detection report.

mAP is not a single 'accuracy'. mAP50 asks whether the box landed on the
right object. mAP50-95 also asks whether the box fits tightly. Precision
counts extra boxes. Recall counts objects the model missed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.detect.boxes import iou_xyxy, parse_yolo_text, yolo_to_xyxy
from src.utils import write_json

NO_TEST_MESSAGE = (
    "No held-out test photos.\n"
    "Refusing to report a test mAP.\n"
    "Validation mAP was already used to pick the best checkpoint, "
    "so it is not an honest final score.\n"
    "Add a test folder, or give one folder of photos and let the program split them."
)


def match_detections(
    ground_truth: list[tuple[int, float, float, float, float]],
    predictions: list[tuple[int, float, float, float, float, float]],
    match_iou: float,
) -> tuple[list[int], list[int]]:
    """Greedy same-class matching.

    ``ground_truth`` items are ``(class, x1, y1, x2, y2)``.
    ``predictions`` items are ``(class, confidence, x1, y1, x2, y2)``,
    highest confidence first.

    Returns the indexes of unmatched ground-truth boxes and unmatched
    predictions. A box on the wrong class does not count as a match, so it
    shows up as both a miss and a false positive.
    """
    used_truth: set[int] = set()
    matched_predictions: set[int] = set()
    order = sorted(range(len(predictions)), key=lambda index: predictions[index][1], reverse=True)
    for prediction_index in order:
        pred_class, _confidence, *pred_box = predictions[prediction_index]
        best_iou = match_iou
        best_truth = None
        for truth_index, truth in enumerate(ground_truth):
            if truth_index in used_truth or truth[0] != pred_class:
                continue
            score = iou_xyxy(tuple(pred_box), truth[1:])
            if score >= best_iou:
                best_iou = score
                best_truth = truth_index
        if best_truth is not None:
            used_truth.add(best_truth)
            matched_predictions.add(prediction_index)
    missed = [index for index in range(len(ground_truth)) if index not in used_truth]
    false_positives = [index for index in range(len(predictions)) if index not in matched_predictions]
    return missed, false_positives


def confusion_from_matches(
    ground_truth: list[tuple[int, float, float, float, float]],
    predictions: list[tuple[int, float, float, float, float, float]],
    class_count: int,
    match_iou: float,
) -> np.ndarray:
    """Build a matrix indexed ``[true class, predicted class]``, plus background.

    Rows are the true class and columns are the prediction. The last column
    is a miss (the object was there, no box landed on it). The last row is a
    false positive (a box was drawn on the background).
    """
    matrix = np.zeros((class_count + 1, class_count + 1), dtype=int)
    background = class_count
    used_truth: set[int] = set()
    order = sorted(range(len(predictions)), key=lambda index: predictions[index][1], reverse=True)
    for prediction_index in order:
        pred_class, _confidence, *pred_box = predictions[prediction_index]
        best_iou = match_iou
        best_truth = None
        for truth_index, truth in enumerate(ground_truth):
            if truth_index in used_truth:
                continue
            score = iou_xyxy(tuple(pred_box), truth[1:])
            if score >= best_iou:
                best_iou = score
                best_truth = truth_index
        if best_truth is None:
            if 0 <= pred_class < class_count:
                matrix[background, pred_class] += 1
            continue
        used_truth.add(best_truth)
        truth_class = ground_truth[best_truth][0]
        if 0 <= truth_class < class_count and 0 <= pred_class < class_count:
            matrix[truth_class, pred_class] += 1
    for truth_index, truth in enumerate(ground_truth):
        if truth_index in used_truth:
            continue
        if 0 <= truth[0] < class_count:
            matrix[truth[0], background] += 1
    return matrix


def write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    labels = list(class_names) + ["background"]
    count = len(labels)
    figure, axis = plt.subplots(figsize=(max(6.0, count * 0.6), max(5.0, count * 0.55)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(count), labels, rotation=45, ha="right")
    axis.set_yticks(range(count), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Detection confusion matrix")
    peak = int(matrix.max()) if matrix.size else 0
    for row in range(count):
        for col in range(count):
            value = int(matrix[row, col])
            color = "white" if peak and value > peak / 2 else "black"
            axis.text(col, row, str(value), ha="center", va="center", color=color)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_error_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "issue", "class_name", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(rows)


def boxes_for_image(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO labels as pixel boxes. Missing files mean no objects."""
    if not label_path.is_file():
        return []
    parsed = parse_yolo_text(label_path.read_text(encoding="utf-8"), str(label_path))
    return [yolo_to_xyxy(box, image_width, image_height) for box in parsed]


def per_class_table(class_names: list[str], box_metrics) -> dict[str, dict[str, float]]:
    """Read per-class precision, recall, and AP from an Ultralytics metric object."""
    table = {
        name: {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}
        for name in class_names
    }
    # ap_class_index is a numpy array. Do not use `or` on it: a multi-value
    # array cannot be treated as true or false.
    raw_indexes = getattr(box_metrics, "ap_class_index", None)
    class_indexes = [] if raw_indexes is None else [int(value) for value in list(raw_indexes)]
    for index, class_id in enumerate(class_indexes):
        if not 0 <= int(class_id) < len(class_names):
            continue
        precision, recall, ap50, ap = box_metrics.class_result(index)
        table[class_names[int(class_id)]] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(ap50),
            "map50_95": float(ap),
        }
    return table


def build_metrics_payload(
    class_names: list[str],
    box_metrics,
    error_rows: list[dict[str, object]],
    image_count: int,
) -> dict:
    per_class = per_class_table(class_names, box_metrics)
    problem_images = sorted({str(row["image"]) for row in error_rows})
    return {
        "split": "test",
        "map50": float(box_metrics.map50),
        "map50_95": float(box_metrics.map),
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "per_class": per_class,
        "images": image_count,
        "images_with_errors": len(problem_images),
        "misses": sum(1 for row in error_rows if row["issue"] == "miss"),
        "false_positives": sum(1 for row in error_rows if row["issue"] == "false_positive"),
        "note": (
            "These figures use only the held-out test photos. "
            "They were not used for training or for choosing the best checkpoint. "
            "mAP sweeps confidence. The miss and false-positive list uses the "
            "confidence threshold in the config, which is the threshold predict.py uses."
        ),
    }


def print_detection_report(metrics: dict, report_dir: Path) -> None:
    print("", flush=True)
    print("=" * 64, flush=True)
    print("Held-out test set", flush=True)
    print("These photos were not used for training, and they were not used", flush=True)
    print("to choose the best checkpoint. That choice used validation only.", flush=True)
    print("=" * 64, flush=True)
    print(f"mAP50:     {metrics['map50']:.3f}", flush=True)
    print(f"mAP50-95:  {metrics['map50_95']:.3f}", flush=True)
    print(f"precision: {metrics['precision']:.3f}", flush=True)
    print(f"recall:    {metrics['recall']:.3f}", flush=True)
    for name, scores in metrics["per_class"].items():
        print(
            f"  {name:<16} precision={scores['precision']:.3f}  "
            f"recall={scores['recall']:.3f}  mAP50={scores['map50']:.3f}",
            flush=True,
        )
    print(f"Confusion matrix: {report_dir / 'confusion_matrix.png'}", flush=True)
    print(
        f"Images with misses or false positives: {metrics['images_with_errors']}. "
        f"List: {report_dir / 'errors.csv'}",
        flush=True,
    )
    print(f"Full numbers: {report_dir / 'test_metrics.json'}", flush=True)


def save_metrics(report_dir: Path, metrics: dict) -> None:
    write_json(report_dir / "test_metrics.json", metrics)

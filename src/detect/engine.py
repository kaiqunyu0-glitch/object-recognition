"""Fine-tune a YOLO detector and score the held-out test photos.

Ultralytics uses the validation split to decide which epoch becomes
``best.pt``. This module does not look at the test split until
``evaluate_detector`` runs, after that choice is finished.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from src.detect.dataset import prepare_detection_dataset
from src.detect.report import (
    NO_TEST_MESSAGE,
    boxes_for_image,
    build_metrics_payload,
    confusion_from_matches,
    match_detections,
    print_detection_report,
    save_metrics,
    write_confusion_matrix,
    write_error_csv,
)
from src.device import describe_device, get_device
from src.duplicates import find_cross_split_duplicates, report_duplicates
from src.utils import write_json

# Medium COCO weights. Larger YOLO11 models memorize a small fixed-background set.
DEFAULT_MODEL = "yolo11m.pt"


def resolve_train_batch(batch, yolo_device: str) -> int | float:
    """Use Ultralytics AutoBatch on NVIDIA GPUs. CPU and MPS get a small fixed batch.

    ``batch=-1`` asks Ultralytics to profile the GPU and pick a size that fits.
    That profiler does not run on CPU or Apple MPS. A large model at batch 16
    would also be a poor default there, so those devices use 2 images at a time.
    A positive ``batch`` is kept as written.
    """
    if isinstance(batch, str) and batch.strip().lower() in {"auto", "-1"}:
        batch = -1
    numeric = float(batch)
    if numeric == -1 or 0 < numeric < 1:
        if yolo_device == "0":
            print("Batch: auto. Ultralytics will pick a size that fits the GPU.", flush=True)
            return numeric
        print(
            "Auto batch only runs on an NVIDIA GPU. Using batch=2 on this device. "
            "On CPU, yolo11n.pt or yolo11s.pt is faster than the default yolo11m.pt.",
            flush=True,
        )
        return 2
    if numeric < 1:
        raise ValueError("batch must be -1 (auto) or a positive integer.")
    return int(numeric)


def resolve_eval_batch(batch, yolo_device: str) -> int:
    """Evaluation needs a fixed batch. Auto (-1) becomes 4 on CUDA and 2 elsewhere."""
    if isinstance(batch, str) and batch.strip().lower() in {"auto", "-1"}:
        batch = -1
    numeric = float(batch)
    if numeric >= 1:
        return int(numeric)
    return 4 if yolo_device == "0" else 2


def ultralytics_device(requested: str | None) -> str:
    """Map our device name onto the string Ultralytics expects."""
    device = get_device(requested)
    if device.type == "cuda":
        return "0"
    return device.type


def _silence_ultralytics_sync() -> None:
    """Skip the Ultralytics phone-home check so training starts offline."""
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS["sync"] = False
    except Exception:
        return


def _load_yolo(model_name: str):
    _silence_ultralytics_sync()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics first: pip install ultralytics") from exc
    try:
        return YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {model_name}. "
            "Check the name (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, yolov8n.pt) "
            "and your network if this is the first time those weights are downloaded. "
            "yolo11n.yaml trains a tiny model from scratch and does not download weights."
        ) from exc


def _run_record(output_dir: Path) -> dict:
    path = output_dir / "detect_run.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Train first:\n"
            "  python detect_train.py --data /path/to/dataset"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def fit_detector(cfg: dict, data_root: Path) -> Path:
    """Prepare the dataset, train, and save ``best.pt``. The test set is not scored here."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_detection_dataset(data_root, cfg, output_dir)
    device = get_device(str(cfg.get("device", "auto")))
    print(f"Device: {describe_device(device)}", flush=True)
    print(f"Layout: {prepared.layout}", flush=True)
    for message in prepared.warnings:
        print(f"Warning: {message}", flush=True)
    print(f"Classes ({len(prepared.class_names)}): {', '.join(prepared.class_names)}", flush=True)
    counts = Counter(row["split"] for row in prepared.rows)
    print(
        f"Images: train={counts['train']} val={counts['val']} test={counts['test']}",
        flush=True,
    )

    if cfg["duplicates"]["enabled"]:
        print(
            f"Checking {len(prepared.rows)} photos for near-duplicates across splits...",
            flush=True,
        )
        pairs = find_cross_split_duplicates(
            prepared.rows,
            hash_size=int(cfg["duplicates"]["hash_size"]),
            max_distance=int(cfg["duplicates"]["max_distance"]),
            max_side=int(cfg.get("max_side", 1600)),
        )
        report_duplicates(pairs, output_dir / "duplicates.csv")
    else:
        print("Near-duplicate check: skipped (duplicates.enabled is false).", flush=True)

    model_name = str(cfg.get("model") or DEFAULT_MODEL)
    yolo_device = ultralytics_device(str(cfg.get("device", "auto")))
    train_batch = resolve_train_batch(cfg["batch"], yolo_device)
    use_amp = bool(cfg["train"]["amp"]) and yolo_device == "0"
    if cfg["train"]["amp"] and yolo_device != "0":
        print(
            "Mixed precision stays off: it is only used on NVIDIA GPUs. Training in full precision.",
            flush=True,
        )
    print(f"Model: {model_name}", flush=True)
    print(
        "Stage: fine-tune the detector. Validation mAP picks best.pt. "
        "The test split is not used for that choice.",
        flush=True,
    )
    model = _load_yolo(model_name)
    train_kwargs = {
        "data": str(prepared.yaml_path),
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": train_batch,
        "device": yolo_device,
        "workers": int(cfg["workers"]),
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
        "patience": int(cfg["patience"]),
        "seed": int(cfg["seed"]),
        "plots": bool(cfg["train"]["plots"]),
        "mosaic": float(cfg["train"]["mosaic"]),
        "amp": use_amp,
        "pretrained": model_name.endswith(".pt"),
        "verbose": True,
        "val": True,
        "cache": False,
        "save": True,
    }
    model.train(**train_kwargs)
    best = output_dir / "train" / "weights" / "best.pt"
    last = output_dir / "train" / "weights" / "last.pt"
    if not best.is_file():
        if last.is_file():
            best = last
        else:
            raise RuntimeError("Training finished without writing best.pt or last.pt.")
    record = {
        "data_yaml": str(prepared.yaml_path.resolve()),
        "best": str(best.resolve()),
        "last": str(last.resolve()) if last.is_file() else "",
        "class_names": prepared.class_names,
        "model_name": model_name,
        "imgsz": int(cfg["imgsz"]),
        "output_dir": str(output_dir.resolve()),
    }
    write_json(output_dir / "detect_run.json", record)
    print(f"Best checkpoint: {best}", flush=True)
    print("Run detect_eval.py for the held-out test mAP.", flush=True)
    return best


def _prediction_rows(result) -> list[tuple[int, float, float, float, float, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    rows = []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    for class_id, confidence, coords in zip(classes, confidences, xyxy):
        x1, y1, x2, y2 = coords
        rows.append((int(class_id), float(confidence), float(x1), float(y1), float(x2), float(y2)))
    return rows


def evaluate_detector(cfg: dict, checkpoint: Path | None, conf: float | None, match_iou: float | None) -> dict | None:
    """Score ``best.pt`` on the test split only."""
    output_dir = Path(cfg["output_dir"])
    record = _run_record(output_dir)
    data_yaml = Path(record["data_yaml"])
    loaded = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not loaded.get("test"):
        print(NO_TEST_MESSAGE, flush=True)
        return None
    dataset_root = Path(loaded["path"])
    test_images = sorted((dataset_root / "images" / "test").glob("*.jpg"))
    if not test_images:
        print(NO_TEST_MESSAGE, flush=True)
        return None
    weights = checkpoint or Path(record["best"])
    if not weights.is_file():
        raise FileNotFoundError(f"Could not find weights at {weights}.")
    yolo_device = ultralytics_device(str(cfg.get("device", "auto")))
    eval_batch = resolve_eval_batch(cfg["batch"], yolo_device)
    model = _load_yolo(str(weights))
    print(f"Device: {describe_device(get_device(str(cfg.get('device', 'auto'))))}", flush=True)
    print(f"Model: {weights}", flush=True)
    # Do not pass the predict confidence here. mAP integrates over the
    # confidence curve, and Ultralytics uses a very low threshold for that.
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(record.get("imgsz", cfg["imgsz"])),
        batch=eval_batch,
        device=yolo_device,
        workers=int(cfg["workers"]),
        plots=False,
        project=str(output_dir),
        name="test_eval",
        exist_ok=True,
        verbose=False,
    )
    class_names = list(record["class_names"])
    operating_conf = float(cfg["conf"] if conf is None else conf)
    iou_threshold = float(cfg["match_iou"] if match_iou is None else match_iou)
    error_rows: list[dict[str, object]] = []
    predictions = model.predict(
        source=[str(path) for path in test_images],
        conf=operating_conf,
        iou=float(cfg["iou"]),
        imgsz=int(record.get("imgsz", cfg["imgsz"])),
        device=yolo_device,
        stream=True,
        verbose=False,
        save=False,
    )
    accumulated = None
    for image_path, result in zip(test_images, predictions):
        label_path = dataset_root / "labels" / "test" / f"{image_path.stem}.txt"
        width = int(result.orig_shape[1])
        height = int(result.orig_shape[0])
        truth = boxes_for_image(label_path, width, height)
        preds = _prediction_rows(result)
        missed, false_positives = match_detections(truth, preds, iou_threshold)
        image_matrix = confusion_from_matches(truth, preds, len(class_names), iou_threshold)
        accumulated = image_matrix if accumulated is None else accumulated + image_matrix
        for index in missed:
            class_id, x1, y1, x2, y2 = truth[index]
            error_rows.append(
                {
                    "image": str(image_path),
                    "issue": "miss",
                    "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                    "confidence": "",
                    "x1": f"{x1:.1f}",
                    "y1": f"{y1:.1f}",
                    "x2": f"{x2:.1f}",
                    "y2": f"{y2:.1f}",
                }
            )
        for index in false_positives:
            class_id, confidence, x1, y1, x2, y2 = preds[index]
            error_rows.append(
                {
                    "image": str(image_path),
                    "issue": "false_positive",
                    "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                    "confidence": f"{confidence:.4f}",
                    "x1": f"{x1:.1f}",
                    "y1": f"{y1:.1f}",
                    "x2": f"{x2:.1f}",
                    "y2": f"{y2:.1f}",
                }
            )
    matrix = accumulated
    if matrix is None:
        import numpy as np

        matrix = np.zeros((len(class_names) + 1, len(class_names) + 1), dtype=int)
    report_dir = output_dir
    write_confusion_matrix(report_dir / "confusion_matrix.png", matrix, class_names)
    write_error_csv(report_dir / "errors.csv", error_rows)
    payload = build_metrics_payload(class_names, metrics.box, error_rows, len(test_images))
    payload["confidence_threshold"] = operating_conf
    payload["match_iou"] = iou_threshold
    save_metrics(report_dir, payload)
    print_detection_report(payload, report_dir)
    return payload


def _print_result_lines(result, title: str) -> None:
    names = result.names or {}
    print(title, flush=True)
    rows = _prediction_rows(result)
    if not rows:
        print("  (no objects above the confidence threshold)", flush=True)
        return
    for class_id, confidence, x1, y1, x2, y2 in rows:
        label = names.get(class_id, str(class_id))
        print(
            f"  {label:<16} {confidence * 100:5.1f}%  "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
            flush=True,
        )


def predict_detector(
    cfg: dict,
    source: str,
    checkpoint: Path | None,
    conf: float | None,
) -> None:
    """Draw boxes for a photo, a folder, a video, or a webcam index."""
    output_dir = Path(cfg["output_dir"])
    record = _run_record(output_dir)
    weights = checkpoint or Path(record["best"])
    if not Path(weights).is_file():
        raise FileNotFoundError(f"Could not find weights at {weights}.")
    yolo_device = ultralytics_device(str(cfg.get("device", "auto")))
    model = _load_yolo(str(weights))
    operating_conf = float(cfg["conf"] if conf is None else conf)
    predict_source: str | int = int(source) if source.isdigit() else source
    print(f"Device: {describe_device(get_device(str(cfg.get('device', 'auto'))))}", flush=True)
    print(f"Model: {weights}", flush=True)
    print(f"Confidence threshold: {operating_conf:.0%}", flush=True)
    results = model.predict(
        source=predict_source,
        conf=operating_conf,
        iou=float(cfg["iou"]),
        imgsz=int(record.get("imgsz", cfg["imgsz"])),
        device=yolo_device,
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
        stream=True,
        verbose=False,
    )
    save_dir = None
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    is_video = isinstance(predict_source, str) and Path(predict_source).suffix.lower() in video_suffixes
    for index, result in enumerate(results):
        save_dir = getattr(result, "save_dir", save_dir)
        path = str(getattr(result, "path", "") or predict_source)
        if isinstance(predict_source, int) or is_video:
            title = f"frame {index}: {path}"
        else:
            title = path
        _print_result_lines(result, title)
    if save_dir is not None:
        print(f"Saved annotated output to {save_dir}", flush=True)

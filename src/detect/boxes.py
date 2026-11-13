"""Read and write bounding boxes.

YOLO stores one object per line: ``class cx cy w h``, all four numbers
normalized to the image width and height, with the origin at the center of
the box. COCO and Label Studio use a top-left box instead, so the converters
below turn those into the YOLO line.
"""

from __future__ import annotations

import json
from pathlib import Path


def clip_yolo_box(
    cx: float,
    cy: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    """Keep a normalized center box inside the image. Drop empty boxes."""
    x1 = max(0.0, cx - width / 2)
    y1 = max(0.0, cy - height / 2)
    x2 = min(1.0, cx + width / 2)
    y2 = min(1.0, cy + height / 2)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w < 1e-4 or clipped_h < 1e-4:
        return None
    return (x1 + clipped_w / 2, y1 + clipped_h / 2, clipped_w, clipped_h)


def parse_yolo_text(text: str, source: str) -> list[tuple[int, float, float, float, float]]:
    """Parse YOLO label text into ``(class, cx, cy, w, h)`` tuples."""
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{source}:{line_number} needs 'class cx cy w h'. Found: {line!r}")
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number} has a non-numeric box: {line!r}") from exc
        if class_id < 0:
            raise ValueError(f"{source}:{line_number} has a negative class id.")
        clipped = clip_yolo_box(cx, cy, width, height)
        if clipped is None:
            continue
        boxes.append((class_id, *clipped))
    return boxes


def format_yolo(boxes: list[tuple[int, float, float, float, float]]) -> str:
    """Write boxes the way YOLO expects to read them back."""
    lines = [f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}" for class_id, cx, cy, width, height in boxes]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def yolo_to_xyxy(
    box: tuple[int, float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float]:
    """Convert one YOLO box to pixel ``(class, x1, y1, x2, y2)``."""
    class_id, cx, cy, width, height = box
    return (
        class_id,
        (cx - width / 2) * image_width,
        (cy - height / 2) * image_height,
        (cx + width / 2) * image_width,
        (cy + height / 2) * image_height,
    )


def iou_xyxy(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    """Intersection over union for two pixel boxes ``(x1, y1, x2, y2)``."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def xywh_top_left_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    """Convert a top-left absolute box to a normalized YOLO box."""
    if image_width <= 0 or image_height <= 0:
        return None
    return clip_yolo_box(
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def percent_top_left_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    """Label Studio stores the top-left box as percentages from 0 to 100."""
    return clip_yolo_box(
        (x + width / 2) / 100.0,
        (y + height / 2) / 100.0,
        width / 100.0,
        height / 100.0,
    )


def load_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def sniff_json(payload) -> str | None:
    """Return ``coco`` or ``labelstudio`` when the file looks like one of those."""
    if isinstance(payload, dict) and "images" in payload and "annotations" in payload:
        return "coco"
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        task = payload[0]
        if "annotations" in task or "result" in task or "data" in task:
            return "labelstudio"
    return None


def coco_to_boxes(payload: dict, image_sizes: dict[int, tuple[int, int]]) -> tuple[dict[int, list], list[str]]:
    """Return ``image_id -> yolo boxes`` and the class-name list.

    COCO category ids are often 1, 2, 3 rather than 0, 1, 2. They are remapped
    to a dense 0-based list in category-id order.
    """
    categories = payload.get("categories") or []
    if not categories:
        raise ValueError("COCO JSON has no categories.")
    ordered = sorted(categories, key=lambda item: int(item["id"]))
    id_map = {int(item["id"]): index for index, item in enumerate(ordered)}
    names = [str(item["name"]) for item in ordered]
    boxes_by_image: dict[int, list] = {int(image["id"]): [] for image in payload.get("images", [])}
    for annotation in payload.get("annotations", []):
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        image_id = int(annotation["image_id"])
        if image_id not in image_sizes:
            continue
        category_id = int(annotation["category_id"])
        if category_id not in id_map:
            raise ValueError(f"COCO annotation uses unknown category id {category_id}.")
        image_width, image_height = image_sizes[image_id]
        converted = xywh_top_left_to_yolo(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), image_width, image_height)
        if converted is None:
            continue
        boxes_by_image.setdefault(image_id, []).append((id_map[category_id], *converted))
    return boxes_by_image, names


def label_studio_results(task: dict) -> list[tuple[str, float, float, float, float]]:
    """Return ``(class_name, cx, cy, w, h)`` from one Label Studio task."""
    annotations = task.get("annotations") or []
    if not annotations and task.get("result"):
        annotations = [{"result": task["result"]}]
    if not annotations:
        return []
    # The last annotation is the newest save in Label Studio.
    results = annotations[-1].get("result") or []
    boxes: list[tuple[str, float, float, float, float]] = []
    for result in results:
        if result.get("type") not in {"rectanglelabels", "rectangle"}:
            continue
        value = result.get("value") or {}
        names = value.get("rectanglelabels") or value.get("labels") or []
        if not names:
            continue
        try:
            x = float(value["x"])
            y = float(value["y"])
            width = float(value["width"])
            height = float(value["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if max(x, y, width, height) > 100:
            original_width = float(result.get("original_width") or value.get("original_width") or 0)
            original_height = float(result.get("original_height") or value.get("original_height") or 0)
            converted = xywh_top_left_to_yolo(x, y, width, height, original_width, original_height)
        else:
            converted = percent_top_left_to_yolo(x, y, width, height)
        if converted is None:
            continue
        boxes.append((str(names[0]), *converted))
    return boxes

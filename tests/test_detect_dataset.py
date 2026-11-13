"""Tests for detection labels, imports, splits, and EXIF. No training."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.config import load_config
from src.detect.boxes import iou_xyxy, percent_top_left_to_yolo
from src.detect.dataset import prepare_detection_dataset
from src.detect.engine import resolve_eval_batch, resolve_train_batch
from src.detect.report import match_detections
from src.detect.synthetic import make_detection_dataset

ROOT = Path(__file__).resolve().parents[1]


def _jpeg(path: Path, size: tuple[int, int], color: tuple[int, int, int] = (20, 20, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _config() -> dict:
    return load_config(ROOT / "config.detect.yaml")


def test_default_detector_is_yolo11x_and_lists_the_smaller_models():
    text = (ROOT / "config.detect.yaml").read_text(encoding="utf-8")
    assert "model: yolo11x.pt" in text
    for name in ("yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11n.pt", "yolov8n.pt"):
        assert name in text
    assert load_config(ROOT / "config.detect.yaml")["model"] == "yolo11x.pt"
    assert load_config(ROOT / "config.detect.yaml")["batch"] == -1


def test_auto_batch_is_only_used_on_nvidia():
    assert resolve_train_batch(-1, "0") == -1
    assert resolve_train_batch(-1, "cpu") == 2
    assert resolve_train_batch(-1, "mps") == 2
    assert resolve_train_batch(8, "cpu") == 8
    assert resolve_eval_batch(-1, "0") == 4
    assert resolve_eval_batch(-1, "cpu") == 2
    assert resolve_eval_batch(8, "mps") == 8


def test_percent_box_becomes_a_yolo_center():
    cx, cy, width, height = percent_top_left_to_yolo(10, 20, 30, 40)
    assert round(cx, 2) == 0.25
    assert round(cy, 2) == 0.40
    assert round(width, 2) == 0.30
    assert round(height, 2) == 0.40


def test_iou_of_the_same_box_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou_xyxy(box, box) == 1.0


def test_same_class_match_marks_a_miss_and_a_false_positive():
    truth = [(0, 0.0, 0.0, 10.0, 10.0)]
    predictions = [(1, 0.9, 0.0, 0.0, 10.0, 10.0)]
    missed, false_positives = match_detections(truth, predictions, match_iou=0.5)
    assert missed == [0]
    assert false_positives == [0]


def test_flat_yolo_folder_is_split_and_disjoint(tmp_path: Path):
    make_detection_dataset(tmp_path / "raw", images=12, image_size=128, seed=0)
    prepared = prepare_detection_dataset(tmp_path / "raw", _config(), tmp_path / "out")
    assert prepared.class_names == ["circle", "square", "triangle"]
    grouped = {"train": set(), "val": set(), "test": set()}
    for row in prepared.rows:
        grouped[row["split"]].add(row["path"])
    assert grouped["train"] and grouped["val"] and grouped["test"]
    assert grouped["train"].isdisjoint(grouped["val"])
    assert grouped["train"].isdisjoint(grouped["test"])
    assert grouped["val"].isdisjoint(grouped["test"])


def test_existing_split_folders_stay_put(tmp_path: Path):
    for split in ("train", "val", "test"):
        image = tmp_path / "images" / split / f"{split}.jpg"
        _jpeg(image, (32, 32))
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("widget\n", encoding="utf-8")
    prepared = prepare_detection_dataset(tmp_path, _config(), tmp_path / "out")
    found = {path.stem: path.parent.name for path in (prepared.root / "images").rglob("*.jpg")}
    assert found == {"train": "train", "val": "val", "test": "test"}


def test_exif_orientation_is_baked_in_and_the_label_is_kept(tmp_path: Path):
    wide = Image.new("RGB", (40, 16), (255, 0, 0))
    wide.putpixel((0, 0), (0, 0, 255))
    exif = wide.getexif()
    exif[274] = 6
    image_path = tmp_path / "images" / "turned.jpg"
    image_path.parent.mkdir(parents=True)
    wide.save(image_path, exif=exif)
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "turned.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    for name in ("b", "c"):
        _jpeg(tmp_path / "images" / f"{name}.jpg", (32, 32))
        (tmp_path / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("widget\n", encoding="utf-8")
    prepared = prepare_detection_dataset(tmp_path, _config(), tmp_path / "out")
    saved = list((prepared.root / "images").rglob("turned.jpg"))
    assert len(saved) == 1
    assert Image.open(saved[0]).size == (16, 40)
    label = (prepared.root / "labels" / saved[0].parent.name / "turned.txt").read_text(encoding="utf-8")
    assert label.startswith("0 0.500000 0.500000")


def test_coco_json_remaps_category_ids(tmp_path: Path):
    _jpeg(tmp_path / "images" / "a.jpg", (40, 40))
    _jpeg(tmp_path / "images" / "b.jpg", (40, 40))
    _jpeg(tmp_path / "images" / "c.jpg", (40, 40))
    payload = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 40, "height": 40},
            {"id": 2, "file_name": "b.jpg", "width": 40, "height": 40},
            {"id": 3, "file_name": "c.jpg", "width": 40, "height": 40},
        ],
        "categories": [{"id": 7, "name": "keys"}, {"id": 3, "name": "mug"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 7, "bbox": [0, 0, 10, 10]},
            {"id": 3, "image_id": 3, "category_id": 3, "bbox": [5, 5, 10, 10]},
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    prepared = prepare_detection_dataset(tmp_path, _config(), tmp_path / "out")
    assert prepared.class_names == ["mug", "keys"]
    texts = [path.read_text(encoding="utf-8") for path in (prepared.root / "labels").rglob("*.txt")]
    assert any(text.startswith("0 ") for text in texts)
    assert any(text.startswith("1 ") for text in texts)


def test_label_studio_json(tmp_path: Path):
    _jpeg(tmp_path / "images" / "scene.jpg", (100, 100))
    _jpeg(tmp_path / "images" / "other.jpg", (100, 100))
    _jpeg(tmp_path / "images" / "third.jpg", (100, 100))
    tasks = []
    for name in ("scene.jpg", "other.jpg", "third.jpg"):
        tasks.append(
            {
                "data": {"image": f"/data/upload/1/{name}"},
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "rectanglelabels",
                                "value": {
                                    "x": 10,
                                    "y": 20,
                                    "width": 30,
                                    "height": 40,
                                    "rectanglelabels": ["mug"],
                                },
                            }
                        ]
                    }
                ],
            }
        )
    (tmp_path / "label_studio.json").write_text(json.dumps(tasks), encoding="utf-8")
    prepared = prepare_detection_dataset(tmp_path, _config(), tmp_path / "out")
    assert prepared.class_names == ["mug"]
    text = next((prepared.root / "labels").rglob("*.txt")).read_text(encoding="utf-8")
    assert text.startswith("0 0.250000 0.400000 0.300000 0.400000")

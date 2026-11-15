"""Tests for the split, EXIF orientation, and near-duplicate check."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.config import apply_overrides, load_config
from src.data import build_manifest, read_image, split_counts
from src.duplicates import dhash, find_cross_split_duplicates, hamming

ROOT = Path(__file__).resolve().parents[1]


def _solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _config() -> dict:
    return load_config(ROOT / "config.yaml")


def test_split_counts_cover_every_photo_and_every_split():
    ratios = (0.70, 0.15, 0.15)
    for count in range(3, 40):
        parts = split_counts(count, ratios)
        assert sum(parts) == count
        assert min(parts) >= 1


def test_split_counts_are_deterministic_for_exact_ratios():
    assert split_counts(20, (0.5, 0.25, 0.25)) == (10, 5, 5)


def test_flat_folder_is_stratified_and_disjoint(tmp_path: Path):
    for label in ("mug", "keys"):
        for index in range(6):
            _solid(tmp_path / label / f"{index}.jpg", (16, 16), (index * 20, 0, 0))
    cfg = _config()
    first = build_manifest(tmp_path, cfg, csv_path=None)
    second = build_manifest(tmp_path, cfg, csv_path=None)
    assert first.rows == second.rows
    assert first.class_names == ["keys", "mug"]
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    labels_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in first.rows:
        by_split[row["split"]].add(row["path"])
        labels_by_split[row["split"]].add(row["label"])
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    for split in by_split:
        assert labels_by_split[split] == {"keys", "mug"}


def test_presplit_folders_are_kept(tmp_path: Path):
    _solid(tmp_path / "train" / "cat" / "a.jpg", (8, 8), (1, 0, 0))
    _solid(tmp_path / "train" / "dog" / "b.jpg", (8, 8), (0, 1, 0))
    _solid(tmp_path / "val" / "cat" / "c.jpg", (8, 8), (0, 0, 1))
    _solid(tmp_path / "val" / "dog" / "d.jpg", (8, 8), (1, 1, 0))
    _solid(tmp_path / "test" / "cat" / "e.jpg", (8, 8), (1, 0, 1))
    _solid(tmp_path / "test" / "dog" / "f.jpg", (8, 8), (0, 1, 1))
    manifest = build_manifest(tmp_path, _config(), csv_path=None)
    splits = {Path(row["path"]).name: row["split"] for row in manifest.rows}
    assert splits == {
        "a.jpg": "train",
        "b.jpg": "train",
        "c.jpg": "val",
        "d.jpg": "val",
        "e.jpg": "test",
        "f.jpg": "test",
    }


def test_missing_test_folder_warns_and_does_not_invent_a_score_split(tmp_path: Path):
    for label, color in (("cat", (255, 0, 0)), ("dog", (0, 255, 0))):
        for index in range(4):
            _solid(tmp_path / "train" / label / f"{index}.jpg", (8, 8), color)
            _solid(tmp_path / "val" / label / f"{index}.jpg", (8, 8), color)
    manifest = build_manifest(tmp_path, _config(), csv_path=None)
    assert not any(row["split"] == "test" for row in manifest.rows)
    assert any("test" in message.lower() for message in manifest.warnings)


def test_csv_labels(tmp_path: Path):
    for label, color in (("mug", (255, 0, 0)), ("keys", (0, 0, 255))):
        for index in range(4):
            _solid(tmp_path / "photos" / f"{label}_{index}.jpg", (8, 8), color)
    csv_path = tmp_path / "labels.csv"
    lines = ["path,label"]
    for label in ("mug", "keys"):
        for index in range(4):
            lines.append(f"photos/{label}_{index}.jpg,{label}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, _config(), csv_path=str(csv_path))
    assert manifest.class_names == ["keys", "mug"]
    assert {row["split"] for row in manifest.rows} == {"train", "val", "test"}


def test_too_few_photos_explains_what_to_do(tmp_path: Path):
    _solid(tmp_path / "mug" / "a.jpg", (8, 8), (255, 0, 0))
    _solid(tmp_path / "keys" / "a.jpg", (8, 8), (0, 0, 255))
    with pytest.raises(RuntimeError, match="fewer than 3"):
        build_manifest(tmp_path, _config(), csv_path=None)


def test_exif_orientation_is_applied(tmp_path: Path):
    wide = Image.new("RGB", (40, 16), (255, 0, 0))
    wide.putpixel((0, 0), (0, 0, 255))
    plain = tmp_path / "plain.jpg"
    turned = tmp_path / "turned.jpg"
    wide.save(plain)
    exif = wide.getexif()
    exif[274] = 6
    wide.save(turned, exif=exif)
    assert read_image(plain).size == (40, 16)
    assert read_image(turned).size == (16, 40)


def test_max_side_shrinks_the_long_edge(tmp_path: Path):
    path = tmp_path / "big.jpg"
    _solid(path, (100, 40), (10, 20, 30))
    assert read_image(path, max_side=20).size == (20, 8)


def test_identical_photos_in_different_splits_are_flagged(tmp_path: Path):
    train = tmp_path / "train.jpg"
    test = tmp_path / "test.jpg"
    _solid(train, (32, 32), (20, 180, 20))
    _solid(test, (32, 32), (20, 180, 20))
    other = tmp_path / "other.jpg"
    Image.effect_noise((32, 32), 80).convert("RGB").save(other)
    manifest = [
        {"path": str(train), "label": "mug", "split": "train"},
        {"path": str(test), "label": "mug", "split": "test"},
        {"path": str(other), "label": "keys", "split": "val"},
    ]
    pairs = find_cross_split_duplicates(manifest, hash_size=8, max_distance=0, max_side=64)
    assert len(pairs) == 1
    assert pairs[0]["distance"] == 0
    assert {pairs[0]["split_a"], pairs[0]["split_b"]} == {"train", "test"}


def test_dhash_distance_is_zero_for_the_same_image():
    image = Image.new("RGB", (24, 24), (8, 8, 8))
    assert hamming(dhash(image), dhash(image)) == 0


def test_config_overrides_and_real_config_keys():
    cfg = _config()
    apply_overrides(cfg, ["train.head_epochs=1", "model.name=tiny", "model.pretrained=false"])
    assert cfg["train"]["head_epochs"] == 1
    assert cfg["model"]["name"] == "tiny"
    assert cfg["model"]["pretrained"] is False
    with pytest.raises(ValueError, match="Unknown config key"):
        apply_overrides(cfg, ["train.not_a_real_key=1"])
    fresh = load_config(ROOT / "config.yaml")
    assert fresh["model"]["name"] == "convnext_tiny"
    assert fresh["split"]["train"] + fresh["split"]["val"] + fresh["split"]["test"] == pytest.approx(1)
    smoke = load_config(ROOT / "config.smoke.yaml")
    assert smoke["model"]["name"] == "tiny"
    assert smoke["model"]["pretrained"] is False

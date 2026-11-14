"""Turn several common exports into one YOLO dataset on disk.

Accepted inputs:

- A YOLO dataset with ``data.yaml`` (including Roboflow and CVAT exports).
- ``images/`` plus ``labels/``, or a folder of photos with matching ``.txt`` files.
- Existing ``train`` / ``val`` / ``test`` folders. ``val`` may be named ``valid``.
- A COCO JSON file.
- A Label Studio JSON export, with the photos beside it or under ``images/``.

If there is only one pool of photos, they are split 70/15/15. The split is
grouped by the rarest class in each photo so a scarce object still shows up
in validation and test when there are enough photos of it. The seed is fixed.

iPhone photos are stored sideways, with the real rotation only in EXIF.
Labels drawn in Roboflow, Label Studio, or CVAT refer to the upright photo
you saw. This module bakes that rotation into the pixels and leaves the
normalized boxes unchanged.
"""

from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.data import is_image, read_image, split_counts
from src.detect.boxes import (
    coco_to_boxes,
    format_yolo,
    label_studio_results,
    load_json_file,
    parse_yolo_text,
    sniff_json,
)

SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}


@dataclass
class DetSample:
    """One photo and the boxes that belong to it."""

    image: Path
    boxes: list[tuple[int, float, float, float, float]]
    split: str | None = None

    @property
    def classes(self) -> set[int]:
        return {box[0] for box in self.boxes}


@dataclass
class PreparedDataset:
    """The YOLO tree written for Ultralytics, plus the file list."""

    yaml_path: Path
    root: Path
    class_names: list[str]
    rows: list[dict[str, str]]
    layout: str
    warnings: list[str] = field(default_factory=list)


def _ratios(cfg: dict) -> tuple[float, float, float]:
    split = cfg["split"]
    ratios = (float(split["train"]), float(split["val"]), float(split["test"]))
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("split.train, split.val, and split.test must all be greater than 0.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-3:
        raise ValueError(
            f"split.train + split.val + split.test must add up to 1. They add up to {total:.3f}."
        )
    return ratios


def layout_help(path: Path) -> str:
    return (
        f"Could not find a detection dataset at {path}.\n"
        "Use one of these:\n"
        "  1. A folder with data.yaml, images/, and labels/ (YOLO / Roboflow / CVAT).\n"
        "  2. images/<photo>.jpg and labels/<photo>.txt, or photos and txt files side by side.\n"
        "  3. train/val/test folders that already exist.\n"
        "  4. A COCO JSON file, or a Label Studio JSON export next to the photos.\n"
        "YOLO label lines look like: class cx cy w h   (normalized, center of the box)."
    )


def parse_names(raw, fallback_count: int | None = None) -> list[str]:
    """Normalize a names list or ``{id: name}`` mapping from data.yaml."""
    if raw is None:
        if not fallback_count:
            return []
        return [f"class_{index}" for index in range(fallback_count)]
    if isinstance(raw, list):
        return [str(name) for name in raw]
    if isinstance(raw, dict):
        pairs = sorted((int(key), str(value)) for key, value in raw.items())
        if not pairs:
            return []
        names = [f"class_{index}" for index in range(pairs[-1][0] + 1)]
        for index, name in pairs:
            names[index] = name
        return names
    raise ValueError(f"Could not read class names from {raw!r}.")


def _read_name_file(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def _find_name_list(root: Path) -> list[str]:
    for candidate in (root / "classes.txt", root / "obj.names", root / "names.txt"):
        if candidate.is_file():
            return _read_name_file(candidate)
    return []


def _check_class_ids(samples: list[DetSample], names: list[str], source: str) -> list[str]:
    """Make sure every box points at a known class. Returns warnings."""
    warnings: list[str] = []
    highest = -1
    for sample in samples:
        for class_id, *_rest in sample.boxes:
            highest = max(highest, class_id)
            if names and class_id >= len(names):
                raise ValueError(
                    f"{sample.image} uses class id {class_id}, but only {len(names)} "
                    f"names were found in {source}."
                )
    if not names:
        if highest < 0:
            raise ValueError(f"No labeled objects were found in {source}.")
        if highest > 999:
            raise ValueError(f"Class id {highest} is unreasonably large. Add a names list.")
        names.extend(f"class_{index}" for index in range(highest + 1))
        warnings.append(
            "No class names file was found. Using class_0, class_1, ... "
            "Add classes.txt or names in data.yaml so the predictions are readable."
        )
    return warnings


def _label_path_for(image: Path) -> Path | None:
    """Find the txt file that belongs to this photo, if there is one."""
    sibling = image.with_suffix(".txt")
    if sibling.is_file():
        return sibling
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.is_file():
                return candidate
            break
    return None


def _boxes_from_label_file(image: Path) -> list[tuple[int, float, float, float, float]]:
    label_path = _label_path_for(image)
    if label_path is None:
        return []
    return parse_yolo_text(label_path.read_text(encoding="utf-8"), str(label_path))


def _iter_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    images: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file() or not is_image(path):
            continue
        if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
            continue
        images.append(path)
    return sorted(images, key=lambda path: path.as_posix())


def _samples_from_images(images: list[Path], split: str | None) -> list[DetSample]:
    samples = []
    missing = 0
    for image in images:
        if _label_path_for(image) is None:
            missing += 1
        samples.append(DetSample(image=image, boxes=_boxes_from_label_file(image), split=split))
    if images and missing == len(images):
        raise ValueError(
            f"Found {len(images)} photos under {images[0].parent} but no YOLO .txt labels. "
            "Each photo needs a label file with the same name."
        )
    return samples


def _resolve_yaml_dir(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _dirs_from_yaml_value(base: Path, value) -> list[Path]:
    """YOLO allows one path or a list of paths, and the path may be a txt list."""
    values = value if isinstance(value, list) else [value]
    folders: list[Path] = []
    for item in values:
        if item is None:
            continue
        path = _resolve_yaml_dir(base, str(item))
        if path.suffix == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image = Path(line)
                if not image.is_absolute():
                    image = (path.parent / image).resolve()
                if image.is_file() and is_image(image):
                    folders.append(image)
            continue
        folders.append(path)
    return folders


def _samples_from_yaml_dirs(entries: list[Path], split: str) -> list[DetSample]:
    samples: list[DetSample] = []
    for entry in entries:
        if entry.is_file() and is_image(entry):
            samples.append(DetSample(entry, _boxes_from_label_file(entry), split))
        else:
            samples.extend(_samples_from_images(_iter_images(entry), split))
    return samples


def load_yolo_yaml(yaml_path: Path) -> tuple[list[DetSample], list[str], str, list[str]]:
    """Load a dataset that already has a YOLO ``data.yaml``."""
    loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{yaml_path} must be a mapping.")
    base = yaml_path.parent
    dataset_root = base
    if loaded.get("path"):
        dataset_root = _resolve_yaml_dir(base, str(loaded["path"]))
        base = dataset_root
    names = parse_names(loaded.get("names"))
    warnings: list[str] = []
    samples: list[DetSample] = []
    found_any_split = False
    for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
        if key not in loaded and not (split == "val" and "valid" in loaded):
            continue
        raw = loaded.get(key, loaded.get("valid"))
        entries = _dirs_from_yaml_value(base, raw)
        split_samples = _samples_from_yaml_dirs(entries, split)
        if split_samples:
            found_any_split = True
            samples.extend(split_samples)
    if not found_any_split:
        raise ValueError(f"{yaml_path} does not point at any photos.")
    # A yaml that lists only one directory of everything is not a real split.
    splits_present = {sample.split for sample in samples}
    if splits_present == {"train"} and len(samples) >= 3:
        for sample in samples:
            sample.split = None
        layout = "data.yaml with one photo pool; automatic split"
    else:
        layout = "existing YOLO data.yaml splits"
    warnings.extend(_check_class_ids(samples, names, str(yaml_path)))
    return samples, names, layout, warnings


def _split_image_dirs(root: Path) -> dict[str, Path]:
    """Find train/val/test image directories under common export layouts."""
    found: dict[str, Path] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            candidates = (
                root / "images" / alias,
                root / alias / "images",
                root / alias,
            )
            for candidate in candidates:
                if candidate.is_dir() and _iter_images(candidate):
                    found[split] = candidate
                    break
            if split in found:
                break
    if "train" not in found:
        return {}
    return found


def load_split_folders(root: Path) -> tuple[list[DetSample], list[str], str, list[str]] | None:
    folders = _split_image_dirs(root)
    if not folders:
        return None
    samples: list[DetSample] = []
    for split, folder in folders.items():
        samples.extend(_samples_from_images(_iter_images(folder), split))
    names = _find_name_list(root)
    warnings = _check_class_ids(samples, names, str(root))
    return samples, names, "existing train/val/test folders", warnings


def load_flat_folders(root: Path) -> tuple[list[DetSample], list[str], str, list[str]] | None:
    images_dir = root / "images" if (root / "images").is_dir() else None
    labels_dir = root / "labels" if (root / "labels").is_dir() else None
    if images_dir and labels_dir and not _split_image_dirs(root):
        images = _iter_images(images_dir)
    elif any(path.is_file() and is_image(path) for path in root.iterdir()):
        images = sorted((path for path in root.iterdir() if path.is_file() and is_image(path)), key=lambda path: path.name)
    else:
        return None
    if not images:
        return None
    samples = _samples_from_images(images, None)
    names = _find_name_list(root)
    warnings = _check_class_ids(samples, names, str(root))
    return samples, names, "one folder of photos; automatic split", warnings


def _find_image_file(file_name: str, search_roots: list[Path]) -> Path | None:
    raw = Path(file_name)
    if raw.is_file():
        return raw.resolve()
    basename = raw.name
    for root in search_roots:
        for candidate in (root / file_name, root / basename, root / "images" / basename, root / "images" / file_name):
            if candidate.is_file():
                return candidate.resolve()
    for root in search_roots:
        matches = [path for path in root.rglob(basename) if path.is_file() and is_image(path)]
        if len(matches) == 1:
            return matches[0].resolve()
    return None


def load_coco(json_path: Path) -> tuple[list[DetSample], list[str], str, list[str]]:
    """Convert a COCO detection JSON into in-memory YOLO boxes."""
    payload = load_json_file(json_path)
    if sniff_json(payload) != "coco":
        raise ValueError(f"{json_path} is not a COCO detection file.")
    search_roots = [json_path.parent, json_path.parent / "images"]
    sizes: dict[int, tuple[int, int]] = {}
    paths: dict[int, Path] = {}
    missing: list[str] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        found = _find_image_file(str(image.get("file_name") or ""), search_roots)
        if found is None:
            missing.append(str(image.get("file_name")))
            continue
        if width <= 0 or height <= 0:
            width, height = read_image(found, max_side=0).size
        sizes[image_id] = (width, height)
        paths[image_id] = found
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"COCO JSON lists photos that were not found, for example: {preview}. "
            "Put the image files next to the JSON or in an images/ folder."
        )
    boxes_by_image, names = coco_to_boxes(payload, sizes)
    samples = [
        DetSample(image=paths[image_id], boxes=boxes_by_image.get(image_id, []), split=None)
        for image_id in paths
    ]
    samples.sort(key=lambda sample: sample.image.as_posix())
    return samples, names, "COCO JSON converted to YOLO; automatic split", []


def load_label_studio(json_path: Path) -> tuple[list[DetSample], list[str], str, list[str]]:
    """Convert a Label Studio JSON export. Boxes are matched to local photos by file name."""
    payload = load_json_file(json_path)
    if sniff_json(payload) != "labelstudio":
        raise ValueError(f"{json_path} is not a Label Studio export.")
    search_roots = [json_path.parent, json_path.parent / "images"]
    names: list[str] = []
    name_to_id: dict[str, int] = {}
    samples: list[DetSample] = []
    missing: list[str] = []
    for task in payload:
        raw_image = ""
        data = task.get("data") or {}
        if isinstance(data, dict):
            raw_image = str(data.get("image") or data.get("img") or "")
        file_name = Path(raw_image.split("?")[0]).name
        if not file_name:
            continue
        found = _find_image_file(file_name, search_roots)
        if found is None:
            missing.append(file_name)
            continue
        boxes = []
        for class_name, cx, cy, width, height in label_studio_results(task):
            if class_name not in name_to_id:
                name_to_id[class_name] = len(names)
                names.append(class_name)
            boxes.append((name_to_id[class_name], cx, cy, width, height))
        samples.append(DetSample(image=found, boxes=boxes, split=None))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Label Studio JSON refers to photos that were not found, for example: {preview}. "
            "Keep the photos in the same folder as the JSON, or in images/."
        )
    if not samples:
        raise ValueError(f"No labeled photos were found in {json_path}.")
    return samples, names, "Label Studio JSON converted to YOLO; automatic split", []


def _sniff_dataset_json(root: Path) -> Path | None:
    candidates = [
        root / "annotations.json",
        root / "instances_default.json",
        root / "label_studio.json",
        root / "annotations" / "instances_default.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    json_files = [path for path in root.glob("*.json") if path.is_file()]
    if len(json_files) == 1:
        return json_files[0]
    return None


def load_dataset(source: Path) -> tuple[list[DetSample], list[str], str, list[str]]:
    """Read whatever export the student pointed ``--data`` at."""
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        return load_yolo_yaml(source)
    if source.is_file() and source.suffix.lower() == ".json":
        payload = load_json_file(source)
        kind = sniff_json(payload)
        if kind == "coco":
            return load_coco(source)
        if kind == "labelstudio":
            return load_label_studio(source)
        raise ValueError(f"Could not tell if {source} is COCO or Label Studio JSON.")
    if not source.is_dir():
        raise FileNotFoundError(layout_help(source))

    for yaml_name in ("data.yaml", "data.yml"):
        yaml_path = source / yaml_name
        if yaml_path.is_file():
            return load_yolo_yaml(yaml_path)

    json_path = _sniff_dataset_json(source)
    if json_path is not None:
        payload = load_json_file(json_path)
        kind = sniff_json(payload)
        if kind == "coco":
            return load_coco(json_path)
        if kind == "labelstudio":
            return load_label_studio(json_path)

    if (source / "obj.names").is_file() and (source / "obj_train_data").is_dir():
        # Classic CVAT YOLO export keeps photos and txt files in obj_train_data.
        names = _read_name_file(source / "obj.names")
        images = _iter_images(source / "obj_train_data")
        samples = _samples_from_images(images, None)
        warnings = _check_class_ids(samples, names, str(source / "obj.names"))
        return samples, names, "CVAT YOLO export; automatic split", warnings

    split = load_split_folders(source)
    if split is not None:
        return split
    flat = load_flat_folders(source)
    if flat is not None:
        return flat
    raise RuntimeError(layout_help(source))


def _stratify_key(sample: DetSample, frequency: Counter) -> str:
    """Group by the scarcest class in the photo so rare objects are not all in train."""
    if not sample.classes:
        return "__none__"
    rarest = min(sample.classes, key=lambda class_id: (frequency[class_id], class_id))
    return str(rarest)


def _apply_counts(group: list[DetSample], counts: tuple[int, int, int]) -> None:
    n_train, n_val, n_test = counts
    for index, sample in enumerate(group):
        if index < n_train:
            sample.split = "train"
        elif index < n_train + n_val:
            sample.split = "val"
        else:
            sample.split = "test"
    if n_test != len(group) - n_train - n_val:
        raise RuntimeError("Internal error while splitting detection photos.")


def assign_splits(
    samples: list[DetSample],
    ratios: tuple[float, float, float],
    seed: int,
) -> list[str]:
    """Fill ``sample.split``. Returns warnings. Does not move photos that already have a split."""
    warnings: list[str] = []
    pending = [sample for sample in samples if sample.split is None]
    if not pending:
        return warnings
    if len(pending) < 3:
        raise RuntimeError(
            "Need at least 3 photos to build train, val, and test. "
            "Label a few more scenes, with the objects in different places."
        )
    frequency: Counter = Counter()
    for sample in pending:
        frequency.update(sample.classes)
    groups: dict[str, list[DetSample]] = defaultdict(list)
    for sample in pending:
        groups[_stratify_key(sample, frequency)].append(sample)
    rng = random.Random(seed)
    small_groups = 0
    for group in groups.values():
        rng.shuffle(group)
        if len(group) < 3:
            small_groups += len(group)
            continue
        _apply_counts(group, split_counts(len(group), ratios))
    # Photos in tiny groups were not assigned. If that would leave val empty,
    # ignore the groups and split the whole pool so training can still run.
    unassigned = [sample for sample in pending if sample.split is None]
    val_count = sum(1 for sample in pending if sample.split == "val")
    if unassigned and (val_count == 0 or len(unassigned) == len(pending)):
        for sample in pending:
            sample.split = None
        rng.shuffle(pending)
        _apply_counts(pending, split_counts(len(pending), ratios))
        warnings.append(
            "Too few photos of some objects to stratify by class. "
            "The whole set was split instead. Add more photos of the rare objects "
            "if a class is missing from the test report."
        )
    else:
        for sample in unassigned:
            sample.split = "train"
        if unassigned:
            warnings.append(
                f"{len(unassigned)} photos belong to a class that appears fewer than 3 times, "
                "so they stayed in train and are not in the test score."
            )
    return warnings


def _hold_out_validation(samples: list[DetSample], fraction: float, seed: int) -> None:
    """Move part of train into val. Test photos stay where they are."""
    rng = random.Random(seed)
    frequency: Counter = Counter(class_id for sample in samples if sample.split == "train" for class_id in sample.classes)
    groups: dict[str, list[DetSample]] = defaultdict(list)
    for sample in samples:
        if sample.split == "train":
            groups[_stratify_key(sample, frequency)].append(sample)
    for group in groups.values():
        rng.shuffle(group)
        n_val = int(round(len(group) * fraction))
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        else:
            n_val = 0
        for index, sample in enumerate(group):
            if index < n_val:
                sample.split = "val"


def _missing_split_warnings(samples: list[DetSample], names: list[str]) -> list[str]:
    warnings = []
    for split in ("val", "test"):
        present_images = [sample for sample in samples if sample.split == split]
        if not present_images:
            continue
        present = {class_id for sample in present_images for class_id in sample.classes}
        missing = [names[class_id] for class_id in range(len(names)) if class_id not in present]
        if missing:
            warnings.append(f"These classes have no {split} boxes: {', '.join(missing)}.")
    if not any(sample.split == "test" for sample in samples):
        warnings.append(
            "No test photos were found, so training cannot report a held-out test score. "
            "Add a test folder, or give one folder of photos and let the program split them."
        )
    return warnings


def _unique_stem(image: Path, used: set[str]) -> str:
    stem = image.stem
    if stem not in used:
        used.add(stem)
        return stem
    suffix = 2
    while f"{stem}_{suffix}" in used:
        suffix += 1
    unique = f"{stem}_{suffix}"
    used.add(unique)
    return unique


def materialize(
    samples: list[DetSample],
    names: list[str],
    dest: Path,
    max_side: int,
) -> list[dict[str, str]]:
    """Write upright photos and YOLO labels. Returns rows for the duplicate check."""
    if dest.exists():
        shutil.rmtree(dest)
    rows: list[dict[str, str]] = []
    used: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for sample in samples:
        split = sample.split or "train"
        stem = _unique_stem(sample.image, used[split])
        image_path = dest / "images" / split / f"{stem}.jpg"
        label_path = dest / "labels" / split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        # EXIF is applied here. Normalized boxes stay the same because they
        # were drawn on the upright view the labeling tool showed.
        read_image(sample.image, max_side=max_side).save(image_path, quality=95)
        label_path.write_text(format_yolo(sample.boxes), encoding="utf-8")
        rows.append({"path": str(image_path.resolve()), "label": names[0] if names else "", "split": split})
    return rows


def write_data_yaml(dest: Path, names: list[str], splits: set[str]) -> Path:
    """Write the yaml Ultralytics trains on. Test is included only when it exists."""
    payload: dict = {
        "path": str(dest.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(names)},
    }
    if "test" in splits:
        payload["test"] = "images/test"
    yaml_path = dest / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def prepare_detection_dataset(source: Path, cfg: dict, output_dir: Path) -> PreparedDataset:
    """Read ``source``, split if needed, and write a YOLO dataset under ``output_dir``."""
    samples, names, layout, warnings = load_dataset(source)
    if len(names) < 1:
        raise ValueError("The dataset has no class names.")
    ratios = _ratios(cfg)
    seed = int(cfg["seed"])
    presplit = all(sample.split is not None for sample in samples)
    if presplit:
        present = {sample.split for sample in samples}
        if "val" not in present:
            train_ratio, val_ratio, _test_ratio = ratios
            fraction = val_ratio / (train_ratio + val_ratio)
            _hold_out_validation(samples, fraction, seed)
            warnings.append(
                "No val folder was found. Holding out part of train for validation "
                f"({fraction:.0%}). The test folder was not touched."
            )
    else:
        warnings.extend(assign_splits(samples, ratios, seed))
        if "automatic" not in layout and "converted" not in layout:
            layout = f"{layout}; automatic {ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%} split (seed {seed})"
        else:
            layout = f"{layout} ({ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%}, seed {seed})"
    if not any(sample.split == "val" for sample in samples):
        raise RuntimeError(
            "The validation set is empty, so there is no way to choose the best checkpoint. "
            "Add more labeled photos, or add a val folder."
        )
    if not any(sample.split == "train" for sample in samples):
        raise RuntimeError("The training set is empty.")
    warnings.extend(_missing_split_warnings(samples, names))
    dest = output_dir / "dataset"
    rows = materialize(samples, names, dest, max_side=int(cfg.get("max_side", 1600)))
    present_splits = {row["split"] for row in rows}
    yaml_path = write_data_yaml(dest, names, present_splits)
    return PreparedDataset(
        yaml_path=yaml_path,
        root=dest,
        class_names=names,
        rows=rows,
        layout=layout,
        warnings=warnings,
    )

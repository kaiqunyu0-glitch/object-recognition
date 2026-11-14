"""Find photos, honor an existing split, or make a stratified one.

Accepted layouts:

1. ``data_root/<class_name>/*.jpg``
   Split 70/15/15 (or whatever ``split`` says) with a fixed seed.
2. ``data_root/train|val|test/<class_name>/*.jpg``
   Those folders are used as they are. ``val`` may also be named
   ``valid`` or ``validation``.
3. A CSV with columns ``path,label``.

iPhone photos are often stored sideways and the true rotation is only in
EXIF. ``read_image`` applies that rotation before the model sees the pixels.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}

_HEIC_READY = False
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIC_READY = True
except ImportError:
    _HEIC_READY = False


def heic_supported() -> bool:
    return _HEIC_READY


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")


def limit_side(image: Image.Image, max_side: int) -> Image.Image:
    """Shrink very large phone photos so one open file cannot exhaust memory.

    The model only sees ``image_size`` (usually 224). Decoding a 48 megapixel
    photo and then cropping it wastes RAM. ``max_side`` of 0 disables this.
    """
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.BILINEAR)


def read_image(path: Path, max_side: int = 0) -> Image.Image:
    """Open an image as upright RGB.

    EXIF orientation is applied first, which is what makes iPhone photos
    appear the same way they looked in the camera app.
    """
    try:
        with Image.open(path) as handle:
            upright = ImageOps.exif_transpose(handle)
            rgb = upright.convert("RGB")
            return limit_side(rgb, max_side)
    except Exception as exc:
        hint = ""
        if path.suffix.lower() in {".heic", ".heif"} and not _HEIC_READY:
            hint = " Install pillow-heif (pip install pillow-heif) or export this photo as JPEG."
        raise RuntimeError(f"Could not read image {path}.{hint}") from exc


def split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Return ``(train, val, test)`` counts that add up to ``n``.

    When there are at least 3 photos, every split gets at least one so both
    the validation score and the held-out test score exist.
    """
    if n < 0:
        raise ValueError("Cannot split a negative number of photos.")
    if n == 0:
        return (0, 0, 0)
    raw = [n * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    leftover = n - sum(counts)
    # Largest fractional part gets the next photo. Ties prefer train, then val.
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in range(leftover):
        counts[order[index]] += 1
    if n >= 3:
        repairs = 0
        while 0 in counts and repairs < 3:
            repairs += 1
            needy = counts.index(0)
            donor = max(
                range(3),
                key=lambda index: counts[index] if index != needy else -1,
            )
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[needy] += 1
    if sum(counts) != n:
        raise RuntimeError(f"Internal error: split counts {counts} do not add up to {n}.")
    return (counts[0], counts[1], counts[2])


@dataclass
class ManifestResult:
    """The file list training, evaluation, and the duplicate check all share."""

    rows: list[dict[str, str]]
    class_names: list[str]
    layout: str
    warnings: list[str] = field(default_factory=list)


class ImageListDataset(Dataset):
    """Photos listed as ``(path, class_index, class_name)``."""

    def __init__(self, samples: list[tuple[str, int, str]], transform, max_side: int) -> None:
        self.samples = samples
        self.transform = transform
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_index, _class_name = self.samples[index]
        image = read_image(Path(path), max_side=self.max_side)
        if self.transform is not None:
            image = self.transform(image)
        return image, class_index


def list_class_folders(root: Path) -> dict[str, list[Path]]:
    """Map each class folder to the images sitting directly inside it."""
    groups: dict[str, list[Path]] = {}
    if not root.is_dir():
        return groups
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name.startswith(".") or child.name == "__MACOSX":
            continue
        images = sorted(
            (path for path in child.iterdir() if path.is_file() and is_image(path)),
            key=lambda path: path.name,
        )
        if images:
            groups[child.name] = images
    return groups


def find_presplit(root: Path) -> dict[str, Path]:
    """Return split folders when ``train/<class>/`` exists. Otherwise {}."""
    found: dict[str, Path] = {}
    for key, names in SPLIT_ALIASES.items():
        for name in names:
            folder = root / name
            if folder.is_dir() and list_class_folders(folder):
                found[key] = folder
                break
    if "train" not in found:
        return {}
    return found


def layout_help(root: Path) -> str:
    return (
        f"Could not find labeled photos in {root}.\n"
        "Use one of these layouts:\n"
        "  1. photos/<class_name>/*.jpg\n"
        "  2. photos/train/<class_name>/*.jpg\n"
        "     photos/val/<class_name>/*.jpg\n"
        "     photos/test/<class_name>/*.jpg\n"
        "  3. python train.py --data photos --csv labels.csv\n"
        "     The CSV needs a header: path,label"
    )


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


def _resolve_csv_path(raw: str, bases: tuple[Path, ...]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(f"CSV points at {path}, but that file does not exist.")
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(base) for base in bases)
    raise FileNotFoundError(f"Could not find {raw}. Looked in: {searched}.")


def read_labeled_csv(csv_path: Path, data_root: Path) -> dict[str, list[Path]]:
    """Read ``path,label`` rows. Paths may be relative to ``data_root``."""
    groups: dict[str, list[Path]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} is empty.")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "path" not in fields or "label" not in fields:
            raise ValueError(
                f"{csv_path} needs a header row with columns path,label. "
                f"Found columns: {list(reader.fieldnames)}."
            )
        for line_number, row in enumerate(reader, start=2):
            raw_path = (row.get(fields["path"]) or "").strip()
            label = (row.get(fields["label"]) or "").strip()
            if not raw_path and not label:
                continue
            if not raw_path or not label:
                raise ValueError(f"{csv_path}:{line_number} needs both a path and a label.")
            image_path = _resolve_csv_path(raw_path, (data_root, csv_path.parent, Path.cwd()))
            groups.setdefault(label, []).append(image_path)
    for label in list(groups):
        groups[label] = sorted(groups[label], key=lambda path: path.as_posix())
    return groups


def dedupe_groups(groups: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], list[str]]:
    """Drop repeated paths. The same file in two classes is an error."""
    warnings: list[str] = []
    seen: dict[str, str] = {}
    cleaned: dict[str, list[Path]] = {}
    for label in sorted(groups):
        unique: list[Path] = []
        local: set[str] = set()
        for path in groups[label]:
            key = str(path.resolve())
            if key in local:
                warnings.append(f"Skipping duplicate path in class {label}: {path}")
                continue
            other = seen.get(key)
            if other is not None and other != label:
                raise RuntimeError(
                    f"{path} is labeled both {other!r} and {label!r}. "
                    "Each photo can only belong to one class."
                )
            local.add(key)
            seen[key] = label
            unique.append(path)
        if unique:
            cleaned[label] = unique
    return cleaned, warnings


def _check_heic(groups: dict[str, list[Path]]) -> None:
    heic = [
        path
        for paths in groups.values()
        for path in paths
        if path.suffix.lower() in {".heic", ".heif"}
    ]
    if heic and not _HEIC_READY:
        raise RuntimeError(
            f"Found {len(heic)} HEIC photo(s), for example {heic[0]}. "
            "Install pillow-heif (pip install pillow-heif) or export the photos as JPEG."
        )


def _rows_from_groups(
    groups: dict[str, list[Path]],
    ratios: tuple[float, float, float],
    seed: int,
    require_three: bool,
) -> list[dict[str, str]]:
    if require_three:
        small = [name for name, paths in sorted(groups.items()) if len(paths) < 3]
        if small:
            listed = ", ".join(small)
            raise RuntimeError(
                "These classes have fewer than 3 photos, so they cannot be split into "
                f"train, val, and test: {listed}. "
                "Take a few more photos of each object, from different angles."
            )
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in sorted(groups):
        paths = list(groups[label])
        rng.shuffle(paths)
        n_train, n_val, n_test = split_counts(len(paths), ratios)
        parts = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        if len(parts["test"]) != n_test:
            raise RuntimeError("Internal error while splitting photos.")
        for split, split_paths in parts.items():
            for path in split_paths:
                rows.append(
                    {"path": str(path.resolve()), "label": label, "split": split}
                )
    return rows


def _rows_from_presplit_deduped(
    folders: dict[str, Path],
) -> tuple[list[dict[str, str]], list[str]]:
    """Read train, then val, then test. The first copy of a file wins.

    Train is visited first so a photo that was copied into the test folder
    by mistake stays in train and does not also count as a test photo.
    """
    warnings: list[str] = []
    seen: dict[str, tuple[str, str]] = {}
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        folder = folders.get(split)
        if folder is None:
            continue
        for label, paths in list_class_folders(folder).items():
            for path in paths:
                key = str(path.resolve())
                previous = seen.get(key)
                if previous is not None:
                    previous_split, previous_label = previous
                    if previous_label != label:
                        raise RuntimeError(
                            f"{path} is labeled both {previous_label!r} and {label!r}. "
                            "Each photo can only belong to one class."
                        )
                    warnings.append(
                        f"Skipping duplicate file {path}. It is already in the {previous_split} split."
                    )
                    continue
                seen[key] = (split, label)
                rows.append({"path": key, "label": label, "split": split})
    return rows, warnings


def hold_out_validation(rows: list[dict[str, str]], fraction: float, seed: int) -> list[dict[str, str]]:
    """Move part of each class from train into val. Test rows stay put."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = {}
    others: list[dict[str, str]] = []
    for row in rows:
        if row["split"] == "train":
            by_class.setdefault(row["label"], []).append(row)
        else:
            others.append(row)
    rewritten = list(others)
    for label in sorted(by_class):
        order = list(by_class[label])
        rng.shuffle(order)
        n_val = int(round(len(order) * fraction))
        if len(order) >= 2:
            n_val = min(max(n_val, 1), len(order) - 1)
        else:
            n_val = 0
        for index, row in enumerate(order):
            split = "val" if index < n_val else "train"
            rewritten.append({**row, "split": split})
    return rewritten


def _class_names(rows: list[dict[str, str]]) -> list[str]:
    train_labels = {row["label"] for row in rows if row["split"] == "train"}
    if len(train_labels) < 2:
        raise RuntimeError(
            "Need at least 2 classes in the training photos. "
            "Put each object in its own folder (mug, keys, bottle, ...)."
        )
    missing = sorted({row["label"] for row in rows if row["label"] not in train_labels})
    if missing:
        listed = ", ".join(missing)
        raise RuntimeError(
            f"These classes have photos outside train but no training photos: {listed}."
        )
    return sorted(train_labels)


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {"train": 0, "val": 1, "test": 2}
    return sorted(rows, key=lambda row: (order[row["split"]], row["label"], row["path"]))


def _missing_split_warnings(rows: list[dict[str, str]], class_names: list[str]) -> list[str]:
    warnings: list[str] = []
    for split in ("val", "test"):
        present = {row["label"] for row in rows if row["split"] == split}
        if not present:
            continue
        missing = [name for name in class_names if name not in present]
        if missing:
            warnings.append(
                f"These classes have no {split} photos: {', '.join(missing)}."
            )
    if not any(row["split"] == "test" for row in rows):
        warnings.append(
            "No test folder was found, so training cannot report a held-out test score. "
            "Add a test folder, or put all photos in class folders and let the program split them."
        )
    if not any(row["split"] == "val" for row in rows):
        warnings.append(
            "The validation set is empty, so there is nothing to use for early stopping."
        )
    return warnings


def build_manifest(data_root: Path, cfg: dict, csv_path: str | None) -> ManifestResult:
    """Build the train/val/test file list without reading pixel data."""
    if not data_root.is_dir():
        raise FileNotFoundError(layout_help(data_root) + f"\nThe folder does not exist: {data_root}")

    ratios = _ratios(cfg)
    seed = int(cfg["seed"])
    warnings: list[str] = []

    if csv_path:
        csv_file = _resolve_csv_path(csv_path, (Path.cwd(), data_root))
        groups, dedupe_warnings = dedupe_groups(read_labeled_csv(csv_file, data_root))
        warnings.extend(dedupe_warnings)
        _check_heic(groups)
        if len(groups) < 2:
            raise RuntimeError(layout_help(data_root))
        rows = _rows_from_groups(groups, ratios, seed, require_three=True)
        layout = (
            "CSV labels, then automatic "
            f"{ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%} split (seed {seed})"
        )
    else:
        presplit = find_presplit(data_root)
        if presplit:
            rows, presplit_warnings = _rows_from_presplit_deduped(presplit)
            warnings.extend(presplit_warnings)
            _check_heic({"all": [Path(row["path"]) for row in rows]})
            layout = "existing train/val/test folders"
            if "val" not in presplit:
                train_ratio, val_ratio, _test_ratio = ratios
                fraction = val_ratio / (train_ratio + val_ratio)
                rows = hold_out_validation(rows, fraction=fraction, seed=seed)
                warnings.append(
                    "No val folder was found. Holding out part of each class from the train "
                    f"folder ({fraction:.0%}) for validation. The test folder was not touched."
                )
        else:
            groups, dedupe_warnings = dedupe_groups(list_class_folders(data_root))
            warnings.extend(dedupe_warnings)
            _check_heic(groups)
            if len(groups) < 2:
                raise RuntimeError(layout_help(data_root))
            rows = _rows_from_groups(groups, ratios, seed, require_three=True)
            layout = (
                "automatic "
                f"{ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%} split (seed {seed})"
            )

    rows = _sort_rows(rows)
    class_names = _class_names(rows)
    warnings.extend(_missing_split_warnings(rows, class_names))
    if not any(row["split"] == "val" for row in rows):
        raise RuntimeError(
            "The validation set is empty, so there is no way to choose the best checkpoint. "
            "Add more photos, or add a val folder."
        )
    return ManifestResult(rows=rows, class_names=class_names, layout=layout, warnings=warnings)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find the split list {path}. "
            "It is written next to best.pt when you run train.py."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "split": row["split"],
                }
            )
    return rows


def samples_for_split(
    rows: list[dict[str, str]],
    split: str,
    class_names: list[str],
) -> list[tuple[str, int, str]]:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    samples: list[tuple[str, int, str]] = []
    for row in rows:
        if row["split"] != split:
            continue
        label = row["label"]
        if label not in class_to_index:
            raise RuntimeError(
                f"{row['path']} has label {label!r}, which is not one of the trained classes."
            )
        samples.append((row["path"], class_to_index[label], label))
    return samples


def datasets_from_manifest(
    rows: list[dict[str, str]],
    class_names: list[str],
    train_tf,
    eval_tf,
    max_side: int,
) -> tuple[ImageListDataset, ImageListDataset, ImageListDataset]:
    def _make(split: str, transform) -> ImageListDataset:
        return ImageListDataset(
            samples_for_split(rows, split, class_names),
            transform=transform,
            max_side=max_side,
        )

    return _make("train", train_tf), _make("val", eval_tf), _make("test", eval_tf)

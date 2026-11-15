"""Download the public chess-piece YOLO set. No Kaggle or Roboflow login.

The Hub repo is public and ungated. ``huggingface_hub`` can download it with
no token. Images and labels are not committed; ``data/`` is gitignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HUB_REPO = "LibreYOLO/chess-pieces-mjzgj"
HUB_URL = f"https://huggingface.co/datasets/{HUB_REPO}"
LICENSE = "CC BY 4.0"
# Counted from the public snapshot: one JPEG and one YOLO txt per image.
EXPECTED_IMAGES = {"train": 202, "valid": 58, "test": 29}


def class_names(config_path: Path | None = None) -> list[str]:
    """Read the 13 class names in label-id order from config.chess.yaml."""
    path = config_path or (ROOT / "config.chess.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = loaded["names"]
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=int)]
    return list(names)


def download(out: Path) -> Path:
    """Download the snapshot into ``out`` and point data.yaml at that folder."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub to download this public dataset: "
            "pip install huggingface_hub\n"
            "No Hugging Face, Kaggle, or Roboflow token is required."
        ) from exc

    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HUB_REPO, repo_type="dataset", local_dir=str(out))
    counts = {}
    for split in EXPECTED_IMAGES:
        images = [path for path in (out / split / "images").glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = list((out / split / "labels").glob("*.txt"))
        if len(images) != len(labels):
            raise RuntimeError(f"{split}: {len(images)} images but {len(labels)} label files.")
        counts[split] = len(images)
    if counts != EXPECTED_IMAGES:
        raise RuntimeError(f"Image counts {counts} do not match the recorded snapshot {EXPECTED_IMAGES}.")

    data_yaml = out / "data.yaml"
    loaded = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    snapshot_names = loaded["names"]
    if isinstance(snapshot_names, dict):
        snapshot_names = [snapshot_names[key] for key in sorted(snapshot_names, key=int)]
    expected = class_names()
    if list(snapshot_names) != expected:
        raise RuntimeError(f"Class list changed.\n snapshot: {snapshot_names}\n config:   {expected}")
    license_name = str((loaded.get("roboflow") or {}).get("license", ""))
    if "CC BY 4.0" not in license_name:
        raise RuntimeError(f"Expected CC BY 4.0 in data.yaml, found {license_name!r}.")

    loaded["path"] = str(out.resolve())
    loaded["names"] = {index: name for index, name in enumerate(expected)}
    data_yaml.write_text(yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Dataset: {HUB_URL}")
    print(f"License: {LICENSE}")
    print(f"Images: train={counts['train']} valid={counts['valid']} test={counts['test']}")
    print(f"Wrote {data_yaml}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public CC BY 4.0 chess-piece YOLO dataset.")
    parser.add_argument("--out", default="data/chess", help="Folder for images, labels, and data.yaml.")
    args = parser.parse_args()
    download(Path(args.out))


if __name__ == "__main__":
    main()

"""Download a few CIFAR-10 classes and save them as folders of images.

Example:
    python scripts/download_sample.py --out data/cifar_sample

CIFAR-10 images are 32x32. They are only for trying the commands. Do not
expect phone-photo accuracy from them. The download is about 170MB the
first time and needs a network connection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_main

DEFAULT_CLASSES = ("automobile", "cat", "dog", "truck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a small CIFAR-10 sample as class folders.")
    parser.add_argument("--out", default="data/cifar_sample", help="Folder to create.")
    parser.add_argument("--per-class", type=int, default=40, help="Images to keep per class.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="CIFAR-10 class names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError("Install torchvision first: pip install torch torchvision") from exc

    out = Path(args.out)
    cache = out.parent / "_download_cache"
    try:
        dataset = CIFAR10(root=cache, train=True, download=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not download CIFAR-10. Check the network and try again."
        ) from exc

    known = set(dataset.classes)
    unknown = [name for name in args.classes if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown CIFAR-10 class(es): {', '.join(unknown)}. "
            f"Choose from: {', '.join(dataset.classes)}."
        )
    wanted = set(args.classes)
    counts = {name: 0 for name in args.classes}
    # Use the CIFAR training half only. This project makes its own val/test split.
    for image, label in dataset:
        name = dataset.classes[label]
        if name not in wanted or counts[name] >= args.per_class:
            continue
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"{counts[name]:03d}.png")
        counts[name] += 1
        if all(value >= args.per_class for value in counts.values()):
            break

    print(f"Wrote {sum(counts.values())} images to {out.resolve()}")
    print("These pictures are 32x32. They are for trying the commands, not for a 99% score.")
    print(f"  python train.py --data {out}")


if __name__ == "__main__":
    run_main(main)

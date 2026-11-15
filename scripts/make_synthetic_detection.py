"""Create a tiny folder of photos that each contain several shapes.

Example:
    python scripts/make_synthetic_detection.py --out data/detect_synthetic

These are drawings, not phone photos. They only check that detection runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_main
from src.detect.synthetic import make_detection_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a tiny synthetic detection dataset.")
    parser.add_argument("--out", default="data/detect_synthetic")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = make_detection_dataset(Path(args.out), images=args.images, image_size=args.image_size, seed=args.seed)
    print(f"Wrote detection photos to {out.resolve()}")
    print("Try the small model (no weight download):")
    print(f"  python detect_train.py --data {out} --config config.detect.smoke.yaml")


if __name__ == "__main__":
    run_main(main)

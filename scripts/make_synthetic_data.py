"""Create a tiny folder of shape photos so you can try the commands.

Example:
    python scripts/make_synthetic_data.py --out data/synthetic

These are drawings, not phone photos. They only check that training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_main
from src.synthetic import make_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a tiny synthetic dataset of shapes.")
    parser.add_argument("--out", default="data/synthetic", help="Folder to create.")
    parser.add_argument("--per-class", type=int, default=18, help="Photos for each shape.")
    parser.add_argument("--image-size", type=int, default=128, help="Width and height in pixels.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = make_synthetic_dataset(
        Path(args.out),
        per_class=args.per_class,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Wrote shape photos to {out.resolve()}")
    print("Try the pipeline with the small model:")
    print(f"  python train.py --data {out} --config config.smoke.yaml")


if __name__ == "__main__":
    run_main(main)

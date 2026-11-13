"""Score the held-out test photos saved during training.

Example:
    python evaluate.py

The test photos are the ones listed as ``test`` in split_manifest.csv next
to best.pt. They were not used to train, and they were not used to pick the
best epoch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint import default_checkpoint
from src.cli import run_main
from src.config import load_config
from src.engine import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report held-out test accuracy, a confusion matrix, and mistakes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="YAML settings file.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to best.pt in output_dir.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval.batch_size.")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Also score a horizontal flip and average the two predictions.",
    )
    parser.add_argument("--no-tta", action="store_true", help="Turn test-time augmentation off.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.config)
    if args.tta and args.no_tta:
        raise ValueError("Pass only one of --tta or --no-tta.")
    if args.tta:
        tta = True
    elif args.no_tta:
        tta = False
    else:
        tta = bool(cfg["eval"]["tta"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["eval"]["batch_size"])
    metrics = evaluate_checkpoint(
        checkpoint,
        cfg,
        tta=tta,
        batch_size=batch_size,
        num_workers=int(cfg["train"]["num_workers"]),
    )
    if metrics is None:
        raise SystemExit(1)


if __name__ == "__main__":
    run_main(main)

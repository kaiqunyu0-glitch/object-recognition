"""Score the held-out test photos from a detection training run.

Example:
    python detect_eval.py

Prints mAP50, mAP50-95, per-class precision and recall, writes a confusion
matrix, and lists photos that have a miss or a false positive. Validation
was already used to pick the checkpoint, so it is not reported as the score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_main
from src.config import load_config
from src.detect.engine import evaluate_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report held-out test mAP, precision, recall, and mistakes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.detect.yaml")
    parser.add_argument("--output-dir", default=None, help="Folder written by detect_train.py.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to the best.pt from training.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence for the miss/false-positive list.")
    parser.add_argument("--match-iou", type=float, default=None, help="IoU required to call a box correct.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    metrics = evaluate_detector(cfg, checkpoint, conf=args.conf, match_iou=args.match_iou)
    if metrics is None:
        raise SystemExit(1)


if __name__ == "__main__":
    run_main(main)

"""Draw a box around every detected object.

Examples:
    python detect_predict.py image.jpg
    python detect_predict.py video.mp4
    python detect_predict.py 0

``0`` is the default webcam. Press Ctrl+C to stop a video or a webcam.
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
from src.detect.engine import DEFAULT_MODEL, predict_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict objects in a photo, a folder, a video, or a webcam. "
            f"detect_train.py defaults to {DEFAULT_MODEL}; this command loads that run's best.pt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="Image, folder, video, or webcam index such as 0.")
    parser.add_argument(
        "--config",
        default="config.detect.yaml",
        help=f"YAML settings file. The default detection model is {DEFAULT_MODEL}.",
    )
    parser.add_argument("--output-dir", default=None, help="Folder written by detect_train.py.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to the best.pt from training.")
    parser.add_argument("--conf", type=float, default=None, help="Hide boxes below this confidence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    predict_detector(cfg, source=args.source, checkpoint=checkpoint, conf=args.conf)


if __name__ == "__main__":
    run_main(main)

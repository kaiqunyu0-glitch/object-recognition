"""Fine-tune a YOLO detector on your labeled photos.

Example:
    python detect_train.py --data /path/to/dataset

The folder may be a YOLO export (data.yaml plus images and labels), a COCO
JSON file, or a Label Studio JSON export next to the photos. Settings live
in config.detect.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_main
from src.config import apply_overrides, load_config
from src.detect.engine import fit_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-object detector on your labeled photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Dataset folder, data.yaml, or JSON export.")
    parser.add_argument("--config", default="config.detect.yaml", help="YAML settings file.")
    parser.add_argument("--output-dir", default=None, help="Where to write the checkpoint and reports.")
    parser.add_argument("--model", default=None, help="For example yolo11n.pt, yolo11s.pt, or yolov8n.pt.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or mps.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value. Example: epochs=40",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set or [])
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.model:
        cfg["model"] = args.model
    if args.device:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    fit_detector(cfg, data_root=Path(args.data))


if __name__ == "__main__":
    run_main(main)

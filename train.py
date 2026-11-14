"""Train an image classifier on your own photos.

Example:
    python train.py --data /path/to/photos

The folder can be class folders, or train/val/test class folders.
Settings live in config.yaml. This file reads the command and starts training.
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
from src.engine import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an image classifier on your own photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Folder of class photos, or a folder that contains train/val/test.",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML settings file.")
    parser.add_argument("--csv", default=None, help="Optional CSV with columns path,label.")
    parser.add_argument("--output-dir", default=None, help="Where to write the checkpoint and reports.")
    parser.add_argument("--model", default=None, help="Backbone name, for example convnext_tiny or tiny.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download ImageNet weights.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value. Example: train.full_epochs=20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set or [])
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.model:
        cfg["model"]["name"] = args.model
    if args.device:
        cfg["device"] = args.device
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    fit(cfg, data_root=Path(args.data), csv_path=args.csv)


if __name__ == "__main__":
    run_main(main)

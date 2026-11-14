"""Fine-tune a YOLO detector on your labeled photos.

Example:
    python detect_train.py --data /path/to/dataset

The folder may be a YOLO export (data.yaml plus images and labels), a COCO
JSON file, or a Label Studio JSON export next to the photos. Settings live
in config.detect.yaml. The default model is yolo11m.pt. Pass ``--model`` to
switch to yolo11n.pt, yolo11s.pt, yolo11l.pt, or yolo11x.pt.
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
from src.detect.engine import DEFAULT_MODEL, fit_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-object detector on your labeled photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Dataset folder, data.yaml, or JSON export.")
    parser.add_argument(
        "--config",
        default="config.detect.yaml",
        help=f"YAML settings file. The default model there is {DEFAULT_MODEL}.",
    )
    parser.add_argument("--output-dir", default=None, help="Where to write the checkpoint and reports.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Detection weights. The config default is %(default)s. "
            "Pass this flag to switch to yolo11n.pt, yolo11s.pt, yolo11l.pt, yolo11x.pt, or yolov8n.pt."
        ),
    )
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


def option_passed(flag: str, argv: list[str] | None = None) -> bool:
    """True when ``flag`` was written on the command line, not filled in by argparse."""
    args = sys.argv[1:] if argv is None else argv
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def resolve_model(config_model: str, cli_model: str, passed: bool) -> str:
    """Use ``--model`` only when the user passed it, so a config file can still choose."""
    return cli_model if passed else config_model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set or [])
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    cfg["model"] = resolve_model(str(cfg.get("model") or DEFAULT_MODEL), args.model, option_passed("--model"))
    if args.device:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    fit_detector(cfg, data_root=Path(args.data))


if __name__ == "__main__":
    run_main(main)

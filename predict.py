"""Predict the class of one or more photos.

Example:
    python predict.py image.jpg

If the top probability is below the threshold, the decision is ``unknown``
instead of a forced class.
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
from src.data import is_image, read_image
from src.device import describe_device, get_device
from src.inference import format_prediction, load_for_inference, probabilities_for_image
from src.transforms import eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the top classes for a photo, or say unknown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", nargs="+", help="Image files, or a folder of images.")
    parser.add_argument("--config", default="config.yaml", help="YAML settings file.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to best.pt in output_dir.")
    parser.add_argument("--top-k", type=int, default=None, help="How many classes to print.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Below this confidence the decision is unknown.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Also score a horizontal flip and average the two predictions.",
    )
    return parser.parse_args()


def expand_inputs(raw_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(child for child in path.iterdir() if child.is_file() and is_image(child))
            if not found:
                raise FileNotFoundError(f"No images found in {path}.")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Could not find {path}.")
    return files


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.config)
    device = get_device(str(cfg.get("device", "auto")))
    model, checkpoint = load_for_inference(checkpoint_path, device)
    image_size = int(checkpoint["image_size"])
    max_side = int(checkpoint.get("max_side", cfg["data"]["max_side"]))
    transform = eval_transform(image_size)
    top_k = args.top_k if args.top_k is not None else int(cfg["predict"]["top_k"])
    threshold = (
        args.threshold if args.threshold is not None else float(cfg["predict"]["unknown_threshold"])
    )
    tta = bool(args.tta or cfg["eval"]["tta"])
    print(f"Device: {describe_device(device)}", flush=True)
    print(
        f"Model: {checkpoint['model_name']}  image_size={image_size}  "
        f"unknown below {threshold:.0%}",
        flush=True,
    )
    for path in expand_inputs(args.images):
        image = read_image(path, max_side=max_side)
        tensor = transform(image)
        probs = probabilities_for_image(model, tensor, device, tta=tta)
        print(
            format_prediction(path, list(checkpoint["class_names"]), probs, top_k, threshold),
            flush=True,
        )


if __name__ == "__main__":
    run_main(main)

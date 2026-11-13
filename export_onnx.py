"""Export the trained model to ONNX for use in another program.

Example:
    python export_onnx.py

The ONNX file returns probabilities in the same order as the labels JSON
written next to it. Input is a float tensor shaped [batch, 3, image_size,
image_size], already resized and normalized the same way as predict.py
(ImageNet mean and standard deviation). For a single photo, prefer predict.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.checkpoint import default_checkpoint
from src.cli import run_main
from src.inference import load_for_inference


class _Probabilities(nn.Module):
    """Wrap the classifier so the exported file returns probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(inputs), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained classifier to an ONNX file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="YAML settings file.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to best.pt in output_dir.")
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to model.onnx next to the checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.config)
    model, checkpoint = load_for_inference(checkpoint_path, torch.device("cpu"))
    image_size = int(checkpoint["image_size"])
    wrapper = _Probabilities(model).eval()
    output = Path(args.output) if args.output else checkpoint_path.resolve().parent / "model.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)
    # dynamo=False keeps the older exporter. It does not need the extra
    # onnxscript package, and it is enough for this feed-forward network.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(output),
            input_names=["image"],
            output_names=["probabilities"],
            opset_version=17,
            dynamo=False,
            dynamic_axes={
                "image": {0: "batch"},
                "probabilities": {0: "batch"},
            },
        )
    labels_path = output.with_name(output.stem + ".labels.json")
    labels_path.write_text(
        json.dumps(list(checkpoint["class_names"]), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}", flush=True)
    print(f"Class order: {labels_path}", flush=True)
    print(
        f"Input: float32 tensor [batch, 3, {image_size}, {image_size}], ImageNet-normalized.",
        flush=True,
    )
    print("Output: probabilities in that class order. They add up to 1.", flush=True)


if __name__ == "__main__":
    run_main(main)

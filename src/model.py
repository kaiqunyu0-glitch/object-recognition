"""Build a classifier: a pretrained backbone plus a new last layer.

Transfer learning keeps the visual features learned on ImageNet and replaces
the original 1000-way layer with one output per class in this project.
Stage 1 trains only that new layer. Stage 2 trains every layer.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Small_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_v2_s,
    mobilenet_v3_small,
    vit_b_16,
)

from src.checkpoint import load_checkpoint

# ``tiny`` is a randomly initialized network for the smoke test. It does not
# download weights and it is not strong enough for real phone photos.
BACKBONES = (
    "convnext_tiny",
    "efficientnet_v2_s",
    "vit_b_16",
    "mobilenet_v3_small",
    "tiny",
)


class TinyCNN(nn.Module):
    """A small network used by the automatic test.

    Real photos should use ``convnext_tiny`` or another pretrained backbone.
    """

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Keep a small spatial grid. Pooling straight to 1x1 would throw
            # away the shape and leave only an average color.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(32 * 4 * 4, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


def build_model(name: str, num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    """Create a model that outputs ``num_classes`` scores.

    ``pretrained=True`` downloads ImageNet weights the first time. predict.py
    and evaluate.py pass ``pretrained=False`` and then load the saved weights,
    so they do not need that download.
    """
    if num_classes < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")
    if name not in BACKBONES:
        choices = ", ".join(BACKBONES)
        raise ValueError(f"Unknown model {name!r}. Choose one of: {choices}.")
    if name == "tiny":
        if pretrained:
            print("The tiny model has no pretrained weights. Starting from random weights.")
        return TinyCNN(num_classes=num_classes, dropout=dropout)

    builders = {
        "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
        "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT),
        "vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT),
        "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
    }
    builder, weights = builders[name]
    model = builder(weights=weights if pretrained else None)
    _replace_last_linear(model, num_classes=num_classes, dropout=dropout)
    return model


def _replace_last_linear(model: nn.Module, num_classes: int, dropout: float) -> None:
    """Discard the ImageNet 1000-way layer and attach one for our classes.

    Dropout sits in front of the new layer so the head memorizes the training
    photos a little less easily. The rest of the network is unchanged.
    """
    last_name = ""
    last_linear: nn.Linear | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_linear = module
    if last_linear is None or not last_name:
        raise RuntimeError("This model has no Linear layer to use as a classification head.")

    parent, child = _parent_of(model, last_name)
    new_head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(last_linear.in_features, num_classes),
    )
    _set_child(parent, child, new_head)


def _parent_of(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parent: nn.Module = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, child: str, module: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child.isdigit():
        parent[int(child)] = module
    else:
        setattr(parent, child, module)


def last_linear(model: nn.Module) -> nn.Linear:
    """The layer that produces one score per class."""
    found: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            found = module
    if found is None:
        raise RuntimeError("No classification layer found.")
    return found


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: train the new last layer, leave the pretrained layers fixed."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in last_linear(model).parameters():
        parameter.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Stage 2: every layer is allowed to learn."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def apply_train_mode(model: nn.Module, head_only: bool) -> None:
    """Set train or eval mode for the current stage.

    Stage 1 keeps the backbone in eval mode. That stops BatchNorm from
    updating its statistics on our small dataset, and it turns off Dropout
    inside the pretrained layers. Only the new head stays in train mode.
    Stage 2 trains the whole network.
    """
    if not head_only:
        model.train()
        return
    model.eval()
    _head_container(model).train()


def _head_container(model: nn.Module) -> nn.Module:
    """The Sequential that holds the final Linear layer, or the layer itself."""
    linear = last_linear(model)
    for module in model.modules():
        if any(child is linear for child in module.children()):
            return module
    return linear


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Rebuild the architecture from a checkpoint and load its weights.

    Pretrained ImageNet weights are not downloaded here. The checkpoint
    already contains the fine-tuned weights.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(
        name=checkpoint["model_name"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint

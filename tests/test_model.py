"""Tests that the head replacement, freezing, and a tiny overfit step work."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model import (
    BACKBONES,
    apply_train_mode,
    build_model,
    freeze_backbone,
    last_linear,
    unfreeze_all,
)


def test_backbones_replace_the_final_layer():
    for name in BACKBONES:
        model = build_model(name, num_classes=5, pretrained=False, dropout=0.1)
        assert last_linear(model).out_features == 5
        del model


def test_convnext_and_tiny_forward_shapes():
    for name, size in (("tiny", 32), ("convnext_tiny", 64), ("mobilenet_v3_small", 64)):
        model = build_model(name, num_classes=4, pretrained=False, dropout=0.0)
        model.eval()
        output = model(torch.zeros(1, 3, size, size))
        assert output.shape == (1, 4)


def test_freeze_trains_only_the_new_layer():
    model = build_model("convnext_tiny", num_classes=3, pretrained=False, dropout=0.2)
    freeze_backbone(model)
    trainable = [module for module in model.modules() if isinstance(module, nn.Linear) and any(p.requires_grad for p in module.parameters())]
    assert len(trainable) == 1
    assert trainable[0].out_features == 3
    apply_train_mode(model, head_only=True)
    assert model.training is False
    assert last_linear(model).training is True
    unfreeze_all(model)
    apply_train_mode(model, head_only=False)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.training is True


def test_tiny_can_overfit_two_solid_colors():
    torch.manual_seed(0)
    model = build_model("tiny", num_classes=2, pretrained=False, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    red = torch.zeros(4, 3, 32, 32)
    red[:, 0] = 1
    green = torch.zeros(4, 3, 32, 32)
    green[:, 1] = 1
    images = torch.cat([red, green], dim=0)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    predictions = model(images).argmax(dim=1)
    assert torch.equal(predictions, labels)

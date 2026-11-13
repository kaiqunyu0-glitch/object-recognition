"""Score one photo and decide whether the answer is confident enough.

``unknown`` means the top class is below the threshold. The photo may be a
class that was never trained, or just a strange angle. Showing the top
guesses as well lets you see what the model almost said.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.model import load_trained_model


def forward_logits(model: torch.nn.Module, images: torch.Tensor, tta: bool) -> torch.Tensor:
    """Average the normal view with a horizontal flip when ``tta`` is on."""
    logits = model(images)
    if tta:
        flipped = torch.flip(images, dims=(3,))
        logits = (logits + model(flipped)) * 0.5
    return logits


def probabilities_for_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    tta: bool,
) -> list[float]:
    """Return one probability per class for a single CHW tensor."""
    batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = forward_logits(model, batch, tta=tta)
        probs = torch.softmax(logits, dim=1)[0]
    return probs.detach().cpu().tolist()


def format_prediction(
    path: Path,
    class_names: list[str],
    probs: list[float],
    top_k: int,
    threshold: float,
) -> str:
    """Text block for one photo, including ``unknown`` when confidence is low."""
    if not 0 <= threshold <= 1:
        raise ValueError("predict.unknown_threshold must be between 0 and 1.")
    if top_k < 1:
        raise ValueError("predict.top_k must be at least 1.")
    order = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)
    lines = [str(path)]
    for rank, index in enumerate(order[:top_k], start=1):
        lines.append(f"  {rank}. {class_names[index]:<16} {probs[index] * 100:5.1f}%")
    best = order[0]
    if probs[best] < threshold:
        lines.append(
            f"  decision: unknown  (top confidence {probs[best] * 100:.1f}% "
            f"is below {threshold * 100:.0f}%)"
        )
    else:
        lines.append(f"  decision: {class_names[best]}")
    return "\n".join(lines)


def load_for_inference(checkpoint_path: Path, device: torch.device):
    """Load a trained model in eval mode."""
    model, checkpoint = load_trained_model(checkpoint_path, device)
    return model, checkpoint

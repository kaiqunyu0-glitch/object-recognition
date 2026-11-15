"""Pick CPU, NVIDIA CUDA, or Apple MPS without a manual switch."""

from __future__ import annotations

import torch


def get_device(requested: str | None = None) -> torch.device:
    """Return the device training should use.

    ``auto`` (the default) prefers CUDA, then Apple MPS, then CPU.
    """
    choice = (requested or "auto").lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "You asked for cuda, but this computer does not have a usable NVIDIA GPU. "
            "Use device: auto or device: cpu."
        )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not torch.backends.mps.is_available():
            raise RuntimeError(
                "You asked for mps, but this Mac is not exposing an Apple GPU. "
                "Use device: auto or device: cpu."
            )
    if device.type not in {"cpu", "cuda", "mps"}:
        raise RuntimeError("device must be auto, cpu, cuda, or mps.")
    return device


def describe_device(device: torch.device) -> str:
    """A short label for the log, including the GPU name when there is one."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return "cpu"

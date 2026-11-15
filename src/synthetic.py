"""Draw a tiny dataset of shapes.

The pictures are not real phone photos. They exist so the training commands
can be tried, and so CI can run, before any personal photos are ready.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ("circle", "square", "triangle")

# Strong, different colors. A tiny network can learn these quickly, which is
# what the smoke test needs. Real phone photos rely on the pretrained backbone.
CLASS_COLORS = {
    "circle": (220, 30, 30),
    "square": (30, 170, 40),
    "triangle": (30, 60, 220),
}


def render_shape(kind: str, size: int, rng: random.Random) -> Image.Image:
    """Draw one colored shape on a noisy background.

    The noise and the random position keep two photos of the same class from
    looking like a burst shot to the near-duplicate check.
    """
    if kind not in CLASS_COLORS:
        raise ValueError(f"Unknown shape {kind!r}.")
    noise = np.random.default_rng(rng.randrange(2**31))
    background = noise.integers(60, 230, size=(size, size, 3), dtype=np.int16)
    image = Image.fromarray(background.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    side = rng.randint(size // 3, size // 2)
    x0 = rng.randint(0, size - side)
    y0 = rng.randint(0, size - side)
    x1 = x0 + side
    y1 = y0 + side
    color = tuple(
        max(0, min(255, channel + rng.randint(-20, 20))) for channel in CLASS_COLORS[kind]
    )
    if kind == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif kind == "square":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    else:
        draw.polygon([(x0 + side // 2, y0), (x0, y1), (x1, y1)], fill=color)
    return image


def make_synthetic_dataset(
    out_dir: Path,
    per_class: int = 18,
    image_size: int = 128,
    seed: int = 0,
) -> Path:
    """Write ``out_dir/<class>/<index>.jpg`` for circle, square, and triangle."""
    if per_class < 1:
        raise ValueError("--per-class must be at least 1.")
    if image_size < 32:
        raise ValueError("--image-size must be at least 32.")
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in CLASS_NAMES:
        folder = out_dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            image = render_shape(kind, image_size, rng)
            image.save(folder / f"{index:03d}.jpg", quality=95)
    return out_dir

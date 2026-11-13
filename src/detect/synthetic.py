"""Draw a tiny detection dataset: several shapes in each photo.

The pictures are not real phone photos. They let the detection commands run,
and they let CI run, before any labeled photos exist.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ("circle", "square", "triangle")
CLASS_COLORS = {
    "circle": (220, 40, 40),
    "square": (40, 180, 60),
    "triangle": (40, 70, 220),
}


def _draw(draw: ImageDraw.ImageDraw, kind: str, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    if kind == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif kind == "square":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    else:
        draw.polygon([(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)], fill=color)


def make_detection_dataset(
    out_dir: Path,
    images: int = 12,
    image_size: int = 160,
    seed: int = 0,
) -> Path:
    """Write ``images/*.jpg`` and ``labels/*.txt`` plus ``classes.txt``.

    Every photo contains one circle, one square, and one triangle in three
    separate areas, so the boxes do not sit on top of each other.
    """
    if images < 1:
        raise ValueError("--images must be at least 1.")
    if image_size < 96:
        raise ValueError("--image-size must be at least 96 so three boxes fit.")
    rng = random.Random(seed)
    image_dir = out_dir / "images"
    label_dir = out_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    side = image_size // 5
    # Three regions: top-left, top-right, bottom-center.
    anchors = (
        (image_size // 12, image_size // 12),
        (image_size - image_size // 12 - side, image_size // 12),
        (image_size // 2 - side // 2, image_size - image_size // 8 - side),
    )
    for index in range(images):
        noise = np.random.default_rng(rng.randrange(2**31))
        background = noise.integers(70, 220, size=(image_size, image_size, 3), dtype=np.int16)
        image = Image.fromarray(background.astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        lines = []
        for kind, (anchor_x, anchor_y) in zip(CLASS_NAMES, anchors):
            jitter_x = rng.randint(-4, 4)
            jitter_y = rng.randint(-4, 4)
            x0 = min(max(0, anchor_x + jitter_x), image_size - side - 1)
            y0 = min(max(0, anchor_y + jitter_y), image_size - side - 1)
            x1 = x0 + side
            y1 = y0 + side
            color = tuple(max(0, min(255, channel + rng.randint(-15, 15))) for channel in CLASS_COLORS[kind])
            _draw(draw, kind, (x0, y0, x1, y1), color)
            class_id = CLASS_NAMES.index(kind)
            cx = ((x0 + x1) / 2) / image_size
            cy = ((y0 + y1) / 2) / image_size
            width = side / image_size
            height = side / image_size
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
        image.save(image_dir / f"{index:03d}.jpg", quality=95)
        (label_dir / f"{index:03d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    return out_dir

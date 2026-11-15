"""Warn when near-copy photos land in different splits.

A burst of almost identical shots can put one copy in train and another in
test. The test score then partly measures "have I seen this exact photo?"
instead of "can I recognize this object?". This is a difference hash
(dHash): shrink the photo, compare neighboring pixels, and count how many
comparisons disagree.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import read_image


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a perceptual hash. Equal images produce equal hashes."""
    if hash_size < 4:
        raise ValueError("duplicates.hash_size must be at least 4.")
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            bits = (bits << 1) | (1 if pixels[row, col] > pixels[row, col + 1] else 0)
    return bits


def hamming(left: int, right: int) -> int:
    """How many bits differ. 0 means the hashes match."""
    return (left ^ right).bit_count()


def find_cross_split_duplicates(
    manifest: list[dict[str, str]],
    hash_size: int,
    max_distance: int,
    max_side: int,
) -> list[dict[str, object]]:
    """Compare photos that are not in the same split.

    Pairs in the same split are allowed: extra views of one object help
    training. Pairs across splits are the leak.
    """
    hashed: list[dict[str, object]] = []
    for row in manifest:
        image = read_image(Path(row["path"]), max_side=max_side)
        hashed.append(
            {
                "path": row["path"],
                "split": row["split"],
                "hash": dhash(image, hash_size=hash_size),
            }
        )

    buckets: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for item in hashed:
        split = str(item["split"])
        if split in buckets:
            buckets[split].append(item)

    pairs: list[dict[str, object]] = []
    comparisons = (("train", "val"), ("train", "test"), ("val", "test"))
    for left_split, right_split in comparisons:
        for left in buckets[left_split]:
            for right in buckets[right_split]:
                distance = hamming(int(left["hash"]), int(right["hash"]))
                if distance <= max_distance:
                    pairs.append(
                        {
                            "path_a": left["path"],
                            "split_a": left_split,
                            "path_b": right["path"],
                            "split_b": right_split,
                            "distance": distance,
                        }
                    )
    pairs.sort(key=lambda item: (int(item["distance"]), str(item["path_a"]), str(item["path_b"])))
    return pairs


def report_duplicates(pairs: list[dict[str, object]], csv_path: Path) -> None:
    """Print a short warning and write every pair to ``duplicates.csv``."""
    print(f"Near-duplicate check: {len(pairs)} suspicious pair(s).", flush=True)
    if not pairs:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path_a", "split_a", "path_b", "split_b", "distance"],
        )
        writer.writeheader()
        writer.writerows(pairs)

    print(
        "WARNING: near-duplicate photos were found in different splits.\n"
        "A photo and a near-copy can make the test score look better than it is.\n"
        f"Open {csv_path} and delete the copies, then train again.",
        flush=True,
    )
    for pair in pairs[:5]:
        print(
            f"  distance {pair['distance']}: [{pair['split_a']}] {pair['path_a']}",
            flush=True,
        )
        print(f"             [{pair['split_b']}] {pair['path_b']}", flush=True)
    extra = len(pairs) - 5
    if extra > 0:
        print(f"  ... {extra} more pair(s) in {csv_path}", flush=True)

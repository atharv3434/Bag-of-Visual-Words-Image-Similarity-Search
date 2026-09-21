"""
Generate synthetic textured-pattern images for Bag-of-Visual-Words
similarity search practice.

Unlike flat, solid-color shapes, these patterns (stripes, dots,
checkerboard, crosshatch) have lots of corners and edges — the kind of
local structure that keypoint detectors like ORB actually need to find
meaningful features. A single solid-color shape gives a handful of corner
keypoints at most; these patterns give dozens to hundreds.

This project ships with pre-generated images already in place
(data/images/, data/metadata.json), so you don't need to run this to try
the project out. Run it again for a fresh random set.

Usage:
    python data/generate_data.py [--n-per-class 15] [--seed 42]
"""

import argparse
import json
import os

import numpy as np
import cv2

PATTERNS = ["stripes", "dots", "checkerboard", "crosshatch"]
CANVAS_SIZE = 160
BACKGROUND_COLOR = (245, 245, 245)  # BGR

PALETTE = [
    (50, 50, 200), (50, 170, 50), (200, 100, 40),
    (180, 60, 180), (40, 160, 210), (90, 90, 90),
]


def _random_color(rng):
    return tuple(int(c) for c in PALETTE[rng.integers(0, len(PALETTE))])


def draw_stripes(canvas, rng):
    """Dashed diagonal stripes rather than continuous lines — a continuous
    straight line has no corners for a keypoint detector like ORB to find
    (confirmed by testing: it produced zero keypoints). Breaking each
    stripe into short dashes gives every dash two endpoints (corners),
    which is what makes this pattern usable for BoVW at all.
    """
    color = _random_color(rng)
    spacing = int(rng.integers(10, 22))
    thickness = int(rng.integers(3, spacing - 2))
    angle = float(rng.uniform(0, 180))
    dash_len = int(rng.integers(10, 22))
    gap_len = int(rng.integers(6, 14))

    line_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for offset in range(-CANVAS_SIZE, CANVAS_SIZE * 2, spacing):
        y = -CANVAS_SIZE
        while y < CANVAS_SIZE * 2:
            cv2.line(line_canvas, (offset, y), (offset, min(y + dash_len, CANVAS_SIZE * 2)), 255, thickness)
            y += dash_len + gap_len

    center = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(line_canvas, rot, (CANVAS_SIZE, CANVAS_SIZE))
    canvas[rotated > 0] = color


def draw_dots(canvas, rng):
    color = _random_color(rng)
    spacing = int(rng.integers(16, 30))
    radius = int(rng.integers(4, spacing // 2))
    jitter = spacing // 4

    for y in range(spacing // 2, CANVAS_SIZE, spacing):
        for x in range(spacing // 2, CANVAS_SIZE, spacing):
            dx = int(rng.integers(-jitter, jitter + 1))
            dy = int(rng.integers(-jitter, jitter + 1))
            cv2.circle(canvas, (x + dx, y + dy), radius, color, thickness=-1)


def draw_checkerboard(canvas, rng):
    color = _random_color(rng)
    cell = int(rng.integers(14, 28))
    for y in range(0, CANVAS_SIZE, cell):
        for x in range(0, CANVAS_SIZE, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                cv2.rectangle(canvas, (x, y), (x + cell, y + cell), color, thickness=-1)


def draw_crosshatch(canvas, rng):
    color = _random_color(rng)
    spacing = int(rng.integers(14, 26))
    thickness = int(rng.integers(2, 4))
    for offset in range(0, CANVAS_SIZE, spacing):
        cv2.line(canvas, (offset, 0), (offset, CANVAS_SIZE), color, thickness)
        cv2.line(canvas, (0, offset), (CANVAS_SIZE, offset), color, thickness)


DRAW_FUNCS = {
    "stripes": draw_stripes,
    "dots": draw_dots,
    "checkerboard": draw_checkerboard,
    "crosshatch": draw_crosshatch,
}


def generate_image(rng, pattern):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BACKGROUND_COLOR, dtype=np.uint8)
    DRAW_FUNCS[pattern](canvas, rng)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic textured-pattern images.")
    parser.add_argument("--n-per-class", type=int, default=15, help="Images per pattern category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records = []
    idx = 0
    for pattern in PATTERNS:
        for _ in range(args.n_per_class):
            canvas = generate_image(rng, pattern)
            filename = f"{pattern}_{idx:03d}.png"
            cv2.imwrite(os.path.join(images_dir, filename), canvas)
            records.append({"file": filename, "category": pattern})
            idx += 1

    rng.shuffle(records)

    metadata_path = os.path.join(args.out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"categories": PATTERNS, "images": records}, f, indent=2)

    print(f"Wrote {len(records)} images to {images_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

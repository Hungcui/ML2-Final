#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def pick_n_per_digit(y: np.ndarray, n_per_digit: int, seed: int) -> dict[int, np.ndarray]:
    """
    Returns a dict: digit -> indices (length n_per_digit) sampled without replacement.
    """
    rng = np.random.default_rng(seed)
    picks: dict[int, np.ndarray] = {}

    for d in range(10):
        idx = np.flatnonzero(y == d)
        if idx.size < n_per_digit:
            raise ValueError(f"Digit {d} only has {idx.size} samples; need {n_per_digit}.")
        picks[d] = rng.choice(idx, size=n_per_digit, replace=False)

    return picks


def to_uint8_grayscale(img_28x28: np.ndarray) -> np.ndarray:
    """
    Ensures a (28,28) image is uint8 in [0,255] for PNG saving.
    """
    if img_28x28.dtype == np.uint8:
        return img_28x28
    # If someone normalized to [0,1], convert back safely
    img = np.asarray(img_28x28, dtype=np.float32)
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Export MNIST test images: N per digit (0-9) to PNG folders.")
    parser.add_argument("--n_per_digit", type=int, default=3, help="How many images per digit (default: 3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection (default: 42).")
    parser.add_argument("--out_dir", type=str, default="mnist_test_png", help="Output directory (default: mnist_test_png).")
    parser.add_argument("--flat_labels", action="store_true",
                        help="If set: save images directly under out_dir instead of digit subfolders.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load MNIST
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    picks = pick_n_per_digit(y_test, n_per_digit=args.n_per_digit, seed=args.seed)

    # Write labels CSV (human-readable)
    labels_path = out_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "original_test_index"])

        total_saved = 0
        for d in range(10):
            digit_indices = picks[d]

            # Create per-digit folder unless user wants flat structure
            if args.flat_labels:
                digit_dir = out_dir
            else:
                digit_dir = out_dir / str(d)
                digit_dir.mkdir(parents=True, exist_ok=True)

            for j, original_idx in enumerate(digit_indices):
                img_u8 = to_uint8_grayscale(x_test[original_idx])  # (28,28) uint8

                filename = f"{j:02d}_label{d}_idx{int(original_idx)}.png"
                filepath = digit_dir / filename

                # Save as grayscale PNG
                Image.fromarray(img_u8, mode="L").save(filepath)

                # Write relative path for portability
                relpath = filepath.relative_to(out_dir)
                writer.writerow([str(relpath), int(d), int(original_idx)])
                total_saved += 1

    # Sanity check counts
    selected_labels = []
    for d in range(10):
        selected_labels.extend([d] * args.n_per_digit)
    counts = np.bincount(np.array(selected_labels), minlength=10)

    print(f"Saved {total_saved} images to: {out_dir.resolve()}")
    print(f"Wrote labels file: {labels_path.resolve()}")
    print("Expected counts per digit 0..9:", counts.tolist())


if __name__ == "__main__":
    main()

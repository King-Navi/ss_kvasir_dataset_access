#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import cv2
import polars as pl

from readmeta import load_csv, filter_missing_coords, FINDING_CLASS_MAP
from video import get_video_frame
from constants import OUTPUT


def compute_axis_aligned_bbox(row) -> tuple[float, float, float, float]:
    """
    Compute an axis-aligned bounding box (x_min, y_min, x_max, y_max)
    from the 4 corner coordinates x1..y4.
    """
    xs = [row["x1"], row["x2"], row["x3"], row["x4"]]
    ys = [row["y1"], row["y2"], row["y3"], row["y4"]]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def yolo_normalize_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """
    Normalize bounding box to YOLO format (x_center, y_center, w, h),
    with all values in [0, 1].
    """
    x_min = max(0.0, min(x_min, img_width - 1.0))
    x_max = max(0.0, min(x_max, img_width - 1.0))
    y_min = max(0.0, min(y_min, img_height - 1.0))
    y_max = max(0.0, min(y_max, img_height - 1.0))

    x_center = ((x_min + x_max) / 2.0) / img_width
    y_center = ((y_min + y_max) / 2.0) / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    return x_center, y_center, w, h


def build_class_mappings():
    """
    Build mappings:
      - full_class_name -> short_key (e.g. 'Polyp' -> 'polyp')
      - full_class_name -> numeric_id (0..13)
    Based directly on FINDING_CLASS_MAP insertion order.
    """
    full_to_key = {v: k for k, v in FINDING_CLASS_MAP.items()}
    full_to_id = {v: idx for idx, (k, v) in enumerate(FINDING_CLASS_MAP.items())}
    return full_to_key, full_to_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 2 frames per class with bounding boxes and YOLO-style normalized "
            "coordinates, using readmeta.py and video.py."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "--separator",
        default=";",
        help="CSV separator (default=';').",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of samples to draw per class (default=2).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT,
        help="Output directory for images and YOLO label files.",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # complete coordinates
    df = load_csv(csv_path, args.separator)
    df = filter_missing_coords(df)

    full_to_key, full_to_id = build_class_mappings()
    df = df.filter(pl.col("finding_class").is_in(list(full_to_key.keys())))

    os.makedirs(args.output_dir, exist_ok=True)

    per_class_counter = {short: 0 for short in FINDING_CLASS_MAP.keys()}

    for short_key, full_name in FINDING_CLASS_MAP.items():
        class_rows = df.filter(pl.col("finding_class") == full_name)
        if class_rows.height == 0:
            print(f"[WARN] No rows found for class '{full_name}' ({short_key})")
            continue

        n = min(args.samples_per_class, class_rows.height)
        # Fix seed for reproducibility
        sampled = class_rows.sample(n=n, shuffle=True, seed=42)

        print(f"[INFO] Class '{full_name}' ({short_key}) -> taking {n} samples.")

        for row in sampled.iter_rows(named=True):
            video_id = row["video_id"]
            frame_number = int(row["frame_number"])

            frame = get_video_frame(frame_number, video_id)

            x_min, y_min, x_max, y_max = compute_axis_aligned_bbox(row)

            img_h, img_w = frame.shape[:2]

            x_center, y_center, bw, bh = yolo_normalize_bbox(
                x_min, y_min, x_max, y_max, img_w, img_h
            )

            # bounding box on the image (using original coords)
            pt1 = (int(round(x_min)), int(round(y_min)))
            pt2 = (int(round(x_max)), int(round(y_max)))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

            class_id = full_to_id[full_name]

            idx = per_class_counter[short_key]
            per_class_counter[short_key] += 1
            base_name = f"{short_key}_{idx}_x_{video_id}"

            img_path = os.path.join(args.output_dir, base_name + ".png")
            label_path = os.path.join(args.output_dir, base_name + ".txt")

            # Save image
            cv2.imwrite(img_path, frame)

            # Save YOLO-style label
            # with open(label_path, "w", encoding="utf-8") as f:
            #     # Format: class_id x_center y_center w h
            #     f.write(
            #         f"{class_id} "
            #         f"{x_center:.6f} {y_center:.6f} "
            #         f"{bw:.6f} {bh:.6f}\n"
            #    )

            print(f"[OK] Saved image: {img_path}")
            print(f"[OK] Saved label: {label_path}")

    return 0

"""
python main.py \
  --csv /home/ivan/Downloads/jose_luis_act/input/metadata.csv \
  --separator ';' \
  --samples-per-class 4
"""


if __name__ == "__main__":
    raise SystemExit(main())

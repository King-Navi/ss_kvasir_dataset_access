#!/usr/bin/env python3
import argparse
import os
import math
from pathlib import Path

import cv2
import polars as pl

from readmeta import (
    load_csv,
    build_class_mappings
)
from video import get_video_frame
from constants import FINDING_CLASS_MAP , COORD_COLS, OUTPUT, THICKNESS


def get_out_of_bounds_points(row, img_width: int, img_height: int):
    """
    Return a list of (i, x, y) for each corner i whose (xi, yi)
    is outside [0, img_width-1] x [0, img_height-1].
    """
    bad_points = []
    for i in range(1, 5):
        x = row[f"x{i}"]
        y = row[f"y{i}"]
        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            bad_points.append((i, x, y))
    return bad_points



def clamp_coords_in_row(row, img_width: int, img_height: int) -> bool:
    """
    Clamp all (xi, yi) to [0, img_width-1] x [0, img_height-1] in-place.

    Returns True if any coordinate was actually clamped.
    """
    clamped = False
    max_x = img_width - 1
    max_y = img_height - 1

    for i in range(1, 5):
        x_key = f"x{i}"
        y_key = f"y{i}"
        old_x = row[x_key]
        old_y = row[y_key]

        new_x = max(0.0, min(float(old_x), float(max_x)))
        new_y = max(0.0, min(float(old_y), float(max_y)))

        if new_x != old_x or new_y != old_y:
            clamped = True
            row[x_key] = new_x
            row[y_key] = new_y

    return clamped

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


def complete_coords_expr() -> pl.Expr:
    """
    Polars expression:
    True if ALL coordinate columns are non-null and non-NaN.
    """
    return pl.all_horizontal(
        [pl.col(c).is_not_null() & ~pl.col(c).is_nan() for c in COORD_COLS]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export frames per class into separate folders. "
            "For each class: images WITH complete bounding box go to 'with_bbox', "
            "images WITHOUT go to 'no_bbox'. A summary index.txt is written per class."
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
        "--output-dir",
        default=OUTPUT,
        help="Output directory for class folders (default is video.OUTPUT).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help=(
            "Optional cap on number of rows per class (0 = no limit, default). "
            "This limit applies separately to with_bbox and no_bbox."
        ),
    )
    parser.add_argument(
        "--keep-first-duplicate",
        action="store_true",
        help=(
            "If set, for duplicated filenames keep the first occurrence "
            "and drop only the subsequent ones. "
            "If not set, drop all rows for any duplicated filename."
        ),
    )
    oob_group = parser.add_mutually_exclusive_group()
    oob_group.add_argument(
        "--drop-out-of-bounds",
        action="store_true",
        help=(
            "If set, skip any row whose coordinates fall outside "
            "the video frame size (no image/label will be written)."
        ),
    )
    oob_group.add_argument(
        "--clamp-out-of-bounds",
        action="store_true",
        help=(
            "If set, clamp any coordinate outside the frame to the "
            "valid min/max border instead of skipping the row."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[INFO] Loading CSV: {csv_path}")
    df = load_csv(str(csv_path), args.separator)
    #Duplicados
    if args.keep_first_duplicate:
        # Flag
        df = df.with_columns(
            pl.col("filename").is_duplicated().alias("is_dup")
        )

        dupes_df = df.filter(pl.col("is_dup"))
        print(f"[INFO] Found {dupes_df.height} duplicated rows by filename (extra occurrences only)")

        if dupes_df.height > 0:
            dupes_df.write_csv(
                os.path.join(args.output_dir, "duplicated_rows_by_filename.csv")
            )

        # only rows are not duplicates (includes the first appearance)

        df = df.filter(~pl.col("is_dup")).drop("is_dup")

    else:
        # No flag
        dupe_names = (
            df
            .group_by("filename")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
            .select("filename")
        )

        dupes_df = df.join(dupe_names, on="filename", how="inner")
        print(f"[INFO] Found {dupes_df.height} rows with duplicated filenames (including first occurrences)")

        if dupes_df.height > 0:
            dupes_df.write_csv(
                os.path.join(args.output_dir, "duplicated_rows_by_filename.csv")
            )

        # rows whose filename is NOT in the duplicate list
        df = df.join(dupe_names, on="filename", how="anti")

    print(f"[INFO] Rows after duplicate-filter: {df.height}")



    full_to_key, full_to_id = build_class_mappings()

    df = df.filter(pl.col("finding_class").is_in(list(full_to_key.keys())))
    print(f"[INFO] Rows after class filter: {df.height}")

    # Precompute for rows with complete coordinates (all non-null / non-NaN)
    complete_expr = complete_coords_expr()
    df_with_bbox = df.filter(complete_expr)
    df_no_bbox = df.filter(~complete_expr)

    print(f"[INFO] Rows WITH complete bbox: {df_with_bbox.height}")
    print(f"[INFO] Rows WITHOUT complete bbox: {df_no_bbox.height}")

    os.makedirs(args.output_dir, exist_ok=True)

    # FINDING_CLASS_MAP 
    for short_key, full_name in FINDING_CLASS_MAP.items():
        if full_name not in full_to_id:
            print(f"[WARN] Class '{full_name}' not found in mapping, skipping.")
            continue

        class_id = full_to_id[full_name]

        class_dir_name = f"{class_id:02d}_{short_key}"
        class_dir = os.path.join(args.output_dir, class_dir_name)
        bbox_dir = os.path.join(class_dir, "with_bbox")
        no_bbox_dir = os.path.join(class_dir, "no_bbox")

        os.makedirs(bbox_dir, exist_ok=True)
        os.makedirs(no_bbox_dir, exist_ok=True)

        class_with_bbox = df_with_bbox.filter(pl.col("finding_class") == full_name)
        class_no_bbox = df_no_bbox.filter(pl.col("finding_class") == full_name)

        if args.max_per_class > 0:
            class_with_bbox = class_with_bbox.head(args.max_per_class)
            class_no_bbox = class_no_bbox.head(args.max_per_class)

        print(
            f"[INFO] Class '{full_name}' ({short_key}, id={class_id}) -> "
            f"{class_with_bbox.height} with_bbox, {class_no_bbox.height} no_bbox"
        )

        bbox_files: list[str] = []
        no_bbox_files: list[str] = []

        # complete bbox
        bbox_counter = 0
        for row in class_with_bbox.iter_rows(named=True):
            video_id = row["video_id"]
            frame_number = int(row["frame_number"])

            try:
                frame = get_video_frame(frame_number, video_id)
            except Exception as e:
                print(
                    f"[WARN] Could not extract frame {frame_number} from video {video_id}: {e}"
                )
                continue

            img_h, img_w = frame.shape[:2]
            if args.drop_out_of_bounds:
                bad_points = get_out_of_bounds_points(row, img_w, img_h)
                if bad_points:
                    print(
                        f"[WARN] Skipping row for filename='{row['filename']}' "
                        f"because some coordinates are outside [0,{img_w-1}]x[0,{img_h-1}]"
                    )
                    for i, x, y in bad_points:
                        print(f"       -> (x{i}, y{i}) = ({x}, {y})")
                    continue
            elif args.clamp_out_of_bounds:
                bad_points = get_out_of_bounds_points(row, img_w, img_h)
                if bad_points:
                    print(
                        f"[INFO] Clamping coordinates for filename='{row['filename']}' "
                        f"to frame bounds [0,{img_w-1}]x[0,{img_h-1}]"
                    )
                    for i, x, y in bad_points:
                        print(f"       -> (x{i}, y{i}) was ({x}, {y})")
                    clamped = clamp_coords_in_row(row, img_w, img_h)
                    if clamped:
                        print("       -> Coordinates updated after clamping.")

            # Use original filename from metadata
            original_filename = row["filename"]  # e.g. "something_0123.jpg"
            original_filename = os.path.basename(original_filename)
            name_root, ext = os.path.splitext(original_filename)
            if not ext:
                ext = ".png"  # fallback if metadata has no extension

            # Compute bbox
            x_min, y_min, x_max, y_max = compute_axis_aligned_bbox(row)
            img_h, img_w = frame.shape[:2]

            # Normalize to YOLO
            x_center, y_center, bw, bh = yolo_normalize_bbox(
                x_min, y_min, x_max, y_max, img_w, img_h
            )

            # Draw rectangle (for visualization only)
            pt1 = (int(round(x_min)), int(round(y_min)))
            pt2 = (int(round(x_max)), int(round(y_max)))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), THICKNESS)

            # Image and label paths based on metadata filename
            img_filename = name_root + ext
            label_filename = name_root + ".txt" # YOLO label

            img_path = os.path.join(bbox_dir, img_filename)
            label_path = os.path.join(bbox_dir, label_filename)

            cv2.imwrite(img_path, frame)

            # YOLO label
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{bw:.6f} {bh:.6f}\n"
                )

            # Store relative path for the summary
            bbox_files.append(os.path.relpath(img_path, class_dir))

        # WITHOUT complete bbox
        no_bbox_counter = 0
        for row in class_no_bbox.iter_rows(named=True):
            video_id = row["video_id"]
            frame_number = int(row["frame_number"])

            try:
                frame = get_video_frame(frame_number, video_id)
            except Exception as e:
                print(
                    f"[WARN] Could not extract frame {frame_number} from video {video_id}: {e}"
                )
                continue

            # Use original filename from metadata
            original_filename = row["filename"]
            original_filename = os.path.basename(original_filename)
            name_root, ext = os.path.splitext(original_filename)
            if not ext:
                ext = ".png"

            img_filename = name_root + ext
            img_path = os.path.join(no_bbox_dir, img_filename)
            cv2.imwrite(img_path, frame)

            no_bbox_files.append(os.path.relpath(img_path, class_dir))

        # summary
        summary_path = os.path.join(class_dir, "index.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"class_id: {class_id}\n")
            f.write(f"class_short: {short_key}\n")
            f.write(f"class_full: {full_name}\n")
            f.write(f"with_bbox_count: {len(bbox_files)}\n")
            f.write(f"no_bbox_count: {len(no_bbox_files)}\n")
            f.write("\nwith_bbox_images:\n")
            for p in bbox_files:
                f.write(f"  {p}\n")
            f.write("\nno_bbox_images:\n")
            for p in no_bbox_files:
                f.write(f"  {p}\n")

        print(f"[OK] Wrote summary: {summary_path}")

    return 0

"""
0 para usar todas

python split_by_dir.py \
  --csv /home/ivan/Downloads/ss_kvasir_dataset_access/input/metadata.csv \
  --separator ';' \
  --output-dir /home/ivan/Downloads/ss_kvasir_dataset_access/output \
  --max-per-class 0 \
  --keep-first-duplicate \
  --drop-out-of-bounds



  


python split_by_dir.py \
  --csv .../metadata.csv \
  --separator ';' \
  --output-dir .../output \
  --max-per-class 0 \
  --keep-first-duplicate \
  --clamp-out-of-bounds
"""

if __name__ == "__main__":
    raise SystemExit(main())

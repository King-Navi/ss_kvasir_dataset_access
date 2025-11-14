#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import polars as pl

from readmeta import load_csv
from video import get_video_frame
from constants import COORD_COLS, OUTPUT, THICKNESS


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


def check_coords_against_frame(row, img_w: int, img_h: int):
    """
    Check whether each (xi, yi) lies within [0, img_w-1] x [0, img_h-1].

    Returns a list of tuples (i, x, y) for points that are out of bounds.
    """
    bad_points = []
    # COORD_COLS = ["x1","y1","x2","y2","x3","y3","x4","y4"]
    for i in range(1, 5):
        x = row[f"x{i}"]
        y = row[f"y{i}"]
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            bad_points.append((i, x, y))
    return bad_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quick preview: given a filename, extract its frame from video and "
            "draw bbox if coordinates exist. Warn if filename appears multiple times."
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
        "--filename",
        required=True,
        help="Filename to search in the metadata (exact match; basename is also tried).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT,
        help="Output directory for the preview image (default=constants.OUTPUT).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[INFO] Loading CSV: {csv_path}")
    df = load_csv(str(csv_path), args.separator)

    df_match = df.filter(pl.col("filename") == args.filename)

    # nothing, try basename
    if df_match.height == 0:
        fname_base = os.path.basename(args.filename)
        if fname_base != args.filename:
            print(f"[INFO] No exact match for '{args.filename}', trying basename '{fname_base}'")
            df_match = df.filter(pl.col("filename") == fname_base)

    if df_match.height == 0:
        print(f"[ERROR] No rows found for filename '{args.filename}'")
        return 1

    print(f"[INFO] Rows found for filename '{args.filename}': {df_match.height}")

    # Warn if duplicated
    if df_match.height > 1:
        print(f"[WARN] Filename appears {df_match.height} times in metadata!")
        os.makedirs(args.output_dir, exist_ok=True)
        dupes_path = os.path.join(
            args.output_dir,
            f"debug_{os.path.basename(args.filename)}_rows.csv",
        )
        df_match.write_csv(dupes_path)
        print(f"[WARN] Saved all matching rows to: {dupes_path}")

    # Use first matched row
    row = next(df_match.iter_rows(named=True))

    video_id = row["video_id"]
    frame_number = int(row["frame_number"])
    print(f"[INFO] Using row -> video_id={video_id}, frame_number={frame_number}")

    try:
        frame = get_video_frame(frame_number, video_id)
    except Exception as e:
        print(f"[ERROR] Could not extract frame {frame_number} from video {video_id}: {e}")
        return 1

    # video size
    img_h, img_w = frame.shape[:2]
    print(f"[INFO] Frame size (max coordinates): width={img_w}, height={img_h}")

    # check if any coordinate exceeds the frame size
    bad_points = check_coords_against_frame(row, img_w, img_h)
    if bad_points:
        print("[WARN] Some coordinates are outside frame bounds:")
        for i, x, y in bad_points:
            print(f"       - (x{i}, y{i}) = ({x}, {y}) is out of [0,{img_w-1}]x[0,{img_h-1}]")
    else:
        print("[INFO] All coordinates are within frame bounds.")

    x_min, y_min, x_max, y_max = compute_axis_aligned_bbox(row)

    # Clamp to frame bounds (like main script)
    x_min = max(0.0, min(x_min, img_w - 1.0))
    x_max = max(0.0, min(x_max, img_w - 1.0))
    y_min = max(0.0, min(y_min, img_h - 1.0))
    y_max = max(0.0, min(y_max, img_h - 1.0))

    pt1 = (int(round(x_min)), int(round(y_min)))
    pt2 = (int(round(x_max)), int(round(y_max)))
    cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)  # blue rectangle

    # draw red points
    for i in range(1, 5):
        x = row[f"x{i}"]
        y = row[f"y{i}"]
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(row["filename"]))[0]
    out_path = os.path.join(args.output_dir, f"{base}_preview_with_bbox.png")

    cv2.imwrite(out_path, frame)
    print(f"[OK] Saved preview image to: {out_path}")

    return 0

"""
python quick_preview_by_filename.py \
  --csv /home/ivan/Downloads/ss_kvasir_dataset_access/input/metadata.csv \
  --separator ';' \
  --filename d369e4f163df4aba_12064.jpg \
  --output-dir /home/ivan/Downloads/ss_kvasir_dataset_access/debug_previews

"""
if __name__ == "__main__":
    raise SystemExit(main())

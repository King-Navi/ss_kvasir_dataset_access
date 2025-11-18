#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import polars as pl

from readmeta import load_csv, build_class_mappings
from split_by_dir import (
    compute_axis_aligned_bbox,
    yolo_normalize_bbox,
    clamp_coords_in_row,
)
from video import get_video_frame
from constants import COORD_COLS, OUTPUT, THICKNESS


def complete_coords_expr() -> pl.Expr:
    """
    True si TODAS las columnas de coordenadas no son null ni NaN.
    """
    return pl.all_horizontal(
        [pl.col(c).is_not_null() & ~pl.col(c).is_nan() for c in COORD_COLS]
    )


def select_rows_for_split(metadata_df: pl.DataFrame, split_csv: str) -> pl.DataFrame:
    """
    Para un split (train/val):

    - Lee el CSV de split (usa solo la columna filename).
    - Filtra metadata a esas imágenes.
    - Si una imagen aparece varias veces en metadata, se queda
      con la fila que tenga coordenadas completas (si hay).
    """
    print(f"[INFO] Loading split file: {split_csv}")
    split_df = pl.read_csv(split_csv)

    # Limpieza mínima
    split_df = split_df.with_columns(
        pl.col("filename").cast(pl.Utf8).str.strip_chars()
    )

    filenames = split_df["filename"].unique()
    print(f"[INFO] Filenames in split ({split_csv}): {filenames.len()}")

    df_split = (
        metadata_df
        .filter(pl.col("filename").is_in(filenames))
        .with_columns(complete_coords_expr().alias("has_complete_coords"))
    )

    print(f"[INFO] Rows in metadata for this split: {df_split.height}")

    # primero las filas con coords completas, luego las demás
    df_split = df_split.with_columns(
        pl.when(pl.col("has_complete_coords").fill_null(False))
        .then(0)
        .otherwise(1)
        .alias("priority")
    )

    # Ordenamos por filename + prioridad y nos quedamos con la primera por filename
    df_split = df_split.sort(["filename", "priority"])
    df_selected = df_split.unique(subset=["filename"], keep="first")

    print(
        f"[INFO] Unique images after duplicate resolution: {df_selected.height}"
    )
    return df_selected


def ensure_dirs(root: str, split_name: str):
    images_dir = os.path.join(root, "images", split_name)
    labels_dir = os.path.join(root, "labels", split_name)
    bbox_dir = os.path.join(root, "bbox", split_name)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)

    return images_dir, labels_dir, bbox_dir


def process_split(
    df_split: pl.DataFrame,
    split_name: str,
    out_root: str,
    full_to_id: dict,
):
    """
    Recorre el split y genera:
      - images/<split>/filename.jpg   (imagen original)
      - labels/<split>/filename.txt   (YOLO: class_id cx cy w h, o vacío)
      - bbox/<split>/filename.jpg     (imagen con rectángulo, solo si hay bbox)
    """
    images_dir, labels_dir, bbox_dir = ensure_dirs(out_root, split_name)

    total = df_split.height
    print(f"[INFO] Processing {total} images for split '{split_name}'")

    for idx, row in enumerate(df_split.iter_rows(named=True), start=1):
        filename = row["filename"]
        video_id = row["video_id"]
        frame_number = int(row["frame_number"])

        has_coords = bool(row.get("has_complete_coords") or False)

        try:
            frame = get_video_frame(frame_number, video_id)
        except Exception as e:
            print(
                f"[WARN] Could not extract frame {frame_number} "
                f"from video {video_id} for '{filename}': {e}"
            )
            continue

        img_h, img_w = frame.shape[:2]

        img_out_path = os.path.join(images_dir, filename)
        cv2.imwrite(img_out_path, frame)

        name_root, _ext = os.path.splitext(filename)
        if not name_root:
            # caso raro de nombre sin base
            name_root = filename

        label_path = os.path.join(labels_dir, name_root + ".txt")

        if not has_coords:
            # No hay bbox -> label vacío
            open(label_path, "w", encoding="utf-8").close()
            continue

        # Tenemos bbox completo: clipear, normalizar y escribir label
        # Usamos una copia del dict para no alterar la fila original
        row_dict = dict(row)

        # Clampear coords a los límites de la imagen
        clamped = clamp_coords_in_row(row_dict, img_w, img_h)
        if clamped:
            print(
                f"[INFO] Clamped bbox for '{filename}' "
                f"to frame bounds [0,{img_w-1}]x[0,{img_h-1}]"
            )

        # Bounding box axis-aligned
        x_min, y_min, x_max, y_max = compute_axis_aligned_bbox(row_dict)

        # formato YOLO (cx, cy, w, h) en [0,1]
        x_center, y_center, bw, bh = yolo_normalize_bbox(
            x_min, y_min, x_max, y_max, img_w, img_h
        )

        finding_class = row["finding_class"]
        class_id = full_to_id.get(finding_class)

        if class_id is None:
            print(
                f"[WARN] Unknown class '{finding_class}' for '{filename}'. "
                f"Writing empty label."
            )
            open(label_path, "w", encoding="utf-8").close()
            continue

        # label YOLO
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(
                f"{class_id} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{bw:.6f} {bh:.6f}\n"
            )

        #guardar imagen con bbox (solo si hay coords)
        frame_bbox = frame.copy()
        pt1 = (int(round(x_min)), int(round(y_min)))
        pt2 = (int(round(x_max)), int(round(y_max)))
        cv2.rectangle(frame_bbox, pt1, pt2, (0, 255, 0), THICKNESS)

        bbox_out_path = os.path.join(bbox_dir, filename)
        cv2.imwrite(bbox_out_path, frame_bbox)

        if idx % 500 == 0:
            print(f"[INFO]   {idx}/{total} images processed in '{split_name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build YOLO-style dataset (images/labels/bbox with "
            "train/val) from metadata.csv + split_0.csv + split_1.csv"
        )
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata.csv (semicolon-separated).",
    )
    parser.add_argument(
        "--split-train",
        required=True,
        help="Path to split_0.csv (train filenames).",
    )
    parser.add_argument(
        "--split-val",
        required=True,
        help="Path to split_1.csv (val filenames).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT,
        help=(
            "Root output directory for images/labels/bbox. "
            "Default is constants.OUTPUT."
        ),
    )
    parser.add_argument(
        "--metadata-separator",
        default=";",
        help="Separator used in metadata.csv (default=';').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    print(f"[INFO] Loading metadata: {meta_path}")
    metadata_df = load_csv(str(meta_path), args.metadata_separator)
    metadata_df = metadata_df.with_columns(
        pl.col("filename").cast(pl.Utf8).str.strip_chars()
    )

    print(
        f"[INFO] Metadata rows: {metadata_df.height}, "
        f"columns: {metadata_df.width}"
    )

    full_to_key, full_to_id = build_class_mappings()
    print(f"[INFO] Number of classes in mapping: {len(full_to_id)}")

    train_df = select_rows_for_split(metadata_df, args.split_train)
    val_df = select_rows_for_split(metadata_df, args.split_val)

    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)
    print(f"[INFO] Output root directory: {out_root}")

    # 4) Procesar train y val
    process_split(train_df, "train", out_root, full_to_id)
    process_split(val_df, "val", out_root, full_to_id)

    print("[OK] Dataset build finished.")
    return 0



"""
python build_yolo_splits.py \
  --metadata /home/ivan/Downloads/ss_kvasir_dataset_access/input/metadata.csv \
  --split-train /home/ivan/Downloads/ss_kvasir_dataset_access/input/split_0.csv \
  --split-val   /home/ivan/Downloads/ss_kvasir_dataset_access/input/split_1.csv \
  --output-dir  /home/ivan/Downloads/ss_kvasir_dataset_access/output

"""

if __name__ == "__main__":
    raise SystemExit(main())

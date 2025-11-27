#!/usr/bin/env python
"""
Fix duplicated filenames in a split CSV and replicate image/label files.

Dataset layout (simplified)
---------------------------

You have three root folders (each with train/ and val/ subfolders):

    bbox/
    images/
    labels/

For now we only care about one subset (usually "train").

You also have:

- A split CSV, e.g. `split_1.csv`, with columns:

      filename,label

- A metadata CSV, e.g. `metadata.csv`, with columns (semicolon separated):

      filename;video_id;frame_number;finding_category;finding_class;x1;y1;x2;y2;x3;y3;x4;y4

  The rows where ANY of x1..y4 is non-empty correspond to images that
  actually have a bbox.

For every duplicated `filename` in the split CSV:

1) Look in the metadata CSV for that filename.

2) Among the metadata rows for that filename, find the rows that:
     - have at least one bbox coordinate (x1..y4 not all empty), and
     - have some `finding_class` value.

3) If the duplicate split rows have a `label` that matches one of the
   `finding_class` values that have bbox, THEN:

   - The *matching* split row is considered the "canonical" row:
     it keeps the original filename (this is the one that must keep the bbox).
   - All other split rows for that filename are treated as duplicates
     and get renamed with suffixes: '_1', '_2', ...

4) If NO label for that filename matches a bbox `finding_class` in
   metadata, we fall back to the previous behaviour: the first row keeps
   the original name and the rest get '_1', '_2', etc.

5) For each renamed duplicate row we:
   - Copy the image from `images/<subset>/<original>` to
     `images/<subset>/<new_with_suffix>`.
   - Create a label file `labels/<subset>/<new_stem>.txt` containing
     the `label` value from the split CSV.

6) The bbox/ folder is **never** modified. As a result, only the
   canonical filename (chosen via metadata) keeps its bbox, exactly as
   you requested.

A new split CSV with the updated filenames is also written.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import shutil


def load_split(csv_path: Path):
    """Load the split CSV and return (rows, fieldnames).

    Each row is a dict. We keep all columns, but the script only needs
    'filename' and 'label'.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "filename" not in fieldnames:
            raise ValueError("CSV must contain at least a 'filename' column.")
        for row in reader:
            rows.append(row)
    return rows, fieldnames


def find_duplicate_filenames(rows):
    """Return (duplicate_names, counts) for the 'filename' column."""
    filenames = [r["filename"] for r in rows]
    counts = Counter(filenames)
    dup_names = {name for name, c in counts.items() if c > 1}
    return dup_names, counts


def load_bbox_classes_by_filename(metadata_csv: Path):
    """
    From the metadata CSV, build a mapping:

        filename -> set of finding_class values that have a bbox.

    We consider that a row "has bbox" if at least one coordinate column
    x1..y4 is non-empty.
    """
    bbox_classes_by_filename: dict[str, set[str]] = defaultdict(set)

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        expected_cols = {"filename", "finding_class",
                         "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"}
        missing = expected_cols.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Metadata CSV is missing expected columns: {sorted(missing)}"
            )

        for row in reader:
            has_bbox = False
            for col in ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]:
                v = row.get(col)
                if v and str(v).strip():
                    has_bbox = True
                    break
            if not has_bbox:
                continue

            filename = (row.get("filename") or "").strip()
            finding_class = (row.get("finding_class") or "").strip()
            if filename and finding_class:
                bbox_classes_by_filename[filename].add(finding_class)

    return bbox_classes_by_filename


def build_new_split_and_ops(rows, dup_names, bbox_classes_by_filename):
    """Rename duplicates and build the file operations plan.

    Strategy for each filename:

    - If not duplicated: keep all rows as they are.
    - If duplicated:
        * If some labels match bbox finding_class from metadata:
              one matching row keeps original filename,
              all others get '_1', '_2', ...
        * Else:
              we fall back to: first row keeps original name,
              all others get '_1', '_2', ...
    """
    rows_by_fname: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        rows_by_fname[row["filename"]].append((idx, row))

    new_rows: list[dict] = [None] * len(rows)
    ops: list[dict[str, str]] = []

    for fname, entries in rows_by_fname.items():
        if fname not in dup_names:
            for idx, row in entries:
                new_rows[idx] = dict(row)
            continue

        bbox_classes = {
            cls.strip().lower()
            for cls in bbox_classes_by_filename.get(fname, set())
            if cls
        }

        primary_entry_index = None

        if bbox_classes:
            for i, (idx, row) in enumerate(entries):
                label_norm = (row.get("label") or "").strip().lower()
                if label_norm in bbox_classes:
                    primary_entry_index = i
                    break

        if primary_entry_index is None:
            primary_entry_index = 0

        suffix = 1
        for i, (idx, row) in enumerate(entries):
            if i == primary_entry_index:
                new_fname = fname
                dup_index = 0
            else:
                stem, ext = os.path.splitext(fname)
                new_fname = f"{stem}_{suffix}{ext}"
                dup_index = suffix
                suffix += 1

                ops.append(
                    {
                        "original": fname,
                        "new": new_fname,
                        "label": row.get("label"),
                        "dup_index": dup_index,
                    }
                )

            new_row = dict(row)
            new_row["filename"] = new_fname
            new_rows[idx] = new_row

    assert all(r is not None for r in new_rows), "Internal error: some rows not filled."

    return new_rows, ops


def check_no_duplicate_filenames(rows):
    """Return a list of filenames that are still duplicated (should be empty)."""
    counts = Counter(r["filename"] for r in rows)
    return [name for name, c in counts.items() if c > 1]


def apply_file_ops(
    ops,
    images_root: Path,
    labels_root: Path,
    subset: str = "train",
    dry_run: bool = False,
):
    """Copy images and create label .txt files for renamed duplicates.

    Only images/<subset> and labels/<subset> are touched. The bbox folder
    is NOT modified, so only the canonical filename keeps its bbox image.
    """
    images_dir = images_root / subset
    labels_dir = labels_root / subset

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for op in ops:
        orig = op["original"]
        new = op["new"]
        label_value = op["label"]

        orig_img = images_dir / orig
        new_img = images_dir / new

        if not orig_img.exists():
            print(f"[WARN] Source image does not exist, skipping copy: {orig_img}")
        else:
            if new_img.exists():
                print(f"[SKIP] Target image already exists, skipping copy: {new_img}")
            else:
                print(f"[COPY] {orig_img} -> {new_img}")
                if not dry_run:
                    shutil.copy2(orig_img, new_img)

        new_stem = Path(new).stem
        new_label_path = labels_dir / f"{new_stem}.txt"
        if new_label_path.exists():
            print(f"[SKIP] Target label already exists, skipping write: {new_label_path}")
        else:
            print(f"[WRITE] Label for {new}: {label_value!r} -> {new_label_path}")
            if not dry_run:
                with open(new_label_path, "w", encoding="utf-8") as f:
                    if label_value is None:
                        f.write("\n")
                    else:
                        f.write(str(label_value).strip() + "\n")


def write_split(rows, fieldnames, output_csv: Path):
    """Write the new split CSV with updated filenames."""
    if "filename" not in fieldnames:
        fieldnames = ["filename"] + [fn for fn in fieldnames if fn != "filename"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(filtered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix duplicated filenames in a split CSV and duplicate image/label\n"
            "files on disk so that each row has its own image + label.\n\n"
            "The canonical occurrence (the one that keeps the original\n"
            "filename and bbox) is chosen based on the metadata CSV: we keep\n"
            "the row whose label matches a finding_class with bbox.\n"
        )
    )
    parser.add_argument(
        "--split-csv",
        dest="split_csv",
        type=Path,
        required=True,
        help="Path to the split CSV (e.g. split_1.csv).",
    )
    parser.add_argument(
        "--metadata-csv",
        dest="metadata_csv",
        type=Path,
        required=True,
        help="Path to the metadata CSV that contains bbox info.",
    )
    parser.add_argument(
        "--images-root",
        dest="images_root",
        type=Path,
        required=True,
        help="Base folder that contains 'train'/'val' subfolders with images.",
    )
    parser.add_argument(
        "--labels-root",
        dest="labels_root",
        type=Path,
        required=True,
        help="Base folder that contains 'train'/'val' subfolders with label .txt files.",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "val"],
        default="train",
        help="Which subfolder to use under images/labels roots (default: train).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional path for the fixed CSV. If omitted, a new file next to the "
            "input is created with suffix '.fixed.csv'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not touch the filesystem; only print what would be done.",
    )

    args = parser.parse_args(argv)

    split_csv: Path = args.split_csv
    metadata_csv: Path = args.metadata_csv
    images_root: Path = args.images_root
    labels_root: Path = args.labels_root
    subset: str = args.subset
    dry_run: bool = args.dry_run

    if not split_csv.is_file():
        parser.error(f"Split CSV does not exist: {split_csv}")
    if not metadata_csv.is_file():
        parser.error(f"Metadata CSV does not exist: {metadata_csv}")
    if not images_root.is_dir():
        parser.error(f"Images root does not exist: {images_root}")
    if not labels_root.is_dir():
        parser.error(f"Labels root does not exist: {labels_root}")

    rows, fieldnames = load_split(split_csv)
    print(f"Total rows in split: {len(rows)}")

    dup_names, counts = find_duplicate_filenames(rows)
    print(f"Found {len(dup_names)} duplicated filenames.")
    if not dup_names:
        print("Nothing to fix: there are no duplicated filenames.")
        return 0

    print("Duplicated filenames (name -> count):")
    for name in sorted(dup_names):
        print(f"  {name}: {counts[name]}")

    print("\nLoading bbox information from metadata...")
    bbox_classes_by_filename = load_bbox_classes_by_filename(metadata_csv)
    print(f"  Loaded bbox info for {len(bbox_classes_by_filename)} filenames.")

    new_rows, ops = build_new_split_and_ops(rows, dup_names, bbox_classes_by_filename)
    print(f"\nPlanned operations for duplicates in subset '{subset}': {len(ops)}")
    for op in ops:
        print(
            f"  dup_index={op['dup_index']} original={op['original']} "
            f"-> new={op['new']} label={op['label']}"
        )

    dup_after = check_no_duplicate_filenames(new_rows)
    if dup_after:
        print("\n[ERROR] After renaming there are still duplicated filenames:")
        for name in dup_after:
            print(f"  {name}")
        print(
            "CSV will NOT be written and no files will be copied. "
            "Please inspect your split manually."
        )
        return 1

    if args.output_csv is None:
        # e.g. split_1.csv -> split_1.fixed.csv
        output_csv = split_csv.with_name(split_csv.stem + ".fixed.csv")
    else:
        output_csv = args.output_csv

    print(f"\nWriting fixed CSV to: {output_csv}")
    if not dry_run:
        write_split(new_rows, fieldnames, output_csv)
    else:
        print("  [dry-run] Skipped writing CSV.")

    print(
        f"\nApplying filesystem operations under subset '{subset}' "
        f"in images_root={images_root} labels_root={labels_root}"
    )
    apply_file_ops(
        ops,
        images_root=images_root,
        labels_root=labels_root,
        subset=subset,
        dry_run=dry_run,
    )

    print("\nDone.")
    if dry_run:
        print("(No files were modified because --dry-run was active.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Usage (change for the real dirs & file):

python fix_it.py \
  --split-csv /ruta/a/split_1.csv \
  --metadata-csv /ruta/a/metadata.csv \
  --images-root /ruta/a/images \
  --labels-root /ruta/a/labels \
  --subset val \
  --dry-run




python fix/fix_it.py \
  --split-csv /home/ivan/Downloads/ss_kvasir_dataset_access/input/split_1.csv \
  --metadata-csv /home/ivan/Downloads/ss_kvasir_dataset_access/input/metadata.csv \
  --images-root /home/ivan/Downloads/borrar_Dirs/images \
  --labels-root /home/ivan/Downloads/borrar_Dirs/labels \
  --subset val 

"""
if __name__ == "__main__":
    raise SystemExit(main())

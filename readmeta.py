#!/usr/bin/env python3
import argparse
from pathlib import Path
import polars as pl

COORD_COLS = [
    "x1", "y1",
    "x2", "y2",
    "x3", "y3",
    "x4", "y4",
]

FINDING_CLASS_MAP = {
    "ampulla": "Ampulla of Vater",
    "angiectasia": "Angiectasia",
    "blood_fresh": "Blood - fresh",
    "blood_hematin": "Blood - hematin",
    "erosion": "Erosion",
    "erythema": "Erythema",
    "foreign_body": "Foreign Body",
    "ileocecal_valve": "Ileocecal valve",
    "lymphangiectasia": "Lymphangiectasia",
    "normal_mucosa": "Normal clean mucosa",
    "polyp": "Polyp",
    "pylorus": "Pylorus",
    "reduced_view": "Reduced Mucosal View",
    "ulcer": "Ulcer",
}

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lee un CSV de metadata/anotaciones con Polars y (opcional) filtra filas sin coordenadas."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Ruta al CSV, por ejemplo: ./input/metadata.csv",
    )
    parser.add_argument(
        "--separator",
        default=";",
        help="Separador del CSV (default=';').",
    )
    parser.add_argument(
        "--drop-missing-coords",
        action="store_true",
        help="Si se pasa, elimina las filas que tengan alguna coord (x1..y4) vacía/null.",
    )
    return parser.parse_args()


def load_csv(path: str, separator: str) -> pl.DataFrame:
    #encabezados
    dtypes = {
        "filename": pl.Utf8,
        "video_id": pl.Utf8,
        "frame_number": pl.Int64,
        "finding_category": pl.Utf8,
        "finding_class": pl.Utf8,
        "x1": pl.Float64,
        "y1": pl.Float64,
        "x2": pl.Float64,
        "y2": pl.Float64,
        "x3": pl.Float64,
        "y3": pl.Float64,
        "x4": pl.Float64,
        "y4": pl.Float64,
    }

    df = pl.read_csv(
        path,
        separator=separator,
        has_header=True,
        dtypes=dtypes,
        null_values=["", "NA", "NaN"],
        ignore_errors=True,
    )
    return df


def filter_missing_coords(df: pl.DataFrame) -> pl.DataFrame:
    # SOLO con las filas donde TODAS las coords no son nulas
    cond = None
    for col in COORD_COLS:
        c = pl.col(col).is_not_null()
        cond = c if cond is None else (cond & c)

    return df.filter(cond)


def filter_complete_coords(df: pl.DataFrame) -> pl.DataFrame:
    # SOLO con filas donde todas las coords:
    # - no sean null
    # - no sean NaN
    cond = pl.all_horizontal([
        pl.col(c).is_not_null() & ~pl.col(c).is_nan()
        for c in COORD_COLS
    ])
    return df.filter(cond)

def filter_finding_class(df: pl.DataFrame, keys) -> pl.DataFrame:
    values = [FINDING_CLASS_MAP[k] for k in keys]
    return df.filter(pl.col("finding_class").is_in(values))

def rows_with_all_coords_missing(df: pl.DataFrame) -> pl.DataFrame:
    cond = pl.all_horizontal([
        pl.col(c).is_null() | pl.col(c).is_nan()
        for c in COORD_COLS
    ])
    return df.filter(cond)

def main() -> int:
    args = parse_args()

    csv_path = Path(args.path)
    print(f"[INFO] leyendo archivo: {csv_path}")

    if not csv_path.exists():
        print(f"[ERROR] no existe el archivo: {csv_path}")
        return 1

    # leer
    df = load_csv(str(csv_path), args.separator)
    print(f"[INFO] filas leídas: {df.height}, columnas: {df.width}")

    # filtrar si lo pidió
    if args.drop_missing_coords:
        df_filtered = filter_missing_coords(df)
        print(f"[INFO] filas después de filtrar coords: {df_filtered.height}")

        df_polyp_ulcer = filter_finding_class(df_filtered, ["polyp", "ulcer"])

        print(f"[INFO] filas después de filtrar coords (clases pedidas): {df_polyp_ulcer.height}")
        # muestra las primeras
        print(df_polyp_ulcer.head(10))
        print(df_polyp_ulcer.head(1).select(["video_id","frame_number"]))
    else:
        print("[INFO] no se aplicó filtro de coordenadas (--drop-missing-coords)")
        print(df.head(10))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python


"""
Search for duplicates & missing rows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

SPLIT_TO_SUBDIR = {
    "split_0": "train",
    "split_1": "val",
}


def load_splits(splits_dir: Path, pattern: str = "split_*.csv") -> pl.DataFrame:
    """Carga todos los csv de splits y añade una columna 'split' con el nombre del csv.

    Columnas esperadas en los csv: 'filename', 'label'
    """
    csv_paths = sorted(splits_dir.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron CSV con patrón {pattern!r} en {splits_dir}")

    dfs: list[pl.DataFrame] = []
    for csv_path in csv_paths:
        split_name = csv_path.stem
        df = pl.read_csv(csv_path)
        if "filename" not in df.columns:
            raise ValueError(f"El archivo {csv_path} no tiene la columna 'filename'")
        df = df.with_columns(pl.lit(split_name).alias("split"))
        dfs.append(df)

    all_df = pl.concat(dfs, how="vertical")
    return all_df


def find_duplicates(all_df: pl.DataFrame) -> pl.DataFrame:
    """Encuentra filenames repetidos entre todos los splits."""
    dup = (
        all_df
        .group_by("filename")
        .agg(
            pl.count().alias("n_rows"),
            pl.col("split").unique().sort().alias("splits"),
            pl.col("label").unique().sort().alias("labels"),
        )
        .filter(pl.col("n_rows") > 1)
        .sort("n_rows", descending=True)
    )
    return dup


def list_files_on_disk(base_dir: Path) -> pl.DataFrame:
    """Lista archivos reales en disco bajo cada subcarpeta de SPLIT_TO_SUBDIR.

    Devuelve columnas: ['split', 'subdir', 'filename', 'filepath']
    """
    rows: list[dict[str, str]] = []

    for split_name, subdir in SPLIT_TO_SUBDIR.items():
        folder = base_dir / subdir
        if not folder.is_dir():
            print(f"[AVISO] La carpeta {folder} no existe, se ignora.")
            continue

        for path in folder.rglob("*"):
            if path.is_file():
                rows.append(
                    {
                        "split": split_name,
                        "subdir": subdir,
                        "filename": path.name,
                        "filepath": str(path),
                    }
                )

    if not rows:
        print("[AVISO] No se encontraron archivos en las subcarpetas configuradas.")
        return pl.DataFrame({"split": [], "subdir": [], "filename": [], "filepath": []})

    return pl.DataFrame(rows)


def find_missing_files(all_df: pl.DataFrame, files_df: pl.DataFrame, dup_df: pl.DataFrame) -> pl.DataFrame:
    """Marca qué filas del csv no tienen archivo en disco y si el filename es duplicado.

    all_df: filas de los splits (todas las filas de los csv)
    files_df: archivos listados en disco
    dup_df: dataframe con filenames duplicados (columna 'filename')
    """
    if files_df.is_empty():
        missing = (
            all_df
            .with_columns(
                pl.lit(None).alias("filepath"),
                pl.lit(True).alias("is_missing"),
            )
        )
    else:
        all_with_path = all_df.join(
            files_df.select(["split", "filename", "filepath"]),
            on=["split", "filename"],
            how="left",
        ).with_columns(
            pl.col("filepath").is_null().alias("is_missing")
        )

        missing = all_with_path.filter(pl.col("is_missing"))

    if not dup_df.is_empty():
        dup_flag = dup_df.select(
            pl.col("filename"),
            pl.lit(True).alias("is_duplicated"),
        )

        missing = missing.join(
            dup_flag,
            on="filename",
            how="left",
        ).with_columns(
            pl.col("is_duplicated").fill_null(False)
        )
    else:
        missing = missing.with_columns(pl.lit(False).alias("is_duplicated"))

    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revisa duplicados de filename en los splits y verifica que existan "
            "los archivos en disco bajo las carpetas train/val."
        )
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("."),
        help="Carpeta donde están los split_0.csv, split_1.csv, etc. (por defecto: cwd)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Carpeta base que contiene las subcarpetas train y val.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=None,
        help="Opcional: prefijo para guardar CSVs de salida (duplicados, faltantes).",
    )

    args = parser.parse_args()

    all_df = load_splits(args.splits_dir)

    print("Total de filas en todos los splits:", all_df.height)

    dup_df = find_duplicates(all_df)
    print("\n=== Filenames duplicados ===")
    print("Total de filenames duplicados:", dup_df.height)
    if dup_df.height > 0:
        print(dup_df.head(20))

    files_df = list_files_on_disk(args.data_dir)
    print("\nTotal de archivos encontrados en disco:", files_df.height)

    missing_df = find_missing_files(all_df, files_df, dup_df)
    print("\n=== Archivos faltantes según los splits ===")
    print("Total de filas sin archivo en disco:", missing_df.height)

    if missing_df.height > 0:
        print(missing_df.select(["split", "filename", "label", "is_duplicated"]).head(30))

    if args.save_prefix is not None:
        base = args.save_prefix
        dup_path = base.with_suffix(".duplicados.csv")
        miss_path = base.with_suffix(".faltantes.csv")

        dup_df_out = dup_df.with_columns(
            pl.col("splits").list.join("|").alias("splits"),
            pl.col("labels").list.join("|").alias("labels"),
        )

        dup_df_out.write_csv(dup_path)
        missing_df.write_csv(miss_path)

        print(f"\nResultados guardados en:")
        print(f"  - Duplicados : {dup_path}")
        print(f"  - Faltantes  : {miss_path}")



"""
python validate_split/valitate_dup_mis.py \
    --data-dir /home/ivan/Downloads/borrar_Dirs/images \
    --splits-dir /home/ivan/Downloads/ss_kvasir_dataset_access/input \
    --save-prefix resultados


"""

if __name__ == "__main__":
    main()

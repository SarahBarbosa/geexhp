"""
Merge all parquet files under parallel/data* into a single parquet file.

Strategy (memory-safe):
  Pass 1 — read only ATMOSPHERE-LAYERS-MOLECULES to collect all unique molecules
            across ALL files and earth types (archean has C2H6 but not O2/O3;
            modern/proterozoic have O2/O3/N2O). Missing molecules are filled with 0.
  Pass 2 — process one file at a time: add Earth_type column, expand molecule
            abundances via fast list-comprehension, write to output via
            ParquetWriter (incremental append).

Never loads more than one source file into memory at a time.

Usage:
    python merge_parquet.py [--output PATH] [--compression snappy|zstd|none]
"""

import argparse
import glob
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _detect_earth_type(filepath: str) -> str:
    fname = os.path.basename(filepath)
    if fname.startswith("modern"):
        return "modern"
    if fname.startswith("proterozoic"):
        return "proterozoic"
    if fname.startswith("archean"):
        return "archean"
    return "unknown"


def fast_extract_abundances(df: pd.DataFrame, sorted_molecules: list) -> pd.DataFrame:
    """
    Vectorized abundance extraction — ~20x faster than row-wise apply.
    Pre-splits both columns as Series of lists, then builds records via
    list comprehension (avoids per-row pd.Series overhead).

    Missing molecules (e.g. O2/O3 absent in archean, C2H6 absent in modern)
    are filled with 0.0 so all rows share the same schema.
    """
    mol_series = df["ATMOSPHERE-LAYERS-MOLECULES"].str.split(",")
    # ATMOSPHERE-LAYER-1 format: T, P, abun1, abun2, ...  → skip first 2
    layer_series = df["ATMOSPHERE-LAYER-1"].str.split(",").apply(lambda x: x[2:])

    records = [
        {mol: float(abun) for mol, abun in zip(mols, abuns)}
        for mols, abuns in zip(mol_series, layer_series)
    ]
    abun_df = pd.DataFrame(records, columns=sorted_molecules, index=df.index).fillna(0.0)
    return pd.concat([df, abun_df], axis=1)


def collect_all_files(parallel_dir: str) -> list:
    pattern = os.path.join(parallel_dir, "data*", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No parquet files found under {parallel_dir}/data*/")
    return files


def pass1_collect_molecules(files: list) -> list:
    """Read only ATMOSPHERE-LAYERS-MOLECULES from every file (cheap)."""
    molecules = set()
    total = len(files)
    for i, f in enumerate(files, 1):
        if i % 100 == 0 or i == total:
            print(f"  Pass 1: {i}/{total}", flush=True)
        df = pd.read_parquet(f, columns=["ATMOSPHERE-LAYERS-MOLECULES"], engine="pyarrow")
        df["ATMOSPHERE-LAYERS-MOLECULES"].apply(lambda x: molecules.update(x.split(",")))
    return sorted(molecules)


def pass2_write_output(files: list, sorted_molecules: list, output_path: str, compression: str):
    """Process one file at a time and write incrementally to output parquet."""
    writer = None
    total = len(files)

    try:
        for i, f in enumerate(files, 1):
            if i % 50 == 0 or i == 1 or i == total:
                print(f"  Pass 2: {i}/{total}  ({os.path.basename(f)})", flush=True)

            earth_type = _detect_earth_type(f)
            df = pd.read_parquet(f, engine="pyarrow")
            df.insert(0, "Earth_type", earth_type)
            df = fast_extract_abundances(df, sorted_molecules)

            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)

            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "combined_parallel.parquet"),
        help="Output parquet file path (default: data/combined_parallel.parquet)",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "zstd", "gzip", "none"],
        help="Parquet compression codec (default: snappy)",
    )
    args = parser.parse_args()

    parallel_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(args.output)
    compression = None if args.compression == "none" else args.compression

    print(f"Parallel dir : {parallel_dir}")
    print(f"Output       : {output_path}")
    print(f"Compression  : {args.compression}")

    files = collect_all_files(parallel_dir)
    print(f"\nFound {len(files)} parquet files.\n")

    print("Pass 1 — collecting molecule names ...")
    sorted_molecules = pass1_collect_molecules(files)
    print(f"  Unique molecules ({len(sorted_molecules)}): {sorted_molecules}\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Pass 2 — merging files ...")
    pass2_write_output(files, sorted_molecules, output_path, compression)

    size_gb = os.path.getsize(output_path) / 1e9
    print(f"\nDone. Output file: {output_path}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()

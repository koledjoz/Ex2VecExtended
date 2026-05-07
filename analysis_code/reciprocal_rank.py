#!/usr/bin/env python3
"""
Create per-example MRR contribution values from prediction files.

Input prediction files must contain:
    userId, trackId, ts, pred_1, pred_2, ..., pred_N

For each row:
    - find the rank of trackId inside pred_1..pred_N
    - if found at rank r, output mrr = 1 / r
    - if not found, output mrr = 0

Outputs:
    For each model:
        <model>_example_mrr.csv

    Each file has two columns:
        example_index,mrr

Example:
    python create_example_mrr_values.py \
      --predictions \
        random=./predictions/baselines/random_output.csv \
        bl_knn=./predictions/baselines/bl_knn_output.csv \
        extended_mlp=./predictions/extendedMLP/extendedmlp_output.csv \
      --output-dir ./predictions/example_mrr \
      --max-k 50
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRED_COL_RE = re.compile(r"^pred_(\d+)$")


def parse_prediction_arg(arg: str) -> Tuple[str, Path]:
    """Parse either 'model_name=path.csv' or just 'path.csv'."""
    if "=" in arg:
        name, path = arg.split("=", 1)
        name = name.strip()
        path = Path(path.strip())

        if not name:
            raise ValueError(f"Empty model name in prediction argument: {arg!r}")

        return name, path

    path = Path(arg)
    stem = path.stem

    for suffix in ("_output", "_predictions", "_preds"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem, path


def safe_filename(name: str) -> str:
    """Make model name safe for a CSV filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def read_header(path: Path) -> List[str]:
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)

    if path.suffix.lower() in {".parquet", ".pq"}:
        return list(pd.read_parquet(path).columns)

    raise ValueError(f"Unsupported prediction file type: {path}. Use CSV or Parquet.")


def get_prediction_columns(columns: Sequence[str], max_k: Optional[int] = None) -> List[str]:
    pred_pairs = []

    for col in columns:
        m = PRED_COL_RE.match(col)
        if m:
            rank = int(m.group(1))
            if max_k is None or rank <= max_k:
                pred_pairs.append((rank, col))

    pred_pairs.sort(key=lambda x: x[0])

    if not pred_pairs:
        raise ValueError("No prediction columns found. Expected columns named pred_1, pred_2, ...")

    observed = [rank for rank, _ in pred_pairs]
    expected = list(range(1, pred_pairs[-1][0] + 1))

    if observed != expected[: len(observed)]:
        raise ValueError(
            "Prediction columns must be contiguous from pred_1. "
            f"Observed ranks start as: {observed[:20]}"
        )

    return [col for _, col in pred_pairs]


def iter_prediction_chunks(
    path: Path,
    usecols: Sequence[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        yield from pd.read_csv(path, usecols=list(usecols), chunksize=chunksize)

    elif suffix in {".parquet", ".pq"}:
        # Same simple behavior as your original script.
        # For very large parquet files, split externally or convert to CSV.
        yield pd.read_parquet(path, columns=list(usecols))

    else:
        raise ValueError(f"Unsupported prediction file type: {path}. Use CSV or Parquet.")


def create_example_mrr_file(
    model_name: str,
    path: Path,
    output_dir: Path,
    target_col: str,
    max_k: Optional[int],
    chunksize: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    columns = read_header(path)

    if target_col not in columns:
        raise ValueError(f"Target column {target_col!r} not found in {path}")

    pred_cols = get_prediction_columns(columns, max_k=max_k)
    usecols = [target_col, *pred_cols]

    output_path = output_dir / f"{safe_filename(model_name)}_example_mrr.csv"

    if output_path.exists():
        output_path.unlink()

    example_offset = 0
    total_mrr_sum = 0.0
    total_rows = 0
    header_written = False

    for chunk in iter_prediction_chunks(path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue

        targets = chunk[target_col].to_numpy()
        preds = chunk[pred_cols].to_numpy()

        matches = preds == targets[:, None]
        found = matches.any(axis=1)

        reciprocal_ranks = np.zeros(len(chunk), dtype=np.float64)

        if found.any():
            ranks = matches[found].argmax(axis=1).astype(np.float64) + 1.0
            reciprocal_ranks[found] = 1.0 / ranks

        example_indices = np.arange(
            example_offset,
            example_offset + len(chunk),
            dtype=np.int64,
        )

        out = pd.DataFrame(
            {
                "example_index": example_indices,
                "mrr": reciprocal_ranks,
            }
        )

        out.to_csv(
            output_path,
            mode="a",
            header=not header_written,
            index=False,
        )

        header_written = True
        example_offset += len(chunk)

        total_mrr_sum += float(reciprocal_ranks.sum())
        total_rows += len(chunk)

    if total_rows == 0:
        raise ValueError(f"No rows found in {path}")

    model_mrr = total_mrr_sum / total_rows

    print(f"Wrote {output_path}")
    print(f"Model: {model_name}")
    print(f"Rows: {total_rows}")
    print(f"MRR:  {model_mrr:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a per-example CSV with example_index and MRR contribution "
            "from prediction CSV/Parquet files with pred_1, pred_2, ... columns."
        )
    )

    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction files. Use either path.csv or model_name=path.csv.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output CSV files.",
    )

    parser.add_argument(
        "--target-col",
        default="trackId",
        help="Column containing the true item ID.",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Use only predictions up to this k. Defaults to all available pred_* columns.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV rows per chunk. Ignored for parquet input.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_specs = [parse_prediction_arg(arg) for arg in args.predictions]

    for model_name, path in prediction_specs:
        print(f"Processing {model_name}: {path}")

        create_example_mrr_file(
            model_name=model_name,
            path=path,
            output_dir=output_dir,
            target_col=args.target_col,
            max_k=args.max_k,
            chunksize=args.chunksize,
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Evaluate the average minimum k needed to recommend the correct item.

Input prediction files must contain one target item per row and prediction columns:
    userId, trackId, ts, pred_1, pred_2, ..., pred_N

For each row, the script finds the 1-indexed rank of trackId inside pred_1..pred_N.
If the target is not present, the row is right-censored: its true rank is > N.

Outputs:
    rank_metrics_tidy.csv   - one row per model with summary rank metrics
    recall_curve_tidy.csv   - Recall@k curve derived from the same ranks
    row_ranks.csv           - optional, one row per prediction row with found rank
    plots/*.png             - optional plots

Example:
    python evaluate_average_required_k.py \
      --predictions \
        random=./predictions/baselines/random_output.csv \
        bl_knn=./predictions/baselines/bl_knn_output.csv \
        extended_mlp=./predictions/extendedMLP/extendedmlp_output.csv \
      --output-dir ./predictions/rank_eval \
      --max-k 50 \
      --save-row-ranks \
      --plot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

    expected = list(range(1, pred_pairs[-1][0] + 1))
    observed = [rank for rank, _ in pred_pairs]
    if observed != expected[: len(observed)]:
        raise ValueError(
            "Prediction columns must be contiguous from pred_1. "
            f"Observed ranks start as: {observed[:20]}"
        )

    return [col for _, col in pred_pairs]


def iter_prediction_chunks(path: Path, usecols: Sequence[str], chunksize: int) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, usecols=list(usecols), chunksize=chunksize)
    elif suffix in {".parquet", ".pq"}:
        # Parquet chunking is intentionally kept simple. If this is too large,
        # convert predictions to CSV or split the parquet externally.
        yield pd.read_parquet(path, columns=list(usecols))
    else:
        raise ValueError(f"Unsupported prediction file type: {path}. Use CSV or Parquet.")


def rank_quantile_from_counts(rank_counts: np.ndarray, q: float) -> float:
    """Quantile over found ranks using integer rank counts."""
    total = int(rank_counts.sum())
    if total == 0:
        return float("nan")
    threshold = q * (total - 1) + 1  # nearest-rank style threshold on 1-indexed count
    cumulative = np.cumsum(rank_counts)
    return float(np.searchsorted(cumulative, threshold, side="left") + 1)


def evaluate_one_file(
    model_name: str,
    path: Path,
    output_dir: Path,
    target_col: str,
    id_cols: Sequence[str],
    max_k: Optional[int],
    chunksize: int,
    save_row_ranks: bool,
    row_ranks_path: Path,
    row_ranks_header_written: bool,
) -> Tuple[Dict[str, object], pd.DataFrame, bool]:
    if not path.exists():
        raise FileNotFoundError(path)

    columns = read_header(path)
    if target_col not in columns:
        raise ValueError(f"Target column {target_col!r} not found in {path}")

    pred_cols = get_prediction_columns(columns, max_k=max_k)
    effective_k = len(pred_cols)

    kept_id_cols = [col for col in id_cols if col in columns]
    usecols = list(dict.fromkeys([target_col, *kept_id_cols, *pred_cols]))

    n_rows = 0
    hit_count = 0
    rank_sum_found = 0.0
    reciprocal_rank_sum = 0.0
    lower_bound_rank_sum = 0.0
    rank_counts = np.zeros(effective_k, dtype=np.int64)

    for chunk in iter_prediction_chunks(path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue

        targets = chunk[target_col].to_numpy()
        preds = chunk[pred_cols].to_numpy()

        matches = preds == targets[:, None]
        found = matches.any(axis=1)

        ranks = np.full(len(chunk), np.nan, dtype=np.float64)
        if found.any():
            # argmax gives first True because True > False. Only valid for found rows.
            ranks[found] = matches[found].argmax(axis=1).astype(np.float64) + 1.0

        found_ranks = ranks[found]
        missing_count = int((~found).sum())
        batch_n = len(chunk)

        n_rows += batch_n
        hit_count += int(found.sum())
        rank_sum_found += float(np.nansum(ranks))
        reciprocal_rank_sum += float(np.sum(1.0 / found_ranks)) if found_ranks.size else 0.0
        lower_bound_rank_sum += float(np.nansum(ranks)) + missing_count * (effective_k + 1)

        if found_ranks.size:
            counts = np.bincount(found_ranks.astype(np.int64), minlength=effective_k + 1)[1 : effective_k + 1]
            rank_counts += counts

        if save_row_ranks:
            out = chunk[kept_id_cols].copy()
            if target_col not in out.columns:
                out[target_col] = chunk[target_col].to_numpy()
            out.insert(0, "model", model_name)
            out["rank"] = ranks
            out["found_in_top_k"] = found
            out["max_prediction_k"] = effective_k
            out["input_file"] = str(path)
            out.to_csv(
                row_ranks_path,
                mode="a",
                header=not row_ranks_header_written,
                index=False,
            )
            row_ranks_header_written = True

    if n_rows == 0:
        raise ValueError(f"No rows found in {path}")

    misses = n_rows - hit_count
    hit_rate = hit_count / n_rows
    mean_rank_found = rank_sum_found / hit_count if hit_count else float("nan")
    mean_rank_lower_bound = lower_bound_rank_sum / n_rows
    mrr = reciprocal_rank_sum / n_rows

    cumulative_hits = np.cumsum(rank_counts)
    recall_curve = pd.DataFrame(
        {
            "model": model_name,
            "k": np.arange(1, effective_k + 1, dtype=np.int64),
            "recall_at_k": cumulative_hits / n_rows,
            "hits_at_k": cumulative_hits,
            "n_rows": n_rows,
            "input_file": str(path),
        }
    )

    metrics = {
        "model": model_name,
        "max_prediction_k": effective_k,
        "n_rows": n_rows,
        "hits_in_top_k": hit_count,
        "misses_outside_top_k": misses,
        "hit_rate_at_max_k": hit_rate,
        "mean_required_k_found_only": mean_rank_found,
        "median_required_k_found_only": rank_quantile_from_counts(rank_counts, 0.50),
        "p90_required_k_found_only": rank_quantile_from_counts(rank_counts, 0.90),
        "p95_required_k_found_only": rank_quantile_from_counts(rank_counts, 0.95),
        "mean_required_k_lower_bound_missing_as_k_plus_1": mean_rank_lower_bound,
        "mrr_at_max_k_missing_as_zero": mrr,
        "exact_mean_required_k_available": misses == 0,
        "exact_mean_required_k": mean_rank_found if misses == 0 else np.nan,
        "input_file": str(path),
    }

    return metrics, recall_curve, row_ranks_header_written


def maybe_plot(metrics_df: pd.DataFrame, recall_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot recall curves.
    plt.figure(figsize=(10, 6))
    for model, sub in recall_df.groupby("model"):
        sub = sub.sort_values("k")
        plt.plot(sub["k"], sub["recall_at_k"], marker="o", markersize=3, label=model)
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Recall curve from prediction ranks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "recall_curve.png", dpi=200)
    plt.close()

    # Plot lower-bound average required k.
    plot_df = metrics_df.sort_values("mean_required_k_lower_bound_missing_as_k_plus_1")
    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model"], plot_df["mean_required_k_lower_bound_missing_as_k_plus_1"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average required k, lower bound")
    plt.title("Average minimum k needed; misses treated as K+1")
    plt.tight_layout()
    plt.savefig(plots_dir / "average_required_k_lower_bound.png", dpi=200)
    plt.close()

    # Plot mean rank among rows where the target was found.
    plot_df = metrics_df.sort_values("mean_required_k_found_only")
    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["model"], plot_df["mean_required_k_found_only"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average rank among hits")
    plt.title("Average required k among targets found in saved top-K")
    plt.tight_layout()
    plt.savefig(plots_dir / "average_required_k_found_only.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the average minimum k needed to recommend the correct item "
            "from prediction CSV/Parquet files with pred_1, pred_2, ... columns."
        )
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction files. Use either path.csv or model_name=path.csv.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV/JSON files.")
    parser.add_argument("--target-col", default="trackId", help="Column containing the true item ID.")
    parser.add_argument(
        "--id-cols",
        default="userId,trackId,ts",
        help="Comma-separated ID/context columns to keep in row_ranks.csv when --save-row-ranks is used.",
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
    parser.add_argument(
        "--save-row-ranks",
        action="store_true",
        help="Also save one row per prediction with the found rank. This can be large.",
    )
    parser.add_argument("--plot", action="store_true", help="Create simple PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    prediction_specs = [parse_prediction_arg(arg) for arg in args.predictions]

    row_ranks_path = output_dir / "row_ranks.csv"
    if row_ranks_path.exists() and args.save_row_ranks:
        row_ranks_path.unlink()
    row_ranks_header_written = False

    all_metrics = []
    all_recall_curves = []

    for model_name, path in prediction_specs:
        print(f"Evaluating {model_name}: {path}")
        metrics, recall_curve, row_ranks_header_written = evaluate_one_file(
            model_name=model_name,
            path=path,
            output_dir=output_dir,
            target_col=args.target_col,
            id_cols=id_cols,
            max_k=args.max_k,
            chunksize=args.chunksize,
            save_row_ranks=args.save_row_ranks,
            row_ranks_path=row_ranks_path,
            row_ranks_header_written=row_ranks_header_written,
        )
        all_metrics.append(metrics)
        all_recall_curves.append(recall_curve)

    metrics_df = pd.DataFrame(all_metrics)
    recall_df = pd.concat(all_recall_curves, ignore_index=True)

    metrics_path = output_dir / "rank_metrics_tidy.csv"
    recall_path = output_dir / "recall_curve_tidy.csv"
    metadata_path = output_dir / "rank_eval_metadata.json"

    metrics_df.to_csv(metrics_path, index=False)
    recall_df.to_csv(recall_path, index=False)

    metadata = {
        "target_col": args.target_col,
        "id_cols": id_cols,
        "max_k_requested": args.max_k,
        "predictions": [{"model": m, "path": str(p)} for m, p in prediction_specs],
        "outputs": {
            "rank_metrics_tidy": str(metrics_path),
            "recall_curve_tidy": str(recall_path),
            "row_ranks": str(row_ranks_path) if args.save_row_ranks else None,
        },
        "metric_notes": {
            "mean_required_k_found_only": "Average rank only over rows where the true item appears in saved pred_1..pred_K.",
            "mean_required_k_lower_bound_missing_as_k_plus_1": "Lower bound on the true average rank; rows not found in top-K are assigned K+1, but their real rank may be larger.",
            "mrr_at_max_k_missing_as_zero": "Mean reciprocal rank with reciprocal rank 0 for rows missing from top-K.",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.plot:
        maybe_plot(metrics_df, recall_df, output_dir)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {recall_path}")
    if args.save_row_ranks:
        print(f"Wrote {row_ranks_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute Recall@K from recommendation prediction files.

Expected prediction-file format:
    userId, trackId, ts, pred_1, pred_2, ..., pred_N

Each row is treated as one prediction event with one relevant item in `trackId`.
For this setup, Recall@K is equivalent to HitRate@K:
    1 if trackId appears anywhere in pred_1..pred_K, else 0.

Outputs:
    recall_tidy.csv
        Long/tidy format, easiest for plotting with pandas/seaborn/matplotlib.
        Columns: model, k, recall_micro, recall_macro_user, n_rows, n_users, input_file

    recall_wide_micro.csv
        Rows are k values, columns are models, values are micro Recall@K.

    recall_wide_macro_user.csv
        Rows are k values, columns are models, values are macro-user Recall@K.

Optional:
    recall_by_user.csv with --save-user
    recall_plot.png with --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRED_COL_RE_TEMPLATE = r"^{prefix}(\d+)$"


def derive_model_name(path: str) -> str:
    """Derive a clean model name from a prediction filename."""
    stem = Path(path).stem
    for suffix in ["_output", "_predictions", "_prediction", "_preds", "_pred"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def parse_prediction_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Expand prediction specs.

    Accepted forms:
        path/to/file.csv
        path/to/*.csv
        model_name=path/to/file.csv
        model_name=path/to/*.csv
    """
    expanded: List[Tuple[str, str]] = []

    for spec in specs:
        model_prefix: Optional[str] = None
        pattern = spec

        # Treat `name=path` as a named spec only when the left side looks like a name,
        # not like part of a path.
        if "=" in spec:
            left, right = spec.split("=", 1)
            if right and not any(sep in left for sep in ["/", "\\"]):
                model_prefix = left.strip()
                pattern = right.strip()

        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        if not matches:
            raise FileNotFoundError(f"No prediction files matched: {spec}")

        for path in matches:
            if model_prefix is None:
                model_name = derive_model_name(path)
            elif len(matches) == 1:
                model_name = model_prefix
            else:
                model_name = f"{model_prefix}_{derive_model_name(path)}"
            expanded.append((model_name, path))

    # Avoid duplicate model names after expansion.
    seen: Dict[str, int] = {}
    unique: List[Tuple[str, str]] = []
    for model_name, path in expanded:
        count = seen.get(model_name, 0)
        seen[model_name] = count + 1
        if count:
            model_name = f"{model_name}_{count + 1}"
        unique.append((model_name, path))

    return unique


def read_columns(path: str) -> List[str]:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        except Exception:
            # Fallback when pyarrow metadata access is unavailable.
            return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def find_prediction_columns(columns: Sequence[str], pred_prefix: str) -> List[str]:
    pattern = re.compile(PRED_COL_RE_TEMPLATE.format(prefix=re.escape(pred_prefix)))
    pred_cols: List[Tuple[int, str]] = []
    for col in columns:
        match = pattern.match(str(col))
        if match:
            pred_cols.append((int(match.group(1)), str(col)))

    if not pred_cols:
        raise ValueError(f"No prediction columns found with prefix '{pred_prefix}'.")

    pred_cols.sort(key=lambda x: x[0])

    # Warn by failing on missing ranks, because Recall@K depends on rank order.
    ranks = [rank for rank, _ in pred_cols]
    expected = list(range(1, max(ranks) + 1))
    if ranks != expected:
        missing = sorted(set(expected) - set(ranks))
        raise ValueError(
            f"Prediction columns must be contiguous from {pred_prefix}1. "
            f"Missing ranks: {missing[:20]}"
        )

    return [col for _, col in pred_cols]


def parse_ks(ks_arg: str, max_available_k: int) -> List[int]:
    if ks_arg.strip().lower() == "all":
        return list(range(1, max_available_k + 1))

    ks: List[int] = []
    for part in ks_arg.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError("All k values must be positive integers.")
        ks.append(k)

    ks = sorted(set(ks))
    if not ks:
        raise ValueError("No k values were provided.")
    if max(ks) > max_available_k:
        raise ValueError(
            f"Requested k={max(ks)}, but the file only contains predictions up to k={max_available_k}."
        )
    return ks


def iter_prediction_chunks(
    path: str,
    usecols: Sequence[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        # For parquet, read once. Prediction CSVs are usually the large files here,
        # and CSV supports chunking directly.
        yield pd.read_parquet(path, columns=list(usecols))
        return

    yield from pd.read_csv(path, usecols=list(usecols), chunksize=chunksize)


def evaluate_one_file(
    model_name: str,
    path: str,
    ks: List[int],
    truth_col: str,
    user_col: Optional[str],
    pred_cols: List[str],
    chunksize: int,
    save_user: bool,
) -> Tuple[List[dict], List[dict]]:
    """Return summary records and optional per-user records for one prediction file."""
    max_k = max(ks)
    selected_pred_cols = pred_cols[:max_k]
    k_positions = np.array([k - 1 for k in ks], dtype=np.int64)

    usecols = [truth_col] + selected_pred_cols
    if user_col is not None:
        usecols.append(user_col)

    total_rows = 0
    total_hits = np.zeros(len(ks), dtype=np.int64)

    # user_id -> count, user_id -> vector of hit sums for each k
    user_counts: Dict[object, int] = {}
    user_hit_sums: Dict[object, np.ndarray] = {}

    for chunk in iter_prediction_chunks(path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue

        truth = chunk[truth_col].to_numpy()
        preds = chunk[selected_pred_cols].to_numpy()

        # matches_by_rank[row, r] is true if pred_{r+1} equals the true item.
        matches_by_rank = preds == truth[:, None]
        hit_at_each_rank = np.maximum.accumulate(matches_by_rank, axis=1)
        selected_hits = hit_at_each_rank[:, k_positions].astype(np.int64, copy=False)

        total_rows += len(chunk)
        total_hits += selected_hits.sum(axis=0)

        if user_col is not None:
            users = chunk[user_col]
            counts = users.value_counts(sort=False)
            for uid, count in counts.items():
                user_counts[uid] = user_counts.get(uid, 0) + int(count)

            grouped_hits = pd.DataFrame(selected_hits, columns=ks)
            grouped_hits[user_col] = users.to_numpy()
            grouped_hits = grouped_hits.groupby(user_col, sort=False)[ks].sum()

            for uid, row in grouped_hits.iterrows():
                if uid not in user_hit_sums:
                    user_hit_sums[uid] = np.zeros(len(ks), dtype=np.int64)
                user_hit_sums[uid] += row.to_numpy(dtype=np.int64)

    if total_rows == 0:
        raise ValueError(f"Prediction file has no rows: {path}")

    micro = total_hits / total_rows

    n_users = len(user_counts) if user_col is not None else 0
    if user_col is not None and n_users > 0:
        user_recall_matrix = np.vstack(
            [user_hit_sums[uid] / user_counts[uid] for uid in user_counts.keys()]
        )
        macro_user = user_recall_matrix.mean(axis=0)
    else:
        macro_user = np.full(len(ks), np.nan, dtype=np.float64)

    summary_records: List[dict] = []
    for idx, k in enumerate(ks):
        summary_records.append(
            {
                "model": model_name,
                "k": int(k),
                "recall_micro": float(micro[idx]),
                "recall_macro_user": float(macro_user[idx]) if not np.isnan(macro_user[idx]) else np.nan,
                "n_rows": int(total_rows),
                "n_users": int(n_users),
                "input_file": str(path),
            }
        )

    user_records: List[dict] = []
    if save_user and user_col is not None:
        for uid, count in user_counts.items():
            recalls = user_hit_sums[uid] / count
            for idx, k in enumerate(ks):
                user_records.append(
                    {
                        "model": model_name,
                        user_col: uid,
                        "k": int(k),
                        "recall": float(recalls[idx]),
                        "n_rows": int(count),
                        "input_file": str(path),
                    }
                )

    return summary_records, user_records


def save_outputs(
    summary_df: pd.DataFrame,
    user_df: Optional[pd.DataFrame],
    output_dir: Path,
    save_parquet: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "recall_tidy.csv"
    summary_df.to_csv(summary_path, index=False)

    wide_micro = summary_df.pivot(index="k", columns="model", values="recall_micro").reset_index()
    wide_macro = summary_df.pivot(index="k", columns="model", values="recall_macro_user").reset_index()

    wide_micro.to_csv(output_dir / "recall_wide_micro.csv", index=False)
    wide_macro.to_csv(output_dir / "recall_wide_macro_user.csv", index=False)

    if user_df is not None:
        user_df.to_csv(output_dir / "recall_by_user.csv", index=False)

    if save_parquet:
        try:
            summary_df.to_parquet(output_dir / "recall_tidy.parquet", index=False)
            wide_micro.to_parquet(output_dir / "recall_wide_micro.parquet", index=False)
            wide_macro.to_parquet(output_dir / "recall_wide_macro_user.parquet", index=False)
            if user_df is not None:
                user_df.to_parquet(output_dir / "recall_by_user.parquet", index=False)
        except Exception as exc:  # pragma: no cover - depends on optional parquet engine
            print(f"Warning: could not save parquet outputs: {exc}")


def save_plot(summary_df: pd.DataFrame, output_dir: Path, metric: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    for model_name, group in summary_df.sort_values(["model", "k"]).groupby("model"):
        plt.plot(group["k"], group[metric], marker="o", label=model_name)

    ylabel = "Micro Recall@K" if metric == "recall_micro" else "Macro-user Recall@K"
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "recall_plot.png", dpi=200)
    plt.close()


def write_metadata(args: argparse.Namespace, prediction_specs: List[Tuple[str, str]], output_dir: Path) -> None:
    metadata = {
        "prediction_files": [{"model": model, "path": path} for model, path in prediction_specs],
        "truth_col": args.truth_col,
        "user_col": args.user_col,
        "pred_prefix": args.pred_prefix,
        "ks": args.ks,
        "chunksize": args.chunksize,
        "definition": "single-relevant-item row-level Recall@K / HitRate@K",
    }
    with open(output_dir / "recall_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Recall@K from prediction CSV/parquet files with pred_1..pred_N columns."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help=(
            "Prediction files or globs. You can optionally name them as model=path. "
            "Examples: ./predictions/*/*.csv ex2vec=./predictions/ex2vec.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory where recall files will be written.")
    parser.add_argument(
        "--ks",
        default="1,5,10,20,50",
        help="Comma-separated k values, e.g. 1,5,10,20,50, or 'all'.",
    )
    parser.add_argument("--truth-col", default="trackId", help="Column containing the true item id.")
    parser.add_argument("--user-col", default="userId", help="Column containing the user id. Use '' to disable macro-user recall.")
    parser.add_argument("--pred-prefix", default="pred_", help="Prediction-column prefix, e.g. pred_ for pred_1.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    parser.add_argument("--save-user", action="store_true", help="Also save per-user Recall@K to recall_by_user.csv.")
    parser.add_argument("--save-parquet", action="store_true", help="Also save parquet versions of the outputs.")
    parser.add_argument("--plot", action="store_true", help="Also save recall_plot.png.")
    parser.add_argument(
        "--plot-metric",
        choices=["recall_micro", "recall_macro_user"],
        default="recall_micro",
        help="Metric used for the optional plot.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_col = args.user_col if args.user_col.strip() else None
    prediction_specs = parse_prediction_specs(args.predictions)

    all_summary_records: List[dict] = []
    all_user_records: List[dict] = []

    for model_name, path in prediction_specs:
        print(f"Evaluating {model_name}: {path}")
        columns = read_columns(path)

        required_cols = [args.truth_col]
        if user_col is not None:
            required_cols.append(user_col)
        missing_required = [col for col in required_cols if col not in columns]
        if missing_required:
            raise ValueError(f"Missing required columns in {path}: {missing_required}")

        pred_cols = find_prediction_columns(columns, args.pred_prefix)
        ks = parse_ks(args.ks, max_available_k=len(pred_cols))

        summary_records, user_records = evaluate_one_file(
            model_name=model_name,
            path=path,
            ks=ks,
            truth_col=args.truth_col,
            user_col=user_col,
            pred_cols=pred_cols,
            chunksize=args.chunksize,
            save_user=args.save_user,
        )
        all_summary_records.extend(summary_records)
        all_user_records.extend(user_records)

    summary_df = pd.DataFrame(all_summary_records).sort_values(["model", "k"]).reset_index(drop=True)
    user_df = None
    if args.save_user:
        user_df = pd.DataFrame(all_user_records).sort_values(["model", user_col or "userId", "k"]).reset_index(drop=True)

    save_outputs(summary_df, user_df, output_dir, save_parquet=args.save_parquet)
    write_metadata(args, prediction_specs, output_dir)

    if args.plot:
        save_plot(summary_df, output_dir, metric=args.plot_metric)

    print("\nSaved:")
    for filename in [
        "recall_tidy.csv",
        "recall_wide_micro.csv",
        "recall_wide_macro_user.csv",
        "recall_metadata.json",
    ]:
        print(f"  {output_dir / filename}")
    if args.save_user:
        print(f"  {output_dir / 'recall_by_user.csv'}")
    if args.plot:
        print(f"  {output_dir / 'recall_plot.png'}")


if __name__ == "__main__":
    main()

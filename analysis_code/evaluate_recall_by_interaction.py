#!/usr/bin/env python3
"""
Compute Recall@K / Accuracy@K grouped by interaction number.

Interaction number definition:
    For each (user, item) pair, interactions are ordered by time in the full
    interaction dataset. The first time the user interacts with that item has
    interaction_number = 1, the second has interaction_number = 2, and so on.

Expected prediction-file format:
    userId, trackId, ts, pred_1, pred_2, ..., pred_N

Each prediction row has one true target item in `trackId`. Therefore, for this
next-item setup, Recall@K is the same as HitRate@K / Accuracy@K:
    1 if trackId is present anywhere in pred_1..pred_K, else 0.

Typical use:
    python evaluate_recall_by_interaction.py \
      --predictions ./predictions/baselines/*_output.csv ./predictions/extendedMLP/extendedmlp_output.csv \
      --data-parquet ../../../../sorted_data.parquet \
      --output-dir ./predictions/interaction_eval \
      --ks 1,5,10,20,50 \
      --plot

Outputs:
    recall_by_interaction_tidy.csv
        Long/tidy file, best for plotting.
        Columns:
            model, k, interaction_number, recall_at_k, accuracy_at_k,
            n_rows, n_users, input_file

    recall_by_interaction_wide.csv
        Wide comparison table.
        Rows are (interaction_number, k), columns are models.

    interaction_counts.csv
        Number of evaluated rows/users per interaction number and model.

    recall_by_interaction_metadata.json
        Reproducibility metadata.

Optional:
    parquet versions with --save-parquet
    one PNG per requested k with --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRED_COL_RE_TEMPLATE = r"^{prefix}(\d+)$"


# ---------------------------------------------------------------------------
# Prediction-file helpers
# ---------------------------------------------------------------------------


def derive_model_name(path: str) -> str:
    """Derive a readable model name from a prediction filename."""
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

        # Treat `name=path` as a named spec only when the left side looks like
        # a model name, not like part of a filesystem path.
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
        # Parquet does not use pandas chunking here. If parquet prediction files
        # become very large, split them before running this script.
        yield pd.read_parquet(path, columns=list(usecols))
        return

    yield from pd.read_csv(path, usecols=list(usecols), chunksize=chunksize)


# ---------------------------------------------------------------------------
# Interaction-number mapping
# ---------------------------------------------------------------------------


def build_interaction_mapping(
    data_parquet: str,
    data_user_col: str,
    data_item_col: str,
    data_time_col: str,
    pred_user_col: str,
    pred_item_col: str,
    pred_time_col: str,
    assume_sorted: bool,
) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Build a dataframe mapping prediction keys to interaction_number.

    Returns:
        mapping_df
        has_duplicate_prediction_keys
        merge_columns
    """
    data_cols = [data_user_col, data_item_col, data_time_col]
    print(f"Loading interaction data from {data_parquet}")
    df = pd.read_parquet(data_parquet, columns=data_cols)

    rename_map = {
        data_user_col: pred_user_col,
        data_item_col: pred_item_col,
        data_time_col: pred_time_col,
    }
    df = df.rename(columns=rename_map)

    key_cols = [pred_user_col, pred_item_col, pred_time_col]

    if not assume_sorted:
        # Stable sort keeps deterministic ordering for equal timestamps.
        print("Sorting interaction data by user, item, and time")
        df = df.sort_values([pred_user_col, pred_item_col, pred_time_col], kind="mergesort")

    # The occurrence count of this item for this user, up to and including this row.
    df["interaction_number"] = (
        df.groupby([pred_user_col, pred_item_col], sort=False).cumcount().astype(np.int64) + 1
    )

    duplicate_mask = df.duplicated(key_cols, keep=False)
    has_duplicate_keys = bool(duplicate_mask.any())

    if has_duplicate_keys:
        print(
            "Found duplicate (user, item, time) keys in the interaction data; "
            "using an additional duplicate-rank key for safe merging."
        )
        df["_dup_rank"] = df.groupby(key_cols, sort=False).cumcount().astype(np.int64)
        merge_cols = key_cols + ["_dup_rank"]
        mapping = df[merge_cols + ["interaction_number"]]
    else:
        merge_cols = key_cols
        mapping = df[merge_cols + ["interaction_number"]].drop_duplicates(merge_cols)

    # Keep the mapping compact.
    mapping = mapping.reset_index(drop=True)
    return mapping, has_duplicate_keys, merge_cols



def add_streaming_duplicate_rank(
    chunk: pd.DataFrame,
    key_cols: Sequence[str],
    seen_counts: Dict[Tuple[object, ...], int],
) -> pd.DataFrame:
    """Add _dup_rank to a prediction chunk, consistent across chunks."""
    chunk = chunk.copy()
    local_rank = chunk.groupby(list(key_cols), sort=False).cumcount().to_numpy(dtype=np.int64)

    keys = list(zip(*(chunk[col].to_numpy() for col in key_cols)))
    prior = np.fromiter((seen_counts.get(key, 0) for key in keys), dtype=np.int64, count=len(chunk))
    chunk["_dup_rank"] = local_rank + prior

    sizes = chunk.groupby(list(key_cols), sort=False).size()
    for key, count in sizes.items():
        if not isinstance(key, tuple):
            key = (key,)
        seen_counts[key] = seen_counts.get(key, 0) + int(count)

    return chunk


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_one_file(
    model_name: str,
    path: str,
    ks: List[int],
    pred_cols: List[str],
    truth_col: str,
    user_col: str,
    time_col: str,
    chunksize: int,
    interaction_mapping: Optional[pd.DataFrame],
    mapping_uses_duplicate_rank: bool,
    merge_cols: Sequence[str],
    interaction_col: str,
    track_users: bool,
) -> Tuple[List[dict], List[dict], int]:
    """Evaluate one prediction file and return tidy records and count records."""
    max_k = max(ks)
    selected_pred_cols = pred_cols[:max_k]
    k_positions = np.array([k - 1 for k in ks], dtype=np.int64)

    usecols = [truth_col, user_col, time_col] + selected_pred_cols
    if interaction_mapping is None and interaction_col not in usecols:
        usecols.append(interaction_col)

    counts_by_interaction: DefaultDict[int, int] = defaultdict(int)
    hit_sums_by_interaction: Dict[int, np.ndarray] = {}
    users_by_interaction: DefaultDict[int, set] = defaultdict(set)
    duplicate_rank_counts: Dict[Tuple[object, ...], int] = {}

    total_rows = 0
    unmatched_rows = 0

    for chunk in iter_prediction_chunks(path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue

        if interaction_mapping is not None:
            if mapping_uses_duplicate_rank:
                chunk = add_streaming_duplicate_rank(
                    chunk,
                    key_cols=[user_col, truth_col, time_col],
                    seen_counts=duplicate_rank_counts,
                )

            before = len(chunk)
            chunk = chunk.merge(interaction_mapping, on=list(merge_cols), how="left")
            missing = int(chunk["interaction_number"].isna().sum())
            unmatched_rows += missing
            if missing:
                chunk = chunk.dropna(subset=["interaction_number"])
            if len(chunk) == 0:
                continue
            current_interaction_col = "interaction_number"
        else:
            current_interaction_col = interaction_col

        truth = chunk[truth_col].to_numpy()
        preds = chunk[selected_pred_cols].to_numpy()
        interactions = chunk[current_interaction_col].to_numpy(dtype=np.int64)

        matches_by_rank = preds == truth[:, None]
        hit_at_each_rank = np.maximum.accumulate(matches_by_rank, axis=1)
        selected_hits = hit_at_each_rank[:, k_positions].astype(np.int64, copy=False)

        total_rows += len(chunk)

        tmp = pd.DataFrame({"interaction_number": interactions})
        for idx, k in enumerate(ks):
            tmp[k] = selected_hits[:, idx]
        if track_users:
            tmp[user_col] = chunk[user_col].to_numpy()

        grouped = tmp.groupby("interaction_number", sort=False)
        count_series = grouped.size()
        hit_sums = grouped[ks].sum()

        for interaction_number, count in count_series.items():
            interaction_number_int = int(interaction_number)
            counts_by_interaction[interaction_number_int] += int(count)
            if interaction_number_int not in hit_sums_by_interaction:
                hit_sums_by_interaction[interaction_number_int] = np.zeros(len(ks), dtype=np.int64)
            hit_sums_by_interaction[interaction_number_int] += hit_sums.loc[interaction_number].to_numpy(
                dtype=np.int64
            )

        if track_users:
            # Exact n_users per interaction number. This can use memory on very
            # large files; disable with --no-track-users if needed.
            user_groups = tmp.groupby("interaction_number", sort=False)[user_col].unique()
            for interaction_number, users in user_groups.items():
                users_by_interaction[int(interaction_number)].update(users.tolist())

    if total_rows == 0:
        raise ValueError(f"Prediction file has no usable rows: {path}")

    records: List[dict] = []
    count_records: List[dict] = []

    for interaction_number in sorted(counts_by_interaction.keys()):
        n_rows = counts_by_interaction[interaction_number]
        n_users = len(users_by_interaction[interaction_number]) if track_users else np.nan
        hit_sums = hit_sums_by_interaction[interaction_number]

        count_records.append(
            {
                "model": model_name,
                "interaction_number": int(interaction_number),
                "n_rows": int(n_rows),
                "n_users": int(n_users) if track_users else np.nan,
                "input_file": str(path),
            }
        )

        for idx, k in enumerate(ks):
            value = float(hit_sums[idx] / n_rows)
            records.append(
                {
                    "model": model_name,
                    "k": int(k),
                    "interaction_number": int(interaction_number),
                    "recall_at_k": value,
                    # Alias included because the user asked for accuracy@interaction.
                    "accuracy_at_k": value,
                    "n_rows": int(n_rows),
                    "n_users": int(n_users) if track_users else np.nan,
                    "input_file": str(path),
                }
            )

    return records, count_records, unmatched_rows


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def save_outputs(
    tidy_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    output_dir: Path,
    save_parquet: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tidy_df.to_csv(output_dir / "recall_by_interaction_tidy.csv", index=False)
    counts_df.to_csv(output_dir / "interaction_counts.csv", index=False)

    wide_df = (
        tidy_df.pivot_table(
            index=["interaction_number", "k"],
            columns="model",
            values="recall_at_k",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["k", "interaction_number"])
    )
    wide_df.to_csv(output_dir / "recall_by_interaction_wide.csv", index=False)

    if save_parquet:
        try:
            tidy_df.to_parquet(output_dir / "recall_by_interaction_tidy.parquet", index=False)
            counts_df.to_parquet(output_dir / "interaction_counts.parquet", index=False)
            wide_df.to_parquet(output_dir / "recall_by_interaction_wide.parquet", index=False)
        except Exception as exc:  # pragma: no cover - optional parquet engine
            print(f"Warning: could not save parquet outputs: {exc}")



def save_plots(
    tidy_df: pd.DataFrame,
    output_dir: Path,
    plot_min_rows: int,
    plot_max_interaction: Optional[int],
    log_x: bool,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for k, k_df in tidy_df.groupby("k", sort=True):
        plot_df = k_df.copy()
        if plot_min_rows > 1:
            plot_df = plot_df[plot_df["n_rows"] >= plot_min_rows]
        if plot_max_interaction is not None:
            plot_df = plot_df[plot_df["interaction_number"] <= plot_max_interaction]

        if plot_df.empty:
            print(f"Skipping plot for k={k}: no rows after plot filters.")
            continue

        plt.figure(figsize=(9, 5.5))
        for model_name, group in plot_df.sort_values(["model", "interaction_number"]).groupby("model"):
            plt.plot(group["interaction_number"], group["recall_at_k"], marker="o", label=model_name)

        plt.xlabel("Interaction number for this user-track pair")
        plt.ylabel(f"Recall@{k} / Accuracy@{k}")
        plt.ylim(0, 1)
        if log_x:
            plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(plot_dir / f"recall_by_interaction_at_{k}.png", dpi=200)
        plt.close()



def write_metadata(
    args: argparse.Namespace,
    prediction_specs: List[Tuple[str, str]],
    output_dir: Path,
    unmatched_by_model: Dict[str, int],
) -> None:
    metadata = {
        "prediction_files": [{"model": model, "path": path} for model, path in prediction_specs],
        "data_parquet": args.data_parquet,
        "truth_col": args.truth_col,
        "user_col": args.user_col,
        "time_col": args.time_col,
        "pred_prefix": args.pred_prefix,
        "ks": args.ks,
        "chunksize": args.chunksize,
        "interaction_definition": (
            "For each user-item pair, interaction_number is the 1-based chronological "
            "occurrence count in the full interaction data."
        ),
        "metric_definition": (
            "single-relevant-item Recall@K, equivalent to HitRate@K/Accuracy@K: "
            "1 if the true item appears in pred_1..pred_K, otherwise 0"
        ),
        "unmatched_prediction_rows_by_model": unmatched_by_model,
    }
    with open(output_dir / "recall_by_interaction_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Recall@K / Accuracy@K grouped by user-track interaction number."
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
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation files will be written.")
    parser.add_argument(
        "--ks",
        default="1,5,10,20,50",
        help="Comma-separated k values, e.g. 1,5,10,20,50, or 'all'.",
    )

    parser.add_argument(
        "--data-parquet",
        default=None,
        help=(
            "Full interaction parquet used to compute interaction_number. "
            "Required unless prediction files already contain --interaction-col."
        ),
    )
    parser.add_argument(
        "--assume-data-sorted",
        action="store_true",
        help=(
            "Skip sorting the full interaction data before counting occurrences. "
            "Use only if the parquet is already sorted chronologically within each user-track pair."
        ),
    )

    parser.add_argument("--truth-col", default="trackId", help="Prediction-file column containing the true item id.")
    parser.add_argument("--user-col", default="userId", help="Prediction-file column containing the user id.")
    parser.add_argument("--time-col", default="ts", help="Prediction-file column containing the prediction timestamp.")
    parser.add_argument("--pred-prefix", default="pred_", help="Prediction-column prefix, e.g. pred_ for pred_1.")
    parser.add_argument(
        "--interaction-col",
        default="interaction_number",
        help="Existing prediction-file interaction-number column to use when --data-parquet is omitted.",
    )

    parser.add_argument("--data-user-col", default="user_id", help="Full-data parquet user column.")
    parser.add_argument("--data-item-col", default="track_id", help="Full-data parquet item column.")
    parser.add_argument("--data-time-col", default="ts", help="Full-data parquet timestamp column.")

    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    parser.add_argument("--save-parquet", action="store_true", help="Also save parquet versions of the outputs.")
    parser.add_argument(
        "--no-track-users",
        action="store_true",
        help="Do not compute exact n_users per interaction number; saves memory on very large files.",
    )

    parser.add_argument("--plot", action="store_true", help="Also save one plot per selected k.")
    parser.add_argument(
        "--plot-min-rows",
        type=int,
        default=1,
        help="For plots only, hide interaction numbers with fewer than this many rows.",
    )
    parser.add_argument(
        "--plot-max-interaction",
        type=int,
        default=None,
        help="For plots only, hide interaction numbers larger than this value.",
    )
    parser.add_argument("--plot-log-x", action="store_true", help="Use a log-scaled x-axis in optional plots.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_specs = parse_prediction_specs(args.predictions)

    # Inspect first file to determine k values. All files are checked later.
    first_columns = read_columns(prediction_specs[0][1])
    first_pred_cols = find_prediction_columns(first_columns, args.pred_prefix)
    ks = parse_ks(args.ks, max_available_k=len(first_pred_cols))

    interaction_mapping: Optional[pd.DataFrame] = None
    mapping_uses_duplicate_rank = False
    merge_cols: List[str] = [args.user_col, args.truth_col, args.time_col]

    if args.data_parquet is not None:
        interaction_mapping, mapping_uses_duplicate_rank, merge_cols = build_interaction_mapping(
            data_parquet=args.data_parquet,
            data_user_col=args.data_user_col,
            data_item_col=args.data_item_col,
            data_time_col=args.data_time_col,
            pred_user_col=args.user_col,
            pred_item_col=args.truth_col,
            pred_time_col=args.time_col,
            assume_sorted=args.assume_data_sorted,
        )
    else:
        # Make sure every prediction file has the interaction column.
        for _, path in prediction_specs:
            columns = read_columns(path)
            if args.interaction_col not in columns:
                raise ValueError(
                    f"{path} does not contain '{args.interaction_col}'. "
                    "Provide --data-parquet to compute interaction numbers from the full dataset."
                )

    all_records: List[dict] = []
    all_count_records: List[dict] = []
    unmatched_by_model: Dict[str, int] = {}

    for model_name, path in prediction_specs:
        print(f"Evaluating {model_name}: {path}")
        columns = read_columns(path)

        required_cols = [args.truth_col, args.user_col, args.time_col]
        if interaction_mapping is None:
            required_cols.append(args.interaction_col)
        missing_required = [col for col in required_cols if col not in columns]
        if missing_required:
            raise ValueError(f"Missing required columns in {path}: {missing_required}")

        pred_cols = find_prediction_columns(columns, args.pred_prefix)
        parse_ks(args.ks, max_available_k=len(pred_cols))  # validates this file

        records, count_records, unmatched_rows = evaluate_one_file(
            model_name=model_name,
            path=path,
            ks=ks,
            pred_cols=pred_cols,
            truth_col=args.truth_col,
            user_col=args.user_col,
            time_col=args.time_col,
            chunksize=args.chunksize,
            interaction_mapping=interaction_mapping,
            mapping_uses_duplicate_rank=mapping_uses_duplicate_rank,
            merge_cols=merge_cols,
            interaction_col=args.interaction_col,
            track_users=not args.no_track_users,
        )
        all_records.extend(records)
        all_count_records.extend(count_records)
        unmatched_by_model[model_name] = int(unmatched_rows)
        if unmatched_rows:
            print(f"Warning: {model_name} had {unmatched_rows} prediction rows not matched to interaction data.")

    tidy_df = (
        pd.DataFrame(all_records)
        .sort_values(["model", "k", "interaction_number"])
        .reset_index(drop=True)
    )
    counts_df = (
        pd.DataFrame(all_count_records)
        .sort_values(["model", "interaction_number"])
        .reset_index(drop=True)
    )

    save_outputs(tidy_df, counts_df, output_dir, save_parquet=args.save_parquet)
    write_metadata(args, prediction_specs, output_dir, unmatched_by_model=unmatched_by_model)

    if args.plot:
        save_plots(
            tidy_df,
            output_dir,
            plot_min_rows=args.plot_min_rows,
            plot_max_interaction=args.plot_max_interaction,
            log_x=args.plot_log_x,
        )

    print("\nSaved:")
    for filename in [
        "recall_by_interaction_tidy.csv",
        "recall_by_interaction_wide.csv",
        "interaction_counts.csv",
        "recall_by_interaction_metadata.json",
    ]:
        print(f"  {output_dir / filename}")
    if args.plot:
        print(f"  {output_dir / 'plots'}")


if __name__ == "__main__":
    main()

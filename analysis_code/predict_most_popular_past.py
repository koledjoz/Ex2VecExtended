#!/usr/bin/env python3
"""
Generate top-k prediction CSVs using a global most-popular-past-item baseline.

For every evaluation interaction (user_id, track_id, ts), the predictions are
ranked by item popularity among ALL interactions with timestamp strictly smaller
than that row's timestamp. Popularity is global, not user-specific.

The output format matches predict_extended_mlp.py and predict_baselines.py:
  userId,trackId,ts,pred_1,...,pred_k

Example:
python predict_most_popular_past.py \
  --data-parquet ../../../../sorted_data.parquet \
  --test-dict ../../../../split_data/test/test_dict.json \
  --output-dir ./predictions/baselines \
  --top-k 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


USER_COL = "user_id"
ITEM_COL = "track_id"
TIME_COL = "ts"


# ---------------------------------------------------------------------------
# Loading / preparation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate predictions for a global most-popular-past-item baseline. "
            "Popularity is computed over all users/items before each query timestamp."
        )
    )
    parser.add_argument("--data-parquet", required=True, help="Path to sorted_data.parquet")
    parser.add_argument("--test-dict", required=True, help="Path to split_data/test/test_dict.json")
    parser.add_argument("--output-dir", required=True, help="Directory where the CSV file will be written")
    parser.add_argument("--top-k", type=int, default=50, help="Number of predictions per row")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for tie-breaking/fill predictions")
    parser.add_argument(
        "--csv-chunk-size",
        type=int,
        default=100_000,
        help="Number of prediction rows buffered before appending to CSV.",
    )
    parser.add_argument(
        "--output-name",
        default="most_popular_past_output.csv",
        help="Output CSV file name inside --output-dir.",
    )
    return parser.parse_args()


def read_main_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {USER_COL, ITEM_COL, TIME_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")

    # Keep the same columns and dtypes expected by the other baseline scripts.
    df = df[[USER_COL, ITEM_COL, TIME_COL]].copy()
    df[USER_COL] = df[USER_COL].astype(np.int64)
    df[ITEM_COL] = df[ITEM_COL].astype(np.int64)
    df = df.sort_values([TIME_COL, USER_COL], kind="mergesort").reset_index(drop=True)
    return df


def _extract_test_rows_from_json(test_obj: dict) -> Tuple[pd.DataFrame, bool]:
    """
    Supports the same test_dict formats as predict_baselines.py:
        {"user_id": [track_id, track_id, ...]}
        {"user_id": [{"track_id": 123, "ts": 456}, ...]}
        {"user_id": [[123, 456], ...]}  # interpreted as [track_id, ts]
    """
    rows = []
    has_ts = False

    for u_raw, values in test_obj.items():
        user_id = int(u_raw)
        for value in values:
            if isinstance(value, dict):
                item = int(value.get(ITEM_COL, value.get("trackId", value.get("item_id"))))
                if TIME_COL in value:
                    rows.append((user_id, item, value[TIME_COL]))
                    has_ts = True
                else:
                    rows.append((user_id, item, None))
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                rows.append((user_id, int(value[0]), value[1]))
                has_ts = True
            else:
                rows.append((user_id, int(value), None))

    if has_ts:
        out = pd.DataFrame(rows, columns=[USER_COL, ITEM_COL, TIME_COL])
        out[TIME_COL] = out[TIME_COL].astype(np.asarray(out[TIME_COL]).dtype)
        return out, True

    out = pd.DataFrame([(u, i) for u, i, _ in rows], columns=[USER_COL, ITEM_COL])
    return out, False


def make_eval_dataframe(df: pd.DataFrame, test_dict_path: str | Path) -> pd.DataFrame:
    """
    Recreates the evaluation dataframe style from predict_baselines.py:
    load test_dict.json, build user_id/track_id pairs, and inner-merge with df.

    If the JSON also contains timestamps, the merge uses user_id/track_id/ts.
    """
    with open(test_dict_path, "r") as f:
        test_obj = json.load(f)

    filter_df, has_ts = _extract_test_rows_from_json(test_obj)
    merge_cols = [USER_COL, ITEM_COL, TIME_COL] if has_ts else [USER_COL, ITEM_COL]
    eval_df = df.merge(filter_df, on=merge_cols, how="inner")
    eval_df = eval_df[[USER_COL, ITEM_COL, TIME_COL]].copy()
    eval_df = eval_df.sort_values([TIME_COL, USER_COL], kind="mergesort").reset_index(drop=True)
    return eval_df


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def prediction_columns(k: int) -> List[str]:
    return [f"pred_{i + 1}" for i in range(k)]


def fill_random(
    preds: Sequence[int],
    rng: np.random.Generator,
    item_count: int,
    top_k: int,
) -> np.ndarray:
    """Fill a partially built recommendation list with distinct random items."""
    out: List[int] = []
    seen = set()
    for p in preds:
        p_int = int(p)
        if 1 <= p_int <= item_count and p_int not in seen:
            out.append(p_int)
            seen.add(p_int)
            if len(out) == top_k:
                return np.asarray(out, dtype=np.int64)

    if len(out) < top_k:
        candidates = np.setdiff1d(
            np.arange(1, item_count + 1, dtype=np.int64),
            np.fromiter(seen, dtype=np.int64),
            assume_unique=False,
        )
        need = top_k - len(out)
        if need > len(candidates):
            raise ValueError(f"Cannot create {top_k} distinct predictions with only {item_count} items.")
        out.extend(rng.choice(candidates, size=need, replace=False).astype(np.int64).tolist())

    return np.asarray(out, dtype=np.int64)


def top_items_from_counts(
    counts: np.ndarray,
    rng: np.random.Generator,
    item_count: int,
    top_k: int,
) -> np.ndarray:
    """
    Return the globally most popular items based on current count state.

    Items with zero previous interactions are not considered popular and are only
    used for random fill if fewer than top_k items have appeared in the past.
    Ties among equally popular items are broken randomly using the provided RNG.
    """
    seen_items = np.flatnonzero(counts[1 : item_count + 1] > 0) + 1
    if len(seen_items) == 0:
        return fill_random([], rng, item_count, top_k)

    # Small random jitter gives reproducible random tie-breaking without changing
    # the order of items with different integer popularity counts.
    jitter = rng.random(len(seen_items)) * 1e-12
    ordered = seen_items[np.argsort(-(counts[seen_items].astype(np.float64) + jitter))]
    return fill_random(ordered[:top_k], rng, item_count, top_k)


def write_prediction_csv(
    output_path: Path,
    rows: Iterable[Sequence[int | float]],
    top_k: int,
    chunk_size: int,
    total: Optional[int] = None,
    desc: str = "Writing predictions",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    cols = ["userId", "trackId", "ts"] + prediction_columns(top_k)
    buffer: List[Sequence[int | float]] = []
    wrote_header = False

    for row in tqdm(rows, total=total, desc=desc):
        buffer.append(row)
        if len(buffer) >= chunk_size:
            pd.DataFrame(buffer, columns=cols).to_csv(
                output_path,
                mode="a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            buffer.clear()

    if buffer:
        pd.DataFrame(buffer, columns=cols).to_csv(
            output_path,
            mode="a",
            header=not wrote_header,
            index=False,
        )


# ---------------------------------------------------------------------------
# Baseline generator
# ---------------------------------------------------------------------------

def iter_most_popular_past_predictions(
    eval_df: pd.DataFrame,
    all_df: pd.DataFrame,
    item_count: int,
    top_k: int,
    rng: np.random.Generator,
) -> Iterable[List[int | float]]:
    """
    Yield predictions for each evaluation row using global past popularity.

    The count vector is updated with every interaction from all_df whose
    timestamp is strictly smaller than the current evaluation timestamp. This
    means that popularity is based on all users, but never uses the current or
    future interactions at the same/later timestamp.
    """
    all_times = all_df[TIME_COL].to_numpy()
    all_items = all_df[ITEM_COL].to_numpy(dtype=np.int64)
    counts = np.zeros(item_count + 1, dtype=np.int64)
    data_pos = 0
    n_interactions = len(all_df)

    for row in eval_df.itertuples(index=False):
        user_id = int(getattr(row, USER_COL))
        true_item = int(getattr(row, ITEM_COL))
        ts = getattr(row, TIME_COL)

        # Strictly past: only interactions with timestamp < ts are counted.
        while data_pos < n_interactions and all_times[data_pos] < ts:
            item = int(all_items[data_pos])
            if 1 <= item <= item_count:
                counts[item] += 1
            data_pos += 1

        preds = top_items_from_counts(counts, rng, item_count, top_k)
        yield [user_id, true_item, ts] + preds.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_main_dataframe(args.data_parquet)
    eval_df = make_eval_dataframe(df, args.test_dict)

    item_count = int(df[ITEM_COL].max())
    if args.top_k > item_count:
        raise ValueError(f"--top-k={args.top_k} is larger than item_count={item_count}")

    print(f"Loaded {len(df):,} interactions")
    print(f"Evaluation rows: {len(eval_df):,}")
    print(f"Users: {int(df[USER_COL].max()):,}; items: {item_count:,}; top_k={args.top_k}")
    print("Baseline: global most-popular-past item prediction")

    rng = np.random.default_rng(args.seed)
    out_path = output_dir / args.output_name
    rows = iter_most_popular_past_predictions(eval_df, df, item_count, args.top_k, rng)
    write_prediction_csv(
        out_path,
        rows,
        args.top_k,
        args.csv_chunk_size,
        total=len(eval_df),
        desc="most_popular_past",
    )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

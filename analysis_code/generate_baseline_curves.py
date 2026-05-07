#!/usr/bin/env python3
"""
Generate score/probability curves for the thesis baselines.

The grouped output format intentionally matches the curve file produced by
`generate_curve_extended_mlp.py`:

    user,item,score,prob

where `score` and `prob` are list-valued columns ordered by the shared time grid.
Optionally, the script can also write a long/tidy CSV with one row per
(model, user, item, time step).

Baselines implemented:
  - random: uniform score/probability for every item
  - last_item: recency-rank score over the user's unique past items
  - bl_proxy: ACT-R/base-level strength per item
  - bl_knn: base-level weighted user vector + pretrained item-embedding kNN score

Example:
python generate_baseline_curves.py \
  --data-parquet ../../../../sorted_data.parquet \
  --test-dict ../../../../split_data/test/test_dict.json \
  --output-dir ./curves/baselines \
  --embedding-parquet /home/koledjoz/Ex2VecExtended/split_data/track_embeddings.parquet \
  --item-mapping /home/koledjoz/Ex2VecExtended/configs/models/item_mapping.json \
  --start-time 1654041600 \
  --end-time 1661990376 \
  --n-steps 500 \
  --baselines all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None


USER_COL = "user_id"
ITEM_COL = "track_id"
TIME_COL = "ts"
BASELINES = ("random", "last_item", "bl_proxy", "bl_knn")


# ---------------------------------------------------------------------------
# CLI / loading
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline score/probability curves for target user-item pairs."
    )
    parser.add_argument("--data-parquet", required=True, help="Path to sorted_data.parquet")
    parser.add_argument("--test-dict", required=True, help="Path to split_data/test/test_dict.json")
    parser.add_argument("--output-dir", required=True, help="Directory where curve CSV files will be written")

    parser.add_argument("--start-time", type=int, default=1654041600, help="First timestamp in the curve grid")
    parser.add_argument("--end-time", type=int, default=1661990376, help="Last timestamp in the curve grid")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of points in each curve")

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["all"],
        choices=["all", *BASELINES],
        help="Baselines to run. Use 'all' for every baseline.",
    )

    parser.add_argument(
        "--embedding-parquet",
        default=None,
        help="Path to Deezer/pretrained item embeddings. Required for bl_knn.",
    )
    parser.add_argument(
        "--item-mapping",
        default=None,
        help="JSON mapping from original Deezer track ids to dense track_id ids. Required/recommended for bl_knn.",
    )
    parser.add_argument(
        "--embedding-id-col",
        default="track_id",
        help="ID column in the pretrained embedding parquet. Default: track_id.",
    )
    parser.add_argument(
        "--embedding-vector-col",
        default="vector",
        help="Nested vector column in the pretrained embedding parquet. Default: vector.",
    )

    parser.add_argument("--decay", type=float, default=0.5, help="ACT-R/base-level decay parameter d")
    parser.add_argument(
        "--cutoff-c",
        type=float,
        default=1.0,
        help="Positive cutoff added to time differences: (t - tj + c)^(-d)",
    )
    parser.add_argument(
        "--bl-transform",
        choices=["strength", "log1p"],
        default="strength",
        help="Score transform for BL proxy. log1p is monotonic and keeps unseen items at score 0.",
    )

    parser.add_argument(
        "--last-item-score-scale",
        type=float,
        default=1.0,
        help="Scale for last-item scores. Score is scale / recency_rank for seen items, 0 otherwise.",
    )
    parser.add_argument(
        "--last-item-max-rank",
        type=int,
        default=0,
        help="Only give positive score to this many most-recent unique items. 0 means all unique past items.",
    )

    parser.add_argument(
        "--knn-score-scale",
        type=float,
        default=1.0,
        help="Multiply BL-kNN scores by this value before softmax.",
    )
    parser.add_argument(
        "--knn-distance",
        choices=["squared", "euclidean"],
        default="squared",
        help="BL-kNN score is -distance. 'squared' matches the top-k prediction script's ordering.",
    )
    parser.add_argument(
        "--knn-time-batch-size",
        type=int,
        default=64,
        help="Number of timestamps processed at once for vectorized BL-kNN curves.",
    )
    parser.add_argument(
        "--knn-missing-score",
        type=float,
        default=-1e9,
        help="Score assigned to candidate items with missing/zero pretrained embeddings.",
    )

    parser.add_argument(
        "--include-times",
        action="store_true",
        help="Include a list-valued ts column in the grouped output files.",
    )
    parser.add_argument(
        "--save-tidy",
        action="store_true",
        help="Also write baseline_curves_tidy.csv with columns: model,user,item,step,ts,score,prob.",
    )
    parser.add_argument(
        "--float-format",
        default=".8g",
        help="Format used inside list-valued score/prob columns. Default: .8g",
    )
    return parser.parse_args()


def normalize_baselines(names: Sequence[str]) -> List[str]:
    if "all" in names:
        return list(BASELINES)
    out: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def read_main_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {USER_COL, ITEM_COL, TIME_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")

    df = df[[USER_COL, ITEM_COL, TIME_COL]].copy()
    df[USER_COL] = df[USER_COL].astype(np.int64)
    df[ITEM_COL] = df[ITEM_COL].astype(np.int64)
    df[TIME_COL] = df[TIME_COL].astype(np.int64)
    return df.sort_values([USER_COL, TIME_COL], kind="mergesort").reset_index(drop=True)


def _extract_item_id(value) -> int:
    """Accept the simple test_dict format and a few more explicit variants."""
    if isinstance(value, dict):
        for key in (ITEM_COL, "trackId", "item_id", "item", "id"):
            if key in value:
                return int(value[key])
        raise ValueError(f"Could not find item id in test_dict entry: {value}")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Found an empty list/tuple in test_dict")
        return int(value[0])
    return int(value)


def load_predict_dict(path: str | Path) -> Dict[int, List[int]]:
    with open(path, "r") as f:
        raw = json.load(f)

    out: Dict[int, List[int]] = {}
    for u_raw, values in raw.items():
        user_id = int(u_raw)
        out[user_id] = [_extract_item_id(v) for v in values]
    return out


def build_histories(df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    histories: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for user_id, group in tqdm(df.groupby(USER_COL, sort=False), desc="Building user histories"):
        histories[int(user_id)] = (
            group[TIME_COL].to_numpy(dtype=np.int64),
            group[ITEM_COL].to_numpy(dtype=np.int64),
        )
    return histories


# ---------------------------------------------------------------------------
# Embedding loading, matching your Ex2VecExtendedMLP loader
# ---------------------------------------------------------------------------


def load_item_extension_from_parquet(path: str | Path, id_col: str = "track_id", emb_col: str = "vector") -> Tuple[List[int], np.ndarray]:
    """Load the nested Deezer embedding parquet the same way as Ex2VecExtendedMLP.

    Equivalent to:
        ids = tbl[id_col].to_pylist()
        raw = tbl[emb_col].to_pylist()
        vec = [d["item"] for d in row["list"]]
    """
    if pq is None:  # pragma: no cover
        raise ImportError("pyarrow is required to read the embedding parquet") from _PYARROW_IMPORT_ERROR

    tbl = pq.read_table(path)
    if id_col not in tbl.column_names:
        raise ValueError(f"Embedding parquet has no id column {id_col!r}. Columns: {tbl.column_names}")
    if emb_col not in tbl.column_names:
        raise ValueError(f"Embedding parquet has no vector column {emb_col!r}. Columns: {tbl.column_names}")

    ids = tbl[id_col].to_pylist()
    raw = tbl[emb_col].to_pylist()

    if any(r is None for r in raw):
        bad = [i for i, r in enumerate(raw) if r is None][:5]
        raise ValueError(f"Found None in embedding column at rows (first 5): {bad}")

    embs: List[List[float]] = []
    for row in raw:
        # The exact format used by your model loader.
        if isinstance(row, Mapping) and "list" in row:
            vec = [float(d["item"]) for d in row["list"]]
        # A couple of defensive fallbacks in case pyarrow/pandas returns a slightly different object.
        elif isinstance(row, list) and row and isinstance(row[0], Mapping) and "item" in row[0]:
            vec = [float(d["item"]) for d in row]
        elif isinstance(row, (list, tuple, np.ndarray)):
            vec = [float(x) for x in row]
        else:
            raise ValueError(f"Unsupported embedding row format: {type(row)}; example={row!r}")
        embs.append(vec)

    return [int(x) for x in ids], np.asarray(embs, dtype=np.float32)


def load_mapped_embedding_matrix(
    embedding_parquet: str | Path,
    item_mapping_path: str | Path,
    n_items: int,
    id_col: str = "track_id",
    emb_col: str = "vector",
) -> np.ndarray:
    ids, embs = load_item_extension_from_parquet(embedding_parquet, id_col=id_col, emb_col=emb_col)

    with open(item_mapping_path, "r") as fp:
        mapping = json.load(fp)

    if not ids:
        raise ValueError("No embeddings were loaded")

    matrix = np.zeros((n_items + 1, embs.shape[1]), dtype=np.float32)
    mapped = 0
    skipped = 0
    for original_id, vec in zip(ids, embs):
        key = str(original_id)
        if key not in mapping:
            skipped += 1
            continue
        dense_id = int(mapping[key])
        if 1 <= dense_id <= n_items:
            matrix[dense_id] = vec
            mapped += 1
        else:
            skipped += 1

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[0] = 0.0
    print(f"Loaded pretrained embeddings: mapped={mapped}, skipped={skipped}, shape={matrix.shape}")
    if mapped == 0:
        raise ValueError("No pretrained embeddings could be mapped. Check --item-mapping and embedding ids.")
    return matrix


# ---------------------------------------------------------------------------
# Score/prob helpers
# ---------------------------------------------------------------------------


def fmt_list(values: Sequence[float], float_format: str) -> str:
    return "[" + ", ".join(format(float(v), float_format) for v in values) + "]"


def past_for_time(times: np.ndarray, items: np.ndarray, query_time: int) -> Tuple[np.ndarray, np.ndarray]:
    end = np.searchsorted(times, query_time, side="left")
    return times[:end], items[:end]


def decay_weights(query_time: int | np.ndarray, past_times: np.ndarray, decay: float, cutoff_c: float) -> np.ndarray:
    dt = np.asarray(query_time - past_times, dtype=np.float64) + float(cutoff_c)
    dt = np.maximum(dt, np.finfo(np.float64).tiny)
    return np.power(dt, -float(decay), dtype=np.float64)


def bl_strength_scores(
    query_time: int,
    past_times: np.ndarray,
    past_items: np.ndarray,
    n_items: int,
    decay: float,
    cutoff_c: float,
    transform: str = "strength",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sparse BL scores as (scored_item_ids, scored_values). Unseen items have score 0."""
    if len(past_items) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    valid = (past_items >= 1) & (past_items <= n_items)
    if not np.any(valid):
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    weights = decay_weights(query_time, past_times[valid], decay, cutoff_c)
    strength = np.bincount(past_items[valid], weights=weights, minlength=n_items + 1)[: n_items + 1]
    ids = np.flatnonzero(strength[1:] > 0.0).astype(np.int64) + 1
    vals = strength[ids].astype(np.float64)
    if transform == "log1p":
        vals = np.log1p(vals)
    elif transform != "strength":
        raise ValueError(f"Unknown BL transform: {transform}")
    return ids, vals


def sparse_softmax_prob(
    target_item: int,
    scored_ids: np.ndarray,
    scored_values: np.ndarray,
    n_items: int,
) -> Tuple[float, float]:
    """Softmax probability when unlisted items have score 0."""
    if target_item < 1 or target_item > n_items:
        return 0.0, 0.0

    if len(scored_ids) == 0:
        return 0.0, 1.0 / float(n_items)

    # There should be one score per id, but if a caller passes duplicates, keep the max.
    if len(scored_ids) != len(np.unique(scored_ids)):
        tmp: Dict[int, float] = {}
        for item, val in zip(scored_ids, scored_values):
            item_i = int(item)
            tmp[item_i] = max(tmp.get(item_i, -np.inf), float(val))
        scored_ids = np.asarray(list(tmp.keys()), dtype=np.int64)
        scored_values = np.asarray(list(tmp.values()), dtype=np.float64)

    pos = np.where(scored_ids == int(target_item))[0]
    target_score = float(scored_values[pos[0]]) if len(pos) else 0.0

    max_score = max(0.0, float(np.max(scored_values)))
    n_unscored = max(n_items - len(scored_ids), 0)
    denom = n_unscored * math.exp(0.0 - max_score) + float(np.exp(scored_values - max_score).sum())
    prob = math.exp(target_score - max_score) / denom if denom > 0.0 else 0.0
    return target_score, prob


def recent_unique_item_scores(
    past_items: np.ndarray,
    n_items: int,
    scale: float,
    max_rank: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score most-recent unique items as scale / recency_rank; unseen items score 0."""
    if len(past_items) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    ids: List[int] = []
    seen = set()
    rank_limit = None if max_rank <= 0 else int(max_rank)

    for item in past_items[::-1]:
        item_i = int(item)
        if item_i < 1 or item_i > n_items or item_i in seen:
            continue
        ids.append(item_i)
        seen.add(item_i)
        if rank_limit is not None and len(ids) >= rank_limit:
            break

    if not ids:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    ranks = np.arange(1, len(ids) + 1, dtype=np.float64)
    vals = float(scale) / ranks
    return np.asarray(ids, dtype=np.int64), vals


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def open_grouped_writer(output_path: Path, include_times: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = output_path.open("w", newline="")
    writer = csv.writer(f)
    if include_times:
        writer.writerow(["user", "item", "ts", "score", "prob"])
    else:
        writer.writerow(["user", "item", "score", "prob"])
    return f, writer


def write_grouped_rows(
    writer,
    user_id: int,
    target_items: Sequence[int],
    time_grid: np.ndarray,
    scores_by_item: Mapping[int, Sequence[float]],
    probs_by_item: Mapping[int, Sequence[float]],
    include_times: bool,
    float_format: str,
) -> None:
    ts_str = fmt_list(time_grid.tolist(), ".0f")
    for item in target_items:
        item_i = int(item)
        row = [user_id, item_i]
        if include_times:
            row.append(ts_str)
        row.extend([
            fmt_list(scores_by_item[item_i], float_format),
            fmt_list(probs_by_item[item_i], float_format),
        ])
        writer.writerow(row)


def append_tidy_rows(
    tidy_writer,
    baseline: str,
    user_id: int,
    target_items: Sequence[int],
    time_grid: np.ndarray,
    scores_by_item: Mapping[int, Sequence[float]],
    probs_by_item: Mapping[int, Sequence[float]],
) -> None:
    if tidy_writer is None:
        return
    for item in target_items:
        item_i = int(item)
        scores = scores_by_item[item_i]
        probs = probs_by_item[item_i]
        for step, (ts, score, prob) in enumerate(zip(time_grid, scores, probs)):
            tidy_writer.writerow([baseline, user_id, item_i, step, int(ts), float(score), float(prob)])


# ---------------------------------------------------------------------------
# Baseline curve generators
# ---------------------------------------------------------------------------


def generate_random_curves(
    predict_dict: Mapping[int, Sequence[int]],
    time_grid: np.ndarray,
    n_items: int,
    output_path: Path,
    include_times: bool,
    float_format: str,
    tidy_writer=None,
) -> None:
    f, writer = open_grouped_writer(output_path, include_times)
    try:
        score_curve = [0.0] * len(time_grid)
        prob_curve = [1.0 / float(n_items)] * len(time_grid)
        for user_id, target_items in tqdm(predict_dict.items(), desc="random curves"):
            scores = {int(item): score_curve for item in target_items}
            probs = {int(item): prob_curve for item in target_items}
            write_grouped_rows(writer, user_id, target_items, time_grid, scores, probs, include_times, float_format)
            append_tidy_rows(tidy_writer, "random", user_id, target_items, time_grid, scores, probs)
    finally:
        f.close()


def generate_last_item_curves(
    predict_dict: Mapping[int, Sequence[int]],
    histories: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    time_grid: np.ndarray,
    n_items: int,
    output_path: Path,
    include_times: bool,
    float_format: str,
    score_scale: float,
    max_rank: int,
    tidy_writer=None,
) -> None:
    f, writer = open_grouped_writer(output_path, include_times)
    try:
        for user_id, target_items in tqdm(predict_dict.items(), desc="last_item curves"):
            target_items_i = [int(x) for x in target_items]
            scores_by_item = {item: [] for item in target_items_i}
            probs_by_item = {item: [] for item in target_items_i}
            hist_times, hist_items = histories.get(user_id, (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)))

            for ts in time_grid:
                _, past_items = past_for_time(hist_times, hist_items, int(ts))
                scored_ids, scored_values = recent_unique_item_scores(
                    past_items, n_items=n_items, scale=score_scale, max_rank=max_rank
                )
                for item in target_items_i:
                    score, prob = sparse_softmax_prob(item, scored_ids, scored_values, n_items)
                    scores_by_item[item].append(score)
                    probs_by_item[item].append(prob)

            write_grouped_rows(writer, user_id, target_items_i, time_grid, scores_by_item, probs_by_item, include_times, float_format)
            append_tidy_rows(tidy_writer, "last_item", user_id, target_items_i, time_grid, scores_by_item, probs_by_item)
    finally:
        f.close()


def generate_bl_proxy_curves(
    predict_dict: Mapping[int, Sequence[int]],
    histories: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    time_grid: np.ndarray,
    n_items: int,
    output_path: Path,
    include_times: bool,
    float_format: str,
    decay: float,
    cutoff_c: float,
    transform: str,
    tidy_writer=None,
) -> None:
    f, writer = open_grouped_writer(output_path, include_times)
    try:
        for user_id, target_items in tqdm(predict_dict.items(), desc="bl_proxy curves"):
            target_items_i = [int(x) for x in target_items]
            scores_by_item = {item: [] for item in target_items_i}
            probs_by_item = {item: [] for item in target_items_i}
            hist_times, hist_items = histories.get(user_id, (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)))

            for ts in time_grid:
                past_times, past_items = past_for_time(hist_times, hist_items, int(ts))
                scored_ids, scored_values = bl_strength_scores(
                    int(ts), past_times, past_items, n_items, decay, cutoff_c, transform=transform
                )
                for item in target_items_i:
                    score, prob = sparse_softmax_prob(item, scored_ids, scored_values, n_items)
                    scores_by_item[item].append(score)
                    probs_by_item[item].append(prob)

            write_grouped_rows(writer, user_id, target_items_i, time_grid, scores_by_item, probs_by_item, include_times, float_format)
            append_tidy_rows(tidy_writer, "bl_proxy", user_id, target_items_i, time_grid, scores_by_item, probs_by_item)
    finally:
        f.close()


def _softmax_probs_from_dense_scores(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax for a dense 2D score matrix."""
    max_scores = np.max(scores, axis=1, keepdims=True)
    exps = np.exp(scores - max_scores)
    denom = np.sum(exps, axis=1, keepdims=True)
    return exps / denom


def generate_bl_knn_curves(
    predict_dict: Mapping[int, Sequence[int]],
    histories: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    time_grid: np.ndarray,
    n_items: int,
    item_embeddings: np.ndarray,
    output_path: Path,
    include_times: bool,
    float_format: str,
    decay: float,
    cutoff_c: float,
    score_scale: float,
    distance_kind: str,
    time_batch_size: int,
    missing_score: float,
    tidy_writer=None,
) -> None:
    emb = item_embeddings.astype(np.float32, copy=False)
    if emb.shape[0] < n_items + 1:
        raise ValueError(f"Embedding matrix has {emb.shape[0]} rows, expected at least {n_items + 1}")

    candidate_emb = emb[1 : n_items + 1].astype(np.float64, copy=False)
    candidate_norm_sq = np.sum(candidate_emb * candidate_emb, axis=1)
    valid_candidate = candidate_norm_sq > 0.0
    valid_cols = np.flatnonzero(valid_candidate)
    valid_emb = candidate_emb[valid_candidate]
    valid_norm_sq = candidate_norm_sq[valid_candidate]

    dense_id_to_valid_col = np.full(n_items + 1, -1, dtype=np.int64)
    dense_id_to_valid_col[valid_cols + 1] = np.arange(len(valid_cols), dtype=np.int64)
    n_missing_candidates = n_items - len(valid_cols)

    if len(valid_cols) == 0:
        raise ValueError("No valid non-zero item embeddings found for BL-kNN.")

    f, writer = open_grouped_writer(output_path, include_times)
    try:
        for user_id, target_items in tqdm(predict_dict.items(), desc="bl_knn curves"):
            target_items_i = [int(x) for x in target_items]
            scores_by_item = {item: [] for item in target_items_i}
            probs_by_item = {item: [] for item in target_items_i}
            hist_times, hist_items = histories.get(user_id, (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)))

            hist_valid = (hist_items >= 1) & (hist_items <= n_items)
            hist_times_v = hist_times[hist_valid]
            hist_items_v = hist_items[hist_valid]
            hist_emb = emb[hist_items_v].astype(np.float64, copy=False)

            if len(hist_items_v) == 0:
                # Cold-start user for the whole grid: no meaningful kNN vector, so uniform.
                for item in target_items_i:
                    scores_by_item[item] = [0.0] * len(time_grid)
                    probs_by_item[item] = [1.0 / float(n_items)] * len(time_grid)
                write_grouped_rows(writer, user_id, target_items_i, time_grid, scores_by_item, probs_by_item, include_times, float_format)
                append_tidy_rows(tidy_writer, "bl_knn", user_id, target_items_i, time_grid, scores_by_item, probs_by_item)
                continue

            for start in range(0, len(time_grid), time_batch_size):
                ts_batch = time_grid[start : start + time_batch_size].astype(np.int64)
                # Shape: B x H. Interactions at exactly the query timestamp are excluded.
                valid_past = ts_batch[:, None] > hist_times_v[None, :]
                if not np.any(valid_past):
                    for item in target_items_i:
                        scores_by_item[item].extend([0.0] * len(ts_batch))
                        probs_by_item[item].extend([1.0 / float(n_items)] * len(ts_batch))
                    continue

                dt = ts_batch[:, None].astype(np.float64) - hist_times_v[None, :].astype(np.float64) + float(cutoff_c)
                dt = np.maximum(dt, np.finfo(np.float64).tiny)
                weights = np.where(valid_past, np.power(dt, -float(decay)), 0.0)
                total_weight = weights.sum(axis=1)

                no_history_rows = total_weight <= 0.0
                # Avoid divide-by-zero; rows without history will be overwritten with uniform scores/probs below.
                safe_weight = total_weight.copy()
                safe_weight[no_history_rows] = 1.0
                user_vecs = (weights @ hist_emb) / safe_weight[:, None]

                user_norm_sq = np.sum(user_vecs * user_vecs, axis=1)
                dists = user_norm_sq[:, None] + valid_norm_sq[None, :] - 2.0 * (user_vecs @ valid_emb.T)
                dists = np.maximum(dists, 0.0)
                if distance_kind == "euclidean":
                    dists = np.sqrt(dists)
                elif distance_kind != "squared":
                    raise ValueError(f"Unknown kNN distance kind: {distance_kind}")

                valid_scores = -float(score_scale) * dists

                # Stable denominator over all n_items. Missing embeddings get a very negative score.
                row_max_valid = np.max(valid_scores, axis=1)
                row_max = np.maximum(row_max_valid, float(missing_score))
                denom = np.exp(valid_scores - row_max[:, None]).sum(axis=1)
                if n_missing_candidates > 0:
                    denom += n_missing_candidates * np.exp(float(missing_score) - row_max)

                for local_idx, no_hist in enumerate(no_history_rows):
                    if no_hist:
                        for item in target_items_i:
                            scores_by_item[item].append(0.0)
                            probs_by_item[item].append(1.0 / float(n_items))
                        continue

                    for item in target_items_i:
                        if 1 <= item <= n_items:
                            col = dense_id_to_valid_col[item]
                        else:
                            col = -1
                        if col >= 0:
                            score = float(valid_scores[local_idx, col])
                        else:
                            score = float(missing_score)
                        prob = math.exp(score - float(row_max[local_idx])) / float(denom[local_idx]) if denom[local_idx] > 0 else 0.0
                        scores_by_item[item].append(score)
                        probs_by_item[item].append(prob)

            write_grouped_rows(writer, user_id, target_items_i, time_grid, scores_by_item, probs_by_item, include_times, float_format)
            append_tidy_rows(tidy_writer, "bl_knn", user_id, target_items_i, time_grid, scores_by_item, probs_by_item)
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    baselines = normalize_baselines(args.baselines)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if args.cutoff_c <= 0:
        raise ValueError("--cutoff-c must be positive")
    if args.knn_time_batch_size <= 0:
        raise ValueError("--knn-time-batch-size must be positive")

    df = read_main_dataframe(args.data_parquet)
    n_users = int(df[USER_COL].max())
    n_items = int(df[ITEM_COL].max())
    print(f"Data: n_users={n_users}, n_items={n_items}, interactions={len(df):,}")

    predict_dict = load_predict_dict(args.test_dict)
    n_pairs = sum(len(v) for v in predict_dict.values())
    print(f"Curve targets: users={len(predict_dict):,}, user-item pairs={n_pairs:,}")

    time_grid = np.linspace(args.start_time, args.end_time, args.n_steps).astype(np.int64)
    histories = build_histories(df)

    tidy_file = None
    tidy_writer = None
    if args.save_tidy:
        tidy_path = output_dir / "baseline_curves_tidy.csv"
        tidy_file = tidy_path.open("w", newline="")
        tidy_writer = csv.writer(tidy_file)
        tidy_writer.writerow(["model", "user", "item", "step", "ts", "score", "prob"])

    try:
        if "random" in baselines:
            generate_random_curves(
                predict_dict,
                time_grid,
                n_items,
                output_dir / "random_predictions_for_curves.csv",
                args.include_times,
                args.float_format,
                tidy_writer=tidy_writer,
            )

        if "last_item" in baselines:
            generate_last_item_curves(
                predict_dict,
                histories,
                time_grid,
                n_items,
                output_dir / "last_item_predictions_for_curves.csv",
                args.include_times,
                args.float_format,
                score_scale=args.last_item_score_scale,
                max_rank=args.last_item_max_rank,
                tidy_writer=tidy_writer,
            )

        if "bl_proxy" in baselines:
            generate_bl_proxy_curves(
                predict_dict,
                histories,
                time_grid,
                n_items,
                output_dir / "bl_proxy_predictions_for_curves.csv",
                args.include_times,
                args.float_format,
                decay=args.decay,
                cutoff_c=args.cutoff_c,
                transform=args.bl_transform,
                tidy_writer=tidy_writer,
            )

        if "bl_knn" in baselines:
            if args.embedding_parquet is None or args.item_mapping is None:
                raise ValueError("--embedding-parquet and --item-mapping are required for bl_knn")
            item_embeddings = load_mapped_embedding_matrix(
                args.embedding_parquet,
                args.item_mapping,
                n_items,
                id_col=args.embedding_id_col,
                emb_col=args.embedding_vector_col,
            )
            generate_bl_knn_curves(
                predict_dict,
                histories,
                time_grid,
                n_items,
                item_embeddings,
                output_dir / "bl_knn_predictions_for_curves.csv",
                args.include_times,
                args.float_format,
                decay=args.decay,
                cutoff_c=args.cutoff_c,
                score_scale=args.knn_score_scale,
                distance_kind=args.knn_distance,
                time_batch_size=args.knn_time_batch_size,
                missing_score=args.knn_missing_score,
                tidy_writer=tidy_writer,
            )
    finally:
        if tidy_file is not None:
            tidy_file.close()

    metadata = {
        "data_parquet": str(args.data_parquet),
        "test_dict": str(args.test_dict),
        "output_dir": str(output_dir),
        "baselines": baselines,
        "start_time": int(args.start_time),
        "end_time": int(args.end_time),
        "n_steps": int(args.n_steps),
        "n_users_in_data": n_users,
        "n_items": n_items,
        "n_target_users": len(predict_dict),
        "n_target_user_item_pairs": n_pairs,
        "decay": float(args.decay),
        "cutoff_c": float(args.cutoff_c),
        "bl_transform": args.bl_transform,
        "last_item_score_definition": "score = last_item_score_scale / recency_rank for unique past items; unseen score = 0",
        "last_item_score_scale": float(args.last_item_score_scale),
        "last_item_max_rank": int(args.last_item_max_rank),
        "random_score_definition": "all items have score 0 and probability 1 / n_items",
        "bl_proxy_score_definition": "sum_j (t - tj + cutoff_c)^(-decay) over past interactions with the same item, optionally log1p-transformed",
        "bl_knn_score_definition": "negative distance from BL-weighted user vector to item embedding; softmax over candidate item scores",
        "knn_score_scale": float(args.knn_score_scale),
        "knn_distance": args.knn_distance,
        "include_times": bool(args.include_times),
        "save_tidy": bool(args.save_tidy),
    }
    with (output_dir / "baseline_curve_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Wrote baseline curve files to: {output_dir}")


if __name__ == "__main__":
    main()

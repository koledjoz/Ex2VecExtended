#!/usr/bin/env python3
"""
Generate top-k prediction CSVs for the thesis baselines:
  1) random prediction
  2) last-item prediction
  3) base-level-activation proxy
  4) base-level-activation-informed kNN

The output format matches predict_extended_mlp.py:
  userId,trackId,ts,pred_1,...,pred_k

Example:
python predict_baselines.py \
  --data-parquet ../../../../sorted_data.parquet \
  --test-dict ../../../../split_data/test/test_dict.json \
  --output-dir ./predictions/baselines \
  --embedding-parquet /home/koledjoz/Ex2VecExtended/split_data/track_embeddings.parquet \
  --item-mapping /home/koledjoz/Ex2VecExtended/configs/models/item_mapping.json \
  --top-k 50 \
  --baselines all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


USER_COL = "user_id"
ITEM_COL = "track_id"
TIME_COL = "ts"


# ---------------------------------------------------------------------------
# Loading / preparation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions for random, last-item, BL-proxy, and BL-kNN baselines."
    )
    parser.add_argument("--data-parquet", required=True, help="Path to sorted_data.parquet")
    parser.add_argument("--test-dict", required=True, help="Path to split_data/test/test_dict.json")
    parser.add_argument("--output-dir", required=True, help="Directory where CSV files will be written")
    parser.add_argument("--top-k", type=int, default=50, help="Number of predictions per row")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for random/tie-fill predictions")

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["all"],
        choices=["all", "random", "last_item", "bl_proxy", "bl_knn"],
        help="Baselines to run. Use 'all' for every baseline.",
    )

    parser.add_argument(
        "--embedding-parquet",
        default=None,
        help="Path to pretrained item embeddings. Required for bl_knn.",
    )
    parser.add_argument(
        "--item-mapping",
        default=None,
        help="Optional JSON mapping between embedding item ids and dense track_id ids.",
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.5,
        help="ACT-R/base-level decay parameter d. Default is 0.5.",
    )
    parser.add_argument(
        "--cutoff-c",
        type=float,
        default=1.0,
        help="Small positive cutoff added to time differences: (t - tj + c)^(-d).",
    )
    parser.add_argument(
        "--use-log-bl",
        action="store_true",
        help=(
            "Use ACT-R log base-level activation as the weighting signal inside BL-kNN as well. "
            "The BL-proxy baseline always uses ACT-R B = log(sum((t - tj + c)^(-d))). "
            "For BL-kNN, the default remains non-log positive strengths because weighted averages "
            "cannot directly use negative log activations."
        ),
    )
    parser.add_argument(
        "--csv-chunk-size",
        type=int,
        default=100_000,
        help="Number of prediction rows buffered before appending to CSV.",
    )
    return parser.parse_args()


def normalize_baseline_names(names: Sequence[str]) -> List[str]:
    if "all" in names:
        return ["random", "last_item", "bl_proxy", "bl_knn"]
    # Preserve order but remove duplicates.
    seen = set()
    out: List[str] = []
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

    # Make sure histories are time sorted even if the parquet name already suggests it.
    df = df[[USER_COL, ITEM_COL, TIME_COL]].copy()
    df[USER_COL] = df[USER_COL].astype(np.int64)
    df[ITEM_COL] = df[ITEM_COL].astype(np.int64)
    df = df.sort_values([USER_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    return df


def _extract_test_rows_from_json(test_obj: dict) -> Tuple[pd.DataFrame, bool]:
    """
    Supports the format used in the example file:
        {"user_id": [track_id, track_id, ...]}

    Also supports a more explicit format if you later create it:
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
    Recreates the evaluation dataframe style from predict_extended_mlp.py:
    load test_dict.json, build user_id/track_id pairs, and inner-merge with df.

    If the JSON also contains timestamps, the merge uses user_id/track_id/ts.
    """
    with open(test_dict_path, "r") as f:
        test_obj = json.load(f)

    filter_df, has_ts = _extract_test_rows_from_json(test_obj)
    merge_cols = [USER_COL, ITEM_COL, TIME_COL] if has_ts else [USER_COL, ITEM_COL]
    eval_df = df.merge(filter_df, on=merge_cols, how="inner")
    eval_df = eval_df[[USER_COL, ITEM_COL, TIME_COL]].copy()
    eval_df = eval_df.sort_values([USER_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    return eval_df


def build_histories(df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return user_id -> (times_sorted, items_sorted)."""
    histories: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for user_id, group in tqdm(df.groupby(USER_COL, sort=False), desc="Building user histories"):
        histories[int(user_id)] = (
            group[TIME_COL].to_numpy(),
            group[ITEM_COL].to_numpy(dtype=np.int64),
        )
    return histories


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
        candidates = np.setdiff1d(np.arange(1, item_count + 1, dtype=np.int64), np.fromiter(seen, dtype=np.int64), assume_unique=False)
        need = top_k - len(out)
        if need > len(candidates):
            raise ValueError(f"Cannot create {top_k} distinct predictions with only {item_count} items.")
        out.extend(rng.choice(candidates, size=need, replace=False).astype(np.int64).tolist())

    return np.asarray(out, dtype=np.int64)


def past_for_query(
    histories: Dict[int, Tuple[np.ndarray, np.ndarray]],
    user_id: int,
    query_time,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return this user's interactions with timestamp strictly smaller than query_time."""
    hist = histories.get(int(user_id))
    if hist is None:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    times, items = hist
    end = np.searchsorted(times, query_time, side="left")
    return times[:end], items[:end]


def decay_weights(query_time, past_times: np.ndarray, decay: float, cutoff_c: float) -> np.ndarray:
    if len(past_times) == 0:
        return np.asarray([], dtype=np.float64)
    dt = np.asarray(query_time - past_times, dtype=np.float64) + float(cutoff_c)
    # Strictly-positive because past_times are selected with side='left'. This guard protects against bad data.
    dt = np.maximum(dt, np.finfo(np.float64).tiny)
    return np.power(dt, -float(decay)).astype(np.float64, copy=False)


def bl_item_scores(
    query_time,
    past_times: np.ndarray,
    past_items: np.ndarray,
    item_count: int,
    decay: float,
    cutoff_c: float,
    use_log: bool = False,
) -> np.ndarray:
    """
    Compute item-level ACT-R base-level evidence for a user at a query time.

    First compute the summed recency/frequency strength for each item:
        strength_i = sum_j (query_time - t_ij + cutoff_c)^(-decay)

    If use_log=True, return the ACT-R base-level activation used by the BL
    proxy baseline:
        B_i = ln(strength_i)

    Items with no previous user-item interaction have no defined activation for
    this proxy ranking, so they are represented as -inf when use_log=True and
    later used only as random fill if top_k is larger than the number of seen
    items. With use_log=False, the non-negative strength is returned; this is
    useful for BL-kNN weighted averages.
    """
    scores = np.zeros(item_count + 1, dtype=np.float64)

    if len(past_items) == 0:
        if use_log:
            return np.full(item_count + 1, -np.inf, dtype=np.float64)
        return scores

    valid = (past_items >= 1) & (past_items <= item_count)
    if not np.any(valid):
        if use_log:
            return np.full(item_count + 1, -np.inf, dtype=np.float64)
        return scores

    weights = decay_weights(query_time, past_times[valid], decay, cutoff_c)
    scores += np.bincount(past_items[valid], weights=weights, minlength=item_count + 1)[: item_count + 1]
    scores[0] = 0.0

    if use_log:
        seen = scores > 0
        log_scores = np.full_like(scores, -np.inf, dtype=np.float64)
        log_scores[seen] = np.log(scores[seen])
        log_scores[0] = -np.inf
        return log_scores

    return scores

def top_items_from_scores(
    scores: np.ndarray,
    rng: np.random.Generator,
    item_count: int,
    top_k: int,
) -> np.ndarray:
    """Top-k scored items, then random fill among all other items.

    For non-log BL strengths, unseen items have score 0 and seen items have
    positive scores. For ACT-R log BL activation, unseen items are -inf and seen
    items can be negative, so we rank all finite seen items.
    """
    candidate_scores = scores[1 : item_count + 1]
    if np.isneginf(candidate_scores).any():
        scored = np.flatnonzero(np.isfinite(candidate_scores)) + 1
    else:
        scored = np.flatnonzero(np.isfinite(candidate_scores) & (candidate_scores > 0.0)) + 1

    if len(scored) == 0:
        return fill_random([], rng, item_count, top_k)

    # Random jitter only for tie-breaking among equal scores.
    jitter = rng.random(len(scored)) * 1e-12
    ordered = scored[np.argsort(-(scores[scored] + jitter))]
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
# Baseline generators
# ---------------------------------------------------------------------------

def iter_random_predictions(
    eval_df: pd.DataFrame,
    item_count: int,
    top_k: int,
    rng: np.random.Generator,
) -> Iterable[List[int | float]]:
    for row in eval_df.itertuples(index=False):
        user_id = int(getattr(row, USER_COL))
        true_item = int(getattr(row, ITEM_COL))
        ts = getattr(row, TIME_COL)
        preds = rng.choice(np.arange(1, item_count + 1, dtype=np.int64), size=top_k, replace=False)
        yield [user_id, true_item, ts] + preds.astype(np.int64).tolist()


def iter_last_item_predictions(
    eval_df: pd.DataFrame,
    histories: Dict[int, Tuple[np.ndarray, np.ndarray]],
    item_count: int,
    top_k: int,
    rng: np.random.Generator,
) -> Iterable[List[int | float]]:
    for row in eval_df.itertuples(index=False):
        user_id = int(getattr(row, USER_COL))
        true_item = int(getattr(row, ITEM_COL))
        ts = getattr(row, TIME_COL)

        _, past_items = past_for_query(histories, user_id, ts)
        recent_unique: List[int] = []
        seen = set()
        for item in past_items[::-1]:
            item_int = int(item)
            if item_int >= 1 and item_int not in seen:
                recent_unique.append(item_int)
                seen.add(item_int)
                if len(recent_unique) >= top_k:
                    break

        preds = fill_random(recent_unique, rng, item_count, top_k)
        yield [user_id, true_item, ts] + preds.tolist()


def iter_bl_proxy_predictions(
    eval_df: pd.DataFrame,
    histories: Dict[int, Tuple[np.ndarray, np.ndarray]],
    item_count: int,
    top_k: int,
    rng: np.random.Generator,
    decay: float,
    cutoff_c: float,
) -> Iterable[List[int | float]]:
    for row in eval_df.itertuples(index=False):
        user_id = int(getattr(row, USER_COL))
        true_item = int(getattr(row, ITEM_COL))
        ts = getattr(row, TIME_COL)

        past_times, past_items = past_for_query(histories, user_id, ts)
        # BL proxy ranks by the ACT-R base-level activation:
        #     B_i = ln(sum_j (t - t_ij + c)^(-d))
        # not by the raw non-log strength.
        scores = bl_item_scores(ts, past_times, past_items, item_count, decay, cutoff_c, use_log=True)
        preds = top_items_from_scores(scores, rng, item_count, top_k)
        yield [user_id, true_item, ts] + preds.tolist()


# ---------------------------------------------------------------------------
# Embedding loading and BL-kNN
# ---------------------------------------------------------------------------

def _read_mapping_json(mapping_path: Optional[str | Path]) -> Optional[dict]:
    if mapping_path is None:
        return None
    with open(mapping_path, "r") as f:
        obj = json.load(f)

    # Common wrappers.
    if isinstance(obj, dict):
        for key in ("item_mapping", "track_mapping", "track_to_idx", "item_to_idx", "mapping"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    return None


def _mapping_to_dense_ids(source_ids: np.ndarray, mapping: Optional[dict], n_items: int) -> Optional[np.ndarray]:
    if mapping is None:
        return None

    # Try source_id -> dense_id.
    dense: List[int] = []
    ok = True
    for sid in source_ids:
        key = str(int(sid)) if isinstance(sid, (int, np.integer, float, np.floating)) and float(sid).is_integer() else str(sid)
        if key not in mapping:
            ok = False
            break
        value = mapping[key]
        try:
            dense_id = int(value)
        except Exception:
            ok = False
            break
        if not (1 <= dense_id <= n_items):
            ok = False
            break
        dense.append(dense_id)
    if ok:
        return np.asarray(dense, dtype=np.int64)

    # Try dense_id -> source_id by reversing.
    rev = {}
    for k, v in mapping.items():
        try:
            source_value = int(v)
            dense_key = int(k)
        except Exception:
            continue
        rev[source_value] = dense_key

    dense = []
    ok = True
    for sid in source_ids:
        sid_int = int(sid)
        if sid_int not in rev or not (1 <= rev[sid_int] <= n_items):
            ok = False
            break
        dense.append(rev[sid_int])
    if ok:
        return np.asarray(dense, dtype=np.int64)

    return None


def load_item_extension_from_parquet(
    path: str | Path,
    id_col: str = ITEM_COL,
    emb_col: str = "vector",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Deezer pretrained track embeddings using the exact nested parquet layout
    used by Ex2VecExtendedMLP.load_item_extension_from_parquet.

    Expected columns:
      - track_id: original item id from the Deezer embedding file
      - vector: nested value like {"list": [{"item": ...}, ...]}

    Returns:
      ids:  shape (num_embedding_rows,)
      embs: shape (num_embedding_rows, embedding_dim), float32
    """
    tbl = pq.read_table(path)
    columns = set(tbl.column_names)
    if id_col not in columns:
        raise ValueError(
            f"Embedding parquet is missing id column {id_col!r}. "
            f"Available columns: {tbl.column_names}"
        )
    if emb_col not in columns:
        raise ValueError(
            f"Embedding parquet is missing embedding column {emb_col!r}. "
            f"Available columns: {tbl.column_names}"
        )

    ids = np.asarray(tbl[id_col].to_pylist())
    raw = tbl[emb_col].to_pylist()

    if any(r is None for r in raw):
        bad = [i for i, r in enumerate(raw) if r is None][:5]
        raise ValueError(f"Found None in embedding column at rows (first 5): {bad}")

    embs: List[List[float]] = []
    for row_idx, row in enumerate(raw):
        # The Deezer file used by the model stores vectors as:
        #   {"list": [{"item": 0.123}, {"item": ...}, ...]}
        if isinstance(row, dict) and "list" in row:
            vec = [float(d["item"]) for d in row["list"]]
        # Be permissive for already-flattened parquet conversions.
        elif isinstance(row, (list, tuple, np.ndarray)):
            vec = [float(x) for x in row]
        else:
            raise ValueError(
                f"Unsupported embedding value at row {row_idx}: {type(row).__name__}. "
                "Expected {'list': [{'item': ...}, ...]} or a flat list."
            )
        embs.append(vec)

    embs_np = np.asarray(embs, dtype=np.float32)
    if embs_np.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embs_np.shape}")
    return ids, embs_np


def load_item_embedding_matrix(
    embedding_parquet: str | Path,
    n_items: int,
    item_mapping_path: Optional[str | Path] = None,
) -> np.ndarray:
    """
    Load Deezer pretrained embeddings into dense rows used by this project.

    This mirrors Ex2VecExtendedMLP:
      ids, embs = load_item_extension_from_parquet(pretrained_path)
      for each original item_id in ids:
          mapped_id = json_mapping[str(item_id)]
          embedding_item_extension.weight[mapped_id] = embs[i]

    The returned matrix has shape (n_items + 1, emb_dim), with row 0 kept as
    all zeros for padding and rows 1..n_items corresponding to dense track_id.
    """
    ids, embs = load_item_extension_from_parquet(embedding_parquet)
    mapping = _read_mapping_json(item_mapping_path)

    matrix = np.zeros((n_items + 1, embs.shape[1]), dtype=np.float32)
    mapped_count = 0

    if mapping is not None:
        # Exact behavior of the model: mapping keys are original ids as strings,
        # values are the dense ids used in sorted_data.parquet.
        for original_id, emb in zip(ids, embs):
            key = str(int(original_id)) if isinstance(original_id, (int, np.integer, float, np.floating)) and float(original_id).is_integer() else str(original_id)
            if key not in mapping:
                continue
            dense_id = int(mapping[key])
            if 1 <= dense_id <= n_items:
                matrix[dense_id] = emb
                mapped_count += 1
    else:
        # Fallback: only use direct ids when they already match dense track_id.
        try:
            dense_ids = ids.astype(np.int64)
        except Exception as exc:
            raise ValueError(
                "--item-mapping is required because embedding ids are not numeric dense ids."
            ) from exc

        valid = (dense_ids >= 1) & (dense_ids <= n_items)
        if not np.any(valid):
            raise ValueError(
                "Could not align embeddings without --item-mapping. "
                "Pass the same item_mapping.json used by Ex2VecExtendedMLP."
            )
        matrix[dense_ids[valid]] = embs[valid]
        mapped_count = int(valid.sum())

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    matrix[0] = 0.0

    if mapped_count == 0:
        raise ValueError(
            "No pretrained embeddings were mapped to dense track ids. "
            "Check that --embedding-parquet and --item-mapping correspond to the same dataset."
        )

    print(
        f"Loaded pretrained embeddings: mapped {mapped_count:,}/{len(ids):,} rows "
        f"into dense matrix shape {matrix.shape}"
    )
    return matrix


def iter_bl_knn_predictions(
    eval_df: pd.DataFrame,
    histories: Dict[int, Tuple[np.ndarray, np.ndarray]],
    item_count: int,
    top_k: int,
    rng: np.random.Generator,
    decay: float,
    cutoff_c: float,
    use_log_bl: bool,
    item_embeddings: np.ndarray,
) -> Iterable[List[int | float]]:
    emb = item_embeddings.astype(np.float32, copy=False)
    if emb.shape[0] < item_count + 1:
        raise ValueError(f"Embedding matrix has {emb.shape[0]} rows, expected at least {item_count + 1}")

    candidate_emb = emb[1 : item_count + 1]
    candidate_ids = np.arange(1, item_count + 1, dtype=np.int64)

    # Exclude missing embeddings (all-zero rows) from kNN ranking; random fill can still use any item.
    valid_emb = np.linalg.norm(candidate_emb, axis=1) > 0
    valid_candidate_emb = candidate_emb[valid_emb]
    valid_candidate_ids = candidate_ids[valid_emb]

    if len(valid_candidate_ids) == 0:
        raise ValueError("No valid non-zero item embeddings found for BL-kNN.")

    for row in eval_df.itertuples(index=False):
        user_id = int(getattr(row, USER_COL))
        true_item = int(getattr(row, ITEM_COL))
        ts = getattr(row, TIME_COL)

        past_times, past_items = past_for_query(histories, user_id, ts)
        if len(past_items) == 0:
            preds = fill_random([], rng, item_count, top_k)
            yield [user_id, true_item, ts] + preds.tolist()
            continue

        scores = bl_item_scores(ts, past_times, past_items, item_count, decay, cutoff_c, use_log=use_log_bl)

        # BL-kNN needs non-negative weights. If log scores were requested, shift seen scores above zero.
        seen = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        if len(seen) == 0 and use_log_bl:
            seen = np.flatnonzero(np.isfinite(scores) & (scores > -np.inf))
            if len(seen) > 0:
                min_seen = np.min(scores[seen])
                scores[seen] = scores[seen] - min_seen + 1e-12

        item_weights = scores[1 : item_count + 1].astype(np.float64)
        item_weights[~np.isfinite(item_weights)] = 0.0
        item_weights[item_weights < 0] = 0.0

        total_weight = float(item_weights.sum())
        if total_weight <= 0.0:
            preds = fill_random([], rng, item_count, top_k)
            yield [user_id, true_item, ts] + preds.tolist()
            continue

        user_vec = (item_weights[:, None] * candidate_emb.astype(np.float64)).sum(axis=0) / total_weight
        # Euclidean nearest neighbours in the pretrained embedding space.
        distances = np.sum((valid_candidate_emb.astype(np.float64) - user_vec[None, :]) ** 2, axis=1)

        k_eff = min(top_k, len(valid_candidate_ids))
        # argpartition is faster than a full sort; we sort the selected candidates afterwards.
        nearest_part = np.argpartition(distances, kth=k_eff - 1)[:k_eff]
        nearest_order = nearest_part[np.argsort(distances[nearest_part])]
        nearest_ids = valid_candidate_ids[nearest_order]
        preds = fill_random(nearest_ids, rng, item_count, top_k)
        yield [user_id, true_item, ts] + preds.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    baselines = normalize_baseline_names(args.baselines)

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

    histories = None
    if any(b in baselines for b in ("last_item", "bl_proxy", "bl_knn")):
        histories = build_histories(df)

    for baseline in baselines:
        rng = np.random.default_rng(args.seed)  # same seed per baseline for reproducibility
        out_path = output_dir / f"{baseline}_output.csv"

        if baseline == "random":
            rows = iter_random_predictions(eval_df, item_count, args.top_k, rng)
            write_prediction_csv(out_path, rows, args.top_k, args.csv_chunk_size, len(eval_df), desc="random")

        elif baseline == "last_item":
            assert histories is not None
            rows = iter_last_item_predictions(eval_df, histories, item_count, args.top_k, rng)
            write_prediction_csv(out_path, rows, args.top_k, args.csv_chunk_size, len(eval_df), desc="last_item")

        elif baseline == "bl_proxy":
            assert histories is not None
            rows = iter_bl_proxy_predictions(
                eval_df,
                histories,
                item_count,
                args.top_k,
                rng,
                args.decay,
                args.cutoff_c,
            )
            write_prediction_csv(out_path, rows, args.top_k, args.csv_chunk_size, len(eval_df), desc="bl_proxy")

        elif baseline == "bl_knn":
            assert histories is not None
            if args.embedding_parquet is None:
                raise ValueError("--embedding-parquet is required for the bl_knn baseline")
            item_embeddings = load_item_embedding_matrix(args.embedding_parquet, item_count, args.item_mapping)
            rows = iter_bl_knn_predictions(
                eval_df,
                histories,
                item_count,
                args.top_k,
                rng,
                args.decay,
                args.cutoff_c,
                args.use_log_bl,
                item_embeddings,
            )
            write_prediction_csv(out_path, rows, args.top_k, args.csv_chunk_size, len(eval_df), desc="bl_knn")

        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

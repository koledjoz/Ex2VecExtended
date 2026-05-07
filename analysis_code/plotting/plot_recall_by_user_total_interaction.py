#!/usr/bin/env python3
r"""
Create thesis-ready Recall@K plots from user-total-interaction evaluation files.

This script is meant for outputs from evaluate_recall_by_user_total_interaction.py:

    recall_by_user_total_interaction_tidy.csv
    recall_by_user_total_interaction_binned_tidy.csv

It can plot either:
  - user_interaction_number: 1, 2, 3, ... in the user's full history
  - history_length_before: 0, 1, 2, ... previous interactions available before prediction

If the input file contains user_interaction_number and you request
history_length_before, the script converts it by subtracting 1 from the x-axis.
So you do not need to rerun the evaluator just to switch between these two views.

It also supports plot-time binning with --bin-size, even if the evaluator did
not create binned output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXACT_X_COL = "user_total_interaction_value"
PAIR_X_COL = "interaction_number"
COMMON_REQUIRED = {"model", "k", "recall_at_k", "n_rows"}

GROUP_USER = "user_interaction_number"
GROUP_HISTORY = "history_length_before"
GROUP_PAIR = "interaction_number"

DEFAULT_MODEL_LABELS: Dict[str, str] = {
    "random": "Random",
    "last_item": "Last item",
    "bl_proxy": "BL proxy",
    "bl_knn": "BL-kNN",
    "original": "Ex2Vec",
    "extendedBase": "Ex2Vec-Sim",
    "extendedbase": "Ex2Vec-Sim",
    "extendeddouble": "Ex2Vec-PSim",
    "extendedmlp": "Ex2Vec-PMLP",
    "extendedmlploss": "Ex2Vec-PMLP-Loss",
}

DEFAULT_MODEL_ORDER = [
    "random",
    "last_item",
    "bl_proxy",
    "bl_knn",
    "original",
    "extendedBase",
    "extendedbase",
    "extendeddouble",
    "extendedmlp",
    "extendedmlploss",
]

DEFAULT_BASELINE_MODELS = (
    "random",
    "last_item",
    "bl_proxy",
    "bl_knn",
    "Random",
    "Last item",
    "Last-item",
    "BL proxy",
    "BL Proxy",
    "BL-kNN",
    "BL kNN",
    "BL-KNN",
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_csv_arg(value: Optional[str]) -> List[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(fmt.strip().lower().lstrip(".") for fmt in value.split(",") if fmt.strip())
    if not formats:
        raise ValueError("At least one output format must be provided.")
    return formats


def parse_rename_arg(value: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in parse_csv_arg(value):
        if "=" not in part:
            raise ValueError(f"Invalid --rename entry {part!r}. Use raw=Label.")
        raw, label = part.split("=", 1)
        raw = raw.strip()
        label = label.strip()
        if not raw or not label:
            raise ValueError(f"Invalid --rename entry {part!r}. Use raw=Label.")
        mapping[raw] = label
    return mapping


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def read_label_map(path: Optional[str], rename: Optional[str]) -> Dict[str, str]:
    labels = dict(DEFAULT_MODEL_LABELS)
    if path:
        with open(path, "r", encoding="utf-8") as fp:
            labels.update(json.load(fp))
    labels.update(parse_rename_arg(rename))
    return labels


def label_for_model(raw_model: str, label_map: Dict[str, str]) -> str:
    return label_map.get(raw_model, raw_model.replace("_", " "))


def load_style_module(style_path: str | Path):
    style_path = Path(style_path)
    if not style_path.exists():
        raise FileNotFoundError(style_path)
    spec = importlib.util.spec_from_file_location("thesis_plot_style", style_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import style module from {style_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordered_display_models(raw_models: Sequence[str], label_map: Dict[str, str], model_order_arg: Optional[str]) -> List[str]:
    labels_in_data = list(dict.fromkeys(label_for_model(m, label_map) for m in raw_models))

    explicit = parse_csv_arg(model_order_arg)
    ordered: List[str] = []
    for name in explicit:
        candidates = [name, label_for_model(name, label_map)]
        for candidate in candidates:
            if candidate in labels_in_data and candidate not in ordered:
                ordered.append(candidate)

    for raw in DEFAULT_MODEL_ORDER:
        label = label_for_model(raw, label_map)
        if label in labels_in_data and label not in ordered:
            ordered.append(label)

    rest = sorted([m for m in labels_in_data if m not in ordered], key=natural_key)
    return ordered + rest


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def read_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = COMMON_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()

    # Native output from evaluate_recall_by_user_total_interaction.py.
    if "group_by" in df.columns:
        df["group_by"] = df["group_by"].astype(str)
    # Compatibility with evaluate_recall_by_interaction.py.
    elif PAIR_X_COL in df.columns:
        df["group_by"] = GROUP_PAIR
        df[EXACT_X_COL] = df[PAIR_X_COL]
    else:
        raise ValueError(
            "Input must contain either 'group_by' + 'user_total_interaction_value' "
            "or an 'interaction_number' column."
        )

    has_exact = EXACT_X_COL in df.columns
    has_binned = {"bin_start", "bin_end", "bin_label"}.issubset(df.columns)

    if not has_exact and not has_binned:
        raise ValueError(
            "Input must contain either 'user_total_interaction_value' for exact data "
            "or 'bin_start', 'bin_end', 'bin_label' for binned data."
        )

    df["model"] = df["model"].astype(str)
    df["k"] = df["k"].astype(int)
    df["recall_at_k"] = pd.to_numeric(df["recall_at_k"], errors="raise")
    df["n_rows"] = pd.to_numeric(df["n_rows"], errors="raise").astype(int)

    if has_exact:
        df[EXACT_X_COL] = pd.to_numeric(df[EXACT_X_COL], errors="raise").astype(int)

    if has_binned:
        df["bin_start"] = pd.to_numeric(df["bin_start"], errors="raise").astype(int)
        df["bin_end"] = pd.to_numeric(df["bin_end"], errors="raise").astype(int)
        df["bin_label"] = df["bin_label"].astype(str)

    if "n_users" in df.columns:
        df["n_users"] = pd.to_numeric(df["n_users"], errors="coerce")

    return df


def add_derived_group_if_needed(df: pd.DataFrame, requested: Optional[str]) -> pd.DataFrame:
    """Allow switching between user_interaction_number and history_length_before.

    user_interaction_number is 1-based. history_length_before is exactly one
    smaller, because the target interaction itself is not in the history.
    """
    if requested is None:
        return df

    available = set(df["group_by"].dropna().astype(str).unique().tolist())
    if requested in available:
        return df

    conversions = {
        (GROUP_USER, GROUP_HISTORY): -1,
        (GROUP_HISTORY, GROUP_USER): 1,
    }

    for (source, target), shift in conversions.items():
        if requested == target and source in available:
            converted = df[df["group_by"] == source].copy()
            converted["group_by"] = target

            if EXACT_X_COL in converted.columns:
                converted[EXACT_X_COL] = converted[EXACT_X_COL].astype(int) + shift
                if target == GROUP_HISTORY:
                    converted = converted[converted[EXACT_X_COL] >= 0].copy()

            if {"bin_start", "bin_end", "bin_label"}.issubset(converted.columns):
                converted["bin_start"] = converted["bin_start"].astype(int) + shift
                converted["bin_end"] = converted["bin_end"].astype(int) + shift
                if target == GROUP_HISTORY:
                    converted = converted[converted["bin_end"] >= 0].copy()
                    converted["bin_start"] = converted["bin_start"].clip(lower=0)
                converted["bin_label"] = (
                    converted["bin_start"].astype(str) + "-" + converted["bin_end"].astype(str)
                )

            print(
                f"Input contains {source!r}; deriving requested --group-by {target!r} "
                f"by shifting the x-axis by {shift:+d}."
            )
            return pd.concat([df, converted], ignore_index=True)

    return df


def determine_group_by(df: pd.DataFrame, requested: Optional[str]) -> str:
    available = sorted(df["group_by"].dropna().astype(str).unique().tolist())

    if requested is not None:
        if requested in available:
            return requested
        if available == [GROUP_PAIR] and requested in {GROUP_USER, GROUP_HISTORY}:
            print(
                f"Warning: input contains repeated user-item interaction numbers, not {requested!r}; "
                f"plotting {GROUP_PAIR!r} instead."
            )
            return GROUP_PAIR
        raise ValueError(f"Requested --group-by {requested!r}, but input contains {available}.")

    if len(available) != 1:
        raise ValueError(f"Input contains multiple group_by values {available}. Pass --group-by.")
    return available[0]


def bin_exact(df: pd.DataFrame, bin_size: int, group_by: str) -> pd.DataFrame:
    if bin_size <= 1 or EXACT_X_COL not in df.columns:
        return df

    out = df.copy()
    values = out[EXACT_X_COL].astype(np.int64)

    if group_by == GROUP_HISTORY:
        bin_start = (values // bin_size) * bin_size
        bin_end = bin_start + bin_size - 1
    else:
        bin_start = ((values - 1) // bin_size) * bin_size + 1
        bin_end = bin_start + bin_size - 1

    out["bin_start"] = bin_start.astype(int)
    out["bin_end"] = bin_end.astype(int)
    out["bin_label"] = out["bin_start"].astype(str) + "-" + out["bin_end"].astype(str)
    out["hit_sum"] = out["recall_at_k"] * out["n_rows"]

    group_cols = ["model", "k", "group_by", "bin_start", "bin_end", "bin_label"]
    agg_spec = {
        "hit_sum": ("hit_sum", "sum"),
        "n_rows": ("n_rows", "sum"),
    }
    if "n_users" in out.columns:
        # Cannot reconstruct exact distinct users from already aggregated exact rows.
        # The recall is correct; n_users is only contextual.
        agg_spec["n_users"] = ("n_users", "max")
    if "input_file" in out.columns:
        agg_spec["input_file"] = ("input_file", "first")

    binned = out.groupby(group_cols, sort=True).agg(**agg_spec).reset_index()
    binned["recall_at_k"] = binned["hit_sum"] / binned["n_rows"]
    binned["accuracy_at_k"] = binned["recall_at_k"]
    binned = binned.drop(columns=["hit_sum"])
    return binned


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def default_x_label(group_by: str) -> str:
    if group_by == GROUP_HISTORY:
        return "Number of previous user interactions"
    if group_by == GROUP_USER:
        return "User interaction number"
    return "Repeated user-item interaction number"


def safe_group_name(group_by: str) -> str:
    return {
        GROUP_HISTORY: "history_length_before",
        GROUP_USER: "user_interaction_number",
        GROUP_PAIR: "interaction_number",
    }.get(group_by, re.sub(r"[^A-Za-z0-9_]+", "_", group_by))


def output_stem(args: argparse.Namespace, selected_k: int, group_by: str, is_binned: bool) -> str:
    if args.output_name:
        return args.output_name

    stem = f"recall_by_{safe_group_name(group_by)}_at_{selected_k}"
    if is_binned:
        stem += "_binned"
        if args.bin_size and args.bin_size > 1:
            stem += f"_{int(args.bin_size)}"
    if args.min_value is not None:
        stem += f"_min_value_{int(args.min_value)}"
    if args.max_value is not None:
        stem += f"_max_value_{int(args.max_value)}"
    return stem


def prepare_plot_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, int, str, bool]:
    df = read_input(args.input)
    df = add_derived_group_if_needed(df, args.group_by)
    label_map = read_label_map(args.label_map_json, args.rename)

    group_by = determine_group_by(df, args.group_by)
    df = df[df["group_by"] == group_by].copy()

    available_ks = sorted(df["k"].unique().tolist())
    if not available_ks:
        raise ValueError("No k values found.")
    selected_k = args.k if args.k is not None else max(available_ks)
    if selected_k not in available_ks:
        raise ValueError(f"Requested --k {selected_k}, but available k values are {available_ks}.")

    df = df[df["k"] == selected_k].copy()

    input_is_binned = {"bin_start", "bin_end", "bin_label"}.issubset(df.columns) and EXACT_X_COL not in df.columns
    if args.bin_size and args.bin_size > 1 and not input_is_binned:
        df = bin_exact(df, int(args.bin_size), group_by)
    elif args.bin_size and args.bin_size > 1 and input_is_binned:
        print("Input is already binned; --bin-size is ignored.")

    is_binned = {"bin_start", "bin_end", "bin_label"}.issubset(df.columns) and EXACT_X_COL not in df.columns
    if is_binned:
        df["x_value"] = (df["bin_start"].astype(float) + df["bin_end"].astype(float)) / 2.0
        df["filter_value"] = df["bin_start"].astype(float)
    else:
        df["x_value"] = df[EXACT_X_COL].astype(float)
        df["filter_value"] = df["x_value"]

    # Raw/display model handling.
    df["model_raw"] = df["model"].astype(str)
    df["model_label"] = df["model_raw"].map(lambda m: label_for_model(m, label_map))

    include_raw = set(parse_csv_arg(args.models)) if args.models else None
    include_display = set(parse_csv_arg(args.display_models)) if args.display_models else None
    exclude_raw = set(parse_csv_arg(args.exclude_models)) if args.exclude_models else set()

    if include_raw is not None:
        df = df[df["model_raw"].isin(include_raw)].copy()
    if include_display is not None:
        df = df[df["model_label"].isin(include_display)].copy()
    if exclude_raw:
        df = df[~df["model_raw"].isin(exclude_raw)].copy()

    if args.min_rows > 1:
        df = df[df["n_rows"] >= args.min_rows].copy()
    if args.min_value is not None:
        df = df[df["filter_value"] >= args.min_value].copy()
    if args.max_value is not None:
        df = df[df["filter_value"] <= args.max_value].copy()

    if df.empty:
        raise ValueError("No data left after applying filters.")

    baseline_names = set(parse_csv_arg(args.baseline_models))
    df["is_baseline"] = df["model_raw"].isin(baseline_names) | df["model_label"].isin(baseline_names)

    multiplier = 100.0 if args.percent else 1.0
    df["recall"] = df["recall_at_k"] * multiplier

    # Combine possible duplicate labels using n_rows-weighted recall.
    records: List[dict] = []
    for (model_label, x_value), group in df.groupby(["model_label", "x_value"], sort=False):
        weights = group["n_rows"].to_numpy(dtype=float)
        values = group["recall"].to_numpy(dtype=float)
        recall = float(np.average(values, weights=weights)) if weights.sum() > 0 else float(np.mean(values))

        rec = {
            "model": str(model_label),
            "x_value": float(x_value),
            "recall": recall,
            "is_baseline": bool(group["is_baseline"].any()),
            "n_rows": int(group["n_rows"].sum()),
            "n_users": int(group["n_users"].max()) if "n_users" in group.columns and group["n_users"].notna().any() else np.nan,
            "model_raw_values": ",".join(sorted(set(group["model_raw"].astype(str).tolist()))),
            "group_by": group_by,
            "is_binned": bool(is_binned),
        }
        if is_binned:
            rec["bin_start"] = int(group["bin_start"].min())
            rec["bin_end"] = int(group["bin_end"].max())
            rec["bin_label"] = str(group["bin_label"].iloc[0])
        else:
            rec[EXACT_X_COL] = int(round(float(x_value)))
        records.append(rec)

    out = pd.DataFrame(records)
    if args.smooth_window > 1:
        window = int(args.smooth_window)
        out = out.sort_values(["model", "x_value"]).copy()
        out["recall_raw"] = out["recall"]
        out["recall"] = out.groupby("model", group_keys=False)["recall"].apply(
            lambda s: s.rolling(window=window, center=True, min_periods=1).mean()
        )

    return out.sort_values(["model", "x_value"]).reset_index(drop=True), selected_k, group_by, is_binned


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_and_save(plot_df: pd.DataFrame, selected_k: int, group_by: str, is_binned: bool, args: argparse.Namespace) -> List[Path]:
    style = load_style_module(args.style_file)
    style.setup_mpl(use_latex_pgf=not args.no_latex_pgf, grayscale=args.grayscale)

    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=style.fig_size(fraction=args.width_fraction, aspect=args.height_ratio))

    # Model order uses display labels.
    label_map = read_label_map(args.label_map_json, args.rename)
    # Recover a useful order by matching raw names from metadata when available.
    raw_for_order: List[str] = []
    for raw_values in plot_df.get("model_raw_values", pd.Series(dtype=str)).astype(str):
        raw_for_order.extend([x for x in raw_values.split(",") if x])
    ordered = ordered_display_models(raw_for_order or plot_df["model"].tolist(), label_map, args.model_order)
    ordered += [m for m in sorted(plot_df["model"].unique().tolist(), key=natural_key) if m not in ordered]

    marker = "o" if args.markers else None
    for model in ordered:
        group = plot_df[plot_df["model"] == model].sort_values("x_value")
        if group.empty:
            continue

        is_baseline = bool(group["is_baseline"].any())
        ax.plot(
            group["x_value"],
            group["recall"],
            linestyle="--" if is_baseline else "-",
            linewidth=1.2 if is_baseline else 1.6,
            alpha=0.75 if is_baseline else 1.0,
            marker=marker,
            label=model,
        )

    ax.set_xlabel(args.xlabel if args.xlabel is not None else default_x_label(group_by))
    if args.ylabel:
        ax.set_ylabel(args.ylabel)
    else:
        ax.set_ylabel(f"Recall@{selected_k}" + (" [\\%]" if args.percent else ""))

    if args.title:
        ax.set_title(args.title)

    if args.percent:
        ax.set_ylim(0, min(100.0, max(1.0, plot_df["recall"].max() * 1.10)))
    else:
        ax.set_ylim(0, min(1.0, max(0.01, plot_df["recall"].max() * 1.10)))

    if args.log_x:
        ax.set_xscale("log")

    if is_binned and "bin_label" in plot_df.columns:
        ticks = plot_df[["x_value", "bin_label"]].drop_duplicates().sort_values("x_value")
        if len(ticks) <= 14:
            ax.set_xticks(ticks["x_value"].to_numpy())
            ax.set_xticklabels(ticks["bin_label"].tolist(), rotation=30, ha="right")

    if args.legend_outside:
        ax.legend(
            title=args.legend_title,
            ncol=args.legend_columns,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
        )
    else:
        ax.legend(title=args.legend_title, ncol=args.legend_columns, loc="best")

    stem = output_stem(args, selected_k, group_by, is_binned)
    written = style.savefig(fig, stem, output_dir=output_dir, formats=parse_formats(args.formats), close=True)
    return written


def save_plot_data(plot_df: pd.DataFrame, selected_k: int, group_by: str, is_binned: bool, args: argparse.Namespace, written: Sequence[Path]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args, selected_k, group_by, is_binned)

    data_path = output_dir / f"{stem}_plot_data.csv"
    plot_df.to_csv(data_path, index=False)

    metadata = {
        "input": str(args.input),
        "selected_k": int(selected_k),
        "metric": "recall_at_k",
        "x_axis": group_by,
        "x_column_in_plot_data": "x_value",
        "is_binned": bool(is_binned),
        "bin_size": int(args.bin_size) if args.bin_size else 0,
        "percent": bool(args.percent),
        "min_rows": int(args.min_rows),
        "min_value": args.min_value,
        "max_value": args.max_value,
        "smooth_window": int(args.smooth_window),
        "models": sorted(plot_df["model"].unique().tolist()),
        "figures": [str(path) for path in written],
        "plot_data": str(data_path),
    }
    with open(output_dir / f"{stem}_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Recall@K by total user interaction/history length using thesis_plot_style.py."
    )
    parser.add_argument("--input", required=True, help="Path to recall_by_user_total_interaction_tidy.csv or binned output.")
    parser.add_argument("--style-file", required=True, help="Path to thesis_plot_style.py.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures and plot data will be saved.")
    parser.add_argument("--k", type=int, default=None, help="Top-k value to plot. Defaults to largest available k.")
    parser.add_argument(
        "--group-by",
        choices=[GROUP_USER, GROUP_HISTORY, GROUP_PAIR],
        default=None,
        help=(
            "Which x-axis to plot. If the file has user_interaction_number and you request "
            "history_length_before, the script subtracts one from the x-axis automatically."
        ),
    )
    parser.add_argument("--formats", default="pgf,pdf", help="Comma-separated formats, e.g. pgf,pdf or pdf,png.")
    parser.add_argument("--percent", action="store_true", help="Plot Recall as percentages instead of fractions.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--ylabel", default=None, help="Optional y-axis label.")
    parser.add_argument("--xlabel", default=None, help="Optional x-axis label.")
    parser.add_argument("--legend-title", default=None, help="Optional legend title.")
    parser.add_argument("--min-rows", type=int, default=1, help="Only plot x-values with at least this many rows per model.")
    parser.add_argument("--min-value", type=int, default=None, help="Only plot x-values/bins starting at least at this value.")
    parser.add_argument("--max-value", type=int, default=None, help="Only plot x-values/bins starting at most at this value. Filename includes it.")
    parser.add_argument("--bin-size", type=int, default=0, help="Create bins from exact input before plotting, e.g. --bin-size 50.")
    parser.add_argument("--log-x", action="store_true", help="Use logarithmic x-axis.")
    parser.add_argument("--models", default=None, help="Comma-separated raw model names to include.")
    parser.add_argument("--display-models", default=None, help="Comma-separated display labels to include after label mapping.")
    parser.add_argument("--exclude-models", default=None, help="Comma-separated raw model names to exclude.")
    parser.add_argument("--label-map-json", default=None, help="Optional JSON raw-name to display-label mapping.")
    parser.add_argument("--rename", default=None, help="Legacy raw=label comma-separated display renames.")
    parser.add_argument("--model-order", default=None, help="Comma-separated raw names or display labels for model order.")
    parser.add_argument("--baseline-models", default=",".join(DEFAULT_BASELINE_MODELS), help="Comma-separated raw/display models drawn dashed.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Centered rolling mean window. Default 1 = no smoothing.")
    parser.add_argument("--markers", action="store_true", help="Draw markers at each x-value.")
    parser.add_argument("--legend-columns", type=int, default=1, help="Number of legend columns.")
    parser.add_argument("--legend-outside", action="store_true", help="Place legend outside to the right.")
    parser.add_argument("--width-fraction", type=float, default=1.0, help="Fraction of thesis text width.")
    parser.add_argument("--height-ratio", type=float, default=0.68, help="Figure height / width ratio.")
    parser.add_argument("--no-latex-pgf", action="store_true", help="Disable PGF/XeLaTeX rcParams.")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale style cycle from thesis_plot_style.py.")
    parser.add_argument("--output-name", default=None, help="Custom output filename stem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_df, selected_k, group_by, is_binned = prepare_plot_data(args)
    written = plot_and_save(plot_df, selected_k, group_by, is_binned, args)
    save_plot_data(plot_df, selected_k, group_by, is_binned, args, written)

    stem = output_stem(args, selected_k, group_by, is_binned)
    print("Saved:")
    for path in written:
        print(f"  {path}")
    print(f"  {Path(args.output_dir) / f'{stem}_plot_data.csv'}")
    print(f"  {Path(args.output_dir) / f'{stem}_metadata.json'}")


if __name__ == "__main__":
    main()

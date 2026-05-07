#!/usr/bin/env python3
r"""
Create thesis-ready Recall@K by interaction-number plots.

This script plots the output created by evaluate_recall_by_interaction.py,
especially recall_by_interaction_tidy.csv, whose expected columns are:

    model, k, interaction_number, recall_at_k, accuracy_at_k,
    n_rows, n_users, input_file

The x-axis is the interaction number for the same user-item pair:
    1 = first time the user interacted with that item
    2 = second time the user interacted with that item
    ...

The legend uses thesis/display names by default instead of raw working names
such as extendedmlploss or bl_proxy. Baselines are drawn with dashed lines by
default so they are visually separated from Ex2Vec-like models.

Example:
    python plot_recall_by_interaction.py \
      --input ./predictions/interaction_eval/recall_by_interaction_tidy.csv \
      --style-file ./thesis_plot_style.py \
      --output-dir ./figures/interaction_recall \
      --k 50 \
      --formats pgf,pdf \
      --percent \
      --max-interaction 20

LaTeX:
    \begin{figure}
        \centering
        \input{figures/interaction_recall/recall_by_interaction_at_50_max_interaction_20.pgf}
        \caption{Recall@50 by repeated user-item interaction number.}
        \label{fig:recall-by-interaction}
    \end{figure}
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


REQUIRED_COLUMNS = {"model", "k", "interaction_number", "recall_at_k", "n_rows"}

# Default display labels. These match the naming style used in plot_recall_at_k.py,
# with an additional explicit label for extendedmlp so both MLP variants can be
# shown if both are present in the same file.
DEFAULT_MODEL_LABELS = {
    "random": "Random",
    "last_item": "Last item",
    'most_popular_past': 'Most popular',
    "bl_proxy": "BL proxy",
    "bl_knn": "BL-kNN",
    "original": "Ex2Vec",
    "extendedBase": "Ex2Vec-Sim",
    "extendedbase": "Ex2Vec-Sim",
    "extendeddouble": "Ex2Vec-PSim",
    "extendedmlploss": "Ex2Vec-PMLP",
}

DEFAULT_MODEL_ORDER = [
    "random",
    "last_item",
    'most_popular_past',
    "bl_proxy",
    "bl_knn",
    "original",
    "extendedBase",
    "extendedbase",
    "extendeddouble",
    "extendedmlp",
    "extendedmlploss",
]

# Baseline detection is based on raw names and display labels. This means dashed
# baseline lines still work even after display-name mapping.
DEFAULT_BASELINE_MODELS = (
    "random",
    "last_item",
    "bl_proxy",
    "bl_knn",
    'most_popular_past',
    "Random",
    "Last item",
    "Last-item",
    "BL proxy",
    "BL Proxy",
    "BL-kNN",
    "BL kNN",
    "BL-KNN",
)

MODEL_COLORS = {
    "random": "#7f7f7f",
    "last_item": "#8c564b",
    "most_popular_past": "#9467bd",
    "bl_proxy": "#1f77b4",
    "bl_knn": "#17becf",
    "original": "#2ca02c",
    "extendedBase": "#ff7f0e",
    "extendedbase": "#ff7f0e",
    "extendeddouble": "#d62728",
    "extendedmlp": "#e377c2",
    "extendedmlploss": "#bcbd22",
}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_csv_arg(value: Optional[str]) -> List[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_rename_arg(value: Optional[str]) -> Dict[str, str]:
    """Parse old=new,old2=new2. Kept for backwards compatibility."""
    mapping: Dict[str, str] = {}
    for part in parse_csv_arg(value):
        if "=" not in part:
            raise ValueError(
                f"Invalid --rename entry {part!r}. Use old=new, for example extendedmlp=Ex2Vec-PMLP."
            )
        old, new = part.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid --rename entry {part!r}.")
        mapping[old] = new
    return mapping


def parse_order_arg(value: Optional[str]) -> List[str]:
    return parse_csv_arg(value)


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(fmt.strip().lower().lstrip(".") for fmt in value.split(",") if fmt.strip())
    if not formats:
        raise ValueError("At least one output format must be provided.")
    return formats


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def read_label_map(path: Optional[str], rename_arg: Optional[str]) -> Dict[str, str]:
    """Build raw-model-name -> display-label mapping.

    Defaults are applied first. A JSON file can override/add labels. The legacy
    --rename option is applied last, so older commands still work.
    """
    labels = dict(DEFAULT_MODEL_LABELS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fp:
            labels.update(json.load(fp))
    labels.update(parse_rename_arg(rename_arg))
    return labels


def label_for_model(raw_model: str, label_map: Dict[str, str]) -> str:
    return label_map.get(raw_model, raw_model.replace("_", " "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Recall@K by user-item interaction number using thesis_plot_style.py."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to recall_by_interaction_tidy.csv produced by evaluate_recall_by_interaction.py.",
    )
    parser.add_argument("--style-file", required=True, help="Path to thesis_plot_style.py.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures and plot data will be saved.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Top-k value to plot. Defaults to the largest k available in the input file.",
    )
    parser.add_argument(
        "--formats",
        default="pgf,pdf",
        help="Comma-separated output formats, for example pgf,pdf or pdf,png. Default: pgf,pdf.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Plot Recall as percentages instead of fractions in [0, 1].",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. By default no title is used, which is usually best for thesis figures.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional y-axis label. Defaults to 'Recall@K' or 'Recall@K [%%]'.",
    )
    parser.add_argument(
        "--xlabel",
        default="Interaction number",
        help="X-axis label. Default: 'Interaction number'.",
    )
    parser.add_argument("--legend-title", default=None, help="Optional legend title. Default: no title.")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Only plot interaction numbers with at least this many evaluated rows per model. Default: 1.",
    )
    parser.add_argument(
        "--max-interaction",
        type=int,
        default=None,
        help="Only plot interaction numbers up to this value. Useful because the tail can be noisy.",
    )
    parser.add_argument("--log-x", action="store_true", help="Use logarithmic x-axis.")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated list of raw model names to include, e.g. extendedmlploss,bl_proxy. "
            "Use --display-models to filter by display labels instead."
        ),
    )
    parser.add_argument(
        "--display-models",
        default=None,
        help=(
            "Comma-separated list of display model labels to include after label mapping, "
            "e.g. Ex2Vec-PMLP,BL proxy."
        ),
    )
    parser.add_argument(
        "--exclude-models",
        default=None,
        help="Comma-separated list of raw model names to exclude before display-name mapping.",
    )
    parser.add_argument(
        "--label-map-json",
        default=None,
        help="Optional JSON file mapping raw model names to display labels. Overrides built-in labels.",
    )
    parser.add_argument(
        "--rename",
        default=None,
        help=(
            "Legacy alias for display labels: comma-separated raw=label pairs. "
            "Applied after --label-map-json."
        ),
    )
    parser.add_argument(
        "--model-order",
        default=None,
        help=(
            "Comma-separated model order. You may use either raw names or display labels. "
            "Models not listed are appended in the default thesis order and then alphabetically."
        ),
    )
    parser.add_argument(
        "--baseline-models",
        default=",".join(DEFAULT_BASELINE_MODELS),
        help=(
            "Comma-separated raw or display model names to draw with dashed lines. "
            "Default covers random,last_item,bl_proxy,bl_knn and common display names."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional centered rolling mean window over interaction_number within each model. Default 1 means no smoothing.",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="Draw markers at each interaction number. Default is line-only.",
    )
    parser.add_argument("--legend-columns", type=int, default=1, help="Number of columns in the legend.")
    parser.add_argument(
        "--legend-outside",
        action="store_true",
        help="Place legend to the right of the plot. Default: use best location inside the axes.",
    )
    parser.add_argument(
        "--width-fraction",
        type=float,
        default=1.0,
        help="Fraction of thesis text width used for the figure. Default: 1.0.",
    )
    parser.add_argument(
        "--height-ratio",
        type=float,
        default=0.68,
        help="Figure height / width ratio. Default: 0.68.",
    )
    parser.add_argument(
        "--no-latex-pgf",
        action="store_true",
        help="Disable PGF/XeLaTeX rcParams. Useful on machines without XeLaTeX; prefer pdf,png formats.",
    )
    parser.add_argument("--grayscale", action="store_true", help="Use the grayscale cycle from thesis_plot_style.py.")
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output filename stem. Default: recall_by_interaction_at_<k>, or "
            "recall_by_interaction_at_<k>_max_interaction_<N> when --max-interaction is passed."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Style loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def read_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["k"] = df["k"].astype(int)
    df["interaction_number"] = df["interaction_number"].astype(int)
    df["recall_at_k"] = df["recall_at_k"].astype(float)
    df["n_rows"] = df["n_rows"].astype(int)
    return df


def select_and_prepare_data(args: argparse.Namespace, label_map: Dict[str, str]) -> Tuple[pd.DataFrame, int]:
    df = read_input(args.input)

    available_ks = sorted(df["k"].dropna().astype(int).unique().tolist())
    if not available_ks:
        raise ValueError("No k values found in input file.")

    selected_k = args.k if args.k is not None else max(available_ks)
    if selected_k not in available_ks:
        raise ValueError(f"Requested --k {selected_k}, but available values are {available_ks}.")

    df = df[df["k"] == selected_k].copy()
    df["model_raw"] = df["model"].astype(str)

    include_models = set(parse_csv_arg(args.models)) if args.models else None
    exclude_models = set(parse_csv_arg(args.exclude_models)) if args.exclude_models else set()

    if include_models is not None:
        df = df[df["model_raw"].isin(include_models)].copy()
    if exclude_models:
        df = df[~df["model_raw"].isin(exclude_models)].copy()

    df["model_label"] = df["model_raw"].map(lambda m: label_for_model(m, label_map))

    display_models = set(parse_csv_arg(args.display_models)) if args.display_models else None
    if display_models is not None:
        df = df[df["model_label"].isin(display_models)].copy()

    if args.min_rows > 1:
        df = df[df["n_rows"] >= args.min_rows].copy()
    if args.max_interaction is not None:
        df = df[df["interaction_number"] <= args.max_interaction].copy()

    if df.empty:
        raise ValueError("No data left after applying filters.")

    baseline_names = set(parse_csv_arg(args.baseline_models))
    df["is_baseline"] = df["model_raw"].isin(baseline_names) | df["model_label"].isin(baseline_names)

    y_multiplier = 100.0 if args.percent else 1.0
    df["recall"] = df["recall_at_k"] * y_multiplier

    # If multiple raw models map to the same display label, keep them separate
    # by including the raw model in the grouping. This avoids accidentally
    # averaging different experimental variants that share similar labels.
    def weighted_recall(group: pd.DataFrame) -> float:
        weights = group["n_rows"].to_numpy(dtype=float)
        vals = group["recall"].to_numpy(dtype=float)
        if weights.sum() <= 0:
            return float(np.mean(vals))
        return float(np.average(vals, weights=weights))

    grouped_records: List[dict] = []
    for (model_raw, model_label, interaction_number), group in df.groupby(
        ["model_raw", "model_label", "interaction_number"], sort=False
    ):
        grouped_records.append(
            {
                "model_raw": str(model_raw),
                "model_label": str(model_label),
                "interaction_number": int(interaction_number),
                "recall": weighted_recall(group),
                "is_baseline": bool(group["is_baseline"].any()),
                "n_rows": int(group["n_rows"].sum()),
                "n_users": (
                    int(group["n_users"].sum())
                    if "n_users" in group.columns and group["n_users"].notna().any()
                    else np.nan
                ),
            }
        )
    grouped = pd.DataFrame(grouped_records)

    if args.smooth_window > 1:
        window = int(args.smooth_window)
        if window < 2:
            raise ValueError("--smooth-window must be at least 1.")
        grouped = grouped.sort_values(["model_raw", "interaction_number"])
        grouped["recall_raw"] = grouped["recall"]
        grouped["recall"] = grouped.groupby("model_raw", group_keys=False)["recall"].apply(
            lambda s: s.rolling(window=window, center=True, min_periods=1).mean()
        )

    return grouped.sort_values(["model_raw", "interaction_number"]).reset_index(drop=True), selected_k


def ordered_raw_models(plot_df: pd.DataFrame, model_order_arg: Optional[str], label_map: Dict[str, str]) -> List[str]:
    observed = list(dict.fromkeys(plot_df["model_raw"].tolist()))
    observed_set = set(observed)

    # Allow --model-order to contain either raw names or display labels.
    label_to_raws: Dict[str, List[str]] = {}
    for raw in observed:
        label_to_raws.setdefault(label_for_model(raw, label_map), []).append(raw)

    ordered: List[str] = []
    for entry in parse_order_arg(model_order_arg):
        if entry in observed_set and entry not in ordered:
            ordered.append(entry)
        elif entry in label_to_raws:
            for raw in label_to_raws[entry]:
                if raw not in ordered:
                    ordered.append(raw)

    # Then use the same default raw order as the recall@K plotting script.
    for raw in DEFAULT_MODEL_ORDER:
        if raw in observed_set and raw not in ordered:
            ordered.append(raw)

    # Remaining unknown models alphabetically by their display label.
    remaining = [raw for raw in observed if raw not in ordered]
    remaining = sorted(remaining, key=lambda raw: natural_key(label_for_model(raw, label_map)))
    return ordered + remaining


def default_output_stem(selected_k: int, args: argparse.Namespace) -> str:
    """Build a filename stem that records important plot filters."""
    stem = f"recall_by_interaction_at_{selected_k}"
    if args.max_interaction is not None:
        stem += f"_max_interaction_{int(args.max_interaction)}"
    return stem


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_plot(
    plot_df: pd.DataFrame,
    selected_k: int,
    args: argparse.Namespace,
    style,
    label_map: Dict[str, str],
) -> List[Path]:
    # Configure Matplotlib before importing pyplot.
    style.setup_mpl(use_latex_pgf=not args.no_latex_pgf, grayscale=args.grayscale)

    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=style.fig_size(fraction=args.width_fraction, aspect=args.height_ratio))

    models = ordered_raw_models(plot_df, args.model_order, label_map)
    marker = "o" if args.markers else None

    for model_raw in models:
        group = plot_df[plot_df["model_raw"] == model_raw].sort_values("interaction_number")
        if group.empty:
            continue

        model_label = str(group["model_label"].iloc[0])
        is_baseline = bool(group["is_baseline"].any()) if "is_baseline" in group.columns else False
        linestyle = "--" if is_baseline else "-"
        linewidth = 1.2 if is_baseline else 1.6
        alpha = 0.75 if is_baseline else 1.0

        ax.plot(
            group["interaction_number"],
            group["recall"],
            color=MODEL_COLORS.get(model_raw),
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=model_label,
        )

    ax.set_xlabel(args.xlabel)
    if args.ylabel is not None:
        ylabel = args.ylabel
    else:
        ylabel = f"Recall@{selected_k}" + (" [\\%]" if args.percent else "")
    ax.set_ylabel(ylabel)

    if args.title:
        ax.set_title(args.title)

    if args.percent:
        ax.set_ylim(0, min(100, max(1.0, plot_df["recall"].max() * 1.10)))
    else:
        ax.set_ylim(0, min(1, max(0.01, plot_df["recall"].max() * 1.10)))

    if args.log_x:
        ax.set_xscale("log")

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

    stem = args.output_name or default_output_stem(selected_k, args)
    formats = parse_formats(args.formats)
    written = style.savefig(fig, stem, output_dir=output_dir, formats=formats, close=True)
    return written


def save_plot_data(
    plot_df: pd.DataFrame,
    selected_k: int,
    args: argparse.Namespace,
    written_figures: Sequence[Path],
    label_map: Dict[str, str],
) -> None:
    output_dir = Path(args.output_dir)
    stem = args.output_name or default_output_stem(selected_k, args)

    ordered = ordered_raw_models(plot_df, args.model_order, label_map)
    out = plot_df.copy()
    out["model_order"] = out["model_raw"].map({m: i for i, m in enumerate(ordered)})
    out = out.sort_values(["model_order", "interaction_number"])

    data_path = output_dir / f"{stem}_plot_data.csv"
    out.to_csv(data_path, index=False)

    metadata = {
        "input": str(args.input),
        "selected_k": selected_k,
        "metric": "recall_at_k",
        "x_axis": "interaction_number",
        "percent": bool(args.percent),
        "min_rows": args.min_rows,
        "max_interaction": args.max_interaction,
        "smooth_window": args.smooth_window,
        "baseline_models": parse_csv_arg(args.baseline_models),
        "model_labels": {raw: label_for_model(raw, label_map) for raw in sorted(out["model_raw"].unique())},
        "models_raw": ordered,
        "models_display": [label_for_model(raw, label_map) for raw in ordered],
        "figures": [str(path) for path in written_figures],
        "plot_data": str(data_path),
    }
    with open(output_dir / f"{stem}_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def main() -> None:
    args = parse_args()
    style = load_style_module(args.style_file)
    label_map = read_label_map(args.label_map_json, args.rename)

    plot_df, selected_k = select_and_prepare_data(args, label_map)
    written = save_plot(plot_df, selected_k, args, style, label_map)
    save_plot_data(plot_df, selected_k, args, written, label_map)

    print("Saved:")
    for path in written:
        print(f"  {path}")
    stem = args.output_name or default_output_stem(selected_k, args)
    print(f"  {Path(args.output_dir) / f'{stem}_plot_data.csv'}")
    print(f"  {Path(args.output_dir) / f'{stem}_metadata.json'}")


if __name__ == "__main__":
    main()

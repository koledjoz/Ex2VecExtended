#!/usr/bin/env python3
r"""
Create thesis-ready Recall@K visualisations from recall_tidy.csv.

This script intentionally plots only the micro-averaged recall column
(`recall_micro`) and labels it simply as Recall, so the final figure can be
used in the thesis without exposing implementation-specific averaging wording.

Input format expected from evaluate_recall_at_k.py:
    model,k,recall_micro,recall_macro_user,n_rows,n_users,input_file

Typical use:
    python plot_recall_at_k.py \
      --input ./predictions/recall_eval/recall_tidy.csv \
      --style-file ./thesis_plot_style.py \
      --output-dir ./figures/recall \
      --formats pgf,pdf \
      --percent

Then in LaTeX:
    \begin{figure}
        \centering
        \input{figures/recall/recall_at_k.pgf}
        \caption{Recall@K comparison.}
        \label{fig:recall-at-k}
    \end{figure}

Notes:
- PGF output lets LaTeX typeset labels in the same font as the thesis.
- If your machine does not have XeLaTeX installed, use:
      --no-latex-pgf --formats pdf,png
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


RECALL_COLUMN = "recall_micro"

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
# CLI / utilities
# ---------------------------------------------------------------------------

def parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(fmt.strip().lower().lstrip(".") for fmt in value.split(",") if fmt.strip())
    if not formats:
        raise ValueError("At least one output format must be provided.")
    return formats


def load_style(style_file: Optional[str]):
    """Load thesis_plot_style.py dynamically or use a small fallback style."""
    if style_file is None:
        return None

    path = Path(style_file)
    if not path.exists():
        raise FileNotFoundError(f"Style file not found: {path}")

    spec = importlib.util.spec_from_file_location("thesis_plot_style", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import style file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_matplotlib(style, args) -> None:
    """Apply the provided thesis style before importing pyplot."""
    if style is not None:
        if args.use_pgf_backend and "pgf" in args.formats:
            style.use_pgf_backend()
        style.setup_mpl(
            use_latex_pgf=not args.no_latex_pgf,
            grayscale=args.grayscale,
            grid=not args.no_grid,
        )
    else:
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
                "savefig.dpi": 300,
                "figure.dpi": 120,
                "axes.grid": not args.no_grid,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.frameon": False,
                "font.size": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8.5,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )


def figure_size(style, width_fraction: float, height: Optional[float] = None):
    if style is not None:
        return style.fig_size(fraction=width_fraction, height_in=height)
    width = 5.1378 * width_fraction
    return (width, height if height is not None else width * 0.618)


def save_figure(fig, style, output_dir: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if style is not None:
        return style.savefig(fig, stem, output_dir=output_dir, formats=formats, close=True)

    import matplotlib.pyplot as plt

    written: List[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(path)
    plt.close(fig)
    return written


def read_label_map(path: Optional[str]) -> Dict[str, str]:
    labels = dict(DEFAULT_MODEL_LABELS)
    if path is None:
        return labels
    with open(path, "r", encoding="utf-8") as fp:
        labels.update(json.load(fp))
    return labels


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prepare_dataframe(
    input_path: Path,
    models: Sequence[str],
    exclude_models: Sequence[str],
    max_k: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"model", "k", RECALL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["k"] = df["k"].astype(int)
    df["recall"] = pd.to_numeric(df[RECALL_COLUMN], errors="raise")

    if models:
        df = df[df["model"].isin(set(models))]
    if exclude_models:
        df = df[~df["model"].isin(set(exclude_models))]
    if max_k is not None:
        df = df[df["k"] <= max_k]

    if df.empty:
        raise ValueError("No rows left after applying filters.")

    return df.sort_values(["model", "k"]).reset_index(drop=True)


def ordered_models(df: pd.DataFrame, explicit_order: Sequence[str]) -> List[str]:
    existing = list(dict.fromkeys(df["model"].tolist()))
    if explicit_order:
        ordered = [m for m in explicit_order if m in existing]
        ordered += [m for m in existing if m not in ordered]
        return ordered

    ordered = [m for m in DEFAULT_MODEL_ORDER if m in existing]
    ordered += sorted([m for m in existing if m not in ordered], key=natural_key)
    return ordered


def label_for_model(model: str, label_map: Dict[str, str]) -> str:
    return label_map.get(model, model.replace("_", " "))


def y_values(series: pd.Series, percent: bool) -> pd.Series:
    return series * 100.0 if percent else series


def y_label(percent: bool) -> str:
    return r"Recall [\%]" if percent else "Recall"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_legend(ax, location: str, ncol: int) -> None:
    if location == "below":
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=ncol,
            columnspacing=1.0,
            handlelength=1.9,
        )
    elif location == "outside":
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    elif location == "inside":
        ax.legend(loc="best")
    else:
        raise ValueError("legend location must be one of: below, outside, inside")


def nice_xticks(k_values: np.ndarray) -> List[int]:
    k_values = np.asarray(sorted(np.unique(k_values)), dtype=int)
    if len(k_values) <= 12:
        return k_values.tolist()
    candidates = np.array([1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000])
    ticks = [int(x) for x in candidates if k_values.min() <= x <= k_values.max()]
    if int(k_values.max()) not in ticks:
        ticks.append(int(k_values.max()))
    if int(k_values.min()) not in ticks:
        ticks.insert(0, int(k_values.min()))
    return ticks


def plot_recall_curve(
    df: pd.DataFrame,
    model_order: Sequence[str],
    label_map: Dict[str, str],
    style,
    args,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figure_size(style, args.width_fraction, args.height))


    baseline_set = set(args.baseline_models)
    highlight_set = set(args.highlight_models)

    for model in model_order:
        sub = df[df["model"] == model].sort_values("k")
        if sub.empty:
            continue

        is_baseline = model in baseline_set
        is_highlight = model in highlight_set

        linestyle = "--" if is_baseline else "-"
        linewidth = 1.2 if is_baseline else 1.6
        alpha = 0.75 if is_baseline else 1.0
        marker = "o" if is_highlight else None

        if is_highlight:
            linewidth = 2.0

        ax.plot(
            sub["k"],
            y_values(sub["recall"], args.percent),
            color=MODEL_COLORS.get(model),
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
            markevery=max(1, len(sub) // 10),
            label=label_for_model(model, label_map),
        )

    ax.set_xlabel("K")
    ax.set_ylabel(y_label(args.percent))
    ax.set_xlim(left=max(1, int(df["k"].min())), right=int(df["k"].max()))
    ax.set_ylim(bottom=0)
    ax.set_xticks(nice_xticks(df["k"].to_numpy()))

    if args.title:
        ax.set_title(args.title)

    add_legend(ax, args.legend, args.legend_columns)
    return fig


def plot_bar_at_k(
    df: pd.DataFrame,
    k: int,
    model_order: Sequence[str],
    label_map: Dict[str, str],
    style,
    args,
):
    import matplotlib.pyplot as plt

    k_df = df[df["k"] == k].copy()
    if k_df.empty:
        available = sorted(df["k"].unique().tolist())
        raise ValueError(f"Requested --bar-k {k}, but available k values are: {available}")

    k_df["_order"] = k_df["model"].map({m: i for i, m in enumerate(model_order)})
    k_df = k_df.sort_values("_order")

    fig, ax = plt.subplots(figsize=figure_size(style, args.width_fraction, args.height))
    x = np.arange(len(k_df))
    bar_colors = [MODEL_COLORS.get(m) for m in k_df["model"]]

    ax.bar(
        x,
        y_values(k_df["recall"], args.percent),
        color=bar_colors,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_model(m, label_map) for m in k_df["model"]], rotation=35, ha="right")
    ax.set_ylabel(y_label(args.percent))
    ax.set_xlabel("Model")
    ax.set_title(f"Recall@{k}")
    ax.set_ylim(bottom=0)
    return fig


def save_plot_data(df: pd.DataFrame, output_dir: Path, model_order: Sequence[str], label_map: Dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["model_label"] = out["model"].map(lambda m: label_for_model(m, label_map))
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(model_order)})
    columns = ["model_order", "model", "model_label", "k", "recall"]
    columns += [c for c in ["n_rows", "n_users", "input_file"] if c in out.columns]
    out[columns].sort_values(["model_order", "k"]).to_csv(output_dir / "recall_plot_data.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create thesis-ready Recall@K plots from recall_tidy.csv. Uses recall_micro and labels it as Recall."
    )
    parser.add_argument("--input", required=True, help="Path to recall_tidy.csv.")
    parser.add_argument("--style-file", default=None, help="Path to thesis_plot_style.py.")
    parser.add_argument("--output-dir", required=True, help="Directory where figures will be written.")
    parser.add_argument(
        "--formats",
        default="pgf,pdf",
        help="Comma-separated output formats, e.g. pgf,pdf or pdf,png.",
    )
    parser.add_argument("--models", default=None, help="Optional comma-separated model subset to include.")
    parser.add_argument("--exclude-models", default=None, help="Optional comma-separated models to exclude.")
    parser.add_argument("--model-order", default=None, help="Optional comma-separated model order.")
    parser.add_argument("--label-map-json", default=None, help="Optional JSON file mapping raw model names to labels.")
    parser.add_argument(
        "--highlight-models",
        # default="extendedmlp,extendeddouble,extendedmlploss",
        help="Comma-separated models to mark/highlight on the curve.",
    )
    parser.add_argument(
        "--baseline-models",
        default="random,last_item,bl_proxy,bl_knn,most_popular_past",
        help="Comma-separated models drawn with dashed lines.",
    )
    parser.add_argument("--max-k", type=int, default=None, help="Optional maximum K to plot.")
    parser.add_argument("--bar-k", type=int, default=None, help="Also create a bar plot at this K.")
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Plot recall values as percentages instead of fractions.",
    )
    parser.add_argument(
        "--legend",
        choices=["below", "outside", "inside"],
        default="below",
        help="Legend placement.",
    )
    parser.add_argument("--legend-columns", type=int, default=3, help="Legend columns when --legend below.")
    parser.add_argument("--width-fraction", type=float, default=1.0, help="Fraction of thesis text width.")
    parser.add_argument("--height", type=float, default=None, help="Explicit figure height in inches.")
    parser.add_argument("--title", default=None, help="Optional title placed above the plot.")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale style cycle from the style file.")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument(
        "--no-latex-pgf",
        action="store_true",
        help="Disable PGF/LaTeX text settings. Useful if XeLaTeX is unavailable.",
    )
    parser.add_argument(
        "--use-pgf-backend",
        action="store_true",
        help="Use Matplotlib PGF backend before importing pyplot. Useful for PGF-only workflows.",
    )

    args = parser.parse_args()
    args.formats = parse_formats(args.formats)
    args.highlight_models = parse_csv_list(args.highlight_models)
    args.baseline_models = parse_csv_list(args.baseline_models)

    style = load_style(args.style_file)
    setup_matplotlib(style, args)

    # Import pyplot only after style/backend setup.
    import matplotlib.pyplot as plt  # noqa: F401

    output_dir = Path(args.output_dir)
    df = prepare_dataframe(
        Path(args.input),
        models=parse_csv_list(args.models),
        exclude_models=parse_csv_list(args.exclude_models),
        max_k=args.max_k,
    )
    model_order = ordered_models(df, parse_csv_list(args.model_order))
    label_map = read_label_map(args.label_map_json)

    written: List[Path] = []
    fig = plot_recall_curve(df, model_order, label_map, style, args)
    written.extend(save_figure(fig, style, output_dir, "recall_at_k", args.formats))

    if args.bar_k is not None:
        fig_bar = plot_bar_at_k(df, args.bar_k, model_order, label_map, style, args)
        written.extend(save_figure(fig_bar, style, output_dir, f"recall_at_{args.bar_k}_bar", args.formats))

    save_plot_data(df, output_dir, model_order, label_map)

    print("Written files:")
    for path in written:
        print(f"  {path}")
    print(f"  {output_dir / 'recall_plot_data.csv'}")


if __name__ == "__main__":
    main()

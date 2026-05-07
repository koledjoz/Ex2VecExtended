#!/usr/bin/env python3
r"""
Create LaTeX-ready probability/score interaction-impact plots for one
(model, user, item) triple.

This script is extracted from the plotting part of `users_clustering.ipynb`,
with the third distance curve removed. It keeps the probability curve, score
curve, interaction-similarity rug markers, colorbar, and log-scale histogram.

Expected curve pickle format, following the notebook:

    data = pd.read_pickle("curves_all_models.pkl")
    df = data[model]

where `df` has at least columns like:

    user, item, score, prob

and each score/prob cell contains a 1D curve array.

Example:

    python plot_user_item_curves.py \
        --pkl curves_all_models.pkl \
        --model extended_doublemlploss \
        --user 13201 \
        --item 985 \
        --interactions-parquet sorted_data.parquet \
        --similarity-matrix sim_matrix.npy \
        --output-dir figures

The default outputs are PGF and PDF, so the PGF can be imported in LaTeX with:

    \input{figures/interaction_curves_extended_doublemlploss_user_13201_item_985.pgf}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Use the shared thesis plotting style supplied alongside this script.
try:
    from thesis_plot_style import fig_size, savefig, setup_mpl
except ImportError as exc:  # pragma: no cover - user-facing failure path
    raise SystemExit(
        "Could not import thesis_plot_style.py. Put thesis_plot_style.py in the "
        "same directory as this script or on PYTHONPATH."
    ) from exc


NOTEBOOK_TIME_START = 1654041600
NOTEBOOK_TIME_END = 1661990376


def _safe_slug(value: object) -> str:
    """Return a filesystem-safe stem component."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "value"


def _load_pickle(path: Path):
    """Load a pickle via pandas, matching the notebook workflow."""
    return pd.read_pickle(path)


def _select_model_frame(data, model: str) -> pd.DataFrame:
    """Select the model DataFrame from the notebook-style pickle."""
    if isinstance(data, dict):
        if model not in data:
            available = ", ".join(map(str, data.keys()))
            raise KeyError(
                f"Model {model!r} was not found in the pickle. "
                f"Available models: {available}"
            )
        df = data[model]
    elif isinstance(data, pd.DataFrame):
        if "model" in data.columns:
            df = data.loc[data["model"] == model]
            if df.empty:
                raise KeyError(f"No rows found for model {model!r} in the DataFrame.")
        else:
            df = data
    else:
        raise TypeError(
            "Unsupported pickle content. Expected a dict of DataFrames or a DataFrame."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The selected model object is {type(df)!r}, not a pandas DataFrame.")
    return df


def _curve_from_row(
    df: pd.DataFrame,
    row: pd.Series,
    column: str,
    fallback_position: int,
    label: str,
) -> np.ndarray:
    """Extract a 1D curve from `column`, falling back to notebook positions."""
    if column in df.columns:
        raw = row[column]
    else:
        try:
            fallback_column = df.columns[fallback_position]
        except IndexError as exc:
            raise KeyError(
                f"Column {column!r} does not exist and fallback position "
                f"{fallback_position} is unavailable. Columns: {list(df.columns)}"
            ) from exc
        print(
            f"Warning: column {column!r} not found. Using positional notebook "
            f"fallback for {label}: column {fallback_position} ({fallback_column!r})."
        )
        raw = row.iloc[fallback_position]

    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} curve must be 1D; got shape {arr.shape}.")
    if len(arr) == 0:
        raise ValueError(f"{label} curve must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} curve contains NaN or infinite values.")
    return arr


def extract_curves(
    pkl_path: Path,
    model: str,
    user_id: int,
    item_id: int,
    *,
    user_col: str = "user",
    item_col: str = "item",
    score_col: str = "score",
    prob_col: str = "prob",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probability_curve, score_curve) for one model/user/item."""
    data = _load_pickle(pkl_path)
    df = _select_model_frame(data, model)

    missing_id_cols = [col for col in (user_col, item_col) if col not in df.columns]
    if missing_id_cols:
        raise KeyError(
            f"Missing id columns {missing_id_cols}. Available columns: {list(df.columns)}"
        )

    match = df.loc[(df[user_col] == user_id) & (df[item_col] == item_id)]
    if match.empty:
        raise ValueError(
            f"No curve row found for model={model!r}, {user_col}={user_id}, "
            f"{item_col}={item_id}."
        )
    if len(match) > 1:
        print(f"Warning: found {len(match)} rows; using the first one.")

    row = match.iloc[0]
    # Notebook used iloc[0, 3] for prob and iloc[0, 2] for score.
    probabilities = _curve_from_row(df, row, prob_col, 3, "probability")
    scores = _curve_from_row(df, row, score_col, 2, "score")

    if len(probabilities) != len(scores):
        raise ValueError(
            f"Probability curve length ({len(probabilities)}) does not match "
            f"score curve length ({len(scores)})."
        )
    return probabilities, scores


def make_times(
    curve_length: int,
    *,
    start: float = NOTEBOOK_TIME_START,
    end: float = NOTEBOOK_TIME_END,
    n_times: int | None = None,
) -> np.ndarray:
    """Create the same linspace time axis used in the notebook."""
    count = int(n_times) if n_times is not None else int(curve_length)
    if count <= 0:
        raise ValueError("n_times must be positive.")
    times = np.linspace(float(start), float(end), count)
    if len(times) != curve_length:
        raise ValueError(
            f"Generated {len(times)} time values, but curves have length {curve_length}. "
            "Set --n-times to match the curve length, or omit it to infer the length."
        )
    return times


def _load_similarity_matrix(path: Path, key: str | None = None) -> np.ndarray:
    """Load a similarity matrix from .npy, .npz, or pickle."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if key is None:
            key = "sim_matrix" if "sim_matrix" in archive.files else archive.files[0]
        matrix = archive[key]
    else:
        obj = pd.read_pickle(path)
        if isinstance(obj, dict):
            if key is None:
                key = "sim_matrix" if "sim_matrix" in obj else next(iter(obj.keys()))
            matrix = obj[key]
        else:
            matrix = obj

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Similarity matrix must be 2D; got shape {matrix.shape}.")
    return matrix


def load_interaction_similarities(
    *,
    interactions_parquet: Path | None,
    similarity_matrix_path: Path | None,
    target_user: int,
    target_item: int,
    ts_user_col: str = "user_id",
    ts_item_col: str = "track_id",
    ts_time_col: str = "ts",
    similarity_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interaction timestamps and similarities for the rug/histogram.

    The notebook computed:

        items = df_full[df_full['user_id'] == u]['track_id'].to_numpy()
        ts = df_full[df_full['user_id'] == u]['ts'].to_numpy()
        similarities = [sim_matrix[i, x] for x in items]

    This function does the same when both the interaction parquet and the
    similarity matrix are supplied. If neither is supplied, it returns empty
    arrays so the curve panels still render.
    """
    if interactions_parquet is None and similarity_matrix_path is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    if interactions_parquet is None or similarity_matrix_path is None:
        raise ValueError(
            "To reproduce the notebook's similarity rug and histogram, provide both "
            "--interactions-parquet and --similarity-matrix. Or omit both to draw "
            "only the probability/score curves with an empty histogram panel."
        )

    interactions = pd.read_parquet(interactions_parquet)
    missing = [c for c in (ts_user_col, ts_item_col, ts_time_col) if c not in interactions.columns]
    if missing:
        raise KeyError(
            f"Missing interaction columns {missing}. Available columns: {list(interactions.columns)}"
        )

    user_history = interactions.loc[interactions[ts_user_col] == target_user]
    if user_history.empty:
        print(f"Warning: no interaction history found for user {target_user}.")
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    matrix = _load_similarity_matrix(similarity_matrix_path, key=similarity_key)
    history_items = user_history[ts_item_col].to_numpy(dtype=int)
    time_sim = user_history[ts_time_col].to_numpy(dtype=float)

    if target_item < 0 or target_item >= matrix.shape[0]:
        raise IndexError(
            f"Target item {target_item} is outside similarity matrix axis 0 "
            f"with size {matrix.shape[0]}."
        )
    if np.any(history_items < 0) or np.any(history_items >= matrix.shape[1]):
        bad = history_items[(history_items < 0) | (history_items >= matrix.shape[1])]
        raise IndexError(
            f"Some history items are outside similarity matrix axis 1 with size "
            f"{matrix.shape[1]}; examples: {bad[:10].tolist()}"
        )

    similarities = matrix[target_item, history_items]
    return time_sim, np.asarray(similarities, dtype=float)


def _prepare_interp_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average duplicate x positions before interpolation, matching notebook code."""
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique_x) == len(x):
        return x, y
    y_sum = np.bincount(inverse, weights=y)
    unique_y = y_sum / counts
    return unique_x, unique_y


def plot_interaction_impact_two_curves(
    time_vals: Sequence[float],
    probabilities: Sequence[float],
    scores: Sequence[float],
    time_sim: Sequence[float] | None = None,
    similarities: Sequence[float] | None = None,
    *,
    model: str | None = None,
    user_id: int | None = None,
    item_id: int | None = None,
    cmap: str = "viridis",
    min_similarity: float | None = None,
    target_bins: int = 80,
    figsize: tuple[float, float] | None = None,
):
    """Plot probability and score curves with similarity rug and histogram.

    This is the notebook's `plot_dual_interaction_impact` with the distance
    curve removed. The visual treatment of the two remaining curves, the log
    color normalization, colorbar, and histogram are preserved.
    """
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    sim_display_min = 1e-4
    sim_display_max = 1.0

    time_vals = np.asarray(time_vals, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    scores = np.asarray(scores, dtype=float)
    time_sim = np.asarray([] if time_sim is None else time_sim, dtype=float)
    similarities = np.asarray([] if similarities is None else similarities, dtype=float)

    for name, arr in (
        ("time_vals", time_vals),
        ("probabilities", probabilities),
        ("scores", scores),
        ("time_sim", time_sim),
        ("similarities", similarities),
    ):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D array-like input.")

    if len(time_vals) == 0:
        raise ValueError("time_vals must not be empty.")
    if len(probabilities) != len(time_vals):
        raise ValueError("probabilities must have the same length as time_vals.")
    if len(scores) != len(time_vals):
        raise ValueError("scores must have the same length as time_vals.")
    if len(time_sim) != len(similarities):
        raise ValueError("time_sim and similarities must have the same length.")
    if target_bins is None or int(target_bins) < 2:
        raise ValueError("target_bins must be an integer >= 2.")
    target_bins = int(target_bins)

    for name, arr in (
        ("time_vals", time_vals),
        ("probabilities", probabilities),
        ("scores", scores),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values.")

    if len(time_sim) > 0:
        valid_interactions = np.isfinite(time_sim) & np.isfinite(similarities)
        dropped = int(np.count_nonzero(~valid_interactions))
        if dropped > 0:
            print(f"Warning: dropped {dropped} interaction(s) with NaN/inf values.")
        time_sim = time_sim[valid_interactions]
        similarities = similarities[valid_interactions]

    # Sort primary series for plotting/interpolation.
    sort_main = np.argsort(time_vals)
    time_vals = time_vals[sort_main]
    probabilities = probabilities[sort_main]
    scores = scores[sort_main]

    interp_time_prob, interp_prob = _prepare_interp_xy(time_vals, probabilities)
    interp_time_score, interp_score = _prepare_interp_xy(time_vals, scores)

    # Filter and sort interactions.
    if min_similarity is not None and len(similarities) > 0:
        mask = similarities >= float(min_similarity)
        removed = int(np.count_nonzero(~mask))
        time_sim = time_sim[mask]
        similarities = similarities[mask]
        if len(similarities) == 0:
            print(f"Warning: no interactions >= {min_similarity}.")
        elif removed > 0 and min_similarity > 0:
            print(
                f"Note: {removed} interaction(s) below min_similarity={min_similarity} "
                "were excluded. Set --min-similarity 0 or omit it to show more."
            )

    if len(similarities) > 0:
        sort_interactions = np.argsort(similarities)
        time_sim = time_sim[sort_interactions]
        similarities = similarities[sort_interactions]

    dates_vals = [_dt.datetime.fromtimestamp(ts) for ts in time_vals]
    dates_sim = [_dt.datetime.fromtimestamp(ts) for ts in time_sim] if len(time_sim) > 0 else []

    if figsize is None:
        figsize = fig_size(fraction=1.0, height_in=6.8)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        3,
        2,
        width_ratios=[40, 1],
        height_ratios=[3, 3, 1.8],
        wspace=0.02,
        hspace=0.28,
    )

    ax_prob = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[1, 0], sharex=ax_prob)
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_cbar = fig.add_subplot(gs[:2, 1])

    cmap_obj = plt.get_cmap(cmap)
    color_norm = LogNorm(vmin=sim_display_min, vmax=sim_display_max, clip=True)

    if len(similarities) > 0:
        sim_for_color = np.clip(similarities, sim_display_min, sim_display_max)
        event_colors = cmap_obj(color_norm(sim_for_color))
    else:
        event_colors = np.empty((0, 4))

    if len(dates_vals) == 1:
        delta = _dt.timedelta(seconds=1)
        xlim_left = dates_vals[0] - delta
        xlim_right = dates_vals[0] + delta
    else:
        xlim_left = dates_vals[0]
        xlim_right = dates_vals[-1]

    def draw_curve(
        ax,
        y_vals: np.ndarray,
        ylabel: str,
        interp_x: np.ndarray,
        interp_y: np.ndarray,
        *,
        title: str | None = None,
        show_x_labels: bool = False,
    ) -> None:
        y_vals = np.asarray(y_vals, dtype=float)
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1e-5

        ymin = y_min - (y_range * 0.10)
        ymax = y_max + (y_range * 0.15)

        ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=1, zorder=0)
        ax.plot(
            dates_vals,
            y_vals,
            color="#151515",
            linewidth=2.5,
            alpha=1.0,
            path_effects=[pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()],
            zorder=4,
        )

        if len(time_sim) > 0:
            rug_height = ymin + (y_range * 0.04)
            ax.vlines(
                x=dates_sim,
                ymin=ymin,
                ymax=rug_height,
                colors=event_colors,
                linewidth=2.5,
                alpha=0.35,
                linestyle="-",
                zorder=2,
            )
            ax.axhline(y=rug_height, color="#dddddd", linewidth=1, zorder=1)

            in_range = (time_sim >= interp_x[0]) & (time_sim <= interp_x[-1])
            if np.any(in_range):
                valid_dates_sim = np.asarray(dates_sim, dtype=object)[in_range]
                valid_time_sim = time_sim[in_range]
                valid_event_colors = event_colors[in_range]
                curve_at_events = np.interp(valid_time_sim, interp_x, interp_y)

                ax.scatter(valid_dates_sim, curve_at_events, c="white", s=45, zorder=5)
                ax.scatter(
                    valid_dates_sim,
                    curve_at_events,
                    c=valid_event_colors,
                    s=18,
                    alpha=0.95,
                    zorder=6,
                )

        ax.set_xlim(xlim_left, xlim_right)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", color="#555555", labelpad=10)

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#222222")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(axis="y", length=0, colors="#555555")

        if show_x_labels:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
            ax.tick_params(axis="x", colors="#555555", rotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        else:
            ax.tick_params(axis="x", length=0, labelbottom=False)

    title_bits = ["Interaction Similarity vs. Probability and Score Over Time"]
    if model is not None or user_id is not None or item_id is not None:
        subtitle = []
        if model is not None:
            subtitle.append(f"model={model}")
        if user_id is not None:
            subtitle.append(f"user={user_id}")
        if item_id is not None:
            subtitle.append(f"item={item_id}")
        title_bits.append(" | ".join(subtitle))

    draw_curve(
        ax_prob,
        probabilities,
        ylabel="Predicted probability",
        interp_x=interp_time_prob,
        interp_y=interp_prob,
        title="\n".join(title_bits),
        show_x_labels=False,
    )
    draw_curve(
        ax_score,
        scores,
        ylabel="Raw score",
        interp_x=interp_time_score,
        interp_y=interp_score,
        show_x_labels=True,
    )
    ax_score.set_xlabel("Time", fontsize=11, fontweight="bold", color="#555555", labelpad=10)

    sm = ScalarMappable(norm=color_norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label(
        "Interaction similarity (log color scale)",
        fontsize=10,
        fontweight="bold",
        color="#555555",
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#555555", size=0)
    cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())

    clipped_low_count = 0
    clipped_high_count = 0
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(sim_display_min, sim_display_max)

    if len(similarities) > 0:
        sims_for_hist = np.asarray(similarities, dtype=float)
        clipped_low_count = int(np.count_nonzero(sims_for_hist < sim_display_min))
        clipped_high_count = int(np.count_nonzero(sims_for_hist > sim_display_max))
        sims_for_hist = np.clip(sims_for_hist, sim_display_min, sim_display_max)

        bin_edges = np.logspace(
            np.log10(sim_display_min),
            np.log10(sim_display_max),
            target_bins + 1,
        )
        counts, bins, patches = ax_hist.hist(
            sims_for_hist,
            bins=bin_edges,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        bin_centers = np.clip(bin_centers, sim_display_min, sim_display_max)
        for center, patch in zip(bin_centers, patches):
            patch.set_facecolor(cmap_obj(color_norm(center)))
            patch.set_alpha(0.92)

        positive_counts = counts[counts > 0]
        if len(positive_counts) > 0:
            ymin_hist = max(float(np.min(positive_counts)) * 0.8, 0.8)
            ymax_hist = float(np.max(positive_counts)) * 1.25
            if ymax_hist <= ymin_hist:
                ymax_hist = ymin_hist * 10
            ax_hist.set_ylim(ymin_hist, ymax_hist)
        else:
            ax_hist.set_ylim(0.8, 10)
    else:
        ax_hist.set_ylim(0.8, 10)
        ax_hist.text(
            0.5,
            0.5,
            "No interaction similarities to display",
            transform=ax_hist.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
        )

    ax_hist.grid(axis="y", color="#eeeeee", linestyle="--", linewidth=1, zorder=0)
    ax_hist.grid(axis="x", color="#f3f3f3", linestyle=":", linewidth=0.8, zorder=0)
    ax_hist.xaxis.set_major_locator(LogLocator(base=10.0))
    ax_hist.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax_hist.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_hist.yaxis.set_major_formatter(LogFormatterSciNotation())

    ax_hist.set_xlabel(
        "Interaction similarity (log scale; fixed range 1e-4 to 1)",
        fontsize=11,
        fontweight="bold",
        color="#555555",
        labelpad=10,
    )
    ax_hist.set_ylabel(
        "Interactions per similarity bin (log count scale)",
        fontsize=10,
        fontweight="bold",
        color="#555555",
        labelpad=10,
    )

    clip_notes = []
    if clipped_low_count > 0:
        clip_notes.append(f"≤1e-4 folded into first bin: {clipped_low_count}")
    if clipped_high_count > 0:
        clip_notes.append(f"≥1 folded into last bin: {clipped_high_count}")
    if clip_notes:
        ax_hist.text(
            0.99,
            0.96,
            "\n".join(clip_notes),
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#777777",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.88),
        )

    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)
    ax_hist.spines["bottom"].set_color("#cccccc")
    ax_hist.tick_params(axis="both", length=0, colors="#555555")

    return fig


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lstrip(".").lower() for part in value.split(",") if part.strip())
    if not formats:
        raise argparse.ArgumentTypeError("At least one output format is required.")
    return formats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot notebook-style probability/score curves for one model/user/item."
    )
    parser.add_argument("--pkl", required=True, type=Path, help="Path to curves_all_models.pkl or equivalent.")
    parser.add_argument("--model", required=True, help="Model key to select from the pickle, e.g. extended_doublemlploss.")
    parser.add_argument("--user", required=True, type=int, help="User id to plot.")
    parser.add_argument("--item", required=True, type=int, help="Item id to plot.")

    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory where figures are saved.")
    parser.add_argument("--output-name", default=None, help="File stem. Defaults to model/user/item-based name.")
    parser.add_argument("--formats", type=_parse_formats, default=("pgf", "pdf"), help="Comma-separated formats, default: pgf,pdf.")

    parser.add_argument("--user-col", default="user", help="User column in the curve DataFrame.")
    parser.add_argument("--item-col", default="item", help="Item column in the curve DataFrame.")
    parser.add_argument("--score-col", default="score", help="Score curve column in the curve DataFrame.")
    parser.add_argument("--prob-col", default="prob", help="Probability curve column in the curve DataFrame.")

    parser.add_argument("--time-start", type=float, default=NOTEBOOK_TIME_START, help="Start UNIX timestamp for the curve grid.")
    parser.add_argument("--time-end", type=float, default=NOTEBOOK_TIME_END, help="End UNIX timestamp for the curve grid.")
    parser.add_argument("--n-times", type=int, default=None, help="Number of time points. Omit to infer from curve length.")

    parser.add_argument("--interactions-parquet", type=Path, default=None, help="Optional path to sorted_data.parquet for user history timestamps.")
    parser.add_argument("--similarity-matrix", type=Path, default=None, help="Optional .npy/.npz/.pkl similarity matrix for histogram/rug colors.")
    parser.add_argument("--similarity-key", default=None, help="Key to read from .npz or dict pickle similarity files.")
    parser.add_argument("--ts-user-col", default="user_id", help="User column in interactions parquet.")
    parser.add_argument("--ts-item-col", default="track_id", help="Item column in interactions parquet.")
    parser.add_argument("--ts-time-col", default="ts", help="Timestamp column in interactions parquet.")

    parser.add_argument("--min-similarity", type=float, default=None, help="Optional lower filter for interaction similarities.")
    parser.add_argument("--target-bins", type=int, default=80, help="Histogram bin count, default 80.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap, default viridis.")

    parser.add_argument("--figure-fraction", type=float, default=1.0, help="Fraction of thesis text width from thesis_plot_style.fig_size().")
    parser.add_argument("--height-in", type=float, default=6.8, help="Figure height in inches, default 6.8.")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale cycle from thesis_plot_style where applicable.")
    parser.add_argument("--no-latex-pgf", action="store_true", help="Disable PGF LaTeX rcParams setup; useful for quick PNG tests.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    setup_mpl(use_latex_pgf=not args.no_latex_pgf, grayscale=args.grayscale)

    probabilities, scores = extract_curves(
        args.pkl,
        args.model,
        args.user,
        args.item,
        user_col=args.user_col,
        item_col=args.item_col,
        score_col=args.score_col,
        prob_col=args.prob_col,
    )
    times = make_times(
        len(probabilities),
        start=args.time_start,
        end=args.time_end,
        n_times=args.n_times,
    )
    time_sim, similarities = load_interaction_similarities(
        interactions_parquet=args.interactions_parquet,
        similarity_matrix_path=args.similarity_matrix,
        target_user=args.user,
        target_item=args.item,
        ts_user_col=args.ts_user_col,
        ts_item_col=args.ts_item_col,
        ts_time_col=args.ts_time_col,
        similarity_key=args.similarity_key,
    )

    fig = plot_interaction_impact_two_curves(
        times,
        probabilities,
        scores,
        time_sim,
        similarities,
        model=args.model,
        user_id=args.user,
        item_id=args.item,
        cmap=args.cmap,
        min_similarity=args.min_similarity,
        target_bins=args.target_bins,
        figsize=fig_size(fraction=args.figure_fraction, height_in=args.height_in),
    )

    output_name = args.output_name or (
        f"interaction_curves_{_safe_slug(args.model)}_user_{args.user}_item_{args.item}"
    )
    written = savefig(
        fig,
        output_name,
        output_dir=args.output_dir,
        formats=args.formats,
        close=True,
    )

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

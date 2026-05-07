#!/usr/bin/env python3
r"""
Recreate the interaction-impact plot from users_clustering.ipynb, but only keep:
    1) predicted probability over time
    2) raw score over time
    3) histogram of interaction similarities

The distance panel from the notebook is intentionally removed.

This version is intentionally minimal and follows the notebook closely:
- input paths are hard-coded from the notebook / project layout
- the plotting logic is kept close to the notebook
- the model-label mapping lives inside this file
- the thesis style file is used for final export

Usage
-----
Run from the plotting directory, for example:
    python plot_user_item_curves_old.py --model original --user 10377 --item 2165
    python plot_user_item_curves_old.py --model extended_doublemlploss --user 10377 --item 2165

Output
------
Files are saved to ./figures/curves/ as both .pdf and .pgf.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Hard-coded paths: keep these aligned with the notebook / project structure.
# -----------------------------------------------------------------------------

CURVES_PKL = Path("./curves_all_models.pkl")
INTERACTIONS_FILE = Path("../sorted_data.parquet")
STYLE_FILE = Path("./thesis_plot_style.py")
OUTPUT_DIR = Path("./figures/curves")

REPO_ROOT = Path(r"Z:/Skola/ING/Recommenders/repo/Ex2VecExtended")
WEIGHTS_PATH = Path(
    r"Z:/Skola/ING/Recommenders/repo/Ex2VecExtended/train_notebooks/weights/extended_doublemlp_loss/weights.pt"
)
PRETRAINED_EMBEDDINGS_PATH = Path(
    r"Z:/Skola/ING/Recommenders/repo/Ex2VecExtended/split_data/track_embeddings.parquet"
)
ITEM_MAPPING_PATH = Path(
    r"Z:/Skola/ING/Recommenders/repo/Ex2VecExtended/split_data/item_mapping.json"
)

NOTEBOOK_TIME_START = 1654041600
NOTEBOOK_TIME_END = 1661990376
NOTEBOOK_NUM_POINTS = 500

SIM_DISPLAY_MIN = 1e-4
SIM_DISPLAY_MAX = 1.0


# -----------------------------------------------------------------------------
# Final thesis labels.
# -----------------------------------------------------------------------------

MODEL_LABELS: Dict[str, str] = {
    "original": "Ex2Vec",
    "extendedbase": "Ex2Vec-Sim",
    "extended_base": "Ex2Vec-Sim",
    "extendedBase": "Ex2Vec-Sim",
    "extendeddouble": "Ex2Vec-PSim",
    "extended_double": "Ex2Vec-PSim",
    "extendedmlploss": "Ex2Vec-PMLP",
    "extended_mlp_loss": "Ex2Vec-PMLP",
    "extended_doublemlploss": "Ex2Vec-PMLP",
    "extended_doublemlp_loss": "Ex2Vec-PMLP",
}


# -----------------------------------------------------------------------------
# Small utilities.
# -----------------------------------------------------------------------------

def _maybe_wsl_path(path: Path) -> Path:
    """Allow the notebook's Windows paths to work under WSL when possible."""
    raw = str(path)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if not m:
        return path
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    candidate = Path("/mnt") / drive / rest
    return candidate if candidate.exists() else path


def _resolve(path: Path) -> Path:
    return _maybe_wsl_path(path)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key, e.g. original or extended_doublemlploss")
    parser.add_argument("--user", required=True, type=int, help="User id")
    parser.add_argument("--item", required=True, type=int, help="Item id")
    return parser.parse_args()


def load_style_module(style_path: Path):
    style_path = _resolve(style_path)
    spec = importlib.util.spec_from_file_location("thesis_plot_style", style_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import style file: {style_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# def setup_plot_style(style_module) -> None:
#     """Apply thesis style. Prefer PGF/XeLaTeX exactly as intended."""
#     try:
#         style_module.use_pgf_backend()
#     except Exception:
#         # If the backend was already set or cannot be switched, continue.
#         pass
#
#     style_module.setup_mpl(use_latex_pgf=True, grid=True)

def setup_plot_style(style_module) -> None:
    """Apply thesis style, but force manual layout for this dense multi-panel plot."""
    style_module.setup_mpl(use_latex_pgf=True, grid=True)

    # The thesis style enables constrained layout globally. This figure uses
    # a hand-controlled GridSpec with a dedicated title row, so constrained
    # layout must stay off or Matplotlib may move the title/panels during save.
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    mpl.rcParams["figure.autolayout"] = False


# -----------------------------------------------------------------------------
# Data loading.
# -----------------------------------------------------------------------------

def load_curves_dict() -> dict:
    path = _resolve(CURVES_PKL)
    data = pd.read_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected curves pickle to contain a dict, got {type(data)!r}")
    return data



def load_interactions() -> pd.DataFrame:
    path = _resolve(INTERACTIONS_FILE)
    return pd.read_parquet(path)



def get_curve_row(df: pd.DataFrame, user_id: int, item_id: int) -> pd.Series:
    if "user" not in df.columns or "item" not in df.columns:
        raise KeyError(f"Expected columns 'user' and 'item'. Available: {list(df.columns)}")
    match = df[(df["user"] == user_id) & (df["item"] == item_id)]
    if match.empty:
        raise ValueError(f"No row found for user={user_id}, item={item_id}")
    return match.iloc[0]



def extract_prob_and_score(row: pd.Series, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the notebook convention:
    - score was accessed with iloc[..., 2]
    - prob  was accessed with iloc[..., 3]

    If named columns exist, prefer them, otherwise fall back to the notebook's
    positional indexing.
    """
    if "score" in df.columns:
        score = row["score"]
    else:
        score = row.iloc[2]

    if "prob" in df.columns:
        prob = row["prob"]
    else:
        prob = row.iloc[3]

    score_arr = np.asarray(score, dtype=float).reshape(-1)
    prob_arr = np.asarray(prob, dtype=float).reshape(-1)

    if len(score_arr) == 0 or len(prob_arr) == 0:
        raise ValueError("Curve arrays must not be empty")
    if len(score_arr) != len(prob_arr):
        raise ValueError("Score and probability curves must have the same length")

    return prob_arr, score_arr



def get_time_axis(n_points: int) -> np.ndarray:
    """Use the notebook time range, but match the actual curve length."""
    return np.linspace(NOTEBOOK_TIME_START, NOTEBOOK_TIME_END, n_points)



def get_user_history(df_full: pd.DataFrame, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
    history = df_full[df_full["user_id"] == user_id].copy()
    history = history.sort_values("ts", kind="stable")
    items = history["track_id"].to_numpy(dtype=int)
    ts = history["ts"].to_numpy(dtype=float)
    return items, ts


# -----------------------------------------------------------------------------
# Similarities.
# -----------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))



def get_dist_weight(dist, s, f):
    return sigmoid(s / (1 + dist) - f * s) / sigmoid(s - f * s)



def get_weighted_dist(dist, s, f):
    return 1.0 / (1.0 + dist) * get_dist_weight(dist, s, f)



def ensure_repo_importable(repo_root: Path) -> None:
    repo_root = _resolve(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))



def load_similarity_row_for_learned_model(df_full: pd.DataFrame, item_id: int) -> np.ndarray:
    """Follow the notebook logic for the learned similarity model."""
    import torch

    ensure_repo_importable(REPO_ROOT)
    from src.models.optimized.extendedMLDistLoss import Ex2VecExtendedMLP

    weights = torch.load(_resolve(WEIGHTS_PATH), map_location="cpu", weights_only=False)
    smooth = weights["model_state_dict"]["smooth"].cpu().numpy()
    force = weights["model_state_dict"]["force"].cpu().numpy()

    user_count = int(df_full["user_id"].max())
    item_count = int(df_full["track_id"].max())

    config = {
        "n_users": user_count,
        "n_items": item_count,
        "latent_d": 64,
        "pretrained_embeddings_path": str(_resolve(PRETRAINED_EMBEDDINGS_PATH)),
        "item_mapping": str(_resolve(ITEM_MAPPING_PATH)),
        "mlp_dist_conf": {
            "emb_dim": 128,
            "hidden_dims": [256, 128],
            "block_size": 512,
            "activation": torch.nn.ReLU,
            "dropout": 0.1,
            "positive_output": True,
        },
    }

    model = Ex2VecExtendedMLP(config)
    model.load_state_dict(weights["model_state_dict"], strict=False)
    model.to("cpu")
    model.eval()

    with torch.no_grad():
        item_dist_matrix = model.metric(model.embedding_item_extension.weight).detach().cpu().numpy()

    sim_matrix = get_weighted_dist(item_dist_matrix, smooth, force)
    if item_id < 0 or item_id >= sim_matrix.shape[0]:
        raise IndexError(f"Target item {item_id} is outside similarity matrix shape {sim_matrix.shape}")
    return np.asarray(sim_matrix[item_id], dtype=float)



def compute_interaction_similarities(model: str, df_full: pd.DataFrame, user_id: int, item_id: int) -> Tuple[np.ndarray, np.ndarray]:
    items, ts = get_user_history(df_full, user_id)

    if model == "original":
        # User explicitly asked for original model similarity to be:
        # same item -> 1, otherwise 0, and zeros should not be displayed.
        similarities = (items == item_id).astype(float)
    else:
        similarity_row = load_similarity_row_for_learned_model(df_full, item_id)
        if np.max(items) >= len(similarity_row):
            raise IndexError(
                f"Interaction item id {np.max(items)} exceeds similarity row length {len(similarity_row)}"
            )
        similarities = similarity_row[items]

    return ts, similarities


# -----------------------------------------------------------------------------
# Plotting.
# -----------------------------------------------------------------------------

def _prepare_interp_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique_x) == len(x):
        return x, y
    y_sum = np.bincount(inverse, weights=y)
    unique_y = y_sum / counts
    return unique_x, unique_y



# def save_figure(fig, style_module, stem: str) -> List[Path]:
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     return style_module.savefig(fig, stem, output_dir=OUTPUT_DIR, formats=("pdf", "pgf"), close=True)

def save_figure(fig, style_module, stem: str) -> List[Path]:
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    written.append(pdf_path)

    # PGF can be very large and may exceed MiKTeX memory for dense interaction plots.
    # Try it, but do not fail the whole script if PGF export breaks.
    pgf_path = OUTPUT_DIR / f"{stem}.pgf"
    try:
        fig.savefig(pgf_path, format="pgf", bbox_inches="tight", pad_inches=0.02)
        written.append(pgf_path)
    except Exception as exc:
        print(f"Warning: PGF export failed, but PDF was saved: {exc}")

    plt.close(fig)
    return written

def plot_dual_interaction_impact(
    time_vals: np.ndarray,
    probabilities: np.ndarray,
    scores: np.ndarray,
    time_sim: np.ndarray,
    similarities: np.ndarray,
    title_model: str,
    user_id: int,
    item_id: int,
):
    import datetime
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    time_vals = np.asarray(time_vals, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    scores = np.asarray(scores, dtype=float)
    time_sim = np.asarray(time_sim, dtype=float)
    similarities = np.asarray(similarities, dtype=float)

    main_order = np.argsort(time_vals)
    time_vals = time_vals[main_order]
    probabilities = probabilities[main_order]
    scores = scores[main_order]

    interp_time_prob, interp_prob = _prepare_interp_xy(time_vals, probabilities)
    interp_time_score, interp_score = _prepare_interp_xy(time_vals, scores)

    # Original model: drop zeros completely, because the intended display is only
    # same-item interactions (similarity 1).
    positive_mask = np.isfinite(time_sim) & np.isfinite(similarities) & (similarities > 0)
    time_sim = time_sim[positive_mask]
    similarities = similarities[positive_mask]

    if len(similarities) > 0:
        interaction_order = np.argsort(similarities)
        time_sim = time_sim[interaction_order]
        similarities = similarities[interaction_order]

    dates_vals = [datetime.datetime.fromtimestamp(ts) for ts in time_vals]
    dates_sim = [datetime.datetime.fromtimestamp(ts) for ts in time_sim]

    # Wider and taller than before so the thesis fonts fit cleanly.
    # The first GridSpec row is reserved for the title. This avoids the
    # recurring suptitle-overlap problem when saving with bbox_inches="tight"
    # or when thesis_plot_style changes layout-related rcParams.
    fig = plt.figure(figsize=(6.3, 7.05), facecolor="white", constrained_layout=False)
    try:
        fig.set_layout_engine(None)
    except AttributeError:
        pass

    gs = gridspec.GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[38, 1.5],
        height_ratios=[0.72, 2.6, 2.6, 1.5],
        wspace=0.05,
        hspace=0.42,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_prob = fig.add_subplot(gs[1, 0])
    ax_score = fig.add_subplot(gs[2, 0], sharex=ax_prob)
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_cbar = fig.add_subplot(gs[1:3, 1])

    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.70,
        "Interaction similarity vs. probability and score over time",
        ha="center",
        va="center",
        transform=ax_title.transAxes,
        fontsize=10,
    )
    ax_title.text(
        0.5,
        0.28,
        f"{title_model} | user={user_id} | item={item_id}",
        ha="center",
        va="center",
        transform=ax_title.transAxes,
        fontsize=8.5,
    )

    cmap = plt.get_cmap("viridis")
    norm = LogNorm(vmin=SIM_DISPLAY_MIN, vmax=SIM_DISPLAY_MAX, clip=True)

    if len(similarities) > 0:
        sim_for_color = np.clip(similarities, SIM_DISPLAY_MIN, SIM_DISPLAY_MAX)
        event_colors = cmap(norm(sim_for_color))
    else:
        event_colors = np.empty((0, 4))

    def draw_curve(ax, y_vals, interp_x, interp_y, ylabel, show_x=False):
        y_vals = np.asarray(y_vals, dtype=float)
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        y_range = y_max - y_min
        if np.isclose(y_range, 0.0):
            y_range = max(abs(y_min), 1.0) * 0.05 + 1e-8

        ymin = y_min - 0.10 * y_range
        ymax = y_max + 0.16 * y_range
        rug_top = ymin + 0.04 * y_range

        ax.plot(dates_vals, y_vals, color="0.15", linewidth=1.25, zorder=3)

        if len(time_sim) > 0:
            ax.vlines(dates_sim, ymin=ymin, ymax=rug_top, colors=event_colors, linewidth=0.7, alpha=0.55, zorder=1)
            in_range = (time_sim >= interp_x[0]) & (time_sim <= interp_x[-1])
            if np.any(in_range):
                curve_at_events = np.interp(time_sim[in_range], interp_x, interp_y)
                ax.scatter(
                    np.asarray(dates_sim, dtype=object)[in_range],
                    curve_at_events,
                    c=event_colors[in_range],
                    s=10,
                    linewidths=0.25,
                    edgecolors="white",
                    zorder=4,
                )

        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", labelbottom=show_x)
        ax.tick_params(axis="both", pad=2)

    draw_curve(ax_prob, probabilities, interp_time_prob, interp_prob, "Predicted\nprobability", show_x=False)
    draw_curve(ax_score, scores, interp_time_score, interp_score, "Raw\nscore", show_x=True)

    ax_score.set_xlabel("Time")
    ax_score.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax_score.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_score.xaxis.get_major_locator()))

    # Histogram: close to notebook behavior, but only for positive similarities.
    bins = np.logspace(np.log10(SIM_DISPLAY_MIN), np.log10(SIM_DISPLAY_MAX), 40)
    if len(similarities) > 0:
        sim_hist = np.clip(similarities, SIM_DISPLAY_MIN, SIM_DISPLAY_MAX)
        counts, edges = np.histogram(sim_hist, bins=bins)
        centers = np.sqrt(edges[:-1] * edges[1:])
        widths = edges[1:] - edges[:-1]
        bar_colors = cmap(norm(np.clip(centers, SIM_DISPLAY_MIN, SIM_DISPLAY_MAX)))
        positive = counts > 0
        ax_hist.bar(
            edges[:-1][positive],
            counts[positive],
            width=widths[positive],
            align="edge",
            color=bar_colors[positive],
            edgecolor="white",
            linewidth=0.25,
        )
    else:
        ax_hist.text(
            0.5,
            0.55,
            "No positive interaction similarities to display",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )

    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(SIM_DISPLAY_MIN, SIM_DISPLAY_MAX)
    ax_hist.set_ylabel("Interactions per bin\n(log count scale)")
    ax_hist.set_xlabel("Interaction similarity (log scale; shown range $10^{-4}$ to 1)")
    ax_hist.grid(True, axis="y")
    ax_hist.grid(False, axis="x")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Interaction similarity (log color scale)")
    cbar.locator = LogLocator(base=10)
    cbar.formatter = LogFormatterSciNotation(base=10)
    cbar.update_ticks()

    # Explicit margins for the whole GridSpec. The title is now inside its own
    # axis, so there is no suptitle that can overlap the first plot.
    fig.subplots_adjust(left=0.16, right=0.88, top=0.98, bottom=0.09)
    return fig


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    style_module = load_style_module(STYLE_FILE)
    setup_plot_style(style_module)

    data = load_curves_dict()
    if args.model not in data:
        available = ", ".join(sorted(map(str, data.keys())))
        raise KeyError(f"Model {args.model!r} not found in curves file. Available: {available}")

    df_curves = data[args.model]
    df_full = load_interactions()

    row = get_curve_row(df_curves, args.user, args.item)
    curve_prob, curve_score = extract_prob_and_score(row, df_curves)
    times = get_time_axis(len(curve_prob))

    time_sim, similarities = compute_interaction_similarities(args.model, df_full, args.user, args.item)

    title_model = model_label(args.model)
    fig = plot_dual_interaction_impact(
        times,
        curve_prob,
        curve_score,
        time_sim,
        similarities,
        title_model=title_model,
        user_id=args.user,
        item_id=args.item,
    )

    stem = f"interaction_curves_{args.model}_user_{args.user}_item_{args.item}"
    written = save_figure(fig, style_module, stem)

    print("Saved:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

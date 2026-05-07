import argparse
import logging
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, dendrogram
from scipy.sparse.linalg import eigsh, LinearOperator

from sklearn.metrics import silhouette_score


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("similarity_clustering")


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@contextmanager
def timed_step(name: str):
    logger.info(f"START: {name}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"DONE:  {name} in {elapsed:.2f} seconds")


# ============================================================
# Utility parsing
# ============================================================

def parse_number_list(text, dtype=float):
    if text is None or text.strip() == "":
        return None
    return [dtype(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================
# Classical MDS / PCoA
# ============================================================

def classical_mds_fast(D, n_components=2, eig_tol=1e-4):
    """
    Fast Classical MDS / PCoA using only the top eigenvectors.

    D is an n x n dissimilarity matrix.
    """

    n = D.shape[0]

    with timed_step("Squaring dissimilarity matrix for Classical MDS"):
        D2 = D ** 2

    if n <= n_components + 1:
        logger.info("Small matrix detected; using full eigendecomposition.")
        row_mean = D2.mean(axis=1, keepdims=True)
        col_mean = D2.mean(axis=0, keepdims=True)
        grand_mean = D2.mean()
        B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

        eigvals, eigvecs = np.linalg.eigh(B)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order][:n_components]
        eigvecs = eigvecs[:, order][:, :n_components]

        eigvals_pos = np.maximum(eigvals, 0)
        coords = eigvecs * np.sqrt(eigvals_pos)
        return coords, eigvals

    def matvec(v):
        """
        Computes B @ v without explicitly forming B.

        B = -0.5 * J D^2 J
        """
        v_centered = v - v.mean()
        w = D2 @ v_centered
        w_centered = w - w.mean()
        return -0.5 * w_centered

    with timed_step("Computing top eigenvectors for Classical MDS"):
        operator = LinearOperator(
            shape=(n, n),
            matvec=matvec,
            dtype=np.float64,
        )

        eigvals, eigvecs = eigsh(
            operator,
            k=n_components,
            which="LA",
            tol=eig_tol,
        )

    with timed_step("Building 2D coordinates"):
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals_pos = np.maximum(eigvals, 0)
        coords = eigvecs * np.sqrt(eigvals_pos)

    return coords, eigvals


# ============================================================
# Pair sampling for approximate medians
# ============================================================

def make_sample_pairs(n, sample_pairs=200_000, seed=42):
    rng = np.random.default_rng(seed)

    max_pairs = n * (n - 1)
    sample_pairs = min(sample_pairs, max_pairs)

    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n - 1, size=sample_pairs)

    # Ensure j != i
    j = j + (j >= i)

    return i, j


# ============================================================
# Cluster metrics
# ============================================================

def evaluate_labels(
    labels,
    S,
    D,
    condensed_S,
    pair_i,
    pair_j,
    sample_i,
    sample_j,
    silhouette_sample_size=1000,
    random_state=42,
    min_cluster_size=5,
):
    """
    Evaluates one clustering solution.

    Uses exact means over all item pairs and approximate medians from sampled pairs.
    """

    labels = np.asarray(labels)
    n = len(labels)

    unique, counts = np.unique(labels, return_counts=True)

    n_clusters = len(unique)
    n_singletons = int(np.sum(counts == 1))
    n_small_clusters = int(np.sum(counts < min_cluster_size))
    largest_cluster = int(counts.max())
    smallest_cluster = int(counts.min())

    singleton_fraction = n_singletons / n
    small_cluster_fraction = np.sum(counts[counts < min_cluster_size]) / n
    largest_cluster_fraction = largest_cluster / n

    # Exact pair-based within/between means using upper triangle
    with timed_step(f"Evaluating exact pair means for {n_clusters} clusters"):
        same_pair = labels[pair_i] == labels[pair_j]

        if np.any(same_pair):
            within_values = condensed_S[same_pair]
            within_mean = float(np.mean(within_values))
        else:
            within_mean = np.nan

        if np.any(~same_pair):
            between_values = condensed_S[~same_pair]
            between_mean = float(np.mean(between_values))
        else:
            between_mean = np.nan

        within_minus_between = within_mean - between_mean

    # Approximate medians from sampled directed pairs
    sample_values = S[sample_i, sample_j]
    sample_same = labels[sample_i] == labels[sample_j]

    if np.any(sample_same):
        within_median_sample = float(np.median(sample_values[sample_same]))
    else:
        within_median_sample = np.nan

    if np.any(~sample_same):
        between_median_sample = float(np.median(sample_values[~sample_same]))
    else:
        between_median_sample = np.nan

    # Silhouette score using precomputed dissimilarity
    sil = np.nan
    try:
        if 2 <= n_clusters <= n - 1:
            sample_size = min(silhouette_sample_size, n)
            sil = float(
                silhouette_score(
                    D,
                    labels,
                    metric="precomputed",
                    sample_size=sample_size,
                    random_state=random_state,
                )
            )
    except Exception as e:
        logger.warning(f"Silhouette failed: {e}")
        sil = np.nan

    return {
        "n_clusters": n_clusters,
        "singletons": n_singletons,
        "small_clusters": n_small_clusters,
        "largest_cluster": largest_cluster,
        "smallest_cluster": smallest_cluster,
        "singleton_fraction": singleton_fraction,
        "small_cluster_fraction": small_cluster_fraction,
        "largest_cluster_fraction": largest_cluster_fraction,
        "within_mean": within_mean,
        "within_median_sample": within_median_sample,
        "between_mean": between_mean,
        "between_median_sample": between_median_sample,
        "within_minus_between": within_minus_between,
        "silhouette": sil,
    }


def scan_k_values(
    S,
    D,
    Z,
    k_values,
    condensed_S,
    pair_i,
    pair_j,
    sample_i,
    sample_j,
    silhouette_sample_size,
    random_state,
    min_cluster_size,
):
    rows = []

    for k in k_values:
        logger.info(f"Scanning k={k}")

        labels = fcluster(Z, t=k, criterion="maxclust")

        metrics = evaluate_labels(
            labels=labels,
            S=S,
            D=D,
            condensed_S=condensed_S,
            pair_i=pair_i,
            pair_j=pair_j,
            sample_i=sample_i,
            sample_j=sample_j,
            silhouette_sample_size=silhouette_sample_size,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
        )

        metrics["selection_type"] = "k"
        metrics["selection_value"] = k
        metrics["k_requested"] = k
        metrics["threshold_requested"] = np.nan

        rows.append(metrics)

    return pd.DataFrame(rows)


def scan_threshold_values(
    S,
    D,
    Z,
    thresholds,
    condensed_S,
    pair_i,
    pair_j,
    sample_i,
    sample_j,
    silhouette_sample_size,
    random_state,
    min_cluster_size,
):
    rows = []

    for threshold in thresholds:
        logger.info(f"Scanning distance threshold={threshold:.5f}")

        labels = fcluster(Z, t=threshold, criterion="distance")

        metrics = evaluate_labels(
            labels=labels,
            S=S,
            D=D,
            condensed_S=condensed_S,
            pair_i=pair_i,
            pair_j=pair_j,
            sample_i=sample_i,
            sample_j=sample_j,
            silhouette_sample_size=silhouette_sample_size,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
        )

        metrics["selection_type"] = "threshold"
        metrics["selection_value"] = threshold
        metrics["k_requested"] = np.nan
        metrics["threshold_requested"] = threshold

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# Automatic choice of clustering
# ============================================================

def choose_best_candidate(
    candidates,
    min_clusters=2,
    max_clusters=200,
    max_largest_fraction=0.80,
    max_singleton_fraction=0.25,
):
    """
    Heuristic automatic choice.

    This is not a mathematical truth. It chooses a cut that tries to balance:
    - high within-cluster similarity
    - low between-cluster similarity
    - decent silhouette
    - avoiding one giant cluster
    - avoiding too many singleton clusters
    """

    df = candidates.copy()

    # Fill missing silhouette with a pessimistic value
    df["silhouette_filled"] = df["silhouette"].fillna(-1.0)

    # Scoring heuristic
    df["auto_score"] = (
        2.0 * df["within_minus_between"].fillna(-1.0)
        + 1.0 * df["silhouette_filled"]
        - 1.0 * df["largest_cluster_fraction"]
        - 0.7 * df["singleton_fraction"]
        - 0.3 * df["small_cluster_fraction"]
    )

    valid = (
        (df["n_clusters"] >= min_clusters)
        & (df["n_clusters"] <= max_clusters)
        & (df["largest_cluster_fraction"] <= max_largest_fraction)
        & (df["singleton_fraction"] <= max_singleton_fraction)
        & (df["within_mean"] > df["between_mean"])
    )

    if valid.any():
        logger.info("Found candidates satisfying automatic selection constraints.")
        chosen = df.loc[valid].sort_values("auto_score", ascending=False).iloc[0]
    else:
        logger.warning(
            "No candidate satisfied all constraints. "
            "Relaxing constraints and choosing the highest-scoring candidate."
        )

        relaxed = (
            (df["n_clusters"] >= min_clusters)
            & (df["within_mean"] > df["between_mean"])
        )

        if relaxed.any():
            chosen = df.loc[relaxed].sort_values("auto_score", ascending=False).iloc[0]
        else:
            logger.warning(
                "No candidate had within_mean > between_mean. "
                "Choosing the highest-scoring candidate overall."
            )
            chosen = df.sort_values("auto_score", ascending=False).iloc[0]

    return chosen, df


def labels_from_candidate(Z, candidate):
    selection_type = candidate["selection_type"]
    value = candidate["selection_value"]

    if selection_type == "k":
        labels = fcluster(Z, t=int(value), criterion="maxclust")
    elif selection_type == "threshold":
        labels = fcluster(Z, t=float(value), criterion="distance")
    else:
        raise ValueError(f"Unknown selection_type: {selection_type}")

    return labels


# ============================================================
# Detailed chosen-cluster summary
# ============================================================

def summarize_chosen_clusters(S, labels, item_indices, max_exact_values=2_000_000, seed=42):
    rng = np.random.default_rng(seed)

    rows = []
    unique = np.unique(labels)

    for c in unique:
        idx = np.where(labels == c)[0]
        other = np.where(labels != c)[0]

        size = len(idx)

        # Within-cluster values
        if size > 1:
            block = S[np.ix_(idx, idx)]
            within_values = block[~np.eye(size, dtype=bool)]

            if len(within_values) > max_exact_values:
                sampled = rng.choice(within_values, size=max_exact_values, replace=False)
                within_median = float(np.median(sampled))
            else:
                within_median = float(np.median(within_values))

            within_mean = float(np.mean(within_values))
        else:
            within_mean = np.nan
            within_median = np.nan

        # Between-cluster values
        if len(other) > 0:
            between_values = S[np.ix_(idx, other)].ravel()

            if len(between_values) > max_exact_values:
                sampled = rng.choice(between_values, size=max_exact_values, replace=False)
                between_median = float(np.median(sampled))
            else:
                between_median = float(np.median(between_values))

            between_mean = float(np.mean(between_values))
        else:
            between_mean = np.nan
            between_median = np.nan

        rows.append({
            "cluster": int(c),
            "size": int(size),
            "within_mean": within_mean,
            "within_median": within_median,
            "between_mean": between_mean,
            "between_median": between_median,
            "within_minus_between": within_mean - between_mean,
            "first_few_item_indices": " ".join(map(str, item_indices[idx[:20]])),
        })

    return pd.DataFrame(rows).sort_values("cluster")


# ============================================================
# Plotting
# ============================================================

def estimate_cut_height_for_k(Z, n, k):
    if k <= 1:
        return float(Z[-1, 2])

    merges_done = n - k

    if merges_done <= 0:
        return float(Z[0, 2]) / 2

    low_idx = merges_done - 1
    high_idx = merges_done

    low = Z[low_idx, 2]

    if high_idx < len(Z):
        high = Z[high_idx, 2]
        return float((low + high) / 2)

    return float(low)


def plot_clustered_heatmap(S, Z, labels, item_indices, output_path):
    with timed_step("Plotting clustered similarity heatmap"):
        order = leaves_list(Z)
        S_ordered = S[np.ix_(order, order)]
        ordered_labels = labels[order]

        fig, ax = plt.subplots(figsize=(11, 10))
        image = ax.imshow(S_ordered, aspect="auto", interpolation="nearest")
        fig.colorbar(image, ax=ax, label="Similarity")

        ax.set_title("Clustered similarity matrix")
        ax.set_xlabel("Items reordered by hierarchical clustering")
        ax.set_ylabel("Items reordered by hierarchical clustering")

        n = S.shape[0]

        if n <= 80:
            ordered_item_indices = item_indices[order]
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(ordered_item_indices, rotation=90, fontsize=6)
            ax.set_yticklabels(ordered_item_indices, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        # Draw cluster boundaries according to the chosen labels
        changes = np.where(ordered_labels[:-1] != ordered_labels[1:])[0] + 0.5

        # for boundary in changes:
        #     ax.axhline(boundary, linewidth=0.5, color="white")
        #     ax.axvline(boundary, linewidth=0.5, color="white")

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def plot_dendrogram(Z, n, candidate, output_path):
    with timed_step("Plotting truncated dendrogram"):
        fig, ax = plt.subplots(figsize=(13, 6))

        if n > 200:
            dendrogram(
                Z,
                truncate_mode="lastp",
                p=80,
                leaf_rotation=90,
                leaf_font_size=8,
                show_contracted=True,
                ax=ax,
            )
            ax.set_xlabel("Clustered item groups")
        else:
            dendrogram(
                Z,
                leaf_rotation=90,
                leaf_font_size=6,
                ax=ax,
            )
            ax.set_xlabel("Item")

        selection_type = candidate["selection_type"]
        value = candidate["selection_value"]

        if selection_type == "threshold":
            cut_height = float(value)
        else:
            cut_height = estimate_cut_height_for_k(Z, n, int(value))

        ax.axhline(cut_height, linestyle="--", linewidth=1.2)
        ax.set_title("Hierarchical clustering dendrogram")
        ax.set_ylabel("Dissimilarity")

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def plot_mds(coords, labels, item_indices, output_path, show_labels=False):
    with timed_step("Plotting 2D Classical MDS / PCoA"):
        fig, ax = plt.subplots(figsize=(9, 7))

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            s=15,
            alpha=0.85,
        )

        fig.colorbar(scatter, ax=ax, label="Cluster")

        if show_labels:
            for idx, x, y in zip(item_indices, coords[:, 0], coords[:, 1]):
                ax.text(x, y, str(idx), fontsize=6, ha="right", va="bottom")

        ax.set_title("2D Classical MDS / PCoA colored by chosen cluster")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def plot_scan_diagnostics(candidates, output_dir):
    with timed_step("Plotting scan diagnostics"):
        df = candidates.copy()

        # k scan
        k_df = df[df["selection_type"] == "k"].copy()
        if len(k_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_df["selection_value"], k_df["largest_cluster_fraction"], marker="o")
            ax.set_xlabel("Requested k")
            ax.set_ylabel("Largest cluster fraction")
            ax.set_title("Largest cluster fraction by k")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_k_largest_cluster_fraction.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_df["selection_value"], k_df["within_minus_between"], marker="o")
            ax.set_xlabel("Requested k")
            ax.set_ylabel("Mean within similarity - mean between similarity")
            ax.set_title("Cluster separation by k")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_k_separation.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_df["selection_value"], k_df["silhouette"], marker="o")
            ax.set_xlabel("Requested k")
            ax.set_ylabel("Silhouette score")
            ax.set_title("Silhouette by k")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_k_silhouette.png", dpi=300)
            plt.close(fig)

        # threshold scan
        t_df = df[df["selection_type"] == "threshold"].copy()
        if len(t_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(t_df["selection_value"], t_df["n_clusters"], marker="o")
            ax.set_xlabel("Distance threshold")
            ax.set_ylabel("Number of clusters")
            ax.set_title("Number of clusters by threshold")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_threshold_n_clusters.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(t_df["selection_value"], t_df["largest_cluster_fraction"], marker="o")
            ax.set_xlabel("Distance threshold")
            ax.set_ylabel("Largest cluster fraction")
            ax.set_title("Largest cluster fraction by threshold")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_threshold_largest_cluster_fraction.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(t_df["selection_value"], t_df["within_minus_between"], marker="o")
            ax.set_xlabel("Distance threshold")
            ax.set_ylabel("Mean within similarity - mean between similarity")
            ax.set_title("Cluster separation by threshold")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_threshold_separation.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(t_df["selection_value"], t_df["silhouette"], marker="o")
            ax.set_xlabel("Distance threshold")
            ax.set_ylabel("Silhouette score")
            ax.set_title("Silhouette by threshold")
            fig.tight_layout()
            fig.savefig(output_dir / "diagnostic_threshold_silhouette.png", dpi=300)
            plt.close(fig)

        # Candidate score overview
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        ax.plot(x, df["auto_score"], marker="o")
        ax.set_xlabel("Candidate index")
        ax.set_ylabel("Automatic score")
        ax.set_title("Automatic score for all clustering candidates")
        fig.tight_layout()
        fig.savefig(output_dir / "diagnostic_all_candidates_auto_score.png", dpi=300)
        plt.close(fig)


def plot_cluster_sizes(labels, output_path):
    with timed_step("Plotting chosen cluster sizes"):
        unique, counts = np.unique(labels, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(unique.astype(str), counts)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of items")
        ax.set_title("Chosen cluster sizes")

        if len(unique) > 40:
            ax.set_xticks([])

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================

def run_analysis(args):
    output_dir = Path(args.out)
    setup_logging(output_dir)

    logger.info("==========================================")
    logger.info("Similarity clustering analysis started")
    logger.info("==========================================")

    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")

    # --------------------------------------------------------
    # Load and prepare matrix
    # --------------------------------------------------------

    with timed_step("Loading similarity matrix"):
        S_full = np.load(args.input)

    logger.info(f"Loaded matrix shape: {S_full.shape}")

    with timed_step("Removing unused index 0"):
        S = S_full[1:, 1:].astype(np.float64, copy=True)

    n = S.shape[0]
    item_indices = np.arange(1, S_full.shape[0])

    logger.info(f"Working matrix shape: {S.shape}")
    logger.info(f"Number of real items: {n}")

    with timed_step("Validating similarity matrix"):
        if S.shape[0] != S.shape[1]:
            raise ValueError("Matrix must be square after removing row/column 0.")

        min_value = float(np.min(S))
        max_value = float(np.max(S))
        logger.info(f"Minimum similarity: {min_value}")
        logger.info(f"Maximum similarity: {max_value}")

        if min_value < -args.tolerance or max_value > 1.0 + args.tolerance:
            raise ValueError("Expected all similarity values to be between 0 and 1.")

    with timed_step("Symmetrizing matrix and forcing diagonal to 1"):
        S = (S + S.T) / 2.0
        np.fill_diagonal(S, 1.0)

    with timed_step("Converting similarity to dissimilarity"):
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)

    # Save cleaned matrices if requested
    if args.save_matrices:
        with timed_step("Saving cleaned similarity and dissimilarity matrices"):
            np.save(output_dir / "cleaned_similarity_matrix.npy", S)
            np.save(output_dir / "dissimilarity_matrix.npy", D)

    # --------------------------------------------------------
    # Hierarchical clustering
    # --------------------------------------------------------

    with timed_step("Creating condensed dissimilarity vector"):
        condensed_D = squareform(D, checks=False)

    with timed_step("Creating condensed similarity vector"):
        condensed_S = 1.0 - condensed_D

    logger.info(f"Condensed vector length: {len(condensed_D)}")

    with timed_step("Running average-linkage hierarchical clustering"):
        Z = linkage(
            condensed_D,
            method="average",
            optimal_ordering=False,
        )

    np.save(output_dir / "linkage_matrix.npy", Z)

    # --------------------------------------------------------
    # Prepare pair indices for exact scan metrics
    # --------------------------------------------------------

    with timed_step("Preparing pair indices for exact metrics"):
        pair_i, pair_j = np.triu_indices(n, k=1)

    with timed_step("Preparing sampled pairs for approximate medians"):
        sample_i, sample_j = make_sample_pairs(
            n=n,
            sample_pairs=args.sample_pairs,
            seed=args.random_state,
        )

    # --------------------------------------------------------
    # Candidate scans
    # --------------------------------------------------------

    if args.k_values is None:
        k_values = [
            2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50,
            75, 100, 150, 200, 300
        ]
    else:
        k_values = parse_number_list(args.k_values, dtype=int)

    k_values = [k for k in k_values if 2 <= k <= n - 1]
    logger.info(f"k values to scan: {k_values}")

    if args.thresholds is None:
        thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    else:
        thresholds = parse_number_list(args.thresholds, dtype=float)

    thresholds = [float(t) for t in thresholds if 0.0 < float(t) < 1.0]
    logger.info(f"Number of thresholds to scan: {len(thresholds)}")

    scan_tables = []

    if len(k_values) > 0:
        with timed_step("Scanning requested k values"):
            k_scan = scan_k_values(
                S=S,
                D=D,
                Z=Z,
                k_values=k_values,
                condensed_S=condensed_S,
                pair_i=pair_i,
                pair_j=pair_j,
                sample_i=sample_i,
                sample_j=sample_j,
                silhouette_sample_size=args.silhouette_sample_size,
                random_state=args.random_state,
                min_cluster_size=args.min_cluster_size,
            )
        k_scan.to_csv(output_dir / "scan_k_values.csv", index=False)
        scan_tables.append(k_scan)

    if len(thresholds) > 0:
        with timed_step("Scanning distance thresholds"):
            threshold_scan = scan_threshold_values(
                S=S,
                D=D,
                Z=Z,
                thresholds=thresholds,
                condensed_S=condensed_S,
                pair_i=pair_i,
                pair_j=pair_j,
                sample_i=sample_i,
                sample_j=sample_j,
                silhouette_sample_size=args.silhouette_sample_size,
                random_state=args.random_state,
                min_cluster_size=args.min_cluster_size,
            )
        threshold_scan.to_csv(output_dir / "scan_threshold_values.csv", index=False)
        scan_tables.append(threshold_scan)

    if not scan_tables:
        raise ValueError("No candidate clusterings were scanned.")

    candidates = pd.concat(scan_tables, ignore_index=True)

    # --------------------------------------------------------
    # Choose best clustering
    # --------------------------------------------------------

    with timed_step("Choosing best clustering candidate"):
        chosen, scored_candidates = choose_best_candidate(
            candidates=candidates,
            min_clusters=args.min_auto_clusters,
            max_clusters=args.max_auto_clusters,
            max_largest_fraction=args.max_largest_fraction,
            max_singleton_fraction=args.max_singleton_fraction,
        )

    scored_candidates.to_csv(output_dir / "all_scored_candidates.csv", index=False)

    chosen.to_frame().T.to_csv(output_dir / "chosen_candidate.csv", index=False)

    logger.info("Chosen clustering candidate:")
    for key, value in chosen.items():
        logger.info(f"  {key}: {value}")

    chosen_labels = labels_from_candidate(Z, chosen)

    # Re-number labels from 1..K in a stable way
    unique_labels = np.unique(chosen_labels)
    remap = {old: new for new, old in enumerate(unique_labels, start=1)}
    chosen_labels = np.array([remap[x] for x in chosen_labels], dtype=int)

    logger.info(f"Final chosen number of clusters: {len(np.unique(chosen_labels))}")

    # --------------------------------------------------------
    # Save chosen labels
    # --------------------------------------------------------

    with timed_step("Saving chosen cluster labels"):
        labels_df = pd.DataFrame({
            "item_index": item_indices,
            "cluster": chosen_labels,
        })
        labels_df.to_csv(output_dir / "chosen_cluster_labels.csv", index=False)

    with timed_step("Creating chosen cluster summary"):
        cluster_summary = summarize_chosen_clusters(
            S=S,
            labels=chosen_labels,
            item_indices=item_indices,
            max_exact_values=args.max_exact_summary_values,
            seed=args.random_state,
        )
        cluster_summary.to_csv(output_dir / "chosen_cluster_summary.csv", index=False)

    logger.info("Chosen cluster sizes:")
    for _, row in cluster_summary.iterrows():
        logger.info(f"  Cluster {int(row['cluster'])}: {int(row['size'])} items")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_scan_diagnostics(scored_candidates, output_dir)

    plot_cluster_sizes(
        chosen_labels,
        output_dir / "chosen_cluster_sizes.png",
    )

    plot_clustered_heatmap(
        S=S,
        Z=Z,
        labels=chosen_labels,
        item_indices=item_indices,
        output_path=output_dir / "chosen_clustered_similarity_heatmap.png",
    )

    plot_dendrogram(
        Z=Z,
        n=n,
        candidate=chosen,
        output_path=output_dir / "chosen_dendrogram.png",
    )

    # --------------------------------------------------------
    # MDS / PCoA
    # --------------------------------------------------------

    if not args.no_mds:
        with timed_step("Running fast Classical MDS / PCoA"):
            coords, eigvals = classical_mds_fast(
                D=D,
                n_components=2,
                eig_tol=args.eig_tol,
            )

        logger.info(f"Top MDS eigenvalues: {eigvals}")

        mds_df = pd.DataFrame({
            "item_index": item_indices,
            "mds_x": coords[:, 0],
            "mds_y": coords[:, 1],
            "cluster": chosen_labels,
        })
        mds_df.to_csv(output_dir / "chosen_mds_coordinates.csv", index=False)

        plot_mds(
            coords=coords,
            labels=chosen_labels,
            item_indices=item_indices,
            output_path=output_dir / "chosen_mds_pcoa_clusters.png",
            show_labels=args.show_mds_labels,
        )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    with timed_step("Writing text summary"):
        summary_path = output_dir / "analysis_summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Similarity clustering analysis summary\n")
            f.write("======================================\n\n")

            f.write(f"Input file: {args.input}\n")
            f.write(f"Matrix size after removing index 0: {n} x {n}\n\n")

            f.write("Chosen clustering candidate\n")
            f.write("---------------------------\n")
            f.write(f"Selection type: {chosen['selection_type']}\n")
            f.write(f"Selection value: {chosen['selection_value']}\n")
            f.write(f"Chosen number of clusters: {len(np.unique(chosen_labels))}\n")
            f.write(f"Automatic score: {chosen['auto_score']}\n")
            f.write(f"Silhouette: {chosen['silhouette']}\n")
            f.write(f"Within mean similarity: {chosen['within_mean']}\n")
            f.write(f"Between mean similarity: {chosen['between_mean']}\n")
            f.write(f"Within - between similarity: {chosen['within_minus_between']}\n")
            f.write(f"Largest cluster fraction: {chosen['largest_cluster_fraction']}\n")
            f.write(f"Singleton fraction: {chosen['singleton_fraction']}\n\n")

            f.write("Important interpretation note\n")
            f.write("-----------------------------\n")
            f.write(
                "The automatically chosen number of clusters is heuristic. "
                "It balances within-cluster similarity, between-cluster similarity, "
                "silhouette score, largest-cluster size, and singleton frequency. "
                "For thesis work, inspect the diagnostic plots and cluster summaries "
                "before treating the selected cluster count as final.\n\n"
            )

            f.write("Key output files\n")
            f.write("----------------\n")
            f.write("chosen_cluster_labels.csv\n")
            f.write("chosen_cluster_summary.csv\n")
            f.write("chosen_clustered_similarity_heatmap.png\n")
            f.write("chosen_dendrogram.png\n")
            f.write("all_scored_candidates.csv\n")
            f.write("scan_k_values.csv\n")
            f.write("scan_threshold_values.csv\n")
            f.write("run.log\n")

            if not args.no_mds:
                f.write("chosen_mds_coordinates.csv\n")
                f.write("chosen_mds_pcoa_clusters.png\n")

    logger.info("==========================================")
    logger.info("Similarity clustering analysis finished")
    logger.info("==========================================")
    logger.info(f"Results saved to: {output_dir}")


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Explore and cluster a 1-indexed similarity matrix."
    )

    parser.add_argument(
        "input",
        help="Path to .npy similarity matrix. Row/column 0 will be ignored.",
    )

    parser.add_argument(
        "--out",
        default="similarity_clustering_results",
        help="Output directory.",
    )

    parser.add_argument(
        "--k-values",
        default=None,
        help=(
            "Comma-separated k values to scan. "
            "Example: 2,3,4,5,10,20,50,100"
        ),
    )

    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Comma-separated distance thresholds to scan. "
            "Example: 0.2,0.3,0.4,0.5,0.6"
        ),
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.05,
        help="Minimum threshold for automatic threshold grid.",
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="Maximum threshold for automatic threshold grid.",
    )

    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=40,
        help="Number of threshold values to scan.",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Clusters smaller than this are counted as small clusters.",
    )

    parser.add_argument(
        "--min-auto-clusters",
        type=int,
        default=2,
        help="Minimum number of clusters allowed for automatic selection.",
    )

    parser.add_argument(
        "--max-auto-clusters",
        type=int,
        default=200,
        help="Maximum number of clusters allowed for automatic selection.",
    )

    parser.add_argument(
        "--max-largest-fraction",
        type=float,
        default=0.80,
        help="Maximum allowed fraction of all items in the largest cluster.",
    )

    parser.add_argument(
        "--max-singleton-fraction",
        type=float,
        default=0.25,
        help="Maximum allowed fraction of singleton clusters.",
    )

    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=200_000,
        help="Number of random item pairs used for approximate median similarities.",
    )

    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=1000,
        help="Number of items sampled for silhouette score.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--eig-tol",
        type=float,
        default=1e-4,
        help="Tolerance for fast MDS eigensolver.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Numerical tolerance for checking similarity range [0, 1].",
    )

    parser.add_argument(
        "--max-exact-summary-values",
        type=int,
        default=2_000_000,
        help="Maximum number of values used exactly for per-cluster medians.",
    )

    parser.add_argument(
        "--no-mds",
        action="store_true",
        help="Skip Classical MDS / PCoA visualization.",
    )

    parser.add_argument(
        "--show-mds-labels",
        action="store_true",
        help="Show item labels on the MDS plot. Not recommended for thousands of items.",
    )

    parser.add_argument(
        "--save-matrices",
        action="store_true",
        help="Save cleaned similarity and dissimilarity matrices.",
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_analysis(args)
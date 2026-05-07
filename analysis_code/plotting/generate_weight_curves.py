r"""Generate Figure 3.1 for the thesis.

Figure 3.1 shows examples of the normalized weighting function

    weight(x) = sigma(p * (x - x0)) / sigma(p * (1 - x0))

for several values of the midpoint parameter x0 and steepness parameter p.

Run from a directory containing `thesis_plot_style.py` or make sure that file
is on your PYTHONPATH:

    python generate_weight_curves.py

By default this writes:

    figures/figure_3_1.pgf
    figures/figure_3_1.pdf

The PGF file is intended for inclusion in XeLaTeX with:

    \begin{figure}
        \centering
        \input{figures/figure_3_1.pgf}
        \caption{Examples of weighting curves for different values of the
        midpoint and steepness parameters. All curves are normalized so that
        a similarity of 1 receives a weight of 1. The figure shows only the
        weighting function; the final relevance is obtained by multiplying
        this weight by the base similarity from Eq.~\ref{eq:...}.}
        \label{fig:weighting-curves}
    \end{figure}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thesis_plot_style import MEDIUM_WIDTH, fig_size, savefig, setup_mpl, use_pgf_backend


# The current thesis text defines the midpoint as x_0. If you want to reproduce
# the old embedded draft figure exactly, set this to r"c".
LEGEND_PARAMETER_NAME = r"x_0"




def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def weight(x: np.ndarray, *, x0: float, p: float) -> np.ndarray:
    """Normalized sigmoid weighting function from Eq. 3.3."""
    numerator = sigmoid(p * (x - x0))
    denominator = sigmoid(p * (1.0 - x0))
    return numerator / denominator


def make_figure():
    import matplotlib.pyplot as plt

    # The thesis figure displays the similarity axis from high to low values.
    # A small margin outside [0, 1] makes it visible that all curves cross 1 at x=1.
    x = np.linspace(-0.1, 1.1, 800)

    parameter_sets = [
        (1.0, 1.0),
        (3.0, 1.0),
        (1.0, 5.0),
        (3.0, 0.2),
    ]

    fig, ax = plt.subplots(figsize=fig_size(fraction=MEDIUM_WIDTH, aspect=0.60))

    for x0, p in parameter_sets:
        label = rf"${LEGEND_PARAMETER_NAME} = {x0:g},\ p = {p:g}$"
        ax.plot(x, weight(x, x0=x0, p=p), label=label)

    ax.set_xlabel(r"Distance-based similarity ($x$)")
    ax.set_ylabel(r"$\operatorname{weight}(x)$")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.30)
    ax.set_xticks(np.arange(1.0, -0.01, -0.2))
    ax.set_yticks(np.arange(0.0, 1.21, 0.2))

    ax.legend(loc="lower right")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis Figure 3.1.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where output files should be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pgf", "pdf"],
        help="Output formats, e.g. pgf pdf png.",
    )
    parser.add_argument(
        "--no-pgf-backend",
        action="store_true",
        help="Do not switch Matplotlib to the PGF backend. Useful for quick PNG previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Use the PGF backend when creating thesis-ready vector output. This should
    # be called before importing matplotlib.pyplot, which happens inside make_figure().
    if not args.no_pgf_backend:
        use_pgf_backend()

    setup_mpl(use_latex_pgf=True)

    fig = make_figure()
    written = savefig(
        fig,
        "weight_curves",
        output_dir=args.output_dir,
        formats=args.formats,
        close=True,
    )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()

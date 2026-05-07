r"""
Shared Matplotlib style settings for figures in a CTU FIT thesis.

This file is tailored to the `ctufit-thesis` LaTeX class configuration you
provided:

    \documentclass[english,master,twoside]{ctufit-thesis}

The class uses A4 paper, 11 pt base document size, and margins that give:

    \textwidth = 13.05 cm = 371.3085 TeX pt = 5.1378 in

For normal one-column thesis figures, \columnwidth is the same as \textwidth.

Recommended workflow
--------------------
1. Put this file somewhere importable by your plotting scripts, for example:

       thesis/scripts/thesis_plot_style.py

2. In each plotting script:

       from thesis_plot_style import setup_mpl, fig_size, savefig
       setup_mpl()

       import matplotlib.pyplot as plt

       fig, ax = plt.subplots(figsize=fig_size())
       ax.plot([0, 1, 2], [0, 1, 4], label="Example")
       ax.set_xlabel("Exposure")
       ax.set_ylabel("Score")
       ax.legend()
       savefig(fig, "example_plot")

3. Include the generated PGF figure in LaTeX like this:

       \begin{figure}
           \centering
           \input{figures/example_plot.pgf}
           \caption{Example plot.}
           \label{fig:example-plot}
       \end{figure}

Notes
-----
- The PGF output lets LaTeX typeset plot labels and math, so the fonts match
  the thesis document better than ordinary PNG/PDF text.
- The class file you showed does not set a custom main text font. It relies on
  LaTeX/XeLaTeX defaults, so this style file also avoids forcing a custom font.
- If you later add `fontspec` and choose a thesis font, set MAIN_FONT below.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import matplotlib as mpl
from cycler import cycler


# ---------------------------------------------------------------------------
# Dimensions from ctufit-thesis.cls
# ---------------------------------------------------------------------------

# A4 width = 21 cm.
# twoside: inner = 4.7 cm, outer = 3.25 cm -> text width = 13.05 cm.
# oneside: left = 3.95 cm, right = 4.0 cm -> text width = 13.05 cm.
TEXT_WIDTH_CM = 13.05
TEXT_WIDTH_PT = 371.30846456692916
TEXT_WIDTH_IN = 5.137795275590551

# The thesis is one-column by default, so column width equals text width.
COLUMN_WIDTH_CM = TEXT_WIDTH_CM
COLUMN_WIDTH_PT = TEXT_WIDTH_PT
COLUMN_WIDTH_IN = TEXT_WIDTH_IN

# Useful figure-width fractions.
FULL_WIDTH = 1.0
WIDE_WIDTH = 0.85
MEDIUM_WIDTH = 0.72
HALF_WIDTH = 0.50

# ---------------------------------------------------------------------------
# Colors from ctufit-thesis.cls
# ---------------------------------------------------------------------------

CTU_BLUE = "#007AC3"          # decoration / heading: RGB(0, 122, 195)
CTU_LIGHT_BLUE = "#C7DBF1"    # headbackgroundgray / backgroundgray
CTU_HEAD_GRAY = "#808082"     # approximate class headgray rgb(0.50,0.50,0.51)
BLACK = "#000000"
DARK_GRAY = "#333333"
MEDIUM_GRAY = "#777777"
LIGHT_GRAY = "#D0D0D0"

# Colorblind-conscious qualitative cycle, starting with CTU blue so plots feel
# connected to the thesis template. It is intentionally not too pastel because
# printed theses often reduce contrast.
COLOR_CYCLE = (
    CTU_BLUE,
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)

# Grayscale-safe cycle for plots that must work in black-and-white printing.
GRAYSCALE_CYCLE = (
    "#000000",
    "#444444",
    "#777777",
    "#AAAAAA",
)


# ---------------------------------------------------------------------------
# Font choices and sizes
# ---------------------------------------------------------------------------

# The class loads book[a4paper,11pt] and does not set a custom text font.
# Axis labels around 10 pt visually match the template's small figure captions.
BASE_FONT_SIZE_PT = 10.0
SMALL_FONT_SIZE_PT = 8.5
TITLE_FONT_SIZE_PT = 10.0
LEGEND_FONT_SIZE_PT = 8.5

# Leave these as None unless you explicitly set fonts in your LaTeX preamble.
# When saving PGF and including it with \input{...}, LaTeX will use the
# surrounding document font.
MAIN_FONT: Optional[str] = None
SANS_FONT: Optional[str] = None
MONO_FONT: Optional[str] = None

# Golden-ratio-ish default figure height/width.
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2

DEFAULT_OUTPUT_DIR = Path("figures")
DEFAULT_FORMATS = ("pgf", "pdf")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def use_pgf_backend() -> None:
    """Use Matplotlib's PGF backend.

    Call this before importing `matplotlib.pyplot` if you want Matplotlib to
    render PDF output through XeLaTeX. For PGF-only output, `setup_mpl()` is
    often enough, but using the PGF backend keeps everything consistent.
    """
    mpl.use("pgf")


def pt_to_in(points: float) -> float:
    """Convert TeX points to inches."""
    return points / 72.27


def cm_to_in(cm: float) -> float:
    """Convert centimeters to inches."""
    return cm / 2.54


def fig_size(
    width: Literal["text", "column"] | float = "text",
    *,
    fraction: float = FULL_WIDTH,
    aspect: float = GOLDEN_RATIO,
    height_in: Optional[float] = None,
) -> tuple[float, float]:
    """Return a thesis-consistent Matplotlib figure size in inches.

    Parameters
    ----------
    width:
        "text", "column", or a numeric width in TeX points.
    fraction:
        Fraction of the selected width. Examples: 1.0 full width, 0.72 medium,
        0.5 half width.
    aspect:
        Height / width. Default is the golden ratio.
    height_in:
        Explicit height in inches. Overrides `aspect` when provided.
    """
    if not (0 < fraction <= 1.5):
        raise ValueError("fraction should usually be in the interval (0, 1.5].")

    if isinstance(width, (int, float)):
        width_pt = float(width)
    elif width == "text":
        width_pt = TEXT_WIDTH_PT
    elif width == "column":
        width_pt = COLUMN_WIDTH_PT
    else:
        raise ValueError("width must be 'text', 'column', or a numeric TeX-point width.")

    width_in = pt_to_in(width_pt) * fraction
    height = height_in if height_in is not None else width_in * aspect
    return (width_in, height)


def _pgf_preamble(
    *,
    main_font: Optional[str],
    sans_font: Optional[str],
    mono_font: Optional[str],
    extra_preamble: Optional[str],
) -> str:
    """Build the LaTeX preamble used by Matplotlib's PGF backend."""
    lines: list[str] = [
        r"\usepackage{fontspec}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
    ]

    if main_font:
        lines.append(rf"\setmainfont{{{main_font}}}")
    if sans_font:
        lines.append(rf"\setsansfont{{{sans_font}}}")
    if mono_font:
        lines.append(rf"\setmonofont{{{mono_font}}}")
    if extra_preamble:
        lines.append(extra_preamble)

    return "\n".join(lines)


def setup_mpl(
    *,
    base_font_size: float = BASE_FONT_SIZE_PT,
    small_font_size: float = SMALL_FONT_SIZE_PT,
    title_font_size: float = TITLE_FONT_SIZE_PT,
    legend_font_size: float = LEGEND_FONT_SIZE_PT,
    font_family: Literal["serif", "sans-serif"] = "serif",
    main_font: Optional[str] = MAIN_FONT,
    sans_font: Optional[str] = SANS_FONT,
    mono_font: Optional[str] = MONO_FONT,
    use_latex_pgf: bool = True,
    pgf_texsystem: Literal["xelatex", "lualatex", "pdflatex"] = "xelatex",
    extra_pgf_preamble: Optional[str] = None,
    color_cycle: Sequence[str] = COLOR_CYCLE,
    grid: bool = True,
    grayscale: bool = False,
) -> None:
    """Apply thesis-wide Matplotlib defaults.

    Call this once near the start of every plotting script before creating
    figures.
    """
    chosen_cycle = GRAYSCALE_CYCLE if grayscale else color_cycle

    rc = {
        # Export defaults.
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "figure.dpi": 120,
        "figure.figsize": fig_size(),
        "figure.constrained_layout.use": True,

        # Fonts.
        "font.family": font_family,
        "font.size": base_font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": small_font_size,
        "ytick.labelsize": small_font_size,
        "legend.fontsize": legend_font_size,
        "figure.titlesize": title_font_size,

        # Text and math fallback when not using PGF.
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,

        # Lines and markers.
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.8,

        # Axes.
        "axes.linewidth": 0.8,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "axes.titleweight": "normal",
        "axes.grid": grid,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=list(chosen_cycle)),

        # Grid and ticks.
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.55,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # Legend.
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 1.0,

        # Make vector text editable/selectable in PDF/SVG outputs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    if use_latex_pgf:
        rc.update(
            {
                "pgf.texsystem": pgf_texsystem,
                "pgf.rcfonts": False,
                "pgf.preamble": _pgf_preamble(
                    main_font=main_font,
                    sans_font=sans_font,
                    mono_font=mono_font,
                    extra_preamble=extra_pgf_preamble,
                ),
                # Keep False for PGF. PGF/XeLaTeX handles text rendering.
                "text.usetex": False,
            }
        )

    mpl.rcParams.update(rc)


@contextmanager
def mpl_style(**kwargs):
    """Temporarily apply the thesis style inside a `with` block."""
    old = mpl.rcParams.copy()
    setup_mpl(**kwargs)
    try:
        yield
    finally:
        mpl.rcParams.update(old)


def savefig(
    fig,
    name: str | Path,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    formats: Iterable[str] = DEFAULT_FORMATS,
    transparent: bool = False,
    close: bool = False,
    **kwargs,
) -> list[Path]:
    """Save a figure in one or more formats using consistent defaults.

    Examples
    --------
    savefig(fig, "my_plot")
    savefig(fig, "my_plot", formats=("pgf",))
    savefig(fig, "my_plot", formats=("pdf", "png"))
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = Path(name).stem
    written: list[Path] = []

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = output_path / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, transparent=transparent, **kwargs)
        written.append(path)

    if close:
        plt.close(fig)

    return written

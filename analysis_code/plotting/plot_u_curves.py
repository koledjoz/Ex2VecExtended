from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from thesis_plot_style import setup_mpl, fig_size, savefig
# Optional, only if you want PGF-native output:
# from thesis_plot_style import use_pgf_backend

# use_pgf_backend()  # enable only if you want Matplotlib PGF backend
setup_mpl(use_latex_pgf=True, grid=True)

MODEL_LABELS = {
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

MODEL_COLORS = {
    "random": "#7f7f7f",
    "last_item": "#8c564b",
    "most_popular_past": "#9467bd",
    "bl_proxy": "#1f77b4",
    "bl_knn": "#17becf",
    "original": "#2ca02c",
    "extendedBase": "#ff7f0e",
    "extendedbase": "#ff7f0e",
    "extended_double": "#d62728",
    "extendedmlp": "#e377c2",
    "extended_doublemlp_loss": "#bcbd22",
}

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=fig_size(fraction=1.0, height_in=2.7))

x = np.linspace(0, 15, 1000)

models = ['original', 'extendedBase', 'extended_double', 'extended_doublemlp_loss']

for m in tqdm(models):
    weights_path = Path("../train_notebooks/weights") / m / "weights.pt"
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)

    alpha = weights["model_state_dict"]["alpha"].detach().cpu().numpy()
    beta = weights["model_state_dict"]["beta"].detach().cpu().numpy()
    gamma = weights["model_state_dict"]["gamma"].detach().cpu().numpy()

    y = x * alpha + x**2 * beta + gamma

    ax.plot(x, y, label=MODEL_LABELS.get(m, m.replace("_", " ")), color=MODEL_COLORS.get(m))

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\alpha x + \beta x^2 + \gamma$")
ax.legend(frameon=False)

savefig(
    fig,
    "model_quadratic_curves",
    output_dir="figures",
    formats=("pdf", "pgf"),
    close=True,
)
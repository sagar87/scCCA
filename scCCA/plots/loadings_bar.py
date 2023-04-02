from typing import List, Union

import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

from .utils import set_up_cmap, set_up_plot


def loading_bar(
    adata,
    model_key: str,
    state: str,
    factor: Union[int, List[int], None] = None,
    vector: str = "W_rna",
    design_dim=0,
    sign=1,
    lowest=4,
    highest=3,
    fat_bar=0.6,
    thin_bar=0.01,
    offset=0.1,
    fontsize=10,
    cmap=cm.RdBu,
    annot_bottom=False,
    ax=None,
):
    """
    Plot factor on a given embedding.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    model_key: str, optional (default: "X_scpca")
        Key for the fitted model.
    embedding: str, optional (default: "X_umap")
        Key for the embedding (e.g. UMAP, T-SNE).
    factor: int, list, optional (default: None)
        Factor(s) to plot. If None, then all factors are plotted.
    sign: float, optional (default: 1.0)
        Sign of the factor. Should be either 1.0 or -1.0.
    cmap: str, optional (default: "PiYG")
        Colormap for the scatterplot.
    colorbar_pos: str, optional (default: "right")
        Position of the colorbar.
    colorbar_width: str, optional (default: "3%")
        Width of the colorbar.
    orientation: str, optional (default: "vertical")
        Orientation of the colorbar. Should be either "vertical" or "horizontal".
    size: float, optional (default: 1)
        Marker/Dot size of the scatterplot.
    ncols: int, optional (default: 4)
        Number of columns for the subplots.
    width: int, optional (default: 4)
        Width of each subplot.
    height: int, optional (default: 3)
        Height of each subplot.
    ax: matplotlib.axes.Axes, optional (default: None)
        Axes object to plot on. If None, then a new figure is created.

    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes object.
    """
    ax = set_up_plot(
        adata,
        model_key,
        factor,
        _loadings_bar,
        state=state,
        vector=vector,
        design_dim=design_dim,
        sign=sign,
        lowest=lowest,
        highest=highest,
        fat_bar=fat_bar,
        thin_bar=thin_bar,
        offset=offset,
        fontsize=fontsize,
        cmap=cmap,
        annot_bottom=annot_bottom,
        ax=ax,
    )
    return ax


def _loadings_bar(
    adata,
    model_key: str,
    factor: int,
    state: Union[str, List[str]],
    vector: str = "W_rna",
    design_dim=0,
    sign=1,
    lowest=4,
    highest=3,
    fat_bar=0.6,
    thin_bar=0.01,
    offset=0.1,
    fontsize=10,
    cmap=cm.RdBu,
    annot_bottom=False,
    ax=None,
):
    model_dict = adata.uns[model_key]
    if isinstance(state, str):
        idx = model_dict["design"][state]
        # loadings = sign * model_dict[vector][idx][factor]
        loadings = sign * adata.varm[f"{model_key}_{vector}"][..., factor, idx]
    else:
        model_design = model_dict["design"]
        state_a = model_design[state[0]]
        state_b = model_design[state[1]]

        # loadings = sign * (model_dict[vector][state_b][factor] - model_dict[vector][state_a][factor])
        loadings = sign * (
            adata.varm[f"{model_key}_{vector}"][..., factor, state_b]
            - adata.varm[f"{model_key}_{vector}"][..., factor, state_a]
        )

    y = loadings
    other = len(loadings) - (lowest + highest)
    loadings_idx = np.argsort(loadings)

    w = np.concatenate(
        [
            np.ones(lowest) * fat_bar,
            np.ones(other) * thin_bar,
            np.ones(highest) * fat_bar,
        ]
    )

    cmap, norm = set_up_cmap(loadings, cmap)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = [mapper.to_rgba(v) for v in y[loadings_idx]]
    xticks = []
    for n, c in enumerate(w):
        xticks.append(sum(w[:n]) + w[n] / 2)

    if ax is None:
        plt.bar(xticks, height=y[loadings_idx], width=w, color=colors, alpha=0.9)
        ax = plt.gca()
    else:
        ax.bar(
            xticks,
            height=y[loadings_idx],
            width=w,
            color=colors,
            alpha=0.9,
        )

    ax.set_xticks([])
    # _ = ax.set_xticklabels(xticks_labels, rotation=90)
    ax.margins(x=0.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"Loading {factor}")

    for name, xtick in zip(adata.var_names[loadings_idx].values[:lowest].tolist(), xticks[:lowest]):
        if not annot_bottom:
            txt = ax.text(
                x=xtick,
                y=-offset,
                s=name,
                rotation=90,
                ha="center",
                color="white",
                va="top",
                fontweight="bold",
                fontsize=fontsize,
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
        else:
            ax.text(
                x=xtick,
                y=offset,
                s=name,
                rotation=90,
                ha="center",
                color="black",
                va="bottom",
                fontsize=fontsize,
            )

    for name, xtick in zip(adata.var_names[loadings_idx].values[-highest:].tolist(), xticks[-highest:]):
        if not annot_bottom:
            txt = ax.text(
                x=xtick,
                y=offset,
                s=name,
                rotation=90,
                ha="center",
                color="white",
                va="bottom",
                fontweight="bold",
                fontsize=fontsize,
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
        else:
            ax.text(
                x=xtick,
                y=-offset,
                s=name,
                rotation=90,
                ha="center",
                color="black",
                va="top",
                fontsize=fontsize,
            )
    return ax

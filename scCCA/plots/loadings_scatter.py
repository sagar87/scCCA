from collections import defaultdict
from typing import List

import matplotlib.cm as cm
import matplotlib.colors as co
import matplotlib.pyplot as plt
import numpy as np

from .utils import rand_jitter, repel_labels


def _get_state_data(
    adata,
    factor: int,
    model_key: str,
    states: List[str],
    sign: float,
    jitter: float,
    size_scale: float,
) -> dict:
    """
    Returns a dictionary with the data to plot the loadings for each state.
    """
    states_data = defaultdict(dict)
    for i, state in enumerate(states):

        loadings = sign * adata.uns[model_key]["states"][state][..., factor]
        order = np.argsort(loadings)

        norm = co.TwoSlopeNorm(vmin=loadings.min(), vmax=loadings.max(), vcenter=0)
        mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu"))
        colors = [mapper.to_rgba(v) for v in loadings]

        x = np.array([i] * loadings.shape[0])
        x = rand_jitter(x, jitter * np.abs(loadings))

        states_data[i]["x"] = x
        states_data[i]["y"] = loadings
        states_data[i]["c"] = colors
        states_data[i]["o"] = order

        states_data[i]["xo"] = x[order]
        states_data[i]["yo"] = loadings[order]
        states_data[i]["sz"] = np.abs(loadings) * size_scale

    return states_data


def _scatter(ax, state):
    ax.scatter(
        state["x"],
        state["y"],
        s=state["sz"],
        c=state["c"],
        zorder=1,
    )


def _style_scatter(ax):
    """
    Helper to stype the scatter plot.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.grid()
    ax.set_axisbelow(True)


def loadings_scatter(
    adata,
    model_key,
    factor,
    states: List[str] = [],
    genes: List[str] = [],
    diff: List[str] = [],
    highest=10,
    lowest=5,
    size_scale=1.0,
    sign=1.0,
    jitter=0.01,
    fontsize=12,
    repel=0.15,
    show_labels=0,
    return_order=False,
    ax=None,
):
    """
    Plot the loadings for a given factor and states.


    Parameters
    ----------
    adata: anndata.AnnData`
        Annotated data matrix.
    factor: int
        The factor to plot.
    model_key: str, optional (default: 'scpca')
        The key in `adata` where the model is stored.
    states: List[str], optional (default: [])
        The states to plot. If empty, all states are plotted.
    genes: List[str], optional (default: [])
        The genes to highlight. If genes and diff are empty, the genes with
        the highest and lowest expression are highlighted.
    diff: List[str], optional (default: [])
        The list must contain two states. The genes with the highest and lowest
    """
    # assert (
    #     (len(genes) > 0) or (len(diff) > 0)
    # ), "Only one of genes or diff can be specified."
    if ax is None:
        plt.figure()
        ax = plt.gca()

    states_data = _get_state_data(adata, factor, model_key, states, sign, jitter, size_scale)

    if len(genes) > 0:
        gene_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))

    if len(diff) > 0:
        diff_matrix = sign * get_loadings_diff(adata, model_key, diff[0], diff[1])
        diff_factor = diff_matrix[..., factor]
        order = np.argsort(diff_factor)

        if highest == 0:
            gene_idx = order[:lowest]
        else:
            gene_idx = np.concatenate([order[:lowest], order[-highest:]])

        magnitude = np.abs(diff_factor[gene_idx])
        genes = adata.var_names.to_numpy()[gene_idx]
        gene_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))

    for i, state in states_data.items():
        _scatter(ax, state)

        if len(genes) > 0:
            coords[i, :, 0] = state["x"][gene_bool]
            coords[i, :, 1] = state["y"][gene_bool]
            if i == show_labels:
                repel_labels(
                    ax,
                    state["x"][gene_bool],
                    state["y"][gene_bool],
                    adata.var_names[gene_bool],
                    label_pos_y=0.15,
                    k=repel,
                    fontsize=fontsize,
                )
        else:
            # mark lowest genes
            lowest_names = adata.var_names[state["o"]].values[:lowest].tolist()
            lowest_x = state["xo"][:lowest].tolist()
            lowest_y = state["yo"][:lowest].tolist()

            repel_labels(
                ax,
                lowest_x,
                lowest_y,
                lowest_names,
                label_pos_y=0.15,
                k=0.15,
                fontsize=fontsize,
            )

            # mark highest genes
            highest_names = adata.var_names[state["o"]].values[-highest:].tolist()
            highest_x = state["xo"][-highest:].tolist()
            highest_y = state["yo"][-highest:].tolist()

            repel_labels(
                ax,
                highest_x,
                highest_y,
                highest_names,
                label_pos_y=0.15,
                k=repel,
                fontsize=fontsize,
            )

    if len(genes) > 0 or len(diff) > 0:
        for j in range(len(states) - 1):
            for g in range(len(genes)):
                ax.plot(
                    [coords[j, g, 0], coords[j + 1, g, 0]],
                    [coords[j, g, 1], coords[j + 1, g, 1]],
                    color="k",
                    linestyle="--",
                    lw=0.5,
                )

    ax.set_xticks(list(states_data.keys()))
    ax.set_xticklabels(states)
    _style_scatter(ax)

    if return_order:
        if highest == 0:
            return order[:lowest]
        else:
            return (order[:lowest], order[-highest:])
    return ax


def loadings_scatter_highlight(
    adata,
    factor,
    model_key="scCCA",
    states: List[str] = [],
    genes=[],
    size_scale=1.0,
    sign=1.0,
    jitter=0.01,
    show_labels=0,
    fontsize=12,
    repel=0.15,
    ax=None,
):

    states_data = _get_state_data(adata, factor, model_key, states, sign, jitter, size_scale)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    gene_bool = adata.var_names.isin(genes)
    coords = np.zeros((len(states), len(genes), 2))
    for i, state in states_data.items():
        _scatter(ax, state)

        coords[i, :, 0] = state["x"][gene_bool]
        coords[i, :, 1] = state["y"][gene_bool]
        if i == show_labels:
            repel_labels(
                ax,
                state["x"][gene_bool],
                state["y"][gene_bool],
                adata.var_names[gene_bool],
                label_pos_y=0.15,
                k=repel,
                fontsize=fontsize,
            )

    for j in range(len(states) - 1):
        for g in range(len(genes)):
            ax.plot(
                [coords[j, g, 0], coords[j + 1, g, 0]],
                [coords[j, g, 1], coords[j + 1, g, 1]],
                color="k",
                linestyle="--",
                lw=0.5,
            )

    ax.set_xticks(list(states_data.keys()))
    ax.set_xticklabels(states)
    _style_scatter(ax)

    return ax

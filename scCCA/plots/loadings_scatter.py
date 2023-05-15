from collections import defaultdict
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.colors import Colormap

from .utils import rand_jitter, set_up_cmap


def _get_state_data(
    adata,
    factor: int,
    model_key: str,
    states: List[str],
    vector: str,
    sign: float,
    jitter: float,
    size_scale: float,
    cmap: Colormap,
) -> dict:
    """
    Returns a dictionary with the data to plot the loadings for each state.
    """
    model_dict = adata.uns[model_key]
    model_design = model_dict["design"]
    states_data = defaultdict(dict)
    for i, state in enumerate(states):
        state_idx = model_design[state]
        # loadings = sign * model_dict[vector][state_idx][factor]
        loadings = sign * adata.varm[f"{model_key}_{vector}"][..., factor, state_idx]
        order = np.argsort(loadings)
        cmap, norm = set_up_cmap(loadings, cmap)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
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


def _annotate_genes(ax, x, y, t, fontsize=10):
    texts = []
    for (x_i, y_i, t_i) in zip(x, y, t):
        texts.append(ax.text(x_i, y_i, t_i, fontsize=fontsize))
    return texts


def loadings_scatter(
    adata,
    model_key,
    factor,
    states: List[str] = [],
    genes: List[str] = [],
    diff: List[str] = [],
    vector: str = "W_rna",
    alpha: float = 1.0,
    highest=3,
    lowest=3,
    size_scale=1.0,
    sign=1.0,
    jitter=0.01,
    fontsize=10,
    show_labels=0,
    plot_diff=False,
    return_order=False,
    annotation_linewidth=0.5,
    cmap=cm.RdBu,
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
        fig = plt.figure()
        ax = plt.gca()

    if isinstance(show_labels, int):
        show_labels = [show_labels]

    states_data = _get_state_data(adata, factor, model_key, states, vector, sign, jitter, size_scale, cmap)
    model_dict = adata.uns[model_key]
    model_design = model_dict["design"]

    if len(genes) > 0:
        gene_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))

    elif len(diff) > 0:
        state_a = model_design[diff[0]]
        state_b = model_design[diff[1]]

        # diff_factor = sign * (model_dict[vector][state_b][factor] - model_dict[vector][state_a][factor])
        diff_factor = sign * (
            adata.varm[f"{model_key}_{vector}"][..., factor, state_b]
            - adata.varm[f"{model_key}_{vector}"][..., factor, state_a]
        )
        order = np.argsort(diff_factor)

        if highest == 0:
            gene_idx = order[:lowest]
        else:
            gene_idx = np.concatenate([order[:lowest], order[-highest:]])

        # magnitude = np.abs(diff_factor[gene_idx])
        diff_genes = adata.var_names.to_numpy()[gene_idx]
        gene_bool = adata.var_names.isin(diff_genes)
        coords = np.zeros((len(states), len(diff_genes), 2))

        if plot_diff:
            diff_genes = np.array([f"{gene} {diff:.2f}" for gene, diff in zip(diff_genes, diff_factor[gene_idx])])
    else:
        diff_genes = []

    texts = []
    for i, state in states_data.items():
        ax.scatter(
            state["x"],
            state["y"],
            s=state["sz"],
            c=state["c"],
            alpha=alpha,
            zorder=1,
        )

        if len(diff_genes) > 0:
            # print(state['c'])
            ax.scatter(
                state["x"][gene_bool],
                state["y"][gene_bool],
                s=state["sz"][gene_bool],
                c=np.asarray(state["c"])[gene_bool],
                alpha=1.0,
                zorder=2,
            )

            coords[i, :, 0] = state["x"][gene_bool]
            coords[i, :, 1] = state["y"][gene_bool]

            if i in show_labels:
                texts += _annotate_genes(ax, state["x"][gene_idx], state["y"][gene_idx], diff_genes, fontsize=fontsize)
                # adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=annotation_linewidth), ax=ax)

            # gene_list = adata.var_names[gene_bool].tolist()
        else:
            # mark lowest diff_genes
            if i in show_labels:
                order = state["o"]

                if lowest != 0:
                    lowest_names = adata.var_names[order].values[:lowest].tolist()
                    lowest_x = state["xo"][:lowest].tolist()
                    lowest_y = state["yo"][:lowest].tolist()

                    texts += _annotate_genes(ax, lowest_x, lowest_y, lowest_names, fontsize=fontsize)

                # mark highest diff_genes
                if highest != 0:
                    highest_names = adata.var_names[order].values[-highest:].tolist()
                    highest_x = state["xo"][-highest:].tolist()
                    highest_y = state["yo"][-highest:].tolist()

                    texts += _annotate_genes(ax, highest_x, highest_y, highest_names, fontsize=fontsize)

    # print(texts)

    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=annotation_linewidth), ax=ax)

    if len(diff) > 0 or len(diff_genes) > 0:
        for j in range(len(states) - 1):
            for g in range(len(diff_genes)):
                ax.plot(
                    [coords[j, g, 0], coords[j + 1, g, 0]],
                    [coords[j, g, 1], coords[j + 1, g, 1]],
                    alpha=0.5,
                    color="k",
                    linestyle="--",
                    lw=0.5,
                )

    ax.set_xticks(list(states_data.keys()))
    ax.set_xticklabels(states)
    _style_scatter(ax)

    if return_order:
        if highest == 0:
            return adata.var_names[order[:lowest]]
        else:
            return (adata.var_names[order[:lowest]], adata.var_names[order[-highest:]])
    return fig, ax


# def loadings_scatter_highlight(
#     adata,
#     factor,
#     model_key="scCCA",
#     vector="W_lin",
#     states: List[str] = [],
#     genes=[],
#     size_scale=1.0,
#     sign=1.0,
#     jitter=0.01,
#     show_labels=0,
#     fontsize=12,
#     repel=0.15,
#     ax=None,
# ):

#     states_data = _get_state_data(adata, factor, model_key, states, vector, sign, jitter, size_scale)

#     if ax is None:
#         plt.figure()
#         ax = plt.gca()

#     gene_bool = adata.var_names.isin(genes)
#     coords = np.zeros((len(states), len(genes), 2))
#     for i, state in states_data.items():
#         _scatter(ax, state)

#         coords[i, :, 0] = state["x"][gene_bool]
#         coords[i, :, 1] = state["y"][gene_bool]
#         if i == show_labels:
#             repel_labels(
#                 ax,
#                 state["x"][gene_bool],
#                 state["y"][gene_bool],
#                 adata.var_names[gene_bool],
#                 label_pos_y=0.15,
#                 k=repel,
#                 fontsize=fontsize,
#             )

#     for j in range(len(states) - 1):
#         for g in range(len(genes)):
#             ax.plot(
#                 [coords[j, g, 0], coords[j + 1, g, 0]],
#                 [coords[j, g, 1], coords[j + 1, g, 1]],
#                 color="k",
#                 linestyle="--",
#                 lw=0.5,
#             )

#     ax.set_xticks(list(states_data.keys()))
#     ax.set_xticklabels(states)
#     _style_scatter(ax)

#     return ax

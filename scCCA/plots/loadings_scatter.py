from collections import defaultdict
from typing import List, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.colors import Colormap

from ..utils import get_diff_genes, get_factor_enrichment
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
        states_data[i]["go"] = adata.var_names[order]

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
    geneset: Union[str, None] = None,
    vector: str = "W_rna",
    alpha: float = 1.0,
    highest=3,
    lowest=3,
    size_scale=1.0,
    sign=1.0,
    jitter=0.01,
    fontsize=10,
    geneset_top_genes: int = 100,
    geneset_bottom_genes: int = 0,
    show_labels=0,
    show_geneset: bool = False,
    show_diff: bool = False,
    return_order=False,
    annotation_linewidth=0.5,
    format_func=lambda x: x,
    cmap=cm.RdBu,
    ax=None,
):
    """
    Scatter plot of factor loadings for a given factor in each state.

    Arguments
    ---------
    adata: AnnData
        AnnData object with the fitted model.
    model_key: str
        Key used to store the fitted model in adata.uns.
    factor: int
        The factor to plot.
    states: List[str], optional (default: [])
        The states to include in the plot.
    genes: List[str], optional (default: [])
        The genes to include in the plot.
    diff: List[str], optional (default: [])
        The genes to highlight in the plot.
    geneset: str or None, optional (default: None)
        Name of a gene set to include in the plot. Requires gseapy package.
    vector: str, optional (default: "W_rna")
        Vector to use for plotting the loadings.
    alpha: float, optional (default: 1.0)
        Transparency of the scatter plot.
    highest: int, optional (default: 3)
        Number of genes with highest loadings to plot per state.
    lowest: int, optional (default: 3)
        Number of genes with lowest loadings to plot per state.
    size_scale: float, optional (default: 1.0)
        Scaling factor for the gene symbol size.
    sign: float, optional (default: 1.0)
        Sign of the loadings.
    jitter: float, optional (default: 0.01)
        Jittering factor for the x-axis to reduce overlap.
    fontsize: int, optional (default: 10)
        Font size for gene labels.
    geneset_top_genes: int, optional (default: 100)
        Number of genes from the gene set to plot with the highest loadings.
    geneset_bottom_genes: int, optional (default: 0)
        Number of genes from the gene set to plot with the lowest loadings.
    show_labels: int, optional (default: 0)
        Show gene labels for top `show_labels` genes with the highest loadings.
    show_geneset: bool, optional (default: False)
        Show the gene set as a solid line.
    show_diff: bool, optional (default: False)
        Show the differential genes as a dashed line.
    return_order: bool, optional (default: False)
        Return the order of genes plotted.
    annotation_kwargs: dict, optional (default: {})
        Additional keyword arguments for gene label annotations.

    Returns
    -------
    order: np.ndarray
        The order of genes plotted (only if `return_order` is True).
    """
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()

    if isinstance(show_labels, int):
        show_labels = [show_labels]

    states_data = _get_state_data(adata, factor, model_key, states, vector, sign, jitter, size_scale, cmap)

    if geneset is not None:
        gene_sets_dict = {}

    if len(genes) > 0:
        # if genes are specified just show them
        gene_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))

    elif geneset is not None and len(diff) == 0:
        # if geneset but diff is specified
        for s, current_state in enumerate(states):
            enrichment_results = get_factor_enrichment(
                adata,
                model_key,
                current_state,
                factor,
                highest=geneset_top_genes,
                lowest=geneset_bottom_genes,
                sign=sign,
                geneset=geneset,
            )
            gene_sets_dict[current_state] = enrichment_results
        # import pdb; pdb.set_trace()
        diff_genes = []

    elif geneset is None and len(diff) > 0:
        df = get_diff_genes(adata, model_key, states, factor, highest=highest, lowest=lowest, sign=sign, vector=vector)
        diff_weights = df["diff"].to_numpy()
        diff_genes = df["gene"].to_numpy()
        gene_idx = df["index"].to_numpy()
        gene_bool = adata.var_names.isin(diff_genes)
        coords = np.zeros((len(states), len(diff_genes), 2))

        if show_diff:
            diff_genes = np.array([f"{gene} {diff:.2f}" for gene, diff in zip(diff_genes, diff_weights)])
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
        elif geneset is None:
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
        elif geneset is not None:
            if states[i] in gene_sets_dict.keys():

                if i in show_labels:
                    order = state["o"]
                    df = gene_sets_dict[states[i]]

                    # gene_set_x = []
                    # gene_set_y = []
                    # gene_set_n = []

                    full_gene_set = defaultdict(dict)

                    for k, row in df.head(5).iterrows():
                        genes = row["Genes"].split(";")
                        genes_index = np.where(state["go"].isin(genes))[0]

                        # gene_set_x += [state["xo"][genes_index[0]]]
                        # gene_set_y += [state["yo"][genes_index[0]]]
                        # gene_set_n += [format_func(row['Term'])]

                        # code here
                        full_gene_set[format_func(row["Term"])]["x"] = state["xo"][genes_index]
                        full_gene_set[format_func(row["Term"])]["y"] = state["yo"][genes_index]
                        full_gene_set[format_func(row["Term"])]["name"] = format_func(row["Term"])

                    gene_names = list(full_gene_set.keys())
                    gene_xcoords = [full_gene_set[k]["x"][0] for k in gene_names]
                    gene_ycoords = [full_gene_set[k]["y"][0] for k in gene_names]
                    texts += _annotate_genes(ax, gene_xcoords, gene_ycoords, gene_names, fontsize=fontsize)

                # df['Genes'].split(';')[9]
            # print('Got here')

    if len(texts) > 0:
        if show_geneset:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0), ax=ax)
        else:
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
    if show_geneset:
        for t in texts:
            x_end, y_end = t.get_position()
            gene_name = t.get_text()
            for x_start, y_start in zip(full_gene_set[gene_name]["x"], full_gene_set[gene_name]["y"]):
                ax.plot(
                    [x_start, x_end],
                    [y_start, y_end],
                    alpha=0.3,
                    color="k",
                    linestyle="--",
                    lw=0.2,
                )

        # import pdb;pdb.set_trace()

    ax.set_xticks(list(states_data.keys()))
    ax.set_xticklabels(states)
    _style_scatter(ax)

    if return_order:
        if highest == 0:
            return adata.var_names[order[:lowest]]
        else:
            return (adata.var_names[order[:lowest]], adata.var_names[order[-highest:]])

    return ax

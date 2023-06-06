import textwrap
from collections import namedtuple
from typing import List, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.colors import Colormap

from ..utils import get_diff_enrichment, get_diff_genes, get_factor_enrichment
from ..utils.data import _get_model_design
from .utils import rand_jitter, set_up_cmap, set_up_plot


def loadings_scatter(
    adata,
    model_key: str,
    factor: Union[int, List[int], None] = None,
    states: List[str] = [],
    genes: List[str] = [],
    diff: List[str] = [],
    geneset=None,
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
    show_geneset=False,
    show_diff: bool = False,
    geneset_top_genes: int = 100,
    geneset_bottom_genes: int = 0,
    organism="Human",
    cmap=cm.RdBu,
    format_func=lambda x: textwrap.fill(x.split(" (")[0], width=20),
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
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
        _loadings_scatter,
        states=states,
        genes=genes,
        diff=diff,
        geneset=geneset,
        vector=vector,
        sign=sign,
        highest=highest,
        lowest=lowest,
        alpha=alpha,
        jitter=jitter,
        fontsize=fontsize,
        show_labels=show_labels,
        show_geneset=show_geneset,
        show_diff=show_diff,
        format_func=format_func,
        geneset_top_genes=geneset_top_genes,
        geneset_bottom_genes=geneset_bottom_genes,
        organism=organism,
        cmap=cmap,
        ncols=ncols,
        width=width,
        height=height,
        ax=ax,
    )
    return ax


def _loadings_scatter(
    adata,
    model_key,
    factor,
    states: List[str] = [],
    genes: List[str] = [],
    diff: List[str] = [],
    geneset: Union[str, None] = None,
    sign: Union[int, float] = 1.0,
    vector: str = "W_rna",
    highest: int = 10,
    lowest: int = 0,
    alpha: float = 1.0,
    size_scale: float = 1.0,
    jitter: float = 0.01,
    fontsize: int = 10,
    geneset_top_genes: int = 100,
    geneset_bottom_genes: int = 0,
    organism="Human",
    show_labels: int = 0,
    show_geneset: bool = False,
    show_diff: bool = False,
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

    if len(diff) > 0 and len(diff) != 2:
        raise ValueError("The argument must specify exactly 2 states.")

    diff_flag = True if len(diff) == 2 else False
    gene_flag = True if len(genes) > 0 else False
    enrichment_flag = True if geneset is not None else False

    # if diff_flag and gene_flag:
    #     # TODO: diff_flag + gene_flag should calculate the difference between specified genes
    #     raise NotImplementedError("Please choose to plot either a specfic set of genes or differences between states.")

    if gene_flag and diff_flag and enrichment_flag:
        raise ValueError("This is an invalid choice.")

    if gene_flag and not diff_flag and enrichment_flag:
        raise ValueError("Please choose to visualise either gene enrichment results or specific genes.")

    # convert show label indices to state
    model_design = _get_model_design(adata, model_key, reverse=True)

    if isinstance(show_labels, int):
        show_labels = [model_design[show_labels]]

    if isinstance(show_labels, list):
        show_labels = [model_design[label] if isinstance(label, int) else label for label in show_labels]

    states_data = _get_state_data(adata, factor, model_key, states, vector, sign, jitter, size_scale, cmap)

    if gene_flag and diff_flag and not enrichment_flag:
        # TODO: compute differece across selected genes
        genes_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))

    elif gene_flag and not diff_flag and not enrichment_flag:
        # just plot genes
        genes_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))
        diff_genes = []

    elif not gene_flag and diff_flag and enrichment_flag:
        # compute enrichment of diff genes
        enrichment_results = get_diff_enrichment(
            adata,
            model_key,
            states,
            factor,
            highest=geneset_top_genes,
            lowest=geneset_bottom_genes,
            sign=sign,
            geneset=geneset,
            organism=organism,
        ).head(highest)

        terms = dict(
            zip(
                [format_func(t) for t in enrichment_results["Term"]],
                enrichment_results["Genes"].apply(lambda x: x.split(";")).tolist(),
            )
        )
        coords = np.zeros((len(states), enrichment_results.shape[0], 2))

    elif not gene_flag and diff_flag and not enrichment_flag:
        diff_genes = get_diff_genes(
            adata, model_key, states, factor, highest=highest, lowest=lowest, sign=sign, vector=vector
        )
        genes = diff_genes["gene"].tolist()
        values = diff_genes["diff"].tolist()
        diff_values = dict(zip(genes, values))

        genes_bool = adata.var_names.isin(genes)
        coords = np.zeros((len(states), len(genes), 2))

    elif not gene_flag and not diff_flag and enrichment_flag:
        # if geneset but diff is specified
        geneset_results = {}

        for state in show_labels:
            enrichment_results = get_factor_enrichment(
                adata,
                model_key,
                state,
                factor,
                highest=geneset_top_genes,
                lowest=geneset_bottom_genes,
                sign=sign,
                geneset=geneset,
                organism=organism,
            ).head(highest)
            geneset_results[state] = dict(
                zip(
                    [format_func(t) for t in enrichment_results["Term"]],
                    enrichment_results["Genes"].apply(lambda x: x.split(";")).tolist(),
                )
            )
    else:
        pass

    texts = []
    for i, (state_name, state) in enumerate(states_data.items()):
        ax.scatter(state.x, state.y, s=state.sz, c=state.c, alpha=alpha, zorder=1)

        if gene_flag and diff_flag and not enrichment_flag:
            # TODO: compute differece across selected genes
            coords[i, :, 0] = state.x[genes_bool]
            coords[i, :, 1] = state.y[genes_bool]

            if state_name in show_labels:
                texts += _annotate_genes(
                    ax, state.x[genes_bool], state.y[genes_bool], state.g[genes_bool], fontsize=fontsize
                )

        elif gene_flag and not diff_flag and not enrichment_flag:
            # just plot genes

            if state_name in show_labels:
                texts += _annotate_genes(
                    ax, state.x[genes_bool], state.y[genes_bool], state.g[genes_bool], fontsize=fontsize
                )

        elif not gene_flag and diff_flag and enrichment_flag:
            # compute enrichment of diff genes

            x_coords, y_coords, labels = [], [], []

            for j, (term, genes) in enumerate(terms.items()):
                genes_bool = state.g.isin(genes)
                x = i  # state.x[genes_bool].median()
                y = state.y[genes_bool].mean()
                x_coords.append(x)
                y_coords.append(y)

                coords[i, j, 0] = x
                coords[i, j, 1] = y

                labels.append(term)

            if state_name in show_labels:
                texts += _annotate_genes(ax, x_coords, y_coords, labels, fontsize=fontsize)

        elif not gene_flag and not diff_flag and enrichment_flag:

            if state_name in geneset_results:
                terms = geneset_results[state_name]
                x_coords, y_coords, labels = [], [], []

                for j, (term, genes) in enumerate(terms.items()):
                    genes_bool = state.g.isin(genes)
                    x = i  # state.x[genes_bool].median()
                    y = state.y[genes_bool].mean()
                    x_coords.append(x)
                    y_coords.append(y)
                    labels.append(term)

                texts += _annotate_genes(ax, x_coords, y_coords, labels, fontsize=fontsize)

        elif not gene_flag and diff_flag and not enrichment_flag:
            coords[i, :, 0] = state.x[genes_bool]
            coords[i, :, 1] = state.y[genes_bool]

            gene_labels = state.g[genes_bool]
            if show_diff:
                gene_labels = np.array([f"{gene} {diff_values[gene]:.2f}" for gene in gene_labels])

            if state_name in show_labels:
                texts += _annotate_genes(ax, state.x[genes_bool], state.y[genes_bool], gene_labels, fontsize=fontsize)

        else:
            # mark lowest diff_genes
            if state_name in show_labels:
                if lowest != 0:
                    texts += _annotate_genes(
                        ax,
                        state.xo[:lowest].tolist(),
                        state.yo[:lowest].tolist(),
                        state.go.values[:lowest].tolist(),
                        fontsize=fontsize,
                    )

                if highest != 0:
                    texts += _annotate_genes(
                        ax,
                        state.xo[-highest:].tolist(),
                        state.yo[-highest:].tolist(),
                        state.go.values[-highest:].tolist(),
                        fontsize=fontsize,
                    )

    if len(texts) > 0:
        if show_geneset:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0), ax=ax)
        else:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=annotation_linewidth), ax=ax)

    if diff_flag:
        for j in range(len(states) - 1):
            for g in range(coords.shape[1]):
                ax.plot(
                    [coords[j, g, 0], coords[j + 1, g, 0]],
                    [coords[j, g, 1], coords[j + 1, g, 1]],
                    alpha=0.5,
                    color="k",
                    linestyle="--",
                    lw=0.5,
                )
    # if show_geneset:
    #     for t in texts:
    #         x_end, y_end = t.get_position()
    #         gene_name = t.get_text()
    #         for x_start, y_start in zip(full_gene_set[gene_name]["x"], full_gene_set[gene_name]["y"]):
    #             ax.plot(
    #                 [x_start, x_end],
    #                 [y_start, y_end],
    #                 alpha=0.3,
    #                 color="k",
    #                 linestyle="--",
    #                 lw=0.2,
    #             )

    ax.set_xticks([i for i in range(len(states))])
    ax.set_title(f'Factor {factor}')
    ax.set_ylabel('Loading weight')
    ax.set_xlabel('State')
    ax.set_xticklabels(states)
    _style_scatter(ax)

    return ax


StateData = namedtuple("StateData", ["x", "y", "c", "o", "g", "xo", "yo", "sz", "go"])


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
    Get the data necessary for plotting the loadings for each state.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data and model information.
    factor : int
        Index of the factor for which to retrieve the loadings.
    model_key : str
        Key to access the model information in `adata.uns`.
    states : List[str]
        List of state names for which to retrieve the loadings.
    vector : str
        Name of the vector to retrieve the loadings from.
    sign : float
        Sign multiplier to apply to the loadings.
    jitter : float
        Amount of jitter to apply to the x-axis values.
    size_scale : float
        Scaling factor for the size of the markers.
    cmap : Colormap
        Matplotlib colormap for coloring the markers.

    Returns
    -------
    states_data : Dict[int, Dict[str, Any]]
        Dictionary containing the data for each state, with the following keys:
        - 'x': x-axis values (jittered)
        - 'y': loadings
        - 'c': marker colors
        - 'o': order of the loadings
        - 'xo': x-axis values (ordered)
        - 'yo': loadings (ordered)
        - 'sz': marker sizes
        - 'go': gene order

    Notes
    -----
    This function retrieves the loadings for each state and prepares the data
    required for plotting the loadings.

    The `vector` should be a valid key to access the loadings in `adata.varm`.
    The `cmap` should be a valid Matplotlib colormap.

    Examples
    --------
    states_data = get_state_data(adata, 0, 'pca', ['state_A', 'state_B'], 'W_rna', 1.0, 0.1, 0.5, 'viridis')
    """
    model_design = _get_model_design(adata, model_key)

    states_data = {}

    for i, state in enumerate(states):
        state_idx = model_design[state]
        loadings = sign * adata.varm[f"{model_key}_{vector}"][..., factor, state_idx]
        order = np.argsort(loadings)
        cmap, norm = set_up_cmap(loadings, cmap)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = [mapper.to_rgba(v) for v in loadings]

        x = np.array([i] * loadings.shape[0])
        x = rand_jitter(x, jitter * np.abs(loadings))

        states_data[state] = StateData(
            x=x,
            y=loadings,
            c=colors,
            o=order,
            g=adata.var_names,
            xo=x[order],
            yo=loadings[order],
            sz=np.abs(loadings) * size_scale,
            go=adata.var_names[order],
        )
        
    
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

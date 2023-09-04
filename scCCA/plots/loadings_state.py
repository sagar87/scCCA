import textwrap
from typing import List, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import get_diff_enrichment, get_diff_genes
from ..utils.data import _get_model_design
from .utils import set_up_cmap, set_up_plot


def loadings_state(
    adata,
    model_key: str,
    states: List[str] = [],
    factor: Union[int, List[int], None] = None,
    variable: str = "W_rna",
    highest=10,
    lowest=10,
    threshold=None,
    sign=1.0,
    geneset=None,
    geneset_top_genes: int = 100,
    geneset_bottom_genes: int = 0,
    organism="Human",
    cmap=cm.RdBu,
    colorbar_pos="right",
    colorbar_width="3%",
    orientation="vertical",
    fontsize=10,
    pad=0.1,
    show_corr=False,
    show_rank=False,
    show_diff=False,
    show_lines=False,
    size_func=lambda x: 10,
    text_func=lambda x: textwrap.fill(x.split(" (")[0], width=20),
    sharey=False,
    sharex=False,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    text_kwargs: dict = {},
    ax=None,
):
    """
    Visualize the loading state of a given model on the data.

    Parameters
    ----------
    adata : 
        The annotated data matrix.
    model_key :
        Key to access the model from the data.
    states : 
        List of states to consider. Default is an empty list.
    factor :
        Factor or factors to consider. Default is None.
    variable :
        Variable to consider. Default is "W_rna".
    highest :
        Number of highest values to consider. Default is 10.
    lowest :
        Number of lowest values to consider. Default is 10.
    threshold : 
        Threshold value to consider. Default is None.
    sign :
        Sign value. Default is 1.0.
    geneset : 
        Set of genes to consider. Default is None.
    geneset_top_genes :
        Number of top genes in the gene set to consider. Default is 100.
    geneset_bottom_genes :
        Number of bottom genes in the gene set to consider. Default is 0.
    organism :
        Organism to consider. Default is "Human".
    cmap 
        Colormap to use. Default is cm.RdBu.
    colorbar_pos :
        Position of the colorbar. Default is "right".
    colorbar_width :
        Width of the colorbar. Default is "3%".
    orientation :
        Orientation of the plot. Default is "vertical".
    fontsize :
        Font size to use. Default is 10.
    pad :
        Padding value. Default is 0.1.
    show_corr :
        Whether to show correlation. Default is False.
    show_rank :
        Whether to show rank. Default is False.
    show_diff :
        Whether to show difference. Default is False.
    show_lines :
        Whether to show lines. Default is False.
    size_func :
        Function to determine size. Default is a function that returns 10.
    text_func :
        Function to format text. Default is a function that wraps text.
    sharey :
        Whether to share y-axis. Default is False.
    sharex :
        Whether to share x-axis. Default is False.
    ncols :
        Number of columns. Default is 4.
    width :
        Width of the plot. Default is 4.
    height :
        Height of the plot. Default is 3.
    text_kwargs :
        Keyword arguments for text formatting. Default is an empty dictionary.
    ax : 
        Axis object. Default is None.

    Returns
    -------
    ax : 
        Configured axis object.

    Notes
    -----
    This function visualizes the loading state of a given model on the data. 
    It provides various customization options for visualization.
    """
    ax = set_up_plot(
        adata,
        model_key,
        factor,
        _loadings_state,
        states=states,
        variable=variable,
        highest=highest,
        lowest=lowest,
        threshold=threshold,
        sign=sign,
        geneset=geneset,
        geneset_top_genes=geneset_top_genes,
        geneset_bottom_genes=geneset_bottom_genes,
        organism=organism,
        size_func=size_func,
        text_func=text_func,
        colorbar_pos=colorbar_pos,
        colorbar_width=colorbar_width,
        orientation=orientation,
        fontsize=fontsize,
        pad=pad,
        cmap=cmap,
        show_corr=show_corr,
        show_rank=show_rank,
        show_diff=show_diff,
        show_lines=show_lines,
        sharey=sharey,
        sharex=sharex,
        ncols=ncols,
        width=width,
        height=height,
        text_kwargs=text_kwargs,
        ax=ax,
    )
    return ax


def _loadings_state(
    adata,
    model_key,
    factor,
    states,
    variable="W_rna",
    highest=10,
    lowest=10,
    threshold=None,
    geneset=None,
    sign=1.0,
    size_func=lambda x: 10,
    text_func=lambda x: x,
    geneset_top_genes: int = 100,
    geneset_bottom_genes: int = 0,
    organism="Human",
    cmap=cm.RdBu,
    colorbar_pos="right",
    colorbar_width="3%",
    orientation="vertical",
    pad=0.1,
    fontsize=10,
    show_corr=False,
    show_lines=False,
    show_rank=False,
    show_diff=False,
    text_kwargs={},
    ax=None,
):

    model_design = _get_model_design(adata, model_key)
    state_a, state_b = model_design[states[0]], model_design[states[1]]
    loadings = adata.varm[f"{model_key}_{variable}"]
    x, y = loadings[..., factor, state_a], loadings[..., factor, state_b]
    diff = sign * (y - x)
    diag = np.linspace(np.quantile(sign * x, 0.01), np.quantile(sign * x, 0.99))
    cmap, norm = set_up_cmap(diff)
    s = size_func(diff)

    # import pdb; pdb.set_trace()

    im = ax.scatter(sign * x, sign * y, s=s, c=diff, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=pad)
    plt.gcf().colorbar(im, cax=cax, orientation=orientation)

    ax.plot(diag, diag, ls="--", color="k", lw=0.5)
    if show_lines:
        ax.axvline(0, ls="--", color="k", lw=0.5)
        ax.axhline(0, ls="--", color="k", lw=0.5)
    ax.set_xlabel(f"Loadings ({states[0]})")
    ax.set_ylabel(f"Loadings ({states[1]})")
    ax.set_title(f"Factor {factor}")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")

    texts = []
    if geneset is not None:
        diff_geneset = get_diff_enrichment(
            adata,
            model_key,
            states,
            factor=factor,
            geneset=geneset,
            sign=sign,
            organism=organism,
            highest=geneset_top_genes,
            lowest=geneset_bottom_genes,
        )
        # import pdb;pdb.set_trace()

        for i, row in diff_geneset.head(highest).iterrows():
            label = str(row["Term"])

            if show_rank:
                label = f"{i+1} " + label
            genes = row["Genes"].split(";")
            # print(genes)
            # print(adata.var_names.isin(genes))
            # import pdb; pdb.set_trace()
            is_upper = np.all([gene.isupper() for gene in genes])

            var_names = adata.var_names
            if is_upper:
                var_names = var_names.str.upper()

            # gene_rep = np.random.choice(adata.var_names[adata.var_names.str.upper().isin(genes)])
            # import pdb;pdb.set_trace()
            # t = ax.text(x[adata.var_names == gene_rep].item(), y[adata.var_names == gene_rep].item(), s=text_func(label), fontsize=fontsize)
            t = ax.text(
                sign * x[var_names.isin(genes)].mean(),
                sign * y[var_names.isin(genes)].mean(),
                s=text_func(label),
                fontsize=fontsize,
            )
            texts.append(t)

    else:
        if threshold:
            diff_genes = get_diff_genes(
                adata,
                model_key,
                states,
                factor,
                vector=variable,
                sign=sign,
                highest=adata.shape[1],
                threshold=threshold,
            )
            diff_genes = diff_genes[diff_genes.significant]
        else:
            diff_genes = get_diff_genes(
                adata, model_key, states, factor, vector=variable, sign=sign, highest=highest, lowest=lowest
            )

        for i, row in diff_genes.iterrows():
            label = str(row["gene"])

            if show_rank:
                label = f"{i} " + label

            if show_diff:
                label += f' {row["diff"]:.2f}'
            t = ax.text(sign * row[states[0]], sign * row[states[1]], s=label, fontsize=fontsize)
            texts.append(t)

    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), ax=ax, **text_kwargs)

    if show_corr:
        correlation = np.corrcoef(x, y)[0, 1]
        ax.text(0.95, 0.95, f"Correlation: {correlation:.2f}", ha="right", va="top", transform=ax.transAxes)

    return ax

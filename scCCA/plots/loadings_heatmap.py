import textwrap

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import masked_array
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

from ..utils import get_significant_genes


def loadings_heatmap(
    adata,
    model_key,
    states=None,
    factor=None,
    highlight=True,
    variable="W_rna",
    annot=False,
    annot_off_diag=False,
    grid_linewidth=1,
    metric="corr",
    fmt="{x:.2f}",
    normalize=False,
    cmap="RdBu",
    vmin=-0.6,
    vmax=0.6,
    colorbar_pos_up=[1.01, 0.05, 0.03, 0.4],
    colorbar_pos_down=[1.01, 0.55, 0.03, 0.4],
    cmap2="Greens",
    vmin2=0,
    vmax2=1,
    ax=None,
):

    if ax is None:
        _ = plt.figure(figsize=(12, 12))
        ax = plt.gca()

    dims = adata.varm[f"{model_key}_{variable}"].shape

    if states is None:
        states = [k for k, v in sorted(list(adata.uns[model_key]["design"].items()), key=lambda x: x[1])]
        states_idx = [v for k, v in sorted(list(adata.uns[model_key]["design"].items()), key=lambda x: x[1])]
    else:
        states_idx = [adata.uns[model_key]["design"][state] for state in states]

    num_states = len(states)

    if factor is None:
        W = adata.varm[f"{model_key}_{variable}"]
        W = W[..., states_idx].reshape(dims[0], -1)

        num_factors = adata.uns[model_key]["model"]["num_factors"]
        factor = [i for i in range(num_factors)]
    else:
        if isinstance(factor, int):
            factor = [factor]
        num_factors = len(factor)
        W = adata.varm[f"{model_key}_{variable}"][..., states_idx]
        W = W[:, factor].reshape(dims[0], -1)

    print(states_idx)
    if normalize:
        W = normalize_columns(W)

    # import pdb;pdb.set_trace()
    if metric == "corr":
        dist = np.corrcoef(W.T)
    else:
        dist = cdist(W.T, W.T, metric=metric)

    labels = [textwrap.fill(f"F{i}|" + f"{j}", width=30) for i in factor for j in states]

    if highlight:
        mask = block_diag_scipy(np.ones((num_states, num_states)), num_factors).astype(bool)
        dist_diag = masked_array(dist, mask)
        dist_off = masked_array(dist, ~mask)
        im_diag, cbar_diag = heatmap(
            dist_diag,
            labels,
            labels,
            cmap=cmap,
            grid_linewidth=grid_linewidth,
            grid_step=num_states,
            colorbar_pos=colorbar_pos_up,  # [1.01, 0.05, 0.03, 0.4],
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        im_off, cbar_off = heatmap(
            dist_off,
            labels,
            labels,
            cmap=cmap2,
            vmin=vmin2,
            vmax=vmax2,
            grid_linewidth=grid_linewidth,
            colorbar_pos=colorbar_pos_down,
            grid_step=num_states,
            ax=ax,
        )
        if annot:
            if annot_off_diag:
                _ = annotate_heatmap(im_diag, valfmt=fmt)
            else:
                _ = annotate_heatmap(im_off, valfmt=fmt)

    else:
        im, cbar = heatmap(
            dist,
            labels,
            labels,
            cmap=cmap,
            grid_linewidth=grid_linewidth,
            grid_step=num_states,
            colorbar_pos=[1.04, 0.2, 0.01, 0.3],
            ax=ax,
        )
        if annot:
            _ = annotate_heatmap(im, valfmt=fmt)

    return W


def loadings_diff_heatmap(
    adata,
    model_key,
    states,
    swap_axes=False,
    annot=True,
    fmt="{x:.0f}",
    threshold=1.96,
    ax=None,
    heatmap_kw={},
    text_kw={},
):
    if ax is None:
        ax = plt.gca()

    data = get_significant_genes(adata, model_key, states, threshold=threshold)

    pivot = data.pivot_table(index="gene", columns="factor", values="diff")
    if swap_axes:
        pivot = pivot.T
    colMap = cm.RdBu
    colMap.set_bad(color="lightgrey")
    im, cbar = heatmap(
        pivot.values, row_labels=pivot.index.tolist(), col_labels=pivot.columns.tolist(), cmap=colMap, *heatmap_kw
    )

    if annot:
        annotate_heatmap(im, valfmt=fmt, *text_kw)
    return im, cbar


def block_diag_scipy(arr, num=3):
    return block_diag(*([arr] * num))


def normalize_columns(matrix):
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / norms


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    grid_linewidth=1,
    grid_step=1,
    cbar_kw=None,
    cbarlabel="",
    colorbar_pos=[1.04, 0.0, 0.05, 0.5],
    colorbar_width="3%",
    orientation="vertical",
    pad=0.1,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=pad)
    if colorbar_pos is not None:
        cax = ax.inset_axes(colorbar_pos)
        cbar = ax.figure.colorbar(im, cax=cax, orientation=orientation, **cbar_kw)
    else:
        cbar = None
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(0, data.shape[1] + 1, grid_step) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, data.shape[0] + 1, grid_step) - 0.5, minor=True)

    # ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    ax.grid(which="minor", color="w", linestyle="-", linewidth=grid_linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

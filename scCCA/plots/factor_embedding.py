from typing import List, Union

import matplotlib.cm as cm
import matplotlib.colors as co
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import set_up_plot


def factor_embedding(
    adata,
    model_key="X_scpca",
    embedding: str = "X_umap",
    factor: Union[int, List[int], None] = None,
    sign: float = 1.0,
    cmap=cm.PiYG,
    colorbar_pos="right",
    colorbar_width="3%",
    orientation="vertical",
    size: float = 1,
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
        _factor_embedding,
        sign=sign,
        cmap=cmap,
        embedding=embedding,
        colorbar_pos=colorbar_pos,
        colorbar_width=colorbar_width,
        orientation=orientation,
        size=size,
        ncols=ncols,
        width=width,
        height=height,
        ax=ax,
    )
    return ax


def _factor_embedding(
    adata,
    model_key: str,
    factor: int,
    embedding: str = "X_umap",
    sign=1.0,
    cmap=cm.PiYG,
    colorbar_pos="right",
    colorbar_width="3%",
    orientation="vertical",
    size: float = 1,
    ax=None,
):

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.gcf()

    factor_num = factor
    factor = sign * adata.obsm[f"X_{model_key}"][..., factor]

    vmin = factor.min()
    vmax = factor.max()
    if vmin < 0 and vmax > 0:
        norm = co.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
    elif vmin < 0 and vmax < 0:
        # print('min color')
        norm = co.Normalize(vmin=vmin, vmax=0)
        cmap = co.LinearSegmentedColormap.from_list("name", [cmap(0.0), "w"])
    else:
        # print('max color')
        cmap = co.LinearSegmentedColormap.from_list("name", ["w", cmap(1.0)])
        norm = co.Normalize(vmin=0, vmax=vmax)

    im = ax.scatter(
        adata.obsm[embedding][:, 0],
        adata.obsm[embedding][:, 1],
        s=size,
        c=factor,
        norm=norm,
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=0.1)
    fig.colorbar(im, cax=cax, orientation=orientation)
    ax.set_title(f"Factor {factor_num}")
    ax.set_xlabel(f"{model_key}1")
    ax.set_ylabel(f"{model_key}2")
    ax.set_xticks([])
    ax.set_yticks([])

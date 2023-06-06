import matplotlib.pyplot as plt
import numpy as np
from patsy.design_info import DesignMatrix


def get_design_matrix(dm, repeats=4, cat_repeats=None):
    if isinstance(dm, DesignMatrix):
        m = np.asmatrix(dm)
    else:
        m = dm

    _, idx = np.unique(m, return_index=True, axis=0)
    mu = m[np.sort(idx)]
    mr = np.repeat(mu, repeats=repeats, axis=0)
    if cat_repeats is not None:
        mr = np.concatenate([mr] * cat_repeats, axis=0)

    return mr


def design_matrix(
    design_matrix,
    repeats=4,
    cat_repeats=None,
    xticklabels=[],
    title="D",
    xlabel=None,
    ylabel=None,
    ylabel_pos=None,
    xlabel_pos=None,
    col=None,
    ax=None,
):
    if ax is None:
        plt.figure(figsize=(0.8, 3))
        ax = plt.gca()

    M = get_design_matrix(design_matrix, repeats=repeats, cat_repeats=cat_repeats)
    if col is None:
        g = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    else:
        M = M[:, [col]]
        g = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    _ = g.axes.set_xticks([i + 0.5 for i in range(M.shape[1])])

    _ = g.axes.set_yticks([])
    if title is not None:
        g.axes.set_title(f"{title}")

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    xgrid = np.arange(M.shape[0] + 1) - 0.5
    ygrid = np.arange(M.shape[1] + 1) - 0.5
    ax.hlines(xgrid, ygrid[0], ygrid[-1], color="k")
    ax.vlines(ygrid, xgrid[0], xgrid[-1], color="k")

    if len(xticklabels) == 0:
        _ = g.axes.set_xticks([])
    else:
        _ = g.axes.set_xticklabels(xticklabels)

    if xlabel is not None:
        ax.set_xlabel("$%s$" % xlabel)
        if xlabel_pos is not None:
            ax.xaxis.set_label_coords(xlabel_pos[0], xlabel_pos[1])
    if ylabel is not None:
        ax.set_ylabel("$%s$" % ylabel)
        if ylabel_pos is not None:
            ax.yaxis.set_label_coords(ylabel_pos[0], ylabel_pos[1])

    return g

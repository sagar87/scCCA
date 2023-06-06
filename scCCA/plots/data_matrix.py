import matplotlib.collections as collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def get_hsvcmap(i, N, rot=0.0):
    nsc = 24
    chsv = mcolors.rgb_to_hsv(plt.cm.hsv(((np.arange(N) / N) + rot) % 1.0)[i, :3])
    rhsv = mcolors.rgb_to_hsv(plt.cm.Reds(np.linspace(0.2, 1, nsc))[:, :3])
    arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
    arhsv[:, 1:] = rhsv[:, 1:]
    rgb = mcolors.hsv_to_rgb(arhsv)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)


def triatpos(pos=(0, 0), rot=0):
    r = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]]) * 0.5
    rm = [
        [np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
        [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))],
    ]
    r = np.dot(rm, r.T).T
    r[:, 0] += pos[0]
    r[:, 1] += pos[1]
    return r


# function that divides number and returns also the remainder
def divmod2(n, d):
    return divmod(n, d) + (n % d,)


def triangle_overlay(a, ax, rot=0, split_rows=2, cmap=["Blues", "Greens"], color=None, vmin=None, vmax=None, **kwargs):
    # segs = []
    nrow, ncol = a.shape

    val_mapping = {i: r for i, r in enumerate(np.split(np.arange(nrow), split_rows))}
    row_mapping = {j: i for i, r in enumerate(np.split(np.arange(nrow), split_rows)) for j in r}
    segs_dict = {i: [] for i in range(split_rows)}

    for i in range(nrow):
        for j in range(ncol):
            k = row_mapping[i]
            segs_dict[k].append(triatpos((j, i), rot=rot))

    for k, v in segs_dict.items():
        if color:
            col = collections.PolyCollection(v, color=color[k], **kwargs)
        else:
            col = collections.PolyCollection(v, cmap=cmap[k], color="w", **kwargs)
        col.set_array(a.flatten()[val_mapping[k]])
        ax.add_collection(col)

    return col


def data_matrix(
    size=(16, 2),
    array=None,
    right_add=0,
    split_rows=2,
    cmaps=None,
    vmin=None,
    vmax=None,
    remove_ticklabels=False,
    xlabel=None,
    ylabel=None,
    ylabel_pos=None,
    xlabel_pos=None,
    hlinewidth=None,
    vlinewidth=None,
    ax=None,
):
    """
    Visualizes a data matrix with multiple colormaps.


    Parameters
    ----------
    size : tuple
        Size of the data matrix. If array is not None, size is ignored.
    array : np.ndarray
        Data matrix. If None, a random matrix is generated.
    right_add : int
        Number of columns to padd to the right of the data matrix. May be
        useful to add annotations on the right side of the matrix.
    split_rows : int or list
        Number of categories to split the data matrix into. Must divide the
        array.shape[0] evenly.
    cmaps : list
        List of colormaps to use. If None, a default colormap is used.


    """
    ax = ax or plt.gca()
    rows = size[0]
    cols = size[1]

    if array is None:
        array = np.random.normal(size=size)

    index_array = []
    # print(type(split_rows) is int)

    if type(split_rows) is int:
        sub_rows = int(rows / split_rows)

        for i in range(split_rows):
            index_array.append(i * np.ones((sub_rows, cols)))
    else:
        sub_rows = len(split_rows)

        for i, j in enumerate(split_rows):
            index_array.append(i * np.ones((j, cols)))

    premask = np.concatenate(index_array, 0)

    if right_add > 0:
        premask_pad = -np.ones((size[0], right_add))
        premask = np.concatenate([premask, premask_pad], 1)

        right_pad = np.zeros((size[0], right_add))
        array = np.concatenate([array, right_pad], 1)

        # print(premask)
    # print(premask)
    images = []

    if cmaps is None:
        cmap = [get_hsvcmap(i, int(np.max(premask)) + 1, rot=0.5) for i in range(int(np.max(premask) + 1))]
    else:
        cmap = cmaps

    for i in range(int(np.min(premask)), int(np.max(premask) + 1)):
        if i == -1:
            continue
        else:
            col = np.ma.array(array, mask=premask != i)
            im = ax.imshow(col, cmap=cmap[i], vmin=vmin, vmax=vmax)
            # sns.heatmap(col,  cmap=cmap[i], linecolor='k', linewidths=.2, cbar=False, ax=ax)
            images.append(im)

    xgrid = np.arange(size[0] + 1) - 0.5
    ygrid = np.arange(size[1] + 1) - 0.5
    ax.hlines(xgrid, ygrid[0], ygrid[-1], color="k", linewidth=hlinewidth)
    ax.vlines(ygrid, xgrid[0], xgrid[-1], color="k", linewidth=vlinewidth)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # ax.set_xticks(np.arange(size[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(size[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False)
    if remove_ticklabels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if xlabel is not None:
        ax.set_xlabel("$%s$" % xlabel)
        if xlabel_pos is not None:
            ax.xaxis.set_label_coords(xlabel_pos[0], xlabel_pos[1])
    if ylabel is not None:
        ax.set_ylabel("$%s$" % ylabel)
        if ylabel_pos is not None:
            ax.yaxis.set_label_coords(ylabel_pos[0], ylabel_pos[1])
    return ax

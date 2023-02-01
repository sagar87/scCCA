from typing import Callable, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from anndata import AnnData


def rand_jitter(arr, stdev=1):
    # stdev = .01 * (max(arr) - min(arr))
    # print(stdev)
    return arr + np.random.randn(len(arr)) * stdev


def repel_labels(
    ax,
    x,
    y,
    labels,
    center=None,
    k=0.01,
    label_pos_x=1.0,
    label_pos_y=1.0,
    fontsize=12,
    expand_limits=False,
):
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    for xi, yi, label in zip(x, y, labels):
        data_str = "data_{0}".format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    if center is None:
        pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)
    else:
        pos = nx.spring_layout(G, pos=init_pos, center=center, k=k)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale_x, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
    # print(scale)
    scale_y, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
    # print(scale)
    shift = np.array([shift_x, shift_y])
    scale = np.array([scale_x, scale_y])
    # print(scale)
    # print(shift)
    for key, val in pos.items():
        # print(val)
        pos[key] = (val * scale) + shift

    for label, data_str in G.edges():
        point_x, point_y = pos[data_str]
        label_x, label_y = pos[label]

        if label_y > point_y and point_y < 0:
            delta_y = 1 - label_pos_y
        elif label_y > point_y and point_y > 0:
            delta_y = 1 + label_pos_y
        elif label_y < point_y and point_y < 0:
            delta_y = 1 + label_pos_y
        else:
            delta_y = 1 - label_pos_y

        # print(f'{data_str} datapoint ({point_x:0.2f}, {point_y:0.2f}) label ({label_x:0.2f} {label_y:0.2f}) -> Δ ({(point_x - label_x):0.2f} {(point_x * label_y):0.2f})')
        # print(f'{data_str} ({label_pos_x, label_pos_y}):  ({label_x:0.2f} {label_y:0.2f}) -> ({(label_x * label_pos_x):0.2f} {(label_y * label_pos_y):0.2f})')
        label_pos = pos[label] * np.array([label_pos_x, delta_y])
        # print(pos[label])
        # print(label_pos)

        ax.annotate(
            label,
            xy=pos[data_str],
            xycoords="data",
            xytext=label_pos,
            textcoords="data",
            fontsize=fontsize,
            arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0, connectionstyle="arc3", color="k"),
        )
    # expand limits
    if expand_limits:
        all_pos = np.vstack(list(pos.values()))
        x_span, y_span = np.ptp(all_pos, axis=0)
        mins = np.min(all_pos - x_span * 0.15, 0)
        maxs = np.max(all_pos + y_span * 0.15, 0)
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])


def set_up_subplots(num_plots, ncols=4, width=4, height=3):
    """Set up subplots for plotting multiple factors."""

    if num_plots == 1:
        fig, ax = plt.subplots()
        return fig, ax

    nrows, reminder = divmod(num_plots, ncols)

    if nrows == 0:
        nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    _ = [ax.axis("off") for ax in axes.flatten()[num_plots:]]
    return fig, axes


def set_up_plot(
    adata: AnnData,
    model_key: str,
    instances: Union[int, List[int], None],
    func: Callable,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Union[plt.Axes, None] = None,
    **kwargs
):
    if isinstance(instances, list):
        num_plots = len(instances)
        fig, ax = set_up_subplots(num_plots, ncols=ncols, width=width, height=height)
    elif isinstance(instances, int):
        num_plots = 1
        if ax is None:
            fig, ax = plt.subplots(1, 1)
    else:
        num_plots = adata.varm[model_key].shape[1]
        instances = [i for i in range(num_plots)]
        fig, ax = set_up_subplots(num_plots, ncols=ncols, width=width, height=height)

    if num_plots == 1:
        func(adata, model_key, instances, ax=ax, **kwargs)
    else:
        for i, ax_i in zip(instances, ax.flatten()):
            func(adata, model_key, i, ax=ax_i, **kwargs)

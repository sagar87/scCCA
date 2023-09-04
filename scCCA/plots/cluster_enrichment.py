import scanpy as sc

from ..utils import get_diff_genes


def cluster_enrichment(
    adata,
    model_key,
    states,
    factor,
    state_key,
    cluster_key,
    cluster,
    dot=False,
    sign=1,
    highest=0,
    lowest=0,
    threshold=1.96,
    return_var_names=False,
    **kwargs,
):
    if isinstance(cluster, str):
        cluster = [cluster]
    if isinstance(state_key, str):
        state_key = [state_key]
    df = get_diff_genes(adata, model_key, states, factor, sign=sign, highest=adata.shape[1], threshold=threshold)
    if highest > 0 or lowest > 0:
        var_names = {
            f"Up in {states[1]}": df.head(highest).gene.tolist(),
            f"Down in {states[1]}": df.tail(lowest).gene.tolist(),
        }
    else:
        var_names = {
            f"Up in {states[1]}": df[df.significant & (df["diff"] > 0)].gene.tolist(),
            f"Down in {states[1]}": df[df.significant & (df["diff"] < 0)].gene.tolist(),
        }

    for k, v in var_names.copy().items():
        if len(v) == 0:
            del var_names[k]

    if dot:
        axes = sc.pl.dotplot(
            adata[adata.obs[cluster_key].isin(cluster)],
            groupby=[cluster_key, *state_key],
            var_names=var_names,
            show=False,
            **kwargs,
        )
        labels = [
            item.get_text() + f" ({df[df.gene==item.get_text()]['diff'].item():.2f})"
            for item in axes["mainplot_ax"].get_xticklabels()
        ]
        axes["mainplot_ax"].set_xticklabels(labels)
    else:
        axes = sc.pl.heatmap(
            adata[adata.obs[cluster_key].isin(cluster)],
            groupby=[cluster_key, *state_key],
            var_names=var_names,
            show=False,
            **kwargs,
        )
        labels = [
            item.get_text() + f" ({df[df.gene==item.get_text()]['diff'].item():.2f})"
            for item in axes["heatmap_ax"].get_xticklabels()
        ]
        axes["heatmap_ax"].set_xticklabels(labels)
        # axes['heatmap_ax'].get_xticklabels()

    if return_var_names:
        return var_names
    return axes

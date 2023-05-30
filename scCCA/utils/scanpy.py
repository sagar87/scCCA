import scanpy as sc
from anndata import AnnData


def umap(adata: AnnData, basis: str, neighbors_kwargs: dict = {}, umap_kwargs: dict = {}):
    """
    Performs UMAP dimensionality reduction on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the data to be processed.
    basis : str
        The basis to use for the UMAP calculation.
    neighbors_kwargs : dict, optional
        Additional keyword arguments to be passed to `sc.pp.neighbors` function.
        Default is an empty dictionary.
    umap_kwargs : dict, optional
        Additional keyword arguments to be passed to `sc.tl.umap` function.
        Default is an empty dictionary.

    Returns
    -------
    None

    Notes
    -----
    This function performs UMAP dimensionality reduction on the input `adata` object
    using the specified `basis`. It first computes the neighbors graph using the
    `sc.pp.neighbors` function, with the option to provide additional keyword arguments
    via `neighbors_kwargs`. Then, it applies the UMAP algorithm using the `sc.tl.umap`
    function, with the option to provide additional keyword arguments via `umap_kwargs`.
    Finally, it stores the UMAP coordinates in the `obsm` attribute of the `adata` object
    under the key `"{basis}_umap"`, and removes the original `"X_umap"` coordinates.

    Example
    -------
    >>> adata = AnnData(X)
    >>> umap(adata, basis="pca", neighbors_kwargs={"n_neighbors": 10}, umap_kwargs={"min_dist": 0.5})

    """
    sc.pp.neighbors(adata, use_rep=f"{basis}", key_added=f"{basis}", **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=f"{basis}", **umap_kwargs)
    adata.obsm[f"{basis}_umap"] = adata.obsm["X_umap"]
    del adata.obsm["X_umap"]

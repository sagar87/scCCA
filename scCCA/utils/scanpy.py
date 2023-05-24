import scanpy as sc
from anndata import AnnData

DESIGN_KEY = "design"


def _get_model_design(adata: AnnData, model_key: str):
    """
    Extracts the design dictionary from an AnnData object.

    This function retrieves the design dictionary associated with a specific model from the `uns` attribute of an AnnData
    object. The `uns` attribute is a dictionary-like storage for unstructured annotation data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the model annotations.
    model_key : str
        The key identifying the model in the `uns` attribute.

    Returns
    -------
    dict
        The design dictionary associated with the specified model.

    Raises
    ------
    ValueError
        If the model key is not found in the `uns` attribute of the AnnData object.
    ValueError
        If the design mapping is not found in the model annotations.

    Example
    -------
    >>> adata = AnnData()
    >>> model_key = "my_model"
    >>> design_mapping = {"Intercept": 0, "stim": 1}
    >>> adata.uns[model_key] = {"design": design_mapping}
    >>> result = _get_model_design(adata, model_key)
    >>> print(result)
    {"Intercept": 0, "stim": 1}
    """

    if model_key not in adata.uns:
        raise ValueError(f"No model with the key {model_key} found.")

    model_dict = adata.uns[model_key]

    if DESIGN_KEY not in model_dict:
        raise ValueError("No design mapping found in model annotations.")

    return model_dict[DESIGN_KEY]


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

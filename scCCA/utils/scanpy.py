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
        raise ValueError(f"No design mapping found in model annotations.")

    return model_dict[DESIGN_KEY]


def umap(adata, basis, neighbors_kwargs={}, umap_kwargs={}):
    sc.pp.neighbors(adata, use_rep=f"{basis}", key_added=f"{basis}", **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=f"{basis}", **umap_kwargs)
    adata.obsm[f"{basis}_umap"] = adata.obsm["X_umap"]
    del adata.obsm["X_umap"]

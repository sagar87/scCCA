from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

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


def extract_counts(adata, layers_key, protein_obsm_key):
    if protein_obsm_key is not None:
        counts = get_protein_counts(adata, protein_obsm_key)
    else:
        counts = get_rna_counts(adata, layers_key)

    return counts


def get_rna_counts(adata: AnnData, layers_key: Union[str, None] = None) -> np.ndarray:
    """
    Extracts RNA counts from AnnData object.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    layers_key: str, optional (default: None)
        If layers_key is None, then the raw counts are extracted from adata.X.
        Otherwise, the counts are extracted from adata.layers[layers_key].

    Returns
    -------
    X: np.ndarray
        RNA counts matrix.
    """
    if layers_key is None:
        X = adata.X
    else:
        if layers_key not in adata.layers:
            raise KeyError("Spefied layers_key was not found in the AnnData object.")
        X = adata.layers[layers_key]

    if issparse(X):
        X = X.toarray()

    return X.astype(np.float32)


def get_protein_counts(adata: AnnData, protein_obsm_key: str) -> np.ndarray:
    """
    Extracts protein counts from AnnData object.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    protein_obsm_key: str
        Key for protein counts in adata.obsm.

    Returns
    -------
    Y: np.ndarray
        Protein counts matrix.
    """
    if isinstance(adata.obsm[protein_obsm_key], pd.DataFrame):
        Y = adata.obsm[protein_obsm_key].values
    else:
        Y = adata.obsm[protein_obsm_key]

    return Y.astype(np.float32)

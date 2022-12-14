from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse


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

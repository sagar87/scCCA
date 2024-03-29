from collections import OrderedDict, namedtuple
from typing import List, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix

from .data import _get_model_design, _validate_sign, _validate_states

StateMapping = namedtuple("StateMapping", "mapping, reverse, encoding, index, columns, states, sparse")


def get_states(design: DesignMatrix) -> namedtuple:
    """Extracts the states from the design matrix.

    Parameters
    ----------
    design: DesignMatrix
        Design matrix of the model.

    Returns
    -------
    StateMapping: namedtuple
        Named tuple with the following fields
    """
    unique_rows, inverse_rows = np.unique(np.asarray(design), axis=0, return_inverse=True)

    combinations = OrderedDict()
    sparse_state = {}
    for j, row in enumerate(range(unique_rows.shape[0])):
        idx = tuple(np.where(unique_rows[row] == 1)[0])
        combinations[idx] = unique_rows[row], j

        state_name = "|".join([design.design_info.column_names[i] for i in np.where(unique_rows[row] == 1)[0]])
        if state_name != "Intercept":
            state_name = state_name.lstrip("Intercept|")
        sparse_state[state_name] = j

    factor_cols = {v: k for k, v, in design.design_info.column_name_indexes.items()}
    state_cols = {v: k for k, v in factor_cols.items()}

    state_mapping = {}
    reverse_mapping = {}
    for idx, (k, v) in enumerate(combinations.items()):
        state = ""
        for idx in k:
            state += factor_cols[idx] + "|"
        state = state.rstrip("|")
        state_mapping[state] = v[1]
        reverse_mapping[v[1]] = state

    return StateMapping(
        state_mapping, reverse_mapping, unique_rows, inverse_rows, factor_cols, state_cols, sparse_state
    )


def get_state_loadings(adata: AnnData, model_key: str) -> dict:
    """
    Computes the loading matrix for each state defined in the
    design matrix of the model.

    Parameters
    ----------
    adata: AnnData
        Anndata object with the fitted scPCA model stored.

    model_key: str
        Key of the model in the AnnData object.

    Returns
    -------
    dict of np.ndarray with
        Dictionary with the loading matrices for each state.
    """
    design = adata.uns[model_key]["design"]

    states = {}
    for k, v in design.items():
        states[k] = adata.varm[model_key][..., v].sum(-1)

    return states


def get_formula(adata: AnnData, formula: str):
    if formula is None:
        batch = dmatrix("1", adata.obs)
    else:
        batch = dmatrix(formula, adata.obs)

    return batch


def _get_gene_idx(array: np.ndarray, highest: int, lowest: int):
    """
    Given an array of indices return the highest and/or lowest
    indices.

    Parameters
    ----------
    array: np.ndarray
        array in which to extract the highest/lowest indices
    highest: int
        number of top indices to extract
    lowest: int
        number of lowest indices to extract

    Returns
    -------
    np.ndarray
    """
    order = np.argsort(array)

    if highest == 0:
        gene_idx = order[:lowest]
    else:
        gene_idx = np.concatenate([order[:lowest], order[-highest:]])

    return gene_idx


def get_ordered_genes(
    adata: AnnData,
    model_key: str,
    state: str,
    factor: int,
    sign: Union[int, float] = 1.0,
    vector: str = "W_rna",
    highest: int = 10,
    lowest: int = 0,
    ascending: bool = False,
):
    """
    Retrieve the ordered genes based on differential factor values.

    Parameters
    ----------
    adata :
        Annotated data object containing gene expression data.
    model_key :
        Key to identify the specific model.
    state :
        Name of the model state from which to extract genes.
    factor :
        Factor index for which differential factor values are calculated.
    sign :
        Sign multiplier for differential factor values. Default is 1.0.
    vector :
        Vector type from which to extract differential factor values. Default is "W_rna".
    highest :
        Number of genes with the highest differential factor values to retrieve. Default is 10.
    lowest :
        Number of genes with the lowest differential factor values to retrieve. Default is 0.
    ascending :
        Flag indicating whether to sort genes in ascending order based on differential factor values.
        Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ordered genes along with their magnitude, differential factor values,
        type (lowest/highest), model state, factor index, and gene index.

    Raises
    ------
    ValueError
        If the specified model key or model state is not found in the provided AnnData object.
    """
    _ = _validate_sign(sign)
    model_design = _get_model_design(adata, model_key)
    state = model_design[state]
    diff_factor = sign * adata.varm[f"{model_key}_{vector}"][..., factor, state]
    gene_idx = _get_gene_idx(diff_factor, highest, lowest)

    magnitude = np.abs(diff_factor[gene_idx])
    genes = adata.var_names.to_numpy()[gene_idx]

    return (
        pd.DataFrame(
            {
                "gene": genes,
                "magnitude": magnitude,
                "diff": diff_factor[gene_idx],
                "type": ["lowest"] * lowest + ["highest"] * highest,
                "state": state,
                "factor": factor,
                "index": gene_idx,
            }
        )
        .sort_values(by="diff", ascending=ascending)
        .reset_index(drop=True)
        .rename(columns={"diff": "value"})
    )


def get_diff_genes(
    adata: AnnData,
    model_key: str,
    states: Union[str, List[str]],
    factor: int,
    sign: Union[int, float] = 1.0,
    vector: str = "W_rna",
    highest: int = 10,
    lowest: int = 0,
    ascending: bool = False,
    threshold=1.96,
):
    """
    Compute the differential genes between two states based on a given model.

    Parameters
    ----------
    adata :
        Annotated data matrix.
    model_key :
        Key to access the model in the adata object.
    states :
        List containing two states for comparison. If a single str is provided
        the base state is assumed to be 'Intercept'.
    factor :
        Factor index to consider for the differential calculation.
    sign :
        Sign to adjust the difference, by default 1.0.
    vector :
        Vector key to access in the model, by default "W_rna".
    highest :
        Number of highest differential genes to retrieve, by default 10.
    lowest :
        Number of lowest differential genes to retrieve, by default 0.
    ascending :
        Whether to sort the results in ascending order, by default False.
    threshold :
        Threshold for significance, by default 1.96.

    Returns
    -------
    pd.DataFrame
        DataFrame containing differential genes, their magnitudes, differences, types, states, factors, indices, and significance.

    Notes
    -----
    This function computes the differential genes between two states based on a given model.
    It first validates the sign, retrieves the model design, and computes
    the difference between the two states for a given factor. The function then
    retrieves the gene indices based on the highest and lowest differences and
    constructs a DataFrame with the results.
    """

    sign = _validate_sign(sign)
    states = _validate_states(states)

    model_design = _get_model_design(adata, model_key)
    state_a = model_design[states[0]]
    state_b = model_design[states[1]]
    a = adata.varm[f"{model_key}_{vector}"][..., factor, state_a]
    b = adata.varm[f"{model_key}_{vector}"][..., factor, state_b]

    # diff_factor = sign * (model_dict[vector][state_b][factor] - model_dict[vector][state_a][factor])
    diff_factor = sign * (b - a)

    gene_idx = _get_gene_idx(diff_factor, highest, lowest)

    magnitude = np.abs(diff_factor[gene_idx])
    genes = adata.var_names.to_numpy()[gene_idx]
    # is_significant = lambda x: x > norm().ppf(1 - significance_level) or x < norm().ppf(significance_level)
    df = (
        pd.DataFrame(
            {
                "gene": genes,
                "magnitude": magnitude,
                "diff": diff_factor[gene_idx],
                "type": ["lowest"] * lowest + ["highest"] * highest,
                "state": states[1] + "-" + states[0],
                "factor": factor,
                "index": gene_idx,
                states[0]: a[gene_idx],
                states[1]: b[gene_idx],
            }
        )
        .sort_values(by="diff", ascending=ascending)
        .reset_index(drop=True)
    )

    df["significant"] = df["magnitude"] > threshold
    return df


def get_significant_genes(adata, model_key, states, filter_genes=True, threshold=1.96):
    all_factors = []
    for i in range(adata.uns[model_key]["model"]["num_factors"]):
        df = get_diff_genes(adata, model_key, states, i, highest=adata.shape[1], threshold=threshold)
        if filter_genes:
            df = df[df.significant]
        all_factors.append(df)

    return pd.concat(all_factors).reset_index(drop=True)

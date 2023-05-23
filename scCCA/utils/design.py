from collections import OrderedDict, namedtuple
from typing import List, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from patsy import dmatrix
from patsy.design_info import DesignMatrix

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
    model_dict = adata.uns[model_key]
    model_design = model_dict["design"]
    state = model_design[state]
    diff_factor = adata.varm[f"{model_key}_{vector}"][..., factor, state]
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
            }
        )
        .sort_values(by="diff", ascending=ascending)
        .reset_index(drop=True)
        .rename(columns={"diff": "value"})
    )


def get_diff_genes(
    adata: AnnData,
    model_key: str,
    state: List[str],
    factor: int,
    sign: Union[int, float] = 1.0,
    vector: str = "W_rna",
    highest: int = 10,
    lowest: int = 0,
    ascending: bool = False,
):
    model_dict = adata.uns[model_key]
    model_design = model_dict["design"]
    state_a = model_design[state[0]]
    state_b = model_design[state[1]]

    # diff_factor = sign * (model_dict[vector][state_b][factor] - model_dict[vector][state_a][factor])
    diff_factor = sign * (
        adata.varm[f"{model_key}_{vector}"][..., factor, state_b]
        - adata.varm[f"{model_key}_{vector}"][..., factor, state_a]
    )

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
                "state": state[1] + "-" + state[0],
                "factor": factor,
            }
        )
        .sort_values(by="diff", ascending=ascending)
        .reset_index(drop=True)
    )

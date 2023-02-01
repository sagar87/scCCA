from collections import OrderedDict, namedtuple

import numpy as np
from patsy import dmatrix
from patsy.design_info import DesignMatrix

StateMapping = namedtuple("StateMapping", "mapping, reverse, encoding, index")


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
    for j, row in enumerate(range(unique_rows.shape[0])):
        idx = tuple(np.where(unique_rows[row] == 1)[0])
        combinations[idx] = unique_rows[row], j

    factor_cols = {v: k for k, v, in design.design_info.column_name_indexes.items()}

    state_mapping = {}
    reverse_mapping = {}
    for idx, (k, v) in enumerate(combinations.items()):
        state = ""
        for idx in k:
            state += factor_cols[idx] + "|"
        state = state.rstrip("|")
        state_mapping[state] = v[1]
        reverse_mapping[v[1]] = state

    return StateMapping(state_mapping, reverse_mapping, unique_rows, inverse_rows)


def get_state_loadings(adata, model_key: str) -> dict:
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


def get_formula(adata, formula):
    if formula is None:
        batch = dmatrix("1", adata.obs)
    else:
        batch = dmatrix(formula, adata.obs)

    return batch

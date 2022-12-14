import numpy as np
from patsy import dmatrix
from patsy.design_info import DesignMatrix


def get_states(
    design: DesignMatrix, return_raw: bool = False, indices: bool = True
) -> dict:
    """
    Extracts states and indices of a patsy design matrix.

    Parameters
    ----------
    design: patsy.DesignMatrix
        Patsy design matrix.
    return_raw: bool, optional (default: False)
        If True, then the raw states and combinations are returned.
    indices: bool, optional (default: True)
        If True, then the indices of the states are returned.

    Returns
    -------
    dict
        States and the corresponding indices of the design matrix.
    """
    unique_rows = np.unique(np.asarray(design), axis=0)

    combinations = {}
    for row in range(unique_rows.shape[0]):
        combinations[tuple(np.where(unique_rows[row] == 1)[0])] = unique_rows[row]

    factor_indices = {k.name(): v for k, v in design.design_info.term_slices.items()}
    factor_levels = {
        f.factor.name(): list(f.categories)
        for f in list(design.design_info.factor_infos.values())
    }

    intercept = False
    if "Intercept" in factor_indices.keys():
        intercept = True

    col_idx = {}
    for k, v in factor_indices.items():
        if k != "Intercept":
            for i, j in enumerate(range(v.start, v.stop)):
                # print(k, j)
                if k in factor_levels.keys():
                    col_idx[j] = {k: factor_levels[k][i + 1 if intercept else i]}
                else:
                    col_idx[j] = {k: k}
                # print(col_idx)
        else:
            col_idx[0] = {k: v[0] for k, v in factor_levels.items()}

    states = {}
    for k, v in combinations.items():
        state_dict = {}
        for i in k:
            for f, l in col_idx[i].items():
                state_dict[f] = l

        states[k] = state_dict

    if return_raw:
        return states, combinations

    state_map = {}
    for k, v in states.items():
        c = ""
        for kk, vv in v.items():
            c += f"{vv}|"

        if indices:
            state_map[c.rstrip("|")] = np.where(combinations[k] == 1)[0]
        else:
            state_map[c.rstrip("|")] = combinations[k]

    if len(state_map) == 1 and "" in state_map.keys():
        v = state_map.pop("")
        state_map["base"] = v

    return state_map


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

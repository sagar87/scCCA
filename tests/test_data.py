import numpy as np
import pytest
from anndata import AnnData

from scCCA.utils import get_diff_genes, get_ordered_genes, get_rna_counts
from scCCA.utils.data import _get_model_design
from scCCA.utils.design import _get_gene_idx


def test_get_rna_counts(test_sparse_anndata):
    X = get_rna_counts(test_sparse_anndata)

    assert type(X) == np.ndarray
    assert X.shape == (100, 2000)


def test_get_gene_idx():
    test = np.random.randn(50)
    arr = _get_gene_idx(test, highest=10, lowest=0)
    assert np.all(arr == np.argsort(test)[-10:])


@pytest.mark.parametrize(
    "model_key, state, factor, highest, lowest, vector, sign, ascending",
    [
        ("m2", "Intercept", 0, 50, 0, "W_rna", 1.0, False),
        ("m2", "Intercept", 5, 50, 0, "W_rna", 1.0, False),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", 1.0, False),
        ("m2", "Intercept", 0, 0, 50, "W_rna", 1.0, False),
        ("m2", "Intercept", 5, 0, 50, "W_rna", 1.0, False),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", 1.0, False),
        ("m2", "Intercept", 0, 25, 25, "W_rna", 1.0, False),
        ("m2", "Intercept", 5, 25, 25, "W_rna", 1.0, False),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", 1.0, False),
        ("m2", "Intercept", 0, 50, 0, "W_rna", 1.0, False),
        ("m2", "Intercept", 5, 50, 0, "W_rna", 1.0, False),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", 1.0, False),
        ("m2", "Intercept", 0, 0, 50, "W_rna", 1.0, False),
        ("m2", "Intercept", 5, 0, 50, "W_rna", 1.0, False),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", 1.0, False),
        ("m2", "Intercept", 0, 25, 25, "W_rna", 1.0, False),
        ("m2", "Intercept", 5, 25, 25, "W_rna", 1.0, False),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", 1.0, False),
        ("m2", "Intercept", 0, 50, 0, "W_rna", 1.0, True),
        ("m2", "Intercept", 5, 50, 0, "W_rna", 1.0, True),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", 1.0, True),
        ("m2", "Intercept", 0, 0, 50, "W_rna", 1.0, True),
        ("m2", "Intercept", 5, 0, 50, "W_rna", 1.0, True),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", 1.0, True),
        ("m2", "Intercept", 0, 25, 25, "W_rna", 1.0, True),
        ("m2", "Intercept", 5, 25, 25, "W_rna", 1.0, True),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", 1.0, True),
        ("m2", "Intercept", 0, 50, 0, "W_rna", 1.0, True),
        ("m2", "Intercept", 5, 50, 0, "W_rna", 1.0, True),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", 1.0, True),
        ("m2", "Intercept", 0, 0, 50, "W_rna", 1.0, True),
        ("m2", "Intercept", 5, 0, 50, "W_rna", 1.0, True),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", 1.0, True),
        ("m2", "Intercept", 0, 25, 25, "W_rna", 1.0, True),
        ("m2", "Intercept", 5, 25, 25, "W_rna", 1.0, True),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", 1.0, True),
        ("m2", "Intercept", 0, 50, 0, "W_rna", -1.0, False),
        ("m2", "Intercept", 5, 50, 0, "W_rna", -1.0, False),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", -1.0, False),
        ("m2", "Intercept", 0, 0, 50, "W_rna", -1.0, False),
        ("m2", "Intercept", 5, 0, 50, "W_rna", -1.0, False),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", -1.0, False),
        ("m2", "Intercept", 0, 25, 25, "W_rna", -1.0, False),
        ("m2", "Intercept", 5, 25, 25, "W_rna", -1.0, False),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", -1.0, False),
        ("m2", "Intercept", 0, 50, 0, "W_rna", -1.0, False),
        ("m2", "Intercept", 5, 50, 0, "W_rna", -1.0, False),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", -1.0, False),
        ("m2", "Intercept", 0, 0, 50, "W_rna", -1.0, False),
        ("m2", "Intercept", 5, 0, 50, "W_rna", -1.0, False),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", -1.0, False),
        ("m2", "Intercept", 0, 25, 25, "W_rna", -1.0, False),
        ("m2", "Intercept", 5, 25, 25, "W_rna", -1.0, False),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", -1.0, False),
        ("m2", "Intercept", 0, 50, 0, "W_rna", -1.0, True),
        ("m2", "Intercept", 5, 50, 0, "W_rna", -1.0, True),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", -1.0, True),
        ("m2", "Intercept", 0, 0, 50, "W_rna", -1.0, True),
        ("m2", "Intercept", 5, 0, 50, "W_rna", -1.0, True),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", -1.0, True),
        ("m2", "Intercept", 0, 25, 25, "W_rna", -1.0, True),
        ("m2", "Intercept", 5, 25, 25, "W_rna", -1.0, True),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", -1.0, True),
        ("m2", "Intercept", 0, 50, 0, "W_rna", -1.0, True),
        ("m2", "Intercept", 5, 50, 0, "W_rna", -1.0, True),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna", -1.0, True),
        ("m2", "Intercept", 0, 0, 50, "W_rna", -1.0, True),
        ("m2", "Intercept", 5, 0, 50, "W_rna", -1.0, True),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna", -1.0, True),
        ("m2", "Intercept", 0, 25, 25, "W_rna", -1.0, True),
        ("m2", "Intercept", 5, 25, 25, "W_rna", -1.0, True),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna", -1.0, True),
    ],
)
def test_get_ordered_genes(model_key, state, factor, highest, lowest, vector, sign, ascending, test_anndata):
    # test that the right order of genes is returned

    df = get_ordered_genes(
        test_anndata,
        model_key=model_key,
        state=state,
        factor=factor,
        highest=highest,
        lowest=lowest,
        vector=vector,
        sign=sign,
        ascending=ascending,
    )

    state_index = test_anndata.uns[model_key]["design"][state]
    factor_weights = sign * test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_index]
    gene_idx = _get_gene_idx(factor_weights, highest, lowest)

    if ascending:
        assert np.all(df["gene"].to_numpy() == test_anndata.var_names[gene_idx].to_numpy())
        assert np.all(df["value"].to_numpy() == factor_weights[gene_idx])
    else:
        assert np.all(df["gene"].to_numpy() == test_anndata.var_names[gene_idx][::-1].to_numpy())
        assert np.all(df["value"].to_numpy() == factor_weights[gene_idx][::-1])


@pytest.mark.parametrize(
    "model_key, state, factor, highest, lowest, vector, sign, ascending",
    [
        ("m2", ["Intercept", "label[T.stim]"], 0, 50, 0, "W_rna", 1.0, False),
        ("m2", ["Intercept", "label[T.stim]"], 5, 50, 0, "W_rna", 1.0, False),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0, False),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0, False),
        ("m2", ["Intercept", "label[T.stim]"], 5, 0, 50, "W_rna", 1.0, False),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0, False),
        ("m2", ["Intercept", "label[T.stim]"], 0, 50, 0, "W_rna", 1.0, True),
        ("m2", ["Intercept", "label[T.stim]"], 5, 50, 0, "W_rna", 1.0, True),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0, True),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0, True),
        ("m2", ["Intercept", "label[T.stim]"], 5, 0, 50, "W_rna", 1.0, True),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0, True),
    ],
)
def test_get_diff_genes(model_key, state, factor, highest, lowest, vector, sign, ascending, test_anndata):
    df = get_diff_genes(
        test_anndata,
        model_key,
        state,
        factor,
        highest=highest,
        lowest=lowest,
        vector=vector,
        sign=sign,
        ascending=ascending,
    )

    model_dict = test_anndata.uns[model_key]
    model_design = model_dict["design"]
    state_a, state_b = model_design[state[0]], model_design[state[1]]
    diff = sign * (
        test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_b]
        - test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_a]
    )
    gene_idx = _get_gene_idx(diff, highest, lowest)

    if ascending:
        assert np.all(df["gene"].to_numpy() == test_anndata.var_names[gene_idx].to_numpy())
        assert np.all(df["diff"].to_numpy() == diff[gene_idx])
    else:
        assert np.all(df["gene"].to_numpy() == test_anndata.var_names[gene_idx][::-1].to_numpy())
        assert np.all(df["diff"].to_numpy() == diff[gene_idx][::-1])


def test_get_model_design_existing_key():
    adata = AnnData()
    model_key = "my_model"
    design_mapping = {"Intercept": 0, "stim": 1}
    adata.uns[model_key] = {"design": design_mapping}

    result = _get_model_design(adata, model_key)

    assert result == design_mapping


def test_get_model_design_missing_key():
    adata = AnnData()
    model_key = "my_model"

    with pytest.raises(ValueError):
        _get_model_design(adata, model_key)


def test_get_model_design_missing_design_mapping():
    adata = AnnData()
    model_key = "my_model"
    adata.uns[model_key] = {}

    with pytest.raises(ValueError):
        _get_model_design(adata, model_key)

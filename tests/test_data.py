import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

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


def test_get_rna_counts_with_X():
    # Test when layers_key is None, extracting from adata.X
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    result = get_rna_counts(adata)
    assert np.array_equal(result, expected)


def test_get_rna_counts_with_layers():
    # Test when layers_key is provided, extracting from adata.layers[layers_key]
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]), layers={"counts": np.array([[7, 8, 9], [10, 11, 12]])})
    expected = np.array([[7, 8, 9], [10, 11, 12]])
    result = get_rna_counts(adata, layers_key="counts")
    assert np.array_equal(result, expected)


def test_get_rna_counts_with_sparse_matrix():
    # Test when layers_key is provided, and X is a sparse matrix
    adata = AnnData(X=csr_matrix([[1, 0, 3], [0, 5, 0]]), layers={"counts": csr_matrix([[7, 0, 9], [0, 11, 0]])})
    expected = np.array([[7, 0, 9], [0, 11, 0]])
    result = get_rna_counts(adata, layers_key="counts")
    assert np.array_equal(result, expected)


def test_get_rna_counts_return_type():
    # Test the return type of the function
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    result = get_rna_counts(adata)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_get_rna_counts_with_invalid_layers_key():
    # Test when invalid layers_key is provided
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]), layers={"counts": np.array([[7, 8, 9], [10, 11, 12]])})
    with pytest.raises(KeyError):
        get_rna_counts(adata, layers_key="invalid_key")

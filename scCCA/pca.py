from functools import partial
from typing import Union

import numpy as np
import torch
from anndata import AnnData
from torch.types import Device

from .model import guide, model
from .train import SUBSAMPLE, SVILocalHandler
from .utils import get_formula, get_rna_counts, get_states


class scPCA(object):
    """
    scPCA model.

    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with the single-cell data.
    num_factors: int (default: 15)
        Number of factors to fit.
    layers_key: str or None (default: None)
        Key to extract single-cell count matrix from adata.layers. If layers_key is None,
        scPCA will try to extract the count matrix from the adata.X.
    batch_formula: str or None (default: None)
        R style formula to extract batch information from adata.obs. If batch_formula is None,
        scPCA assumes a single batch. Usually `batch_column - 1`.
    design_formula: str or None (default: None)
        R style formula to construct the design matrix from adata.obs. If design_formula is None,
        scPCA fits a normal PCA.
    subsampling: int (default: 4096)
        Number of cells to subsample for training. A larger number will result in a more accurate
        computation of the gradients, but will also increase the training time and memory usage.
    device: torch.device (default: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        Device to run the model on. A GPU is highly recommended.
    model_key: str (default: "scpca")
        Key to store the model in the AnnData object.
    model_kwargs: dict
        Model parameters. See sccca.model.model for more details.
    training_kwargs: dict
        Training parameters. See sccca.handler for more details.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        batch_formula: Union[str, None] = None,
        design_formula: Union[str, None] = None,
        subsampling: int = 4096,
        device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed: Union[int, None] = None,
        model_key: str = "scpca",
        model_kwargs: dict = {
            "β_rna_sd": 0.01,
            "β_rna_mean": 3.0,
            "fixed_beta": True,
            "intercept": True,
            "batch_beta": False,
            "horseshoe": False,
        },
        training_kwargs: dict = SUBSAMPLE,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            self.seed = seed
        else:
            self.seed = None

        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.batch_formula = batch_formula
        self.design_formula = design_formula
        self.model_key = model_key

        self.subsampling = min([subsampling, adata.shape[0]])
        self.device = device

        # prepare design and batch matrix
        self.batch_matrix = get_formula(self.adata, self.batch_formula)
        self.design_matrix = get_formula(self.adata, self.design_formula)
        self.batch_states = get_states(self.batch_matrix)
        self.design_states = get_states(self.design_matrix)

        #
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs

        # setup data
        self.data = self._setup_data()
        self.handler = self._setup_handler()

    def _to_torch(self, data):
        """
        Helper method to convert numpy arrays of a dict to torch tensors.
        """
        return {k: torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v for k, v in data.items()}

    def _setup_data(self):
        """
        Sets up the data.
        """
        X = get_rna_counts(self.adata, self.layers_key)
        X_size = np.log(X.sum(axis=1, keepdims=True))
        batch = np.asarray(self.batch_states.encoding).astype(np.float32)
        design = np.asarray(self.design_states.encoding).astype(np.float32)
        batch_idx = self.batch_states.index
        design_idx = self.design_states.index

        num_genes = X.shape[1]
        num_cells = X.shape[0]
        num_batches = batch.shape[1]
        idx = np.arange(num_cells)

        data = dict(
            num_factors=self.num_factors,
            X=X,
            X_size=X_size,
            Y=None,
            Y_size=None,
            design=design,
            batch=batch,
            design_idx=design_idx,
            batch_idx=batch_idx,
            idx=idx,
            num_genes=num_genes,
            num_proteins=None,
            num_batches=num_batches,
            num_cells=num_cells,
        )
        return self._to_torch(data)

    def _setup_handler(self):
        """
        Sets up the handler for training the model.
        """
        train_model = partial(
            model,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        train_guide = partial(
            guide,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        idx = self.data.pop("idx")

        predict_model = partial(
            model,
            subsampling=0,
            minibatches=True,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        predict_guide = partial(
            guide,
            subsampling=0,
            minibatches=True,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        return SVILocalHandler(
            model=train_model,
            guide=train_guide,
            predict_model=predict_model,
            predict_guide=predict_guide,
            idx=idx,
            **self.training_kwargs,
        )

    def fit(self, *args, **kwargs):
        self.handler.fit(*args, **kwargs)

    def _meta_to_anndata(self, model_key=None, num_samples=25):
        model_key = self.model_key if model_key is None else model_key
        adata = self.adata

        adata.uns[f"{model_key}"] = {}
        res = adata.uns[f"{model_key}"]

        res["design"] = self.design_states.sparse
        res["intercept"] = self.batch_states.sparse

        res["design_index"] = self.design_states.index
        res["intercept_index"] = self.batch_states.index

        res["loss"] = self.handler.loss
        res["model"] = {"num_factors": self.num_factors, "seed": self.seed, **self.model_kwargs}

        return res

    def posterior_to_anndata(self, model_key=None, num_samples=25):
        _ = self._meta_to_anndata(model_key, num_samples)
        adata = self.adata

        adata.varm[f"{model_key}_W_rna"] = (
            self.handler.predict_global_variable("W_lin", num_samples=num_samples).T.swapaxes(-1, -3).swapaxes(-1, -2)
        )
        adata.varm[f"{model_key}_V_rna"] = self.handler.predict_global_variable(
            "W_add", num_samples=num_samples
        ).T.swapaxes(-1, -2)

        α_rna = self.handler.predict_global_variable("α_rna", num_samples=num_samples).T

        if α_rna.ndim == 2:
            α_rna = np.expand_dims(α_rna, 1)

        adata.varm[f"{model_key}_α_rna"] = α_rna.swapaxes(-1, -2)

        σ_rna = self.handler.predict_global_variable("σ_rna", num_samples=num_samples).T

        if σ_rna.ndim == 2:
            σ_rna = np.expand_dims(σ_rna, 1)

        adata.varm[f"{model_key}_σ_rna"] = σ_rna.swapaxes(-1, -2)

        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable("z", num_samples=num_samples).swapaxes(0, 1)

    def mean_to_anndata(self, model_key=None, num_samples=25, num_split=2048):
        _ = self._meta_to_anndata(model_key, num_samples)
        adata = self.adata

        adata.layers[f"{model_key}_μ_rna"] = self.handler.predict_local_variable(
            "μ_rna", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.layers[f"{model_key}_offset_rna"] = self.handler.predict_local_variable(
            "offset_rna", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
            "z", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.varm[f"{model_key}_W_rna"] = (
            self.handler.predict_global_variable("W_lin", num_samples=num_samples).mean(0).T
        )
        adata.varm[f"{model_key}_V_rna"] = (
            self.handler.predict_global_variable("W_add", num_samples=num_samples).mean(0).T
        )
        adata.varm[f"{model_key}_α_rna"] = self.handler.predict_global_variable("α_rna").mean(0).T
        adata.varm[f"{model_key}_σ_rna"] = self.handler.predict_global_variable("σ_rna").mean(0).T

    def to_anndata(self, model_key=None, num_samples=25, num_split=2048):
        res = self._meta_to_anndata(model_key, num_samples)
        adata = self.adata
        res["α_rna"] = self.handler.predict_global_variable("α_rna", num_samples=num_samples).mean(0)
        res["W_fac"] = self.handler.predict_global_variable("W_fac", num_samples=num_samples).mean(0)
        res["W_vec"] = self.handler.predict_global_variable("W_vec", num_samples=num_samples).mean(0)
        res["W_lin"] = self.handler.predict_global_variable("W_lin", num_samples=num_samples).mean(0)
        res["W_add"] = self.handler.predict_global_variable("W_add", num_samples=num_samples).mean(0)

        res["μ_rna"] = self.handler.predict_local_variable("μ_rna", num_samples=num_samples, num_split=num_split).mean(
            0
        )
        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
            "z", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.obsm[f"Z_{model_key}"] = self.handler.predict_local_variable(
            "z_vec", num_samples=num_samples, num_split=num_split
        ).mean(0)

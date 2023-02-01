from typing import Union

import numpy as np
import torch
from anndata import AnnData
from torch.types import Device

from .pca import scPCA
from .train import SUBSAMPLE
from .utils import get_protein_counts, get_rna_counts, get_states


class scCCA(scPCA):
    """
    scCCA model.

    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with the single-cell data.
    num_factors: int
        Number of factors to fit.
    protein_obsm_key: str or None (default: None)
        Key to extract single-cell protein matrix from `adata.obsm`.
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
    device: torch.device (default: torch.device("cuda") if a GPU is available)
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
        protein_obsm_key: str,
        layers_key: Union[str, None] = None,
        batch_formula: Union[str, None] = None,
        design_formula: Union[str, None] = None,
        subsampling: int = 4096,
        device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_key: str = "sccca",
        model_kwargs: dict = {
            "β_rna_sd": 0.01,
            "β_rna_mean": 3,
            "intercept": True,
            "batch_beta": False,
            "horseshoe": False,
        },
        training_kwargs: dict = SUBSAMPLE,
    ):
        self.protein_obsm_key = protein_obsm_key

        super().__init__(
            adata=adata,
            num_factors=num_factors,
            layers_key=layers_key,
            batch_formula=batch_formula,
            design_formula=design_formula,
            subsampling=subsampling,
            device=device,
            model_key=model_key,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
        )

    def _setup_data(self):
        """
        Sets up the data.
        """
        X = get_rna_counts(self.adata, self.layers_key)
        Y = get_protein_counts(self.adata, self.protein_obsm_key)
        X_size = np.log(X.sum(axis=1, keepdims=True))
        Y_size = np.log(Y.sum(axis=1, keepdims=True))
        batch = np.asarray(self.batch_states.encoding).astype(np.float32)
        design = np.asarray(self.design_states.encoding).astype(np.float32)
        batch_idx = self.batch_states.index
        design_idx = self.design_states.index

        num_genes = X.shape[1]
        num_cells = X.shape[0]
        num_batches = batch.shape[1]
        num_proteins = Y.shape[1]
        idx = np.arange(num_cells)

        data = dict(
            X=X,
            X_size=X_size,
            Y=Y,
            Y_size=Y_size,
            design=design,
            batch=batch,
            design_idx=design_idx,
            batch_idx=batch_idx,
            idx=idx,
            num_genes=num_genes,
            num_proteins=num_proteins,
            num_batches=num_batches,
            num_cells=num_cells,
        )
        return self._to_torch(data)

    def to_anndata(self, adata=None, model_key=None, num_samples=25):
        if model_key is None:
            model_key = self.model_key

        if adata is None:
            adata = self.adata

        # set up unstructured metadata
        adata.uns[f"{model_key}"] = {}
        adata.uns[f"{model_key}"]["posterior"] = {
            "μ_rna": self.handler.predict_local_variable("μ_rna", num_samples=num_samples).mean(0),
            "μ_prot": self.handler.predict_local_variable("μ_prot", num_samples=num_samples).mean(0),
            "α_rna": self.handler.predict_global_variable("α_rna", num_samples=num_samples).mean(0),
            "α_prot": self.handler.predict_global_variable("α_prot", num_samples=num_samples).mean(0),
            "V_prot": self.handler.predict_global_variable("V_fac", num_samples=num_samples).mean(0).T,
        }

        W_fac = self.handler.predict_global_variable("W_fac", num_samples=num_samples).mean(0)
        W_add = self.handler.predict_global_variable("W_add", num_samples=num_samples).mean(0)

        V_fac = self.handler.predict_global_variable("V_fac", num_samples=num_samples).mean(0)
        V_add = self.handler.predict_global_variable("V_add", num_samples=num_samples).mean(0)

        adata.uns[f"{model_key}"]["design"] = get_states(self.design_matrix)
        adata.uns[f"{model_key}"]["model"] = self.model_kwargs

        adata.uns[f"{model_key}"]["W_fac"] = {}
        adata.uns[f"{model_key}"]["W_add"] = {}
        adata.uns[f"{model_key}"]["V_fac"] = {}
        adata.uns[f"{model_key}"]["V_add"] = {}
        adata.uns[f"{model_key}"]["W_vec"] = {}
        adata.uns[f"{model_key}"]["W_scale"] = {}
        adata.uns[f"{model_key}"]["W_unit"] = {}
        adata.uns[f"{model_key}"]["V_vec"] = {}
        adata.uns[f"{model_key}"]["V_scale"] = {}
        adata.uns[f"{model_key}"]["V_unit"] = {}

        for i, (k, v) in enumerate(adata.uns[f"{model_key}"]["design"].items()):
            adata.uns[f"{model_key}"]["W_fac"][k] = W_fac[v[-1]]
            adata.uns[f"{model_key}"]["W_vec"][k] = W_fac[v].sum(axis=0)
            adata.uns[f"{model_key}"]["W_scale"][k] = np.linalg.norm(W_fac[v].sum(axis=0), ord=2, axis=1, keepdims=True)
            adata.uns[f"{model_key}"]["W_unit"][k] = W_fac[v].sum(axis=0) / np.linalg.norm(
                W_fac[v].sum(axis=0), ord=2, axis=1, keepdims=True
            )

            adata.uns[f"{model_key}"]["V_fac"][k] = V_fac[v[-1]]
            adata.uns[f"{model_key}"]["V_vec"][k] = V_fac[v].sum(axis=0)
            adata.uns[f"{model_key}"]["V_scale"][k] = np.linalg.norm(V_fac[v].sum(axis=0), ord=2, axis=1, keepdims=True)
            adata.uns[f"{model_key}"]["V_unit"][k] = V_fac[v].sum(axis=0) / np.linalg.norm(
                V_fac[v].sum(axis=0), ord=2, axis=1, keepdims=True
            )

        for k, v in get_states(self.batch_matrix).items():
            adata.uns[f"{model_key}"]["W_add"][k] = W_add[v[-1]]
            adata.uns[f"{model_key}"]["V_add"][k] = V_add[v[-1]]

        adata.uns[f"{model_key}"]["design"] = get_states(self.design_matrix)
        # adata.uns[f"{model_key}"]["batch"] = get_states(self.batch)
        adata.uns[f"{model_key}"]["model"] = self.model_kwargs

        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable("z", num_samples=num_samples).mean(0)

        adata.varm[f"{model_key}"] = self.handler.predict_global_variable("W_fac", num_samples=num_samples).mean(0).T

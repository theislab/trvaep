from model import CVAE
from model import ModelTrainer
import scanpy as sc

adata = sc.read("./data/kang_seurat.h5ad")
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.filter_genes_dispersion(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000)
adata = adata[:, adata.var['highly_variable']]
n_conditions = adata.obs["condition"].unique().shape[0]
model = CVAE(adata.n_vars, num_classes=n_conditions,
             encoder_layer_sizes=[64], decoder_layer_sizes=[64], latent_dim=10, alpha=0.0001,
             use_mmd=True, beta=10)
trainer = ModelTrainer(model, adata)
trainer.train_trvae(20, 64)
data = model.get_y(adata.X.A, model.label_encoder.transform(adata.obs["condition"]))
adata_latent = sc.AnnData(data)
adata_latent.obs["cell_type"] = adata.obs["cell_type"].tolist()
adata_latent.obs["condition"] = adata.obs["condition"].tolist()
sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent, color=["condition", "cell_type"])

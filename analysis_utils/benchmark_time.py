from scipy.stats import spearmanr
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import sys

if sys.path[0] != "../":
    sys.path.insert(0, "../")
from analysis_utils.utils_qc import complete_trajectory_analysis
import analysis_utils.utils_qc as utils_qc
import analysis_utils.utils as utils


time_mapping = {'E3': 3, 'E4': 4, 'E5': 5, 'E6': 6, 'E7': 7}
root_cell = 'E3'

adata_g = utils.get_adata_g('time3929_04n2', 'time3929_04n2_with_rebuild', 100)
adata_g, pos_value, p_value = complete_trajectory_analysis(adata_g, root_cell, time_mapping, time_column='time', use_latent=False)

import matplotlib.pyplot as plt

# 创建子图，并列显示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 按DPT伪时间着色
sc.pl.umap(adata_g, color='dpt_pseudotime', ax=ax1, show=False,
           cmap='viridis', title='UMAP - Diffusion Pseudotime')
# 添加颜色条标签
cbar1 = ax1.collections[0].colorbar
if cbar1 is not None:
    cbar1.set_label('Pseudotime')

# 按真实时间标签着色
sc.pl.umap(adata_g, color='time', ax=ax2, show=False,
           title='UMAP - Developmental Time', legend_loc='right')

plt.tight_layout()
plt.show()
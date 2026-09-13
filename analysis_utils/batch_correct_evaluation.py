import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from typing import Union, Dict
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Optional
import sys


if sys.path[0] != "../":
    sys.path.insert(0, "../")
import analysis_utils.utils as utils
from analysis_utils.cluster_evaluation_v2 import compare_clusters, cluster, cluster_latent


def knn_divergence_per_class(Z_s: np.ndarray, Z_t: np.ndarray, k: int = 30) -> float:
    """
    计算单个细胞类型的 kNN 散度（从源到目标方向）。

    参数
    ----------
    Z_s : np.ndarray, shape (m, d)
        源批次中该细胞类型的嵌入矩阵。
    Z_t : np.ndarray, shape (n, d)
        目标批次中该细胞类型的嵌入矩阵。
    k : int, default=30
        近邻个数。

    返回
    -------
    float
        该细胞类型的散度值。若任一批次样本数不足 k，则返回 NaN。
    """
    m, d = Z_s.shape
    n = Z_t.shape[0]
    if m == 0 or n == 0:
        return np.nan
    # 实际使用的 k 不能超过样本数
    k_s = min(k, m)
    k_t = min(k, n)

    # 计算源批次内每个点的第 k 近邻距离
    nn_s = NearestNeighbors(n_neighbors=k_s, metric='euclidean')
    nn_s.fit(Z_s)
    dist_s, _ = nn_s.kneighbors(Z_s, n_neighbors=k_s)
    rho_s = dist_s[:, -1]  # 第 k 近邻距离

    # 计算源批次每个点到目标批次的第 k 近邻距离
    nn_t = NearestNeighbors(n_neighbors=k_t, metric='euclidean')
    nn_t.fit(Z_t)
    dist_t, _ = nn_t.kneighbors(Z_s, n_neighbors=k_t)
    v_s = dist_t[:, -1]

    # 防止距离为零取对数
    eps = 1e-10
    rho_s = np.maximum(rho_s, eps)
    v_s = np.maximum(v_s, eps)

    sum_log = np.sum(np.log(v_s / rho_s))
    div = (d / n) * sum_log + np.log(m / (n - 1 + eps))
    return div





def evaluate_batch_correction(
        adata_src: ad.AnnData,
        adata_tgt: ad.AnnData,
        latent_key: str = 'latent',
        celltype_key: str = 'celltype',
        k: int = 30,
        use_gex: bool = False,
        reduction_method: str = 'pca',
        n_components: int = 50
) -> Dict[str, float]:
    """
    评估两个批次校正后的嵌入质量，返回轮廓系数和 kNN 散度。
    参数
    ----------
    adata_src, adata_tgt : AnnData
        源和目标批次数据，需包含：
        - .obsm[latent_key] : 校正后嵌入矩阵，形状 (n_cells, n_dim) 或
        - .X : 基因表达矩阵（当 use_reduction=True 时）
        - .obs[celltype_key] : 细胞类型标签
    latent_key : str, default='latent'
        嵌入在 .obsm 中的键名。
    celltype_key : str, default='celltype'
        细胞类型在 .obs 中的列名。
    k : int, default=30
        kNN 散度中使用的近邻个数。
    use_gex : bool, default=False
        是否使用降维后的基因表达数据。如果为 True，则从 .X 中提取基因表达并进行降维。
    reduction_method : str, default='pca'
        降维方法，可选 'pca' 或 'umap'。
    n_components : int, default=50
        降维后的维度数。
    返回
    -------
    dict
        {'silhouette_score': float, 'divergence_score': float}
    """
    # 检查细胞类型标签是否存在
    if celltype_key not in adata_src.obs or celltype_key not in adata_tgt.obs:
        raise KeyError(f"'{celltype_key}' not found in .obs of both AnnData objects.")
    # 提取细胞类型标签
    labels_src = adata_src.obs[celltype_key].values
    labels_tgt = adata_tgt.obs[celltype_key].values
    # 提取或计算嵌入
    # 使用基因表达数据并进行降维
    # 先合并两个批次，降维后再分开
    src = adata_src.copy()
    tgt = adata_tgt.copy()

    # 添加批次标识
    src.obs['batch'] = 'source'
    tgt.obs['batch'] = 'target'

    # 合并两个 AnnData 对象
    adata_combined = src.concatenate(tgt, batch_key='batch', batch_categories=['source', 'target'])

    if use_gex:

        # 对合并后的数据进行预处理和降维
        # sc.pp.scale(adata_combined, max_value=10)

        if reduction_method == 'pca':
            sc.tl.pca(adata_combined, n_comps=n_components)
            embedding_key = 'X_pca'
        elif reduction_method == 'umap':
            # 先做 PCA 作为 UMAP 的输入
            sc.tl.pca(adata_combined, n_comps=50)
            sc.pp.neighbors(adata_combined, n_neighbors=10, use_rep='X_pca')
            sc.tl.umap(adata_combined, n_components=n_components)
            embedding_key = 'X_umap'
        else:
            raise ValueError(f"Unsupported reduction method: {reduction_method}. Use 'pca' or 'umap'.")

        # 分离源批次和目标批次的嵌入
        src_mask = adata_combined.obs['batch'] == 'source'
        tgt_mask = adata_combined.obs['batch'] == 'target'
        X_src = adata_combined.obsm[embedding_key][src_mask]
        X_tgt = adata_combined.obsm[embedding_key][tgt_mask]

        adata_combined = cluster(adata_combined)
    else:
        # 使用现有的嵌入矩阵
        if latent_key not in adata_src.obsm or latent_key not in adata_tgt.obsm:
            raise KeyError(f"'{latent_key}' not found in .obsm of both AnnData objects.")
        X_src = adata_src.obsm[latent_key]
        X_tgt = adata_tgt.obsm[latent_key]

        adata_combined = cluster_latent(adata_combined)
    # 合并数据用于轮廓系数计算
    X_combined = np.vstack([X_src, X_tgt])
    labels_combined = np.concatenate([labels_src, labels_tgt])

    # 轮廓系数（使用欧氏距离）
    sil_label = silhouette_score(X_combined, labels_combined, metric='euclidean')

    batch_combined = np.concatenate([src.obs['batch'].values, tgt.obs['batch'].values])
    sil_batch = silhouette_score(X_combined, batch_combined, metric='euclidean')

    # 找出两个批次共有的细胞类型
    common_types = set(labels_src) & set(labels_tgt)
    if len(common_types) == 0:
        raise ValueError("No common cell types found between the two batches.")

    # 对每个共有类型计算双向 kNN 散度并平均
    div_scores = []
    for ct in common_types:
        # 获取当前类型的嵌入
        src_idx = np.where(labels_src == ct)[0]
        tgt_idx = np.where(labels_tgt == ct)[0]
        Z_src = X_src[src_idx]
        Z_tgt = X_tgt[tgt_idx]

        # 两个方向散度
        div_src2tgt = knn_divergence_per_class(Z_src, Z_tgt, k=k)
        div_tgt2src = knn_divergence_per_class(Z_tgt, Z_src, k=k)

        # 取平均（忽略 NaN，例如某类样本数太少导致无法计算）
        div_mean = np.nanmean([div_src2tgt, div_tgt2src])
        if not np.isnan(div_mean):
            div_scores.append(div_mean)

    if len(div_scores) == 0:
        divergence = np.nan
    else:
        divergence = np.mean(div_scores)

    ari, nmi, purity = compare_clusters(adata_combined.obs['leiden'], adata_combined.obs[celltype_key])

    return {
        'silhouette_score_batch': sil_batch,
        'divergence_score': divergence,
        'leiden_ari': ari,
        'leiden_nmi': nmi,
        'leiden_purity': purity,
        'silhouette_score_label': sil_label,
    }


def plot_umap_batch_celltype(
        adata_src: ad.AnnData,
        adata_tgt: ad.AnnData,
        latent_key: str = 'latent',
        batch_name: str = 'batch',
        celltype_key: str = 'celltype',
        save: Optional[str] = None,
        figsize: tuple = (12, 5),
        use_reduction: bool = False,
        reduction_method: str = 'pca',
        n_components: int = 50
):
    """
    将两个批次的 AnnData 对象合并，基于 latent_key 中的嵌入计算 UMAP，
    并分别按批次和细胞类型着色。
    参数
    ----------
    adata_src, adata_tgt : AnnData
        源和目标批次数据，需包含：
        - .obsm[latent_key] : 低维嵌入（当 use_reduction=False 时）或
        - .X : 基因表达矩阵（当 use_reduction=True 时）
        - .obs[celltype_key] : 细胞类型标签
    latent_key : str, default='latent'
        低维嵌入在 .obsm 中的键名（如校正后的 2~50 维向量）。
    batch_name : str, default='batch'
        合并后用于标识批次的列名。
    save : str or None, default=None
        若提供路径，则保存图像（如 'umap.png'）。
    figsize : tuple, default=(12,5)
        图像大小。
    use_reduction : bool, default=False
        是否使用降维后的基因表达数据。如果为 True，则从 .X 中提取基因表达并进行降维。
    reduction_method : str, default='pca'
        降维方法，可选 'pca' 或 'umap'。
    n_components : int, default=50
        降维后的维度数。
    """
    # 1. 复制数据，避免修改原始对象
    src = adata_src.copy()
    tgt = adata_tgt.copy()
    # 2. 检查必要字段
    if celltype_key not in src.obs or celltype_key not in tgt.obs:
        raise KeyError(f"'{celltype_key}' not found in .obs of both AnnData objects.")
    # 3. 添加批次标识
    tgt.obs[batch_name] = 'target'
    # 4. 合并两个 AnnData
    adata_combined = src.concatenate(tgt, batch_key=batch_name, batch_categories=['source', 'target'])
    # 5. 提取或计算嵌入
    if use_reduction:
        # 使用基因表达数据并进行降维
        # 先合并两个批次，降维后再分开
        # 对合并后的数据进行预处理和降维
        sc.pp.scale(adata_combined, max_value=10)

        if reduction_method == 'pca':
            sc.tl.pca(adata_combined, n_comps=n_components)
            embedding_key = 'X_pca'
        elif reduction_method == 'umap':
            # 先做 PCA 作为 UMAP 的输入
            sc.tl.pca(adata_combined, n_comps=50)
            sc.pp.neighbors(adata_combined, n_neighbors=10, use_rep='X_pca')
            sc.tl.umap(adata_combined, n_components=n_components)
            embedding_key = 'X_umap'
        else:
            raise ValueError(f"Unsupported reduction method: {reduction_method}. Use 'pca' or 'umap'.")

        # 分离源批次和目标批次的嵌入
        src_mask = adata_combined.obs[batch_name] == 'source'
        tgt_mask = adata_combined.obs[batch_name] == 'target'
        X_src = adata_combined.obsm[embedding_key][src_mask]
        X_tgt = adata_combined.obsm[embedding_key][tgt_mask]
    else:
        # 使用现有的嵌入矩阵
        if latent_key not in src.obsm or latent_key not in tgt.obsm:
            raise KeyError(f"'{latent_key}' not found in .obsm of both AnnData objects.")
        X_src = src.obsm[latent_key]
        X_tgt = tgt.obsm[latent_key]
    # 合并嵌入矩阵
    X_combined = np.vstack([X_src, X_tgt])
    # 创建一个只包含嵌入的临时 AnnData 用于 UMAP 计算
    adata_emb = sc.AnnData(X_combined)
    adata_emb.obs = adata_combined.obs.copy()  # 继承合并后的 obs（包含 batch 和 celltype）

    # 5. 基于嵌入计算 UMAP
    sc.pp.neighbors(adata_emb, use_rep='X')  # 使用嵌入作为表达矩阵
    sc.tl.umap(adata_emb)

    # 6. 绘制 UMAP
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 按批次着色
    sc.pl.umap(adata_emb, color=batch_name, ax=axes[0], show=False, title='Colored by Batch')
    # 按细胞类型着色
    sc.pl.umap(adata_emb, color=celltype_key, ax=axes[1], show=False, title='Colored by Cell Type')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()

    return adata_emb  # 返回包含 UMAP 坐标的对象，以便后续使用
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,silhouette_score
import numpy as np
from sklearn import metrics
import scanpy as sc
import seaborn as sns
import sys

if sys.path[0] != "../":
    sys.path.insert(0, "../")
import analysis_utils.utils as utils


def compare_clusters(cluster_labels1, cluster_labels2):
    """比较两个聚类结果的相似性
    Args:
        cluster_labels1: 第一个聚类结果标签数组
        cluster_labels2: 第二个聚类结果标签数组
    Returns:
        ari: 调整兰德指数
        nmi: 标准化互信息
        purity: 纯度得分
    """
    # 计算ARI和NMI
    ari = adjusted_rand_score(cluster_labels1, cluster_labels2)
    nmi = normalized_mutual_info_score(cluster_labels1, cluster_labels2)

    # 计算纯度
    def purity_score(y_true, y_pred):
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    purity = purity_score(cluster_labels1, cluster_labels2)

    # 计算轮廓系数
    # if len(cluster_labels1.unique()) > 1:  # 确保至少有2个聚类
    #     silhouette_avg = silhouette_score(X_pca, adata.obs["cluster"].astype(int))
    #     print(f"Silhouette Score: {silhouette_avg:.4f}")
    # else:
    #     silhouette_avg = np.nan
    #     print("Silhouette Score: Not applicable (only one cluster)")

    print(f"Clusters Comparison - ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {purity:.4f}")
    return ari, nmi, purity


def cluster(adata, draw=False, seed=42):
    # PCA降维
    sc.tl.pca(adata, svd_solver='arpack')
    # 2. 构建邻接图（基于PCA结果）
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    # 3. Leiden聚类（分辨率参数调整聚类粒度）
    sc.tl.leiden(adata, resolution=0.5, random_state=seed)
    # 4. 结果存储在 adata.obs['leiden']
    print(adata.obs['leiden'].head())

    # 可选：可视化聚类结果（UMAP）
    if draw:
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['leiden'])

    return adata


def cluster_latent(adata, draw=False, seed=42):
    # 检查是否存在latent空间
    if 'latent' not in adata.obsm:
        raise ValueError(
            "adata.obsm['latent'] does not exist. Please ensure the latent space is computed before clustering.")

    # 构建邻接图（基于latent空间）
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='latent')
    # Leiden聚类（分辨率参数调整聚类粒度）
    sc.tl.leiden(adata, resolution=0.5, random_state=seed)
    # 结果存储在 adata.obs['leiden']
    print(adata.obs['leiden'].head())
    # 可选：可视化聚类结果（UMAP）
    if draw:
        sc.tl.umap(adata, use_rep='latent')
        sc.pl.umap(adata, color=['leiden'])
    return adata


def compare_cluster_visulise(adata_before, adata_after):

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 处理前的UMAP
    sc.pl.umap(adata_before, color='leiden', ax=ax1, show=False)
    ax1.set_title('Before Treatment')

    # 处理后的UMAP
    sc.pl.umap(adata_after, color='leiden', ax=ax2, show=False)
    ax1.set_title('After Treatment')

    plt.tight_layout()
    plt.show()

    # 创建交叉表显示聚类对应关系
    cross_tab = pd.crosstab(
        adata_before.obs['leiden'],
        adata_after.obs['leiden'],
        normalize='index'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_tab, annot=True, cmap='viridis')
    plt.title('Cluster Correspondence Between Conditions')
    plt.xlabel('After Treatment Clusters')
    plt.ylabel('Before Treatment Clusters')
    plt.show()

    # 在处理前后数据上分别进行轨迹推断
    sc.tl.diffmap(adata_before)
    sc.tl.diffmap(adata_after)

    # 可视化扩散图组件
    sc.pl.diffmap(adata_before, color='leiden')
    sc.pl.diffmap(adata_after, color='leiden')


if __name__ == '__main__':
    from train.train_map_single_base import *
    import analysis_utils.utils_qc as utils_qc
    from train.generate_map_single_v2 import generate_map

    device = 'cuda'  # todo: check the device
    seed = 42
    seed_everything(seed)
    tag_t = 'LIHC_GSE140228'
    tag_s = tag_t

    save_path = '../analysis_results/qc250812/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_type = '10x'

    sava_tag = 'LIHC_GSE140228_260105'

    save_dir_root = f'../output_single/{sava_tag}/'

    # 当使用非配对数据集时，需要修改下面的参数
    data_t = f'../data/single_train/{tag_t}_10X_10x_immune.h5ad'
    data_s = f'../data/single_train/{tag_s}_Smartseq2_smartseq_immune.h5ad'

    save_dir_t = f'{save_dir_root}10x_{tag_t}/'
    save_dir_s = f'{save_dir_root}smartseq_{tag_s}/'
    save_dir_g = f'{save_dir_root}{tag_t}_to_{tag_s}/'

    args_ed = {'loss_ae': 'mse',
               'decoder_activation': 'ReLU',
               'local_rank': 0,
               'seed': seed,
               'hparams': '',
               'epochs_t': 100,
               'epochs_s': 100,
               'batch_size': 128,
               'negative_selection': 'hardest',

               'data_t': data_t,
               'data_s': data_s,
               'state_dict': '../pretrain/annotation_model_v1',
               'save_dir_t': save_dir_t,
               'save_dir_s': save_dir_s,
               'device': device
               }

    epochs_t = args_ed['epochs_t']
    epochs_s = args_ed['epochs_s']
    epochs_g = 100
    # state_dict_t = save_dir_t + f'model_seed={seed}_step={epochs_t - 1}.pt'
    # state_dict_s = save_dir_s + f'model_seed={seed}_step={epochs_s - 1}.pt'
    state_dict_t = save_dir_g + f'ae_10x_seed={seed}_step={epochs_g - 1}.pt'
    state_dict_s = save_dir_g + f'ae_smartseq_seed={seed}_step={epochs_g - 1}.pt'

    args_map = {'loss_ae': 'mse',
                'decoder_activation': 'ReLU',

                'seed': seed,
                'epochs': epochs_g,
                'batch_size': 512,
                'lr': 2e-3,

                'state_dict_t': state_dict_t,
                'state_dict_s': state_dict_s,
                'save_dir': save_dir_g,
                'data_dir_t': data_t,
                'data_dir_s': data_s,
                'device': device,

                'latent_dim': 128,
                'cyc_coef': 10,
                'sty_coef': 0,
                'cont_coef': 0,
                }

    args_map['state_dict_g'] = args_map['save_dir'] + f'model_seed={seed}_step={epochs_g - 1}.pt'

    seed = args_ed["seed"]

    # 获取两个数据集中的共同类型的细胞
    adata_10x = sc.read_h5ad(data_t)
    adata_smartseq = sc.read_h5ad(data_s)

    # 过滤掉细胞数量少于500的细胞类型
    adata_10x = utils.filter_cell_types_by_count(adata_10x)
    adata_smartseq = utils.filter_cell_types_by_count(adata_smartseq)

    type_smart = adata_smartseq.obs['celltype'].value_counts()
    type_10x = adata_10x.obs['celltype'].value_counts()

    cross_type = type_smart.index.intersection(type_10x.index)
    diff_type = type_smart.index.difference(type_10x.index)

    adata_smartseq_common = adata_smartseq[adata_smartseq.obs['celltype'].isin(cross_type)]
    adata_10x_common = adata_10x[adata_10x.obs['celltype'].isin(cross_type)]

    adata_10x = adata_10x_common.copy()
    adata_smartseq = adata_smartseq_common.copy()

    adata_10x, adata_smartseq, adata_g = generate_map(adata_10x, adata_smartseq, args_map)

    adata_10x = cluster(adata_10x)
    adata_g = cluster(adata_g)

    res = compare_clusters(adata_10x.obs['leiden'], adata_g.obs['leiden'])

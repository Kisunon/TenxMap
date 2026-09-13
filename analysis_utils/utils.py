import os
import sys
import anndata
import h5py
import numpy as np
import scanpy as sc
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, issparse
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

if sys.path[0] != "../":
    sys.path.insert(0, "../")
from train.generate_map_single_v2 import generate_map


def random_sample_from_adata(adata, num_sample, cell_type=None, batch=None, cell_type_name='celltype',
                             batch_name='batch', seed=None):
    '''
    从adata中随机采样num_sample个细胞，可选过滤条件：
    1. cell_type: 细胞类型
    2. batch: 批次（筛选条件2）
    3. cell_type_name: 细胞类型列名
    :param adata: 要采样的adata对象
    :param num_sample: 采样的细胞数量
    :param cell_type: 列表，要筛选的细胞类型，为None时表示考虑所有细胞类型
    :param batch: 列表，要筛选的批次，为None时表示不考虑该筛选条件
    :param cell_type_name: 细胞类型在adata.obs中的列名，默认为'celltype'
    :param batch_name: 批次在adata.obs中的列名，默认为'batch'
    :param seed: 随机种子，默认为None
    :return:随机采样的adata对象
    '''
    if seed is not None:
        np.random.seed(seed)

    if cell_type is None:
        cell_type = adata.obs[cell_type_name].unique()
    celltypes = adata.obs[cell_type_name].values

    if batch is None:
        mask = np.isin(celltypes, cell_type)
    else:
        batchs = adata.obs[batch_name].values
        mask = np.isin(batchs, batch) & np.isin(celltypes, cell_type)

    idxs = np.where(mask)[0]
    if len(idxs) >= num_sample:
        random_indices = np.random.choice(idxs, size=num_sample, replace=False)
    else:
        print('sample num is larger than total num, use all samples')
        random_indices = idxs

    adata_filtered = adata[random_indices].copy()

    return adata_filtered


def random_sample_every_cell_type(adata, cell_types, num_sample, seed=None):
    all_adata_types = []
    for type_ in cell_types:
        adata_type = random_sample_from_adata(adata, num_sample, [type_], seed=seed)
        all_adata_types.append(adata_type)
    combined_adata = sc.concat(all_adata_types)
    return combined_adata


def draw_umap_with_generate_smart(adata_pred, adata_smartseq, draw_latent=False, save_path=None):
    if draw_latent:
        z_smart_pred = adata_pred.obsm['latent']
        z_smart_true = adata_smartseq.obsm['latent']
    else:
        z_smart_pred = adata_pred.X
        z_smart_true = adata_smartseq.X

    combined_data = np.vstack([z_smart_pred, z_smart_true])
    labels_batch = ['batch_generate'] * len(z_smart_pred) + ['batch_target'] * len(z_smart_true)
    celltype = np.concatenate([adata_pred.obs.celltype.values, adata_smartseq.obs.celltype.values])

    adata = sc.AnnData(X=combined_data)
    adata.obs['batch'] = labels_batch
    adata.obs['celltype'] = celltype

    # sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(8, 6), frameon=True)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # 6. 绘制 UMAP
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 按批次着色
    sc.pl.umap(adata, color='batch', ax=axes[0], show=False, title='Colored by Batch')
    # 按细胞类型着色
    sc.pl.umap(adata, color='celltype', ax=axes[1], show=False, title='Colored by Cell Type')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # if save_path is not None:
    #     # Draw the category plot
    #     ax1 = sc.pl.umap(adata, color=['batch'], legend_fontsize=10, show=False)
    #     fig1 = ax1.get_figure()  # Get the Figure object from the Axes object
    #     plt.tight_layout()
    #     fig1.savefig(save_path + 'batch.png')
    #     plt.close(fig1)
    #
    #     # Draw the celltype plot
    #     ax2 = sc.pl.umap(adata, color=['celltype'], legend_fontsize=10, show=False)
    #     fig2 = ax2.get_figure()  # Get the Figure object from the Axes object
    #     plt.tight_layout()
    #     fig2.savefig(save_path + '_celltype.png')
    #     plt.close(fig2)
    # else:
    #     sc.pl.umap(adata, color=['batch'], legend_fontsize=10)
    #     sc.pl.umap(adata, color=['celltype'], legend_fontsize=10)
    #     plt.tight_layout()


def read_single_cell_h5_to_dataframe(file_path):
    """
        从指定的HDF5文件中读取单细胞数据，并将其转换为稀疏的Pandas DataFrame。
        参数:
        file_path (str): 包含单细胞数据的HDF5文件的路径。
        返回:
        pandas.DataFrame: 一个稀疏的DataFrame，其中行表示细胞，列表示基因。
        提示: 要将返回的稀疏DataFrame转换为密集DataFrame，可以使用以下方法:
              read_single_cell_h5_to_dataframe(single_cell_gex_path).sparse.to_dense()
        """
    # output a sparse df, to dense: read_single_cell_h5_to_dataframe(single_cell_gex_path).sparse.to_dense()

    with h5py.File(file_path, 'r') as f:
        matrix_group = f['matrix']
        data = matrix_group['data'][:]
        barcodes = np.array(matrix_group['barcodes'][:])
        features_group = matrix_group['features']
        if 'name' in features_group and isinstance(features_group['name'], h5py.Dataset):
            features = np.array(features_group['name'][:])
        else:
            raise ValueError("'name' dataset not found in 'features' group.")
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        shape = f['matrix/shape'][:]
        matrix_shape = (shape[1], shape[0])

    expression_matrix = csr_matrix((data, indices, indptr), shape=matrix_shape)
    barcodes_utf = [x.decode('utf-8') for x in barcodes]
    features_utf = [x.decode('utf-8') for x in features]

    df = pd.DataFrame.sparse.from_spmatrix(expression_matrix, index=barcodes_utf, columns=features_utf)

    return df


def filter_and_insert_genes(adata, gene_set):
    """
    根据指定基因集筛选 AnnData 对象中的基因，插入缺失基因并设表达值为 0，
    同时打印删除和插入的基因数量。

    参数:
    adata (anndata.AnnData): 输入的 AnnData 对象。
    gene_set (list): 指定的基因集。

    返回:
    anndata.AnnData: 处理后的 AnnData 对象。
    """
    # 获取当前 AnnData 中的基因名
    current_genes = adata.var_names.tolist()

    # 找出需要删除的基因
    genes_to_delete = [gene for gene in current_genes if gene not in gene_set]
    # 找出需要插入的基因
    genes_to_insert = [gene for gene in gene_set if gene not in current_genes]

    # 打印删除和插入的基因数量
    print(f"删除的基因数量: {len(genes_to_delete)}")
    print(f"插入的基因数量: {len(genes_to_insert)}")

    # 筛选出基因集中存在的基因
    adata = adata[:, [gene for gene in current_genes if gene in gene_set]]

    # 插入缺失的基因
    if genes_to_insert:
        new_columns = np.zeros((adata.n_obs, len(genes_to_insert)))
        new_var = pd.DataFrame(index=genes_to_insert)
        new_adata = anndata.AnnData(
            X=new_columns,
            obs=adata.obs,
            var=new_var
        )
        adata = anndata.concat([adata, new_adata], axis=1)
        # 重新排序基因以匹配基因集顺序
        adata = adata[:, gene_set]

    return adata


def calculate_mean_scc(real_data, generated_data, gene_wise=True):
    """
    计算两组单细胞数据的Spearman相关系数 (SCC)

    参数:
        real_data (np.ndarray): 真实细胞数据，形状为 (n_cells_real, n_genes)
        generated_data (np.ndarray): 生成细胞数据，形状为 (n_cells_gen, n_genes)
        gene_wise (bool): 若为True，按基因计算平均表达值的SCC；若为False，按单细胞直接计算（需n_cells_real = n_cells_gen）

    返回:
        scc (float): Spearman相关系数值
    """
    # 检查基因维度是否一致
    if real_data.shape[1] != generated_data.shape[1]:
        raise ValueError("基因数量不一致！")

    if gene_wise:
        # 按基因计算平均表达值
        real_mean = np.mean(real_data, axis=0)
        gen_mean = np.mean(generated_data, axis=0)
        # 计算SCC

        scc, _ = spearmanr(real_mean, gen_mean)
    else:  # TODO: 测试这一部分的情况
        # 按单细胞直接计算（需样本数相同）
        if real_data.shape[0] != generated_data.shape[0]:
            raise ValueError("样本数不一致，无法逐细胞计算！")
        # 展平为向量（假设所有基因独立）
        real_flat = real_data.flatten()
        gen_flat = generated_data.flatten()
        scc, _ = spearmanr(real_flat, gen_flat)

    return scc


def calculate_cellwise_pcc(adata_gen, adata_real, norm=False, parallel=-1):
    """
    逐细胞计算生成数据与真实数据的SCC，并返回全局统计量和相关性矩阵

    Parameters
    ----------
    adata_real : AnnData
        真实数据的AnnData对象
    adata_gen : AnnData
        生成数据的AnnData对象

    Returns
    -------
    global_scc : float
        全局SCC（所有细胞对的均值）
    global_std : float
        全局SCC的标准差
    corr_matrix : np.ndarray
        相关性矩阵（形状：n_gen_cells × n_real_cells）
    """

    # 1. 数据预处理
    # --------------------------------------------------
    if norm:
        # 总计数归一化至1e4（CPM）
        sc.pp.normalize_total(adata_real, target_sum=1e4)
        sc.pp.normalize_total(adata_gen, target_sum=1e4)

        # 对数转换（log1p）
        sc.pp.log1p(adata_real)
        sc.pp.log1p(adata_gen)

    # 基因对齐（按真实数据顺序）
    shared_genes = adata_real.var_names.intersection(adata_gen.var_names)
    adata_real = adata_real[:, shared_genes].copy()
    adata_gen = adata_gen[:, shared_genes].copy()
    assert np.all(adata_real.var_names == adata_gen.var_names), "基因名称或顺序不一致！"

    # 转换为稠密数组（若内存允许）
    X_real = adata_real.X.toarray() if hasattr(adata_real.X, 'toarray') else adata_real.X
    X_gen = adata_gen.X.toarray() if hasattr(adata_gen.X, 'toarray') else adata_gen.X

    # 2. 逐细胞计算SCC
    # --------------------------------------------------
    n_gen = X_gen.shape[0]
    n_real = X_real.shape[0]
    corr_matrix = np.zeros((n_gen, n_real))

    if parallel is not None:
        from joblib import Parallel, delayed

        def calculate_row(i, X_gen, X_real):
            row = np.zeros(X_real.shape[0])
            for j in range(X_real.shape[0]):
                # row[j], _ = spearmanr(X_gen[i], X_real[j])
                row[j], _ = pearsonr(X_gen[i], X_real[j])
            return row

        corr_matrix = Parallel(n_jobs=-1)(
            delayed(calculate_row)(i, X_gen, X_real)
            for i in tqdm(range(n_gen)))
        corr_matrix = np.array(corr_matrix)

    else:
        # 遍历每个生成细胞
        for i in tqdm(range(n_gen), desc="计算生成细胞与真实细胞的SCC"):
            gen_cell = X_gen[i, :]
            # 计算与所有真实细胞的SCC
            for j in range(n_real):
                real_cell = X_real[j, :]
                # scc, _ = spearmanr(gen_cell, real_cell)
                scc, _ = pearsonr(gen_cell, real_cell)
                corr_matrix[i, j] = scc

    # 3. 计算全局统计量
    # --------------------------------------------------
    # global_scc = np.nanmean(corr_matrix)  # 忽略NaN
    # global_std = np.nanstd(corr_matrix)
    max_values = np.max(corr_matrix, axis=1)
    global_scc = np.mean(max_values)
    global_std = np.std(max_values)

    return global_scc, global_std, corr_matrix


def compute_scc_for_each_cell_type(adata1, adata2, cell_types, results_df, results_column):
    print('calculate mean scc')
    for type_ in cell_types:
        adata1_type = adata1[adata1.obs.celltype == type_].copy()
        adata2_type = adata2[adata2.obs.celltype == type_].copy()
        scc = calculate_mean_scc(adata1_type.X, adata2_type.X)
        print(type_, scc)
        results_df.loc[type_, results_column] = scc
    return results_df


def compute_pcc_cellwise_for_each_cell_type(adata1, adata2, cell_types, results_df, results_column):
    print('calculate cellwise pcc')
    for type_ in cell_types:
        adata1_type = adata1[adata1.obs.celltype == type_].copy()
        adata2_type = adata2[adata2.obs.celltype == type_].copy()
        scc, sigma, _ = calculate_cellwise_pcc(adata1_type, adata2_type)
        print(type_, scc, sigma)
        results_df.loc[type_, results_column] = scc
        results_df.loc[type_, results_column + 3] = sigma
    return results_df


def calculate_results_for_test_data_all_sample(test_tag, gen_cell_types, adata_smartseq, args, top_indices=None,
                                               save_path_tag=None, is_multi_type=False):
    adata_test = sc.read_h5ad(f'../data/test_data/{test_tag}_immune.h5ad')
    gen_tissue_cell_types = [test_tag.split('_')[0] + '_' + cell_type for cell_type in gen_cell_types]

    adata_test_gen = adata_test[adata_test.obs.tissue_celltype.isin(gen_tissue_cell_types)]
    adata_smartseq_gen = adata_smartseq[adata_smartseq.obs.tissue_celltype.isin(gen_tissue_cell_types)]

    adata_test_source, adata_smartseq_gen, adata_test_gen = generate_map(adata_test_gen, adata_smartseq_gen, args,
                                                                         is_multi_type=is_multi_type)

    if top_indices is None:
        adata_hvg_s = adata_smartseq_gen
        adata_hvg_g = adata_test_gen
    else:
        adata_hvg_s = adata_smartseq_gen[:, top_indices]
        adata_hvg_g = adata_test_gen[:, top_indices]

    # all_types = ['B', 'CD4Tconv', 'CD8T', 'CD8Tex', 'Mast', 'Mono/Macro', 'NK', 'Plasma']
    all_types = gen_cell_types

    if issparse(adata_hvg_s.X):
        adata_hvg_s = adata_hvg_s.to_memory()
        adata_hvg_s.X = adata_hvg_s.X.toarray()

    if issparse(adata_hvg_g.X):
        # 如果是稀疏矩阵，转换为密集矩阵
        adata_hvg_g.X = adata_hvg_g.X.toarray()

    scc_mean_results = pd.DataFrame()
    scc_mean_results = compute_scc_for_each_cell_type(adata_hvg_g, adata_hvg_s, all_types, scc_mean_results, 0)

    scc_cellwise_results = pd.DataFrame()
    scc_cellwise_results = compute_pcc_cellwise_for_each_cell_type(adata_hvg_g, adata_hvg_s, all_types,
                                                                   scc_cellwise_results, 0)

    # scc_mean_results['mean'] = scc_mean_results.mean(axis=1)
    # scc_mean_results['std'] = scc_mean_results.std(axis=1)
    #
    # scc_cellwise_results['mean'] = scc_cellwise_results.loc[:, [0, 1, 2]].mean(axis=1)
    # scc_cellwise_results['std'] = scc_cellwise_results.loc[:, [0, 1, 2]].std(axis=1)

    if save_path_tag is not None:
        save_path = f'../output/{save_path_tag}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        scc_mean_results.round(4).to_csv(f'{save_path}{test_tag}_scc_mean_results.csv')
        scc_cellwise_results.round(4).to_csv(f'{save_path}{test_tag}_scc_cellwise_results.csv')

        draw_umap_with_generate_smart(adata_hvg_g, adata_hvg_s, draw_latent=False,
                                      save_path=f'{save_path}{test_tag}_umap')
    else:
        draw_umap_with_generate_smart(adata_hvg_g, adata_hvg_s, draw_latent=False,
                                      save_path=None)
        return scc_mean_results, scc_cellwise_results


def calculate_results_for_test_data(adata_test, adata_smartseq, test_tag, gen_cell_types, args, top_indices=None, draw_umap=False,
                                    save_path_tag=None, is_multi_type=False, num_sample=100, vision=None, calculate_cellwise=False):
    gen_tissue_cell_types = [test_tag.split('_')[0] + '_' + cell_type for cell_type in gen_cell_types]
    adata_10x_gen = adata_test[adata_test.obs.tissue_celltype.isin(gen_tissue_cell_types)]
    adata_smartseq_gen = adata_smartseq[adata_smartseq.obs.tissue_celltype.isin(gen_tissue_cell_types)]

    adata_test_source, adata_smartseq_gen, adata_test_gen = generate_vision(adata_10x_gen, adata_smartseq_gen,
                                                                              args,
                                                                              vision=vision,
                                                                              is_multi_type=is_multi_type)

    if top_indices is None:
        adata_hvg_s = adata_smartseq_gen
        adata_hvg_g = adata_test_gen
    else:
        adata_hvg_s = adata_smartseq_gen[:, top_indices]
        adata_hvg_g = adata_test_gen[:, top_indices]

    # all_types = ['B', 'CD4Tconv', 'CD8T', 'CD8Tex', 'Mast', 'Mono/Macro', 'NK', 'Plasma']
    all_types = gen_cell_types

    adata_g_sample1 = random_sample_every_cell_type(adata_hvg_g, all_types, num_sample, seed=2486)
    adata_g_sample2 = random_sample_every_cell_type(adata_hvg_g, all_types, num_sample, seed=2442)
    adata_g_sample3 = random_sample_every_cell_type(adata_hvg_g, all_types, num_sample, seed=3407)

    adata_smartseq_sample = random_sample_every_cell_type(adata_hvg_s, all_types, num_sample, seed=2486)
    if issparse(adata_smartseq_sample.X):
        # 如果是稀疏矩阵，转换为密集矩阵
        adata_smartseq_sample.X = adata_smartseq_sample.X.toarray()

    g_list = [adata_g_sample1, adata_g_sample2, adata_g_sample3]

    scc_mean_results = pd.DataFrame()
    for i, g in enumerate(g_list):
        scc_mean_results = compute_scc_for_each_cell_type(g, adata_smartseq_sample, all_types, scc_mean_results, i)

    scc_cellwise_results = pd.DataFrame()
    if calculate_cellwise:
        for i, g in enumerate(g_list):
            scc_cellwise_results = compute_pcc_cellwise_for_each_cell_type(g, adata_smartseq_sample, all_types,
                                                                           scc_cellwise_results, i)
        scc_cellwise_results['mean'] = scc_cellwise_results.loc[:, [0, 1, 2]].mean(axis=1)
        scc_cellwise_results['std'] = scc_cellwise_results.loc[:, [0, 1, 2]].std(axis=1)

    scc_mean_results['mean'] = scc_mean_results.mean(axis=1)
    scc_mean_results['std'] = scc_mean_results.std(axis=1)

    if save_path_tag is not None:
        save_path = f'../output/{save_path_tag}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        scc_mean_results.round(4).to_csv(f'{save_path}{test_tag}_scc_mean_results.csv')
        scc_cellwise_results.round(4).to_csv(f'{save_path}{test_tag}_scc_cellwise_results.csv')

        if draw_umap:
            draw_umap_with_generate_smart(adata_g_sample2, adata_smartseq_sample, draw_latent=False,
                                          save_path=f'{save_path}{test_tag}_umap')
    else:
        if draw_umap:
            draw_umap_with_generate_smart(adata_g_sample2, adata_smartseq_sample, draw_latent=False,
                                          save_path=None)
        return scc_mean_results, scc_cellwise_results


def calculate_results_for_all_test_data(ae_10x_tag, ae_smartseq_tag, g_tag, args=None, is_multi_type=False):
    test_tags = ['CRC_GSE136394', 'CRC_GSE139555', 'LIHC_GSE125449_aPDL1aCTLA4', 'LIHC_GSE179795',
                 'SKCM_GSE159251', 'SKCM_GSE179373']
    cell_types = [['CD8T', 'Mono/Macro'],
                  ['CD8T', 'CD8Tex', 'Mono/Macro', 'NK', 'Plasma'],
                  ['CD8Tex', 'Plasma'],
                  ['NK'],
                  ['CD8Tex', 'CD4Tconv'],
                  ['CD8Tex']]

    if args is None:
        args = {'loss_ae': 'mse',
                'decoder_activation': 'ReLU',
                'local_rank': 0,
                'seed': None,
                'hparams': '',
                'epochs': 50,
                'batch_size': 128,

                'state_dict_t': f'../output/{ae_10x_tag}/model_seed=None_step=99.pt',
                'state_dict_s': f'../output/{ae_smartseq_tag}/model_seed=None_step=99.pt',
                'state_dict_g': f'../output/{g_tag}/model_seed=None_step=199.pt',
                'label_encoder': f'../output/{g_tag}/label_encoder.pkl',
                'data_dir_s': '../data/train_data/training_smartseq2_immune_common.h5ad',
                'device': 'cuda'
                }
    save_path_tag = f'{g_tag}/{ae_10x_tag}_{ae_smartseq_tag}'

    adata_smartseq = sc.read_h5ad(args['data_dir_s'])

    top_2000_indices = get_top_highly_variable_genes(adata_smartseq, top=2000)

    for test_tag, cell_type in zip(test_tags, cell_types):
        calculate_results_for_test_data(test_tag, cell_type, adata_smartseq, args, top_2000_indices, save_path_tag,
                                        is_multi_type=is_multi_type)


def get_top_highly_variable_genes(adata, top=2000):
    # 识别高变基因
    sc.pp.highly_variable_genes(adata)
    adata_hvg_s = adata[:, adata.var['highly_variable']]

    # 计算每个基因的缺失率
    missing_rate = np.sum(adata_hvg_s.X == 0, axis=0) / adata_hvg_s.n_obs
    if hasattr(missing_rate, 'A1'):  # 处理稀疏矩阵结果
        missing_rate = missing_rate.A1

    # 获取缺失率最低的 2000 个基因的索引
    top_indices = np.argsort(missing_rate)[:top]

    return top_indices


def get_top_lowly_loses_genes(adata, top=2000, min_cells=100):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 计算每个基因的缺失率
    missing_rate = np.sum(adata.X == 0, axis=0) / adata.n_obs
    if hasattr(missing_rate, 'A1'):  # 处理稀疏矩阵结果
        missing_rate = missing_rate.A1

    # 获取缺失率最低的 2000 个基因的索引
    top_indices = np.argsort(missing_rate)[:top]

    return top_indices


def filter_cell_types_by_count(adata, cell_type_column='celltype', min_count=500):
    """
    过滤掉细胞数量少于指定阈值的细胞类型

    参数:
    adata: scanpy的AnnData对象
    cell_type_column: 存储细胞类型信息的列名，默认为'celltype'
    min_count: 最小细胞数量阈值，默认为500

    返回:
    过滤后的AnnData对象
    """
    # 统计每种细胞类型的细胞数量
    cell_type_counts = adata.obs[cell_type_column].value_counts()

    # 筛选出细胞数量大于等于min_count的细胞类型
    filtered_cell_types = cell_type_counts[cell_type_counts >= min_count].index.tolist()

    # 打印过滤前后的信息
    print(f"过滤前的细胞类型数量: {len(cell_type_counts)}")
    print(f"过滤后的细胞类型数量: {len(filtered_cell_types)}")
    print(f"被过滤掉的细胞类型: {', '.join(cell_type_counts[cell_type_counts < min_count].index.tolist())}")

    # 根据筛选出的细胞类型来子集化adata对象
    filtered_adata = adata[adata.obs[cell_type_column].isin(filtered_cell_types)].copy()

    return filtered_adata


def get_adata_g(tag, model_tag, epochs_g=100, device='cuda', return_model=False):
    import sys
    if sys.path[0] != "../":
        sys.path.insert(0, "../")
    import analysis_utils.utils_qc as utils_qc
    from train.generate_map_single_v2 import generate_map, generate_10x_cycle
    from train.train_map_single_base_mix import seed_everything

    seed = 42
    seed_everything(seed)

    tag_t = tag
    tag_s = tag_t

    sava_tag = model_tag

    save_dir_root = f'output_single/{sava_tag}/'

    # 当使用非配对数据集时，需要修改下面的参数
    data_t = f'data/single_train/{tag_t}_10X_10x_immune.h5ad'
    data_s = f'data/single_train/{tag_s}_Smartseq2_smartseq_immune.h5ad'

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
                'is_celltype': True,
                }

    args_map['state_dict_g'] = args_map['save_dir'] + f'model_seed={seed}_step={epochs_g - 1}.pt'

    seed = args_ed["seed"]

    # 获取两个数据集中的共同类型的细胞
    adata_10x = sc.read_h5ad(data_t)
    adata_smartseq = sc.read_h5ad(data_s)

    type_smart = adata_smartseq.obs['celltype'].value_counts()
    type_10x = adata_10x.obs['celltype'].value_counts()

    cross_type = type_smart.index.intersection(type_10x.index)
    diff_type = type_smart.index.difference(type_10x.index)

    adata_smartseq_common = adata_smartseq[adata_smartseq.obs['celltype'].isin(cross_type)]
    adata_10x_common = adata_10x[adata_10x.obs['celltype'].isin(cross_type)]

    adata_10x = adata_10x_common.copy()
    adata_smartseq = adata_smartseq_common.copy()

    if return_model:
        adata_10x, adata_smartseq, adata_g, generator = generate_map(adata_10x, adata_smartseq, args_map)
        return adata_g, generator
    else:
        adata_10x, adata_smartseq, adata_g = generate_map(adata_10x, adata_smartseq, args_map)
        return adata_g

def get_adata_g_pert(tag, model_tag, epochs_g=100, tag_s=None, is_test=False, device='cuda'):
    import sys
    if sys.path[0] != "../":
        sys.path.insert(0, "../")
    import analysis_utils.utils_qc as utils_qc
    from train.generate_map_single_v2 import generate_map, generate_10x_cycle
    from train.train_map_single_base_mix import seed_everything

    # utils.get_adata_g_for_pert('replogle_hepg2_cont', 'replogle_hepg2_cdc20_251030', 200, tag_s='replogle_hepg2_cdc20', is_test=False)

    seed = 42
    seed_everything(seed)

    tag_t = tag
    if tag_s is None:
        tag_s = tag_t

    sava_tag = model_tag

    save_dir_root = f'../output_single/{sava_tag}/'

    # 当使用非配对数据集时，需要修改下面的参数
    if is_test:
        data_t = f"../data/pert_test/{tag_t}_test.h5ad"
        data_s = f"../data/pert_test/{tag_s}_test.h5ad"
    else:
        data_t = f"../data/pert_train/{tag_t}_train.h5ad"
        data_s = f"../data/pert_train/{tag_s}_train.h5ad"

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
                'is_celltype': True,
                }

    args_map['state_dict_g'] = args_map['save_dir'] + f'model_seed={seed}_step={epochs_g - 1}.pt'

    seed = args_ed["seed"]

    # 获取两个数据集中的共同类型的细胞
    adata_10x = sc.read_h5ad(data_t)
    adata_smartseq = sc.read_h5ad(data_s)

    type_smart = adata_smartseq.obs['celltype'].value_counts()
    type_10x = adata_10x.obs['celltype'].value_counts()

    cross_type = type_smart.index.intersection(type_10x.index)
    diff_type = type_smart.index.difference(type_10x.index)

    adata_smartseq_common = adata_smartseq[adata_smartseq.obs['celltype'].isin(cross_type)]
    adata_10x_common = adata_10x[adata_10x.obs['celltype'].isin(cross_type)]

    adata_10x = adata_10x_common.copy()
    adata_smartseq = adata_smartseq_common.copy()

    adata_10x, adata_smartseq, adata_g = generate_map(adata_10x, adata_smartseq, args_map)

    return adata_g

def generate_vision(adata_10x_gen, adata_smartseq_gen, args, vision=None, is_multi_type=False):
    if vision == 'v2':
        from train.generate_map_v2 import generate_map as generate_map_v2
        adata_test_source, adata_smartseq_gen, adata_test_gen = generate_map_v2(adata_10x_gen, adata_smartseq_gen, args,
                                                                               is_multi_type=is_multi_type)
    elif vision == 'single':
        from train.generate_map_single import generate_map as generate_map_single
        adata_test_source, adata_smartseq_gen, adata_test_gen = generate_map_single(adata_10x_gen, adata_smartseq_gen, args,)
    elif vision == 'single_vae':
        from train.generate_map_single_vae import generate_map as generate_map_single_vae
        adata_test_source, adata_smartseq_gen, adata_test_gen = generate_map_single_vae(adata_10x_gen, adata_smartseq_gen, args,)
    else:
        from train_history.generate_map import generate_map
        adata_test_source, adata_smartseq_gen, adata_test_gen = generate_map(adata_10x_gen, adata_smartseq_gen, args,
                                                                               is_multi_type=is_multi_type)
    return adata_test_source, adata_smartseq_gen, adata_test_gen

if __name__ == '__main__':
    # import pandas as pd
    #
    # adata = sc.read_h5ad('../data/LIHC_GSE140228_Smartseq2_exm.h5ad')
    # print(adata.obs['celltype'].value_counts())
    #
    # cell_types = ['NK', 'CD8Tex']
    #
    # adata_filtered = random_sample_from_adata(adata, 100, cell_type=cell_types, seed=1234)
    #
    # print(adata_filtered)
    # print(adata_filtered.obs['celltype'].value_counts())

    # debug 250427
    tag = 'CRC_GSE139555'
    cell_type = ['CD8T', 'CD8Tex', 'Mono/Macro', 'NK', 'Plasma']

    ae_10x_tag = '10X_immune_common'
    ae_smartseq_tag = 'Smartseq2_immune_common'

    args = {'loss_ae': 'mse',
            'decoder_activation': 'ReLU',
            'local_rank': 0,
            'seed': None,
            'hparams': '',
            'epochs': 50,
            'batch_size': 128,

            'state_dict_t': f'../output/{ae_10x_tag}/model_seed=None_step=99.pt',
            'state_dict_s': f'../output/{ae_smartseq_tag}/model_seed=None_step=99.pt',
            'device': 'cuda'
            }

    g_tag = 'immune_common'

    args['state_dict_g'] = f'../output/{g_tag}/model_seed=None_step=199.pt'
    args['label_encoder'] = f'../output/{g_tag}/label_encoder.pkl'

    adata_smartseq = sc.read_h5ad('../data/train_data/training_smartseq2_immune_common.h5ad')
    top_2000_indices = get_top_highly_variable_genes(adata_smartseq, top=2000)

    scc_mean_results, scc_cellwise_results = calculate_results_for_test_data_all_sample(tag, cell_type,
                                                                                        adata_smartseq, args,
                                                                                        top_2000_indices,
                                                                                        is_multi_type=False)

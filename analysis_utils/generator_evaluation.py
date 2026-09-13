import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

if sys.path[0] != "../":
    sys.path.insert(0, "../")
import analysis_utils.utils as utils


def calculate_content_results_for_test_data(adata_test, adata_smartseq, test_tag, gen_cell_types, args, top_indices=None,
                                            save_path_tag=None, is_multi_type=False, num_sample=100, vision=None):
    """
        对真实10x和对应的生成smartseq按配对计算pcc
        :param test_tag:
        :param gen_cell_types:
        :param adata_smartseq:
        :param args:
        :param top_indices:
        :param save_path_tag:
        :param is_multi_type:
        :param num_sample:
        :return:
    """
    print('check adata_test shape:')
    print(adata_test)


    gen_tissue_cell_types = [test_tag.split('_')[0] + '_' + cell_type for cell_type in gen_cell_types]
    adata_10x_gen = adata_test[adata_test.obs.tissue_celltype.isin(gen_tissue_cell_types)]
    adata_smartseq_gen = adata_smartseq[adata_smartseq.obs.tissue_celltype.isin(gen_tissue_cell_types)]

    adata_test_source, adata_smartseq_gen, adata_test_gen = utils.generate_vision(adata_10x_gen, adata_smartseq_gen, args,
                                                                                  vision=vision, is_multi_type=is_multi_type)

    if top_indices is None:
        adata_hvg_s = adata_smartseq_gen
        adata_hvg_g = adata_test_gen
    else:
        adata_hvg_s = adata_smartseq_gen[:, top_indices]
        adata_hvg_g = adata_test_gen[:, top_indices]

    # all_types = ['B', 'CD4Tconv', 'CD8T', 'CD8Tex', 'Mast', 'Mono/Macro', 'NK', 'Plasma']
    all_types = gen_cell_types

    adata_g_sample1 = utils.random_sample_every_cell_type(adata_hvg_g, all_types, num_sample, seed=2486)
    adata_g_sample2 = utils.random_sample_every_cell_type(adata_hvg_g, all_types, num_sample, seed=2442)
    adata_g_sample3 = utils.random_sample_every_cell_type(adata_hvg_g, all_types, num_sample, seed=3407)

    if issparse(adata_test_source.X):
        adata_test_source = adata_test_source.to_memory()
        adata_test_source.X = adata_test_source.X.toarray()

    adata_10x_sample1 = adata_test_source[adata_g_sample1.obs_names]
    adata_10x_sample2 = adata_test_source[adata_g_sample2.obs_names]
    adata_10x_sample3 = adata_test_source[adata_g_sample3.obs_names]

    tenx_list = [adata_10x_sample1, adata_10x_sample2, adata_10x_sample3]

    adata_smartseq_sample = utils.random_sample_every_cell_type(adata_hvg_s, all_types, num_sample, seed=2486)

    if issparse(adata_smartseq_sample.X):
        # 如果是稀疏矩阵，转换为密集矩阵
        adata_smartseq_sample.X = adata_smartseq_sample.X.toarray()

    g_list = [adata_g_sample1, adata_g_sample2, adata_g_sample3]

    scc_mean_results = pd.DataFrame()
    for i, g in enumerate(g_list):
        shared_genes = tenx_list[i].var_names.intersection(g.var_names)
        print('shared_genes:', len(shared_genes))
        print('g.var_names:', g.var_names.shape)
        print('tenx_list[i].var_names:', tenx_list[i].var_names.shape)
        adata_real = tenx_list[i][:, shared_genes].copy()
        adata_gen = g[:, shared_genes].copy()
        scc_mean_results = utils.compute_scc_for_each_cell_type(adata_gen, adata_real, all_types,
                                                                scc_mean_results, i)

    pcc_content_results = pd.DataFrame()
    for i, g in enumerate(g_list):
        pcc_content_results = compute_content_pcc_for_each_cell_type(g, tenx_list[i], all_types,
                                                                     pcc_content_results, i)

    scc_mean_results['mean'] = scc_mean_results.mean(axis=1)
    scc_mean_results['std'] = scc_mean_results.std(axis=1)

    pcc_content_results['mean'] = pcc_content_results.loc[:, [0, 1, 2]].mean(axis=1)
    pcc_content_results['std'] = pcc_content_results.loc[:, [0, 1, 2]].std(axis=1)

    return scc_mean_results, pcc_content_results


def compute_content_pcc_for_each_cell_type(adata1, adata2, cell_types, results_df, results_column):
    print('calculate cellwise pcc')
    for type_ in cell_types:
        adata1_type = adata1[adata1.obs.celltype == type_].copy()
        adata2_type = adata2[adata2.obs.celltype == type_].copy()
        scc, sigma, _ = calculate_content_pcc(adata1_type, adata2_type)
        print(type_, scc, sigma)
        results_df.loc[type_, results_column] = scc
        results_df.loc[type_, results_column + 3] = sigma
    return results_df


def calculate_content_pcc(adata_gen, adata_source, norm=False):
    # 1. 数据预处理
    # --------------------------------------------------
    if norm:
        # 总计数归一化至1e4（CPM）
        sc.pp.normalize_total(adata_source, target_sum=1e4)
        sc.pp.normalize_total(adata_gen, target_sum=1e4)

        # 对数转换（log1p）
        sc.pp.log1p(adata_source)
        sc.pp.log1p(adata_gen)

    # 基因对齐（按真实数据顺序）
    shared_genes = adata_source.var_names.intersection(adata_gen.var_names)
    print('shared_genes for calculate_content_pcc:', len(shared_genes))
    adata_source = adata_source[:, shared_genes].copy()
    adata_gen = adata_gen[:, shared_genes].copy()
    assert np.all(adata_source.var_names == adata_gen.var_names), "基因名称或顺序不一致！"

    # 转换为稠密数组（若内存允许）
    X_source = adata_source.X.toarray() if hasattr(adata_source.X, 'toarray') else adata_source.X
    X_gen = adata_gen.X.toarray() if hasattr(adata_gen.X, 'toarray') else adata_gen.X

    n_gen = X_gen.shape[0]
    n_source = X_source.shape[0]

    assert n_gen == n_source

    corr_list = np.zeros(n_source)
    # scaled_array = np.arange(len(X_gen[0])) * 0.05
    # scaled_array = scaled_array.reshape(X_gen[0].shape)
    print('attention the ConstantInputWarning !!!')

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        print(f"警告发生在第 {i} 次迭代: {message}")
        # 这里可以添加你想要执行的其他语句
        print(f'X_gen[i]: max {X_gen[i].max()}, min {X_gen[i].min()}, mean {X_gen[i].mean()}, std {X_gen[i].std()}')
        print(f'X_source[i]: max {X_source[i].max()}, min {X_source[i].min()}, mean {X_source[i].mean()}, std {X_source[i].std()}')


    for i in range(n_source):
        # print('\ncheck content pcc:')
        # print(X_gen[i][:20], X_source[i][:20])
        # corr_list[i], _ = pearsonr(X_gen[i], X_source[i])
        # corr_list[i], _ = pearsonr(X_gen[i], scaled_array)
        with warnings.catch_warnings():
            warnings.showwarning = warning_handler
            warnings.filterwarnings("always", category=RuntimeWarning)

            corr_list[i], _ = pearsonr(X_gen[i], X_source[i])
            # corr_list[i], _ = spearmanr(X_gen[i], X_source[i])

    # 3. 计算全局统计量
    # --------------------------------------------------
    # global_scc = np.nanmean(corr_matrix)  # 忽略NaN
    # global_std = np.nanstd(corr_matrix)
    global_pcc = np.nanmean(corr_list)
    global_std = np.nanstd(corr_list)

    return global_pcc, global_std, corr_list


def evaluation(adata_smartseq, generator_list, test_dataset_info, testdata_root_path, testdata_suffix, model_root_path, args, top_2000_indices=None,
               num_sample=100, vision=None, calculate_pcc_smartseq=False):
    results_scc_mean = pd.DataFrame()
    results_pcc_content = pd.DataFrame()
    results_pcc_smartseq = pd.DataFrame()
    seed = args['seed']
    for g_tag in generator_list:
        print(f'\nevaluating {g_tag}\n')
        args['state_dict_g'] = f'{model_root_path}{g_tag}/model_seed={seed}_step=199.pt'
        args['label_encoder'] = f'{model_root_path}{g_tag}/label_encoder.pkl'
        print('loading pretrained model from: \n', args["state_dict_g"])
        for tag in test_dataset_info:
            print(f'evaluating {tag}')
            adata_test = sc.read_h5ad(f'{testdata_root_path}{tag}{testdata_suffix}')
            cell_type = test_dataset_info[tag]
            scc_mean_content_results, pcc_content_results = calculate_content_results_for_test_data(adata_test,adata_smartseq,tag, cell_type,
                                                                                                    args,
                                                                                                    top_indices=top_2000_indices,
                                                                                                    save_path_tag=None,
                                                                                                    is_multi_type=False,
                                                                                                    num_sample=num_sample,
                                                                                                    vision=vision)
            results_pcc_content.loc[g_tag, tag] = pcc_content_results['mean'].mean()

            scc_mean_results, scc_cellwise_results = utils.calculate_results_for_test_data(adata_test, adata_smartseq,
                                                                                           tag, cell_type, args,
                                                                                           top_indices=top_2000_indices,
                                                                                           save_path_tag=None,
                                                                                           is_multi_type=False,
                                                                                           num_sample=num_sample,
                                                                                           vision=vision,
                                                                                           calculate_cellwise=calculate_pcc_smartseq)

            results_scc_mean.loc[g_tag, tag] = scc_mean_results['mean'].mean()
            if calculate_pcc_smartseq:
                results_pcc_smartseq.loc[g_tag, tag] = scc_cellwise_results['mean'].mean()
    return results_pcc_content, results_scc_mean, results_pcc_smartseq


def evaluation_v2(adata_smartseq, adata_10x, adata_gen, all_types, num_sample=100, i=0):
    scc_mean_results = pd.DataFrame()
    pcc_content_results = pd.DataFrame()

    scc_mean_results = utils.compute_scc_for_each_cell_type(adata_gen, adata_smartseq, all_types,
                                                            scc_mean_results, i)


    pcc_content_results = compute_content_pcc_for_each_cell_type(adata_gen, adata_10x, all_types,
                                                                     pcc_content_results, i)

    return scc_mean_results, pcc_content_results


if __name__ == '__main__':
    # generator_list = ['immune_common', 'immune_common_v2d1_minibatch', 'immune_common_v2d1_bigbatch',
    #                   'immune_common_onlystyloss_v2d1_minibatch', 'immune_common_onlystyloss_v2d1_bigbatch',
    #                   'immune_common_onlycontloss_v2d1_minibatch', 'immune_common_onlycontloss_v2d1_bigbatch',
    #                   'immune_common_hvgt_v2d1_minibatch', 'immune_common_hvgt_v2d1_bigbatch',
    #                   'immune_common_hvgt_onlystyloss_v2d1_minibatch', 'immune_common_hvgt_onlystyloss_v2d1_bigbatch',
    #                   'immune_common_hvgt_onlycontloss_v2d1_minibatch', 'immune_common_hvgt_onlycontloss_v2d1_bigbatch',
    #                   ]

    # test_dataset_info = {
    #     'CRC_GSE139555': ['CD8T', 'CD8Tex', 'Mono/Macro', 'NK', 'Plasma'],
    #     'LIHC_GSE125449_aPDL1aCTLA4': ['CD8Tex', 'Plasma'],
    #     'SKCM_GSE159251': ['CD8Tex', 'CD4Tconv'],
    # }

    # test_dataset_info = {
    #     'LIHC_GSE140228_10X': ['CD4Tconv', 'Mono/Macro', 'NK', 'CD8T', 'CD8Tex', 'B', 'Treg'],
    #     'CRC_GSE146771_10X': ['Mono/Macro', 'NK', 'CD8T', 'Plasma', 'CD8Tex'],
    #     'CRC_GSE139555': ['CD8T', 'CD8Tex', 'Mono/Macro', 'NK', 'Plasma'],
    #     'LIHC_GSE125449_aPDL1aCTLA4': ['CD8Tex', 'Plasma'],
    #     'SKCM_GSE159251': ['CD8Tex', 'CD4Tconv'],
    # }
    from train_history.train_map_single import *

    device = 'cuda:0'  # todo: check the device
    seed = 2486
    seed_everything(seed)

    args = {'loss_ae': 'mse',
            'decoder_activation': 'ReLU',
            'local_rank': 0,
            'seed': seed,
            'hparams': '',
            'epochs': 100,
            'batch_size': 128,

            'state_dict': '../pretrain/annotation_model_v1',
            # 'save_dir': '../output/test/',
            # 'data_dir': '../data/LIHC_GSE140228_10X_exm.h5ad',
            'device': device
            }

    data_t = '../data/single_train/CRC_GSE146771_10X_10x_immune.h5ad'
    data_s = '../data/single_train/CRC_GSE146771_Smartseq2_smartseq_immune.h5ad'

    args['data_dir'] = data_t
    args['save_dir'] = '../output_single/10x_CRC_GSE146771_10X/'

    generator_list = ['immune_10x_CRC_GSE146771_10X_to_CRC_GSE146771_Smartseq2']
    test_dataset_info = {'CRC_GSE146771_10X_10x': ['NK', 'CD4Tconv', 'Mono/Macro', 'CD8Tex', 'Plasma', 'CD8T']}
    model_root_path = '../output_single/'
    testdata_root_path = '../data/single_train/'
    testdata_suffix = '_immune.h5ad'

    ae_10x_tag = '10x_CRC_GSE146771_10X'
    ae_smartseq_tag = 'smartseq_CRC_GSE146771_Smartseq2'

    args = {'loss_ae': 'mse',
            'decoder_activation': 'ReLU',
            'local_rank': 0,
            'seed': seed,
            'hparams': '',
            'epochs': 50,
            'batch_size': 128,

            'state_dict_t': f'{model_root_path}{ae_10x_tag}/model_seed={seed}_step=99.pt',
            'state_dict_s': f'{model_root_path}{ae_smartseq_tag}/model_seed={seed}_step=99.pt',
            'device': 'cuda'
            }

    if torch.cuda.is_available() is False:
        args["device"] = "cpu"

    adata_smartseq = sc.read_h5ad(data_s)

    top_2000_indices = utils.get_top_highly_variable_genes(adata_smartseq, top=2000)

    results_pcc_content, results_scc_mean, results_pcc_smartseq = evaluation(adata_smartseq, generator_list,
                                                                             test_dataset_info, testdata_root_path,
                                                                             testdata_suffix, model_root_path, args,
                                                                             top_2000_indices=None, num_sample=100,
                                                                             vision='single')

    print('\npcc_content')
    print(results_pcc_content)

    print('\nscc_mean')
    print(results_scc_mean)

    print('\npcc_smartseq')
    print(results_pcc_smartseq)
import sys
import pandas as pd

if sys.path[0] != "../":
    sys.path.insert(0, "../")
from train.train_map_single_base_mix import *
import analysis_utils.utils as utils
import analysis_utils.generator_evaluation as ge


def evaluation_v2(adata_smartseq, adata_10x, adata_gen, all_types, i=0):
    scc_mean_results = pd.DataFrame()
    pcc_content_results = pd.DataFrame()

    if issparse(adata_smartseq.X):
        # 如果是稀疏矩阵，转换为密集矩阵
        adata_smartseq.X = adata_smartseq.X.toarray()

    if issparse(adata_10x.X):
        adata_10x.X = adata_10x.X.toarray()

    scc_mean_results = utils.compute_scc_for_each_cell_type(adata_gen, adata_smartseq, all_types,
                                                            scc_mean_results, i)

    pcc_content_results = ge.compute_content_pcc_for_each_cell_type(adata_gen, adata_10x, all_types,
                                                                    pcc_content_results, i)

    return scc_mean_results, pcc_content_results


if __name__ == "__main__":
    import numpy as np
    import scanpy as sc
    from train.generate_map_single_v2 import generate_map

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # todo: check the device
    seed = 42
    seed_everything(seed)
    tag_t = 'LIHC_GSE140228'
    tag_s = tag_t

    is_mix = True

    model_tag = 'LIHC_GSE140228_260105'

    epochs_g = 100

    model_dir_root = f'../output_single/{model_tag}/'
    num_sample = 200

    # 当使用非配对数据集时，需要修改下面的参数
    data_t = f'../data/single_train/{tag_t}_10X_10x_immune.h5ad'
    data_s = f'../data/single_train/{tag_s}_Smartseq2_smartseq_immune.h5ad'

    save_dir_t = f'{model_dir_root}10x_{tag_t}/'
    save_dir_s = f'{model_dir_root}smartseq_{tag_s}/'
    save_dir_g = f'{model_dir_root}{tag_t}_to_{tag_s}/'
    # save_dir_g += 'epochs_t_100_epochs_s_100_batch_size_128_beta_0_add_noise_dbinary/epochs_100_batch_size_32_lr_0p0001_cyc_coef_0p5_cyc_g_coef_0_sty_coef_5_cont_coef_5_ae_coef_0/'


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
    if is_mix:
        state_dict_t = save_dir_g + f'ae_10x_seed={seed}_step={epochs_g - 1}.pt'
        state_dict_s = save_dir_g + f'ae_smartseq_seed={seed}_step={epochs_g - 1}.pt'
    else:
        state_dict_t = save_dir_t + f'model_seed={seed}_step={epochs_t - 1}.pt'
        state_dict_s = save_dir_s + f'model_seed={seed}_step={epochs_s - 1}.pt'

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

    epochs_g = args_map['epochs']
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

    all_types = adata_g.obs['celltype'].unique()
    adata_g_sample1 = utils.random_sample_every_cell_type(adata_g, all_types, num_sample, seed=2486)
    adata_g_sample2 = utils.random_sample_every_cell_type(adata_g, all_types, num_sample, seed=2442)
    adata_g_sample3 = utils.random_sample_every_cell_type(adata_g, all_types, num_sample, seed=3407)

    adata_10x_sample1 = adata_10x[adata_g_sample1.obs_names].copy()
    adata_10x_sample2 = adata_10x[adata_g_sample2.obs_names].copy()
    adata_10x_sample3 = adata_10x[adata_g_sample3.obs_names].copy()

    adata_smartseq_sample = utils.random_sample_every_cell_type(adata_smartseq, all_types, num_sample, seed=2486)

    pcc_content_results = pd.DataFrame()
    i = 0
    pcc_content_results_s = ge.compute_content_pcc_for_each_cell_type(adata_smartseq_sample, adata_10x_sample2,
                                                                      all_types, pcc_content_results, i)
    scc_mean_results, pcc_content_results = evaluation_v2(adata_smartseq_sample, adata_10x_sample2, adata_g_sample2,
                                                          all_types)

    print('\nscc_mean_results', scc_mean_results)
    print('\npcc_content_results', pcc_content_results)
    print('\npcc_content_results_s', pcc_content_results_s)

    print('\nscc_mean_results describe', scc_mean_results.describe())
    print('\npcc_content_results describe', pcc_content_results.describe())
    print('\npcc_content_results_s describe', pcc_content_results_s.describe())

    # scc_mean_results.round(4).to_csv('../scc_mean_250918.csv')
    # pcc_content_results.round(4).to_csv('../pcc_content_250918.csv')

import sys
import pandas as pd

if sys.path[0] != "../":
    sys.path.insert(0, "../")
from train.train_map_single_base_mix import *
import analysis_utils.utils as utils
import analysis_utils.generator_evaluation as ge
from analysis_utils.batch_correct_evaluation import evaluate_batch_correction, plot_umap_batch_celltype


if __name__ == "__main__":
    from train.generate_map_single_v2 import generate_map

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # todo: check the device
    seed = 42
    seed_everything(seed)

    # fac = 1
    #
    # tag_t = f'sim_batchfac{fac}'
    # tag_t = f'CRC_GSE146771'
    # tag_t = 'GSE75748_ct_04n2'
    tag_t = f'LIHC_GSE140228'
    tag_s = tag_t

    # results = pd.DataFrame()

    is_mix = True

    model_tag = f'{tag_t}_with_rebuild'

    epochs_g = 100

    model_dir_root = f'../output_single/{model_tag}/'

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
                'is_celltype': True
                }

    epochs_g = args_map['epochs']
    args_map['state_dict_g'] = args_map['save_dir'] + f'model_seed={seed}_step={epochs_g - 1}.pt'

    seed = args_ed["seed"]

    # 获取两个数据集中的共同类型的细胞
    adata_10x = sc.read_h5ad(data_t)
    adata_smartseq = sc.read_h5ad(data_s)

    # 过滤掉细胞数量少于500的细胞类型
    adata_10x = utils.filter_cell_types_by_count(adata_10x, min_count=500)
    adata_smartseq = utils.filter_cell_types_by_count(adata_smartseq, min_count=500)

    type_smart = adata_smartseq.obs['celltype'].value_counts()
    type_10x = adata_10x.obs['celltype'].value_counts()

    cross_type = type_smart.index.intersection(type_10x.index)
    diff_type = type_smart.index.difference(type_10x.index)

    adata_smartseq_common = adata_smartseq[adata_smartseq.obs['celltype'].isin(cross_type)]
    adata_10x_common = adata_10x[adata_10x.obs['celltype'].isin(cross_type)]

    adata_10x = adata_10x_common.copy()
    adata_smartseq = adata_smartseq_common.copy()
    adata_10x, adata_smartseq, adata_g = generate_map(adata_10x, adata_smartseq, args_map, recon_tgt=True)

    metrics = evaluate_batch_correction(adata_g, adata_smartseq, k=30, use_gex=False)
    print(metrics)

    results = pd.DataFrame(metrics, index=[model_tag], columns=list(metrics.keys())).transpose()
    results.to_csv(f'{model_tag}_results.csv')

    # plot_umap_batch_celltype(
    #     adata_src=adata_g,
    #     adata_tgt=adata_smartseq,
    #     latent_key='latent',
    #     celltype_key='celltype',
    #     save='umap_comparison.png',
    #     use_reduction=True,
    # )

    all_types = adata_g.obs['celltype'].unique()
    num_sample = 500

    adata_g_sample1 = utils.random_sample_every_cell_type(adata_g, all_types, num_sample, seed=2486)
    adata_g_sample2 = utils.random_sample_every_cell_type(adata_g, all_types, num_sample, seed=2442)
    adata_g_sample3 = utils.random_sample_every_cell_type(adata_g, all_types, num_sample, seed=3407)

    adata_10x_sample1 = adata_10x[adata_g_sample1.obs_names].copy()
    adata_10x_sample2 = adata_10x[adata_g_sample2.obs_names].copy()
    adata_10x_sample3 = adata_10x[adata_g_sample3.obs_names].copy()

    adata_smartseq_sample = utils.random_sample_every_cell_type(adata_smartseq, all_types, num_sample, seed=2486)

    # save_path = f'{model_tag}_batch_correct'
    save_path = f'{model_tag}_batch_correct.png'
    # save_path = None
    utils.draw_umap_with_generate_smart(adata_g_sample2, adata_smartseq_sample, draw_latent=True, save_path=save_path)
    # utils.draw_umap_with_generate_smart(adata_g_sample2, adata_smartseq_sample, draw_latent=True, save_path=save_path)
    # utils.draw_umap_with_generate_smart(adata_g_sample3, adata_smartseq_sample, draw_latent=True)

    # plot_umap_batch_celltype(
    #     adata_src=adata_g_sample1,
    #     adata_tgt=adata_smartseq_sample,
    #     latent_key='latent',
    #     celltype_key='celltype',
    #     save='umap_comparison.png',
    #     use_reduction=False,
    # )
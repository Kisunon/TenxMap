import sys
import os
import logging
import scanpy as sc
import torch

if sys.path[0] != "../":
    sys.path.insert(0, "../")
# todo: 选择训练的模型：
# from train.train_map_single_base_v2 import *
from train.train_map_single_base_mix import *
from analysis_utils import utils


def train(args_ed, args_map, continue_train=None):
    print("")

    # 确保必要的目录存在
    for dir_path in [args_ed['save_dir_t'], args_ed['save_dir_s'], args_map['save_dir']]:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 配置日志记录
    log_file = os.path.join(os.path.dirname(args_ed['save_dir_t']), 'training.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 同时将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    # 记录 args_ed 的键值对
    logger.info("args_ed:")
    for key, value in args_ed.items():
        logger.info(f"  {key}: {value}")
    # 记录 args_map 的键值对
    logger.info("args_map:")
    for key, value in args_map.items():
        logger.info(f"  {key}: {value}")

    seed_everything(args_ed["seed"])

    seed = args_ed["seed"]
    epochs_t = args_ed["epochs_t"]
    epochs_s = args_ed["epochs_s"]
    device = args_ed["device"]

    # 获取两个数据集中的共同类型的细胞
    adata_10x = sc.read_h5ad(args_ed['data_t'])
    adata_smartseq = sc.read_h5ad(args_ed['data_s'])

    # 过滤掉细胞数量少于500的细胞类型
    adata_10x = utils.filter_cell_types_by_count(adata_10x, min_count=500)
    adata_smartseq = utils.filter_cell_types_by_count(adata_smartseq, min_count=500)

    type_smart = adata_smartseq.obs['celltype'].value_counts()
    type_10x = adata_10x.obs['celltype'].value_counts()

    cross_type = type_smart.index.intersection(type_10x.index)
    diff_type = type_smart.index.difference(type_10x.index)
    logger.info('cross_type: %s', cross_type)
    logger.info('diff_type: %s', diff_type)

    adata_smartseq_common = adata_smartseq[adata_smartseq.obs['celltype'].isin(cross_type)]
    adata_10x_common = adata_10x[adata_10x.obs['celltype'].isin(cross_type)]

    adata_10x = adata_10x_common.copy()
    adata_smartseq = adata_smartseq_common.copy()

    # 计算state_dict路径
    state_dict_t = None
    state_dict_s = None
    if args_ed['save_dir_t'] is not None:
        state_dict_t = os.path.join(args_ed['save_dir_t'], f'model_seed={seed}_step={epochs_t - 1}.pt')
    if args_ed['save_dir_s'] is not None:
        state_dict_s = os.path.join(args_ed['save_dir_s'], f'model_seed={seed}_step={epochs_s - 1}.pt')

    # 训练自动编码器
    seed_everything(seed)
    args_ed['data_dir'] = args_ed["data_t"]
    args_ed['save_dir'] = args_ed["save_dir_t"]
    args_ed['epochs'] = epochs_t
    add_noise = args_ed["add_noise"]

    if add_noise is not None:
        if add_noise[0] == 'd':
            args_ed["add_noise"] = add_noise[1:]
        else:
            args_ed["add_noise"] = None
    if args_ed['save_dir_t'] is not None:
        if not os.path.exists(state_dict_t):
            train_ed(adata_10x, args_ed, continue_train=None)

    args_ed['data_dir'] = args_ed["data_s"]
    args_ed['save_dir'] = args_ed["save_dir_s"]
    args_ed['epochs'] = epochs_s

    if add_noise is not None:
        if add_noise[0] == 'd':
            args_ed["add_noise"] = add_noise[1:]
        else:
            args_ed["add_noise"] = add_noise
    if args_ed['save_dir_s'] is not None:
        if not os.path.exists(state_dict_s):
            train_ed(adata_smartseq, args_ed, continue_train=None)

    # 更新args_map中的state_dict路径
    args_map['state_dict_t'] = state_dict_t
    args_map['state_dict_s'] = state_dict_s

    # 训练生成器
    seed_everything(seed)
    train_map(adata_10x, adata_smartseq, args_map, use_hvg_10x=False, continue_train=continue_train)


def main(args_ed, args_map):
    """
    主函数，用于从浏览器调用
    """
    seed_everything(args_ed['seed'])
    train(args_ed, args_map)

if __name__ == "__main__":
    # 命令行运行时的默认参数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 42
    seed_everything(seed)

    tag_t = 'CRC_GSE146771'
    tag_s = tag_t

    sava_tag = f'{tag_t}_with_rebuild'

    save_dir_root = f'../output_single/{sava_tag}/'
    if not os.path.exists(save_dir_root):
        os.makedirs(save_dir_root)

    # 当使用非配对数据集时，需要修改下面的参数
    data_t = f'../data/single_train/{tag_t}_10X_10x_immune.h5ad'
    data_s = f'../data/single_train/{tag_s}_Smartseq2_smartseq_immune.h5ad'

    save_dir_t = f'{save_dir_root}10x_{tag_t}/'
    save_dir_s = f'{save_dir_root}smartseq_{tag_s}/'
    save_dir_g = f'{save_dir_root}{tag_t}_to_{tag_s}/'

    args_ed = {
        'loss_ae': 'mse',
        'decoder_activation': 'ReLU',
        'local_rank': 0,
        'seed': seed,
        'hparams': '',
        'epochs_t': 2,
        'epochs_s': 2,
        'batch_size': 128,
        'negative_selection': 'semihard',  # "semihard", "random"
        'beta': 0,
        'add_noise': "dbinary",  # 'gaussian','binary'
        'noise_prob': 0.2,
        'data_t': data_t,
        'data_s': data_s,
        'state_dict': '../pretrain/annotation_model_v1',
        'save_dir_t': save_dir_t,
        'save_dir_s': save_dir_s,
        'device': device
    }

    args_map = {
        'loss_ae': 'mse',
        'decoder_activation': 'ReLU',
        'seed': seed,
        'epochs': 2,
        'batch_size': 128,
        'lr': 2e-4,
        'state_dict': None,
        'save_dir': save_dir_g,
        'data_dir_t': data_t,
        'data_dir_s': data_s,
        'device': device,
        'latent_dim': 128,
        'cyc_coef': 10,
        'cyc_g_coef': 10,
        'sty_coef': 5,
        'cont_coef': 5,
        'ae_coef': 2,
        'is_celltype': True
    }

    main(args_ed, args_map)
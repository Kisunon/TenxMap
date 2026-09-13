import os
import pickle
import sys
import scanpy as sc
import numpy as np
import torch

from scipy.sparse import issparse

if sys.path[0] != "../":
    sys.path.insert(0, "../")
from model.cycle_gan_celltype import CellMapCycleGAN
from model.VAE_model_v2 import VAE


def generate_map(adata_10x, adata_smartseq, args, recon_tgt=False):
    """
    输入处理好的数据
    :param args:
    :return:
    """
    device = args["device"]

    # 获取两个数据集中的共同类型的细胞
    # 过滤掉细胞数量少于500的细胞类型
    from analysis_utils import utils

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

    print('-'*16,'10x','-'*16)
    print(adata_10x.obs['celltype'].value_counts())
    print('-'*16,'smartseq','-'*16)
    print(adata_smartseq.obs['celltype'].value_counts())

    # 编码潜变量
    ae_10x = VAE(
        num_genes=adata_10x.shape[1],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )
    ae_smartseq = VAE(
        num_genes=adata_smartseq.shape[1],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )

    ae_10x.load_state_dict(torch.load(args["state_dict_t"], map_location=device, weights_only=True))
    print('loading pretrained model from: \n', args["state_dict_t"])
    ae_smartseq.load_state_dict(torch.load(args["state_dict_s"], map_location=device, weights_only=True))
    print('loading pretrained model from: \n', args["state_dict_s"])

    ae_10x.eval()
    ae_smartseq.eval()

    if issparse(adata_10x.X):
        data_10x = adata_10x.X.toarray().astype(np.float32)
    else:
        data_10x = adata_10x.X.astype(np.float32)

    if issparse(adata_smartseq.X):
        data_smartseq = adata_smartseq.X.toarray().astype(np.float32)
    else:
        data_smartseq = adata_smartseq.X.astype(np.float32)

    x_10x = ae_10x.encoder(torch.tensor(data_10x, device=device)).detach()
    x_smartseq = ae_smartseq.encoder(torch.tensor(data_smartseq, device=device)).detach()

    if args["is_celltype"]:
        label_encoder_path = args['save_dir'] + 'label_encoder.pkl'
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        print(label_encoder.classes_)
        tenx_labels = label_encoder.transform(adata_10x.obs['celltype'])

        n_classes = len(label_encoder.classes_)

        print(dict(zip(
            label_encoder.classes_,
            label_encoder.transform(label_encoder.classes_)
        )))
    else:
        n_classes = 0
        tenx_labels = torch.zeros(adata_10x.shape[0])
    generator = CellMapCycleGAN(n_classes=n_classes, is_celltype=args['is_celltype']).to(device)
    generator.load_state_dict(torch.load(args["state_dict_g"], map_location=device, weights_only=True))

    generator.eval()
    g_10x = generator.G(x_10x, torch.tensor(tenx_labels, device=device))
    # g_10x = generator.G(x_10x, torch.tensor(tenx_labels.astype(np.int64), device=device))

    ae_smartseq.eval()
    g_gex = ae_smartseq.decoder(g_10x)

    if recon_tgt:
        tgt_gex = ae_smartseq.decoder(x_smartseq).detach().cpu().numpy()
        adata_smartseq.X = tgt_gex

    g_10x = g_10x.detach().cpu().numpy()
    x_smartseq = x_smartseq.detach().cpu().numpy()
    g_gex = g_gex.detach().cpu().numpy()
    x_10x = x_10x.detach().cpu().numpy()




    print('-'*16,'g','-'*16)
    print(g_10x.shape)
    print(g_gex.shape)
    print(adata_smartseq.shape)
    print('-'*16,'smartseq','-'*16)
    # 创建新的 AnnData 对象
    adata_g = sc.AnnData(
        X=g_gex,
        obs=adata_10x.obs.copy(),  # 复制 adata_10x 的观测信息
        var=adata_smartseq.var.copy(),  # 初始化新的变量信息
        uns=adata_10x.uns.copy(),  # 复制 adata_10x 的未结构化数据
        obsm=adata_10x.obsm.copy(),  # 复制 adata_10x 的观测矩阵
    )

    adata_g.obs_names = adata_10x.obs_names
    adata_g.var_names = adata_smartseq.var_names

    adata_g.obsm['latent'] = g_10x
    adata_smartseq.obsm['latent'] = x_smartseq
    adata_10x.obsm['latent'] = x_10x

    return adata_10x, adata_smartseq, adata_g


def generate_10x_cycle(adata_10x, adata_g, args):
    device = args["device"]

    # 编码潜变量
    ae_10x = VAE(
        num_genes=adata_10x.shape[1],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )

    ae_10x.load_state_dict(torch.load(args["state_dict_t"], map_location=device, weights_only=True))
    print('loading pretrained model from: \n', args["state_dict_t"])

    ae_10x.eval()

    x_g = adata_g.obsm['latent']

    label_encoder_path = args['save_dir'] + 'label_encoder.pkl'
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    tenx_labels = label_encoder.transform(adata_g.obs['celltype'])

    n_classes = len(label_encoder.classes_)

    print(dict(zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    )))

    generator = CellMapCycleGAN(n_classes=n_classes).to(device)
    generator.load_state_dict(torch.load(args["state_dict_g"], map_location=device, weights_only=True))

    generator.eval()
    cycle_10x = generator.F(torch.tensor(x_g, device=device), torch.tensor(tenx_labels, device=device))

    cycle_gex = ae_10x.decoder(cycle_10x)

    cycle_10x = cycle_10x.detach().cpu().numpy()
    cycle_gex = cycle_gex.detach().cpu().numpy()

    print('-' * 16, 'cycle', '-' * 16)
    print(cycle_10x.shape)
    print(cycle_gex.shape)
    print(adata_g.shape)
    print('-' * 16, 'cycle', '-' * 16)
    # 创建新的 AnnData 对象
    adata_cycle = sc.AnnData(
        X=cycle_gex,
        obs=adata_g.obs.copy(),  # 复制 adata_10x 的观测信息
        var=adata_10x.var.copy(),  # 初始化新的变量信息
        uns=adata_g.uns.copy(),  # 复制 adata_10x 的未结构化数据
        obsm=adata_g.obsm.copy(),  # 复制 adata_10x 的观测矩阵
    )

    adata_cycle.obs_names = adata_g.obs_names
    adata_cycle.var_names = adata_10x.var_names

    adata_cycle.obsm['latent'] = cycle_10x

    return adata_cycle






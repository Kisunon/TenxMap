import logging
import os
import pickle
import random
import scanpy as sc
import numpy as np
import torch
import sys
from torch import optim
from torch import nn
import torch.nn.functional as F

from scipy.sparse import issparse

if sys.path[0] != "../":
    sys.path.insert(0, "../")
from model.VAE_model_v2 import VAE
from model.dataset import CellDatasetV2 as CellDataset
from model.dataset import BalancedDataset
from model.cycle_gan_celltype import CellMapCycleGAN


def seed_everything(seed):
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 添加 CUDA 确定性配置
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    print('set seed: ', seed)


class CellMapTrainer:
    def __init__(self, ae_10x, ae_smartseq, tenx_gene_names, smart_gene_names, n_classes, args):
        self.lr = args["lr"]
        self.device = args["device"]
        self.is_celltype = args["is_celltype"]

        self.model = CellMapCycleGAN(n_classes, latent_dim=args["latent_dim"], is_celltype=self.is_celltype).to(self.device)

        self.ae_10x = ae_10x.to(self.device)
        self.ae_smartseq = ae_smartseq.to(self.device)

        self.cyc_coef = args["cyc_coef"]
        self.cyc_g_coef = args["cyc_g_coef"]
        self.style_coef = args["sty_coef"]
        self.cont_coef = args["cont_coef"]
        self.ae_coef = args["ae_coef"]

        # 新增基因名称列表
        self.tenx_gene_names = tenx_gene_names
        self.smart_gene_names = smart_gene_names

        # 找出共有基因索引
        tenx_gene_names = list(tenx_gene_names)
        smart_gene_names = list(smart_gene_names)
        # self.common_genes = list(set(tenx_gene_names) & set(smart_gene_names))
        self.common_genes = sorted(list(set(tenx_gene_names) & set(smart_gene_names)))

        self.tenx_mask = torch.zeros(len(tenx_gene_names), dtype=torch.bool).to(self.device)
        self.smart_mask = torch.zeros(len(smart_gene_names), dtype=torch.bool).to(self.device)
        self.tenx_common_indices = [tenx_gene_names.index(gene) for gene in self.common_genes]
        self.smart_common_indices = [smart_gene_names.index(gene) for gene in self.common_genes]

        # 优化器
        self.optimizer_G = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.F.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            list(self.model.D_smart.parameters()) + list(self.model.D_tenx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_AE = optim.Adam(
            list(self.ae_10x.parameters()) + list(self.ae_smartseq.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

        # 损失函数
        self.criterion_adv = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_autoencoder = nn.MSELoss(reduction='mean')

    def pearson_loss(self, x, y):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        covariance = (x_centered * y_centered).mean()
        x_std = x_centered.std()
        y_std = y_centered.std()
        return 1 - covariance / (x_std * y_std + 1e-8)

    def batch_pearson_loss(self, x, y):
        # x 和 y 形状: (batch_size, feature_dim)
        # 计算每个样本的均值（沿特征维度）
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        # 中心化
        x_centered = x - x_mean
        y_centered = y - y_mean

        # 协方差（每个样本单独计算）
        covariance = (x_centered * y_centered).mean(dim=1)  # 形状: (batch_size,)

        # 标准差（无偏估计设为 False，与协方差计算一致）
        x_std = x_centered.std(dim=1, unbiased=False)
        y_std = y_centered.std(dim=1, unbiased=False)

        # 计算每个样本的 Pearson 系数
        pearson = covariance / (x_std * y_std + 1e-8)  # 防止除零

        # 使用Huber损失使计算更稳健
        huber_loss = F.huber_loss(pearson, torch.ones_like(pearson), delta=0.1)

        return huber_loss  # 返回批量平均损失

    def batch_cosine_loss(self, x, y):
        # 计算每个样本的余弦相似度（沿特征维度）
        if x.shape[0] != y.shape[0]:
            min_length = min(x.shape[0], y.shape[0])
            x = x[:min_length]
            y = y[:min_length]
        cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=1)  # 形状: (batch_size,)
        return 1 - cos_sim.mean()  # 返回批量平均损失

    def style_loss(self, x, y):
        # 计算每个样本的均值和标准差
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        y_std = y.std(dim=0, keepdim=True)
        # 计算均值和标准差的差异
        mean_diff = (x_mean - y_mean).abs().mean()
        std_diff = (x_std - y_std).abs().mean()
        return mean_diff + std_diff

    def _get_conditional_input(self, x, labels): # TODO: 直接给判别器一个embedding，而不是onehot
        """ 为判别器生成带类别条件的信息 """
        batch_size = labels.size(0)
        onehot_labels = torch.zeros(batch_size, self.model.n_classes).to(self.device)

        labels = labels.to(torch.int64)
        onehot_labels.scatter_(1, labels.unsqueeze(1), 1)
        return torch.cat([x, onehot_labels], dim=1)

    def train_step(self, batch, use_identity_loss=False, use_self_constrained_loss=False):
        tenx_x = batch[0].to(self.device)
        smart_x = batch[1].to(self.device)
        label = batch[2].to(self.device)

        # ----------------------
        #  训练生成器
        # ----------------------
        self.optimizer_G.zero_grad()
        self.optimizer_AE.zero_grad()

        # 前向传播
        tenx_z = self.ae_10x.encoder(tenx_x)
        smart_z = self.ae_smartseq.encoder(smart_x)

        outputs = self.model(tenx_z, smart_z, label)

        if self.is_celltype:
            d_input_smart = self._get_conditional_input(outputs['fake_smart'], label)
        else:
            d_input_smart = outputs['fake_smart']
        pred_fake_smart = self.model.D_smart(d_input_smart)
        loss_G_smart = self.criterion_adv(pred_fake_smart, torch.ones_like(pred_fake_smart))

        if self.is_celltype:
            d_input_tenx = self._get_conditional_input(outputs['fake_tenx'], label)
        else:
            d_input_tenx = outputs['fake_tenx']
        pred_fake_tenx = self.model.D_tenx(d_input_tenx)
        loss_G_tenx = self.criterion_adv(pred_fake_tenx, torch.ones_like(pred_fake_tenx))
        # print('pred adv G:',pred_fake_smart[0].item(), pred_fake_tenx[0].item())
        # print('loss adv G:',loss_G_smart.item(), loss_G_tenx.item())

        # 循环一致性损失
        loss_cycle = self.criterion_cycle(outputs['cycled_tenx'], tenx_z) + \
                     self.criterion_cycle(outputs['cycled_smart'], smart_z)

        loss_G = (loss_G_smart + loss_G_tenx) + self.cyc_coef * loss_cycle

        if self.cyc_g_coef > 0:
            tenx_cycle = self.ae_10x.decoder(outputs['cycled_tenx'])
            smart_cycle = self.ae_smartseq.decoder(outputs['cycled_smart'])

            # 循环一致性损失
            loss_cycle_g = self.criterion_cycle(tenx_cycle, tenx_x) + \
                         self.criterion_cycle(smart_cycle, smart_x)
            loss_G += self.cyc_g_coef * loss_cycle_g

        # 身份损失

        if use_identity_loss:
            loss_id_smart = self.criterion_identity(self.model.G(smart_z), smart_z)  # todo: input
            loss_id_tenx = self.criterion_identity(self.model.F(tenx_z), tenx_z)  # todo: input

            loss_G += 7.0 * (loss_id_smart + loss_id_tenx)

        # 对生成样本和目标域真实样本计算L1损失或许一定程度上能够拉近这种弱配对样本之间的关系？
        if use_self_constrained_loss:
            loss_sc_smart = self.criterion_identity(outputs['fake_smart'], smart_z)
            loss_sc_tenx = self.criterion_identity(outputs['fake_tenx'], tenx_z)

            loss_G += 2.0 * (loss_sc_smart + loss_sc_tenx)

        # todo: 内容和风格损失是否需要对反向过程也同样进行计算呢？直觉上来说应该是需要的，这样可能能够更好的循环
        if self.cont_coef > 0:
            fake_smart_recon = self.ae_smartseq.decoder(outputs['fake_smart'])
            fake_tenx_recon = self.ae_10x.decoder(outputs['fake_tenx'])

            # 提取共有基因数据
            fake_smart_common = fake_smart_recon[:, self.smart_common_indices]
            tenx_common = tenx_x[:, self.tenx_common_indices]
            loss_content = self.batch_cosine_loss(fake_smart_common, tenx_common)

            # 提取共有基因数据
            fake_tenx_common = fake_tenx_recon[:, self.tenx_common_indices]
            smart_common = smart_x[:, self.smart_common_indices]
            loss_content += self.batch_cosine_loss(fake_tenx_common, smart_common)
            loss_G += self.cont_coef * loss_content

        if self.style_coef > 0:
            fake_smart_recon = self.ae_smartseq.decoder(outputs['fake_smart'])
            loss_style = self.style_loss(fake_smart_recon, smart_x)
            fake_tenx_recon = self.ae_10x.decoder(outputs['fake_tenx'])
            loss_style += self.style_loss(fake_tenx_recon, tenx_x)
            loss_G += self.style_coef * loss_style

        loss_G.backward()
        self.optimizer_G.step()

        # 更新自动编码器
        loss_t = self.criterion_autoencoder(self.ae_10x(tenx_x), tenx_x)
        loss_s = self.criterion_autoencoder(self.ae_smartseq(smart_x), smart_x)
        # todo: 自动编码器的损失的权重
        loss_autoencoder = self.ae_coef * (loss_t + loss_s)
        loss_autoencoder.backward()
        self.optimizer_AE.step()

        # ----------------------
        #  训练判别器
        # ----------------------
        self.optimizer_D.zero_grad()

        # 真实样本
        if self.is_celltype:
            real_smart_input = self._get_conditional_input(smart_z.detach(), label)
        else:
            real_smart_input = smart_z.detach()
        pred_real_smart = self.model.D_smart(real_smart_input)
        loss_real_smart = self.criterion_adv(pred_real_smart, torch.ones_like(pred_real_smart))

        if self.is_celltype:
            real_tenx_input = self._get_conditional_input(tenx_z.detach(), label)
        else:
            real_tenx_input = tenx_z.detach()
        pred_real_tenx = self.model.D_tenx(real_tenx_input)
        loss_real_tenx = self.criterion_adv(pred_real_tenx, torch.ones_like(pred_real_tenx))

        # 生成样本
        if self.is_celltype:
            fake_smart_input = self._get_conditional_input(outputs['fake_smart'].detach(), label)
        else:
            fake_smart_input = outputs['fake_smart'].detach()
        pred_fake_smart = self.model.D_smart(fake_smart_input)
        loss_fake_smart = self.criterion_adv(pred_fake_smart, torch.zeros_like(pred_fake_smart))

        if self.is_celltype:
            fake_tenx_input = self._get_conditional_input(outputs['fake_tenx'].detach(), label)
        else:
            fake_tenx_input = outputs['fake_tenx'].detach()
        pred_fake_tenx = self.model.D_tenx(fake_tenx_input)
        loss_fake_tenx = self.criterion_adv(pred_fake_tenx, torch.zeros_like(pred_fake_tenx))

        # print('pred adv D:',pred_real_smart[0].item(), pred_fake_smart[0].item(), pred_real_tenx[0].item(), pred_fake_tenx[0].item())
        # print('loss adv D:',loss_real_smart.item(), loss_fake_smart.item(), loss_real_tenx.item(), loss_fake_tenx.item())

        # 总判别器损失
        loss_D = (loss_real_smart + loss_fake_smart + loss_real_tenx + loss_fake_tenx) / 2
        loss_D.backward()
        self.optimizer_D.step()

        loss_dict = {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_cycle': loss_cycle.item(),
        }

        if self.cont_coef > 0:
            loss_dict['loss_content'] = loss_content.item()
        if self.style_coef > 0:
            loss_dict['loss_style'] = loss_style.item()
        loss_dict['loss_autoencoder'] = loss_autoencoder.item()

        return loss_dict


def train_ed(adata, args, continue_train=None):
    logger = logging.getLogger(__name__)

    device = args["device"]
    epochs = args["epochs"]
    beta = args["beta"]
    add_noise = args["add_noise"]
    noise_prob = args["noise_prob"]
    """
    Trains a autoencoder
    """
    dataset = CellDataset(adata, use_celltype=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    autoencoder = VAE(
        num_genes=dataset.num_genes,
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
        negative_selection=args["negative_selection"],
        beta=beta,
    )

    if continue_train is None:
        if args["state_dict"] is not None:
            filenames = {}
            checkpoint_path = {
                "encoder": os.path.join(
                    args["state_dict"], filenames.get("model", "encoder.ckpt")
                ),
                "decoder": os.path.join(
                    args["state_dict"], filenames.get("model", "decoder.ckpt")
                ),
                "gene_order": os.path.join(
                    args["state_dict"], filenames.get("gene_order", "gene_order.tsv")
                ),
            }
            logger.info('loading pretrained model from: \n %s', args["state_dict"])
            use_gpu = device[:4] == "cuda"
            autoencoder.encoder.load_state(checkpoint_path["encoder"], use_gpu)
            autoencoder.decoder.load_state(checkpoint_path["decoder"], use_gpu)
    else:
        logger.info('loading pretrained model from: \n %s', args["state_dict"])
        autoencoder.load_state_dict(torch.load(args["state_dict"], map_location=device, weights_only=True))

    args["hparams"] = autoencoder.hparams

    training_losses = []
    if continue_train is None:
        epochs_iter = range(epochs)
    else:
        epochs_iter = range(continue_train, epochs)
    for i in epochs_iter:
        for batch in dataloader:
            minibatch_training_stats = autoencoder.train_step(batch, add_noise=add_noise, noise_prob=noise_prob)
            # for key, val in minibatch_training_stats.items():
            #     training_losses.append(val)
            batch_loss = []
            for key, val in minibatch_training_stats.items():
                batch_loss.append(val)
            training_losses.append(batch_loss)

        # for key, val in minibatch_training_stats.items():
        #     print('epoch ', i, f'loss {key}', val)
        logger.info('epoch %d/%d %s', i + 1, epochs, minibatch_training_stats)

        if (i % 10 == 0 & i != 0) | (i == epochs - 1):
            os.makedirs(args["save_dir"], exist_ok=True)
            torch.save(
                autoencoder.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], i),
                ),
            )
            training_losses_np = np.array(training_losses)
            if continue_train is None:
                i_str = str(i)
            else:
                i_str = f'{continue_train}to{i}'
            np.save(
                os.path.join(
                    args["save_dir"],
                    "training_losses_seed={}_step={}.npy".format(args["seed"], i_str),
                ), training_losses_np
            )


def train_map(adata_10x, adata_smartseq, args, use_hvg_10x=False, continue_train=None):
    # 假设日志已经在主文件中配置好，直接使用
    logger = logging.getLogger(__name__)

    device = args["device"]

    # cell_types_10x = adata_10x.obs['tissue_celltype'].values
    # cell_types_smartseq = adata_smartseq.obs['tissue_celltype'].values

    if use_hvg_10x:
        sc.pp.highly_variable_genes(adata_10x)
        tenx_gene_names = adata_10x.var_names[adata_10x.var['highly_variable']]
    else:
        tenx_gene_names = adata_10x.var_names

    smartseq_gene_names = adata_smartseq.var_names

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

    if args["state_dict_t"] is not None:
        ae_10x.load_state_dict(torch.load(args["state_dict_t"], map_location=device, weights_only=True))
        logger.info('loading pretrained model from: \n %s', args["state_dict_t"])
    elif args["state_dict"] is not None:
        filenames = {}
        checkpoint_path = {
            "encoder": os.path.join(
                args["state_dict"], filenames.get("model", "encoder.ckpt")
            ),
            "decoder": os.path.join(
                args["state_dict"], filenames.get("model", "decoder.ckpt")
            ),
            "gene_order": os.path.join(
                args["state_dict"], filenames.get("gene_order", "gene_order.tsv")
            ),
        }
        logger.info('loading pretrained model from: \n %s', args["state_dict"])
        use_gpu = device[:4] == "cuda"
        ae_10x.encoder.load_state(checkpoint_path["encoder"], use_gpu)
        ae_10x.decoder.load_state(checkpoint_path["decoder"], use_gpu)

    if args["state_dict_s"] is not None:
        ae_smartseq.load_state_dict(torch.load(args["state_dict_s"], map_location=device, weights_only=True))
        logger.info('loading pretrained model from: \n %s', args["state_dict_s"])
    elif args["state_dict"] is not None:
        filenames = {}
        checkpoint_path = {
            "encoder": os.path.join(
                args["state_dict"], filenames.get("model", "encoder.ckpt")
            ),
            "decoder": os.path.join(
                args["state_dict"], filenames.get("model", "decoder.ckpt")
            ),
            "gene_order": os.path.join(
                args["state_dict"], filenames.get("gene_order", "gene_order.tsv")
            ),
        }
        logger.info('loading pretrained model from: \n %s', args["state_dict"])
        use_gpu = device[:4] == "cuda"
        ae_smartseq.encoder.load_state(checkpoint_path["encoder"], use_gpu)
        ae_smartseq.decoder.load_state(checkpoint_path["decoder"], use_gpu)

    dataset = BalancedDataset(adata_10x, adata_smartseq)
    n_classes = dataset.n_classes
    logger.info('dataset.shape: %d', len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    trainer = CellMapTrainer(ae_10x, ae_smartseq, tenx_gene_names, smartseq_gene_names, n_classes, args)

    epochs = args["epochs"]

    training_losses = []
    for epoch in range(epochs):
        # np.random.seed(args["seed"] + epoch)  # 每个epoch使用不同但可预测的种子
        # dataloader.dataset.rebalance()

        for batch in dataloader:
            # trainer.train_step(x_10x_batch, x_smartseq_batch, cell_types_batch)
            loss = trainer.train_step(batch)
            batch_loss = []
            for key, val in loss.items():
                batch_loss.append(val)
            training_losses.append(batch_loss)
        logger.info('epoch %d/%d %s', epoch + 1, epochs, loss)

        if (epoch % 10 == 0 & epoch != 0) | (epoch == epochs - 1):
            os.makedirs(args["save_dir"], exist_ok=True)
            torch.save(
                trainer.model.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], epoch),
                ),
            )
            torch.save(
                trainer.ae_10x.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "ae_10x_seed={}_step={}.pt".format(args["seed"], epoch),
                ),
            )
            torch.save(
                trainer.ae_smartseq.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "ae_smartseq_seed={}_step={}.pt".format(args["seed"], epoch),
                ),
            )

            training_losses_np = np.array(training_losses)
            if continue_train is None:
                i_str = str(epoch)
            else:
                i_str = f'{continue_train}to{epoch}'
            np.save(
                os.path.join(
                    args["save_dir"],
                    "training_losses_seed={}_step={}.npy".format(args["seed"], i_str),
                ), training_losses_np
            )
    if args["is_celltype"]:
        with open(os.path.join(args["save_dir"], 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(dataset.label_encoder, f)

    logger.info('end')


if __name__ == "__main__":
    device = 'cuda:0'  # todo: check the device
    seed = 42
    seed_everything(seed)

    args = {'loss_ae': 'mse',
            'decoder_activation': 'ReLU',
            'local_rank': 0,
            'seed': seed,
            'hparams': '',
            'epochs': 100,
            'batch_size': 128,
            'beta': 0.001,

            'state_dict': '../pretrain/annotation_model_v1',
            # 'save_dir': '../output/test/',
            # 'data_dir': '../data/LIHC_GSE140228_10X_exm.h5ad',
            'device': device
            }

    data_t = '../data/single_train/CRC_GSE146771_10X_10x_immune.h5ad'
    data_s = '../data/single_train/CRC_GSE146771_Smartseq2_smartseq_immune.h5ad'

    save_dir_t = '../backup/trian_state_dict_before250626/10x_CRC_GSE146771_10X_common_trip17/'
    save_dir_s = '../backup/trian_state_dict_before250626/smartseq_CRC_GSE146771_Smartseq2_common_trip17/'
    save_dir_g = '../backup/trian_state_dict_before250626/immune_CRC_GSE146771_to_CRC_GSE146771_common_trip_test/'

    epochs_t = 50
    epochs_s = 100

    adata_10x = sc.read_h5ad(data_t)
    adata_smartseq = sc.read_h5ad(data_s)

    type_smart = adata_smartseq.obs['celltype'].value_counts()
    type_10x = adata_10x.obs['celltype'].value_counts()

    cross_type = type_smart.index.intersection(type_10x.index)
    diff_type = type_smart.index.difference(type_10x.index)
    print('cross_type: ', cross_type)
    print('diff_type', diff_type)

    adata_smartseq_common = adata_smartseq[adata_smartseq.obs['celltype'].isin(cross_type)]
    adata_10x_common = adata_10x[adata_10x.obs['celltype'].isin(cross_type)]

    adata_10x = adata_10x_common.copy()
    adata_smartseq = adata_smartseq_common.copy()

    args['data_dir'] = data_t
    args['save_dir'] = save_dir_t
    args['epochs'] = epochs_t
    state_dict_t = save_dir_t + f'model_seed={seed}_step={epochs_t - 1}.pt'
    if os.path.exists(state_dict_t) is False:
        train_ed(adata_10x, args, continue_train=None)

    args['data_dir'] = data_s
    args['save_dir'] = save_dir_s
    args['epochs'] = epochs_s
    state_dict_s = save_dir_s + f'model_seed={seed}_step={epochs_s - 1}.pt'
    if os.path.exists(state_dict_s) is False:
        train_ed(adata_smartseq, args, continue_train=None)

    args_map = {'loss_ae': 'mse',
                'decoder_activation': 'ReLU',
                'local_rank': 0,
                'seed': seed,
                'hparams': '',
                'epochs': 200,
                'batch_size': 512,
                'lr': 2e-3,

                'state_dict_t': state_dict_t,
                'state_dict_s': state_dict_s,
                'save_dir': save_dir_g,
                'data_dir_t': data_t,
                'data_dir_s': data_s,
                'device': device
                }

    train_map(adata_10x, adata_smartseq, args_map, use_hvg_10x=False, continue_train=None)
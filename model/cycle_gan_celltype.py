import torch
from torch import optim
import torch.nn as nn

class Generator(nn.Module):
    """ 加入细胞类型信息的生成器 """

    def __init__(self, n_classes, latent_dim=128, is_celltype=True):
        super().__init__()
        self.is_celltype = is_celltype
        self.n_classes = n_classes
        if is_celltype:
            self.label_emb = nn.Embedding(n_classes, latent_dim)
            G_input_dim = latent_dim * 2
        else:
            G_input_dim = latent_dim

        # todo: 测试是否需要这个BatchNorm
        self.main = nn.Sequential(
            nn.Linear(G_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, cell_type_labels):
        if self.is_celltype:
            # 拼接潜变量和类别嵌入
            cell_emb = self.label_emb(cell_type_labels)  # TODO: label_embedding放在cycleganer不是G中，直接给G一个embedding
            combined = torch.cat([x, cell_emb], dim=1)
        else:
            combined = x
        return self.main(combined)


class CellMapCycleGAN(nn.Module):
    def __init__(self, n_classes, latent_dim=128, is_celltype=True):
        super().__init__()
        self.is_celltype = is_celltype
        self.n_classes = n_classes

        # 生成器网络
        self.G = Generator(n_classes, latent_dim, is_celltype)  # 10x → SmartSeq
        self.F = Generator(n_classes, latent_dim, is_celltype)  # SmartSeq → 10x

        if is_celltype:
            D_input_dim = latent_dim + n_classes
        else:
            D_input_dim = latent_dim

        # todo: 判别器可以尝试使用支持mse损失的网络
        # 判别器网络
        self.D_smart = nn.Sequential(
            nn.Linear(D_input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.D_tenx = nn.Sequential(
            nn.Linear(D_input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, tenx_z, smart_z, cell_types):
        # 前向映射
        fake_smart = self.G(tenx_z, cell_types)
        cycled_tenx = self.F(fake_smart, cell_types)

        # 反向映射
        fake_tenx = self.F(smart_z, cell_types)
        cycled_smart = self.G(fake_tenx, cell_types)

        return {
            'fake_smart': fake_smart,
            'cycled_tenx': cycled_tenx,
            'fake_tenx': fake_tenx,
            'cycled_smart': cycled_smart
        }
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
import anndata as ad
import scanpy as sc
from typing import List


class Encoder(nn.Module):
    """A class that encapsulates the encoder."""

    def __init__(
            self,
            n_genes: int,
            latent_dim: int = 128,
            hidden_dim: List[int] = [1024, 1024],
            dropout: float = 0.5,
            input_dropout: float = 0.4,
            residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        input_dropout: float, default: 0.4
            The dropout rate for the input layer
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> F.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool
            Boolean indicating whether or not to use GPUs.
        """
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename, weights_only=True)
        state_dict = ckpt['state_dict']

        # print('----------check----------')
        # for key in state_dict.keys():
        #     print(key)
        # print('----------check----------')

        first_layer_key = ['network.0.1.weight',
                           'network.0.1.bias',
                           'network.0.2.weight',
                           'network.0.2.bias',
                           'network.0.2.running_mean',
                           'network.0.2.running_var',
                           'network.0.2.num_batches_tracked',
                           'network.0.3.weight]', ]
        for key in first_layer_key:
            if key in state_dict:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

        # print('----------check----------')
        # for key in self.state_dict().keys():
        #     print(key)
        # print('----------check----------')


class Decoder(nn.Module):
    """A class that encapsulates the decoder."""

    def __init__(
            self,
            n_genes: int,
            latent_dim: int = 128,
            hidden_dim: List[int] = [1024, 1024],
            dropout: float = 0.5,
            residual: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        n_genes: int
            The number of genes in the gene space, representing the input dimensions.
        latent_dim: int, default: 128
            The latent space dimensions
        hidden_dim: List[int], default: [1024, 1024]
            A list of hidden layer dimensions, describing the number of layers and their dimensions.
            Hidden layers are constructed in the order of the list for the encoder and in reverse
            for the decoder.
        dropout: float, default: 0.5
            The dropout rate for hidden layers
        residual: bool, default: False
            Use residual connections.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))
        self.network.append(nn.ReLU())

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save state dictionary.

        Parameters
        ----------
        filename: str
            Filename to save the state dictionary.
        """
        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Parameters
        ----------
        filename: str
            Filename containing the model state.
        use_gpu: bool
            Boolean indicating whether to use GPUs.
        """
        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename, weights_only=True)
        state_dict = ckpt['state_dict']
        last_layer_key = ['network.3.weight',
                          'network.3.bias', ]
        for key in last_layer_key:
            if key in state_dict:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)
        # self.load_state_dict(ckpt["state_dict"])


class VAE(torch.nn.Module):
    """
    VAE base on compositional perturbation autoencoder (CPA)
    """

    def __init__(
            self,
            num_genes,
            device="cuda",
            seed=0,
            loss_ae="gauss",
            decoder_activation="linear",
            hidden_dim=128,
            margin=0.05,
            negative_selection="hardest",
            beta=0.001,
    ):
        super(VAE, self).__init__()
        self.margin = margin
        self.negative_selection = negative_selection
        self.beta = beta

        # set generic attributes
        self.num_genes = num_genes
        self.device = device
        self.seed = seed
        self.loss_ae = loss_ae
        # early-stopping
        self.best_score = -1e3
        self.patience_trials = 0

        # set hyperparameters
        self.set_hparams_(hidden_dim)

        # set models
        self.hidden_dim = [1024, 1024, 1024]
        self.dropout = 0.0
        self.input_dropout = 0.0
        self.residual = False
        self.encoder = Encoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.num_genes,
            latent_dim=self.hparams["dim"],
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )

        # losses
        self.loss_autoencoder = nn.MSELoss(reduction='mean')
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

        self.iteration = 0

        self.to(self.device)

        # optimizers
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
                get_params(self.encoder, True)
                + get_params(self.decoder, True)
        )
        self.optimizer_autoencoder = torch.optim.AdamW(_parameters, lr=self.hparams["autoencoder_lr"],
                                                       weight_decay=self.hparams["autoencoder_wd"], )

    def forward(self, genes, return_latent=False, return_decoded=False):
        """
        If return_latent=True, act as encoder only. If return_decoded, genes should 
        be the latent representation and this act as decoder only.
        """
        if return_decoded:
            gene_reconstructions = self.decoder(genes)
            gene_reconstructions = nn.ReLU()(gene_reconstructions)  # only relu when inference
            return gene_reconstructions

        latent_basal = self.encoder(genes)
        if return_latent:
            return latent_basal

        gene_reconstructions = self.decoder(latent_basal)

        return gene_reconstructions

    def set_hparams_(self, hidden_dim):
        """
        Set hyper-parameters to default values or values fixed by user.
        """

        self.hparams = {
            "dim": hidden_dim,
            "autoencoder_width": 5000,
            "autoencoder_depth": 3,
            "adversary_lr": 3e-4,
            "autoencoder_wd": 0.01,
            "autoencoder_lr": 5e-4,
        }

        return self.hparams

    def pdist(self, vectors: np.ndarray):
        """Get pair-wise distance between all cell embeddings.

        Parameters
        ----------
        vectors: numpy.ndarray
            Cell embeddings.

        Returns
        -------
        numpy.ndarray
            Distance matrix of cell embeddings.
        """
        vectors_squared_sum = (vectors ** 2).sum(axis=1)
        distance_matrix = (
                -2 * np.matmul(vectors, np.matrix.transpose(vectors))
                + vectors_squared_sum.reshape(1, -1)
                + vectors_squared_sum.reshape(-1, 1)
        )
        return distance_matrix

    def hardest_negative(self, loss_values):
        """Get hardest negative.

        Parameters
        ----------
        loss_values: numpy.ndarray
            Triplet loss of all negatives for given anchor positive pair.
        margin: float
            Triplet loss margin.

        Returns
        -------
        int
            Index of selection.
        """
        hard_negative = np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative] > 0 else None

    def random_negative(self, loss_values):
        """Get random negative.

        Parameters
        ----------
        loss_values: numpy.ndarray
            Triplet loss of all negatives for given anchor positive pair.

        Returns
        -------
        int
            Index of selection.
        """

        hard_negatives = np.where(loss_values > 0)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

    def semihard_negative(self, loss_values):
        """Get a random semihard negative.

        Parameters
        ----------
        loss_values: numpy.ndarray
            Triplet loss of all negatives for given anchor positive pair.

        Returns
        -------
        int
            Index of selection.
        """

        semihard_negatives = np.where(
            np.logical_and(loss_values < self.margin, loss_values > 0)
        )[0]
        return (
            np.random.choice(semihard_negatives)
            if len(semihard_negatives) > 0
            else None
        )

    def generate_triplets(self, embeddings: np.ndarray, labels: np.ndarray, margin: float = 0.05):
        """Generate triplets as anchor, positive, and negative cell indices.

        Parameters
        ----------
        embeddings: numpy.ndarray
            Cell embeddings.
        labels: numpy.ndarray
            Cell labels.
        margin: float, default: 0.05
            Triplet loss margin.

        Returns
        -------
        tuple[list, list, list]
            A tuple of lists containing anchor, positive, and negative cell indices.
        """
        # 计算距离矩阵
        if isinstance(embeddings, torch.Tensor):
            distance_matrix = self.pdist(embeddings.detach().cpu().numpy())
        else:
            distance_matrix = self.pdist(embeddings)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        unique_labels = np.unique(labels)
        triplets = []

        for label in unique_labels:
            # 找出当前标签对应的细胞索引
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]

            # 若当前标签的细胞数量少于 2 个，则跳过
            if len(label_indices) < 2:
                continue

            # 找出负样本的细胞索引
            negative_indices = np.where(np.logical_not(label_mask))[0]

            # 生成所有可能的锚点 - 正样本对
            anchor_positives = list(combinations(label_indices, 2))

            for anchor_positive in anchor_positives:
                # 计算三元组损失
                loss_values = (
                        distance_matrix[anchor_positive[0], anchor_positive[1]]
                        - distance_matrix[[anchor_positive[0]], negative_indices]
                        + margin
                )

                # 选择最难的负样本
                # hard_negative = self.hardest_negative(loss_values, margin)

                # select one negative for anchor positive pair based on selection function
                if self.negative_selection == "semihard":
                    hard_negative = self.semihard_negative(loss_values)
                elif self.negative_selection == "hardest":
                    hard_negative = self.hardest_negative(loss_values)
                elif self.negative_selection == "random":
                    hard_negative = self.random_negative(loss_values)
                else:
                    hard_negative = None

                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if not triplets:
            triplets.append([0, 0, 0])

        anchor_idx, positive_idx, negative_idx = zip(*triplets)
        return list(anchor_idx), list(positive_idx), list(negative_idx)

    def train_step(self, batch, add_noise=None, noise_prob=0.2):
        """
        Train VAE.
        """
        genes, labels = batch
        genes = genes.to(self.device)

        if add_noise == 'binary':
            # 生成布尔噪音掩码 (伯努利分布)
            # 以noise_prob概率将特征置零，保留(1-noise_prob)比例的特征
            mask = torch.bernoulli(torch.full_like(genes, 1 - noise_prob))
            genes = genes * mask  # 应用掩码
        elif add_noise == 'gaussian':
            # 添加高斯噪声
            noisy_genes = genes + torch.normal(mean=0, std=0.1, size=genes.size(), device=self.device)
            genes = torch.clamp(noisy_genes, min=0)  # 确保非负

        embeddings = self.encoder(genes)
        gene_reconstructions = self.decoder(embeddings)

        anchor_idx, positive_idx, negative_idx = self.generate_triplets(
            embeddings, labels, margin=self.triplet_loss.margin
        )
        anchor = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[negative_idx]

        # autoencoder loss
        reconstruction_loss = self.loss_autoencoder(gene_reconstructions, genes)
        # triplet loss
        if self.beta != 0:
            triplet_loss = self.triplet_loss(anchor, positive, negative)
            loss = (1-self.beta)*reconstruction_loss + self.beta*triplet_loss
        else:
            loss = reconstruction_loss

        self.optimizer_autoencoder.zero_grad()
        loss.backward()
        self.optimizer_autoencoder.step()

        self.iteration += 1

        loss_dict = {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss": loss.item(),
        }
        if self.beta != 0:
            loss_dict['loss_triplet'] = triplet_loss.item()

        return loss_dict

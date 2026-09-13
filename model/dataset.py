from collections import defaultdict

import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import SpectralClustering
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CellDataset(Dataset):
    def __init__(
        self,
        adata
    ):
        super().__init__()
        if issparse(adata.X):
            self.data = adata.X.toarray().astype(np.float32)
        else:
            self.data = adata.X.astype(np.float32)

        classes = adata.obs['celltype'].values
        label_encoder = LabelEncoder()
        labels = classes
        label_encoder.fit(labels)
        classes = label_encoder.transform(labels)
        self.class_name = classes
        self.num_genes = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        y = None
        if self.class_name is not None:
            y = np.array(self.class_name[idx], dtype=np.int64)
        return arr, y


class CellDatasetV2(Dataset):
    def __init__(
        self,
        adata,
        n_clusters=None,
        use_celltype=False
    ):
        super().__init__()
        if issparse(adata.X):
            self.data = adata.X.toarray().astype(np.float32)
            adata.X = self.data
        else:
            self.data = adata.X.astype(np.float32)

        if n_clusters is None:
            n_clusters = len(adata.obs["celltype"].unique())

        if use_celltype:
            classes = adata.obs["celltype"].values
        else:
            adata = self.get_cluster_labels(adata, n_clusters=n_clusters)
            # adata = self.cluster(adata, n_clusters=n_clusters)

            classes = adata.obs["cluster"].values

        label_encoder = LabelEncoder()
        labels = classes
        label_encoder.fit(labels)
        classes = label_encoder.transform(labels)
        self.class_name = classes
        self.num_genes = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        y = None
        if self.class_name is not None:
            y = np.array(self.class_name[idx], dtype=np.int64)
        return arr, y

    def get_cluster_labels(self, adata, n_clusters=9):
        sc.pp.pca(adata, n_comps=50)
        X_pca = adata.obsm["X_pca"]  # 取前20个主成分

        # 谱聚类（假设真实类簇数为n_clusters）
        clusters = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors").fit_predict(X_pca)
        adata.obs["cluster"] = clusters.astype(str)
        return adata

    def cluster(self, adata, n_clusters=9, npc=15):
        if n_clusters > 1:
            n = np.min((adata.shape[0], adata.shape[1]))
            pca = PCA(n_components=n)
            pcs = pca.fit_transform(adata.X)
            var = (pca.explained_variance_ratio_).cumsum()
            npc_raw = (np.where(var > 0.7))[0].min()  # number of PC used in K-means
            print("npc_raw: ", npc_raw)
            if npc_raw > npc:
                npc_raw = npc
            pcs = pcs[:, :npc_raw]
            # K-means clustering on PCs
            kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit( \
                StandardScaler().fit_transform(pcs))
            clustering_label = kmeans.labels_
            adata.obs["cluster"] = clustering_label.astype(str)
            return adata


class SmartseqDataset(Dataset):
    def __init__(self, smartseq_data):
        """
        Args:
            smartseq_data: SmartSeq潜变量矩阵 (m_samples, 128)
        """
        self.smartseq_data = torch.FloatTensor(smartseq_data)

    def __len__(self):
        return len(self.smartseq_data)

    def __getitem__(self, idx):
        return self.smartseq_data[idx]


class TenxDataset(Dataset):
    def __init__(self, tenx_data):
        """
        Args:
            tenx_data: 10x潜变量矩阵 (n_samples, 128)
        """
        self.tenx_data = torch.FloatTensor(tenx_data)

    def __len__(self):
        return len(self.tenx_data)

    def __getitem__(self, idx):
        return self.tenx_data[idx]


class BalancedDataset(Dataset):
    def __init__(self, adata_t, adata_s, cell_type_key='celltype'):
        self.cell_type_key = cell_type_key
        self.adata_t = adata_t
        self.adata_s = adata_s

        data_t, data_s = self.balance(self.adata_t, self.adata_s)
        self.data_t = self.adata2tensor(data_t)
        self.data_s = self.adata2tensor(data_s)
        # 确保两个数据集的细胞类型标签一致
        assert set(data_t.obs[cell_type_key]) == set(data_s.obs[cell_type_key]), "两个数据集的细胞类型标签不一致"

        labels = data_t.obs[cell_type_key]

        self.n_classes = len(labels.unique())
        self.label_encoder = LabelEncoder().fit(labels)

        # 转换为整型标签
        self.labels = self.label_encoder.transform(labels)

        # 建立类别到索引的映射
        self.cls_dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.cls_dict[label].append(idx)

    def balance(self, adata_t, adata_s):
        # cell_type = adata.obs[self.opt.cell_type]
        ctrl = adata_t
        stim = adata_s
        ctrl_cell_type = ctrl.obs[self.cell_type_key]
        stim_cell_type = stim.obs[self.cell_type_key]
        class_num = np.unique(ctrl_cell_type)
        max_num = {}

        for i in class_num:
            # x = ctrl_cell_type[ctrl_cell_type == i].shape[0]
            # y = stim_cell_type[stim_cell_type == i].shape[0]
            # maxm = max(x, y)
            max_num[i] = (
                max(ctrl_cell_type[ctrl_cell_type == i].shape[0], stim_cell_type[stim_cell_type == i].shape[0]))

        ctrl_index_add = []
        strl_index_add = []

        for i in class_num:
            ctrl_class_index = np.array(ctrl_cell_type == i)
            stim_class_index = np.array(stim_cell_type == i)
            stim_fake = np.ones(len(stim_cell_type))

            ctrl_index_cls = np.nonzero(ctrl_class_index)[0]
            stim_index_cls = np.nonzero(stim_class_index)[0]
            stim_fake = np.nonzero(stim_fake)[0]
            # print(stim_fake)

            # print(ctrl_index_cls)
            # print(type(ctrl_index_cls))

            # print(max_num[i])
            # print(len(ctrl_index_cls))
            # print(len(stim_index_cls))
            stim_len = len(stim_index_cls)

            if stim_len == 0:
                # todo: 目前的做法是，如果刺激组中该细胞类型的样本数量为 0，则从整个刺激组中随机选择样本。
                stim_len = len(stim_cell_type)
                ctrl_index_cls = ctrl_index_cls[np.random.choice(len(ctrl_index_cls), max_num[i])]
                stim_index_cls = stim_fake[np.random.choice(stim_len, max_num[i])]

            else:
                ctrl_index_cls = ctrl_index_cls[np.random.choice(len(ctrl_index_cls), max_num[i])]
                stim_index_cls = stim_index_cls[np.random.choice(stim_len, max_num[i])]

            ctrl_index_add.append(ctrl_index_cls)
            strl_index_add.append(stim_index_cls)

        balanced_data_ctrl = ctrl[np.concatenate(ctrl_index_add)]
        balanced_data_stim = stim[np.concatenate(strl_index_add)]

        return balanced_data_ctrl, balanced_data_stim

    def rebalance(self):
        """在每个epoch开始时重新平衡数据"""
        data_t, data_s = self.balance(self.adata_t, self.adata_s)
        self.data_t = self.adata2tensor(data_t)
        self.data_s = self.adata2tensor(data_s)

        # 确保两个数据集的细胞类型标签一致
        assert set(data_t.obs[self.cell_type_key]) == set(data_s.obs[self.cell_type_key]), "两个数据集的细胞类型标签不一致"

        labels = data_t.obs[self.cell_type_key]
        self.n_classes = len(labels.unique())

        # 转换为整型标签
        self.labels = self.label_encoder.transform(labels)

        # 建立类别到索引的映射
        self.cls_dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.cls_dict[label].append(idx)

    def numpy2tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        else:
            Exception("This is not a numpy")
        return data

    def tensor2numpy(self, data):
        data = data.cpu().detach().numpy()
        return data

    def adata2numpy(self, adata):
        if sparse.issparse(adata.X):
            return adata.X.toarray().astype(np.float32)
        else:
            return adata.X.astype(np.float32)

    def adata2tensor(self, adata):
        return self.numpy2tensor(self.adata2numpy(adata))

    def __getitem__(self, idx):
        # return self.data_t[idx], self.data_s[idx], self.labels[idx], self.cls_dict[self.labels[idx]] # 报错：尺寸不一
        return self.data_t[idx], self.data_s[idx], self.labels[idx]

    def __len__(self):
        return len(self.data_t)


if __name__ == "__main__":
    data_t = '../data/single_train/CRC_GSE146771_10X_10x_immune.h5ad'
    adata = sc.read_h5ad(data_t)
    dataset = CellDatasetV2(adata)
    print(dataset.class_name)
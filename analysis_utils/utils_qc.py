import scanpy as sc
import scipy
from scipy.sparse import issparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ranksums, kstest
from sklearn.metrics import mean_squared_error, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel, wilcoxon, ks_2samp, ttest_ind, entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import trustworthiness
import umap
from matplotlib.lines import Line2D
import torch


# data quality control
def calculate_dropout_rate(adata):
    if issparse(adata.X):
        data = adata.X.toarray()
    else:
        data = adata.X

    # 计算表达矩阵中零元素的数量
    num_zeros = (data == 0).sum()
    # 计算表达矩阵中元素的总数
    total_elements = data.size
    # 计算 dropout 率
    dropout_rate = num_zeros / total_elements
    return dropout_rate


def check_genes_mean_variance_relationship(adata):
    if issparse(adata.X):
        adata = adata.copy()
        adata.X = adata.X.toarray()

    # 计算每个基因的均值和方差
    # sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化（可选但推荐）
    mean = np.array(adata.X.mean(axis=0)).flatten()  # 基因平均表达量
    variance = np.array(adata.X.var(axis=0)).flatten()  # 基因表达方差

    # 过滤极低表达基因（避免log坐标轴问题）
    min_threshold = 1e-5
    valid_idx = np.where(mean > min_threshold)[0]
    mean_filtered = mean[valid_idx]
    variance_filtered = variance[valid_idx]

    # 绘制均值-方差关系图
    plt.figure(figsize=(8, 6))
    plt.scatter(np.log10(mean_filtered),
                np.log10(variance_filtered),
                s=5, alpha=0.5)
    plt.xlabel('log10(Mean Expression)')
    plt.ylabel('log10(Variance)')
    plt.title('Mean-Variance Relationship')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加参考线 y=x（泊松分布线）
    xlim = plt.xlim()
    ylim = plt.ylim()
    max_val = max(xlim[1], ylim[1])
    plt.plot([-5, max_val], [-5, max_val], 'r--', linewidth=1, label="Poisson (mean = variance)")
    plt.legend()

    plt.show()


def check_housekeeping_genes(adata):
    if issparse(adata.X):
        adata = adata.copy()
        adata.X = adata.X.toarray()

    # 定义常见内参基因列表（根据研究物种调整）
    housekeeping_genes = ['ACTB', 'GAPDH', 'TUBB', 'PPIA', 'RPLP0',
                          'B2M', 'HPRT1', 'PGK1', 'RPL13A', 'SDHA']

    # 检查哪些内参基因存在于数据集中
    existing_hk_genes = [gene for gene in housekeeping_genes if gene in adata.var_names]
    print(f"找到 {len(existing_hk_genes)}/{len(housekeeping_genes)} 个内参基因: {', '.join(existing_hk_genes)}")

    # 可视化每个内参基因的表达分布
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 调整布局以适应基因数量
    axs = axs.flatten()

    for i, gene in enumerate(existing_hk_genes):
        # 计算表达该基因的细胞比例
        expr_percentage = (adata[:, gene].X > 0).mean() * 100

        # 绘制小提琴图
        sc.pl.violin(adata, gene, groupby=None, ax=axs[i], show=False)
        axs[i].set_title(f"{gene}\n({expr_percentage:.1f}%cell expression)")
        axs[i].set_ylabel('expression')

        # 添加参考线
        median_expr = np.median(adata[:, gene].X)
        axs[i].axhline(y=median_expr, color='r', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 高级分析：内参基因相关性
    if len(existing_hk_genes) > 1:
        # 创建内参基因表达子矩阵
        hk_matrix = adata[:, existing_hk_genes].X

        # 计算基因间相关性
        import pandas as pd
        from scipy.stats import spearmanr

        corr_matrix = np.zeros((len(existing_hk_genes), len(existing_hk_genes)))
        for i in range(len(existing_hk_genes)):
            for j in range(len(existing_hk_genes)):
                corr_matrix[i, j] = spearmanr(hk_matrix[:, i], hk_matrix[:, j])[0]

        # 可视化相关性热图
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # 添加标签
        ax.set_xticks(np.arange(len(existing_hk_genes)))
        ax.set_yticks(np.arange(len(existing_hk_genes)))
        ax.set_xticklabels(existing_hk_genes)
        ax.set_yticklabels(existing_hk_genes)

        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Spearman correlation', rotation=-90, va="bottom")

        # 添加数值标签
        for i in range(len(existing_hk_genes)):
            for j in range(len(existing_hk_genes)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                               ha="center", va="center", color="w")

        plt.title("housekeeping genes correlation")
        plt.tight_layout()
        plt.show()

    # 按细胞类型分组查看
    sc.pl.violin(adata, existing_hk_genes, groupby='celltype', rotation=90)


def calculate_marker_specificity(adata, marker_dict, celltype_col='celltype'):
    """
    计算每个标记基因在其对应细胞类型中的特异性指标
    """
    results = []

    for celltype, markers in marker_dict.items():
        # 检查该细胞类型是否存在
        if celltype not in adata.obs[celltype_col].unique():
            print(f"警告: 细胞类型 '{celltype}' 不存在于数据集中")
            continue

        for gene in markers:
            # 检查基因是否存在
            if gene not in adata.var_names:
                print(f"警告: 基因 '{gene}' 不存在于数据集中")
                continue

            # 获取目标细胞类型和其他细胞类型的表达数据
            target_cells = adata[adata.obs[celltype_col] == celltype]
            other_cells = adata[adata.obs[celltype_col] != celltype]

            # 计算目标细胞类型中的表达特征
            target_expr = target_cells[:, gene].X
            if scipy.sparse.issparse(target_expr):
                target_expr = target_expr.toarray().flatten()

            # 计算指标
            expr_percentage_target = (target_expr > 0).mean() * 100
            mean_expr_target = target_expr.mean()

            # 计算其他细胞类型中的表达特征
            other_expr = other_cells[:, gene].X
            if scipy.sparse.issparse(other_expr):
                other_expr = other_expr.toarray().flatten()

            expr_percentage_other = (other_expr > 0).mean() * 100
            mean_expr_other = other_expr.mean()

            # 计算特异性分数 (目标表达量/其他表达量)
            specificity_score = mean_expr_target / (mean_expr_other + 0.001)  # 避免除零

            # Wilcoxon秩和检验 (目标 vs 其他)
            _, p_value = ranksums(target_expr, other_expr)

            results.append({
                'CellType': celltype,
                'MarkerGene': gene,
                'Target_MeanExpr': mean_expr_target,
                'Target_ExprPct': expr_percentage_target,
                'Other_MeanExpr': mean_expr_other,
                'Other_ExprPct': expr_percentage_other,
                'SpecificityScore': specificity_score,
                'Log2FoldChange': np.log2((mean_expr_target + 0.001) / (mean_expr_other + 0.001)),
                'P_value': p_value
            })

    return pd.DataFrame(results)


def check_cell_annotation(adata, marker_dict, celltype_col='celltype', draw_hot_map=False, draw_deg=False,
                          draw_confusion_matrix=False, marker_genes_dict_generation=None):
    if marker_genes_dict_generation is None:
        marker_genes_dict_generation = {
            'T细胞': ['CD3D', 'CD3E', 'CD8A', 'CD4'],
            'B细胞': ['CD19', 'MS4A1', 'CD79A'],
            '髓系细胞': ['CD14', 'FCGR3A', 'LYZ'],
            'NK细胞': ['NKG7', 'GNLY', 'FCGR3A'],
            '造血干细胞': ['CD34', 'PROM1'],
            '上皮细胞': ['EPCAM', 'KRT19'],
            '成纤维细胞': ['COL1A1', 'DCN'],
            '内皮细胞': ['PECAM1', 'VWF'],
            # '耗竭': ['HAVCR2', 'PDCD1', 'ENTPD1', 'CTLA4', 'TIGIT', 'TNFRSF9', 'CD27'],
            # 添加更多细胞类型和标记...
        }

    if issparse(adata.X):
        adata = adata.copy()
        adata.X = adata.X.toarray()

    marker_results = calculate_marker_specificity(adata, marker_dict)
    print("\n标记基因特异性分析结果:")
    print(marker_results.sort_values(['CellType', 'SpecificityScore'], ascending=[True, False]))

    marker_set = [gene for genes in marker_genes_dict_generation.values() for gene in genes if
                   gene in adata.var_names]
    # 对marker_set 去重
    # marker_set = list(set(marker_set))

    # 点图可视化表达模式
    sc.pl.dotplot(
        adata,
        var_names=marker_set,
        groupby=celltype_col,
        standard_scale='var',  # 按基因标准化
        cmap='Reds',
        figsize=(12, 8),
        title='Cell type markers for gene expression patterns'
    )

    if draw_hot_map:
        # 热图展示表达模式
        sc.pl.heatmap(
            adata,
            var_names=[gene for genes in marker_genes_dict_generation.values() for gene in genes if
                       gene in adata.var_names],
            groupby=celltype_col,
            standard_scale='var',
            figsize=(12, 10),
            cmap='viridis',
            dendrogram=True,
            swap_axes=True,
            # title='Cell type marker gene expression heat map'
        )

    if draw_deg:
        # 自动标记基因检测
        # 使用Scanpy的差异表达分析验证注释
        sc.tl.rank_genes_groups(adata, groupby='celltype', method='wilcoxon')
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)

    if draw_confusion_matrix:
        # 构建细胞类型-标记基因混淆矩阵
        confusion_matrix = pd.crosstab(
            marker_results['CellType'],
            marker_results['MarkerGene'],
            values=marker_results['SpecificityScore'],
            aggfunc='mean'
        )
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt=".1f")
        plt.title('细胞类型-标记基因特异性矩阵')
        plt.xlabel('标记基因')
        plt.ylabel('细胞类型')
        plt.show()


def plot_combined_feature_view(adata, marker_genes, reduction='umap', figsize=(20, 15)):
    """
    绘制组合视图：左侧为细胞类型注释，右侧为标记基因表达
    """
    # 创建图形和网格布局
    fig = plt.figure(figsize=figsize)

    # 创建右侧子图的网格布局
    n_genes = sum(len(genes) for genes in marker_genes.values())
    n_cols = 4  # 每行显示4个基因
    n_rows = (n_genes + n_cols - 1) // n_cols

    gs_right = GridSpec(n_rows, n_cols, figure=fig)

    # 遍历所有标记基因并绘制
    gene_index = 0
    for celltype, genes in marker_genes.items():
        for gene in genes:
            if gene not in adata.var_names:
                print(f"跳过: 基因 '{gene}' 不存在")
                continue

            # 计算当前基因在子网格中的位置
            row = gene_index // n_cols
            col = gene_index % n_cols
            ax2 = plt.subplot(gs_right[row, col])

            # 绘制该基因的表达
            sc.pl.embedding(
                adata,
                basis=reduction,
                color=gene,
                title=gene,
                cmap='viridis',  # 使用适合表达数据的颜色映射
                vmax='p99.5',  # 使用99.5%分位数作为上限，避免异常值影响
                show=False,
                ax=ax2,
                frameon=False  # 不显示坐标轴边框
            )
            gene_index += 1

    # 调整布局
    plt.tight_layout()
    plt.suptitle(f"Cell Type Annotations vs Marker Gene Expression ({reduction.upper()})", fontsize=16)
    plt.subplots_adjust(top=0.9)  # 为总标题留出空间

    return fig


def draw_marker_genes_plot(adata, marker_genes):
    if 'X_umap' not in adata.obsm_keys() and 'X_tsne' in adata.obsm_keys():
        reduction = 'tsne'
        reduction_key = 'X_tsne'
    elif 'X_umap' in adata.obsm_keys():
        reduction = 'umap'
        reduction_key = 'X_umap'
    else:
        print("未找到降维坐标，正在计算UMAP...")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        reduction = 'umap'
        reduction_key = 'X_umap'

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.embedding(
        adata,
        basis=reduction,
        color='celltype',
        palette=sc.pl.palettes.default_20,  # 使用Scanpy默认调色板
        title='Cell Type Annotations',
        show=False,
        ax=ax,
        legend_loc='on data'  # 将图例放在图上
    )
    plt.show()

    fig = plot_combined_feature_view(adata, marker_genes, reduction=reduction)
    plt.show()

# autoencoder quality control
def plot_gene_reconstruction(adata, reconstructed, marker_genes=None,
                             n_hvgs=20, figsize=(15, 10), point_size=15,
                             alpha=0.5, hexbin=False, save_path=None):
    """
    评估并可视化关键基因的重建质量

    参数:
    adata: AnnData对象 (包含原始数据)
    reconstructed: 重建的表达矩阵 (numpy数组或AnnData)
    marker_genes: 重要标记基因列表
    n_hvgs: 显示的高变基因数量
    figsize: 图像大小
    point_size: 散点大小
    alpha: 透明度
    hexbin: 是否使用六边形分箱图
    save_path: 图像保存路径
    """
    # 确保重建数据格式正确
    if isinstance(reconstructed, sc.AnnData):
        reconstructed = reconstructed.X
    reconstructed = np.array(reconstructed)

    # 获取原始表达矩阵
    original = adata.X

    # 自动识别高变基因
    if 'highly_variable' not in adata.var.columns:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    # 组合基因列表
    hvgs = adata.var_names[adata.var.highly_variable].tolist()[:n_hvgs]
    genes_to_plot = list(set(hvgs + (marker_genes or [])))

    # 创建绘图布局
    n_cols = 4
    n_rows = int(np.ceil(len(genes_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # 存储指标结果
    metrics = []

    for i, gene in enumerate(genes_to_plot):
        if gene not in adata.var_names:
            print(f"Warning: Gene {gene} not found in dataset. Skipping.")
            continue

        ax = axes[i]
        gene_idx = adata.var_names.tolist().index(gene)
        orig_vals = original[:, gene_idx].toarray().flatten() if scipy.sparse.issparse(original) else original[:,
                                                                                                      gene_idx]
        rec_vals = reconstructed[:, gene_idx]

        # 计算指标
        mse = mean_squared_error(orig_vals, rec_vals)
        pearson_corr, _ = pearsonr(orig_vals, rec_vals)
        spearman_corr, _ = spearmanr(orig_vals, rec_vals)

        num_zeros = (orig_vals == 0).sum()
        # 计算表达矩阵中元素的总数
        total_elements = orig_vals.size
        # 计算 dropout 率
        orig_dropout_rate = num_zeros / total_elements

        metrics.append({
            'gene': gene,
            'mse': mse,
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'type': 'Marker' if gene in (marker_genes or []) else 'HVG'
        })

        # 绘制散点图
        if hexbin:
            hb = ax.hexbin(orig_vals, rec_vals, gridsize=50, cmap='viridis', mincnt=1)
            fig.colorbar(hb, ax=ax)
        else:
            ax.scatter(orig_vals, rec_vals, s=point_size, alpha=alpha, edgecolor='none')

        # 添加参考线
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # 添加统计信息
        stats_text = (f"MSE: {mse:.4f}\n"
                      f"Pearson: {pearson_corr:.3f}\n"
                      f"Spearman: {spearman_corr:.3f}\n"
                      f"Orig dpr: {orig_dropout_rate:.3f}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

        ax.set_title(f"{gene} ({'Marker' if gene in (marker_genes or []) else 'HVG'})")
        ax.set_xlabel('Original Expression')
        ax.set_ylabel('Reconstructed Expression')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()

    # 返回指标数据框
    return pd.DataFrame(metrics)


def latent_space_interpolation(decoder, adata, latent, cell_type_key='celltype',
                               cell_type_A=None, cell_type_B=None, n_points=20,
                               k=10, random_state=42, figsize=(18, 10), draw_umap=False, device=None):
    """
    执行潜在空间线性插值实验并分析结果

    参数:
    encoder: 自动编码器的编码器模型
    decoder: 自动编码器的解码器模型
    adata: AnnData对象 (包含原始数据和细胞类型注释)
    cell_type_key: 细胞类型注释的列名
    cell_type_A: 起始细胞类型
    cell_type_B: 目标细胞类型
    n_points: 插值点的数量
    k: 用于UMAP投影的最近邻数量
    random_state: 随机种子

    返回:
    插值结果字典 (包含所有中间数据)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置随机种子
    np.random.seed(random_state)

    if draw_umap:
        # 1. 准备参考UMAP (原始数据空间)
        print("准备参考UMAP...")
        if 'X_umap' not in adata.obsm_keys():
            sc.pp.neighbors(adata, n_neighbors=k, use_rep='X')
            sc.tl.umap(adata, random_state=random_state)

    # 2. 获取潜在表示
    print("编码原始数据到潜在空间...")
    if 'X_latent' not in adata.obsm_keys():
        adata.obsm['X_latent'] = latent

    # 3. 选择起始和目标细胞类型
    unique_cell_types = adata.obs[cell_type_key].unique()
    if cell_type_A is None or cell_type_B is None:
        # 自动选择两个不同的细胞类型
        if len(unique_cell_types) < 2:
            raise ValueError("需要至少两种细胞类型进行插值实验")

        # 选择样本数量最多的两种细胞类型
        cell_type_counts = adata.obs[cell_type_key].value_counts()
        cell_type_A = cell_type_counts.index[0]
        cell_type_B = cell_type_counts.index[1]

    print(f"选择的细胞类型: {cell_type_A} -> {cell_type_B}")

    # 4. 选择代表细胞 (每个类型的质心)
    print("选择代表细胞...")

    def get_centroid_cell(adata, cell_type):
        # 获取该类型的所有细胞
        cell_idx = np.where((adata.obs[cell_type_key] == cell_type) & (adata.obs['celltype_p'] == cell_type))[0]
        cells = adata[cell_idx]

        # 计算质心
        if scipy.sparse.issparse(cells.X):
            centroid = np.array(cells.X.mean(axis=0)).flatten()
        else:
            centroid = cells.X.mean(axis=0)

        # 找到最接近质心的细胞
        dists = np.linalg.norm(cells.X - centroid, axis=1)
        closest_idx = np.argmin(dists)
        return cells[closest_idx].obs.index[0], cells[closest_idx].obsm['X_latent'][0]

    cell_id_A, z_A = get_centroid_cell(adata, cell_type_A)
    cell_id_B, z_B = get_centroid_cell(adata, cell_type_B)

    print(f"代表细胞: {cell_id_A} ({cell_type_A}) -> {cell_id_B} ({cell_type_B})")

    # 5. 创建线性插值路径
    print("创建插值路径...")
    alphas = np.linspace(0, 1, n_points)
    z_interp = np.array([(1 - alpha) * z_A + alpha * z_B for alpha in alphas])

    # 解码插值点
    print("解码插值点...")
    z_interp_t = torch.tensor(z_interp.astype(np.float32), device=device)
    print('shape_z:',z_interp_t.shape)
    X_interp = decoder(z_interp_t).detach().cpu().numpy()

    if draw_umap:
        # 将插值点投影到原始UMAP空间
        print("投影到参考UMAP空间...")
        # 使用kNN回归将插值点投影到UMAP空间
        knn_reg = KNeighborsRegressor(n_neighbors=k)
        knn_reg.fit(adata.X, adata.obsm['X_umap'])

        # 预测插值点的UMAP坐标
        umap_interp = knn_reg.predict(X_interp)

        # 8. 可视化插值轨迹
        print("可视化插值轨迹...")
        plt.figure(figsize=figsize)

        # 绘制原始数据的UMAP
        sc.pl.umap(adata, color=cell_type_key, show=False, title='Interpolation Path in Reference UMAP')

        # 绘制插值路径
        plt.plot(umap_interp[:, 0], umap_interp[:, 1], 'r-', linewidth=2, alpha=0.7, label='Interpolation Path')
        plt.scatter(umap_interp[:, 0], umap_interp[:, 1], c=alphas, cmap='viridis', s=100, edgecolor='k')

        # 标记起点和终点
        plt.scatter(umap_interp[0, 0], umap_interp[0, 1], s=200, marker='*', c='gold', edgecolor='k',
                    label=f'Start: {cell_type_A}')
        plt.scatter(umap_interp[-1, 0], umap_interp[-1, 1], s=200, marker='*', c='red', edgecolor='k',
                    label=f'End: {cell_type_B}')

        plt.legend()
        plt.title(f'Interpolation from {cell_type_A} to {cell_type_B}')
        plt.tight_layout()
        plt.show()

    # 9. 细胞类型概率分析 (如果有预训练的分类器)
    if draw_umap:
        results = {
            'z_interp': z_interp,
            'X_interp': X_interp,
            'umap_interp': umap_interp,
            'alphas': alphas,
            'cell_type_A': cell_type_A,
            'cell_type_B': cell_type_B,
            'cell_id_A': cell_id_A,
            'cell_id_B': cell_id_B
        }
    else:
        results = {
            'z_interp': z_interp,
            'X_interp': X_interp,
            'alphas': alphas,
            'cell_type_A': cell_type_A,
            'cell_type_B': cell_type_B,
            'cell_id_A': cell_id_A,
            'cell_id_B': cell_id_B
        }

    if 'cell_type_classifier' in adata.uns_keys():
        print("分析细胞类型概率...")
        classifier = adata.uns['cell_type_classifier']
        proba_interp = classifier.predict_proba(X_interp)
        cell_types = classifier.classes_

        # 创建概率变化图
        plt.figure(figsize=(12, 6))

        # 绘制主要细胞类型的概率变化
        for i, ct in enumerate(cell_types):
            if ct in [cell_type_A, cell_type_B] or proba_interp[:, i].max() > 0.2:
                plt.plot(alphas, proba_interp[:, i], label=ct, linewidth=2.5)

        # 标记起点和终点
        plt.axvline(0, color='gold', linestyle='--', alpha=0.7)
        plt.axvline(1, color='red', linestyle='--', alpha=0.7)

        plt.xlabel('Interpolation Position (alpha)')
        plt.ylabel('Cell Type Probability')
        plt.title(f'Cell Type Probability Along Interpolation Path\n{cell_type_A} to {cell_type_B}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

        # 添加概率结果
        results['cell_type_probs'] = proba_interp
        results['cell_type_classes'] = cell_types

        # 创建热图显示所有细胞类型的概率
        plt.figure(figsize=(10, 8))
        sns.heatmap(proba_interp.T, cmap='viridis',
                    xticklabels=[f"{a:.2f}" for a in alphas],
                    yticklabels=cell_types)
        plt.title('Cell Type Probability Heatmap')
        plt.xlabel('Interpolation Position (alpha)')
        plt.ylabel('Cell Type')
        plt.tight_layout()
        plt.show()

    if issparse(adata.X):
        adata = adata.copy()
        adata.X = adata.X.toarray()
    # 10. 关键基因表达动态分析
    print("分析关键基因表达动态...")
    # 选择细胞类型标记基因
    marker_genes = []
    if cell_type_A in adata.uns.get('marker_genes', {}):
        marker_genes.extend(adata.uns['marker_genes'][cell_type_A][:3])
    if cell_type_B in adata.uns.get('marker_genes', {}):
        marker_genes.extend(adata.uns['marker_genes'][cell_type_B][:3])

    if marker_genes:
        # 确保基因在数据中
        marker_genes = [g for g in marker_genes if g in adata.var_names]

        if marker_genes:
            n_genes = len(marker_genes)
            fig, axes = plt.subplots(n_genes, 1, figsize=(10, 3 * n_genes), sharex=True)

            # 获取基因索引
            gene_indices = [adata.var_names.tolist().index(g) for g in marker_genes]

            for i, (gene, idx) in enumerate(zip(marker_genes, gene_indices)):
                ax = axes[i] if n_genes > 1 else axes
                gene_expr = X_interp[:, idx]

                # 绘制表达变化
                ax.plot(alphas, gene_expr, 'o-', label=gene, linewidth=2)

                # 添加原始细胞的表达作为参考
                expr_A = adata[cell_id_A].X.flatten()[idx]
                expr_B = adata[cell_id_B].X.flatten()[idx]
                ax.axhline(expr_A, color='gold', linestyle='--', alpha=0.7)
                ax.axhline(expr_B, color='red', linestyle='--', alpha=0.7)

                ax.set_ylabel('Expression Level')
                ax.set_title(f'{gene} Expression Along Interpolation')
                ax.legend()
                ax.grid(alpha=0.2)

            plt.xlabel('Interpolation Position (alpha)')
            plt.tight_layout()
            plt.show()

            results['marker_genes'] = marker_genes
            results['gene_expression'] = {gene: X_interp[:, idx] for gene, idx in zip(marker_genes, gene_indices)}

    print("插值实验完成!")
    return results


def evaluate_perturbation_sensitivity(decoder, adata, latent, recon, device, n_perturbations=100, noise_std=0.01,
                                      save_path='./'):
    """
    评估自动编码器对潜在空间微小扰动的敏感性

    参数:
    autoencoder: 训练好的自动编码器模型
    data_loader: 数据加载器 (包含单个批次)
    device: 计算设备 ('cuda' 或 'cpu')
    n_perturbations: 扰动次数
    noise_std: 高斯噪声标准差
    """
    results = {
        'gene_diff': [],
        'latent_dist': [],
        'expr_dist': []
    }

    if issparse(adata.X):
        adata = adata.copy()
        adata.X = adata.X.toarray()

    batch_size = 4096
    num_batches = (len(latent) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(latent))
        batch_latent = latent[start_idx:end_idx]
        batch_recon = recon[start_idx:end_idx]
        X = adata.X[start_idx:end_idx]

        # 原始重建
        z = batch_latent
        X_recon = batch_recon

        z = torch.tensor(z.astype(np.float32), device=device)
        X_recon = torch.tensor(X_recon.astype(np.float32), device=device)
        X = torch.tensor(X.astype(np.float32), device=device)

        # 随机选择5个细胞进行可视化
        cell_indices = np.random.choice(len(z), min(5, len(z)), replace=False)

        for idx in range(len(z)):
            # 获取当前细胞的潜在向量和重建表达
            z_orig = z[idx]
            X_recon_orig = X_recon[idx]

            for _ in range(n_perturbations):
                # 生成微小高斯噪声
                epsilon = torch.randn_like(z_orig) * noise_std
                z_perturbed = z_orig + epsilon

                # 解码扰动后的潜在向量
                X_perturbed_recon = decoder(z_perturbed.unsqueeze(0)).detach().squeeze()

                # 计算指标
                gene_diff = (X_recon_orig - X_perturbed_recon).abs().mean().item()
                latent_dist = torch.norm(epsilon).item()
                expr_dist = torch.norm(X_recon_orig - X_perturbed_recon).item()

                # 存储结果
                results['gene_diff'].append(gene_diff)
                results['latent_dist'].append(latent_dist)
                results['expr_dist'].append(expr_dist)

            # 可视化示例细胞
            if idx in cell_indices:
                visualize_perturbation(X[idx].cpu(),
                                       X_recon_orig.cpu(),
                                       X_perturbed_recon.cpu(),
                                       gene_diff,
                                       save_path)

    # 分析结果
    analyze_results(results, save_path)
    return results


def visualize_perturbation(original, recon_orig, recon_perturbed, gene_diff, save_path='./'):
    """可视化原始表达、重建表达和扰动重建"""
    plt.figure(figsize=(15, 4))

    # 原始表达 vs 原始重建
    plt.subplot(131)
    plt.scatter(original.numpy(), recon_orig.numpy(), alpha=0.5)
    plt.plot([0, original.max()], [0, original.max()], 'r--')
    plt.title('Original vs Reconstruction')
    plt.xlabel('Original Expression')
    plt.ylabel('Reconstructed')

    # 原始重建 vs 扰动重建
    plt.subplot(132)
    plt.scatter(recon_orig.numpy(), recon_perturbed.numpy(), alpha=0.5)
    lim_max = max(recon_orig.max(), recon_perturbed.max())
    plt.plot([0, lim_max], [0, lim_max], 'r--')
    plt.title(f'Recon vs Perturbed (Avg Diff: {gene_diff:.4f})')
    plt.xlabel('Original Reconstruction')
    plt.ylabel('Perturbed Reconstruction')

    # 差异分布
    plt.subplot(133)
    diff = (recon_orig - recon_perturbed).abs().numpy()
    plt.hist(diff[diff > 0], bins=50, log=True)
    plt.title('Expression Difference Distribution')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency (log)')

    plt.tight_layout()
    plt.savefig(f'{save_path}perturbation_visualization_{np.random.randint(100)}.png')
    plt.close()


def analyze_results(results, save_path='./'):
    """分析扰动实验结果"""
    print("\n===== 扰动实验结果分析 =====")
    print(f"测试细胞数量: {len(results['gene_diff']) // len(results['gene_diff'])}")
    print(f"总扰动次数: {len(results['gene_diff'])}")

    # 计算关键指标
    avg_gene_diff = np.mean(results['gene_diff'])
    avg_expr_dist = np.mean(results['expr_dist'])
    max_diff = np.max(results['gene_diff'])

    print(f"\n平均基因表达差异: {avg_gene_diff:.5f}")
    print(f"平均表达谱欧氏距离: {avg_expr_dist:.5f}")
    print(f"最大单基因差异: {max_diff:.5f}")

    # 潜在空间距离 vs 表达空间距离
    plt.figure(figsize=(10, 6))
    plt.scatter(results['latent_dist'], results['expr_dist'], alpha=0.3)
    plt.xlabel('Latent Space Distance (||ε||)')
    plt.ylabel('Expression Space Distance')
    plt.title('Perturbation Sensitivity')
    plt.savefig(f'{save_path}perturbation_sensitivity.png')
    plt.close()

    # 检查异常扰动
    threshold = np.percentile(results['gene_diff'], 99)
    outliers = [d for d in results['gene_diff'] if d > threshold]
    print(f"\n异常扰动比例 (>99%分位数): {len(outliers) / len(results['gene_diff']) * 100:.2f}%")

    # 建议评估标准
    print("\n===== 评估建议 =====")
    print("1. 平均基因差异 < 0.01: 优秀 (局部平滑性良好)")
    print("2. 平均基因差异 0.01-0.05: 良好")
    print("3. 平均基因差异 > 0.05: 需要检查潜在空间连续性")
    print(f"4. 异常扰动比例 > 5%: 可能表示潜在空间存在不连续区域")


def evaluate_latent_space_dim(latent_vectors, save_path='./'):
    """
    评估自动编码器潜在空间的维度分布和相关性

    """

    n_cells, latent_dim = latent_vectors.shape

    print(f"评估潜在空间: {n_cells}个细胞, {latent_dim}个维度")

    # 1. 维度分布可视化
    visualize_dimension_distributions(latent_vectors, latent_dim, save_path)

    # 2. 维度相关性分析
    analyze_dimension_correlations(latent_vectors, latent_dim, save_path)

    # 3. 维度统计指标
    analyze_dimension_statistics(latent_vectors, latent_dim, save_path)


def visualize_dimension_distributions(latent_vectors, latent_dim, save_path='./'):
    """可视化潜在空间各维度的分布"""
    print("\n===== 潜在维度分布可视化 =====")

    # 设置绘图布局
    n_cols = 4
    n_rows = int(np.ceil(latent_dim / n_cols))

    plt.figure(figsize=(16, 4 * n_rows))

    # 计算每个维度的统计量
    dim_stats = []

    for i in range(latent_dim):
        dim_data = latent_vectors[:, i]

        # 计算统计指标
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        range_val = max_val - min_val

        # 正态性检验 (KS检验)
        _, ks_p = kstest((dim_data - mean_val) / std_val, 'norm')

        dim_stats.append({
            'dim': i,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'ks_p': ks_p
        })

        # 绘制子图
        plt.subplot(n_rows, n_cols, i + 1)

        # 直方图与KDE曲线
        sns.histplot(dim_data, kde=True, bins=50, stat='density', color='skyblue')

        # 标记统计信息
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
        plt.axvline(mean_val - std_val, color='g', linestyle=':', alpha=0.5)
        plt.axvline(mean_val + std_val, color='g', linestyle=':', alpha=0.5)

        # 设置标题和标签
        plt.title(f'Dim {i} (std={std_val:.3f})')
        plt.xlabel('Value')
        plt.ylabel('Density')

        # 添加图例
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}latent_dim_distributions.png')
    plt.close()

    # 创建小提琴图 (所有维度一起)
    plt.figure(figsize=(14, 8))
    plt.title('Latent Dimension Distributions (Violin Plots)')
    sns.violinplot(data=pd.DataFrame(latent_vectors), inner="quartile", palette="Set3")
    plt.xlabel('Dimension Index')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}latent_dim_violins.png')
    plt.close()

    # 分析维度分布问题
    analyze_distribution_issues(dim_stats, latent_dim)


def analyze_distribution_issues(dim_stats, latent_dim=128):
    """分析潜在维度分布中的问题"""
    print("\n===== 维度分布问题分析 =====")

    # 识别塌缩维度 (标准差接近零)
    collapsed_dims = [d for d in dim_stats if d['std'] < 1e-3]
    print(f"塌缩维度数量 (std < 0.001): {len(collapsed_dims)}")
    if collapsed_dims:
        print(f"塌缩维度索引: {[d['dim'] for d in collapsed_dims]}")

    # 识别低方差维度 (标准差 < 0.1)
    low_var_dims = [d for d in dim_stats if d['std'] < 0.1 and d['std'] >= 1e-3]
    print(f"低方差维度数量 (0.001 <= std < 0.1): {len(low_var_dims)}")

    # 识别离群值严重的维度 (范围 > 10倍标准差)
    outlier_dims = [d for d in dim_stats if d['range'] > 10 * d['std']]
    print(f"高离群值维度数量 (range > 10×std): {len(outlier_dims)}")

    # 识别非正态分布维度 (KS检验p<0.05)
    non_normal_dims = [d for d in dim_stats if d['ks_p'] < 0.05]
    print(f"非正态分布维度数量 (KS p<0.05): {len(non_normal_dims)}")

    # 建议
    print("\n===== 改进建议 =====")
    if collapsed_dims:
        print("警告: 存在塌缩维度! 建议:")
        print("1. 检查网络激活函数 (如使用ReLU可能导致神经元死亡)")
        print("2. 增加权重正则化 (如L2正则化)")
        print("3. 调整学习率或优化器")

    if len(low_var_dims) > latent_dim * 0.5:
        print(f"警告: 超过50%的维度方差较低! 建议:")
        print("1. 考虑降低潜在空间维度")
        print("2. 增加重构损失的权重")
        print("3. 在网络中使用Batch Normalization")

    if len(outlier_dims) > 0:
        print("警告: 存在高离群值维度! 建议:")
        print("1. 在潜在空间中使用Layer Normalization")
        print("2. 使用梯度裁剪防止梯度爆炸")
        print("3. 检查输入数据是否需要更严格的归一化")


def analyze_dimension_correlations(latent_vectors, latent_dim, save_path='./'):
    """分析潜在维度之间的相关性"""
    print("\n===== 维度相关性分析 =====")

    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(latent_vectors, rowvar=False)

    # 可视化相关系数矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix,
                annot=False,
                fmt=".2f",
                cmap='coolwarm',
                vmin=-1, vmax=1,
                square=True,
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
                cbar_kws={"shrink": 0.8})

    plt.title('Latent Dimension Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{save_path}/latent_dim_correlations.png')
    plt.close()

    # 识别高度相关的维度对
    high_corr_pairs = []
    for i in range(latent_dim):
        for j in range(i + 1, latent_dim):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    print(f"高度相关维度对 (|r| > 0.8) 数量: {len(high_corr_pairs)}")

    # 识别高度相关的维度组
    correlated_groups = find_correlated_dimension_groups(corr_matrix, threshold=0.7)
    print(f"高度相关维度组数量 (r > 0.7): {len(correlated_groups)}")

    # 打印高度相关的维度对
    if high_corr_pairs:
        print("\n高度相关维度对:")
        for i, j, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"维度 {i} 和 {j}: r = {r:.3f}")

    # 打印高度相关的维度组
    if correlated_groups:
        print("\n高度相关维度组:")
        for i, group in enumerate(correlated_groups):
            print(f"组 {i + 1}: {sorted(group)}")

    # 建议
    print("\n===== 改进建议 =====")
    if len(high_corr_pairs) > latent_dim * 0.5:
        print("警告: 过度相关的潜在空间! 建议:")
        print("1. 显著降低潜在空间维度")
        print("2. 增加潜在空间的正则化 (如L1正则化)")
        print("3. 使用解纠缠正则化方法 (如β-VAE)")
    elif high_corr_pairs:
        print("存在冗余维度: 建议:")
        print("1. 考虑移除高度相关的维度")
        print("2. 增加三元损失权重以解耦维度")
        print("3. 在网络中使用Dropout层")


def find_correlated_dimension_groups(corr_matrix, threshold=0.7):
    """识别高度相关的维度组"""
    latent_dim = corr_matrix.shape[0]
    visited = set()
    groups = []

    for i in range(latent_dim):
        if i in visited:
            continue

        # 找到与当前维度高度相关的所有维度
        group = {i}
        queue = [i]

        while queue:
            current = queue.pop(0)
            for j in range(latent_dim):
                if j not in group and corr_matrix[current, j] > threshold:
                    group.add(j)
                    queue.append(j)
                    visited.add(j)

        # 只保留包含2个以上维度的组
        if len(group) > 1:
            groups.append(group)
            visited |= group

    return groups


def analyze_dimension_statistics(latent_vectors, latent_dim, save_path='./'):
    """计算潜在维度的统计指标"""
    print("\n===== 维度统计指标 =====")

    # 计算每个维度的方差
    variances = np.var(latent_vectors, axis=0)

    # 计算方差解释率
    total_variance = np.sum(variances)
    explained_variance_ratio = variances / total_variance

    # 计算累积方差解释率
    sorted_indices = np.argsort(explained_variance_ratio)[::-1]
    sorted_variance_ratio = explained_variance_ratio[sorted_indices]
    cumulative_variance = np.cumsum(sorted_variance_ratio)

    # 可视化方差解释
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.bar(range(latent_dim), sorted_variance_ratio)
    plt.title('Sorted Explained Variance Ratio')
    plt.xlabel('Dimension Index (Sorted)')
    plt.ylabel('Variance Ratio')

    plt.subplot(122)
    plt.plot(range(latent_dim), cumulative_variance, 'o-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Cumulative Variance Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_path}/latent_dim_variance.png')
    plt.close()

    # 打印关键指标
    dim_90 = np.argmax(cumulative_variance >= 0.90) + 1
    dim_95 = np.argmax(cumulative_variance >= 0.95) + 1

    print(f"总潜在维度: {latent_dim}")
    print(f"解释90%方差所需维度数: {dim_90}")
    print(f"解释95%方差所需维度数: {dim_95}")
    print(f"最信息丰富的5个维度: {sorted_indices[:5]}")
    print(f"最不活跃的5个维度: {np.argsort(variances)[:5]}")

    # 建议
    if dim_95 < latent_dim * 0.7:
        print("\n===== 改进建议 =====")
        print(f"警告: 潜在空间维度可能过高! 建议:")
        print(f"1. 考虑将维度从{latent_dim}降低到{dim_95}左右")
        print("2. 添加潜在空间瓶颈正则化")
        print("3. 使用PCA分析确定最优维度数")


# generator quality control
def compute_mmd(X, Y, gamma=None):
    """
    计算两个分布X和Y之间的最大均值差异(MMD)

    参数:
        X: 样本集1 (n_samples, n_features)
        Y: 样本集2 (m_samples, n_features)
        gamma: RBF核参数，如果为None则自动计算

    返回:
        mmd_value: MMD值
    """
    if gamma is None:
        # 使用中位数启发式方法设置gamma
        XX = np.vstack([X, Y])
        pairwise_dists = np.sqrt(np.sum((XX[:, None, :] - XX[None, :, :]) ** 2, axis=-1))
        gamma = 1.0 / (2 * (np.median(pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)])) ** 2)
        if gamma <= 0:
            gamma = 1.0  # 避免gamma为0或负值

    # 计算核矩阵
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    # 计算MMD
    n = X.shape[0]
    m = Y.shape[0]

    mmd_value = (np.sum(K_XX) / (n * (n - 1)) +
                 np.sum(K_YY) / (m * (m - 1)) -
                 2 * np.sum(K_XY) / (n * m))

    return mmd_value


def domain_classifier_evaluation(gen_emb, real_emb, test_size=0.3, print_result=False):
    """
    训练分类器区分生成嵌入和真实嵌入

    参数:
        gen_emb: 生成嵌入 (n_samples, n_features)
        real_emb: 真实嵌入 (m_samples, n_features)
        test_size: 测试集比例

    返回:
        test_acc: 测试集准确率
        auroc: AUROC值
    """
    # 创建标签：生成嵌入为0，真实嵌入为1
    X = np.vstack([gen_emb, real_emb])
    y = np.zeros(len(gen_emb) + len(real_emb))
    y[len(gen_emb):] = 1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    # 训练分类器
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)

    if print_result:
        print(classification_report(y_test, y_pred))

    return test_acc, auroc


def evaluate_celltype_classifier(adata_smartseq, adata_g, celltype_key='cell_type',
                                 test_size=0.2, random_state=42):
    """
    评估生成嵌入在目标域(Smart-seq2)的细胞类型判别能力

    参数:
        adata_smartseq: AnnData对象，包含真实的Smart-seq2嵌入和细胞类型标签
        adata_g: AnnData对象，包含生成的Smart-seq2嵌入和细胞类型标签
        celltype_key: 包含细胞类型标签的obs列名
        test_size: 测试集比例
        random_state: 随机种子

    返回:
        results: 包含评估指标的字典
        classifier: 训练好的分类器
        le: 标签编码器
    """
    # 检查输入数据
    assert celltype_key in adata_smartseq.obs.columns, \
        f"celltype_key '{celltype_key}' not found in adata_smartseq.obs"
    assert celltype_key in adata_g.obs.columns, \
        f"celltype_key '{celltype_key}' not found in adata_g.obs"

    # 提取数据和标签
    X_real = adata_smartseq.obsm['latent']
    y_real = adata_smartseq.obs[celltype_key].values

    X_gen = adata_g.obsm['latent']
    y_gen = adata_g.obs[celltype_key].values

    # 确保嵌入维度相同
    assert X_real.shape[1] == X_gen.shape[1], \
        f"嵌入维度不一致: 真实嵌入维度{X_real.shape[1]}, 生成嵌入维度{X_gen.shape[1]}"

    # 编码细胞类型标签
    le = LabelEncoder()
    y_real_encoded = le.fit_transform(y_real)
    y_gen_encoded = le.transform(y_gen)  # 使用相同的编码器

    # 划分真实数据的训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real_encoded, test_size=test_size,
        random_state=random_state, stratify=y_real_encoded
    )

    # 训练分类器
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced'  # 处理类别不平衡
    )
    classifier.fit(X_train, y_train)

    # 评估在真实测试集上的性能
    y_pred_real = classifier.predict(X_test)
    real_accuracy = accuracy_score(y_test, y_pred_real)
    real_f1 = f1_score(y_test, y_pred_real, average='weighted')

    # 评估在生成嵌入上的性能
    y_pred_gen = classifier.predict(X_gen)
    gen_accuracy = accuracy_score(y_gen_encoded, y_pred_gen)
    gen_f1 = f1_score(y_gen_encoded, y_pred_gen, average='weighted')

    # 计算每个细胞类型的准确率
    cell_types = le.classes_
    celltype_acc_real = {}
    celltype_acc_gen = {}

    for celltype in cell_types:
        # 获取当前细胞类型的索引
        idx_real = (y_real == celltype)
        idx_gen = (y_gen == celltype)

        # 检查是否有该细胞类型的样本
        if np.sum(idx_real) > 0 and np.sum(idx_gen) > 0:
            # 真实数据中该细胞类型的准确率
            y_cell_real = y_real_encoded[idx_real]
            y_pred_cell_real = classifier.predict(X_real[idx_real])
            celltype_acc_real[celltype] = accuracy_score(y_cell_real, y_pred_cell_real)

            # 生成数据中该细胞类型的准确率
            y_cell_gen = y_gen_encoded[idx_gen]
            y_pred_cell_gen = classifier.predict(X_gen[idx_gen])
            celltype_acc_gen[celltype] = accuracy_score(y_cell_gen, y_pred_cell_gen)

    # 创建结果字典
    results = {
        'real_test_accuracy': real_accuracy,
        'real_test_f1': real_f1,
        'gen_accuracy': gen_accuracy,
        'gen_f1': gen_f1,
        'accuracy_difference': real_accuracy - gen_accuracy,
        'celltype_acc_real': celltype_acc_real,
        'celltype_acc_gen': celltype_acc_gen,
        'confusion_matrix_real': confusion_matrix(y_test, y_pred_real),
        'confusion_matrix_gen': confusion_matrix(y_gen_encoded, y_pred_gen),
        'class_names': le.classes_
    }

    return results, classifier, le


def plot_celltype_evaluation(results, figsize=(14, 6)):
    """可视化细胞类型评估结果"""
    # 创建图形
    plt.figure(figsize=figsize)

    # 1. 整体准确率比较
    plt.subplot(121)
    metrics = ['Real Test Set Accuracy', 'Generative Embedding Accuracy']
    values = [results['real_test_accuracy'], results['gen_accuracy']]

    plt.bar(metrics, values, color=['blue', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('Comparison of cell type classification accuracy')
    plt.ylim(0, 1.05)

    # 在柱子上添加数值标签
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

    # 2. 每个细胞类型的准确率比较
    plt.subplot(122)
    cell_types = list(results['celltype_acc_real'].keys())
    real_acc = [results['celltype_acc_real'][ct] for ct in cell_types]
    gen_acc = [results['celltype_acc_gen'][ct] for ct in cell_types]

    x = np.arange(len(cell_types))
    width = 0.35

    plt.bar(x - width / 2, real_acc, width, label='Real data', color='blue')
    plt.bar(x + width / 2, gen_acc, width, label='Generate data', color='orange')

    plt.xticks(x, cell_types, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Classification accuracy of each cell type')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()

    # 添加整体标题
    plt.suptitle(
        f"Generate an embedded cell type classification assessment (Overall difference: {results['accuracy_difference']:.3f})",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 3. 绘制混淆矩阵
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # 真实测试集的混淆矩阵
    sns.heatmap(results['confusion_matrix_real'],
                annot=True, fmt='d', cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=ax[0])
    ax[0].set_title('Real data confusion matrix')
    ax[0].set_xlabel('Prediction label')
    ax[0].set_ylabel('Authentic labels')

    # 生成嵌入的混淆矩阵
    sns.heatmap(results['confusion_matrix_gen'],
                annot=True, fmt='d', cmap='Oranges',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=ax[1])
    ax[1].set_title('Generate data confusion matrix')
    ax[1].set_xlabel('Prediction label')

    plt.tight_layout()
    plt.show()

    return fig


def print_evaluation_summary(results):
    """打印评估结果摘要"""
    print("\n" + "=" * 60)
    print("细胞类型分类器评估摘要")
    print("=" * 60)
    print(f"真实测试集准确率: {results['real_test_accuracy']:.4f}")
    print(f"真实测试集F1分数: {results['real_test_f1']:.4f}")
    print(f"生成嵌入准确率: {results['gen_accuracy']:.4f}")
    print(f"生成嵌入F1分数: {results['gen_f1']:.4f}")
    print(f"准确率差异(真实-生成): {results['accuracy_difference']:.4f}")

    print("\n各细胞类型准确率:")
    print(f"{'细胞类型':<20}{'真实测试集':<15}{'生成嵌入':<15}{'差异':<10}")
    for ct in results['celltype_acc_real']:
        real_acc = results['celltype_acc_real'][ct]
        gen_acc = results['celltype_acc_gen'][ct]
        diff = real_acc - gen_acc
        print(f"{ct:<20}{real_acc:.4f}{'':<10}{gen_acc:.4f}{'':<10}{diff:.4f}")


def nearest_neighbor_distance_evaluation(adata_g, adata_smartseq, k=15,
                                         min_cells_per_type=5, n_jobs=-1):
    """
    评估生成嵌入的最近邻距离分布，生成嵌入到真实和真实到真实的最近邻距离分布

    参数:
        adata_g: 生成的Smart-seq2嵌入数据 (包含obsm['latent']和细胞类型)
        adata_smartseq: 真实的Smart-seq2数据 (包含obsm['latent']和细胞类型)
        k: 最近邻数量
        min_cells_per_type: 最小细胞数量要求
        n_jobs: 并行作业数

    返回:
        results: 包含评估结果的字典
    """
    # 提取嵌入数据
    gen_emb = adata_g.obsm['latent']
    real_emb = adata_smartseq.obsm['latent']

    # 提取细胞类型标签
    if 'celltype' in adata_g.obs.columns:
        gen_celltypes = adata_g.obs['celltype'].values
        real_celltypes = adata_smartseq.obs['celltype'].values
        celltype_available = True
    else:
        celltype_available = False

    # 构建最近邻搜索器
    # 在真实Smart-seq空间中的搜索器 (用于计算到真实点的距离)
    nn_real = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
    nn_real.fit(real_emb)

    # 在生成嵌入空间中的搜索器 (用于计算生成点间的距离)
    nn_gen = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs)  # k+1 包含自身
    nn_gen.fit(gen_emb)

    # 计算每个生成点到真实点的距离
    distances_to_real, _ = nn_real.kneighbors(gen_emb)
    d_gen_to_real = distances_to_real.mean(axis=1)

    # 计算每个生成点到其他生成点的距离
    distances_to_gen, _ = nn_gen.kneighbors(gen_emb)
    # 排除自身 (第一个邻居)
    d_gen_to_gen = distances_to_gen[:, 1:].mean(axis=1)  # 取第2到k+1个邻居

    # 计算距离比率
    distance_ratios = d_gen_to_real / d_gen_to_gen

    # 计算整体统计量
    mean_d_to_real = np.mean(d_gen_to_real)
    mean_d_to_gen = np.mean(d_gen_to_gen)
    mean_ratio = np.mean(distance_ratios)
    median_ratio = np.median(distance_ratios)

    # 执行统计检验
    t_stat, t_p = ttest_rel(d_gen_to_real, d_gen_to_gen)
    w_stat, w_p = wilcoxon(d_gen_to_real, d_gen_to_gen)

    # 按细胞类型分组结果
    celltype_results = {}
    if celltype_available:
        unique_celltypes = np.unique(gen_celltypes)

        for celltype in unique_celltypes:
            # 获取当前细胞类型的索引
            idx = gen_celltypes == celltype
            if np.sum(idx) < min_cells_per_type:
                continue

            # 计算当前细胞类型的统计量
            ct_d_to_real = d_gen_to_real[idx]
            ct_d_to_gen = d_gen_to_gen[idx]
            ct_ratios = distance_ratios[idx]

            # 计算平均值
            ct_mean_d_to_real = np.mean(ct_d_to_real)
            ct_mean_d_to_gen = np.mean(ct_d_to_gen)
            ct_mean_ratio = np.mean(ct_ratios)

            # 统计检验
            ct_t_stat, ct_t_p = ttest_rel(ct_d_to_real, ct_d_to_gen)
            ct_w_stat, ct_w_p = wilcoxon(ct_d_to_real, ct_d_to_gen)

            # 存储结果
            celltype_results[celltype] = {
                'd_to_real': ct_d_to_real,
                'd_to_gen': ct_d_to_gen,
                'ratios': ct_ratios,
                'mean_d_to_real': ct_mean_d_to_real,
                'mean_d_to_gen': ct_mean_d_to_gen,
                'mean_ratio': ct_mean_ratio,
                't_test_p': ct_t_p,
                'wilcoxon_p': ct_w_p
            }

    # 创建结果字典
    results = {
        'd_gen_to_real': d_gen_to_real,
        'd_gen_to_gen': d_gen_to_gen,
        'distance_ratios': distance_ratios,
        'mean_d_to_real': mean_d_to_real,
        'mean_d_to_gen': mean_d_to_gen,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        't_test': {'statistic': t_stat, 'p_value': t_p},
        'wilcoxon': {'statistic': w_stat, 'p_value': w_p},
        'celltype_results': celltype_results if celltype_available else None,
        'celltype_available': celltype_available
    }

    return results


def plot_nn_distance_evaluation(results, figsize=(16, 12)):
    """可视化最近邻距离评估结果"""
    plt.figure(figsize=figsize)

    # 1. 距离分布直方图
    plt.subplot(221)
    sns.histplot(results['d_gen_to_real'], color='blue', alpha=0.5, label='Distance to real points', kde=True)
    sns.histplot(results['d_gen_to_gen'], color='orange', alpha=0.5, label='Distance to generated points', kde=True)
    plt.axvline(results['mean_d_to_real'], color='blue', linestyle='--',
                label=f'Mean to real points: {results["mean_d_to_real"]:.4f}')
    plt.axvline(results['mean_d_to_gen'], color='orange', linestyle='--',
                label=f'Mean to generated points: {results["mean_d_to_gen"]:.4f}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Nearest Neighbor Distance Distribution')
    plt.legend()

    # 2. 距离比率分布
    plt.subplot(222)
    sns.histplot(results['distance_ratios'], bins=30, kde=True)
    plt.axvline(1.0, color='r', linestyle='--', label='Ideal value (1.0)')
    plt.axvline(results['mean_ratio'], color='g', linestyle='--',
                label=f'Mean: {results["mean_ratio"]:.3f}')
    plt.xlabel('Distance Ratio (d_gen_to_real / d_gen_to_gen)')
    plt.ylabel('Frequency')
    plt.title('Distance Ratio Distribution')
    plt.legend()

    # 3. 散点图：到真实点距离 vs 到生成点距离
    plt.subplot(223)
    plt.scatter(results['d_gen_to_gen'], results['d_gen_to_real'],
                alpha=0.5, s=10)
    plt.plot([0, max(results['d_gen_to_gen'])],
             [0, max(results['d_gen_to_gen'])],
             'r--', label='y=x')
    plt.xlabel('Average distance to generated points')
    plt.ylabel('Average distance to real points')
    plt.title('Distance Comparison Scatter Plot')
    plt.legend()

    # 4. 箱线图：按距离比率分组
    plt.subplot(224)
    # 创建分组 (0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5+)
    ratio_groups = []
    group_labels = []

    for i in range(len(results['distance_ratios'])):
        ratio = results['distance_ratios'][i]
        d_real = results['d_gen_to_real'][i]
        d_gen = results['d_gen_to_gen'][i]

        if ratio < 0.5:
            ratio_groups.append(d_real - d_gen)
            group_labels.append('<0.5')
        elif ratio < 1.0:
            ratio_groups.append(d_real - d_gen)
            group_labels.append('0.5-1.0')
        elif ratio < 1.5:
            ratio_groups.append(d_real - d_gen)
            group_labels.append('1.0-1.5')
        else:
            ratio_groups.append(d_real - d_gen)
            group_labels.append('>1.5')

    sns.boxplot(x=group_labels, y=ratio_groups)
    plt.xlabel('Distance Ratio Group')
    plt.ylabel('Distance Difference (d_real - d_gen)')
    plt.title('Distance Difference Across Ratio Groups')

    plt.tight_layout()
    plt.show()

    # 5. 按细胞类型分组结果 (如果可用)
    if results['celltype_available'] and results['celltype_results']:
        celltypes = list(results['celltype_results'].keys())
        mean_ratios = [results['celltype_results'][ct]['mean_ratio'] for ct in celltypes]
        mean_d_to_real = [results['celltype_results'][ct]['mean_d_to_real'] for ct in celltypes]
        mean_d_to_gen = [results['celltype_results'][ct]['mean_d_to_gen'] for ct in celltypes]

        # 按平均比率排序
        sorted_idx = np.argsort(mean_ratios)
        celltypes_sorted = [celltypes[i] for i in sorted_idx]
        mean_ratios_sorted = [mean_ratios[i] for i in sorted_idx]
        mean_d_to_real_sorted = [mean_d_to_real[i] for i in sorted_idx]
        mean_d_to_gen_sorted = [mean_d_to_gen[i] for i in sorted_idx]

        # 创建分组柱状图
        plt.figure(figsize=(14, 8))
        x = np.arange(len(celltypes_sorted))
        width = 0.35

        plt.bar(x - width / 2, mean_d_to_real_sorted, width, label='Distance to real points', color='blue')
        plt.bar(x + width / 2, mean_d_to_gen_sorted, width, label='Distance to generated points', color='orange')

        plt.xticks(x, celltypes_sorted, rotation=45, ha='right')
        plt.ylabel('Average Distance')
        plt.title('Comparison of Nearest Neighbor Distances by Cell Type')
        plt.legend()

        # 在顶部添加比率值
        for i, v in enumerate(mean_ratios_sorted):
            plt.text(i, max(mean_d_to_real_sorted[i], mean_d_to_gen_sorted[i]) + 0.05,
                     f'Ratio: {v:.2f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        # 创建比率热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(np.array(mean_ratios_sorted).reshape(1, -1),
                    annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=celltypes_sorted, yticklabels=["Distance Ratio"])
        plt.title('Distance Ratios by Cell Type (d_gen_to_real / d_gen_to_gen)')
        plt.tight_layout()
        plt.show()


def print_nn_distance_summary(results):
    """打印最近邻距离评估摘要"""
    print("\n" + "=" * 60)
    print("最近邻距离评估摘要")
    print("=" * 60)
    print(f"到真实点的平均距离: {results['mean_d_to_real']:.4f}")
    print(f"到生成点的平均距离: {results['mean_d_to_gen']:.4f}")
    print(f"距离比率均值 (d_real/d_gen): {results['mean_ratio']:.4f}")
    print(f"距离比率中位数: {results['median_ratio']:.4f}")

    # 解释比率
    if results['mean_ratio'] < 0.8:
        interpretation = "警告：生成点过于聚集（可能模式坍塌）"
    elif results['mean_ratio'] < 1.2:
        interpretation = "良好：生成点合理融入真实分布"
    else:
        interpretation = "警告：生成点未充分融入真实分布"

    print(f"\n解释: {interpretation}")

    # 统计检验结果
    print(f"\n配对t检验: t = {results['t_test']['statistic']:.4f}, p = {results['t_test']['p_value']:.4e}")
    print(
        f"Wilcoxon符号秩检验: statistic = {results['wilcoxon']['statistic']:.4f}, p = {results['wilcoxon']['p_value']:.4e}")

    # 按细胞类型的结果
    if results['celltype_available'] and results['celltype_results']:
        print("\n各细胞类型结果:")
        print(f"{'细胞类型':<20}{'d_to_real':<12}{'d_to_gen':<12}{'比率':<10}{'p值'}")
        for ct, res in results['celltype_results'].items():
            p_value = min(res['t_test_p'], res['wilcoxon_p'])
            print(
                f"{ct:<20}{res['mean_d_to_real']:.4f}{'':<4}{res['mean_d_to_gen']:.4f}{'':<4}{res['mean_ratio']:.4f}{'':<4}{p_value:.4e}")


def visualize_embeddings(adata_smartseq, adata_g, adata_10x=None, adata_cycle=None,
                         celltype_key='cell_type', n_neighbors=15, min_dist=0.1,
                         random_state=42, point_size=10, alpha=0.6):
    """
    可视化嵌入空间：评估生成嵌入的质量和一致性

    参数:
        adata_smartseq: 真实Smart-seq2数据 (包含obsm['latent']和细胞类型)
        adata_g: 生成的Smart-seq2嵌入数据
        adata_10x: 原始10x嵌入数据 (可选)
        adata_cycle: 循环重建的10x嵌入数据 (可选)
        celltype_key: 细胞类型标签的obs列名
        n_neighbors: UMAP的邻居数量参数
        min_dist: UMAP的最小距离参数
        random_state: 随机种子
        point_size: 散点大小
        alpha: 透明度

    返回:
        umap_result: UMAP降维结果
    """
    # 收集所有嵌入数据
    embeddings = []
    labels = []
    sources = []

    # 添加真实Smart-seq嵌入
    embeddings.append(adata_smartseq.obsm['latent'])
    labels.append(adata_smartseq.obs[celltype_key].values)
    sources.append(['Real Smart-seq'] * len(adata_smartseq))

    # 添加生成的Smart-seq嵌入
    embeddings.append(adata_g.obsm['latent'])
    labels.append(adata_g.obs[celltype_key].values)
    sources.append(['Generated Smart-seq'] * len(adata_g))

    # 添加原始10x嵌入 (如果提供)
    if adata_10x is not None:
        embeddings.append(adata_10x.obsm['latent'])
        labels.append(adata_10x.obs[celltype_key].values)
        sources.append(['Original 10x'] * len(adata_10x))

    # 添加循环重建的10x嵌入 (如果提供)
    if adata_cycle is not None:
        embeddings.append(adata_cycle.obsm['latent'])
        labels.append(adata_cycle.obs[celltype_key].values)
        sources.append(['Cycled 10x'] * len(adata_cycle))

    # 合并所有数据
    all_embeddings = np.vstack(embeddings)
    all_labels = np.concatenate(labels)
    all_sources = np.concatenate(sources)

    # 创建UMAP转换器
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=2
    )

    # 执行降维
    umap_emb = reducer.fit_transform(all_embeddings)

    # 创建图形
    plt.figure(figsize=(20, 16))

    # 1. 按数据源着色 (来源视图)
    plt.subplot(221)
    palette_source = {
        'Real Smart-seq': 'blue',
        'Generated Smart-seq': 'orange',
        'Original 10x': 'green',
        'Cycled 10x': 'red'
    }

    # 为每个来源绘制点
    for source in np.unique(all_sources):
        idx = all_sources == source
        plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1],
                    c=palette_source[source],
                    s=point_size, alpha=alpha, label=source)

    plt.title('Embedding Space by Source')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Data Source')

    # 2. 按细胞类型着色 (类型视图)
    plt.subplot(222)
    # 获取唯一细胞类型并创建调色板
    unique_celltypes = np.unique(all_labels)
    palette_type = sns.color_palette('husl', n_colors=len(unique_celltypes))
    celltype_palette = {ct: palette_type[i] for i, ct in enumerate(unique_celltypes)}

    # 为每个细胞类型绘制点
    for celltype in unique_celltypes:
        idx = all_labels == celltype
        plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1],
                    c=[celltype_palette[celltype]] * np.sum(idx),
                    s=point_size, alpha=alpha, label=celltype)

    plt.title('Embedding Space by Cell Type')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. 按数据源和细胞类型组合着色 (详细视图)
    plt.subplot(223)
    # 创建组合图例
    legend_elements = []

    # 为每个细胞类型和来源组合创建图例项
    for celltype in unique_celltypes:
        for source in ['Real Smart-seq', 'Generated Smart-seq']:
            # 创建图例项
            color = celltype_palette[celltype]
            marker = 'o' if source == 'Real Smart-seq' else 's'
            label = f"{celltype} ({source})"
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                          markerfacecolor=color, markersize=10, label=label))

    # 绘制点 (仅显示真实和生成)
    for celltype in unique_celltypes:
        # 真实Smart-seq
        idx_real = (all_labels == celltype) & (all_sources == 'Real Smart-seq')
        if np.sum(idx_real) > 0:
            plt.scatter(umap_emb[idx_real, 0], umap_emb[idx_real, 1],
                        c=[celltype_palette[celltype]] * np.sum(idx_real),
                        s=point_size, alpha=alpha, marker='o')

        # 生成Smart-seq
        idx_gen = (all_labels == celltype) & (all_sources == 'Generated Smart-seq')
        if np.sum(idx_gen) > 0:
            plt.scatter(umap_emb[idx_gen, 0], umap_emb[idx_gen, 1],
                        c=[celltype_palette[celltype]] * np.sum(idx_gen),
                        s=point_size, alpha=alpha, marker='s')

    plt.title('Real vs Generated by Cell Type')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. 循环一致性可视化 (如果提供原始和循环10x数据)
    if adata_10x is not None and adata_cycle is not None:
        plt.subplot(224)

        # 提取原始和循环10x的索引
        idx_orig = all_sources == 'Original 10x'
        idx_cycle = all_sources == 'Cycled 10x'

        # 绘制原始10x点
        plt.scatter(umap_emb[idx_orig, 0], umap_emb[idx_orig, 1],
                    c='green', s=point_size, alpha=alpha,
                    label='Original 10x')

        # 绘制循环10x点
        plt.scatter(umap_emb[idx_cycle, 0], umap_emb[idx_cycle, 1],
                    c='red', s=point_size, alpha=alpha,
                    label='Cycled 10x')

        # 计算并显示平均偏差
        orig_points = umap_emb[idx_orig]
        cycle_points = umap_emb[idx_cycle]
        deviations = np.linalg.norm(orig_points - cycle_points, axis=1)
        avg_deviation = np.mean(deviations)

        # 添加箭头显示偏差
        for i in range(min(20, len(orig_points))):  # 最多显示20个箭头
            plt.arrow(orig_points[i, 0], orig_points[i, 1],
                      cycle_points[i, 0] - orig_points[i, 0],
                      cycle_points[i, 1] - orig_points[i, 1],
                      head_width=0.2, head_length=0.3,
                      fc='purple', ec='purple', alpha=0.3)

        plt.title(f'Cycle Consistency (Avg Deviation: {avg_deviation:.2f})')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 5. 轨迹连续性分析 (如果数据有轨迹结构)
    # 假设我们有轨迹标签 (例如伪时间)
    if 'pseudotime' in adata_smartseq.obs.columns:
        # 合并真实和生成的Smart-seq数据
        idx_smart = all_sources == 'Real Smart-seq'
        idx_gen = all_sources == 'Generated Smart-seq'

        # 提取轨迹信息
        pseudotimes = np.concatenate([
            adata_smartseq.obs['pseudotime'].values,
            adata_g.obs['pseudotime'].values
        ])

        # 创建轨迹可视化
        plt.figure(figsize=(12, 10))

        # 绘制真实Smart-seq轨迹
        plt.scatter(umap_emb[idx_smart, 0], umap_emb[idx_smart, 1],
                    c=pseudotimes[:np.sum(idx_smart)],
                    cmap='viridis', s=point_size, alpha=alpha,
                    label='Real Smart-seq')

        # 绘制生成Smart-seq轨迹
        plt.scatter(umap_emb[idx_gen, 0], umap_emb[idx_gen, 1],
                    c=pseudotimes[np.sum(idx_smart):np.sum(idx_smart) + np.sum(idx_gen)],
                    cmap='viridis', s=point_size, alpha=alpha,
                    marker='s', label='Generated Smart-seq')

        plt.colorbar(label='Pseudotime')
        plt.title('Trajectory Continuity Analysis')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.show()

    # 6. 按细胞类型分面可视化 (详细检查)
    unique_celltypes = np.unique(all_labels)
    n_cols = 4
    n_rows = int(np.ceil(len(unique_celltypes) / n_cols))

    plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for i, celltype in enumerate(unique_celltypes):
        plt.subplot(n_rows, n_cols, i + 1)

        # 获取当前细胞类型的索引
        idx_celltype = all_labels == celltype

        # 真实Smart-seq (当前类型)
        idx_real = idx_celltype & (all_sources == 'Real Smart-seq')
        if np.sum(idx_real) > 0:
            plt.scatter(umap_emb[idx_real, 0], umap_emb[idx_real, 1],
                        c='blue', s=point_size, alpha=0.8, label='Real')

        # 生成Smart-seq (当前类型)
        idx_gen = idx_celltype & (all_sources == 'Generated Smart-seq')
        if np.sum(idx_gen) > 0:
            plt.scatter(umap_emb[idx_gen, 0], umap_emb[idx_gen, 1],
                        c='orange', s=point_size, alpha=0.8,
                        marker='s', label='Generated')

        # 计算混合度指标 (重叠区域的比例)
        if np.sum(idx_real) > 0 and np.sum(idx_gen) > 0:
            # 计算凸包重叠 (简化方法)
            from scipy.spatial import ConvexHull

            try:
                # 真实点凸包
                hull_real = ConvexHull(umap_emb[idx_real])
                # 生成点凸包
                hull_gen = ConvexHull(umap_emb[idx_gen])

                # 计算重叠面积 (简化方法)
                overlap = len(set(hull_real.vertices) & set(hull_gen.vertices)) / min(len(hull_real.vertices),
                                                                                      len(hull_gen.vertices))
            except:
                overlap = 0.0

            plt.title(f"{celltype}\nOverlap: {overlap:.2f}")
        else:
            plt.title(celltype)

        plt.legend()

    plt.tight_layout()
    plt.suptitle('Cell Type Specific Analysis', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

    return umap_emb


def nearest_neighbor_variance_analysis(real_embeddings, gen_embeddings, k=15, n_jobs=-1):
    """
    分析最近邻距离分布，检测模式坍塌，生成到生成和真实到真实的最近邻距离分布

    参数:
        real_embeddings: 真实Smart-seq2嵌入 (n_real_cells, n_features)
        gen_embeddings: 生成的Smart-seq2嵌入 (n_gen_cells, n_features)
        k: 最近邻数量
        n_jobs: 并行计算作业数

    返回:
        results: 包含分析结果的字典
    """
    # 1. 计算真实嵌入的最近邻距离分布
    nn_real = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs)  # k+1包含自身
    nn_real.fit(real_embeddings)
    real_distances, _ = nn_real.kneighbors(real_embeddings)
    real_nn_distances = real_distances[:, 1:].mean(axis=1)  # 排除自身

    # 2. 计算生成嵌入的最近邻距离分布
    nn_gen = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs)
    nn_gen.fit(gen_embeddings)
    gen_distances, _ = nn_gen.kneighbors(gen_embeddings)
    gen_nn_distances = gen_distances[:, 1:].mean(axis=1)  # 排除自身

    # 3. 计算统计量
    real_mean = np.mean(real_nn_distances)
    real_std = np.std(real_nn_distances)
    real_var = np.var(real_nn_distances)

    gen_mean = np.mean(gen_nn_distances)
    gen_std = np.std(gen_nn_distances)
    gen_var = np.var(gen_nn_distances)

    # 4. 计算距离比率
    mean_ratio = gen_mean / real_mean
    std_ratio = gen_std / real_std
    var_ratio = gen_var / real_var

    # 5. 统计检验
    # Kolmogorov-Smirnov检验比较分布
    ks_stat, ks_p = ks_2samp(real_nn_distances, gen_nn_distances)

    # T检验比较均值
    t_stat, t_p = ttest_ind(real_nn_distances, gen_nn_distances, equal_var=False)

    # 创建结果字典
    results = {
        'real_nn_distances': real_nn_distances,
        'gen_nn_distances': gen_nn_distances,
        'real_mean': real_mean,
        'real_std': real_std,
        'real_var': real_var,
        'gen_mean': gen_mean,
        'gen_std': gen_std,
        'gen_var': gen_var,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'var_ratio': var_ratio,
        'ks_test': {'statistic': ks_stat, 'p_value': ks_p},
        't_test': {'statistic': t_stat, 'p_value': t_p},
        'k': k
    }

    return results


def plot_nn_variance_analysis(results, figsize=(16, 12)):
    """可视化最近邻距离/方差分析结果"""
    plt.figure(figsize=figsize)

    # 1. 距离分布直方图
    plt.subplot(221)
    sns.histplot(results['real_nn_distances'], color='blue', alpha=0.5,
                 label=f'reality (mean={results["real_mean"]:.4f}, standard deviation={results["real_std"]:.4f})',
                 kde=True)
    sns.histplot(results['gen_nn_distances'], color='orange', alpha=0.5,
                 label=f'generate (mean={results["gen_mean"]:.4f}, standard deviation={results["gen_std"]:.4f})',
                 kde=True)
    plt.axvline(results['real_mean'], color='blue', linestyle='--')
    plt.axvline(results['gen_mean'], color='orange', linestyle='--')
    plt.xlabel('Average nearest neighbor distance')
    plt.ylabel('density')
    plt.title('Nearest neighbor distance distribution')
    plt.legend()

    # 2. 箱线图比较
    plt.subplot(222)
    data = pd.DataFrame({
        'Distance': np.concatenate([results['real_nn_distances'], results['gen_nn_distances']]),
        'Source': ['reality'] * len(results['real_nn_distances']) + ['generate'] * len(results['gen_nn_distances'])
    })
    sns.boxplot(x='Source', y='Distance', data=data)
    plt.title('Comparison of nearest neighbor distance distribution')

    # 添加统计检验结果
    plt.text(0.5, plt.ylim()[1] * 0.9,
             f"KS-test p value: {results['ks_test']['p_value']:.2e}\nT-test p value: {results['t_test']['p_value']:.2e}",
             ha='center')

    # 3. 方差比较
    plt.subplot(223)
    metrics = ['Mean', 'Standard Deviation', 'Variance']
    real_values = [results['real_mean'], results['real_std'], results['real_var']]
    gen_values = [results['gen_mean'], results['gen_std'], results['gen_var']]
    ratios = [results['mean_ratio'], results['std_ratio'], results['var_ratio']]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, real_values, width, label='reality', color='blue')
    plt.bar(x + width / 2, gen_values, width, label='generate', color='orange')

    # 添加比率值
    for i, r in enumerate(ratios):
        plt.text(x[i], max(real_values[i], gen_values[i]) * 1.05, f'rate: {r:.2f}', ha='center')

    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.title('Statistics comparison')
    plt.legend()

    # 4. 模式坍塌检测指标
    plt.subplot(224)
    collapse_metrics = ['Mean rate', 'Standard Deviation rate', 'Variance rate']
    values = [results['mean_ratio'], results['std_ratio'], results['var_ratio']]

    plt.bar(collapse_metrics, values, color=['blue', 'green', 'red'])
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7)

    # 添加解释文本
    # for i, v in enumerate(values):
    #     if v < 0.8:
    #         plt.text(i, v - 0.05, "risk of model collapse", ha='center', color='red')
    #     elif v < 0.95:
    #         plt.text(i, v + 0.05, "slightly gathered", ha='center', color='orange')
    #     else:
    #         plt.text(i, v + 0.05, "normal", ha='center', color='green')

    plt.ylim(0, max(values) * 1.2)
    plt.ylabel('rate (generate/reality)')
    plt.title('Model collapse risk indicators')

    plt.tight_layout()
    plt.show()

    # 5. 距离分布散点图 (可选)
    plt.figure(figsize=(10, 8))
    plt.scatter(results['real_nn_distances'], results['gen_nn_distances'],
                alpha=0.5, s=20)
    plt.plot([0, max(results['real_nn_distances'])],
             [0, max(results['real_nn_distances'])],
             'r--', label='y=x')
    plt.xlabel('reality nearest neighbor distance')
    plt.ylabel('generate nearest neighbor distance')
    plt.title('Nearest neighbor distance correlation')
    plt.legend()
    plt.show()

    # 6. 累积分布函数比较
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(results['real_nn_distances'], label='reality', color='blue')
    sns.ecdfplot(results['gen_nn_distances'], label='generate', color='orange')
    plt.xlabel('Average nearest neighbor distance')
    plt.ylabel('Cumulative probability')
    plt.title('Comparison of cumulative distribution functions')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_nn_variance_summary(results):
    """打印最近邻距离/方差分析摘要"""
    print("\n" + "=" * 60)
    print("最近邻距离/方差分析摘要")
    print("=" * 60)
    print(f"使用 k={results['k']} 最近邻")
    print(
        f"真实嵌入 - 均值距离: {results['real_mean']:.4f}, 标准差: {results['real_std']:.4f}, 方差: {results['real_var']:.4f}")
    print(
        f"生成嵌入 - 均值距离: {results['gen_mean']:.4f}, 标准差: {results['gen_std']:.4f}, 方差: {results['gen_var']:.4f}")
    print(f"均值比率 (生成/真实): {results['mean_ratio']:.4f}")
    print(f"标准差比率 (生成/真实): {results['std_ratio']:.4f}")
    print(f"方差比率 (生成/真实): {results['var_ratio']:.4f}")

    # 解释结果
    print("\n模式坍塌风险评估:")
    if results['mean_ratio'] < 0.8:
        print("  ⚠️ 警告: 均值比率 < 0.8 - 生成点过于集中，可能模式坍塌")
    elif results['mean_ratio'] < 0.95:
        print("  ⚠️ 注意: 均值比率 < 0.95 - 生成点轻微聚集")
    else:
        print("  ✅ 良好: 均值比率正常")

    if results['var_ratio'] < 0.5:
        print("  ⚠️ 警告: 方差比率 < 0.5 - 生成点多样性不足")
    elif results['var_ratio'] < 0.8:
        print("  ⚠️ 注意: 方差比率 < 0.8 - 生成点多样性降低")
    else:
        print("  ✅ 良好: 方差比率正常")

    # 统计检验结果
    print(f"\n统计检验:")
    print(
        f"  Kolmogorov-Smirnov检验: D = {results['ks_test']['statistic']:.4f}, p = {results['ks_test']['p_value']:.4e}")
    print(f"  T检验 (均值差异): t = {results['t_test']['statistic']:.4f}, p = {results['t_test']['p_value']:.4e}")

    # 最终评估
    if results['mean_ratio'] < 0.8 or results['var_ratio'] < 0.5:
        print("\n结论: ⚠️ 检测到模式坍塌风险 - 生成点过于集中且多样性不足")
    elif results['mean_ratio'] < 0.95 or results['var_ratio'] < 0.8:
        print("\n结论: ⚠️ 轻微聚集 - 生成点分布比真实点更集中")
    else:
        print("\n结论: ✅ 良好 - 生成点分布与真实点相似")


def cluster_entropy_analysis(real_embeddings, gen_embeddings,
                             resolution_range=(0.1, 2.0), n_steps=10,
                             real_celltypes=None, gen_celltypes=None,
                             random_state=42):
    """
    通过聚类熵分析评估生成嵌入的多样性

    参数:
        real_embeddings: 真实Smart-seq2嵌入 (n_real_cells, n_features)
        gen_embeddings: 生成的Smart-seq2嵌入 (n_gen_cells, n_features)
        resolution_range: 聚类分辨率范围 (min, max)
        n_steps: 分辨率参数扫描步数
        real_celltypes: 真实数据的细胞类型标签 (可选)
        gen_celltypes: 生成数据的细胞类型标签 (可选)
        random_state: 随机种子

    返回:
        results: 包含分析结果的字典
    """
    # 创建AnnData对象
    real_adata = sc.AnnData(real_embeddings)
    gen_adata = sc.AnnData(gen_embeddings)

    # 添加细胞类型标签 (如果提供)
    if real_celltypes is not None:
        real_adata.obs['cell_type'] = real_celltypes
    if gen_celltypes is not None:
        gen_adata.obs['cell_type'] = gen_celltypes

    # 预处理
    sc.pp.neighbors(real_adata, random_state=random_state)
    sc.pp.neighbors(gen_adata, random_state=random_state)

    # 扫描不同分辨率参数
    resolutions = np.linspace(resolution_range[0], resolution_range[1], n_steps)
    real_entropies = []
    gen_entropies = []
    real_n_clusters = []
    gen_n_clusters = []
    aris = []  # 调整兰德指数 (如果提供细胞类型)
    nmis = []  # 标准化互信息 (如果提供细胞类型)

    for res in resolutions:
        # 真实数据聚类
        sc.tl.leiden(real_adata, resolution=res, random_state=random_state)
        real_clusters = real_adata.obs['leiden']

        # 计算真实数据熵
        cluster_counts = real_clusters.value_counts()
        p_real = cluster_counts / cluster_counts.sum()
        real_entropy = entropy(p_real, base=2)  # 使用以2为底的对数，结果在比特单位
        real_entropies.append(real_entropy)
        real_n_clusters.append(len(cluster_counts))

        # 生成数据聚类
        sc.tl.leiden(gen_adata, resolution=res, random_state=random_state)
        gen_clusters = gen_adata.obs['leiden']

        # 计算生成数据熵
        cluster_counts = gen_clusters.value_counts()
        p_gen = cluster_counts / cluster_counts.sum()
        gen_entropy = entropy(p_gen, base=2)
        gen_entropies.append(gen_entropy)
        gen_n_clusters.append(len(cluster_counts))

        # 如果提供细胞类型，计算聚类-类型一致性
        if real_celltypes is not None:
            ari = adjusted_rand_score(real_celltypes, real_clusters)
            nmi = normalized_mutual_info_score(real_celltypes, real_clusters)
            aris.append(ari)
            nmis.append(nmi)

    # 找到熵差异最小的分辨率 (最可比)
    entropy_diffs = np.abs(np.array(real_entropies) - np.array(gen_entropies))
    optimal_idx = np.argmin(np.abs(entropy_diffs))
    optimal_res = resolutions[optimal_idx]

    # 在最优分辨率下重新聚类以获取详细结果
    sc.tl.leiden(real_adata, resolution=optimal_res, random_state=random_state)
    sc.tl.leiden(gen_adata, resolution=optimal_res, random_state=random_state)

    # 计算最优分辨率下的详细指标
    real_clusters_opt = real_adata.obs['leiden']
    gen_clusters_opt = gen_adata.obs['leiden']

    # 真实数据熵
    cluster_counts = real_clusters_opt.value_counts()
    p_real = cluster_counts / cluster_counts.sum()
    real_entropy_opt = entropy(p_real, base=2)

    # 生成数据熵
    cluster_counts = gen_clusters_opt.value_counts()
    p_gen = cluster_counts / cluster_counts.sum()
    gen_entropy_opt = entropy(p_gen, base=2)

    # 熵比率
    entropy_ratio = gen_entropy_opt / real_entropy_opt

    # 创建结果字典
    results = {
        'resolutions': resolutions,
        'real_entropies': real_entropies,
        'gen_entropies': gen_entropies,
        'real_n_clusters': real_n_clusters,
        'gen_n_clusters': gen_n_clusters,
        'optimal_resolution': optimal_res,
        'real_entropy_opt': real_entropy_opt,
        'gen_entropy_opt': gen_entropy_opt,
        'entropy_ratio': entropy_ratio,
        'real_cluster_dist': p_real,
        'gen_cluster_dist': p_gen,
        'real_adata': real_adata,
        'gen_adata': gen_adata
    }

    # 添加聚类-类型一致性结果
    if real_celltypes is not None:
        results['real_aris'] = aris
        results['real_nmis'] = nmis
        results['real_ari_opt'] = adjusted_rand_score(real_celltypes, real_clusters_opt)
        results['real_nmi_opt'] = normalized_mutual_info_score(real_celltypes, real_clusters_opt)

    return results


def plot_entropy_analysis(results, figsize=(16, 12)):
    """可视化熵分析结果"""
    plt.figure(figsize=figsize)

    real_adata = results['real_adata']
    gen_adata = results['gen_adata']

    # 1. 熵随分辨率变化曲线
    plt.subplot(221)
    plt.plot(results['resolutions'], results['real_entropies'],
             'b-o', label='real embedding', alpha=0.8)
    plt.plot(results['resolutions'], results['gen_entropies'],
             'r--s', label='generate embeddings', alpha=0.8)
    plt.axvline(results['optimal_resolution'], color='g', linestyle=':',
                label=f'Optimal Resolution: {results["optimal_resolution"]:.2f}')
    plt.xlabel('Resolution Parameter')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy vs Clustering Resolution')
    plt.legend()
    plt.grid(True)

    # 添加最优分辨率的熵值
    plt.annotate(f'Real Entropy: {results["real_entropy_opt"]:.2f}',
                 (results['optimal_resolution'], results['real_entropy_opt']),
                 textcoords="offset points", xytext=(0, 10), ha='center')
    plt.annotate(f'Generated Entropy: {results["gen_entropy_opt"]:.2f}',
                 (results['optimal_resolution'], results['gen_entropy_opt']),
                 textcoords="offset points", xytext=(0, -15), ha='center')

    # 2. 聚类数量随分辨率变化
    plt.subplot(222)
    plt.plot(results['resolutions'], results['real_n_clusters'],
             'b-o', label='Real Embedding', alpha=0.8)
    plt.plot(results['resolutions'], results['gen_n_clusters'],
             'r--s', label='Generated Embedding', alpha=0.8)
    plt.axvline(results['optimal_resolution'], color='g', linestyle=':')
    plt.xlabel('Resolution Parameter')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters vs Resolution')
    plt.legend()
    plt.grid(True)

    # 3. 最优分辨率下的聚类分布
    plt.subplot(223)
    # 真实数据聚类分布
    plt.bar(results['real_cluster_dist'].index,
            results['real_cluster_dist'].values,
            alpha=0.7, label='Real')
    # 生成数据聚类分布
    plt.bar(results['gen_cluster_dist'].index,
            results['gen_cluster_dist'].values,
            alpha=0.7, label='Generated')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cell Proportion')
    plt.title(f'Cluster Distribution at Optimal Resolution (res={results["optimal_resolution"]:.2f})')
    plt.legend()

    # 4. 熵比较和模式坍塌评估
    plt.subplot(224)
    metrics = ['Real Entropy', 'Generated Entropy', 'Entropy Ratio']
    values = [results['real_entropy_opt'], results['gen_entropy_opt'], results['entropy_ratio']]

    plt.bar(metrics[:2], values[:2], color=['blue', 'red'])
    plt.ylabel('Entropy (bits)')

    # 添加熵比率作为第二条Y轴
    ax2 = plt.twinx()
    ax2.bar(metrics[2], values[2], color='green', alpha=0.7)
    ax2.set_ylabel('Ratio (Generated/Real)')
    ax2.set_ylim(0, 1.5)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.7)

    plt.title('Entropy Comparison and Mode Collapse Evaluation')

    # 添加评估注释
    # ratio = results['entropy_ratio']
    # if ratio < 0.7:
    #     plt.text(1, values[1]*0.9, "⚠️ Severe Mode Collapse", ha='center', color='red', fontsize=12)
    #     plt.text(2, ratio*0.9, f"Ratio: {ratio:.2f}\nSevere Lack of Diversity", ha='center', color='red')
    # elif ratio < 0.9:
    #     plt.text(1, values[1]*0.9, "⚠️ Mild Mode Collapse", ha='center', color='orange', fontsize=12)
    #     plt.text(2, ratio*0.9, f"Ratio: {ratio:.2f}\nPartial Lack of Diversity", ha='center', color='orange')
    # else:
    #     plt.text(1, values[1]*0.9, "✅ Good Diversity", ha='center', color='green', fontsize=12)
    #     plt.text(2, ratio*0.9, f"Ratio: {ratio:.2f}\nGood Diversity Preservation", ha='center', color='green')

    plt.tight_layout()
    plt.show()

    # 5. 聚类-细胞类型一致性 (如果提供细胞类型)
    if 'real_aris' in results:
        plt.figure(figsize=(14, 5))

        plt.subplot(121)
        plt.plot(results['resolutions'], results['real_aris'], 'b-o', label='Real Embedding ARI')
        plt.plot(results['resolutions'], results['real_nmis'], 'g--s', label='Real Embedding NMI')
        plt.axvline(results['optimal_resolution'], color='r', linestyle=':', label='Optimal Resolution')
        plt.xlabel('Resolution Parameter')
        plt.ylabel('Consistency Score')
        plt.title('Cluster-Cell Type Consistency')
        plt.legend()
        plt.grid(True)

        plt.subplot(122)
        # 创建聚类-细胞类型热力图
        cross_tab = pd.crosstab(real_adata.obs['leiden'], real_adata.obs['cell_type'])
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Real Embedding Cluster-Cell Type Correspondence (res={results["optimal_resolution"]:.2f})')
        plt.xlabel('Cell Type')
        plt.ylabel('Cluster')

        plt.tight_layout()
        plt.show()

    # 6. UMAP可视化聚类结果
    plt.figure(figsize=(16, 6))

    # 真实嵌入聚类
    plt.subplot(121)
    sc.pp.neighbors(real_adata)
    sc.tl.umap(real_adata)
    sc.pl.umap(real_adata, color='leiden', show=False, title='Real embedding clustering')

    # 生成嵌入聚类
    plt.subplot(122)
    sc.pp.neighbors(gen_adata)
    sc.tl.umap(gen_adata)
    sc.pl.umap(gen_adata, color='leiden', show=False, title='Generate embedding clustering')

    plt.tight_layout()
    plt.show()


def print_entropy_summary(results):
    """打印熵分析摘要"""
    print("\n" + "=" * 60)
    print("聚类熵多样性分析摘要")
    print("=" * 60)
    print(f"最优分辨率: {results['optimal_resolution']:.4f}")
    print(f"真实嵌入熵: {results['real_entropy_opt']:.4f} bits")
    print(f"生成嵌入熵: {results['gen_entropy_opt']:.4f} bits")
    print(f"熵比率 (生成/真实): {results['entropy_ratio']:.4f}")

    # 熵比率解释
    ratio = results['entropy_ratio']
    if ratio < 0.7:
        print("\n结论: ⚠️ 严重模式坍塌 - 生成嵌入多样性严重不足")
        print("      可能原因: 生成器崩溃、损失函数不平衡、训练不充分")
    elif ratio < 0.9:
        print("\n结论: ⚠️ 轻微模式坍塌 - 生成嵌入多样性部分缺失")
        print("      可能原因: 生成器保守、数据覆盖不全、特定细胞类型缺失")
    else:
        print("\n结论: ✅ 多样性良好 - 生成嵌入保留了原始多样性")

    # 聚类分布比较
    print("\n聚类分布比较:")
    print(f"真实聚类数: {len(results['real_cluster_dist'])}")
    print(f"生成聚类数: {len(results['gen_cluster_dist'])}")

    # 最大聚类占比
    real_max_prop = results['real_cluster_dist'].max()
    gen_max_prop = results['gen_cluster_dist'].max()
    print(f"最大聚类占比 - 真实: {real_max_prop:.4f}, 生成: {gen_max_prop:.4f}")

    # 熵差异解释
    entropy_diff = results['real_entropy_opt'] - results['gen_entropy_opt']
    if entropy_diff > 1.0:
        print(f"熵差异: {entropy_diff:.4f} bits (显著)")
    elif entropy_diff > 0.5:
        print(f"熵差异: {entropy_diff:.4f} bits (明显)")
    else:
        print(f"熵差异: {entropy_diff:.4f} bits (微小)")

    # 如果提供细胞类型
    if 'real_ari_opt' in results:
        print("\n聚类-细胞类型一致性:")
        print(f"调整兰德指数 (ARI): {results['real_ari_opt']:.4f}")
        print(f"标准化互信息 (NMI): {results['real_nmi_opt']:.4f}")


def evaluate_local_structure_preservation(emb_10x, emb_smart, k=15, n_samples=1000, random_state=42):
    """
    综合评估局部结构保持度

    参数:
        emb_10x: 原始10x嵌入 (n_cells, n_features)
        emb_smart: 映射后的Smart-seq2嵌入 (n_cells, n_features)
        k: 最近邻数量
        n_samples: 抽样细胞数量 (用于大型数据集)
        random_state: 随机种子

    返回:
        results: 包含所有评估指标的字典
    """
    # 确保细胞数量一致
    assert emb_10x.shape[0] == emb_smart.shape[0], "细胞数量不一致"
    n_cells = emb_10x.shape[0]

    # 抽样处理大型数据集
    if n_samples and n_cells > n_samples:
        np.random.seed(random_state)
        sample_idx = np.random.choice(n_cells, n_samples, replace=False)
        emb_10x = emb_10x[sample_idx]
        emb_smart = emb_smart[sample_idx]
        n_cells = n_samples

    # 1. 最近邻保持率
    nn_preservation = nearest_neighbor_preservation(emb_10x, emb_smart, k)

    # 2. 局部距离相关性
    local_corr = local_distance_correlation(emb_10x, emb_smart, k)

    # 3. 信任度指标
    trust_score = trustworthiness(emb_10x, emb_smart, n_neighbors=k)

    # 4. 图结构保持度
    graph_preservation = graph_structure_preservation(emb_10x, emb_smart, k)

    # 5. 局部应力
    stress = local_stress(emb_10x, emb_smart, k)

    # 创建结果字典
    results = {
        'nn_preservation': nn_preservation,
        'local_correlation': local_corr,
        'trustworthiness': trust_score,
        'graph_preservation': graph_preservation,
        'local_stress': stress,
        'k': k,
        'n_cells': n_cells
    }

    return results


def nearest_neighbor_preservation(emb_source, emb_target, k):
    """
    计算最近邻保持率

    参数:
        emb_source: 源空间嵌入 (10x)
        emb_target: 目标空间嵌入 (Smart-seq2)
        k: 最近邻数量

    返回:
        preservation_rate: 平均最近邻保持率
    """
    # 在源空间找到k个最近邻
    nn_source = NearestNeighbors(n_neighbors=k + 1)  # k+1包含自身
    nn_source.fit(emb_source)
    _, indices_source = nn_source.kneighbors(emb_source)

    # 在目标空间找到k个最近邻
    nn_target = NearestNeighbors(n_neighbors=k + 1)
    nn_target.fit(emb_target)
    _, indices_target = nn_target.kneighbors(emb_target)

    # 计算每个细胞的最近邻保持率
    preservation_rates = []
    for i in range(len(emb_source)):
        # 获取源空间和目标空间的最近邻索引 (排除自身)
        source_nn = set(indices_source[i, 1:])
        target_nn = set(indices_target[i, 1:])

        # 计算交集大小
        intersection = source_nn & target_nn
        preservation_rate = len(intersection) / k
        preservation_rates.append(preservation_rate)

    # 返回平均保持率
    return np.mean(preservation_rates)


def local_distance_correlation(emb_source, emb_target, k):
    """
    计算局部距离相关性

    参数:
        emb_source: 源空间嵌入
        emb_target: 目标空间嵌入
        k: 局部邻居数量

    返回:
        corr: 平均Spearman相关系数
    """
    # 在源空间找到k个最近邻
    nn_source = NearestNeighbors(n_neighbors=k + 1)
    nn_source.fit(emb_source)
    _, indices_source = nn_source.kneighbors(emb_source)

    # 计算每个细胞的局部距离相关性
    correlations = []
    for i in range(len(emb_source)):
        # 获取当前细胞和它的k个最近邻
        neighbors_idx = indices_source[i]

        # 源空间距离
        source_dist = pdist(emb_source[neighbors_idx])
        source_dist = squareform(source_dist)[0, 1:]  # 获取当前细胞到邻居的距离

        # 目标空间距离
        target_dist = pdist(emb_target[neighbors_idx])
        target_dist = squareform(target_dist)[0, 1:]  # 获取当前细胞到邻居的距离

        # 计算Spearman相关系数
        corr, _ = spearmanr(source_dist, target_dist)

        # 如果有效则存储
        if not np.isnan(corr):
            correlations.append(corr)

    # 返回平均相关系数
    return np.mean(correlations) if correlations else 0


def graph_structure_preservation(emb_source, emb_target, k):
    """
    计算图结构保持度

    参数:
        emb_source: 源空间嵌入
        emb_target: 目标空间嵌入
        k: 构建图的邻居数量

    返回:
        jaccard_sim: 图结构的平均Jaccard相似度
    """
    # 在源空间构建kNN图
    nn_source = NearestNeighbors(n_neighbors=k + 1)
    nn_source.fit(emb_source)
    _, indices_source = nn_source.kneighbors(emb_source)

    # 在目标空间构建kNN图
    nn_target = NearestNeighbors(n_neighbors=k + 1)
    nn_target.fit(emb_target)
    _, indices_target = nn_target.kneighbors(emb_target)

    # 计算每个细胞的图结构相似度
    jaccard_sims = []
    for i in range(len(emb_source)):
        # 获取源空间和目标空间的邻居集 (排除自身)
        source_nn = set(indices_source[i, 1:])
        target_nn = set(indices_target[i, 1:])

        # 计算Jaccard相似度
        intersection = source_nn & target_nn
        union = source_nn | target_nn
        jaccard_sim = len(intersection) / len(union) if union else 0
        jaccard_sims.append(jaccard_sim)

    # 返回平均Jaccard相似度
    return np.mean(jaccard_sims)


def local_stress(emb_source, emb_target, k):
    """
    计算局部应力

    参数:
        emb_source: 源空间嵌入
        emb_target: 目标空间嵌入
        k: 局部邻居数量

    返回:
        stress: 平均局部应力
    """
    # 在源空间找到k个最近邻
    nn_source = NearestNeighbors(n_neighbors=k + 1)
    nn_source.fit(emb_source)
    _, indices_source = nn_source.kneighbors(emb_source)

    # 计算每个细胞的局部应力
    stress_values = []
    for i in range(len(emb_source)):
        # 获取当前细胞和它的k个最近邻
        neighbors_idx = indices_source[i]

        # 源空间距离
        source_dists = pdist(emb_source[neighbors_idx])

        # 目标空间距离
        target_dists = pdist(emb_target[neighbors_idx])

        # 计算应力
        stress = np.sum((source_dists - target_dists) ** 2) / np.sum(source_dists ** 2)
        stress_values.append(stress)

    # 返回平均应力
    return np.mean(stress_values)


def plot_local_structure_results(results, figsize=(14, 10)):
    """可视化局部结构保持度评估结果"""
    plt.figure(figsize=figsize)

    # 1. 指标雷达图
    metrics = ['NN Preservation', 'Local Correlation', 'Trustworthiness', 'Graph Preservation', 'Local Stress']
    values = [
        results['nn_preservation'],
        results['local_correlation'],
        results['trustworthiness'],
        results['graph_preservation'],
        1 - results['local_stress']  # 应力取反，使值越大越好
    ]

    # 使指标数量适合雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    values += values[:1]  # 闭合雷达图
    angles = np.concatenate((angles, [angles[0]]))

    ax = plt.subplot(221, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, metrics)
    ax.set_ylim(0, 1)
    plt.title('Local Structure Preservation', y=1.1)

    # 2. 指标条形图
    plt.subplot(222)
    metrics = metrics[:-1]  # 排除应力
    values = values[:-2]  # 排除闭合点

    bars = plt.bar(metrics, values, color=sns.color_palette("viridis", len(metrics)))
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Local Structure Preservation Metrics')

    # 在条形上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    # 3. 应力分布解释
    plt.subplot(223)
    stress = results['local_stress']
    plt.bar(['Local Stress'], [stress], color='red' if stress > 0.2 else 'orange')
    plt.axhline(0.1, color='green', linestyle='--', label='Excellent')
    plt.axhline(0.2, color='orange', linestyle='--', label='Good')
    plt.axhline(0.3, color='red', linestyle='--', label='Poor')
    plt.ylim(0, min(1.0, stress * 1.5))
    plt.title(f'Local Stress: {stress:.4f}')
    plt.legend()

    # 4. 综合评分
    weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # 加权权重
    weighted_scores = [
        results['nn_preservation'] * weights[0],
        results['local_correlation'] * weights[1],
        results['trustworthiness'] * weights[2],
        results['graph_preservation'] * weights[3],
        (1 - results['local_stress']) * weights[4]
    ]
    overall_score = np.sum(weighted_scores)

    plt.subplot(224)
    plt.bar(['Overall Score'], [overall_score], color='skyblue')
    plt.axhline(0.9, color='green', linestyle='--', label='Excellent')
    plt.axhline(0.7, color='orange', linestyle='--', label='Good')
    plt.axhline(0.5, color='red', linestyle='--', label='Poor')
    plt.ylim(0, 1)
    plt.title(f'Overall Score: {overall_score:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 5. 评分解释
    print("\n局部结构保持度评估:")
    print(f"  1. 最近邻保持率 ({results['nn_preservation']:.3f}) - 映射后保留的最近邻比例")
    print(f"  2. 局部相关性 ({results['local_correlation']:.3f}) - 局部距离关系的相关性")
    print(f"  3. 信任度 ({results['trustworthiness']:.3f}) - 避免产生虚假邻居的程度")
    print(f"  4. 图结构保持度 ({results['graph_preservation']:.3f}) - kNN图结构的相似度")
    print(f"  5. 局部应力 ({results['local_stress']:.3f}) - 局部距离变形的程度")

    print("\n综合评估:")
    if overall_score > 0.85:
        print("  ✅ 优秀: 局部结构高度保持")
    elif overall_score > 0.7:
        print("  ⚠️ 良好: 局部结构基本保持，有轻微变形")
    elif overall_score > 0.5:
        print("  ⚠️ 一般: 局部结构部分保持，有明显变形")
    else:
        print("  ❌ 较差: 局部结构严重变形")


def visualize_local_structure(emb_10x, emb_smart, cell_idx, k=15, title=None):
    """
    可视化特定细胞的局部结构保持情况

    参数:
        emb_10x: 原始10x嵌入
        emb_smart: 映射后的Smart-seq2嵌入
        cell_idx: 目标细胞索引
        k: 显示邻居数量
        title: 图标题
    """
    # 在源空间找到最近邻
    nn_10x = NearestNeighbors(n_neighbors=k + 1)
    nn_10x.fit(emb_10x)
    _, indices_10x = nn_10x.kneighbors(emb_10x)

    # 在目标空间找到最近邻
    nn_smart = NearestNeighbors(n_neighbors=k + 1)
    nn_smart.fit(emb_smart)
    _, indices_smart = nn_smart.kneighbors(emb_smart)

    # 获取目标细胞的邻居
    source_neighbors = indices_10x[cell_idx]
    target_neighbors = indices_smart[cell_idx]

    # 创建图形
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    if title:
        fig.suptitle(title, fontsize=16)

    # 源空间可视化
    axs[0].scatter(emb_10x[:, 0], emb_10x[:, 1], alpha=0.1, s=10, c='gray', label='Other Cells')
    axs[0].scatter(emb_10x[source_neighbors[1:], 0], emb_10x[source_neighbors[1:], 1],
                   alpha=0.7, s=50, c='blue', label='Neighbors')
    axs[0].scatter(emb_10x[cell_idx, 0], emb_10x[cell_idx, 1],
                   s=200, c='red', marker='*', label='Target Cell')

    # 添加连接线
    for neighbor in source_neighbors[1:]:
        axs[0].plot([emb_10x[cell_idx, 0], emb_10x[neighbor, 0]],
                    [emb_10x[cell_idx, 1], emb_10x[neighbor, 1]],
                    'b-', alpha=0.3)

    axs[0].set_title('Original 10x Space')
    axs[0].set_xlabel('Dimension 1')
    axs[0].set_ylabel('Dimension 2')
    axs[0].legend()

    # 目标空间可视化
    axs[1].scatter(emb_smart[:, 0], emb_smart[:, 1], alpha=0.1, s=10, c='gray', label='Other Cells')
    axs[1].scatter(emb_smart[target_neighbors[1:], 0], emb_smart[target_neighbors[1:], 1],
                   alpha=0.7, s=50, c='green', label='Neighbors')
    axs[1].scatter(emb_smart[cell_idx, 0], emb_smart[cell_idx, 1],
                   s=200, c='red', marker='*', label='Target Cell')

    # 添加连接线
    for neighbor in target_neighbors[1:]:
        axs[1].plot([emb_smart[cell_idx, 0], emb_smart[neighbor, 0]],
                    [emb_smart[cell_idx, 1], emb_smart[neighbor, 1]],
                    'g-', alpha=0.3)

    axs[1].set_title('Mapped Smart-seq2 Space')
    axs[1].set_xlabel('Dimension 1')
    axs[1].set_ylabel('Dimension 2')
    axs[1].legend()

    plt.tight_layout()

    # 计算并显示保持率
    preserved_neighbors = set(source_neighbors[1:]) & set(target_neighbors[1:])
    preservation_rate = len(preserved_neighbors) / k
    plt.figtext(0.5, 0.01,
                f"Nearest Neighbor Preservation Rate: {preservation_rate:.2f} ({len(preserved_neighbors)}/{k} neighbors preserved)",
                ha='center', fontsize=12)

    plt.show()

    return preservation_rate


def calculate_pos(adata, pseudotime_col='dpt_pseudotime', time_col='time_numeric'):
    """
    计算POS (Pseudotime Ordering Score) 值
    """
    pseudotime = adata.obs[pseudotime_col].values
    real_time = adata.obs[time_col].values

    # 检查inf值
    valid_mask = np.isfinite(pseudotime)
    inf_count = np.sum(~valid_mask)
    total_count = len(pseudotime)
    inf_ratio = inf_count / total_count
    print('number of inf_count:', inf_count)
    print('number of total_count:', total_count)
    print('ratio of inf pseudotime:', inf_ratio)

    pos_value, p_value = spearmanr(pseudotime, real_time)

    penalty = 1.0 - inf_ratio * 0.5  # 每10%的inf细胞降低5%的分数
    # penalty = 1.0
    penalized_pos = pos_value * penalty

    return penalized_pos, p_value


def complete_trajectory_analysis(adata, root_tag='E3', time_mapping=None, time_column='time', use_latent=True):
    """
    完整的轨迹分析流程 - 修正版
    """
    # 1. 预处理
    print("Step 1: Data preprocessing...")
    # sc.pp.filter_cells(adata, min_genes=200)

    # 2. 时间转换
    print("Step 2: Time label conversion...")
    if time_mapping is None:
        time_mapping = {'E3': 3, 'E4': 4, 'E5': 5, 'E6': 6, 'E7': 7}
    adata.obs['time_numeric'] = adata.obs[time_column].map(time_mapping)

    # 3. 聚类
    print("Step 3: Clustering...")
    if use_latent:
        sc.pp.neighbors(adata, n_neighbors=20, use_rep='latent')
    else:
        sc.pp.filter_genes(adata, min_cells=3)
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        # adata = adata[:, adata.var.highly_variable]
        # sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs=40)

    sc.tl.leiden(adata, resolution=0.5)

    # 4. PAGA分析 - 修正部分
    print("Step 4: PAGA analysis_3...")
    sc.tl.paga(adata, groups='leiden')

    # 关键修正：先绘制PAGA图来生成位置信息
    sc.pl.paga(adata, threshold=0.1, show=False)  # 这会生成 adata.uns['paga']['pos']

    # 现在可以使用PAGA初始化UMAP
    sc.tl.umap(adata, init_pos='paga')

    # 5. DPT分析
    print("Step 5: DPT analysis_3...")

    # 选择根细胞（E3时期）
    e3_cells = adata.obs[adata.obs[time_column] == root_tag].index
    if len(e3_cells) == 0:
        print("Warning: No E3 cells found. Using alternative root selection.")
        # 如果没有E3细胞，使用其他方法选择根细胞
        sc.tl.diffmap(adata)
        # 选择在扩散组件中最早时间点的细胞
        earliest_time = adata.obs['time_numeric'].min()
        earliest_cells = adata.obs[adata.obs['time_numeric'] == earliest_time].index
        # 使用PCA空间中心点
        earliest_pca = adata[earliest_cells].obsm['X_pca']
        root_cell_idx = np.argmin(np.linalg.norm(earliest_pca - np.median(earliest_pca, axis=0), axis=1))
        root_cell = earliest_cells[root_cell_idx]
    else:
        # 使用E3细胞
        if use_latent:
            e3_pca = adata[e3_cells].obsm['latent']
        else:
            e3_pca = adata[e3_cells].obsm['X_pca']
        root_cell_idx = np.argmin(np.linalg.norm(e3_pca - np.median(e3_pca, axis=0), axis=1))
        root_cell = e3_cells[root_cell_idx]

    # 设置根细胞并运行DPT
    adata.uns['iroot'] = np.where(adata.obs_names == root_cell)[0][0]
    sc.tl.dpt(adata)

    # 6. 计算POS
    print("Step 6: Calculating POS...")
    pos_value, p_value = calculate_pos(adata)

    print(f"\n=== Results ===")
    print(f"POS值 (Spearman相关系数): {pos_value:.4f}")
    print(f"显著性P值: {p_value:.4e}")

    return adata, pos_value, p_value



"""
特征可视化模块

实现DNA序列特征的可视化分析，包括：
- t-SNE降维：将高维特征映射到2D/3D空间
- PCA降维：主成分分析，获取解释方差比
- 可视化绘图：生成t-SNE和PCA散点图
- 聚类分析：计算轮廓系数等聚类指标

用于评估自监督学习学到的表示是否具有区分性。
"""
import os
import inspect
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 物种标签映射
SPECIES_NAMES = {0: 'Human', 1: 'Mouse', 2: 'Fly', 3: 'Worm'}
SPECIES_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}


class FeatureVisualizer:
    """特征可视化器

    提供t-SNE降维、PCA降维、可视化绘图和聚类分析功能。

    Attributes:
        pca_model: PCA模型实例（用于获取解释方差比）
        tsne_model: t-SNE模型实例
    """

    def __init__(self, random_state: int = 42):
        """初始化特征可视化器

        Args:
            random_state: 随机种子，确保结果可复现
        """
        self.random_state = random_state
        self.pca_model: Optional[PCA] = None
        self.tsne_model: Optional[TSNE] = None
        self.pca_explained_variance_ratio_: Optional[np.ndarray] = None

        logger.info("FeatureVisualizer初始化完成")

    def tsne_reduce(
        self,
        features: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000
    ) -> np.ndarray:
        """t-SNE降维

        将高维特征映射到低维空间（通常2D或3D）用于可视化。

        Args:
            features: 输入特征矩阵，形状为 (n_samples, n_features)
            n_components: 降维后的维度（2或3）
            perplexity: 困惑度参数，控制局部和全局结构的平衡
            learning_rate: 学习率
            n_iter: 迭代次数

        Returns:
            降维后的特征矩阵，形状为 (n_samples, n_components)
        """
        logger.info(f"开始t-SNE降维: {features.shape} -> {n_components}D")

        # 兼容不同版本的 scikit-learn：
        #  - >= 1.5 将 TSNE 的 n_iter 更名为 max_iter
        #  - >= 1.5 的 learning_rate 只接受 'auto' / 'warn'，不再接受浮点数
        tsne_params = inspect.signature(TSNE).parameters

        tsne_kwargs = {
            'n_components': n_components,
            'perplexity': perplexity,
            'random_state': self.random_state,
            'init': 'pca',
        }
        tsne_kwargs['max_iter' if 'max_iter' in tsne_params else 'n_iter'] = n_iter

        lr_default = tsne_params['learning_rate'].default
        tsne_kwargs['learning_rate'] = (
            learning_rate if isinstance(lr_default, (int, float)) else 'auto'
        )

        self.tsne_model = TSNE(**tsne_kwargs)

        tsne_result = self.tsne_model.fit_transform(features)

        logger.info(f"t-SNE降维完成，结果形状: {tsne_result.shape}")
        return tsne_result

    def pca_reduce(
        self,
        features: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """PCA降维

        使用主成分分析将特征降维到指定维度。

        Args:
            features: 输入特征矩阵，形状为 (n_samples, n_features)
            n_components: 降维后的维度

        Returns:
            降维后的特征矩阵，形状为 (n_samples, n_components)
        """
        logger.info(f"开始PCA降维: {features.shape} -> {n_components}D")

        self.pca_model = PCA(
            n_components=n_components,
            random_state=self.random_state
        )

        pca_result = self.pca_model.fit_transform(features)
        self.pca_explained_variance_ratio_ = self.pca_model.explained_variance_ratio_

        logger.info(f"PCA降维完成，解释方差比: {self.pca_explained_variance_ratio_}")
        return pca_result

    def get_pca_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """获取PCA解释方差比

        Returns:
            解释方差比数组，如果PCA未执行则返回None
        """
        return self.pca_explained_variance_ratio_

    def plot_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        perplexity: float = 30.0,
        title: str = 't-SNE Visualization of DNA Features'
    ) -> str:
        """绘制t-SNE可视化图

        Args:
            features: 输入特征矩阵
            labels: 标签数组
            save_path: 保存路径
            perplexity: 困惑度参数
            title: 图标题

        Returns:
            保存的图片路径
        """
        logger.info("生成t-SNE可视化图")

        # 执行t-SNE降维
        tsne_result = self.tsne_reduce(features, n_components=2, perplexity=perplexity)

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图，按物种着色
        for species_id in np.unique(labels):
            mask = labels == species_id
            species_name = SPECIES_NAMES.get(species_id, f'Species {species_id}')
            color = SPECIES_COLORS.get(species_id, '#888888')

            ax.scatter(
                tsne_result[mask, 0],
                tsne_result[mask, 1],
                c=color,
                label=species_name,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )

        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 保存图形
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"t-SNE可视化图已保存到: {save_path}")
        return save_path

    def plot_pca(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        title: str = 'PCA Visualization of DNA Features'
    ) -> str:
        """绘制PCA可视化图

        Args:
            features: 输入特征矩阵
            labels: 标签数组
            save_path: 保存路径
            title: 图标题

        Returns:
            保存的图片路径
        """
        logger.info("生成PCA可视化图")

        # 执行PCA降维
        pca_result = self.pca_reduce(features, n_components=2)

        # 获取解释方差比
        explained_variance = self.pca_explained_variance_ratio_

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图，按物种着色
        for species_id in np.unique(labels):
            mask = labels == species_id
            species_name = SPECIES_NAMES.get(species_id, f'Species {species_id}')
            color = SPECIES_COLORS.get(species_id, '#888888')

            ax.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=color,
                label=species_name,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )

        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 保存图形
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"PCA可视化图已保存到: {save_path}")
        return save_path

    def compute_clustering_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """计算聚类指标

        Args:
            features: 特征矩阵（通常是降维后的结果）
            labels: 真实标签

        Returns:
            包含聚类指标的字典：
            - silhouette_score: 轮廓系数 [-1, 1]，越高越好
            - calinski_harabasz_score: CH指数，越高越好
            - davies_bouldin_score: DB指数，越低越好
        """
        logger.info("计算聚类指标")

        # 轮廓系数
        silhouette = silhouette_score(features, labels)

        # Calinski-Harabasz指数
        calinski_harabasz = calinski_harabasz_score(features, labels)

        # Davies-Bouldin指数
        davies_bouldin = davies_bouldin_score(features, labels)

        metrics = {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(calinski_harabasz),
            'davies_bouldin_score': float(davies_bouldin)
        }

        logger.info(f"聚类指标: {metrics}")
        return metrics

    def extract_features(
        self,
        encoder: torch.nn.Module,
        input_data: torch.Tensor,
        device: str = 'cpu'
    ) -> np.ndarray:
        """从编码器提取特征

        Args:
            encoder: PyTorch编码器模型
            input_data: 输入数据张量
            device: 设备（'cpu' 或 'cuda'）

        Returns:
            提取的特征矩阵，形状为 (n_samples, output_dim)
        """
        logger.info("从编码器提取特征")

        encoder.eval()
        encoder.to(device)

        with torch.no_grad():
            input_data = input_data.to(device)
            features = encoder(input_data)

        # 转换为numpy数组
        features_np = features.cpu().numpy()

        logger.info(f"特征提取完成，形状: {features_np.shape}")
        return features_np

    def generate_visualization_report(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        output_dir: str,
        prefix: str = 'visualization'
    ) -> Dict[str, object]:
        """生成完整可视化报告

        Args:
            features: 输入特征矩阵
            labels: 标签数组
            output_dir: 输出目录
            prefix: 文件名前缀

        Returns:
            包含可视化结果的字典：
            - tsne_plot_path: t-SNE图路径
            - pca_plot_path: PCA图路径
            - clustering_metrics: 聚类指标
            - pca_explained_variance: PCA解释方差比
        """
        logger.info("生成完整可视化报告")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 生成t-SNE图
        tsne_path = os.path.join(output_dir, f'{prefix}_tsne.png')
        self.plot_tsne(features, labels, tsne_path)

        # 生成PCA图
        pca_path = os.path.join(output_dir, f'{prefix}_pca.png')
        self.plot_pca(features, labels, pca_path)

        # 计算聚类指标（使用PCA降维后的特征）
        pca_features = self.pca_reduce(features, n_components=10)
        clustering_metrics = self.compute_clustering_metrics(pca_features, labels)

        # 重新执行PCA用于报告
        pca_2d = self.pca_reduce(features, n_components=2)

        report = {
            'tsne_plot_path': tsne_path,
            'pca_plot_path': pca_path,
            'clustering_metrics': clustering_metrics,
            'pca_explained_variance': self.pca_explained_variance_ratio_.tolist()
        }

        logger.info(f"可视化报告生成完成，输出目录: {output_dir}")
        return report

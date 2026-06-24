"""
特征可视化模块测试

测试内容：
- t-SNE降维功能
- PCA降维功能
- 可视化绘图功能
- 聚类分析功能

遵循TDD原则：先写测试 -> 看失败 -> 实现 -> 看通过 -> 提交
"""
import os
import sys
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFeatureVisualization:
    """特征可视化测试类"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟的特征数据
        self.n_samples = 100
        self.n_features = 256  # 与CNN编码器输出维度一致
        self.features = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.labels = np.random.randint(0, 4, size=self.n_samples)  # 4个物种

        # 创建模拟的编码器
        self.mock_encoder = MagicMock()
        self.mock_encoder.get_output_dim.return_value = self.n_features

    def test_tsne_reduce_dimensionality(self):
        """测试t-SNE降维结果维度正确"""
        from feature_visualization import FeatureVisualizer

        visualizer = FeatureVisualizer()
        tsne_result = visualizer.tsne_reduce(self.features, n_components=2)

        # 验证降维结果维度正确
        assert tsne_result.shape == (self.n_samples, 2), \
            f"t-SNE结果维度错误: 期望 ({self.n_samples}, 2), 得到 {tsne_result.shape}"

    def test_tsne_reduce_3d(self):
        """测试t-SNE降维到3维"""
        from feature_visualization import FeatureVisualizer

        visualizer = FeatureVisualizer()
        tsne_result = visualizer.tsne_reduce(self.features, n_components=3)

        assert tsne_result.shape == (self.n_samples, 3), \
            f"t-SNE 3D结果维度错误: 期望 ({self.n_samples}, 3), 得到 {tsne_result.shape}"

    def test_pca_reduce_dimensionality(self):
        """测试PCA降维结果维度正确"""
        from feature_visualization import FeatureVisualizer

        visualizer = FeatureVisualizer()
        pca_result = visualizer.pca_reduce(self.features, n_components=2)

        # 验证降维结果维度正确
        assert pca_result.shape == (self.n_samples, 2), \
            f"PCA结果维度错误: 期望 ({self.n_samples}, 2), 得到 {pca_result.shape}"

    def test_pca_explained_variance(self):
        """测试PCA解释方差比"""
        from feature_visualization import FeatureVisualizer

        visualizer = FeatureVisualizer()
        pca_result = visualizer.pca_reduce(self.features, n_components=10)

        # 获取解释方差比
        explained_variance_ratio = visualizer.get_pca_explained_variance_ratio()

        # 验证解释方差比存在且和为<=1
        assert explained_variance_ratio is not None, "解释方差比为None"
        assert len(explained_variance_ratio) == 10, \
            f"解释方差比长度错误: 期望 10, 得到 {len(explained_variance_ratio)}"
        assert np.sum(explained_variance_ratio) <= 1.0 + 1e-6, \
            f"解释方差比之和超过1: {np.sum(explained_variance_ratio)}"

    def test_plot_tsne_visualization(self):
        """测试t-SNE可视化图可生成"""
        from feature_visualization import FeatureVisualizer
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互后端

        visualizer = FeatureVisualizer()

        # 创建临时输出目录
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'test_tsne_plot.png')

        # 生成可视化图
        save_path = visualizer.plot_tsne(
            features=self.features,
            labels=self.labels,
            save_path=output_path
        )

        # 验证图可生成
        assert os.path.exists(save_path), f"t-SNE可视化图未生成: {save_path}"
        assert os.path.getsize(save_path) > 0, "t-SNE可视化图文件为空"

        # 清理
        if os.path.exists(save_path):
            os.remove(save_path)

    def test_plot_pca_visualization(self):
        """测试PCA可视化图可生成"""
        from feature_visualization import FeatureVisualizer
        import matplotlib
        matplotlib.use('Agg')

        visualizer = FeatureVisualizer()

        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'test_pca_plot.png')

        save_path = visualizer.plot_pca(
            features=self.features,
            labels=self.labels,
            save_path=output_path
        )

        assert os.path.exists(save_path), f"PCA可视化图未生成: {save_path}"
        assert os.path.getsize(save_path) > 0, "PCA可视化图文件为空"

        if os.path.exists(save_path):
            os.remove(save_path)

    def test_compute_clustering_metrics(self):
        """测试聚类指标可计算"""
        from feature_visualization import FeatureVisualizer

        visualizer = FeatureVisualizer()

        # 使用降维后的特征计算聚类指标
        tsne_result = visualizer.tsne_reduce(self.features, n_components=2)
        metrics = visualizer.compute_clustering_metrics(tsne_result, self.labels)

        # 验证指标包含必要字段
        assert 'silhouette_score' in metrics, "缺少silhouette_score指标"
        assert 'calinski_harabasz_score' in metrics, "缺少calinski_harabasz_score指标"
        assert 'davies_bouldin_score' in metrics, "缺少davies_bouldin_score指标"

        # 验证指标值在合理范围内
        assert -1 <= metrics['silhouette_score'] <= 1, \
            f"轮廓系数超出范围: {metrics['silhouette_score']}"
        assert metrics['calinski_harabasz_score'] >= 0, \
            f"Calinski-Harabasz指数应为非负: {metrics['calinski_harabasz_score']}"
        assert metrics['davies_bouldin_score'] >= 0, \
            f"Davies-Bouldin指数应为非负: {metrics['davies_bouldin_score']}"

    def test_extract_features_from_encoder(self):
        """测试从编码器提取特征"""
        from feature_visualization import FeatureVisualizer

        visualizer = FeatureVisualizer()

        # 创建模拟的PyTorch模型和数据
        batch_size = 10
        seq_len = 512
        mock_input = torch.randint(0, 68, (batch_size, seq_len))
        output_dim = self.n_features

        # 创建简单的模拟编码器
        class MockEncoder(torch.nn.Module):
            def __init__(self, output_dim):
                super().__init__()
                self.output_dim = output_dim
                self.linear = torch.nn.Linear(10, output_dim)

            def get_output_dim(self):
                return self.output_dim

            def forward(self, x):
                return torch.randn(x.shape[0], self.output_dim)

        encoder = MockEncoder(output_dim)
        features = visualizer.extract_features(encoder, mock_input, device='cpu')

        assert features.shape == (batch_size, self.n_features), \
            f"提取的特征维度错误: 期望 ({batch_size}, {self.n_features}), 得到 {features.shape}"

    def test_generate_full_visualization_report(self):
        """测试生成完整可视化报告"""
        from feature_visualization import FeatureVisualizer
        import matplotlib
        matplotlib.use('Agg')

        visualizer = FeatureVisualizer()

        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(output_dir, exist_ok=True)

        # 生成完整报告
        report = visualizer.generate_visualization_report(
            features=self.features,
            labels=self.labels,
            output_dir=output_dir,
            prefix='test_report'
        )

        # 验证报告包含必要字段
        assert 'tsne_plot_path' in report, "报告缺少tsne_plot_path"
        assert 'pca_plot_path' in report, "报告缺少pca_plot_path"
        assert 'clustering_metrics' in report, "报告缺少clustering_metrics"
        assert 'pca_explained_variance' in report, "报告缺少pca_explained_variance"

        # 验证图文件生成
        assert os.path.exists(report['tsne_plot_path']), "t-SNE图未生成"
        assert os.path.exists(report['pca_plot_path']), "PCA图未生成"

        # 清理测试文件
        for key in ['tsne_plot_path', 'pca_plot_path']:
            if os.path.exists(report[key]):
                os.remove(report[key])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

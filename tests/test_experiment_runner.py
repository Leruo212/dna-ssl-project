"""
实验运行器测试模块

测试实验运行和结果收集功能，包括：
- 验证实验完成无错误
- 验证结果文件生成
- 验证指标合理
"""
import os
import sys
import json
import pytest
import torch
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiment_runner import ExperimentRunner


@pytest.fixture
def data_dir():
    """数据目录路径"""
    return str(Path(__file__).parent.parent / "data")


@pytest.fixture
def results_dir():
    """结果目录路径"""
    return str(Path(__file__).parent.parent / "results" / "test_experiment")


@pytest.fixture
def runner(data_dir, results_dir):
    """创建实验运行器实例"""
    return ExperimentRunner(
        data_dir=data_dir,
        results_dir=results_dir,
        num_epochs=2,  # 测试时使用少量epoch
        batch_size=8,
        learning_rate=1e-3,
        device='cpu'
    )


class TestExperimentRunner:
    """实验运行器测试类"""

    def test_initialization(self, runner, data_dir, results_dir):
        """测试初始化"""
        assert runner.data_dir == data_dir
        assert runner.results_dir == results_dir
        assert runner.num_epochs == 2
        assert runner.batch_size == 8
        assert runner.device == 'cpu'

    def test_run_pretraining(self, runner):
        """测试预训练实验"""
        # 运行预训练
        pretraining_results = runner.run_pretraining()

        # 验证返回结果结构
        assert 'train_losses' in pretraining_results
        assert 'val_losses' in pretraining_results
        assert 'model_path' in pretraining_results

        # 验证损失是列表
        assert isinstance(pretraining_results['train_losses'], list)
        assert isinstance(pretraining_results['val_losses'], list)

        # 验证损失数量与epoch数量一致
        assert len(pretraining_results['train_losses']) == runner.num_epochs

        # 验证损失值合理（应该大于0）
        for loss in pretraining_results['train_losses']:
            assert loss > 0

        # 验证模型文件存在
        assert os.path.exists(pretraining_results['model_path'])

    def test_run_classification(self, runner):
        """测试分类实验"""
        # 先运行预训练
        runner.run_pretraining()

        # 运行分类
        classification_results = runner.run_classification()

        # 验证返回结果结构
        assert 'metrics' in classification_results
        assert 'train_history' in classification_results

        # 验证指标结构
        metrics = classification_results['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

        # 验证指标值在合理范围内 [0, 1]
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            assert 0 <= metrics[metric_name] <= 1

    def test_generate_visualizations(self, runner):
        """测试特征可视化生成"""
        # 先运行预训练
        runner.run_pretraining()

        # 生成可视化
        visualization_results = runner.generate_visualizations()

        # 验证返回结果结构
        assert 'tsne_plot_path' in visualization_results
        assert 'pca_plot_path' in visualization_results
        assert 'clustering_metrics' in visualization_results

        # 验证图片文件存在
        assert os.path.exists(visualization_results['tsne_plot_path'])
        assert os.path.exists(visualization_results['pca_plot_path'])

        # 验证聚类指标
        clustering_metrics = visualization_results['clustering_metrics']
        assert 'silhouette_score' in clustering_metrics
        assert 'calinski_harabasz_score' in clustering_metrics
        assert 'davies_bouldin_score' in clustering_metrics

    def test_collect_metrics(self, runner):
        """测试收集实验指标"""
        # 运行完整实验
        runner.run_pretraining()
        runner.run_classification()
        runner.generate_visualizations()

        # 收集指标
        all_metrics = runner.collect_metrics()

        # 验证返回结果结构
        assert 'pretraining' in all_metrics
        assert 'classification' in all_metrics
        assert 'visualization' in all_metrics
        assert 'summary' in all_metrics

        # 验证摘要
        summary = all_metrics['summary']
        assert 'best_train_loss' in summary
        assert 'best_val_loss' in summary
        assert 'classification_accuracy' in summary
        assert 'silhouette_score' in summary

    def test_save_results(self, runner):
        """测试保存结果"""
        # 运行完整实验
        runner.run_pretraining()
        runner.run_classification()
        runner.generate_visualizations()
        runner.collect_metrics()

        # 保存结果
        results_path = runner.save_results()

        # 验证结果文件存在
        assert os.path.exists(results_path)

        # 验证结果文件是有效的JSON
        with open(results_path, 'r') as f:
            saved_results = json.load(f)

        assert 'pretraining' in saved_results
        assert 'classification' in saved_results
        assert 'visualization' in saved_results
        assert 'summary' in saved_results

    def test_run_full_experiment(self, runner):
        """测试完整实验流程"""
        # 运行完整实验
        results = runner.run_full_experiment()

        # 验证所有结果都已收集
        assert 'pretraining' in results
        assert 'classification' in results
        assert 'visualization' in results
        assert 'summary' in results
        assert 'results_path' in results

        # 验证结果文件存在
        assert os.path.exists(results['results_path'])

        # 验证指标合理
        summary = results['summary']
        assert summary['best_train_loss'] > 0
        assert 0 <= summary['classification_accuracy'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

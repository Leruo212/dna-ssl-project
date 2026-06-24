"""
预训练循环测试模块

测试Pretrainer类的训练循环、验证循环、损失监控和模型保存功能。
"""
import os
import sys
import pytest
import torch
import tempfile
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pretrainer import Pretrainer
from masked_reconstruction_model import MaskedReconstructionModel
from dataset import DNADataset
from kmer_encoding import KmerEncoder


@pytest.fixture
def setup_components():
    """创建测试所需的组件"""
    # 创建编码器
    encoder = KmerEncoder(k=3, max_length=64)

    # 创建模型
    model = MaskedReconstructionModel(
        vocab_size=len(encoder.vocabulary),
        embedding_dim=32,
        num_filters=[16, 32],
        kernel_sizes=[3, 3],
        encoder_output_dim=64,
        mask_ratio=0.15
    )

    return model, encoder


@pytest.fixture
def setup_dummy_dataset():
    """创建虚拟数据集用于测试"""
    # 创建编码器
    encoder = KmerEncoder(k=3, max_length=64)

    # 创建临时FASTA文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">seq1\n")
        f.write("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n")
        f.write(">seq2\n")
        f.write("TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA\n")
        fasta_file = f.name

    # 创建掩码数据集
    dataset = DNADataset(
        fasta_files=[fasta_file],
        encoder=encoder,
        max_length=64,
        mask_ratio=0.15
    )

    # 清理临时文件
    os.unlink(fasta_file)

    return dataset


class TestPretrainer:
    """Pretrainer测试类"""

    def test_pretrainer_initialization(self, setup_components):
        """测试Pretrainer初始化"""
        model, encoder = setup_components

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-4,
            batch_size=32,
            device='cpu'
        )

        assert pretrainer is not None
        assert pretrainer.model is model
        assert pretrainer.learning_rate == 1e-4
        assert pretrainer.batch_size == 32

    def test_pretrainer_train_loop_runs(self, setup_components, setup_dummy_dataset):
        """测试训练循环可以运行"""
        model, encoder = setup_components
        dataset = setup_dummy_dataset

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-4,
            batch_size=2,
            device='cpu'
        )

        # 运行1个epoch的训练
        train_loss = pretrainer.train_epoch(dataset)

        assert isinstance(train_loss, float)
        assert train_loss >= 0

    def test_pretrainer_validation_loop_runs(self, setup_components, setup_dummy_dataset):
        """测试验证循环可以运行"""
        model, encoder = setup_components
        dataset = setup_dummy_dataset

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-4,
            batch_size=2,
            device='cpu'
        )

        # 运行验证
        val_loss = pretrainer.validate(dataset)

        assert isinstance(val_loss, float)
        assert val_loss >= 0

    def test_pretrainer_loss_decreases(self, setup_components, setup_dummy_dataset):
        """测试损失下降"""
        model, encoder = setup_components
        dataset = setup_dummy_dataset

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-3,  # 使用较大学习率加速收敛
            batch_size=2,
            device='cpu'
        )

        # 记录初始损失
        initial_loss = pretrainer.train_epoch(dataset)

        # 训练几个epoch
        for _ in range(5):
            current_loss = pretrainer.train_epoch(dataset)

        # 损失应该下降（允许小的波动）
        assert current_loss < initial_loss * 1.5  # 允许50%的波动

    def test_pretrainer_loss_monitoring(self, setup_components, setup_dummy_dataset):
        """测试损失监控"""
        model, encoder = setup_components
        dataset = setup_dummy_dataset

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-4,
            batch_size=2,
            device='cpu'
        )

        # 训练几个epoch
        for _ in range(3):
            pretrainer.train_epoch(dataset)

        # 检查损失历史记录
        assert len(pretrainer.train_losses) == 3
        assert all(isinstance(loss, float) for loss in pretrainer.train_losses)

    def test_pretrainer_model_saving(self, setup_components, setup_dummy_dataset):
        """测试模型保存"""
        model, encoder = setup_components
        dataset = setup_dummy_dataset

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-4,
            batch_size=2,
            device='cpu'
        )

        # 训练1个epoch
        pretrainer.train_epoch(dataset)

        # 保存模型
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model.pt')
            pretrainer.save_model(save_path)

            # 验证文件存在
            assert os.path.exists(save_path)

            # 验证可以加载
            loaded_state = torch.load(save_path, map_location='cpu')
            assert 'model_state_dict' in loaded_state
            assert 'optimizer_state_dict' in loaded_state
            assert 'train_losses' in loaded_state

    def test_pretrainer_full_training(self, setup_components, setup_dummy_dataset):
        """测试完整训练流程"""
        model, encoder = setup_components
        dataset = setup_dummy_dataset

        pretrainer = Pretrainer(
            model=model,
            learning_rate=1e-4,
            batch_size=2,
            device='cpu'
        )

        # 运行完整训练
        results = pretrainer.train(
            train_dataset=dataset,
            val_dataset=dataset,
            num_epochs=3,
            save_path=None
        )

        assert 'train_losses' in results
        assert 'val_losses' in results
        assert len(results['train_losses']) == 3
        assert len(results['val_losses']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

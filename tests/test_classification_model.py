"""
分类头实现测试模块

测试ClassificationModel类的实例化、微调循环和分类指标计算功能。
"""
import os
import sys
import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cnn_encoder import CNNEncoder
from classification_model import ClassificationModel, compute_classification_metrics


class TestClassificationModelInit:
    """ClassificationModel实例化测试"""

    def test_model_instantiation(self):
        """测试分类模型可以实例化"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )
        model = ClassificationModel(encoder=encoder, num_classes=4, dropout=0.2)

        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_output_shape(self):
        """测试模型输出形状正确"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )
        model = ClassificationModel(encoder=encoder, num_classes=4, dropout=0.2)

        # 模拟输入: (batch_size=2, seq_len=32)
        x = torch.randint(0, 68, (2, 32))
        output = model(x)

        assert output.shape == (2, 4), f"期望形状(2, 4)，实际得到{output.shape}"

    def test_encoder_parameters_frozen(self):
        """测试编码器参数被冻结"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )
        model = ClassificationModel(encoder=encoder, num_classes=4, dropout=0.2)

        # 检查编码器参数是否被冻结
        for name, param in model.encoder.named_parameters():
            assert not param.requires_grad, f"编码器参数 {name} 未被冻结"

    def test_classifier_parameters_trainable(self):
        """测试分类头参数可训练"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )
        model = ClassificationModel(encoder=encoder, num_classes=4, dropout=0.2)

        # 检查分类头参数是否可训练
        for name, param in model.classifier.named_parameters():
            assert param.requires_grad, f"分类头参数 {name} 不可训练"

    def test_model_with_pretrained_encoder(self):
        """测试使用预训练编码器"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )

        # 模拟预训练权重
        pretrained_state = encoder.state_dict()

        # 创建分类模型
        model = ClassificationModel(
            encoder=encoder,
            num_classes=4,
            dropout=0.2,
            pretrained_weights=pretrained_state
        )

        # 验证编码器权重已加载
        for key in pretrained_state:
            assert torch.equal(
                model.encoder.state_dict()[key],
                pretrained_state[key]
            ), f"预训练权重 {key} 未正确加载"


class TestClassificationModelFinetune:
    """微调循环测试"""

    @pytest.fixture
    def setup_finetune(self):
        """创建微调测试所需的组件"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )
        model = ClassificationModel(encoder=encoder, num_classes=4, dropout=0.2)

        # 创建虚拟数据集
        num_samples = 20
        seq_len = 32
        sequences = torch.randint(0, 68, (num_samples, seq_len))
        labels = torch.randint(0, 4, (num_samples,))

        return model, sequences, labels

    def test_finetune_loop_runs(self, setup_finetune):
        """测试微调循环可以运行"""
        model, sequences, labels = setup_finetune

        # 创建数据集和数据加载器
        dataset = torch.utils.data.TensorDataset(sequences, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # 初始化优化器
        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=1e-3
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 运行1个epoch的微调
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_sequences, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert num_batches > 0

    def test_finetune_only_updates_classifier(self, setup_finetune):
        """测试微调只更新分类头参数"""
        model, sequences, labels = setup_finetune

        # 记录编码器初始权重
        encoder_initial_weights = {
            name: param.clone()
            for name, param in model.encoder.named_parameters()
        }

        # 创建数据集和数据加载器
        dataset = torch.utils.data.TensorDataset(sequences, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        # 初始化优化器（只优化分类头）
        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=1e-3
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 运行微调
        model.train()
        for batch_sequences, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # 验证编码器权重未改变
        for name, param in model.encoder.named_parameters():
            assert torch.equal(
                param,
                encoder_initial_weights[name]
            ), f"编码器参数 {name} 在微调过程中被修改"

    def test_finetune_multiple_epochs(self, setup_finetune):
        """测试多轮微调"""
        model, sequences, labels = setup_finetune

        dataset = torch.utils.data.TensorDataset(sequences, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=1e-3
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 运行多轮微调
        epoch_losses = []
        for epoch in range(3):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_sequences, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)

        # 验证损失记录
        assert len(epoch_losses) == 3
        assert all(isinstance(loss, float) for loss in epoch_losses)
        assert all(loss >= 0 for loss in epoch_losses)


class TestClassificationMetrics:
    """分类指标计算测试"""

    def test_compute_classification_metrics(self):
        """测试分类指标计算"""
        # 创建模拟预测和标签
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # 完美预测

        metrics = compute_classification_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

        # 完美预测的指标应该都是1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_compute_metrics_imperfect_prediction(self):
        """测试不完美预测的指标计算"""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 0, 0, 1, 2, 3])  # 1个错误

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics['accuracy'] == 7/8  # 7/8正确
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1

    def test_compute_metrics_with_model(self):
        """测试使用模型计算指标"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=32,
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            output_dim=64,
            dropout=0.1
        )
        model = ClassificationModel(encoder=encoder, num_classes=4, dropout=0.2)

        # 创建测试数据
        sequences = torch.randint(0, 68, (10, 32))
        labels = torch.randint(0, 4, (10,))

        # 获取预测
        model.eval()
        with torch.no_grad():
            outputs = model(sequences)
            predictions = torch.argmax(outputs, dim=1)

        # 计算指标
        metrics = compute_classification_metrics(
            labels.numpy(),
            predictions.numpy()
        )

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_compute_metrics_weighted_average(self):
        """测试加权平均指标"""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred, average='weighted')

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

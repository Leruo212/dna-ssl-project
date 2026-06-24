"""
CNN编码器测试

测试CNNEncoder类的：
- 模型实例化
- 前向传播无错误
- 输出维度正确
"""
import pytest
import torch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cnn_encoder import CNNEncoder


class TestCNNEncoder:
    """CNNEncoder测试类"""

    def test_model_instantiation(self):
        """测试模型可实例化"""
        encoder = CNNEncoder(
            vocab_size=68,  # 4个特殊token + 64个3-mer
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )
        assert encoder is not None
        assert isinstance(encoder, CNNEncoder)

    def test_forward_pass_no_error(self):
        """测试前向传播无错误"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        # 创建模拟输入（batch_size=2, seq_len=512）
        batch_size = 2
        seq_len = 512
        x = torch.randint(0, 68, (batch_size, seq_len))

        # 前向传播不应抛出异常
        output = encoder(x)
        assert output is not None

    def test_output_dimension(self):
        """测试输出维度正确"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        batch_size = 4
        seq_len = 512
        x = torch.randint(0, 68, (batch_size, seq_len))

        output = encoder(x)

        # 输出应为 (batch_size, output_dim)
        assert output.shape == (batch_size, 256)

    def test_different_batch_sizes(self):
        """测试不同batch size"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        for batch_size in [1, 8, 16]:
            x = torch.randint(0, 68, (batch_size, 512))
            output = encoder(x)
            assert output.shape == (batch_size, 256)

    def test_embedding_layer_exists(self):
        """测试嵌入层存在"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        assert hasattr(encoder, 'embedding')
        assert isinstance(encoder.embedding, torch.nn.Embedding)
        assert encoder.embedding.num_embeddings == 68
        assert encoder.embedding.embedding_dim == 128

    def test_conv_layers_exist(self):
        """测试卷积层存在"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        assert hasattr(encoder, 'conv_layers')
        assert len(encoder.conv_layers) == 3

    def test_fc_layer_exists(self):
        """测试全连接层存在"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        assert hasattr(encoder, 'fc')
        assert isinstance(encoder.fc, torch.nn.Sequential)
        # 验证Sequential中第一个层是Linear
        assert isinstance(encoder.fc[0], torch.nn.Linear)

    def test_output_is_float(self):
        """测试输出为浮点类型"""
        encoder = CNNEncoder(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            output_dim=256
        )

        x = torch.randint(0, 68, (2, 512))
        output = encoder(x)
        assert output.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

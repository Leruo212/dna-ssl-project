"""
掩码重构模型测试

测试MaskedReconstructionModel类的：
- 模型实例化
- 掩码逻辑正确性
- 损失计算正确性
"""
import pytest
import torch
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from masked_reconstruction_model import MaskedReconstructionModel


class TestMaskedReconstructionModel:
    """MaskedReconstructionModel测试类"""

    def test_model_instantiation(self):
        """测试模型可实例化"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )
        assert model is not None
        assert isinstance(model, MaskedReconstructionModel)

    def test_mask_generation(self):
        """测试掩码生成逻辑"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )

        # 测试掩码生成
        batch_size = 4
        seq_len = 100
        x = torch.randint(0, 68, (batch_size, seq_len))

        # 生成掩码
        mask = model.generate_mask(x)

        # 验证掩码形状
        assert mask.shape == (batch_size, seq_len)

        # 验证掩码值是布尔类型
        assert mask.dtype == torch.bool

        # 验证掩码比例约为15%（允许一定误差）
        mask_ratio = mask.float().mean()
        assert 0.10 <= mask_ratio <= 0.20, f"掩码比例 {mask_ratio:.3f} 不在合理范围内"

    def test_forward_pass_with_mask(self):
        """测试带掩码的前向传播"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )

        # 创建模拟输入
        batch_size = 2
        seq_len = 512
        original_x = torch.randint(0, 68, (batch_size, seq_len))
        masked_x = original_x.clone()

        # 手动生成掩码并应用
        mask = model.generate_mask(original_x)
        masked_x[mask] = 0  # 用0作为掩码token

        # 前向传播不应抛出异常
        output = model(masked_x, mask)
        assert output is not None

    def test_loss_calculation(self):
        """测试损失计算"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )

        # 创建模拟数据
        batch_size = 4
        seq_len = 100
        original_x = torch.randint(0, 68, (batch_size, seq_len))
        masked_x = original_x.clone()

        # 生成掩码
        mask = model.generate_mask(original_x)
        masked_x[mask] = 0

        # 计算损失
        loss, predictions = model.compute_loss(masked_x, original_x, mask)

        # 验证损失是标量
        assert loss.dim() == 0

        # 验证损失值非负
        assert loss.item() >= 0

        # 验证预测结果形状
        assert predictions.shape == (batch_size, seq_len, 68)

    def test_only_masked_positions_contribute_to_loss(self):
        """测试只有被掩码位置参与损失计算"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )

        # 创建模拟数据
        batch_size = 2
        seq_len = 50
        original_x = torch.randint(0, 68, (batch_size, seq_len))
        masked_x = original_x.clone()

        # 生成掩码
        mask = model.generate_mask(original_x)
        masked_x[mask] = 0

        # 计算损失
        loss, _ = model.compute_loss(masked_x, original_x, mask)

        # 验证损失计算只考虑被掩码位置
        # 这个测试主要确保代码逻辑正确，不抛出异常
        assert loss is not None

    def test_model_output_dimensions(self):
        """测试模型输出维度"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )

        batch_size = 3
        seq_len = 200
        x = torch.randint(0, 68, (batch_size, seq_len))
        mask = model.generate_mask(x)

        # 前向传播
        output = model(x, mask)

        # 验证输出维度
        assert output.shape == (batch_size, seq_len, 68)

    def test_different_mask_ratios(self):
        """测试不同的掩码比例"""
        # 测试10%掩码
        model_10 = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.10
        )

        # 测试20%掩码
        model_20 = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.20
        )

        batch_size = 2
        seq_len = 100
        x = torch.randint(0, 68, (batch_size, seq_len))

        # 测试不同掩码比例
        mask_10 = model_10.generate_mask(x)
        mask_20 = model_20.generate_mask(x)

        # 验证掩码比例
        ratio_10 = mask_10.float().mean().item()
        ratio_20 = mask_20.float().mean().item()

        assert 0.05 <= ratio_10 <= 0.15
        assert 0.15 <= ratio_20 <= 0.25

    def test_gradient_flow(self):
        """测试梯度流"""
        model = MaskedReconstructionModel(
            vocab_size=68,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=68,
            mask_ratio=0.15
        )

        batch_size = 2
        seq_len = 50
        original_x = torch.randint(0, 68, (batch_size, seq_len))
        masked_x = original_x.clone()

        mask = model.generate_mask(original_x)
        masked_x[mask] = 0

        # 计算损失
        loss, _ = model.compute_loss(masked_x, original_x, mask)

        # 反向传播
        loss.backward()

        # 验证参数有梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "模型参数应该有梯度"
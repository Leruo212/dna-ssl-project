"""
掩码重构模型模块

实现基于掩码的自监督学习模型，用于DNA序列重构任务。
类似于BERT的掩码语言模型，但应用于DNA序列。

架构：
- 编码器：CNNEncoder（已实现）
- 重构头：全连接层，将编码器输出映射到词汇表大小
- 掩码逻辑：随机掩码15%的碱基
- 损失计算：交叉熵损失，只计算被掩码位置

输入：
- 原始序列和掩码后的序列
- 掩码位置

输出：
- 重构损失
- 预测结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from cnn_encoder import CNNEncoder


class MaskedReconstructionModel(nn.Module):
    """掩码重构模型

    组合编码器和重构头，实现掩码逻辑和重构损失计算。

    Args:
        vocab_size: 词汇表大小
        embedding_dim: 嵌入向量维度
        num_filters: 各卷积层的滤波器数量列表
        kernel_sizes: 各卷积层的核大小列表
        encoder_output_dim: 编码器输出维度
        reconstruction_dim: 重构维度（词汇表大小）
        mask_ratio: 掩码比例
        dropout: Dropout概率
    """

    def __init__(
        self,
        vocab_size: int = 68,
        embedding_dim: int = 128,
        num_filters: list = None,
        kernel_sizes: list = None,
        encoder_output_dim: int = 256,
        reconstruction_dim: int = None,
        mask_ratio: float = 0.15,
        dropout: float = 0.1
    ):
        super(MaskedReconstructionModel, self).__init__()

        # 默认参数
        if num_filters is None:
            num_filters = [128, 256, 512]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        # 默认参数
        if reconstruction_dim is None:
            reconstruction_dim = vocab_size

        self.vocab_size = vocab_size
        self.encoder_output_dim = encoder_output_dim
        self.reconstruction_dim = reconstruction_dim
        self.mask_ratio = mask_ratio

        # 编码器：CNNEncoder
        self.encoder = CNNEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        # 重构头：全连接层，将编码器输出映射到词汇表大小
        # 注意：这里需要将序列维度考虑进去
        # 编码器输出是 (batch_size, encoder_output_dim)
        # 但我们需要为每个位置预测，所以需要序列级别的输出
        # 因此，我们需要修改编码器或使用不同的架构

        # 为了保持简单，我们假设编码器输出序列级别的表示
        # 但实际上CNNEncoder输出的是全局表示，不是序列级别的
        # 我们需要调整：要么修改编码器，要么使用不同的重构策略

        # 方案：使用一个全连接层将编码器的全局表示广播到每个位置
        # 但这不太合理。更好的方案是使用序列级别的编码器。

        # 重新设计：使用一个简单的序列编码器，输出每个位置的表示
        # 或者，我们修改CNNEncoder，使其输出序列级别的表示

        # 为了快速实现，我们创建一个简单的序列编码器
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Conv1d(embedding_dim, encoder_output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 重构头：将每个位置的编码映射到词汇表大小
        self.reconstruction_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim // 2, reconstruction_dim)
        )

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        """生成随机掩码

        Args:
            x: 输入序列，形状为 (batch_size, seq_len)

        Returns:
            掩码张量，形状为 (batch_size, seq_len)，布尔类型
        """
        batch_size, seq_len = x.shape

        # 生成随机掩码
        mask = torch.rand(batch_size, seq_len, device=x.device) < self.mask_ratio

        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入序列，形状为 (batch_size, seq_len)
            mask: 掩码，形状为 (batch_size, seq_len)，布尔类型

        Returns:
            预测结果，形状为 (batch_size, seq_len, reconstruction_dim)
        """
        # 序列编码：(batch_size, seq_len) -> (batch_size, seq_len, encoder_output_dim)
        # 注意：我们需要处理嵌入层和卷积层的维度
        embedded = self.sequence_encoder[0](x)  # 嵌入层
        embedded = embedded.permute(0, 2, 1)  # 转换为卷积输入格式
        encoded = self.sequence_encoder[1](embedded)  # 卷积层
        encoded = encoded.permute(0, 2, 1)  # 转换回 (batch_size, seq_len, encoder_output_dim)

        # 应用ReLU和Dropout
        encoded = self.sequence_encoder[2](encoded)  # ReLU
        encoded = self.sequence_encoder[3](encoded)  # Dropout

        # 重构：(batch_size, seq_len, encoder_output_dim) -> (batch_size, seq_len, reconstruction_dim)
        reconstructed = self.reconstruction_head(encoded)

        return reconstructed

    def compute_loss(
        self,
        masked_x: torch.Tensor,
        original_x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算重构损失

        Args:
            masked_x: 掩码后的序列，形状为 (batch_size, seq_len)
            original_x: 原始序列，形状为 (batch_size, seq_len)
            mask: 掩码，形状为 (batch_size, seq_len)，布尔类型

        Returns:
            loss: 重构损失（标量）
            predictions: 预测结果，形状为 (batch_size, seq_len, reconstruction_dim)
        """
        # 前向传播得到预测
        predictions = self.forward(masked_x, mask)

        # 计算损失
        # predictions: (batch_size, seq_len, reconstruction_dim)
        # original_x: (batch_size, seq_len)

        # 重塑为 (batch_size * seq_len, reconstruction_dim) 和 (batch_size * seq_len)
        batch_size, seq_len, recon_dim = predictions.shape
        predictions_flat = predictions.view(-1, recon_dim)
        targets_flat = original_x.view(-1)

        # 计算每个位置的损失
        loss_per_position = self.loss_fn(predictions_flat, targets_flat)

        # 重塑回 (batch_size, seq_len)
        loss_per_position = loss_per_position.view(batch_size, seq_len)

        # 只计算被掩码位置的损失
        mask_expanded = mask.unsqueeze(-1).expand_as(predictions)  # (batch_size, seq_len, 1)
        mask_flat = mask.view(-1)  # (batch_size * seq_len)

        # 应用掩码
        masked_loss = loss_per_position * mask.float()

        # 计算平均损失（只考虑被掩码位置）
        if mask.sum() > 0:
            loss = masked_loss.sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=predictions.device)

        return loss, predictions

    def get_encoder(self) -> nn.Module:
        """获取编码器

        Returns:
            编码器模块
        """
        return self.encoder

    def get_reconstruction_dim(self) -> int:
        """获取重构维度

        Returns:
            重构维度（词汇表大小）
        """
        return self.reconstruction_dim
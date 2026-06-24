"""
CNN编码器模块

实现基于卷积神经网络的DNA序列编码器，用于将k-mer编码的序列
转换为固定长度的表示向量。

架构：
- 嵌入层：将k-mer token ID转换为稠密向量
- 卷积层序列：多层1D卷积 + ReLU激活
- 全局平均池化：压缩序列维度
- 全连接层：映射到最终表示维度
"""
import torch
import torch.nn as nn
from typing import List


class CNNEncoder(nn.Module):
    """CNN编码器

    将k-mer编码的DNA序列转换为固定长度的表示向量。

    Args:
        vocab_size: 词汇表大小（特殊token + k-mer数量）
        embedding_dim: 嵌入向量维度
        num_filters: 各卷积层的滤波器数量列表
        kernel_sizes: 各卷积层的核大小列表
        output_dim: 最终输出维度
        dropout: Dropout概率
    """

    def __init__(
        self,
        vocab_size: int = 68,
        embedding_dim: int = 128,
        num_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super(CNNEncoder, self).__init__()

        # 默认参数
        if num_filters is None:
            num_filters = [128, 256, 512]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim

        # 嵌入层：token ID -> 稠密向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=None  # 不设置padding_idx，让模型学习PAD的表示
        )

        # 卷积层序列
        self.conv_layers = nn.ModuleList()
        in_channels = embedding_dim

        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # 保持序列长度不变
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层：映射到输出维度
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len)，
               包含k-mer token的整数ID

        Returns:
            输出张量，形状为 (batch_size, output_dim)，
            表示DNA序列的特征向量
        """
        # 嵌入层：(batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # 转换为卷积输入格式：(batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # 卷积层序列
        hidden = embedded
        for conv_layer in self.conv_layers:
            hidden = conv_layer(hidden)

        # 全局平均池化：(batch_size, num_filters[-1], seq_len) -> (batch_size, num_filters[-1], 1)
        pooled = self.global_avg_pool(hidden)

        # 压缩最后一维：(batch_size, num_filters[-1], 1) -> (batch_size, num_filters[-1])
        pooled = pooled.squeeze(-1)

        # 全连接层：(batch_size, num_filters[-1]) -> (batch_size, output_dim)
        output = self.fc(pooled)

        return output

    def get_output_dim(self) -> int:
        """获取输出维度

        Returns:
            输出特征向量的维度
        """
        return self.output_dim

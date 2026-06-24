"""
分类头实现模块

实现DNA序列物种分类的微调模型，包括：
- ClassificationModel类：组合预训练编码器和分类头
- 编码器参数冻结：只训练分类头
- 分类指标计算：准确率、精确率、召回率、F1分数
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('macro', 'micro', 'weighted')

    Returns:
        包含accuracy, precision, recall, f1的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class ClassificationModel(nn.Module):
    """分类模型

    组合预训练编码器和分类头，用于DNA序列物种分类。

    架构：
    - 预训练编码器（冻结参数）：将DNA序列转换为特征向量
    - 分类头（可训练）：将特征向量映射到物种类别

    Args:
        encoder: 预训练的CNNEncoder实例
        num_classes: 分类类别数（默认4：human, mouse, fly, worm）
        dropout: Dropout概率
        pretrained_weights: 可选的预训练权重字典
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 4,
        dropout: float = 0.2,
        pretrained_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        super(ClassificationModel, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes

        # 加载预训练权重（如果提供）
        if pretrained_weights is not None:
            self.encoder.load_state_dict(pretrained_weights)
            logger.info("已加载预训练权重")

        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 获取编码器输出维度
        encoder_output_dim = encoder.get_output_dim()

        # 分类头：encoder_output_dim -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        logger.info(f"ClassificationModel初始化完成，编码器输出维度: {encoder_output_dim}，类别数: {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len)，
               包含k-mer token的整数ID

        Returns:
            输出张量，形状为 (batch_size, num_classes)，
            表示每个类别的logits
        """
        # 冻结编码器，不计算梯度
        with torch.no_grad():
            features = self.encoder(x)

        # 分类头
        logits = self.classifier(features)

        return logits

    def get_encoder(self) -> nn.Module:
        """获取编码器

        Returns:
            冻结的编码器实例
        """
        return self.encoder

    def get_classifier(self) -> nn.Module:
        """获取分类头

        Returns:
            可训练的分类头实例
        """
        return self.classifier

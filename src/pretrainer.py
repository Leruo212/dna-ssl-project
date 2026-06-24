"""
预训练循环模块

实现DNA序列自监督学习的预训练循环，包括：
- 训练循环：前向传播、损失计算、反向传播、参数更新
- 验证循环：评估模型性能
- 损失监控：记录和跟踪训练/验证损失
- 模型保存：保存模型检查点

使用MaskedReconstructionModel进行掩码重构预训练任务。
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pretrainer:
    """预训练器类

    实现完整的预训练流程，包括训练循环、验证循环、损失监控和模型保存。

    Args:
        model: MaskedReconstructionModel实例
        learning_rate: 学习率
        batch_size: 批次大小
        device: 训练设备（'cpu' 或 'cuda'）
        weight_decay: 权重衰减系数
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = 'cpu',
        weight_decay: float = 0.01
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.weight_decay = weight_decay

        # 将模型移到指定设备
        self.model.to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 损失历史记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        logger.info(f"Pretrainer初始化完成，设备: {self.device}")

    def train_epoch(self, dataset: Dataset) -> float:
        """训练一个epoch

        Args:
            dataset: 训练数据集

        Returns:
            平均训练损失
        """
        self.model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # 解包数据
            if len(batch) == 3:
                original, masked, _ = batch
            else:
                raise ValueError("数据集应返回(original, masked, label)三元组")

            # 移到设备
            original = original.to(self.device)
            masked = masked.to(self.device)

            # 生成掩码（与数据集中的掩码位置相同）
            mask = (original != masked).float()

            # 前向传播
            predictions = self.model(masked, mask.bool())

            # 计算损失
            loss, _ = self.model.compute_loss(masked, original, mask.bool())

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)

        logger.info(f"训练Epoch完成，平均损失: {avg_loss:.4f}")
        return avg_loss

    def validate(self, dataset: Dataset) -> float:
        """验证模型

        Args:
            dataset: 验证数据集

        Returns:
            平均验证损失
        """
        self.model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # 解包数据
                if len(batch) == 3:
                    original, masked, _ = batch
                else:
                    raise ValueError("数据集应返回(original, masked, label)三元组")

                # 移到设备
                original = original.to(self.device)
                masked = masked.to(self.device)

                # 生成掩码
                mask = (original != masked).float()

                # 计算损失
                loss, _ = self.model.compute_loss(masked, original, mask.bool())

                total_loss += loss.item()
                num_batches += 1

        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)

        logger.info(f"验证完成，平均损失: {avg_loss:.4f}")
        return avg_loss

    def save_model(self, save_path: str) -> None:
        """保存模型检查点

        Args:
            save_path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存检查点
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到: {save_path}")

    def load_model(self, load_path: str) -> None:
        """加载模型检查点

        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        logger.info(f"模型已从 {load_path} 加载")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_epochs: int = 50,
        save_path: Optional[str] = None,
        save_every: int = 10
    ) -> Dict[str, List[float]]:
        """完整训练流程

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集（可选）
            num_epochs: 训练轮次
            save_path: 模型保存路径（可选）
            save_every: 每隔多少epoch保存一次

        Returns:
            训练结果字典，包含train_losses和val_losses
        """
        logger.info(f"开始训练，共 {num_epochs} 个epoch")

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # 训练
            train_loss = self.train_epoch(train_dataset)

            # 验证
            if val_dataset is not None:
                val_loss = self.validate(val_dataset)
                logger.info(f"  训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            else:
                logger.info(f"  训练损失: {train_loss:.4f}")

            # 定期保存模型
            if save_path and (epoch + 1) % save_every == 0:
                epoch_save_path = os.path.join(
                    save_path,
                    f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self.save_model(epoch_save_path)

        # 保存最终模型
        if save_path:
            final_save_path = os.path.join(save_path, "final_model.pt")
            self.save_model(final_save_path)

        logger.info("训练完成")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def get_training_history(self) -> Dict[str, List[float]]:
        """获取训练历史

        Returns:
            包含训练和验证损失的字典
        """
        return {
            'train_losses': self.train_losses.copy(),
            'val_losses': self.val_losses.copy()
        }

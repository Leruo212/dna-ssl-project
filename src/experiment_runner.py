"""
实验运行器模块

实现完整的实验流程，包括：
- 预训练实验：运行掩码重构预训练
- 下游分类实验：运行物种分类任务
- 特征可视化：生成t-SNE和PCA可视化
- 指标收集：收集所有实验指标并保存

使用所有已实现的组件完成端到端实验。
"""
import os
import sys
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from kmer_encoding import KmerEncoder
from dataset import DNADataset
from cnn_encoder import CNNEncoder
from masked_reconstruction_model import MaskedReconstructionModel
from pretrainer import Pretrainer
from classification_model import ClassificationModel, compute_classification_metrics
from feature_visualization import FeatureVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器

    整合所有组件，运行完整的实验流程。

    Args:
        data_dir: 数据目录路径
        results_dir: 结果保存目录
        num_epochs: 预训练epoch数量
        batch_size: 批次大小
        learning_rate: 学习率
        device: 训练设备
        max_length: 序列最大长度
        k: k-mer大小
        mask_ratio: 掩码比例
    """

    def __init__(
        self,
        data_dir: str,
        results_dir: str,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        classification_epochs: int = 50,
        classification_lr: float = 1e-3,
        device: str = 'cpu',
        max_length: int = 512,
        k: int = 3,
        mask_ratio: float = 0.15
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classification_epochs = classification_epochs
        self.classification_lr = classification_lr
        self.device = device
        self.max_length = max_length
        self.k = k
        self.mask_ratio = mask_ratio

        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)

        # 初始化组件
        self.encoder = KmerEncoder(k=k, max_length=max_length)
        self.model = None
        self.pretrainer = None
        self.classification_model = None
        self.visualizer = FeatureVisualizer(random_state=42)

        # 存储结果
        self.pretraining_results = {}
        self.classification_results = {}
        self.visualization_results = {}
        self.all_metrics = {}

        logger.info(f"ExperimentRunner初始化完成，设备: {device}")

    def _load_datasets(self):
        """加载数据集"""
        logger.info("加载数据集...")

        # 获取FASTA文件路径
        fasta_files = []
        for species in ['human', 'mouse', 'fly', 'worm']:
            fasta_path = os.path.join(self.data_dir, f"{species}.fasta")
            if os.path.exists(fasta_path):
                fasta_files.append(fasta_path)
            else:
                logger.warning(f"FASTA文件不存在: {fasta_path}")

        if not fasta_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到FASTA文件")

        # 创建掩码数据集（用于预训练）
        masked_dataset = DNADataset(
            fasta_files=fasta_files,
            encoder=self.encoder,
            max_length=self.max_length,
            k=self.k,
            mask_ratio=self.mask_ratio
        )

        # 创建非掩码数据集（用于分类）
        classification_dataset = DNADataset(
            fasta_files=fasta_files,
            encoder=self.encoder,
            max_length=self.max_length,
            k=self.k,
            mask_ratio=None
        )

        logger.info(f"数据集加载完成，共 {len(masked_dataset)} 条序列")

        return masked_dataset, classification_dataset

    def _split_dataset(self, dataset, train_ratio: float = 0.8):
        """划分训练集和验证集"""
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"数据集划分: 训练集 {train_size}, 验证集 {val_size}")
        return train_dataset, val_dataset

    def run_pretraining(self) -> Dict[str, Any]:
        """运行预训练实验

        Returns:
            预训练结果字典
        """
        logger.info("=" * 50)
        logger.info("开始预训练实验")
        logger.info("=" * 50)

        # 加载数据集
        masked_dataset, _ = self._load_datasets()
        train_dataset, val_dataset = self._split_dataset(masked_dataset)

        # 创建模型
        vocab_size = len(self.encoder.vocabulary)
        self.model = MaskedReconstructionModel(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=[128, 256, 512],
            kernel_sizes=[3, 3, 3],
            encoder_output_dim=256,
            reconstruction_dim=vocab_size,
            mask_ratio=self.mask_ratio,
            dropout=0.1
        )

        # 创建预训练器
        self.pretrainer = Pretrainer(
            model=self.model,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            weight_decay=0.01
        )

        # 运行训练
        save_path = os.path.join(self.results_dir, "pretraining")
        training_history = self.pretrainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=self.num_epochs,
            save_path=save_path,
            save_every=self.num_epochs
        )

        # 保存结果
        self.pretraining_results = {
            'train_losses': training_history['train_losses'],
            'val_losses': training_history['val_losses'],
            'model_path': os.path.join(save_path, "final_model.pt"),
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }

        logger.info("预训练实验完成")
        return self.pretraining_results

    @torch.no_grad()
    def _precompute_features(self, dataset):
        """用冻结的编码器一次性提取并缓存整个数据集的特征

        编码器参数已冻结，其特征与分类头无关，没必要每轮重算。
        实测在 CPU 上重跑一遍 CNN 约 50ms/样本，若每轮都重算，
        50 轮需要约 100 分钟。

        Args:
            dataset: 返回 (sequence, label) 或 (original, masked, label) 的数据集

        Returns:
            (features, labels) 张量元组
        """
        encoder = self.model.get_encoder()
        encoder.eval()
        encoder.to(self.device)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        feats, labels = [], []
        for batch in loader:
            if len(batch) == 2:
                sequences, y = batch
            else:
                sequences, _, y = batch
            feats.append(encoder(sequences.to(self.device)).cpu())
            labels.append(y)

        return torch.cat(feats), torch.cat(labels)

    def run_classification(self) -> Dict[str, Any]:
        """运行分类实验

        Returns:
            分类结果字典
        """
        logger.info("=" * 50)
        logger.info("开始分类实验")
        logger.info("=" * 50)

        if self.model is None:
            raise RuntimeError("请先运行预训练实验")

        # 加载数据集
        _, classification_dataset = self._load_datasets()
        train_dataset, val_dataset = self._split_dataset(classification_dataset)

        # 创建分类模型
        encoder = self.model.get_encoder()
        self.classification_model = ClassificationModel(
            encoder=encoder,
            num_classes=4,
            dropout=0.2
        )

        # 预提取冻结特征，并在特征上构建数据加载器
        logger.info("提取并缓存编码器特征（只需一次）...")
        train_feats, train_labels = self._precompute_features(train_dataset)
        val_feats, val_labels = self._precompute_features(val_dataset)
        logger.info(f"特征缓存完成: train={tuple(train_feats.shape)} val={tuple(val_feats.shape)}")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_feats, train_labels),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_feats, val_labels),
            batch_size=self.batch_size,
            shuffle=False
        )

        # 训练分类头：只优化 requires_grad=True 的参数（feature_norm + classifier）。
        # 编码器参数已冻结，不纳入优化器。
        # 学习率不能沿用预训练的 1e-4 —— 冻结特征 L2 量级仅约 0.46，
        # 在 1e-4 下梯度小到几乎不更新（实测 5 轮后 loss 反而上升）。
        trainable_params = [
            p for p in self.classification_model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(trainable_params, lr=self.classification_lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_losses = []
        val_losses = []
        val_accuracies = []
        self.classification_model.to(self.device)

        # 训练循环
        num_classification_epochs = self.classification_epochs
        best_val_acc = 0.0
        best_state = None

        for epoch in range(num_classification_epochs):
            self.classification_model.train()
            total_loss = 0
            num_batches = 0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.classification_model.forward_from_features(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)

            # 每轮在验证集上评估，用于确认模型是否真的在收敛
            self.classification_model.eval()
            val_loss_total = 0
            val_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.classification_model.forward_from_features(features)

                    val_loss_total += criterion(outputs, labels).item()
                    val_batches += 1
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            val_loss = val_loss_total / max(val_batches, 1)
            val_acc = correct / max(total, 1)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.detach().clone()
                    for k, v in self.classification_model.state_dict().items()
                }

            logger.info(
                f"分类训练 Epoch {epoch + 1}/{num_classification_epochs} "
                f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # 回滚到验证集上表现最好的权重
        if best_state is not None:
            self.classification_model.load_state_dict(best_state)

        # 评估
        self.classification_model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                outputs = self.classification_model.forward_from_features(features)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        # 计算指标
        metrics = compute_classification_metrics(
            np.array(all_labels),
            np.array(all_predictions)
        )

        self.classification_results = {
            'metrics': metrics,
            'train_history': train_losses,
            'val_loss_history': val_losses,
            'val_accuracy_history': val_accuracies,
            'best_val_accuracy': best_val_acc,
            'num_epochs': num_classification_epochs,
            'learning_rate': self.classification_lr
        }

        logger.info(f"分类实验完成，准确率: {metrics['accuracy']:.4f}")
        return self.classification_results

    def generate_visualizations(self) -> Dict[str, Any]:
        """生成特征可视化

        Returns:
            可视化结果字典
        """
        logger.info("=" * 50)
        logger.info("生成特征可视化")
        logger.info("=" * 50)

        if self.model is None:
            raise RuntimeError("请先运行预训练实验")

        # 加载数据集
        _, classification_dataset = self._load_datasets()

        # 准备数据
        dataloader = torch.utils.data.DataLoader(
            classification_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # 提取特征
        encoder = self.model.get_encoder()
        encoder.eval()
        encoder.to(self.device)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    sequences, labels = batch
                else:
                    sequences, _, labels = batch

                sequences = sequences.to(self.device)
                features = encoder(sequences)

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        features_np = np.concatenate(all_features, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        logger.info(f"特征提取完成，形状: {features_np.shape}")

        # 生成可视化报告
        viz_output_dir = os.path.join(self.results_dir, "visualizations")
        self.visualization_results = self.visualizer.generate_visualization_report(
            features=features_np,
            labels=labels_np,
            output_dir=viz_output_dir,
            prefix='dna_features'
        )

        logger.info("特征可视化生成完成")
        return self.visualization_results

    def collect_metrics(self) -> Dict[str, Any]:
        """收集所有实验指标

        Returns:
            所有指标的字典
        """
        logger.info("收集实验指标...")

        # 计算摘要
        summary = {}

        if self.pretraining_results:
            summary['best_train_loss'] = min(self.pretraining_results['train_losses'])
            if self.pretraining_results['val_losses']:
                summary['best_val_loss'] = min(self.pretraining_results['val_losses'])
            else:
                summary['best_val_loss'] = None

        if self.classification_results:
            summary['classification_accuracy'] = self.classification_results['metrics']['accuracy']
            summary['classification_f1'] = self.classification_results['metrics']['f1']

        if self.visualization_results:
            summary['silhouette_score'] = self.visualization_results['clustering_metrics']['silhouette_score']

        self.all_metrics = {
            'pretraining': self.pretraining_results,
            'classification': self.classification_results,
            'visualization': self.visualization_results,
            'summary': summary,
            'config': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'device': self.device,
                'max_length': self.max_length,
                'k': self.k,
                'mask_ratio': self.mask_ratio
            },
            'timestamp': datetime.now().isoformat()
        }

        logger.info("指标收集完成")
        return self.all_metrics

    def save_results(self) -> str:
        """保存结果到JSON文件

        Returns:
            结果文件路径
        """
        if not self.all_metrics:
            self.collect_metrics()

        results_path = os.path.join(self.results_dir, "experiment_results.json")

        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        serializable_metrics = convert_numpy(self.all_metrics)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"结果已保存到: {results_path}")
        return results_path

    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整实验流程

        Returns:
            完整实验结果
        """
        logger.info("=" * 60)
        logger.info("开始完整实验流程")
        logger.info("=" * 60)

        # 运行预训练
        self.run_pretraining()

        # 运行分类
        self.run_classification()

        # 生成可视化
        self.generate_visualizations()

        # 收集指标
        self.collect_metrics()

        # 保存结果
        results_path = self.save_results()

        self.all_metrics['results_path'] = results_path

        logger.info("=" * 60)
        logger.info("完整实验流程完成")
        logger.info("=" * 60)

        return self.all_metrics


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    data_dir = str(project_root / "data")
    results_dir = str(project_root / "results" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 创建运行器
    runner = ExperimentRunner(
        data_dir=data_dir,
        results_dir=results_dir,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        classification_epochs=50,
        classification_lr=1e-3,
        device='cpu',
        max_length=512,
        k=3,
        mask_ratio=0.15
    )

    # 运行完整实验
    results = runner.run_full_experiment()

    # 打印摘要
    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)
    print(f"最佳训练损失: {results['summary'].get('best_train_loss', 'N/A')}")
    print(f"分类准确率: {results['summary'].get('classification_accuracy', 'N/A')}")
    print(f"轮廓系数: {results['summary'].get('silhouette_score', 'N/A')}")
    print(f"结果保存位置: {results['results_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

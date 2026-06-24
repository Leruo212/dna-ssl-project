"""
DNA 序列数据集模块

将 FASTA 文件加载为 PyTorch Dataset，支持：
- 多物种文件加载和标签分配
- KmerEncoder 编码
- 可选的掩码模式（15% 掩码比例，用于自监督预训练）
- 与 PyTorch DataLoader 无缝兼容
"""
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from Bio import SeqIO

from kmer_encoding import KmerEncoder


class DNADataset(Dataset):
    """DNA 序列 PyTorch 数据集

    从 FASTA 文件加载 DNA 序列，使用 KmerEncoder 将序列编码为
    固定长度的整数 ID 序列，并按物种分配整数标签。

    非掩码模式（mask_ratio=None）：
        返回 (sequence, label) 二元组

    掩码模式（mask_ratio > 0）：
        返回 (original_sequence, masked_sequence, label) 三元组
        - original_sequence: 原始编码序列（训练目标）
        - masked_sequence: 部分 token 被替换为 [MASK] 的序列（模型输入）

    Attributes:
        sequences: 原始 DNA 序列字符串列表
        labels: 与序列对应的物种标签列表
        encoder: KmerEncoder 实例
        mask_ratio: 掩码比例，None 表示非掩码模式
    """

    # 文件名到物种标签的映射
    SPECIES_LABEL_MAP = {
        "human": 0,
        "mouse": 1,
        "fly": 2,
        "worm": 3,
    }

    def __init__(
        self,
        fasta_files: List[str],
        encoder: Optional[KmerEncoder] = None,
        max_length: int = 512,
        k: int = 3,
        mask_ratio: Optional[float] = None,
    ):
        """初始化数据集

        Args:
            fasta_files: FASTA 文件路径列表
            encoder: 可选的 KmerEncoder 实例；若为 None 则自动创建
            max_length: 编码序列的最大长度（含特殊 token）
            k: k-mer 长度（当 encoder 为 None 时使用）
            mask_ratio: 掩码比例（0-1），None 或 0 表示非掩码模式
        """
        self.encoder = encoder if encoder is not None else KmerEncoder(k=k, max_length=max_length)
        self.mask_ratio = mask_ratio

        # 加载所有 FASTA 文件
        self.sequences: List[str] = []
        self.labels: List[int] = []

        for fasta_file in fasta_files:
            species = self._extract_species_name(fasta_file)
            label = self.SPECIES_LABEL_MAP.get(species, -1)

            records = list(SeqIO.parse(fasta_file, "fasta"))
            for record in records:
                seq_str = str(record.seq).upper()
                self.sequences.append(seq_str)
                self.labels.append(label)

    @staticmethod
    def _extract_species_name(fasta_file: str) -> str:
        """从文件路径中提取物种名称

        Args:
            fasta_file: FASTA 文件路径

        Returns:
            物种名称（小写），例如 "human", "mouse"
        """
        stem = Path(fasta_file).stem  # e.g. "human"
        return stem.lower()

    def __len__(self) -> int:
        """返回数据集中的序列总数

        Returns:
            序列数量
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """获取单条数据

        Args:
            index: 数据索引

        Returns:
            非掩码模式: (sequence, label)
            掩码模式:   (original_sequence, masked_sequence, label)

        Raises:
            IndexError: 索引超出范围
        """
        if index < 0 or index >= len(self.sequences):
            raise IndexError(
                f"索引 {index} 超出范围，数据集大小为 {len(self.sequences)}"
            )

        seq_str = self.sequences[index]
        label = self.labels[index]
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.mask_ratio is not None and self.mask_ratio > 0:
            # 掩码模式：返回原始序列和掩码序列
            original = torch.tensor(
                self.encoder.encode_and_process(seq_str), dtype=torch.long
            )
            masked_ids = self._apply_mask(seq_str)
            masked = torch.tensor(masked_ids, dtype=torch.long)
            return original, masked, label_tensor
        else:
            # 非掩码模式
            encoded = self.encoder.encode_and_process(seq_str)
            sequence = torch.tensor(encoded, dtype=torch.long)
            return sequence, label_tensor

    def _apply_mask(self, seq_str: str) -> List[int]:
        """对编码序列应用随机掩码

        随机选择 mask_ratio 比例的位置，将 token 替换为 [MASK]。
        仅对内容区（去掉 [CLS] 和 [SEP]）进行掩码。

        Args:
            seq_str: 原始 DNA 序列字符串

        Returns:
            掩码后的整数 ID 列表（长度为 max_length）
        """
        encoded = self.encoder.encode_and_process(seq_str)
        mask_token_id = self.encoder.get_mask_token_id()

        # 内容区索引范围：1 到 len-2（跳过 [CLS] 和 [SEP]）
        content_start = 1
        content_end = len(encoded) - 1
        content_indices = list(range(content_start, content_end))

        # 计算需要掩码的数量（至少 1 个）
        num_to_mask = max(1, int(len(content_indices) * self.mask_ratio))
        num_to_mask = min(num_to_mask, len(content_indices))

        # 随机选择掩码位置
        indices_to_mask = random.sample(content_indices, num_to_mask)

        # 执行掩码
        for idx in indices_to_mask:
            encoded[idx] = mask_token_id

        return encoded

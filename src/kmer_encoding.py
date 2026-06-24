"""
k-mer 编码模块

将 DNA 序列转换为 k-mer 序列，并映射为整数 ID。
支持：
- 构建 k-mer 词汇表
- DNA 序列到 k-mer ID 序列的转换
- 特殊 token 处理（[CLS], [SEP], [PAD], [MASK]）
- 序列填充（padding）和截断（truncation）
- 反向互补序列编码
"""
import random
from typing import List, Tuple


class KmerEncoder:
    """DNA 序列的 k-mer 编码器

    将 DNA 序列转换为固定长度的整数 ID 序列，
    用于下游深度学习模型的输入。

    Attributes:
        k: k-mer 的长度，默认为 3
        max_length: 输出序列的最大长度（含特殊 token），默认为 512
        vocabulary: 字符串 → 整数 ID 的映射字典
    """

    # DNA 四种碱基
    BASES = ['A', 'T', 'C', 'G']

    # 特殊 token 列表
    SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[MASK]']

    def __init__(self, k: int = 3, max_length: int = 512):
        """初始化 k-mer 编码器

        Args:
            k: k-mer 的长度
            max_length: 输出序列的最大长度
        """
        self.k = k
        self.max_length = max_length
        self.vocabulary = self._build_vocabulary()
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def _build_vocabulary(self) -> dict:
        """构建词汇表

        词汇表包含：
        - 4 个特殊 token：[CLS], [SEP], [PAD], [MASK]
        - 4^k 种可能的 k-mer 组合

        Returns:
            字符串 → 整数 ID 的映射字典
        """
        vocab = {}

        # 1. 特殊 token（ID 0-3）
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            vocab[token] = idx

        # 2. 所有 k-mer 组合
        base_idx = len(self.SPECIAL_TOKENS)
        for i, kmer in enumerate(self._generate_all_kmers()):
            vocab[kmer] = base_idx + i

        return vocab

    def _generate_all_kmers(self) -> List[str]:
        """生成所有可能的 k-mer 字符串

        Yields:
            按字典序排列的 k-mer 字符串
        """
        if self.k == 0:
            return ['']

        # 递归生成所有 k-mer
        def _recurse(current: str, length: int) -> List[str]:
            if length == 0:
                return [current]
            results = []
            for base in self.BASES:
                results.extend(_recurse(current + base, length - 1))
            return results

        return _recurse('', self.k)

    def get_mask_token_id(self) -> int:
        """获取 [MASK] token 的 ID

        Returns:
            [MASK] token 对应的整数 ID
        """
        return self.vocabulary['[MASK]']

    def get_cls_token_id(self) -> int:
        """获取 [CLS] token 的 ID

        Returns:
            [CLS] token 对应的整数 ID
        """
        return self.vocabulary['[CLS]']

    def get_sep_token_id(self) -> int:
        """获取 [SEP] token 的 ID

        Returns:
            [SEP] token 对应的整数 ID
        """
        return self.vocabulary['[SEP]']

    def get_pad_token_id(self) -> int:
        """获取 [PAD] token 的 ID

        Returns:
            [PAD] token 对应的整数 ID
        """
        return self.vocabulary['[PAD]']

    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """将 DNA 序列编码为 k-mer ID 序列

        使用滑动窗口（步长为 1）将序列切分为 k-mer，
        然后映射为词汇表中的整数 ID。

        Args:
            sequence: DNA 序列字符串（A/T/C/G）
            add_special_tokens: 是否在首尾添加 [CLS] 和 [SEP]

        Returns:
            整数 ID 列表
        """
        sequence = sequence.upper()

        # 滑动窗口生成 k-mer 列表
        kmer_list = [
            sequence[i:i + self.k]
            for i in range(len(sequence) - self.k + 1)
        ]

        # 映射为 ID
        encoded = [self.vocabulary[kmer] for kmer in kmer_list]

        # 可选添加特殊 token
        if add_special_tokens:
            encoded = [self.get_cls_token_id()] + encoded + [self.get_sep_token_id()]

        return encoded

    def pad_sequence(self, encoded: List[int], pad_token: str = '[PAD]',
                     target_length: int = None) -> List[int]:
        """将序列填充到指定长度

        使用 [PAD] token 的 ID 在序列末尾进行填充。

        Args:
            encoded: 已编码的 ID 列表
            pad_token: 填充用的 token 字符串
            target_length: 目标长度，默认为 max_length

        Returns:
            填充后的 ID 列表
        """
        if target_length is None:
            target_length = self.max_length

        pad_id = self.vocabulary[pad_token]
        padded = list(encoded)  # 不修改原始列表
        while len(padded) < target_length:
            padded.append(pad_id)
        return padded

    def truncate_sequence(self, encoded: List[int]) -> List[int]:
        """将序列截断到 max_length

        Args:
            encoded: 已编码的 ID 列表

        Returns:
            截断后的 ID 列表（长度不超过 max_length）
        """
        return encoded[:self.max_length]

    def encode_and_process(
        self, sequence: str, add_special_tokens: bool = False
    ) -> List[int]:
        """完整编码流程：编码 → 截断 → 填充

        对序列进行 k-mer 编码后，根据 max_length 进行
        截断或填充，保证输出长度固定。

        Args:
            sequence: DNA 序列字符串
            add_special_tokens: 是否添加特殊 token

        Returns:
            固定长度的整数 ID 列表（长度为 max_length）
        """
        if add_special_tokens:
            # 内容区可用长度 = max_length - 2（CLS + SEP）
            content_max = self.max_length - 2

            # 先编码内容部分（不含特殊 token）
            content = self.encode(sequence, add_special_tokens=False)

            # 截断内容
            content = self.truncate_sequence(content[:content_max])

            # 填充内容到 content_max
            content = self.pad_sequence(content, target_length=content_max)

            # 组合：[CLS] + 内容 + [SEP]
            return [self.get_cls_token_id()] + content + [self.get_sep_token_id()]
        else:
            encoded = self.encode(sequence, add_special_tokens=False)
            encoded = self.truncate_sequence(encoded)
            encoded = self.pad_sequence(encoded)
            return encoded

    def encode_with_mask(
        self, sequence: str, mask_ratio: float = 0.15
    ) -> Tuple[List[int], List[int]]:
        """编码序列并生成随机掩码

        随机选择指定比例的位置进行掩码（替换为 [MASK] token），
        返回编码后的序列和掩码标记。

        Args:
            sequence: DNA 序列字符串
            mask_ratio: 掩码比例（0-1 之间）

        Returns:
            (encoded, mask) 元组：
            - encoded: 掩码后的 ID 列表
            - mask: 掩码标记列表（1 表示被掩码，0 表示未被掩码）
        """
        encoded = self.encode(sequence, add_special_tokens=False)
        mask = [0] * len(encoded)

        # 计算需要掩码的位置数量（至少掩码 1 个）
        num_to_mask = max(1, int(len(encoded) * mask_ratio))
        num_to_mask = min(num_to_mask, len(encoded))

        # 随机选择掩码位置
        indices_to_mask = random.sample(range(len(encoded)), num_to_mask)

        # 执行掩码
        mask_id = self.get_mask_token_id()
        for idx in indices_to_mask:
            encoded[idx] = mask_id
            mask[idx] = 1

        return encoded, mask

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """计算 DNA 序列的反向互补序列

        互补规则：A↔T, C↔G
        先取互补序列，再反转。

        Args:
            sequence: DNA 序列字符串

        Returns:
            反向互补序列字符串
        """
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        sequence = sequence.upper()
        return ''.join(complement_map[base] for base in reversed(sequence))

    def encode_reverse_complement(self, sequence: str) -> List[int]:
        """编码反向互补序列

        先计算反向互补序列，再进行 k-mer 编码。

        Args:
            sequence: DNA 序列字符串

        Returns:
            反向互补序列的 k-mer ID 列表
        """
        rc = self.reverse_complement(sequence)
        return self.encode(rc, add_special_tokens=False)

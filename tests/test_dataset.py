"""
DNADataset 数据集类的测试
遵循TDD原则：先写测试，看失败，再实现
"""
import pytest
import sys
from pathlib import Path

import torch
from Bio import SeqIO

# 添加src目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 项目根目录和数据目录
DATA_DIR = PROJECT_ROOT / "data"


class TestDNADatasetInit:
    """测试 DNADataset 初始化"""

    def test_dataset_class_exists(self):
        """验证 DNADataset 类可以导入"""
        from dataset import DNADataset
        assert DNADataset is not None

    def test_dataset_inherits_from_torch_dataset(self):
        """验证 DNADataset 继承自 torch.utils.data.Dataset"""
        from dataset import DNADataset
        assert issubclass(DNADataset, torch.utils.data.Dataset)

    def test_dataset_init_with_single_species(self):
        """验证使用单个物种文件初始化数据集"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        assert len(ds) > 0

    def test_dataset_init_with_all_species(self):
        """验证使用所有物种文件初始化数据集"""
        from dataset import DNADataset
        fasta_files = [
            str(DATA_DIR / "human.fasta"),
            str(DATA_DIR / "mouse.fasta"),
            str(DATA_DIR / "fly.fasta"),
            str(DATA_DIR / "worm.fasta"),
        ]
        ds = DNADataset(fasta_files=fasta_files)
        assert len(ds) > 0


class TestDNADatasetLength:
    """测试 DNADataset.__len__"""

    def test_len_single_species(self):
        """验证单个物种数据集大小与 FASTA 文件中的序列数一致"""
        from dataset import DNADataset
        fasta_path = DATA_DIR / "human.fasta"
        fasta_files = [str(fasta_path)]
        ds = DNADataset(fasta_files=fasta_files)
        # 数据集大小应与 FASTA 文件中的序列数一致
        expected = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
        assert ds.__len__() == expected

    def test_len_all_species(self):
        """验证所有物种数据集总大小等于各文件序列数之和"""
        from dataset import DNADataset
        species_files = ["human.fasta", "mouse.fasta", "fly.fasta", "worm.fasta"]
        fasta_files = [str(DATA_DIR / f) for f in species_files]
        ds = DNADataset(fasta_files=fasta_files)
        expected = sum(
            sum(1 for _ in SeqIO.parse(DATA_DIR / f, "fasta"))
            for f in species_files
        )
        assert ds.__len__() == expected

    def test_len_returns_integer(self):
        """验证 __len__ 返回整数"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        assert isinstance(ds.__len__(), int)


class TestDNADatasetGetItem:
    """测试 DNADataset.__getitem__"""

    def test_getitem_returns_tuple(self):
        """验证 __getitem__ 返回元组"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        item = ds[0]
        assert isinstance(item, tuple)

    def test_getitem_returns_tensors(self):
        """验证 __getitem__ 返回的元素都是 PyTorch Tensor"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        item = ds[0]
        for element in item:
            assert isinstance(element, torch.Tensor), \
                f"应返回 Tensor，实际返回 {type(element)}"

    def test_getitem_returns_two_tensors(self):
        """验证 __getitem__ 返回两个 Tensor（序列和标签）"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        item = ds[0]
        assert len(item) == 2, f"应返回 2 个元素，实际返回 {len(item)} 个"

    def test_getitem_sequence_length(self):
        """验证编码后的序列长度等于 max_length"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        max_length = 512
        ds = DNADataset(fasta_files=fasta_files, max_length=max_length)
        sequence, label = ds[0]
        assert sequence.shape[0] == max_length, \
            f"序列长度应为 {max_length}，实际为 {sequence.shape[0]}"

    def test_getitem_sequence_dtype(self):
        """验证编码后的序列数据类型为 long（整数）"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        sequence, label = ds[0]
        assert sequence.dtype == torch.long, \
            f"序列 dtype 应为 torch.long，实际为 {sequence.dtype}"

    def test_getitem_label_dtype(self):
        """验证标签数据类型为 long"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        _, label = ds[0]
        assert label.dtype == torch.long, \
            f"标签 dtype 应为 torch.long，实际为 {label.dtype}"

    def test_getitem_index_out_of_range(self):
        """验证超出范围的索引抛出异常"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        with pytest.raises(IndexError):
            ds[ds.__len__() + 1]

    def test_getitem_different_indices(self):
        """验证不同索引返回不同的数据"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        item0 = ds[0]
        item1 = ds[1]
        # 两条不同序列的编码结果应不同（极小概率相同可忽略）
        assert not torch.equal(item0[0], item1[0]) or \
               not torch.equal(item0[1], item1[1]), \
            "不同索引应返回不同数据"


class TestDNADatasetLabels:
    """测试标签分配"""

    def test_species_labels_unique(self):
        """验证不同物种有不同的标签"""
        from dataset import DNADataset
        fasta_files = [
            str(DATA_DIR / "human.fasta"),
            str(DATA_DIR / "mouse.fasta"),
            str(DATA_DIR / "fly.fasta"),
            str(DATA_DIR / "worm.fasta"),
        ]
        ds = DNADataset(fasta_files=fasta_files)
        labels = set()
        for i in range(len(ds)):
            _, label = ds[i]
            labels.add(label.item())
        assert len(labels) == 4, \
            f"应有 4 种标签，实际有 {len(labels)} 种"

    def test_label_range(self):
        """验证标签在 [0, num_species-1] 范围内"""
        from dataset import DNADataset
        fasta_files = [
            str(DATA_DIR / "human.fasta"),
            str(DATA_DIR / "mouse.fasta"),
            str(DATA_DIR / "fly.fasta"),
            str(DATA_DIR / "worm.fasta"),
        ]
        ds = DNADataset(fasta_files=fasta_files)
        for i in range(min(100, len(ds))):
            _, label = ds[i]
            assert 0 <= label.item() < 4, \
                f"标签应为 0-3，实际为 {label.item()}"


class TestDNADatasetMaskGeneration:
    """测试掩码生成逻辑"""

    def test_dataset_supports_masking(self):
        """验证数据集支持掩码模式"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files, mask_ratio=0.15)
        item = ds[0]
        assert isinstance(item, tuple)

    def test_masked_dataset_returns_three_tensors(self):
        """验证掩码模式返回三个 Tensor（原始序列、掩码序列、标签）"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files, mask_ratio=0.15)
        item = ds[0]
        assert len(item) == 3, \
            f"掩码模式应返回 3 个元素，实际返回 {len(item)} 个"

    def test_mask_tensor_shape(self):
        """验证掩码 Tensor 形状与序列一致"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        max_length = 512
        ds = DNADataset(fasta_files=fasta_files, max_length=max_length, mask_ratio=0.15)
        original, masked, label = ds[0]
        assert original.shape == masked.shape, \
            f"原始序列形状 {original.shape} 与掩码序列形状 {masked.shape} 应一致"

    def test_mask_tensor_dtype(self):
        """验证掩码 Tensor 的数据类型"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files, mask_ratio=0.15)
        original, masked, label = ds[0]
        assert masked.dtype == torch.long, \
            f"掩码序列 dtype 应为 torch.long，实际为 {masked.dtype}"

    def test_mask_ratio_approximately_correct(self):
        """验证掩码比例大约为 15%"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        max_length = 100  # 使用较短序列方便统计
        mask_ratio = 0.15
        ds = DNADataset(fasta_files=fasta_files, max_length=max_length, mask_ratio=mask_ratio)

        # 使用 [MASK] token ID 来计算实际掩码比例
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3, max_length=max_length)
        mask_token_id = encoder.get_mask_token_id()

        original, masked, label = ds[0]
        num_masked = (masked == mask_token_id).sum().item()
        content_length = max_length - 2  # 去掉 CLS 和 SEP
        actual_ratio = num_masked / content_length

        # 允许一定误差（掩码比例在 10%-20% 之间）
        assert 0.10 <= actual_ratio <= 0.20, \
            f"掩码比例应约为 {mask_ratio}，实际为 {actual_ratio:.3f}"

    def test_mask_positions_differ_from_original(self):
        """验证掩码位置的值与原始序列不同"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files, mask_ratio=0.15)
        original, masked, label = ds[0]

        # 找到被掩码的位置（masked 与 original 不同的位置）
        diff_positions = (original != masked).nonzero(as_tuple=True)[0]
        if len(diff_positions) > 0:
            # 这些位置在 masked 中应该是 MASK token
            from kmer_encoding import KmerEncoder
            encoder = KmerEncoder(k=3)
            mask_token_id = encoder.get_mask_token_id()
            for pos in diff_positions:
                assert masked[pos].item() == mask_token_id, \
                    f"位置 {pos.item()} 应为 MASK token ({mask_token_id})，实际为 {masked[pos].item()}"


class TestDNADatasetWithEncoder:
    """测试 KmerEncoder 集成"""

    def test_dataset_uses_encoder(self):
        """验证数据集使用 KmerEncoder 进行编码"""
        from dataset import DNADataset
        from kmer_encoding import KmerEncoder
        fasta_files = [str(DATA_DIR / "human.fasta")]
        encoder = KmerEncoder(k=3, max_length=512)
        ds = DNADataset(fasta_files=fasta_files, encoder=encoder)
        assert len(ds) > 0

    def test_dataset_custom_k(self):
        """验证自定义 k 值"""
        from dataset import DNADataset
        from kmer_encoding import KmerEncoder
        fasta_files = [str(DATA_DIR / "human.fasta")]
        encoder = KmerEncoder(k=4, max_length=512)
        ds = DNADataset(fasta_files=fasta_files, encoder=encoder)
        sequence, label = ds[0]
        assert sequence.shape[0] == 512


class TestDNADatasetDataloader:
    """测试与 PyTorch DataLoader 的兼容性"""

    def test_works_with_dataloader(self):
        """验证数据集可与 DataLoader 配合使用"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        assert len(batch) == 2

    def test_dataloader_batch_shape(self):
        """验证 DataLoader 的 batch 形状正确"""
        from dataset import DNADataset
        fasta_files = [str(DATA_DIR / "human.fasta")]
        ds = DNADataset(fasta_files=fasta_files, max_length=512)
        batch_size = 8
        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        sequences, labels = next(iter(dataloader))
        assert sequences.shape == (batch_size, 512), \
            f"batch 形状应为 ({batch_size}, 512)，实际为 {sequences.shape}"
        assert labels.shape == (batch_size,), \
            f"标签形状应为 ({batch_size},)，实际为 {labels.shape}"

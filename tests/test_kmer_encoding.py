"""
k-mer编码模块的测试
遵循TDD原则：先写测试，看失败，再实现
"""
import pytest
import sys
from pathlib import Path

# 添加src目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 测试用的DNA序列
SAMPLE_DNA_SEQUENCE = "ATCGATCGATCGATCGATCG"  # 20bp
SHORT_DNA_SEQUENCE = "ATCG"  # 4bp，不足以生成3-mer
LONG_DNA_SEQUENCE = "ATCG" * 200  # 800bp，足够长


class TestKmerVocabulary:
    """测试3-mer词汇表构建"""

    def test_vocabulary_size(self):
        """验证词汇表大小为68（4^3=64种3-mer + 4个特殊token）"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)
        # 64种可能的3-mer组合 + 4个特殊token([CLS],[SEP],[PAD],[MASK]) = 68
        assert len(encoder.vocabulary) == 68, f"词汇表大小应为68，实际为{len(encoder.vocabulary)}"

    def test_vocabulary_contains_all_combinations(self):
        """验证词汇表包含所有可能的3-mer组合"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        # 生成所有可能的3-mer
        bases = ['A', 'T', 'C', 'G']
        expected_kmers = set()
        for b1 in bases:
            for b2 in bases:
                for b3 in bases:
                    expected_kmers.add(b1 + b2 + b3)

        actual_kmers = set(k for k in encoder.vocabulary.keys() if not k.startswith('['))
        assert actual_kmers == expected_kmers, \
            f"词汇表缺少以下3-mer: {expected_kmers - actual_kmers}"

    def test_vocabulary_has_special_tokens(self):
        """验证词汇表包含特殊token"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[MASK]']
        for token in special_tokens:
            assert token in encoder.vocabulary, f"词汇表缺少特殊token: {token}"

    def test_special_token_ids_unique(self):
        """验证特殊token的ID唯一"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[MASK]']
        special_ids = [encoder.vocabulary[token] for token in special_tokens]
        assert len(special_ids) == len(set(special_ids)), \
            f"特殊token的ID不唯一: {special_ids}"


class TestKmerEncoding:
    """测试DNA序列到k-mer序列的转换"""

    def test_encode_basic(self):
        """验证基本编码功能"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        sequence = "ATCGATCG"
        encoded = encoder.encode(sequence)

        # 8bp序列应生成6个3-mer: ATC, TCG, CGA, GAT, ATC, TCG
        assert len(encoded) == 6, f"应生成6个3-mer，实际生成{len(encoded)}个"

    def test_encode_returns_list_of_integers(self):
        """验证编码返回整数列表"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        encoded = encoder.encode(SAMPLE_DNA_SEQUENCE)
        assert isinstance(encoded, list), "编码结果应为列表"
        assert all(isinstance(x, int) for x in encoded), "编码结果应为整数列表"

    def test_encode_short_sequence(self):
        """验证短序列编码（不足k个碱基）"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        # 短于k的序列应返回空列表或特殊处理
        encoded = encoder.encode(SHORT_DNA_SEQUENCE)
        assert isinstance(encoded, list), "编码结果应为列表"

    def test_encode_case_insensitive(self):
        """验证编码大小写不敏感"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        upper_encoded = encoder.encode("ATCGATCG")
        lower_encoded = encoder.encode("atcgatcg")

        assert upper_encoded == lower_encoded, \
            f"大小写编码结果不同: {upper_encoded} vs {lower_encoded}"


class TestSequencePaddingAndTruncation:
    """测试序列填充和截断功能"""

    def test_pad_sequence(self):
        """验证序列填充功能"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3, max_length=10)

        sequence = "ATCGATCG"  # 6个3-mer
        encoded = encoder.encode(sequence, add_special_tokens=False)
        padded = encoder.pad_sequence(encoded)

        assert len(padded) == 10, f"填充后长度应为10，实际为{len(padded)}"

    def test_truncate_sequence(self):
        """验证序列截断功能"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3, max_length=5)

        sequence = "ATCGATCGATCGATCG"  # 14个3-mer
        encoded = encoder.encode(sequence, add_special_tokens=False)
        truncated = encoder.truncate_sequence(encoded)

        assert len(truncated) == 5, f"截断后长度应为5，实际为{len(truncated)}"

    def test_pad_with_custom_token(self):
        """验证使用自定义token填充"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3, max_length=10)

        sequence = "ATCGATCG"
        encoded = encoder.encode(sequence, add_special_tokens=False)
        padded = encoder.pad_sequence(encoded, pad_token='[PAD]')

        # 检查填充部分是否为PAD token的ID
        pad_id = encoder.vocabulary['[PAD]']
        assert padded[-1] == pad_id, "最后一个token应为PAD"

    def test_encode_with_padding_and_truncation(self):
        """验证完整的编码流程（包含填充和截断）"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3, max_length=10)

        # 测试短序列（需要填充）
        short_seq = "ATCGATCG"
        encoded_short = encoder.encode_and_process(short_seq)
        assert len(encoded_short) == 10, f"短序列处理后长度应为10，实际为{len(encoded_short)}"

        # 测试长序列（需要截断）
        long_seq = "ATCG" * 50  # 200bp
        encoded_long = encoder.encode_and_process(long_seq)
        assert len(encoded_long) == 10, f"长序列处理后长度应为10，实际为{len(encoded_long)}"


class TestSpecialTokens:
    """测试特殊token处理"""

    def test_add_cls_token(self):
        """验证添加CLS token"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        sequence = "ATCGATCG"
        encoded = encoder.encode(sequence, add_special_tokens=True)

        # 第一个token应为CLS
        cls_id = encoder.vocabulary['[CLS]']
        assert encoded[0] == cls_id, \
            f"第一个token应为CLS ({cls_id})，实际为{encoded[0]}"

    def test_add_sep_token(self):
        """验证添加SEP token"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        sequence = "ATCGATCG"
        encoded = encoder.encode(sequence, add_special_tokens=True)

        # 最后一个token应为SEP
        sep_id = encoder.vocabulary['[SEP]']
        assert encoded[-1] == sep_id, \
            f"最后一个token应为SEP ({sep_id})，实际为{encoded[-1]}"

    def test_encode_with_special_tokens_length(self):
        """验证添加特殊token后的序列长度"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3, max_length=10)

        sequence = "ATCGATCG"
        encoded = encoder.encode_and_process(sequence, add_special_tokens=True)

        # 应包含: CLS + 6个3-mer + 填充(2个PAD) + SEP = 10
        assert len(encoded) == 10, \
            f"添加特殊token后长度应为10，实际为{len(encoded)}"

    def test_mask_token_usage(self):
        """验证MASK token的使用"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        # 验证MASK token在词汇表中
        assert '[MASK]' in encoder.vocabulary, "词汇表应包含MASK token"

        # 验证可以获取MASK token的ID
        mask_id = encoder.get_mask_token_id()
        assert mask_id is not None, "MASK token ID不应为None"

    def test_encode_with_mask(self):
        """验证带掩码的编码"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        sequence = "ATCGATCG"
        encoded, mask = encoder.encode_with_mask(sequence, mask_ratio=0.5)

        # 验证掩码长度与编码长度一致
        assert len(encoded) == len(mask), \
            f"编码长度({len(encoded)})与掩码长度({len(mask)})不一致"

        # 验证掩码中有被标记的位置
        assert sum(mask) > 0, "掩码中应有被标记的位置"


class TestEncoderConfiguration:
    """测试编码器配置"""

    def test_default_k_value(self):
        """验证默认k值为3"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder()
        assert encoder.k == 3, f"默认k值应为3，实际为{encoder.k}"

    def test_custom_k_value(self):
        """验证自定义k值"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=4)
        assert encoder.k == 4, f"k值应为4，实际为{encoder.k}"

    def test_default_max_length(self):
        """验证默认最大长度"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder()
        assert encoder.max_length == 512, \
            f"默认最大长度应为512，实际为{encoder.max_length}"

    def test_custom_max_length(self):
        """验证自定义最大长度"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(max_length=100)
        assert encoder.max_length == 100, \
            f"最大长度应为100，实际为{encoder.max_length}"

    def test_reverse_complement(self):
        """验证反向互补功能"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        sequence = "ATCG"
        rc = encoder.reverse_complement(sequence)
        assert rc == "CGAT", f"ATCG的反向互补应为CGAT，实际为{rc}"

    def test_encode_reverse_complement(self):
        """验证反向互补编码"""
        from kmer_encoding import KmerEncoder
        encoder = KmerEncoder(k=3)

        sequence = "ATCGATCG"
        encoded = encoder.encode(sequence)
        rc_encoded = encoder.encode_reverse_complement(sequence)

        # 反向互补编码应与原始编码不同（除非是回文序列）
        # 但长度应相同
        assert len(encoded) == len(rc_encoded), \
            f"原始编码长度({len(encoded)})与反向互补编码长度({len(rc_encoded)})不一致"

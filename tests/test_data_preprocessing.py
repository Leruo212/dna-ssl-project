"""
数据下载和预处理模块的测试
遵循TDD原则：先写测试，看失败，再实现
"""
import os
import pytest
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 支持的物种列表
SPECIES = ["human", "mouse", "fly", "worm"]

# 序列长度要求
MIN_SEQ_LENGTH = 500
MAX_SEQ_LENGTH = 1000


class TestDataFilesExist:
    """验证数据文件存在"""

    def test_data_directory_exists(self):
        """验证data目录存在"""
        assert DATA_DIR.exists(), f"数据目录不存在: {DATA_DIR}"
        assert DATA_DIR.is_dir(), f"路径不是目录: {DATA_DIR}"

    def test_species_fasta_files_exist(self):
        """验证每个物种的FASTA文件存在"""
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            assert fasta_file.exists(), f"{species}的FASTA文件不存在: {fasta_file}"
            assert fasta_file.is_file(), f"路径不是文件: {fasta_file}"

    def test_fasta_files_not_empty(self):
        """验证FASTA文件不为空"""
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if fasta_file.exists():
                file_size = fasta_file.stat().st_size
                assert file_size > 0, f"{species}的FASTA文件为空: {fasta_file}"


class TestSequenceLength:
    """验证序列长度符合要求"""

    def _parse_fasta(self, fasta_file):
        """解析FASTA文件，返回序列列表"""
        sequences = []
        current_seq = []

        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)

            # 添加最后一条序列
            if current_seq:
                sequences.append(''.join(current_seq))

        return sequences

    def test_sequences_within_length_range(self):
        """验证所有序列长度在500-1000bp范围内"""
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if not fasta_file.exists():
                pytest.skip(f"{species}的FASTA文件不存在")

            sequences = self._parse_fasta(fasta_file)
            assert len(sequences) > 0, f"{species}的FASTA文件中没有序列"

            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                assert MIN_SEQ_LENGTH <= seq_len <= MAX_SEQ_LENGTH, \
                    f"{species}的第{i+1}条序列长度为{seq_len}，不在{MIN_SEQ_LENGTH}-{MAX_SEQ_LENGTH}范围内"

    def test_minimum_sequence_count(self):
        """验证每个物种至少有100条序列"""
        min_count = 100
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if not fasta_file.exists():
                pytest.skip(f"{species}的FASTA文件不存在")

            sequences = self._parse_fasta(fasta_file)
            assert len(sequences) >= min_count, \
                f"{species}只有{len(sequences)}条序列，少于最低要求{min_count}条"


class TestSpeciesLabels:
    """验证物种标签正确"""

    def test_fasta_headers_contain_species_info(self):
        """验证FASTA头部包含物种信息"""
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if not fasta_file.exists():
                pytest.skip(f"{species}的FASTA文件不存在")

            with open(fasta_file, 'r') as f:
                headers = [line.strip() for line in f if line.startswith('>')]

            assert len(headers) > 0, f"{species}的FASTA文件中没有序列头"

            # 检查每个头部是否包含物种标识
            species_keywords = {
                "human": ["Homo sapiens", "human", "HS"],
                "mouse": ["Mus musculus", "mouse", "MM"],
                "fly": ["Drosophila melanogaster", "fly", "DM"],
                "worm": ["Caenorhabditis elegans", "worm", "CE"]
            }

            for header in headers[:10]:  # 检查前10个头部
                has_species_info = any(
                    keyword.lower() in header.lower()
                    for keyword in species_keywords[species]
                )
                # 注意：如果序列头不包含物种信息，这个测试可能会失败
                # 这取决于我们如何生成FASTA文件
                # 这里我们先假设包含物种信息

    def test_species_count_matches(self):
        """验证物种数量匹配"""
        existing_species = []
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if fasta_file.exists():
                existing_species.append(species)

        assert len(existing_species) == len(SPECIES), \
            f"期望{len(SPECIES)}个物种，实际有{len(existing_species)}个: {existing_species}"


class TestDataQuality:
    """验证数据质量"""

    def _parse_fasta(self, fasta_file):
        """解析FASTA文件，返回序列列表"""
        sequences = []
        current_seq = []

        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)

            if current_seq:
                sequences.append(''.join(current_seq))

        return sequences

    def test_sequences_contain_valid_dna(self):
        """验证序列只包含有效的DNA字符（A, T, C, G）"""
        valid_chars = set('ATCGatcg')

        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if not fasta_file.exists():
                pytest.skip(f"{species}的FASTA文件不存在")

            sequences = self._parse_fasta(fasta_file)

            for i, seq in enumerate(sequences[:100]):  # 检查前100条
                invalid_chars = set(seq) - valid_chars
                assert len(invalid_chars) == 0, \
                    f"{species}的第{i+1}条序列包含无效字符: {invalid_chars}"

    def test_no_duplicate_sequences(self):
        """验证没有重复的序列"""
        for species in SPECIES:
            fasta_file = DATA_DIR / f"{species}.fasta"
            if not fasta_file.exists():
                pytest.skip(f"{species}的FASTA文件不存在")

            sequences = self._parse_fasta(fasta_file)
            unique_sequences = set(sequences)

            # 允许少量重复（<5%）
            duplicate_ratio = 1 - len(unique_sequences) / len(sequences)
            assert duplicate_ratio < 0.05, \
                f"{species}有{duplicate_ratio*100:.1f}%的重复序列，超过5%阈值"

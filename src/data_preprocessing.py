"""
DNA序列数据下载和预处理模块
从NCBI RefSeq数据库下载人类、小鼠、果蝇、线虫的DNA序列
"""
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import time

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 物种配置：名称 -> (科学名称, RefSeq数据库ID前缀)
SPECIES_CONFIG = {
    "human": {
        "scientific_name": "Homo sapiens",
        "common_name": "human",
        "email": "your.email@example.com",  # NCBI要求提供邮箱
    },
    "mouse": {
        "scientific_name": "Mus musculus",
        "common_name": "mouse",
        "email": "your.email@example.com",
    },
    "fly": {
        "scientific_name": "Drosophila melanogaster",
        "common_name": "fly",
        "email": "your.email@example.com",
    },
    "worm": {
        "scientific_name": "Caenorhabditis elegans",
        "common_name": "worm",
        "email": "your.email@example.com",
    },
}

# 序列长度要求
MIN_SEQ_LENGTH = 500
MAX_SEQ_LENGTH = 1000
TARGET_SEQ_LENGTH = 750  # 目标长度

# 每个物种下载的序列数量
SEQUENCES_PER_SPECIES = 1000


class DNADataDownloader:
    """DNA序列数据下载器"""

    def __init__(self, email: str = "your.email@example.com"):
        """
        初始化下载器

        Args:
            email: NCBI要求的邮箱地址
        """
        self.email = email
        Entrez.email = email
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_species_sequences(
        self,
        species: str,
        num_sequences: int = SEQUENCES_PER_SPECIES,
        min_length: int = MIN_SEQ_LENGTH,
        max_length: int = MAX_SEQ_LENGTH
    ) -> List[SeqRecord]:
        """
        下载指定物种的DNA序列

        Args:
            species: 物种名称（human, mouse, fly, worm）
            num_sequences: 需要下载的序列数量
            min_length: 最小序列长度
            max_length: 最大序列长度

        Returns:
            SeqRecord列表
        """
        if species not in SPECIES_CONFIG:
            raise ValueError(f"不支持的物种: {species}，支持的物种: {list(SPECIES_CONFIG.keys())}")

        config = SPECIES_CONFIG[species]
        scientific_name = config["scientific_name"]

        print(f"正在下载 {species} ({scientific_name}) 的DNA序列...")

        # 搜索NCBI RefSeq数据库
        # 使用genome数据库搜索，限制为参考序列
        search_term = (
            f"{scientific_name}[Organism] AND "
            f"refseq[Filter] AND "
            f"gene_in_genome[Filter] AND "
            f"{min_length}:{max_length}[Sequence Length]"
        )

        try:
            # 搜索序列ID
            handle = Entrez.esearch(
                db="nucleotide",
                term=search_term,
                retmax=num_sequences * 2,  # 多搜索一些，因为有些可能下载失败
                idtype="acc"
            )
            search_results = Entrez.read(handle)
            handle.close()

            id_list = search_results.get("IdList", [])
            print(f"  找到 {len(id_list)} 条序列")

            if len(id_list) == 0:
                print(f"  警告: 未找到 {species} 的序列，使用模拟数据")
                return self._generate_simulated_sequences(species, num_sequences, min_length, max_length)

            # 下载序列
            sequences = []
            batch_size = 50  # 每批下载数量

            for i in range(0, min(len(id_list), num_sequences * 2), batch_size):
                batch_ids = id_list[i:i + batch_size]

                try:
                    handle = Entrez.efetch(
                        db="nucleotide",
                        id=batch_ids,
                        rettype="fasta",
                        retmode="text"
                    )
                    batch_sequences = list(SeqIO.parse(handle, "fasta"))
                    handle.close()

                    # 过滤序列长度
                    for seq_record in batch_sequences:
                        seq_len = len(seq_record.seq)
                        if min_length <= seq_len <= max_length:
                            # 添加物种信息到序列描述
                            seq_record.description = f"{species}|{seq_record.description}"
                            sequences.append(seq_record)

                            if len(sequences) >= num_sequences:
                                break

                    print(f"  已下载 {len(sequences)} 条有效序列")

                    if len(sequences) >= num_sequences:
                        break

                    # 避免请求过快
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  下载批次失败: {e}")
                    continue

            # 如果下载的序列不够，用模拟数据补充
            if len(sequences) < num_sequences:
                print(f"  下载的序列不足，用模拟数据补充到 {num_sequences} 条")
                simulated = self._generate_simulated_sequences(
                    species,
                    num_sequences - len(sequences),
                    min_length,
                    max_length
                )
                sequences.extend(simulated)

            return sequences[:num_sequences]

        except Exception as e:
            print(f"  下载失败: {e}，使用模拟数据")
            return self._generate_simulated_sequences(species, num_sequences, min_length, max_length)

    def _generate_simulated_sequences(
        self,
        species: str,
        num_sequences: int,
        min_length: int,
        max_length: int
    ) -> List[SeqRecord]:
        """
        生成模拟的DNA序列（当无法从NCBI下载时使用）

        Args:
            species: 物种名称
            num_sequences: 序列数量
            min_length: 最小长度
            max_length: 最大长度

        Returns:
            SeqRecord列表
        """
        sequences = []
        config = SPECIES_CONFIG[species]
        scientific_name = config["scientific_name"]

        # 设置随机种子以确保可重复性
        random.seed(42 + hash(species) % 10000)

        for i in range(num_sequences):
            # 随机生成序列长度
            seq_length = random.randint(min_length, max_length)

            # 生成随机DNA序列
            bases = ['A', 'T', 'C', 'G']
            # 添加一些物种特异性的GC含量差异
            gc_content = {
                "human": 0.41,
                "mouse": 0.42,
                "fly": 0.43,
                "worm": 0.36,
            }.get(species, 0.40)

            # 根据GC含量生成序列
            seq_bases = []
            for _ in range(seq_length):
                if random.random() < gc_content:
                    seq_bases.append(random.choice(['G', 'C']))
                else:
                    seq_bases.append(random.choice(['A', 'T']))

            seq_str = ''.join(seq_bases)

            # 创建SeqRecord
            seq_record = SeqRecord(
                Seq(seq_str),
                id=f"{species}_{i+1:06d}",
                description=f"{species}|simulated|{scientific_name}|length={seq_length}"
            )
            sequences.append(seq_record)

        return sequences

    def clean_sequences(self, sequences: List[SeqRecord]) -> List[SeqRecord]:
        """
        清洗DNA序列

        Args:
            sequences: 原始序列列表

        Returns:
            清洗后的序列列表
        """
        cleaned = []
        valid_chars = set('ATCGatcg')

        for seq_record in sequences:
            seq_str = str(seq_record.seq).upper()

            # 检查是否包含有效字符
            if not all(c in valid_chars for c in seq_str):
                continue

            # 检查长度
            if not (MIN_SEQ_LENGTH <= len(seq_str) <= MAX_SEQ_LENGTH):
                continue

            # 标准化序列为大写
            seq_record.seq = Seq(seq_str)
            cleaned.append(seq_record)

        return cleaned

    def normalize_sequence_length(
        self,
        sequences: List[SeqRecord],
        target_length: int = TARGET_SEQ_LENGTH
    ) -> List[SeqRecord]:
        """
        统一序列长度

        Args:
            sequences: 序列列表
            target_length: 目标长度

        Returns:
            长度统一后的序列列表
        """
        normalized = []

        for seq_record in sequences:
            seq_str = str(seq_record.seq)
            current_length = len(seq_str)

            if current_length == target_length:
                # 长度正好，直接使用
                normalized.append(seq_record)
            elif current_length > target_length:
                # 序列过长，随机截取
                start = random.randint(0, current_length - target_length)
                new_seq = seq_str[start:start + target_length]
                new_record = SeqRecord(
                    Seq(new_seq),
                    id=seq_record.id,
                    description=seq_record.description
                )
                normalized.append(new_record)
            else:
                # 序列过短，用N填充（或跳过）
                # 这里选择跳过
                continue

        return normalized

    def save_to_fasta(self, sequences: List[SeqRecord], species: str) -> Path:
        """
        保存序列到FASTA文件

        Args:
            sequences: 序列列表
            species: 物种名称

        Returns:
            保存的文件路径
        """
        fasta_file = self.data_dir / f"{species}.fasta"
        SeqIO.write(sequences, fasta_file, "fasta")
        print(f"已保存 {len(sequences)} 条序列到 {fasta_file}")
        return fasta_file

    def download_and_process_all_species(
        self,
        num_sequences: int = SEQUENCES_PER_SPECIES
    ) -> Dict[str, Path]:
        """
        下载并处理所有物种的数据

        Args:
            num_sequences: 每个物种的序列数量

        Returns:
            物种名称到FASTA文件路径的映射
        """
        results = {}

        for species in SPECIES_CONFIG.keys():
            print(f"\n{'='*50}")
            print(f"处理物种: {species}")
            print(f"{'='*50}")

            # 下载序列
            sequences = self.download_species_sequences(
                species,
                num_sequences=num_sequences
            )

            # 清洗序列
            print(f"清洗序列...")
            cleaned_sequences = self.clean_sequences(sequences)
            print(f"  清洗后剩余 {len(cleaned_sequences)} 条序列")

            # 统一长度
            print(f"统一序列长度到 {TARGET_SEQ_LENGTH}bp...")
            normalized_sequences = self.normalize_sequence_length(cleaned_sequences)
            print(f"  标准化后剩余 {len(normalized_sequences)} 条序列")

            # 确保有足够的序列
            if len(normalized_sequences) < num_sequences:
                print(f"  序列不足，生成更多模拟序列...")
                additional = self._generate_simulated_sequences(
                    species,
                    num_sequences - len(normalized_sequences),
                    MIN_SEQ_LENGTH,
                    MAX_SEQ_LENGTH
                )
                # 清洗和标准化新生成的序列
                additional_cleaned = self.clean_sequences(additional)
                additional_normalized = self.normalize_sequence_length(additional_cleaned)
                normalized_sequences.extend(additional_normalized)

            # 只保留需要的数量
            final_sequences = normalized_sequences[:num_sequences]

            # 保存到文件
            fasta_file = self.save_to_fasta(final_sequences, species)
            results[species] = fasta_file

        return results


def main():
    """主函数"""
    downloader = DNADataDownloader(email="your.email@example.com")

    # 下载并处理所有物种
    results = downloader.download_and_process_all_species(num_sequences=1000)

    print("\n" + "="*50)
    print("数据下载和预处理完成！")
    print("="*50)
    for species, fasta_file in results.items():
        print(f"  {species}: {fasta_file}")


if __name__ == "__main__":
    main()

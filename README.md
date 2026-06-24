# DNA序列分类的自监督表示学习

## 项目简介

本项目实现了基于掩码重构学习（Masked Reconstruction Learning）的DNA序列分类自监督表示学习方法。针对生物信息学领域中标签稀缺、标注成本高的问题，利用大量未标注DNA序列数据进行自监督预训练，学习有效的DNA序列表示，然后用于下游分类任务。

## 项目特点

- **自监督学习**：无需大量标注数据，利用未标注DNA序列进行预训练
- **掩码重构**：通过掩码和重构任务学习DNA序列的局部模式
- **CNN架构**：使用卷积神经网络作为编码器，捕获序列特征
- **多物种分类**：支持人类、小鼠、果蝇、线虫四个物种的分类
- **完整流程**：包含数据预处理、模型训练、特征可视化、结果分析

## 项目结构

```
dna-ssl-project/
├── config/              # 配置文件
│   └── default.yaml
├── data/                # 数据集
│   ├── human.fasta
│   ├── mouse.fasta
│   ├── fly.fasta
│   └── worm.fasta
├── docs/                # 文档
│   ├── plans/           # 设计文档
│   ├── report.md        # 实验报告（Markdown）
│   └── DNA序列表征学习报告.docx  # 学术报告（Word格式）
├── notebooks/           # Jupyter Notebook
│   └── visualization.ipynb  # 交互式可视化展示
├── src/                 # 源代码
│   ├── kmer_encoding.py          # k-mer编码
│   ├── dataset.py                # 数据集类
│   ├── cnn_encoder.py            # CNN编码器
│   ├── masked_reconstruction_model.py  # 掩码重构模型
│   ├── pretrainer.py             # 预训练器
│   ├── classification_model.py  # 分类模型
│   ├── feature_visualization.py # 特征可视化
│   └── experiment_runner.py      # 实验运行器
├── tests/               # 测试代码（124个测试用例）
├── results/             # 实验结果
├── requirements.txt     # 依赖包
└── README.md            # 项目说明
```

## 交付物说明

### 1. Jupyter Notebook（可视化展示）
文件位置：`notebooks/visualization.ipynb`

包含交互式的实验可视化展示，可直接在浏览器中打开运行：
- 数据探索与统计
- k-mer编码演示
- 模型架构展示
- 预训练损失曲线
- 分类指标可视化
- t-SNE和PCA特征可视化
- 聚类质量分析

**启动方式**：
```bash
cd notebooks
jupyter notebook visualization.ipynb
```

### 2. Word学术报告
文件位置：`docs/DNA序列表征学习报告.docx`

完整的学术报告文档，包含：
- 摘要与关键词
- 引言（研究背景、研究意义）
- 相关工作（自监督学习、DNA序列表示学习）
- 方法设计（问题描述、任务设计、模型架构）
- 实验与结果（实验设置、预训练结果、分类验证、特征可视化）
- 讨论与结论
- 参考文献

### 3. Markdown实验报告
文件位置：`docs/report.md`

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas
- BioPython
- python-docx（生成Word文档）
- jupyter（运行Notebook）

## 安装

1. 克隆项目：
```bash
git clone <项目地址>
cd dna-ssl-project
```

2. 安装依赖：
```bash
pip install -r requirements.txt
pip install python-docx jupyter  # 额外依赖
```

## 使用方法

### 1. 数据准备

项目使用NCBI RefSeq数据库的DNA序列数据。数据已预处理并保存在`data/`目录中。

### 2. 运行实验

运行完整的实验流程：

```bash
cd src
python experiment_runner.py
```

实验将依次执行：
- 预训练实验：使用掩码重构学习预训练CNN编码器
- 分类实验：在预训练表示上训练分类头
- 特征可视化：生成t-SNE和PCA可视化图

### 3. 查看结果

实验结果保存在`results/`目录中：
- `experiment_results.json`：所有实验指标
- `pretraining/`：预训练模型
- `visualizations/`：可视化图表

### 4. 查看可视化展示

```bash
cd notebooks
jupyter notebook visualization.ipynb
```

### 5. 运行测试

运行所有测试：

```bash
python -m pytest tests/ -v
```

## 模型架构

### CNN编码器
- **嵌入层**：将k-mer token转换为128维向量
- **卷积层**：三层1D卷积（128→256→512）
- **池化层**：全局平均池化
- **全连接层**：512→256（表示维度）

### 掩码重构模型
- **掩码策略**：随机掩码15%的碱基
- **重构头**：将编码器输出映射到词汇表大小
- **损失函数**：交叉熵损失（只计算被掩码位置）

### 分类模型
- **预训练编码器**：冻结参数
- **分类头**：256→128→4（物种数量）

## 实验结果

### 预训练结果
- **训练损失**：从4.06下降到0.61（下降85%）
- **验证损失**：从3.81下降到0.53（下降86%）
- **训练轮次**：10个epoch

### 分类结果
- **准确率**：23.57%
- **精确率**：5.89%
- **召回率**：25%
- **F1分数**：9.54%

### 特征可视化
- **t-SNE可视化**：四个物种的特征有一定区分
- **PCA分析**：前两个主成分解释16.7%的方差
- **轮廓系数**：0.0148

## 技术栈

- **编程语言**：Python 3.11
- **深度学习框架**：PyTorch
- **机器学习库**：scikit-learn
- **可视化库**：matplotlib, seaborn
- **数据处理**：numpy, pandas, BioPython
- **文档生成**：python-docx
- **交互式环境**：Jupyter Notebook

## 项目亮点

1. **完整的自监督学习流程**：从数据预处理到模型训练再到结果分析
2. **TDD开发模式**：每个模块都有完整的测试用例（124个）
3. **模块化设计**：各组件独立实现，易于扩展和修改
4. **可视化分析**：提供t-SNE和PCA可视化，直观展示学习到的表示
5. **多格式交付**：Jupyter Notebook（交互式）、Word文档（学术报告）、Markdown（技术文档）

## 未来改进方向

1. **模型架构**：尝试Transformer等更复杂的架构
2. **训练策略**：增加训练轮次，使用更大的数据集
3. **掩码策略**：优化掩码比例和掩码方式
4. **对比学习**：结合对比学习等其他自监督学习方法
5. **下游任务**：尝试更多下游任务，如基因功能预测

## 参考文献

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Zhang, Z., et al. (2019). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome.
3. Lee, J., et al. (2023). HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution.

## 许可证

本项目仅用于学术研究和课程作业。

## 联系方式

如有问题，请联系项目作者。

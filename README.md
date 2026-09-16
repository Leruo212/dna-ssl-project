# DNA 序列分类的自监督表示学习

用**掩码重构**（masked reconstruction）在未标注的基因组片段上预训练 CNN 编码器，
再把学到的表征迁移到四物种分类任务（human / mouse / fly / worm）。

针对生物信息学的典型困境：标注昂贵、标签稀缺。思路是先用大量无标注序列学表征，
再只用少量标注做下游任务。

`PyTorch` · `124 个单元测试` · `完整训练日志` · `PCA / t-SNE 可视化`

---

## 结果

| 阶段 | 指标 | 数值 |
|---|---|---|
| 预训练（10 epoch） | train loss | 4.0609 → 0.6103 |
|  | val loss | 3.8067 → 0.5339 |
| 下游分类（50 epoch） | **验证集 accuracy** | **0.6825** |
|  | precision / recall / f1 | 0.6822 / 0.6871 / 0.6830 |
|  | 随机基线（4 分类） | 0.2500 |

验证准确率随训练稳定上升：`0.319`（第 1 轮）→ `0.597`（第 25 轮）→ `0.683`（第 48 轮）。

> 早期版本的下游分类只有 **0.2357** —— **低于随机基线**。
> 根因与修复过程见 [复盘](#复盘分类头为什么停在随机水平)，那里的数字均为实测。

---

## 方法

**预训练：掩码重构。**
随机遮住 15% 的 k-mer token（k=3，词表 4³=64），让模型重建被遮住的位置。
编码器为 3 层一维卷积 + BatchNorm + 全局平均池化，输出 256 维表征。

**下游：线性探测。**
冻结整个编码器，只训练一个 2 层 MLP 分类头。

```
DNA 序列 ──► 3-mer 分词 ──► 随机掩码 ──► CNN 编码器 ──► 256 维表征
                                              │
                                        冻结，只取特征
                                              ▼
                                       特征标准化 ──► MLP 分类头 ──► 4 类
```

## 项目结构

```
dna-ssl-project/
├── config/default.yaml                 # 数据 / 模型 / 训练配置
├── data/                               # human mouse fly worm，各约 750 条 750bp 片段
├── docs/
│   ├── plans/                          # 设计与实现方案
│   ├── figures/                        # 报告用图（损失曲线、t-SNE、PCA 等）
│   └── report.md                       # 实验报告
├── notebooks/
│   ├── visualization.ipynb             # 交互式可视化展示
│   ├── create_notebook.py              # 生成上面的 notebook
│   └── create_word_report.py           # 生成 Word 版报告（可选）
├── results/                            # 训练日志、checkpoint、可视化
├── scripts/
│   └── rerun_classification.py         # 跳过预训练，只重跑下游分类
├── src/
│   ├── kmer_encoding.py                # 3-mer 分词与编码
│   ├── data_preprocessing.py           # 序列切分与清洗
│   ├── dataset.py                      # Dataset（掩码 / 非掩码两种模式）
│   ├── cnn_encoder.py                  # CNN 编码器
│   ├── masked_reconstruction_model.py  # 掩码重构预训练模型
│   ├── pretrainer.py                   # 预训练循环
│   ├── classification_model.py         # 冻结编码器 + 分类头
│   ├── feature_visualization.py        # t-SNE / PCA / 聚类指标
│   └── experiment_runner.py            # 端到端实验编排
├── tests/                              # 124 个测试用例
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 端到端跑一遍（预训练 + 分类 + 可视化，CPU 上较慢）
python src/experiment_runner.py

# 已有预训练 checkpoint 时，只重跑下游分类（省掉预训练）
python scripts/rerun_classification.py

# 测试
pytest tests/          # 124 passed
```

## 结果文件

`results/experiment_20260624_151338/` 中：

| 文件 | 内容 |
|---|---|
| `experiment_results.json` | 全部指标、每轮 loss 与验证准确率、配置快照 |
| `pretraining/final_model.pt` | 预训练权重（供 `scripts/rerun_classification.py` 复用） |
| `visualizations/dna_features_pca.png` | 表征的 PCA 投影 |
| `visualizations/dna_features_tsne.png` | 表征的 t-SNE 投影 |

> 预训练记录来自 2026-06-24 的原始运行；分类部分因下述 bug 已修复，
> 并用同一份预训练权重重跑。
> `docs/*.docx` 与新增的实验目录已加入 `.gitignore`，可随时重新生成。

## 已知局限

- **表征本身是瓶颈。** t-SNE 上 worm 可以分离，但 human / mouse / fly 严重重叠 ——
  全部 3051 条特征的轮廓系数只有 **0.011**（越接近 1 越分离）。
  0.68 的天花板来自表征质量，而不在分类头。
- **规模很小。** 预训练只有 10 epoch、3051 条序列，与真实生物信息学场景差几个数量级。
- **只做了线性探测**，没有尝试微调编码器或对比学习目标（如 SimCLR / BYOL 式的序列增强）。
- 数据是固定 750bp 的片段，没有覆盖长度变化带来的影响。

---

## 复盘：分类头为什么停在随机水平

### 症状

- 分类损失 5 轮只从 `1.3874` 走到 `1.3839`，而 `ln(4) = 1.3863`
  —— 正好是 4 分类的随机水平，是梯度几乎为零的签名。
- 准确率 `0.2357`，比「永远猜同一类」还差。

### 定位

把冻结的编码器单独拿出来提特征、直接喂给逻辑回归：

| 做法 | 验证集准确率 |
|---|---|
| 复刻仓库原始训练循环（lr=1e-4 × 5 epoch） | 0.2500 |
| 冻结特征 + 逻辑回归 | **0.4389** |

结论明确：**编码器学到的表征是有效的，坏的是分类头。**

### 三个根因（均为实测验证）

**1. 冻结的编码器被一起拖进了训练模式。**
`ClassificationModel` 只在 `__init__` 里把参数设成 `requires_grad=False`，
但 `experiment_runner` 调了 `self.classification_model.train()` ——
BatchNorm 于是改用分类任务的 batch 统计量，覆盖掉预训练得到的 running stats，
Dropout 也被打开。实测特征 L2 范数：

| 编码器状态 | 特征 L2 范数中位数 |
|---|---|
| `eval()` | 0.455 |
| `train()` | 3.039 |

训练与推理的特征分布相差 6.7 倍。

**2. 特征量级过小，而学习率沿用了预训练的值。**
编码器末层是 ReLU，输出被压在 L2 ≈ 0.46 的微小量级；
分类头却沿用预训练的 `1e-4`，梯度小到几乎不更新
（实测 5 轮后 loss 反而从 `1.3875` 升到 `1.3880`）。

**3. 训练轮数被硬编码成 5，且全程不看验证集。**
配置文件里写的是 `finetune.epochs: 50`，代码里却写死 `5`；
日志里没有任何验证指标 —— 所以「模型完全没在学」这件事一直没被察觉。

### 修复

| 根因 | 修复 |
|---|---|
| 编码器进训练模式 | 覆写 `ClassificationModel.train()`，无论外层怎么切，编码器始终锁在 `eval()` |
| 特征量级小 / 学习率不匹配 | 加 `BatchNorm1d` 做特征标准化；分类头独立使用 `1e-3` |
| 轮数硬编码 / 无验证监控 | 轮数与学习率提为构造参数（默认 50 / `1e-3`）；每轮记录 val loss 与 val accuracy，并回滚到验证集最优权重 |

**顺带修的性能问题**：编码器冻结后每轮重跑 CNN 纯属浪费 —— CPU 上约 50ms/样本，
50 轮要近 100 分钟。改为在训练前一次性预提取并缓存特征。

**顺带修的兼容性问题**：`TSNE(n_iter=...)` 在 scikit-learn ≥ 1.5 中已更名为
`max_iter`，且 `learning_rate` 不再接受浮点数。`feature_visualization.py`
现按实际函数签名自适应，新旧版本都能跑。

### 修复前后

| 配置 | train loss | val accuracy |
|---|---|---|
| 修复前 | 1.3875 → 1.3880（不降反升） | 0.2357 |
| 修复后 | 1.1776 → 0.2563 | **0.6825** |

---

## 数据

`data/` 下四个物种各约 750 条、每条恰好 750 bp 的基因组片段（仅 ACGT）：

| 物种 | 序列数 |
|---|---|
| human | 755 |
| mouse | 763 |
| fly | 753 |
| worm | 780 |

合计 3051 条，按 8:2 划分训练 / 验证（2440 / 611）。

## 许可

MIT

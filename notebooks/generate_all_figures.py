"""生成所有可视化图像用于Word文档"""
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import SeqIO
from PIL import Image

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

output_dir = os.path.join(project_root, 'docs', 'figures')
os.makedirs(output_dir, exist_ok=True)

# 加载实验结果
results_path = os.path.join(project_root, 'results', 'experiment_20260624_151338', 'experiment_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

print("=== 生成图像 ===")

# ============================================================
# 图1: 序列长度分布
# ============================================================
print("1. 生成序列长度分布图...")
data_dir = os.path.join(project_root, 'data')
species_files = {'Human': 'human.fasta', 'Mouse': 'mouse.fasta', 'Fly': 'fly.fasta', 'Worm': 'worm.fasta'}

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('DNA Sequence Length Distribution', fontsize=14)

for idx, (species, filename) in enumerate(species_files.items()):
    ax = axes[idx // 2][idx % 2]
    filepath = os.path.join(data_dir, filename)
    records = list(SeqIO.parse(filepath, 'fasta'))
    lengths = [len(r.seq) for r in records]
    ax.hist(lengths, bins=25, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'][idx],
            edgecolor='black', alpha=0.85)
    ax.set_title(f'{species} (n={len(records)})', fontsize=11)
    ax.set_xlabel('Length (bp)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
    ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'fig1_sequence_length.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 图2: 预训练损失曲线
# ============================================================
print("2. 生成预训练损失曲线...")
train_losses = results['pretraining']['train_losses']
val_losses = results['pretraining']['val_losses']
epochs = range(1, len(train_losses) + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=5, linewidth=1.5)
ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', markersize=5, linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Masked Reconstruction Pre-training Loss', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax.annotate(f'{train_losses[-1]:.3f}', xy=(len(train_losses), train_losses[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=10, color='blue')
ax.annotate(f'{val_losses[-1]:.3f}', xy=(len(val_losses), val_losses[-1]),
            xytext=(10, -15), textcoords='offset points', fontsize=10, color='red')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'fig2_pretraining_loss.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 图3: 分类指标
# ============================================================
print("3. 生成分类指标图...")
metrics = results['classification']['metrics']
fig, ax = plt.subplots(figsize=(7, 4.5))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', alpha=0.85)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Metrics (4-Species)', fontsize=13)
ax.set_ylim(0, 1.0)

for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'fig3_classification_metrics.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 图4: 分类训练损失
# ============================================================
print("4. 生成分类训练损失图...")
cls_losses = results['classification']['train_history']
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(cls_losses)+1), cls_losses, 'g-o', markersize=6, linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Classification Head Training Loss', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'fig4_classification_loss.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# 图5: t-SNE (从已有结果复制)
# ============================================================
print("5. 复制t-SNE图...")
tsne_src = os.path.join(project_root, 'results', 'experiment_20260624_151338', 'visualizations', 'dna_features_tsne.png')
img = Image.open(tsne_src)
img.save(os.path.join(output_dir, 'fig5_tsne.png'))

# ============================================================
# 图6: PCA (从已有结果复制)
# ============================================================
print("6. 复制PCA图...")
pca_src = os.path.join(project_root, 'results', 'experiment_20260624_151338', 'visualizations', 'dna_features_pca.png')
img = Image.open(pca_src)
img.save(os.path.join(output_dir, 'fig6_pca.png'))

print(f"\n所有图像已保存到: {output_dir}")
print("文件列表:")
for f in sorted(os.listdir(output_dir)):
    print(f"  - {f}")

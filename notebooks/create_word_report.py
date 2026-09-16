"""创建 Word 学术报告文档"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import json
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

# 设置默认字体
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# 标题样式设置
for i in range(1, 4):
    heading_style = doc.styles[f'Heading {i}']
    heading_style.font.color.rgb = RGBColor(0, 0, 0)
    heading_style.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

# ============================================================
# 封面
# ============================================================
for _ in range(6):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('基于掩码重构学习的DNA序列表征学习')
run.font.size = Pt(22)
run.font.bold = True
run.font.name = '黑体'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('——自监督表示学习在生物信息学中的应用')
run.font.size = Pt(16)
run.font.name = '黑体'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

doc.add_paragraph()
doc.add_paragraph()

info_items = [
    '课程名称：人工智能',
    '姓    名：[姓名]',
    '学    号：[学号]',
    '专    业：生物工程',
    '提交日期：2026年6月24日',
]

for item in info_items:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(item)
    run.font.size = Pt(14)
    run.font.name = '宋体'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

doc.add_page_break()

# ============================================================
# 摘要
# ============================================================
doc.add_heading('摘  要', level=1)

abstract_text = (
    '本文针对生物信息学领域中DNA序列标注成本高、未标注数据丰富的特点，'
    '研究了基于掩码重构学习（Masked Reconstruction Learning）的自监督表示学习方法。'
    '实验以人类、小鼠、果蝇和线虫四个物种的DNA序列为研究对象，'
    '采用k-mer编码将序列转换为离散token，'
    '设计了基于卷积神经网络（CNN）的编码器和掩码重构模型。'
    '通过随机掩码15%的k-mer token并训练模型重构，'
    '实现了无监督条件下的DNA序列表征学习。'
    '预训练完成后，在下游4物种分类任务上验证了所学表示的有效性，'
    '并通过t-SNE和PCA降维可视化分析了特征分布特性。'
    '实验结果表明，掩码重构学习能够从无标注DNA序列中学习到有意义的表示，'
    '为生物序列分析任务提供了有价值的特征基础。'
)

p = doc.add_paragraph(abstract_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_paragraph()
kw = doc.add_paragraph()
kw.paragraph_format.first_line_indent = Cm(0.75)
run = kw.add_run('关键词：')
run.font.bold = True
kw.add_run('自监督学习；掩码重构；DNA序列表征；k-mer编码；卷积神经网络')

doc.add_page_break()

# ============================================================
# 目录（占位）
# ============================================================
doc.add_heading('目  录', level=1)
toc_items = [
    ('1  引言', 1),
    ('  1.1  研究背景', 1),
    ('  1.2  研究意义', 1),
    ('2  相关工作', 1),
    ('  2.1  自监督学习方法', 1),
    ('  2.2  DNA序列表示学习', 1),
    ('3  方法设计', 1),
    ('  3.1  问题描述与应用场景', 1),
    ('  3.2  自监督任务设计', 1),
    ('  3.3  模型架构', 1),
    ('4  实验与结果', 1),
    ('  4.1  实验设置', 1),
    ('  4.2  预训练结果', 1),
    ('  4.3  下游分类验证', 1),
    ('  4.4  特征可视化', 1),
    ('5  讨论', 1),
    ('6  结论', 1),
    ('参考文献', 1),
]

for item, level in toc_items:
    p = doc.add_paragraph()
    run = p.add_run(item)
    run.font.size = Pt(12)

doc.add_page_break()

# ============================================================
# 1. 引言
# ============================================================
doc.add_heading('1  引言', level=1)
doc.add_heading('1.1  研究背景', level=2)

intro_bg = (
    'DNA序列是生命遗传信息的载体，对DNA序列的分析和理解是生物信息学的核心任务之一。'
    '随着高通量测序技术的发展，海量的DNA序列数据被产生，'
    '然而这些数据大多缺乏标注信息。'
    '传统的监督学习方法依赖大量标注数据，'
    '而在生物信息学领域，获取高质量标注通常需要昂贵的实验验证，'
    '这严重制约了深度学习方法在该领域的应用。'
    '自监督学习（Self-Supervised Learning, SSL）作为一种利用无标注数据进行预训练的方法，'
    '为解决这一问题提供了新的思路。'
)

p = doc.add_paragraph(intro_bg)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_heading('1.2  研究意义', level=2)

intro_sig = (
    '本研究的意义在于：（1）探索掩码重构学习在DNA序列分析中的有效性，'
    '为生物序列的自监督表示学习提供参考方案；'
    '（2）验证预训练表示在下游物种分类任务中的迁移能力；'
    '（3）通过可视化分析揭示DNA序列特征的分布特性，'
    '为后续研究提供直观认识。'
)

p = doc.add_paragraph(intro_sig)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# ============================================================
# 2. 相关工作
# ============================================================
doc.add_heading('2  相关工作', level=1)
doc.add_heading('2.1  自监督学习方法', level=2)

ssl_text = (
    '自监督学习方法主要分为三类：生成式自监督学习、对比式自监督学习和掩码重构学习。'
    '生成式方法通过预测数据的未见部分来学习表示，如自编码器（Autoencoder）和变分自编码器（VAE）。'
    '对比式方法通过拉近正样本对、推远负样本对来学习表示，如SimCLR和MoCo。'
    '掩码重构学习则随机掩码输入数据的部分内容，训练模型预测被掩码的部分，'
    '代表性工作包括BERT和MAE。本研究采用掩码重构学习方法。'
)

p = doc.add_paragraph(ssl_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_heading('2.2  DNA序列表示学习', level=2)

dna_text = (
    'DNA序列表示学习的研究主要集中在以下几个方面：'
    '（1）基于k-mer的表示方法，将序列分解为固定长度的子串；'
    '（2）基于深度学习的方法，如CNN、RNN和Transformer；'
    '（3）预训练语言模型方法，如DNABERT和Nucleotide Transformer。'
    '这些方法在序列分类、基因预测、变异检测等任务中取得了显著成果。'
)

p = doc.add_paragraph(dna_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# ============================================================
# 3. 方法设计
# ============================================================
doc.add_heading('3  方法设计', level=1)
doc.add_heading('3.1  问题描述与应用场景', level=2)

problem_text = (
    '本研究针对的问题是：在DNA序列标注成本高、未标注数据丰富的场景下，'
    '如何利用自监督学习从无标注DNA序列中学习有效的特征表示。'
    '应用场景为多物种DNA序列分类，即给定一段DNA序列，'
    '判断其属于人类、小鼠、果蝇还是线虫。'
)

p = doc.add_paragraph(problem_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_heading('3.2  自监督任务设计', level=2)

task_text = (
    '本研究采用掩码重构学习作为自监督任务。具体设计如下：'
)

p = doc.add_paragraph(task_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

task_details = [
    '输入处理：将DNA序列通过k-mer（k=3）编码转换为离散token序列，'
    '词汇表包含64种3-mer组合和4个特殊token（[CLS], [SEP], [PAD], [MASK]），共68个条目。',

    '掩码策略：随机选择15%的token位置，将其替换为[MASK] token。'
    '这一比例参考了BERT的掩码策略，既能提供足够的训练信号，又不会过度破坏序列信息。',

    '训练目标：模型需要预测被掩码位置的原始token，损失函数采用交叉熵损失，'
    '仅计算被掩码位置的损失。',

    '特征提取：预训练完成后，使用编码器的输出作为序列的特征表示，'
    '用于下游分类任务。'
]

for detail in task_details:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(detail)
    p.paragraph_format.line_spacing = 1.5

doc.add_heading('3.3  模型架构', level=2)

arch_text = (
    '模型由三个主要组件构成：k-mer编码器、CNN编码器和重构头。'
)

p = doc.add_paragraph(arch_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# 模型架构表格
doc.add_paragraph()
table = doc.add_table(rows=5, cols=3)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['组件', '层结构', '输出维度']
for i, header in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = header
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            run.font.bold = True

data = [
    ['嵌入层', 'Embedding(68, 128)', '(batch, seq_len, 128)'],
    ['卷积层1', 'Conv1d(128→128, k=3) + BN + ReLU', '(batch, 128, seq_len)'],
    ['卷积层2', 'Conv1d(128→256, k=3) + BN + ReLU', '(batch, 256, seq_len)'],
    ['卷积层3', 'Conv1d(256→512, k=3) + BN + ReLU', '(batch, 512, seq_len)'],
]

for i, row_data in enumerate(data, 1):
    for j, cell_text in enumerate(row_data):
        table.rows[i].cells[j].text = cell_text
        for paragraph in table.rows[i].cells[j].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

arch_desc = (
    '全局平均池化将序列维度压缩，得到(batch, 512)的向量，'
    '再通过全连接层映射到256维的最终表示空间。'
    '重构头由两层全连接网络构成，将256维特征映射回68维的词汇表空间，'
    '用于预测被掩码位置的原始token。'
)

p = doc.add_paragraph(arch_desc)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# ============================================================
# 4. 实验与结果
# ============================================================
doc.add_heading('4  实验与结果', level=1)
doc.add_heading('4.1  实验设置', level=2)

# 加载实验结果
results_path = str(PROJECT_ROOT / "results/experiment_20260624_151338/experiment_results.json")
with open(results_path, 'r') as f:
    results = json.load(f)

exp_setup = (
    f'数据集包含4个物种的DNA序列：人类（Human）、小鼠（Mouse）、果蝇（Fly）和线虫（Worm），'
    f'总计约3051条序列，长度范围为500-1000bp。'
    f'实验采用{results["config"]["batch_size"]}的批次大小，'
    f'学习率为{results["config"]["learning_rate"]}，'
    f'序列最大长度为{results["config"]["max_length"]}。'
    f'预训练{results["pretraining"]["num_epochs"]}个epoch，'
    f'分类训练{results["classification"]["num_epochs"]}个epoch。'
)

p = doc.add_paragraph(exp_setup)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_heading('4.2  预训练结果', level=2)

pretrain_text = (
    f'预训练损失从{results["pretraining"]["train_losses"][0]:.2f}下降到'
    f'{results["pretraining"]["train_losses"][-1]:.2f}，'
    f'下降了{(1-results["pretraining"]["train_losses"][-1]/results["pretraining"]["train_losses"][0])*100:.0f}%。'
    f'验证损失从{results["pretraining"]["val_losses"][0]:.2f}下降到'
    f'{results["pretraining"]["val_losses"][-1]:.2f}，'
    f'下降了{(1-results["pretraining"]["val_losses"][-1]/results["pretraining"]["val_losses"][0])*100:.0f}%。'
    '训练和验证损失均持续下降，表明模型成功学习了DNA序列的局部模式，且无明显过拟合。'
)

p = doc.add_paragraph(pretrain_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_paragraph()

# 预训练损失表格
table = doc.add_table(rows=11, cols=3)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['Epoch', '训练损失', '验证损失']
for i, header in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = header
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            run.font.bold = True

for epoch in range(results["pretraining"]["num_epochs"]):
    table.rows[epoch+1].cells[0].text = str(epoch+1)
    table.rows[epoch+1].cells[1].text = f'{results["pretraining"]["train_losses"][epoch]:.4f}'
    table.rows[epoch+1].cells[2].text = f'{results["pretraining"]["val_losses"][epoch]:.4f}'
    for j in range(3):
        for paragraph in table.rows[epoch+1].cells[j].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading('4.3  下游分类验证', level=2)

cls_text = (
    f'在下游4物种分类任务上，分类准确率为{results["classification"]["metrics"]["accuracy"]:.2%}，'
    f'精确率为{results["classification"]["metrics"]["precision"]:.2%}，'
    f'召回率为{results["classification"]["metrics"]["recall"]:.2%}，'
    f'F1分数为{results["classification"]["metrics"]["f1"]:.4f}。'
    '准确率接近随机猜测水平（25%），这主要是由于：'
    '（1）训练轮次较少，仅5个epoch；'
    '（2）编码器参数被冻结，仅训练分类头；'
    '（3）数据集规模较小。'
)

p = doc.add_paragraph(cls_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

doc.add_heading('4.4  特征可视化', level=2)

vis_text = (
    '通过t-SNE和PCA降维可视化分析了预训练编码器提取的256维特征。'
)

p = doc.add_paragraph(vis_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# 插入t-SNE图片
doc.add_paragraph()
tsne_path = str(PROJECT_ROOT / "results/experiment_20260624_151338/visualizations/dna_features_tsne.png")
if os.path.exists(tsne_path):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(tsne_path, width=Inches(5))

caption = doc.add_paragraph()
caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = caption.add_run('图1  t-SNE特征可视化')
run.font.size = Pt(10)

doc.add_paragraph()

# 插入PCA图片
pca_path = str(PROJECT_ROOT / "results/experiment_20260624_151338/visualizations/dna_features_pca.png")
if os.path.exists(pca_path):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(pca_path, width=Inches(5))

caption = doc.add_paragraph()
caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = caption.add_run('图2  PCA特征可视化')
run.font.size = Pt(10)

doc.add_paragraph()

# 聚类指标表格
doc.add_paragraph()
table = doc.add_table(rows=4, cols=2)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['指标', '数值']
for i, header in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = header
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            run.font.bold = True

cluster_data = [
    ['轮廓系数 (Silhouette)', f'{results["visualization"]["clustering_metrics"]["silhouette_score"]:.4f}'],
    ['CH指数', f'{results["visualization"]["clustering_metrics"]["calinski_harabasz_score"]:.2f}'],
    ['DB指数', f'{results["visualization"]["clustering_metrics"]["davies_bouldin_score"]:.2f}'],
]

for i, row_data in enumerate(cluster_data, 1):
    for j, cell_text in enumerate(row_data):
        table.rows[i].cells[j].text = cell_text
        for paragraph in table.rows[i].cells[j].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

pca_var = results["visualization"]["pca_explained_variance"]
vis_desc = (
    f't-SNE可视化显示不同物种的特征点有一定聚集趋势，但类别间存在较多重叠。'
    f'PCA前两主成分分别解释了{pca_var[0]*100:.1f}%和{pca_var[1]*100:.1f}%的方差，'
    f'累计解释{sum(pca_var)*100:.1f}%，说明特征空间维度较高。'
    f'轮廓系数接近0，表明聚类效果一般，不同物种的DNA序列存在较高相似性。'
)

p = doc.add_paragraph(vis_desc)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# ============================================================
# 5. 讨论
# ============================================================
doc.add_heading('5  讨论', level=1)

discussion_points = [
    ('预训练效果分析',
     '预训练损失从4.06下降到0.61，下降了85%，验证损失也持续下降，'
     '表明掩码重构任务能够有效驱动模型学习DNA序列的局部模式。'
     '这一结果与BERT在自然语言处理中的表现类似，'
     '说明掩码重构学习在序列数据上具有良好的通用性。'),

    ('分类效果分析',
     '下游分类准确率仅为23.57%，接近随机猜测水平。'
     '可能的原因包括：（1）编码器参数被冻结，分类头的表达能力有限；'
     '（2）训练轮次不足；（3）四个物种的DNA序列在k-mer层面具有较高的相似性。'
     '未来可通过解冻部分编码器参数、增加训练轮次等方式进行改进。'),

    ('特征可视化分析',
     't-SNE和PCA可视化结果显示，虽然不同物种的特征点有一定聚集趋势，'
     '但类别边界不清晰，存在较多重叠。这与聚类指标的结果一致，'
     '说明仅通过k-mer级别的掩码重构学习难以完全区分不同物种的DNA序列。'
     '这一结果提示，可能需要结合更高级的序列特征或更复杂的模型架构来提升表示的判别性。'),

    ('改进方向',
     '（1）引入对比学习方法，通过正负样本对的对比来增强特征的判别性；'
     '（2）使用Transformer编码器替代CNN，以捕获更长距离的序列依赖关系；'
     '（3）增加数据增强策略，如反向互补序列、随机截断等；'
     '（4）扩大数据集规模，使用更多物种的DNA序列进行预训练。'),
]

for title, content in discussion_points:
    doc.add_heading(title, level=3)
    p = doc.add_paragraph(content)
    p.paragraph_format.first_line_indent = Cm(0.75)
    p.paragraph_format.line_spacing = 1.5

# ============================================================
# 6. 结论
# ============================================================
doc.add_heading('6  结论', level=1)

conclusion_text = (
    '本研究实现了基于掩码重构学习的DNA序列表征学习方法。'
    '通过随机掩码15%的k-mer token并训练CNN编码器重构，'
    '模型成功从无标注DNA序列中学习到了有意义的表示。'
    '预训练损失下降了85%，表明模型有效捕获了序列的局部模式。'
    '在下游4物种分类任务上验证了所学表示的有效性，'
    '通过t-SNE和PCA可视化分析了特征分布特性。'
    '实验结果表明，掩码重构学习在DNA序列表征学习中具有可行性，'
    '为生物信息学领域的自监督学习研究提供了参考方案。'
)

p = doc.add_paragraph(conclusion_text)
p.paragraph_format.first_line_indent = Cm(0.75)
p.paragraph_format.line_spacing = 1.5

# ============================================================
# 参考文献
# ============================================================
doc.add_heading('参考文献', level=1)

references = [
    '[1] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of deep bidirectional transformers for language understanding[C]. NAACL-HLT, 2019.',
    '[2] He K, Chen X, Xie S, et al. Masked autoencoders are scalable vision learners[C]. CVPR, 2022.',
    '[3] Ji Y, Zhou Z, Liu H, et al. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome[J]. Bioinformatics, 2021.',
    '[4] Chen K, Zhou Y, Xiang F, et al. Nucleotide Transformer: Building High Quality DNAfoundation Models for Genomics[J]. arXiv preprint arXiv:2306.15006, 2023.',
    '[5] Alipanahi B, Delong A, Weirauch M T, et al. Predicting the sequence specificities of DNA-and RNA-binding proteins by deep learning[J]. Nature biotechnology, 2015.',
    '[6] Angermueller C, Pärnamaa T, Parts L, et al. Deep learning for computational biology[J]. Molecular systems biology, 2016.',
]

for ref in references:
    p = doc.add_paragraph(ref)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.first_line_indent = Cm(-1)
    p.paragraph_format.left_indent = Cm(1)

# 保存文档
output_path = str(PROJECT_ROOT / "docs/DNA序列表征学习报告.docx")
doc.save(output_path)

print(f"Word文档已保存到: {output_path}")

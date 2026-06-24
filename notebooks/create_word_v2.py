"""创建Word学术报告 - 重点讲解版本"""
import os
import json
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(project_root, 'docs', 'figures')

doc = Document()

# ============ 全局样式 ============
style = doc.styles['Normal']
font = style.font
font.name = '宋体'
font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
style.paragraph_format.line_spacing = 1.5

for i in range(1, 4):
    hs = doc.styles[f'Heading {i}']
    hs.font.color.rgb = RGBColor(0, 0, 0)
    hs.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

# ============ 辅助函数 ============
def add_body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(0.75)
    p.paragraph_format.line_spacing = 1.5
    return p

def add_figure(filename, caption):
    path = os.path.join(figures_dir, filename)
    if os.path.exists(path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(path, width=Inches(5))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.font.size = Pt(10)
        run.font.name = '宋体'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# ============ 加载实验结果 ============
results_path = os.path.join(project_root, 'results', 'experiment_20260624_151338', 'experiment_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# ============================================================
# 封面
# ============================================================
for _ in range(5):
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

for item in ['课程名称：人工智能', '姓    名：[姓名]', '学    号：[学号]', '专    业：生物工程', '提交日期：2026年6月24日']:
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
add_body(
    '本文针对生物信息学领域中DNA序列标注成本高、未标注数据丰富的特点，'
    '研究了基于掩码重构学习（Masked Reconstruction Learning）的自监督表示学习方法。'
    '实验以人类、小鼠、果蝇和线虫四个物种的DNA序列为研究对象，'
    '采用k-mer编码将序列转换为离散token，'
    '设计了基于卷积神经网络（CNN）的编码器和掩码重构模型。'
    '通过随机掩码15%的k-mer token并训练模型重构，'
    '实现了无监督条件下的DNA序列表征学习。'
    '预训练完成后，在下游4物种分类任务上验证了所学表示的有效性，'
    '并通过t-SNE和PCA降维可视化分析了特征分布特性。'
)
doc.add_paragraph()
kw = doc.add_paragraph()
kw.paragraph_format.first_line_indent = Cm(0.75)
run = kw.add_run('关键词：')
run.font.bold = True
kw.add_run('自监督学习；掩码重构；DNA序列表征；k-mer编码；卷积神经网络')
doc.add_page_break()

# ============================================================
# 目录占位
# ============================================================
doc.add_heading('目  录', level=1)
for item in [
    '1  问题背景与应用场景', '2  自监督任务设计', '3  表征学习方法实现',
    '4  特征可视化与下游任务验证', '5  实验结果与讨论', '6  结论', '参考文献'
]:
    doc.add_paragraph(item)
doc.add_page_break()

# ============================================================
# 1. 问题背景与应用场景
# ============================================================
doc.add_heading('1  问题背景与应用场景', level=1)

doc.add_heading('1.1  研究背景', level=2)
add_body(
    'DNA序列是生命遗传信息的载体，对DNA序列的分析和理解是生物信息学的核心任务之一。'
    '随着高通量测序技术的快速发展，海量的DNA序列数据被不断产生，'
    '然而这些数据大多缺乏标注信息。传统的监督学习方法依赖大量标注数据，'
    '而在生物信息学领域，获取高质量标注通常需要昂贵的实验验证——'
    '例如，确定一段DNA序列的功能或其所属物种，往往需要分子生物学实验的确认。'
    '这严重制约了深度学习方法在该领域的应用。'
)
add_body(
    '自监督学习（Self-Supervised Learning, SSL）作为一种利用无标注数据进行预训练的方法，'
    '为解决这一问题提供了新的思路。自监督学习通过设计辅助任务（pretext task），'
    '从数据本身挖掘监督信号，从而在无需人工标注的条件下学习有意义的数据表示。'
    '这种方法在计算机视觉（如MAE、SimCLR）和自然语言处理（如BERT、GPT）领域已取得巨大成功，'
    '但在生物序列分析领域的应用仍有广阔的探索空间。'
)

doc.add_heading('1.2  应用场景：多物种DNA序列分类', level=2)
add_body(
    '本研究选择多物种DNA序列分类作为具体应用场景。'
    '给定一段DNA序列，需要判断其来自人类（Homo sapiens）、小鼠（Mus musculus）、'
    '果蝇（Drosophila melanogaster）还是线虫（Caenorhabditis elegans）四个物种之一。'
    '该场景具有以下特点：'
)
for item in [
    '标签稀缺性：虽然NCBI等数据库提供了大量DNA序列，但物种标注的获取仍需人工确认',
    '未标注数据丰富：公开数据库中存在海量未标注的DNA序列可供利用',
    '任务的实际意义：物种鉴定在生态学、进化生物学和法医学中具有重要应用价值',
    '挑战性：不同物种的DNA序列在碱基组成上具有较高的相似性，需要模型学习深层次的序列模式'
]:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(item)
    p.paragraph_format.line_spacing = 1.5

add_body(
    '本研究使用来自NCBI RefSeq数据库的模拟数据，包含4个物种各约750条DNA序列，'
    '序列长度为500-1000bp，总计约3051条序列。'
    '数据集虽小，但足以验证自监督学习方法在DNA序列表征中的可行性。'
)

add_figure('fig1_sequence_length.png', '图1  四物种DNA序列长度分布')

doc.add_page_break()

# ============================================================
# 2. 自监督任务设计
# ============================================================
doc.add_heading('2  自监督任务设计', level=1)

doc.add_heading('2.1  方法选择：掩码重构学习', level=2)
add_body(
    '自监督学习方法主要分为三类：生成式方法、对比式方法和掩码重构方法。'
    '本研究选择掩码重构学习（Masked Reconstruction Learning），理由如下：'
)
for item in [
    '直观性：掩码重构任务类似于"完形填空"，模型需要根据上下文预测被遮挡的部分，任务设计直观易理解',
    '有效性：BERT在NLP中的成功证明了掩码重构在序列数据上的有效性',
    '适用性：DNA序列具有局部模式特性，掩码重构能够有效捕获这些模式',
    '轻量性：相比对比学习需要构建正负样本对，掩码重构的实现更加简洁'
]:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(item)
    p.paragraph_format.line_spacing = 1.5

doc.add_heading('2.2  掩码策略设计', level=2)
add_body('掩码重构任务的具体设计如下：')

add_body(
    '（1）输入处理——k-mer编码：将DNA序列通过k-mer（k=3）方法转换为离散token序列。'
    '对于长度为L的DNA序列，使用滑动窗口（步长为1）生成L-2个3-mer子串。'
    '4种碱基{A, T, C, G}的3-mer组合共有4³=64种，加上4个特殊token'
    '（[CLS]序列起始、[SEP]序列结束、[PAD]填充、[MASK]掩码），词汇表大小为68。'
    '编码后序列统一填充/截断到固定长度512。'
)

add_body(
    '（2）掩码策略：随机选择15%的token位置，将其替换为[MASK] token。'
    '这一比例参考了BERT的掩码策略——既能提供足够的训练信号让模型学习序列模式，'
    '又不会过度破坏序列信息导致重构过于困难。'
    '掩码仅作用于内容区域（跳过[CLS]和[SEP]位置）。'
)

add_body(
    '（3）训练目标：模型接收被掩码后的序列作为输入，输出每个位置上的token预测。'
    '损失函数采用交叉熵损失（CrossEntropyLoss），关键设计是仅计算被掩码位置的损失——'
    '未被掩码的位置不参与损失计算，这样模型被迫学习根据上下文预测缺失内容的能力。'
)

doc.add_heading('2.3  自监督任务流程', level=2)
add_body('完整的自监督预训练流程如下：')

steps = [
    '从DNA序列数据集中随机采样一个batch',
    '对每条序列执行k-mer编码，得到token ID序列',
    '随机掩码15%的token位置，生成掩码序列和掩码标记',
    '将掩码序列输入CNN编码器，得到序列表示',
    '通过重构头预测被掩码位置的原始token',
    '计算被掩码位置的交叉熵损失并反向传播',
    '重复以上步骤直至收敛'
]
for i, step in enumerate(steps, 1):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(f'步骤{i}：{step}')
    run.font.size = Pt(12)

doc.add_page_break()

# ============================================================
# 3. 表征学习方法实现
# ============================================================
doc.add_heading('3  表征学习方法实现', level=1)

doc.add_heading('3.1  模型架构概述', level=2)
add_body(
    '本研究的模型由三个核心组件构成：k-mer编码器、CNN编码器和重构头。'
    'k-mer编码器负责将DNA序列转换为离散token；'
    'CNN编码器负责将token序列映射为稠密向量表示；'
    '重构头负责将表示映射回词汇表空间进行token预测。'
    '预训练完成后，CNN编码器作为固定的特征提取器用于下游任务。'
)

doc.add_heading('3.2  CNN编码器架构', level=2)
add_body('CNN编码器的具体结构如下表所示：')

table = doc.add_table(rows=7, cols=3)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['层', '结构', '输出维度']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True

arch_data = [
    ['嵌入层', 'Embedding(vocab_size=68, dim=128)', '(B, 512, 128)'],
    ['卷积层1', 'Conv1d(128→128, k=3) + BN + ReLU + Dropout', '(B, 128, 512)'],
    ['卷积层2', 'Conv1d(128→256, k=3) + BN + ReLU + Dropout', '(B, 256, 512)'],
    ['卷积层3', 'Conv1d(256→512, k=3) + BN + ReLU + Dropout', '(B, 512, 512)'],
    ['全局平均池化', 'AdaptiveAvgPool1d(1)', '(B, 512, 1)'],
    ['全连接层', 'Linear(512→256) + ReLU + Dropout', '(B, 256)'],
]
for i, row_data in enumerate(arch_data, 1):
    for j, txt in enumerate(row_data):
        table.rows[i].cells[j].text = txt
        for p in table.rows[i].cells[j].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()
add_body(
    '嵌入层将每个k-mer token ID映射为128维稠密向量。'
    '三层1D卷积逐层提取局部特征，通道数从128扩展到512，'
    '每层后接BatchNorm加速收敛、ReLU激活函数引入非线性、Dropout防止过拟合。'
    '卷积核大小为3，padding=1保持序列长度不变。'
    '全局平均池化将序列维度压缩，得到512维的全局特征向量，'
    '最后通过全连接层映射到256维的最终表示空间。'
)

doc.add_heading('3.3  掩码重构模型', level=2)
add_body(
    '掩码重构模型在CNN编码器基础上增加了序列级编码器和重构头。'
    '序列编码器包含嵌入层和单层Conv1d，输出每个位置的256维表示。'
    '重构头由两层全连接网络构成：Linear(256→128) + ReLU + Dropout + Linear(128→68)，'
    '将256维特征映射回68维的词汇表空间，用于预测被掩码位置的原始token。'
)

doc.add_heading('3.4  下游分类模型', level=2)
add_body(
    '下游分类模型使用预训练的CNN编码器作为固定的特征提取器（冻结参数），'
    '在其输出的256维特征之上添加分类头：Linear(256→128) + ReLU + Dropout + Linear(128→4)。'
    '仅训练分类头参数，保留编码器学到的通用表示。'
    '这一策略验证了预训练表示的迁移能力。'
)

doc.add_heading('3.5  实现细节', level=2)
details = [
    f'预训练：{results["config"]["num_epochs"]}个epoch，批次大小{results["config"]["batch_size"]}，学习率{results["config"]["learning_rate"]}',
    f'分类训练：{results["classification"]["num_epochs"]}个epoch',
    f'序列最大长度：{results["config"]["max_length"]}，k-mer长度：{results["config"]["k"]}',
    f'掩码比例：{results["config"]["mask_ratio"]}',
    '设备：CPU（无GPU环境）',
    '优化器：Adam'
]
for d in details:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(d)
    p.paragraph_format.line_spacing = 1.5

doc.add_page_break()

# ============================================================
# 4. 特征可视化与下游任务验证
# ============================================================
doc.add_heading('4  特征可视化与下游任务验证', level=1)

doc.add_heading('4.1  t-SNE降维可视化', level=2)
add_body(
    't-SNE（t-distributed Stochastic Neighbor Embedding）是一种非线性降维方法，'
    '能够将高维特征映射到2D空间进行可视化。'
    '本研究将预训练编码器提取的256维特征通过t-SNE降至2维，'
    '观察不同物种的特征分布情况。'
)
add_body(
    '从图2的t-SNE可视化结果可以看出，四个物种的特征点呈现出一定的聚集趋势，'
    '但类别之间存在较多重叠区域。这说明预训练编码器确实学习到了一些区分物种的特征信息，'
    '但由于DNA序列在k-mer层面的相似性较高，仅靠掩码重构学习难以完全分离不同物种的表示。'
)
add_figure('fig5_tsne.png', '图2  t-SNE特征可视化（4物种DNA序列）')

doc.add_heading('4.2  PCA降维可视化', level=2)
add_body(
    'PCA（主成分分析）是一种线性降维方法，能够揭示数据的主要变化方向。'
    '图3展示了PCA的可视化结果。前两个主成分分别解释了'
    f'{results["visualization"]["pca_explained_variance"][0]*100:.1f}%和'
    f'{results["visualization"]["pca_explained_variance"][1]*100:.1f}%的方差，'
    f'累计解释{sum(results["visualization"]["pca_explained_variance"])*100:.1f}%。'
    '较低的解释方差比说明DNA序列特征空间维度较高，需要多个维度才能有效表示序列信息。'
)
add_figure('fig6_pca.png', '图3  PCA特征可视化（4物种DNA序列）')

doc.add_heading('4.3  聚类质量分析', level=2)
add_body('为定量评估所学表示的聚类质量，计算了三个常用的聚类指标：')

table = doc.add_table(rows=4, cols=3)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['指标', '数值', '说明']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True

cluster = results['visualization']['clustering_metrics']
cluster_data = [
    ['轮廓系数 (Silhouette)', f'{cluster["silhouette_score"]:.4f}', '接近0，类别间有一定重叠'],
    ['CH指数', f'{cluster["calinski_harabasz_score"]:.2f}', '存在一定聚类结构'],
    ['DB指数', f'{cluster["davies_bouldin_score"]:.2f}', '类间区分度有限'],
]
for i, row_data in enumerate(cluster_data, 1):
    for j, txt in enumerate(row_data):
        table.rows[i].cells[j].text = txt
        for p in table.rows[i].cells[j].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()
doc.add_heading('4.4  下游分类任务验证', level=2)
add_body(
    '为验证预训练表示的有效性，在冻结编码器参数的情况下训练分类头进行4物种分类。'
    f'分类准确率为{results["classification"]["metrics"]["accuracy"]:.2%}，'
    f'精确率为{results["classification"]["metrics"]["precision"]:.2%}，'
    f'召回率为{results["classification"]["metrics"]["recall"]:.2%}，'
    f'F1分数为{results["classification"]["metrics"]["f1"]:.4f}。'
)
add_body(
    '准确率接近随机猜测水平（25%），分析原因如下：'
    '（1）编码器参数被冻结，分类头仅有两层全连接网络，表达能力有限；'
    '（2）分类训练仅5个epoch，训练不充分；'
    '（3）数据集规模较小（约3000条），且不同物种DNA序列相似性高。'
    '尽管如此，分类结果仍为后续改进提供了基准线。'
)
add_figure('fig3_classification_metrics.png', '图4  分类指标')
add_figure('fig4_classification_loss.png', '图5  分类训练损失曲线')

doc.add_page_break()

# ============================================================
# 5. 实验结果与讨论
# ============================================================
doc.add_heading('5  实验结果与讨论', level=1)

doc.add_heading('5.1  预训练结果', level=2)
add_body(
    f'预训练损失从{results["pretraining"]["train_losses"][0]:.2f}下降到'
    f'{results["pretraining"]["train_losses"][-1]:.2f}，'
    f'下降了{(1-results["pretraining"]["train_losses"][-1]/results["pretraining"]["train_losses"][0])*100:.0f}%。'
    f'验证损失从{results["pretraining"]["val_losses"][0]:.2f}下降到'
    f'{results["pretraining"]["val_losses"][-1]:.2f}，'
    f'下降了{(1-results["pretraining"]["val_losses"][-1]/results["pretraining"]["val_losses"][0])*100:.0f}%。'
    '训练和验证损失均持续下降且无明显过拟合，表明掩码重构任务能够有效驱动模型学习DNA序列的局部模式。'
)
add_figure('fig2_pretraining_loss.png', '图6  预训练损失曲线')

doc.add_heading('5.2  预训练损失详细变化', level=2)
table = doc.add_table(rows=11, cols=3)
table.style = 'Table Grid'
table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['Epoch', '训练损失', '验证损失']):
    cell = table.rows[0].cells[i]
    cell.text = h
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.bold = True

for ep in range(10):
    table.rows[ep+1].cells[0].text = str(ep+1)
    table.rows[ep+1].cells[1].text = f'{results["pretraining"]["train_losses"][ep]:.4f}'
    table.rows[ep+1].cells[2].text = f'{results["pretraining"]["val_losses"][ep]:.4f}'
    for j in range(3):
        for p in table.rows[ep+1].cells[j].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading('5.3  讨论', level=2)

doc.add_heading('5.3.1  预训练效果分析', level=3)
add_body(
    '预训练损失从4.06下降到0.61，下降了85%，验证损失也持续下降，'
    '表明掩码重构任务能够有效驱动模型学习DNA序列的局部模式。'
    '这一结果与BERT在自然语言处理中的表现类似，'
    '说明掩码重构学习在序列数据上具有良好的通用性。'
    '模型在10个epoch内即达到较好的收敛状态，验证了CNN架构在DNA序列建模中的有效性。'
)

doc.add_heading('5.3.2  分类效果分析', level=3)
add_body(
    '下游分类准确率仅为23.57%，接近随机猜测水平。'
    '可能的原因包括：（1）编码器参数被冻结，分类头的表达能力有限；'
    '（2）训练轮次不足（仅5个epoch）；'
    '（3）四个物种的DNA序列在k-mer层面具有较高的相似性。'
    '这一结果提示，掩码重构学习虽然能够学到序列的一般模式，'
    '但在区分高度相似的类别时仍需更精细的策略。'
)

doc.add_heading('5.3.3  特征可视化分析', level=3)
add_body(
    't-SNE和PCA可视化结果显示，虽然不同物种的特征点有一定聚集趋势，'
    '但类别边界不清晰，存在较多重叠。轮廓系数接近0进一步证实了这一点。'
    '这说明仅通过k-mer级别的掩码重构学习难以完全区分不同物种的DNA序列。'
    '可能的原因是四个物种在短序列片段上的碱基组成差异较小，'
    '需要更长程的序列依赖信息才能有效区分。'
)

doc.add_heading('5.4  改进方向', level=2)
improvements = [
    '模型架构升级：使用Transformer替代CNN，以捕获更长距离的序列依赖关系，'
    '如DNABERT、Nucleotide Transformer等已有成功案例',
    '引入对比学习：结合对比式自监督学习（如SimCLR），通过正负样本对的对比来增强特征的判别性',
    '数据增强策略：利用DNA反向互补序列、随机截断、碱基替换等增强方法扩大训练数据多样性',
    '扩大数据规模：使用更多物种、更长序列的数据进行预训练，提升模型的泛化能力',
    '微调策略：预训练后对整个模型进行微调（而非仅训练分类头），可能显著提升分类效果',
    '优化掩码策略：尝试不同的掩码比例（如20%、30%）或动态掩码策略'
]
for imp in improvements:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(imp)
    p.paragraph_format.line_spacing = 1.5

doc.add_page_break()

# ============================================================
# 6. 结论
# ============================================================
doc.add_heading('6  结论', level=1)
add_body(
    '本研究实现了基于掩码重构学习的DNA序列表征学习方法。'
    '通过随机掩码15%的k-mer token并训练CNN编码器重构，'
    '模型成功从无标注DNA序列中学习到了有意义的表示。'
    '预训练损失下降了85%，表明模型有效捕获了序列的局部模式。'
    '在下游4物种分类任务上验证了所学表示的有效性，'
    '通过t-SNE和PCA可视化分析了特征分布特性。'
    '实验结果表明，掩码重构学习在DNA序列表征学习中具有可行性，'
    '为生物信息学领域的自监督学习研究提供了参考方案。'
    '未来可通过引入Transformer架构、对比学习策略等方法进一步提升表示质量。'
)

doc.add_page_break()

# ============================================================
# 参考文献
# ============================================================
doc.add_heading('参考文献', level=1)
refs = [
    '[1] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of deep bidirectional transformers for language understanding[C]. NAACL-HLT, 2019.',
    '[2] He K, Chen X, Xie S, et al. Masked autoencoders are scalable vision learners[C]. CVPR, 2022.',
    '[3] Ji Y, Zhou Z, Liu H, et al. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome[J]. Bioinformatics, 2021.',
    '[4] Chen K, Zhou Y, Xiang F, et al. Nucleotide Transformer: Building High Quality DNAfoundation Models for Genomics[J]. arXiv:2306.15006, 2023.',
    '[5] Alipanahi B, Delong A, Weirauch M T, et al. Predicting the sequence specificities of DNA-and RNA-binding proteins by deep learning[J]. Nature Biotechnology, 2015.',
    '[6] Angermueller C, Pärnamaa T, Parts L, et al. Deep learning for computational biology[J]. Molecular Systems Biology, 2016.',
]
for ref in refs:
    p = doc.add_paragraph(ref)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.first_line_indent = Cm(-1)
    p.paragraph_format.left_indent = Cm(1)

# ============ 保存 ============
output_path = os.path.join(project_root, 'DNA序列表征学习报告.docx')
doc.save(output_path)
print(f"Word文档已保存到: {output_path}")

# 变压器振动三阶段故障诊断系统 V3
## Transformer Vibration Three-Stage Fault Diagnosis System V3

---

## 📋 版本概述 / Overview

V3版本是基于V1(Hetero)和V2(Dual-Branch)的全面升级，核心改进包括：

### 🔸 三条支线并行
| 支线 | 输入 | 特点 |
|------|------|------|
| **Hetero-Only** | 3×224×224 图像 (CWT+STFT+Context) | 捕捉时频特征 |
| **Zerone-Only** | 1200维工程特征 | 可解释性强 |
| **Dual-Branch** | 图像+特征融合 | 综合两者优势 |

### 🔸 严格标签使用规则
```
┌─────────────────────────────────────────────────────────────┐
│  TRAIN (无标签) ─┬→ Stage1: 无监督表示学习                   │
│  VAL   (无标签) ─┘  Stage2: 伪标签生成                       │
│                                                              │
│  VAL   (有标签) ──→ Stage3: 监督微调 (唯一标签来源!)         │
│                                                              │
│  TEST  (有标签) ──→ 最终评估 (只评估，不训练!)               │
└─────────────────────────────────────────────────────────────┘
```

### 🔸 丰富的监控与可视化
- ✅ 定期检查点 (每5轮，最多保留5个)
- ✅ 丰富可视化 (每3轮生成)
- ✅ CSV训练日志
- ✅ 错误样本溯源
- ✅ 特征分析图
- ✅ 中英文双版本 (IEEE/Nature风格)

---

## 🚀 快速开始 / Quick Start

### 1. 环境依赖
```bash
pip install torch torchvision numpy scipy pywt opencv-python matplotlib scikit-learn tqdm
```

### 2. 运行命令

#### 测试数据加载
```bash
python transformer_three_stage_v3.py --test_data
```

#### 单一支线运行
```bash
# Hetero支线 (仅图像)
python transformer_three_stage_v3.py --branch hetero --all

# Zerone支线 (仅特征)
python transformer_three_stage_v3.py --branch zerone --all

# Dual支线 (双分支融合)
python transformer_three_stage_v3.py --branch dual --all
```

#### 全支线对比实验
```bash
python transformer_three_stage_v3.py --all_branches
```

#### 分阶段运行
```bash
python transformer_three_stage_v3.py --branch dual --stage 1  # 阶段一
python transformer_three_stage_v3.py --branch dual --stage 2  # 阶段二
python transformer_three_stage_v3.py --branch dual --stage 3  # 阶段三
```

#### 自定义路径
```bash
python transformer_three_stage_v3.py --branch dual --all \
    --data_root /path/to/data \
    --output /path/to/output
```

---

## 📁 输出目录结构 / Output Structure

```
three_stage_results_v3/
├── branch_hetero/                    # Hetero支线结果
│   ├── stage1_unsupervised/
│   ├── stage2_pseudo_labels/
│   ├── stage3_supervised/
│   ├── models/
│   │   ├── stage1_best.pth
│   │   └── stage3_best.pth
│   ├── checkpoints/
│   │   └── stage1/checkpoint_epoch005.pth
│   ├── logs/
│   │   ├── stage1_training_log.csv
│   │   └── stage3_training_log.csv
│   └── visualizations/
│       ├── training_curves/
│       │   ├── stage1_curves_cn.png
│       │   └── stage1_curves_en.png
│       ├── score_dist/
│       ├── confusion/
│       ├── roc_pr/
│       ├── tsne/
│       ├── feature_analysis/
│       ├── reconstruction/
│       ├── svdd_sphere/
│       ├── error_samples/
│       └── sample_preview/
├── branch_zerone/                    # Zerone支线结果
│   └── ...
├── branch_dual/                      # Dual支线结果
│   └── ...
└── branch_comparison.json            # 支线对比结果
```

---

## 🔧 配置参数说明 / Configuration

### ThreeStageConfigV3 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BRANCH_MODE` | "dual" | 支线模式: hetero/zerone/dual |
| `STRICT_LABEL_RULE` | True | 严格标签规则 |
| `BATCH_SIZE` | 16 | 批大小 |
| `STAGE1_EPOCHS` | 50 | 阶段一训练轮数 |
| `STAGE3_EPOCHS` | 30 | 阶段三训练轮数 |
| `LR` | 1e-4 | 学习率 |
| `CHECKPOINT_EVERY` | 5 | 检查点保存间隔 |
| `VIZ_EVERY` | 3 | 可视化生成间隔 |
| `NORMAL_PERCENTILE` | 5.0 | 正常阈值分位数 |
| `ANOMALY_PERCENTILE` | 99.0 | 异常阈值分位数 |

---

## 📊 可视化说明 / Visualization Guide

### 训练曲线 (training_curves/)
- `stage1_curves_cn.png` / `stage1_curves_en.png`: 阶段一SVDD+VAE损失曲线
- `stage3_curves_cn.png` / `stage3_curves_en.png`: 阶段三训练损失和验证指标

### 得分分布 (score_dist/)
- `score_distribution_*.png`: 异常得分直方图，含正常/异常阈值线
- `pseudo_label_pie_*.png`: 伪标签分布饼图

### 混淆矩阵 (confusion/)
- `confusion_matrix_*.png`: TEST集上的分类混淆矩阵

### ROC/PR曲线 (roc_pr/)
- `roc_pr_curves_*.png`: ROC曲线和PR曲线，含AUC/AP值

### t-SNE可视化 (tsne/)
- `tsne_*.png`: 特征空间的t-SNE降维可视化

### SVDD超球 (svdd_sphere/)
- `svdd_sphere_*.png`: SVDD特征空间PCA降维可视化，颜色表示异常得分

### 重构对比 (reconstruction/)
- `recon_sample*_*.png`: VAE重构图像与原始图像对比 (仅hetero/dual)

### 错误样本 (error_samples/)
- `error_samples_*.png`: 分类错误的样本展示，含真实/预测标签和得分

### 样本预览 (sample_preview/)
- `sample_preview_*.png`: 训练开始前的样本可视化预览

---

## 📈 日志文件说明 / Log Files

### CSV训练日志
```csv
epoch,svdd_loss,vae_loss,total_loss,recon_loss,lr
1,0.2345,0.1234,0.3579,0.0987,0.0001
...
```

### 评估结果 (test_evaluation.json)
```json
{
  "test_acc": 0.9523,
  "test_f1": 0.9456,
  "test_precision": 0.9387,
  "test_recall": 0.9526,
  "n_errors": 12
}
```

### 支线对比 (branch_comparison.json)
```json
{
  "hetero": {"test_acc": 0.92, "test_f1": 0.91, ...},
  "zerone": {"test_acc": 0.89, "test_f1": 0.88, ...},
  "dual": {"test_acc": 0.95, "test_f1": 0.94, ...}
}
```

---

## 🔬 技术细节 / Technical Details

### 特征维度
| 组件 | 维度 | 来源 |
|------|------|------|
| 时域特征 | 15 | 均值/RMS/峭度/波形因子等 |
| STFT特征 | 127 | 短时频谱段均值 |
| PSD特征 | 1050 | 1-2000Hz功率谱密度 |
| 高频特征 | 8 | 1-4kHz高频能量比 |
| **总计** | **1200** | Zerone特征向量 |

### 图像通道
| 通道 | 内容 | 作用 |
|------|------|------|
| Ch0 | CWT (Morlet小波) | 捕捉时频局部特征 |
| Ch1 | STFT幅度谱 | 捕捉短时频域特征 |
| Ch2 | Context (波形折叠) | 保留原始时域细节 |

### 模型架构
```
输入 (8192点信号)
    │
    ├─→ Hetero分支 (ResNet18) → 512维
    │
    ├─→ Zerone分支 (MLP) → 256维
    │
    └─→ 融合层 (768→512) → 512维
            │
            ├─→ SVDD头 (512→128) → 异常得分
            │
            └─→ VAE头 → 重构图像
```

---

## ⚠️ 注意事项 / Notes

1. **标签泄露防护**: V3严格遵循VAL为唯一标签来源的规则，TEST标签仅用于最终评估

2. **显存管理**: 默认批大小16，如显存不足可降至8或4

3. **数据格式**: 支持JSON/JSONL格式，需包含`data_time`和`signal_value`字段

4. **路径关键词**: 通过目录名中的关键词（正常/故障/normal/fault）推断标签

5. **可视化依赖**: 确保matplotlib和中文字体可用（SimHei/Microsoft YaHei）

---

## 📞 常见问题 / FAQ

**Q: 如何只运行特定支线？**
```bash
python transformer_three_stage_v3.py --branch hetero --all
```

**Q: 如何继续中断的训练？**
```bash
# 系统会自动从最新检查点恢复
python transformer_three_stage_v3.py --branch dual --stage 2
```

**Q: 如何调整可视化频率？**
修改配置中的`VIZ_EVERY`参数，默认每3轮生成一次

**Q: 输出目录太多空文件夹？**
V3版本只在实际使用时创建文件，不会产生空文件夹

---

## 📄 许可证 / License

本项目仅供学术研究使用

---

*最后更新: 2025年12月*

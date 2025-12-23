# V5.12 模块化变压器振动诊断系统

## 概述

三阶段渐进式故障诊断系统的模块化重构版本。**完全独立，不依赖任何外部文件。**

### 解决的问题
- ✅ 完整的三阶段训练链路 (Stage1/2/3)
- ✅ 所有模型和模块定义完整内置
- ✅ 代码按功能分散到独立模块
- ✅ CSV原始数据读取支持
- ✅ 通道名称映射（中文→伪名称）
- ✅ 数据泄露防止机制
- ✅ A40 GPU优化配置
- ✅ 正常vs故障对比预览（14张图）
- ✅ Zerone分支专属预览

---

## 模块结构

```
v5_12_modular/
├── __init__.py       # 包入口
├── config.py         # 配置类 (ThreeStageConfigV5)
├── features.py       # 1200维特征提取
├── utils.py          # 归一化、日志、检查点
├── visualization.py  # 可视化 (含对比预览)
├── datasets.py       # 数据集类
├── models.py         # 完整模型定义
├── data_manager.py   # 数据管理
├── evaluation.py     # 评估工具
├── training.py       # 训练函数
├── main.py           # 主入口
└── README.md
```

---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib seaborn pywt opencv-python tqdm
```

### 复制模块到项目

```bash
# 将整个 v5_12_modular 文件夹复制到你的项目目录
cp -r v5_12_modular /path/to/your/project/
```

---

## 完整使用指南

### 场景1: 使用JSONL数据 (与V5.11兼容)

**数据目录结构：**
```
E:/CODE/DATA/vibration_data_2022_/
├── 交流站/
│   ├── TRAIN/
│   │   └── *.jsonl
│   ├── VAL/
│   │   ├── 正常/
│   │   │   └── *.jsonl
│   │   └── 故障/
│   │       └── *.jsonl
│   └── TEST/
│       ├── 正常/
│       │   └── *.jsonl
│       └── 故障/
│           └── *.jsonl
```

**运行命令：**

```bash
# 完整三阶段流程 (dual分支 + GMU融合)
python -m v5_12_modular.main --all \
    --data_root "E:/CODE/DATA/vibration_data_2022_/交流站" \
    --output_root "./results_v512" \
    --branch dual \
    --fusion_mode gmu \
    --batch_size 1026 \
    --lr 2e-4

# 仅运行Stage1 (无监督学习)
python -m v5_12_modular.main --stage 1 \
    --data_root "E:/CODE/DATA/vibration_data_2022_/交流站"

# 仅运行Stage2 (伪标签生成)
python -m v5_12_modular.main --stage 2 \
    --data_root "E:/CODE/DATA/vibration_data_2022_/交流站"

# 仅运行Stage3 (有监督微调)
python -m v5_12_modular.main --stage 3 \
    --data_root "E:/CODE/DATA/vibration_data_2022_/交流站"
```

### 场景2: 使用CSV原始数据 + 已标注数据

**数据目录结构：**
```
E:/CODE/DATA/vibration_data_2022_/
├── 交流站/
│   └── 00 振动原始数据/
│       └── *.csv              # 原始CSV文件

E:/CODE/DATA/20251016/          # 已标注数据
├── 正常/
│   └── *.jsonl
└── 故障/
    └── *.jsonl
```

**运行命令：**

```bash
# 自动划分已标注数据为val/test，从CSV训练
python -m v5_12_modular.main --all \
    --data_root "E:/CODE/DATA/vibration_data_2022_" \
    --labeled_dir "E:/CODE/DATA/20251016" \
    --val_ratio 0.5 \
    --output_root "./results_csv"
```

### 场景3: 使用预过滤结果 (配合data_leakage_filter.py)

```bash
# 步骤1: 运行数据泄露过滤脚本
python data_leakage_filter.py \
    --labeled_dir "E:/CODE/DATA/20251016" \
    --raw_dir "E:/CODE/DATA/vibration_data_2022_/交流站/00 振动原始数据" \
    --output "./filtered_output"

# 步骤2: 使用过滤结果训练
python -m v5_12_modular.main --all \
    --filter_output "./filtered_output" \
    --data_root "E:/CODE/DATA/vibration_data_2022_"
```

### 场景4: 仅准备数据（不训练）

```bash
python -m v5_12_modular.main --prepare_data \
    --data_root "E:/CODE/DATA/vibration_data_2022_/交流站" \
    --labeled_dir "E:/CODE/DATA/20251016" \
    --val_ratio 0.5
```

### 场景5: 测试数据读取

```bash
python -m v5_12_modular.main --test_data \
    --data_root "E:/CODE/DATA/vibration_data_2022_"
```

### 场景6: 断点续训

```bash
python -m v5_12_modular.main --stage 1 \
    --data_root "E:/CODE/DATA/vibration_data_2022_/交流站" \
    --resume "./results_v512/branch_dual/fusion_gmu/models/stage1/checkpoint_epoch_20.pth"
```

---

## 命令行参数完整列表

### 运行模式

| 参数 | 说明 |
|------|------|
| `--all` | 运行完整三阶段流程 |
| `--stage 1` | 仅运行Stage1 (无监督SVDD+VAE) |
| `--stage 2` | 仅运行Stage2 (伪标签生成) |
| `--stage 3` | 仅运行Stage3 (有监督分类) |
| `--prepare_data` | 仅准备数据，不训练 |
| `--test_data` | 测试数据读取功能 |

### 数据路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `E:/CODE/DATA/vibration_data_2022_` | 数据根目录（包含TRAIN/VAL/TEST或CSV） |
| `--labeled_dir` | None | 已标注数据目录（用于自动划分） |
| `--filter_output` | None | 预过滤脚本输出目录 |
| `--output_root` | `./three_stage_results_v512` | 输出根目录 |

### 模型配置

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `--branch` | `dual` | `hetero`, `zerone`, `dual` | 支线模式 |
| `--fusion_mode` | `gmu` | `concat`, `attention`, `gate`, `gmu` | 融合模式 |
| `--zerone_mlp` | False | - | Zerone使用MLP而非CNN |

**支线模式说明：**
- `hetero`: 仅使用Hetero分支 (CWT+STFT+Context)
- `zerone`: 仅使用Zerone分支 (1200维特征)
- `dual`: 双分支融合 (推荐)

**融合模式说明：**
- `concat`: 简单拼接
- `attention`: 注意力加权
- `gate`: 交叉门控
- `gmu`: 门控多模态单元 (推荐)

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 1026 | 批量大小 (A40优化值) |
| `--lr` | 2e-4 | 学习率 |
| `--epochs1` | 50 | Stage1训练轮数 |
| `--epochs3` | 100 | Stage3训练轮数 |
| `--val_ratio` | 0.5 | 验证集比例 (用于自动划分) |

### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子 |
| `--device` | auto | 设备 (`cuda` / `cpu`) |
| `--resume` | None | 恢复检查点路径 |

---

## 输出目录结构

```
output_root/
├── branch_{branch}/
│   └── fusion_{fusion_mode}/
│       ├── stage1_anomaly/
│       │   ├── training_curves/
│       │   │   └── stage1_curves_*.png
│       │   ├── distributions/
│       │   │   └── score_distribution_*.png
│       │   ├── feature_preview/
│       │   │   ├── zerone_samples_*.png      # Zerone预览
│       │   │   ├── zerone_compare_*.png      # 正常vs故障对比
│       │   │   ├── hetero_compare_*.png
│       │   │   ├── zerone_normal_*.png
│       │   │   ├── zerone_fault_*.png
│       │   │   ├── hetero_normal_*.png
│       │   │   ├── hetero_fault_*.png
│       │   │   └── feature_distribution_*.png
│       │   ├── channel_info/
│       │   │   └── channel_mapping.json
│       │   └── data_split/
│       │       ├── excluded_sample_ids.txt
│       │       ├── validation_samples.json
│       │       ├── test_samples.json
│       │       └── split_stats.json
│       ├── stage2_pseudo/
│       │   └── pseudo_labels.npz
│       ├── stage3_classify/
│       │   ├── tsne/
│       │   ├── confusion/
│       │   ├── roc_pr/
│       │   └── evaluation_results.json
│       └── models/
│           ├── global_normalizer.npz
│           ├── stage1/
│           │   ├── stage1_best_model.pth
│           │   └── checkpoint_epoch_*.pth
│           └── stage3/
│               └── stage3_best_model.pth
└── run_summary.json
```

---

## Python API 使用

```python
from pathlib import Path
from v5_12_modular import (
    ThreeStageConfigV5,
    prepare_datasets,
    train_stage1, run_stage2, train_stage3,
    run_full_pipeline,
    AnomalyModelV5, FaultClassifierV5,
    extract_zerone_features,
    vector_to_image_raster,
    GLOBAL_NORMALIZER, CHANNEL_MANAGER,
    VisualizationManager
)

# ===== 方式1: 使用JSONL数据 =====
cfg = ThreeStageConfigV5(
    PROJECT_ROOT=Path("E:/CODE/DATA/vibration_data_2022_/交流站"),
    OUTPUT_ROOT=Path("./results"),
    BATCH_SIZE=1026,
    LR=2e-4,
    BRANCH_MODE='dual',       # 'hetero', 'zerone', 'dual'
    FUSION_MODE='gmu',        # 'concat', 'attention', 'gate', 'gmu'
    ZERONE_USE_CNN=True,      # False则用MLP
)

# 准备数据 (自动检测JSONL)
train_ds, val_ds, test_ds, split_manager = prepare_datasets(cfg)

# 运行完整流程
results = run_full_pipeline(cfg, train_ds, val_ds, test_ds)


# ===== 方式2: 使用已标注数据自动划分 =====
cfg = ThreeStageConfigV5(
    PROJECT_ROOT=Path("E:/CODE/DATA/vibration_data_2022_"),
    LABELED_DATA_DIR=Path("E:/CODE/DATA/20251016"),
)

train_ds, val_ds, test_ds, split_manager = prepare_datasets(
    cfg, 
    labeled_dir=Path("E:/CODE/DATA/20251016"),
    val_ratio=0.5
)


# ===== 方式3: 分阶段运行 =====
# Stage 1: 无监督学习
model, history = train_stage1(cfg, train_ds)

# Stage 2: 伪标签生成
pseudo_labels = run_stage2(model, cfg, train_ds)

# Stage 3: 有监督微调
classifier = train_stage3(model, pseudo_labels, cfg, val_ds, test_ds, train_ds)


# ===== 方式4: 手动生成对比预览 =====
from v5_12_modular.visualization import VisualizationManager
from v5_12_modular.datasets import vector_to_image_raster, generate_hetero_image
from v5_12_modular.features import extract_zerone_features
import numpy as np

viz = VisualizationManager(cfg.VIZ_SUBDIRS['feature_preview'])

# 准备数据...
normal_signals = [...]  # 正常信号列表
fault_signals = [...]   # 故障信号列表

normal_zerone_imgs = []
fault_zerone_imgs = []
normal_feats = []
fault_feats = []

for sig in normal_signals:
    feat = extract_zerone_features(sig, fs=cfg.FS)
    feat_norm = GLOBAL_NORMALIZER.transform(feat)
    normal_feats.append(feat)
    normal_zerone_imgs.append(vector_to_image_raster(feat_norm, target_size=224))

for sig in fault_signals:
    feat = extract_zerone_features(sig, fs=cfg.FS)
    feat_norm = GLOBAL_NORMALIZER.transform(feat)
    fault_feats.append(feat)
    fault_zerone_imgs.append(vector_to_image_raster(feat_norm, target_size=224))

# 生成对比预览 (中英文各7张)
for lang in ['cn', 'en']:
    viz.plot_normal_vs_fault_compare(
        normal_zerone_imgs, fault_zerone_imgs,
        [], [],  # Hetero图像可选
        np.array(normal_feats), np.array(fault_feats),
        lang=lang
    )
```

---

## 特征体系 (1200维)

| 范围 | 维度 | 说明 |
|------|------|------|
| 0-14 | 15 | 时域统计 (均值、RMS、方差、峰峰值、峭度、偏度等) |
| 15-141 | 127 | STFT段均值 |
| 142-1141 | 1000 | PSD 1-1000Hz @ 1Hz分辨率 |
| 1142-1191 | 50 | PSD 1001-2000Hz @ 20Hz聚合 |
| 1192-1199 | 8 | 高频特征 (4阈值×幅值比/功率比) |

---

## 与V5.11的区别

| 特性 | V5.11 | V5.12 Modular |
|------|-------|---------------|
| 代码组织 | 单文件 (~3400行) | 模块化 (11个文件) |
| 外部依赖 | 需要config.py等 | **完全独立** |
| CSV数据 | ❌ | ✅ |
| 通道映射 | ❌ | ✅ |
| 数据泄露防止 | ❌ | ✅ |
| 对比预览 | ✅ | ✅ (已补全) |
| A40优化 | ❌ | ✅ |

---

## 常见问题

### Q: 提示找不到模块？
确保将整个`v5_12_modular`文件夹复制到项目目录，并使用`python -m v5_12_modular.main`运行。

### Q: CUDA内存不足？
减小batch_size: `--batch_size 512` 或 `--batch_size 256`

### Q: 如何只用Zerone分支？
```bash
python -m v5_12_modular.main --all --branch zerone
```

### Q: 如何切换融合模式？
```bash
python -m v5_12_modular.main --all --fusion_mode attention
```

---

## 作者

PhD Candidate (Electrical Eng.)

## 版本

V5.12 Modular

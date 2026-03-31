# 变压器振动故障诊断系统 — Deep SVDD V5.12

---

## TII Paper Submission

**"Anti-Collapse Deep SVDD for Unsupervised Fault Detection in Power Transformers"**
Submitted to *IEEE Transactions on Industrial Informatics* (Regular Paper)

This repository contains the full implementation supporting the above paper. The proposed framework is a three-stage unsupervised pipeline trained on unlabeled vibration recordings from 200+ in-service HVDC converter transformers (2,024,188 segments, ~200 GB):

- **Stage 1** — Anti-collapse Deep SVDD with VAE pretraining (Hetero branch: ResNet-18 + 3-channel CWT/STFT/raw images at 224×224)
- **Stage 2** — Percentile-based pseudo-labeling (20th/80th percentile score thresholds)
- **Stage 3** — Lightweight MLP fine-tuning with optional MMD domain alignment

Fleet-scale test results (2,250 labeled samples): **Acc 60.6% / macro-F1 59.6% / AUC 0.666**, outperforming OC-SVM by +20.6 pp and Isolation Forest by +26.3 pp in macro-F1.

Paper support materials (figures, result JSONs, hyperparameter config) are in [`paper_support/`](paper_support/).

---

基于深度半监督学习的变压器振动信号故障检测与分类系统。采用三阶段训练流程，结合 Deep SVDD 无监督异常检测与有监督故障分类，支持多模态双分支输入（时频图像 + 1200 维工程特征）。

---

## 目录

- [项目概述](#项目概述)
- [文件结构](#文件结构)
- [模型结构](#模型结构)
- [特征维度详解](#特征维度详解)
- [损失函数详解](#损失函数详解)
- [数据流向](#数据流向)
- [三阶段训练流程](#三阶段训练流程)
- [实验模式](#实验模式)
- [快速开始](#快速开始)
- [核心配置参数](#核心配置参数)
- [性能对比](#性能对比)

---

## 项目概述

| 属性 | 说明 |
|------|------|
| 任务 | 变压器振动信号异常检测 + 故障分类 |
| 学习方式 | 半监督（无标签数据预训练 → 伪标签 → 有标签微调） |
| 核心算法 | Deep SVDD + VAE 重建辅助损失 |
| 输入模态 | 时频图像（Hetero）+ 工程特征向量（Zerone） |
| 版本 | V5.12 Modular |

---

## 文件结构

```
Deep SVDD/
├── main.py               # 主入口（argparse，三阶段流程调度）
├── config.py             # ThreeStageConfigV5 配置类
├── models.py             # 全部神经网络模块定义
├── datasets.py           # 数据集类（CSV / JSONL / 带标签）
├── features.py           # 1200维 Zerone 特征提取
├── training.py           # 三阶段训练函数
├── gpu_transforms.py     # GPU 批量 CWT/STFT 加速
├── visualization.py      # VisualizationManager 可视化
├── utils.py              # GlobalNormalizer / 检查点 / 早停
├── data_manager.py       # 数据划分与通道管理
├── evaluation.py         # 评估指标计算
├── data_leakage_filter.py# 数据泄漏过滤预处理
├── filtered_output/      # 预过滤后的 JSONL 数据
│   ├── val/正常/ 故障/
│   └── test/正常/ 故障/
└── three_stage_results_v512/   # 训练输出目录
    ├── branch_hetero/
    ├── branch_zerone/
    └── branch_dual/fusion_gmu/
        ├── stage1_anomaly/
        ├── stage2_pseudo/
        ├── stage3_classify/
        └── models/
```

---

## 模型结构

### 总体架构图

```
振动信号 (8192点, fs=8192Hz)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
【Hetero 分支】                             【Zerone 分支】
GPU: CWT + STFT + 时域reshape              1200维工程特征提取
        │                                          │
        ▼                                          ▼
3通道 224×224 图像                         归一化后:
[R=CWT, G=STFT, B=时域]                   ├── CNN模式: 转为224×224 raster图像
        │                                  └── MLP模式: 保持1200维向量
        ▼                                          │
  HeteroCNN                                        ▼
(ResNet18 backbone)                      ZeroneCNN (ResNet18)
        │                                或 ZeroneMLP (2层MLP)
        ▼                                          │
  h_img (512维)                             h_feat (512维)
        │                                          │
        └─────────────┬─────────────────────────────┘
                      │
              ModalityDropout
            (训练时随机丢弃一模态，
             可学习缺失token补偿)
                      │
                      ▼
              ┌── 融合模块 ──┐
              │ ConcatFusion │ 拼接 + Linear
              │ AttentionFusion│ Softmax门控加权
              │ GatedFusion  │ 交叉门控
              │ GMUFusion ★  │ 门控多模态单元（推荐）
              └──────────────┘
                      │
                  h_fused (512维)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  【Stage 1: 异常检测】      【Stage 3: 故障分类】
   SVDD投影头                 分类器头
   512→128→64 (latent)       512→256→128→2
          │                       │
   z_svdd (64维)            logits (2类)
          │                       │
   SVDD score =             CrossEntropy Loss
   ||z - center||²                │
          │                  可选: DomainDiscriminator
   可选VAE解码:              (DANN 梯度反转)
   h → μ,σ → 重建图像
   L1重建损失 + KL散度
```

### 各模块说明

#### 1. HeteroCNN / ZeroneCNN
- 骨干网络：ResNet18（预训练）
- 去掉最后的全连接层（`children()[:-1]`），取 512 维全局平均池化特征
- 接一个线性层投影到 `cnn_feat_dim`（默认 512 维）

#### 2. ZeroneMLP（兼容模式）
```
输入 1200维 → Linear(1200, 512) → BN/LN → ReLU → Dropout
           → Linear(512, 512) → BN/LN → ReLU → Dropout
           → Linear(512, 512)
```

#### 3. GMUFusion（推荐融合方式）
```python
h_img_t = Tanh(Linear(h_img))        # (B, 512)
h_feat_t = Tanh(Linear(h_feat))      # (B, 512)
z = Sigmoid(Linear(cat(h_img, h_feat)))  # 门控 (B, 512)
h_fused = z * h_img_t + (1-z) * h_feat_t  # 软选择
h_fused = BN/LN(h_fused)
```

#### 4. AnomalyModelV5（异常检测主模型）
- **SVDD 投影头**：`512 → 128 (BN + LeakyReLU) → 64`
- **SVDD 超球中心**：用训练集均值初始化，注册为 buffer
- **VAE 解码器**（仅 hetero/dual 模式）：
  - `h → μ, logvar` (形状 `latent_channels × 7 × 7`)
  - 5 层 ConvTranspose2d 上采样到 `224×224`
  - 重建损失 = L1 + KL
- **综合异常分**：`α × SVDD_norm + (1-α) × VAE_norm`（α=0.6）

#### 5. FaultClassifierV5（故障分类器）
- 复用 Stage 1 训练好的编码器（默认冻结）
- 分类头：`512 → 256 (BN+ReLU+Drop) → 128 (BN+ReLU+Drop) → 2`
- 可选 DANN 域判别器（梯度反转层 + 3层 MLP）

---

## 特征维度详解

### Zerone 1200 维工程特征

| 子集 | 维度 | 计算方式 | 特征列表 |
|------|------|---------|---------|
| 时域统计 | **15** | 直接统计 | mean, rms, var, std, max, min, p2p, kurtosis, skewness, zero_cross_rate, mean_abs, crest_factor, impulse_factor, margin_factor, waveform_factor |
| STFT 段均值 | **127** | `scipy.stft`，nperseg=128, noverlap=64，去 DC bin，取各时间帧的频率幅值均值 | 频带 1~127（第 0 bin 去除） |
| PSD 频带 | **1050** | `scipy.welch`，nperseg=4096，插值到 1 Hz 栅格 | 1–1000 Hz：1 Hz 分辨率（1000 维）；1001–2000 Hz：每 20 Hz 聚合（50 维） |
| 高频能量比 | **8** | 4 个截止频率 × 2 指标 | 截止：1000/2000/3000/4000 Hz；每个截止点输出幅值比 + 功率比 |
| **合计** | **1200** | `np.concatenate([time, stft, psd, hf])` | — |

各指标定义（时域）：

| 指标 | 公式 |
|------|------|
| RMS | $\sqrt{\frac{1}{N}\sum x_i^2}$ |
| 峰峰值 | $\max(x) - \min(x)$ |
| 峭度 | $\mu_4 / \mu_2^2$（标准四阶矩） |
| 偏度 | $\mu_3 / \sigma^3$ |
| 过零率 | 相邻采样点符号变化次数 / $(N-1)$ |
| 波峰因子 | $\max(x) / \text{RMS}$ |
| 脉冲因子 | $\max(x) / \bar{|x|}$ |
| 裕度因子 | $\max(x) / \sqrt[4]{\overline{|x|^4}}$ |
| 波形因子 | $\text{RMS} / \bar{|x|}$ |

### 网络各层维度

```
输入信号        [B, 8192]
GPU CWT         [B, 64, 8192] → 压缩 → [B, 1, 224, 224]
GPU STFT        [B, 129, T]   → 归一化 → [B, 1, 224, 224]
时域 reshape    [B, 8192]     → reshape → [B, 1, 224, 224]
三通道拼接      [B, 3, 224, 224]
ResNet18        [B, 512, 1, 1] → squeeze → [B, 512]
Linear(512,512) [B, 512]       h_img / h_feat

GMU 门控        [B, 512]       h_fused
SVDD 投影头     512 → 128 → 64  z_svdd
VAE μ/σ        [B, 64, 7, 7]
VAE 解码        [B, 64,7,7] → [B,256,14,14] → [B,128,28,28]
                → [B,64,56,56] → [B,32,112,112] → [B,3,224,224]
分类头          512 → 256 → 128 → 2  logits
```

---

## 损失函数详解

### Stage 1：无监督 SVDD 预训练

**总损失**（有 VAE 时）：

$$L_{\text{total}} = 0.5 \times L_{\text{SVDD}}^{*} + 0.5 \times L_{\text{VAE}}$$

**SVDD 损失（含防崩塌正则）**：

$$L_{\text{SVDD}}^{*} = \underbrace{\frac{1}{B}\sum_{i=1}^B \|z_i - c\|^2}_{L_{\text{SVDD}}} + 0.01 \times \underbrace{(-\log(R_{90} + \epsilon))}_{L_{\text{radius}}} + 0.001 \times \underbrace{\frac{1}{\text{Var}(z) + \epsilon}}_{L_{\text{diversity}}}$$

- $c$：SVDD 超球中心（训练集均值初始化，注册为 buffer）
- $R_{90}$：batch 内 SVDD 分数的 90% 分位数，作为软边界半径
- $L_{\text{radius}}$：半径正则，防止中心塌缩（半径趋零时给大惩罚）
- $L_{\text{diversity}}$：多样性正则，鼓励 $z$ 在各维度有方差，防止输出退化

**VAE 损失**：

$$L_{\text{VAE}} = \underbrace{\frac{1}{B}\sum \|x - \hat{x}\|_1}_{L_1\text{ 重建}} + \beta_t \times 0.01 \times \underbrace{\frac{1}{B}\sum \left(-\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)\right)}_{L_{\text{KL}}}$$

- $\beta_t = \min(1.0,\ t / 10)$：KL warmup，前 10 epoch 线性增长至 1
- $\text{logvar}$ 被 clamp 到 $[-10, 10]$ 防止数值爆炸

**优化器分组**（防止 SVDD 头过快学习）：

| 参数组 | 学习率 | 权重衰减 |
|--------|--------|---------|
| 编码器（encoder） | $lr$ | $10^{-4}$ |
| SVDD 投影头（svdd_proj） | $lr \times 0.1$ | $10^{-3}$ |

梯度裁剪：`clip_grad_norm_(model.parameters(), max_norm=1.0)`

调度器：`CosineAnnealingLR`

---

### Stage 2：异常分计算与伪标签阈值

$$\text{score}_i = \alpha \cdot \frac{s_i^{\text{SVDD}}}{\bar{s}^{\text{SVDD}}} + (1-\alpha) \cdot \frac{s_i^{\text{VAE}}}{\bar{s}^{\text{VAE}}}, \quad \alpha = 0.6$$

$$\text{label}_i = \begin{cases} 0\ (\text{正常}) & \text{score}_i < Q_{0.2} \\ 1\ (\text{故障}) & \text{score}_i > Q_{0.8} \\ \text{丢弃} & \text{其他} \end{cases}$$

---

### Stage 3：有监督分类微调

$$L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{DA}} \times L_{\text{MMD}}$$

$$L_{\text{CE}} = -\sum_k y_k' \log p_k, \quad y_k' = (1 - \epsilon) y_k + \frac{\epsilon}{K}$$

- Label smoothing：$\epsilon = 0.05$，$K = 2$
- $L_{\text{MMD}}$：对 batch 内正常/故障样本的编码器特征 $h$ 计算 RBF-MMD，$\lambda_{\text{DA}} = 0.1$（`DA_WEIGHT`）

**MMD / CORAL 公式**：

$$L_{\text{MMD}} = \frac{1}{n_s^2}\sum_{i,j} k(s_i, s_j) + \frac{1}{n_t^2}\sum_{i,j} k(t_i, t_j) - \frac{2}{n_s n_t}\sum_{i,j} k(s_i, t_j)$$

$$k(x, y) = \exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right), \quad \sigma = 1.0$$

$$L_{\text{CORAL}} = \frac{1}{4d^2}\|C_s - C_t\|_F^2$$

其中 $C_s, C_t$ 为源域/目标域特征的协方差矩阵，$d = 512$。

---

## 数据流向

### 输入数据格式

- 原始格式：JSONL 文件，每行一个 JSON，`signal_value` 字段存储逗号分隔的振动采样点
- 信号长度：8192 点，采样率 8192 Hz
- 目录结构：`val/正常/*.jsonl`、`test/故障/*.jsonl`（标签由父目录名推断）

### Hetero 特征提取流程（GPU 加速）

```
原始信号 [B, 8192]
    │
    ├── CWT (Morlet 小波, 64尺度)    → [B, 64, 8192]
    │       GPU 预计算小波核
    │
    ├── STFT (n_fft=256, hop=64)     → [B, 129, T]
    │
    └── 时域 reshape                 → [B, 1, H, W]
            │
            ▼
    拼接为 3通道图像                  → [B, 3, 224, 224]
    ├── R通道: CWT 压缩
    ├── G通道: STFT 归一化
    └── B通道: 时域二维化
```

> CPU 单样本 ~50ms → GPU 批量(128) ~0.5ms/样本，**提速约 100 倍**

### Zerone 特征提取流程（1200维）

```
原始信号 (8192点)
    │
    ├── 时域统计 (15维)
    │   mean, rms, var, std, max, min, p2p,
    │   kurtosis, skewness, zero_cross_rate,
    │   mean_abs, crest_factor, impulse_factor,
    │   margin_factor, waveform_factor
    │
    ├── STFT 段均值 (127维, 去DC)
    │   nperseg=128, noverlap=64
    │
    ├── PSD 频带均值 (1050维)
    │   功率谱密度分段统计
    │
    └── 高频能量 (8维)
            │
            ▼
    拼接 → 1200维特征向量
            │
    全局 Z-score 归一化 (GlobalNormalizer)
            │
            ├── CNN模式: 光栅化为 224×224 图像
            └── MLP模式: 直接输入 MLP 编码器
```

### 数据集加载优先级

```
main.py → prepare_datasets()
    │
    ├── 优先: filter_output_dir（预过滤 JSONL）
    │         ├── 训练集: CSV 原始数据（排除已标注样本）
    │         └── 验证/测试: filtered_output/val, test
    │
    ├── 其次: labeled_dir（自动按 val_ratio 划分）
    │
    └── 默认: PROJECT_ROOT 目录结构加载
```

---

## 三阶段训练流程

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1: 无监督异常检测预训练                                      │
│                                                                    │
│  训练数据: 大量无标签正常(含少量未知故障)振动数据                   │
│                                                                    │
│  损失函数:                                                         │
│    L = L_svdd + λ_vae × L_vae + λ_da × L_domain               │
│                                                                    │
│    L_svdd = mean(||z - center||²)        [SVDD超球压缩]           │
│    L_vae  = L1(recon, img) + β×KL       [VAE重建辅助]            │
│    L_da   = MMD(train, val) 或 CORAL     [域适应]                 │
│                                                                    │
│  输出: 训练好的 AnomalyModelV5 + SVDD center                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 2: 伪标签生成                                                │
│                                                                    │
│  对训练集所有样本计算异常分 score                                   │
│  按 quantile 分位数阈值二分:                                        │
│    score < Q_low  → 伪标签: 正常 (0)                               │
│    score > Q_high → 伪标签: 故障 (1)                               │
│    中间区域        → 丢弃 (uncertain)                              │
│                                                                    │
│  输出: pseudo_labels.npz                                           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 3: 有监督故障分类微调                                        │
│                                                                    │
│  训练数据:                                                         │
│    真实标签 (val/test 的有标注数据)                                 │
│  + 伪标签 (Stage 2 高置信度样本)                                   │
│                                                                    │
│  模型: FaultClassifierV5                                           │
│    = Stage1 冻结编码器 + 新分类头                                   │
│    可选: 逐步解冻 fusion_only → last_layers → all                  │
│                                                                    │
│  损失: CrossEntropy (label smoothing=0.05)                         │
│        + 可选 DANN 域对齐                                           │
│                                                                    │
│  输出: 故障分类器 + 评估报告 (Acc / Precision / Recall / F1)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 实验模式

系统支持 8 种模式组合（可通过 `--run_all_modes` 一次性跑完并生成对比图）：

| # | 模式名 | branch | fusion | Zerone架构 |
|---|--------|--------|--------|-----------|
| 1 | hetero | hetero | — | — |
| 2 | zerone_cnn | zerone | — | CNN |
| 3 | zerone_mlp | zerone | — | MLP |
| 4 | dual_concat | dual | concat | CNN |
| 5 | dual_attention | dual | attention | CNN |
| 6 | dual_gate | dual | gate | CNN |
| 7 | **dual_gmu_cnn** | dual | **gmu** | CNN（推荐） |
| 8 | dual_gmu_mlp | dual | gmu | MLP |

---

## 快速开始

### 完整流程（推荐）

```cmd
python main.py --all ^
    --data_root "E:/CODE/DATA/vibration_data_2022_" ^
    --filter_output "./filtered_output" ^
    --output_root "./three_stage_results_v512" ^
    --branch dual --fusion_mode gmu ^
    --batch_size 128 --num_workers 4
```

### 运行所有 8 种模式对比

```cmd
python main.py --run_all_modes ^
    --filter_output "./filtered_output" ^
    --batch_size 128 --num_workers 0
```

### 单独运行某阶段

```cmd
python main.py --stage 1   # 仅 Stage 1 异常检测预训练
python main.py --stage 2   # 仅 Stage 2 伪标签生成
python main.py --stage 3   # 仅 Stage 3 分类微调
```

### 断点续跑（从第 N 个模式开始）

```cmd
python main.py --run_all_modes --start_from 5 --filter_output "./filtered_output"
```

---

## 核心配置参数

```python
# config.py — ThreeStageConfigV5

BRANCH_MODE   = 'dual'      # 'hetero' / 'zerone' / 'dual'
FUSION_MODE   = 'gmu'       # 'concat' / 'attention' / 'gate' / 'gmu'
ZERONE_USE_CNN = True       # Zerone 分支用 CNN(图像) 还是 MLP(向量)
USE_GPU_HETERO = True       # GPU 加速 CWT/STFT（推荐开启）

BATCH_SIZE    = 128         # A40 建议 128-256，笔记本 16-32
LR            = 1e-4        # Deep SVDD 官方推荐
STAGE1_EPOCHS = 50          # Stage 1 轮数
STAGE3_EPOCHS = 100         # Stage 3 轮数
NUM_WORKERS   = 4           # Windows 建议 2-4，卡死用 0

SVDD_LATENT_DIM  = 64      # SVDD 投影维度
QUANTILE_LOW     = 0.2     # 伪标签正常阈值
QUANTILE_HIGH    = 0.8     # 伪标签故障阈值

USE_MODALITY_DROPOUT = True        # 模态随机丢弃
MODALITY_DROPOUT_RATE = 0.2
USE_DOMAIN_ADAPTATION = True       # 域适应
DA_MODE = 'mmd'                    # 'mmd' / 'coral'
```

---

## 性能对比

### 训练速度

| 配置 | 速度 |
|------|------|
| 原始 CPU，workers=0 | ~27s/batch |
| +workers=4 + pin_memory | ~4s/batch |
| **+GPU CWT/STFT（V5.12）** | **~1s/batch** |

### GPU 内存占用估计

| batch_size | 显存占用 |
|-----------|---------|
| 64 | ~1.5 GB |
| 128 | ~2.5 GB |
| 256 | ~4.5 GB |

---

## 注意事项

1. **首次运行**：会自动计算并缓存 zerone 特征（`OUTPUT_ROOT/cache/`）和归一化参数（`OUTPUT_ROOT/global_normalizer.npz`），所有模式共用
2. **Windows 多进程**：若 DataLoader 卡死，设置 `--num_workers 0`
3. **SVDD 崩塌防护**：投影头使用 LeakyReLU（非 ReLU），center 初始化时 eps 截断避免接近零
4. **标签推断**：由 JSONL 父目录名关键字自动推断（"正常/normal" → 0，"故障/fault" → 1）
5. **关闭 GPU 加速**：在 `config.py` 中设 `USE_GPU_HETERO = False`，回退到 CPU CWT/STFT

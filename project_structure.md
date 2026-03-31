# 项目详细结构记录

## 模型层次（models.py）

```
AnomalyModelV5
├── encoder: BranchEncoderV5
│   ├── hetero_branch: HeteroCNN (ResNet18, 输出 512维)
│   ├── zerone_branch: ZeroneCNN (ResNet18) 或 ZeroneMLP (1200→512)
│   ├── modality_dropout: ModalityDropout (p=0.2, learnable token)
│   └── fusion_module: ConcatFusion / AttentionFusion / GatedFusion / GMUFusion
├── svdd_proj: Linear(512,128) → BN → LeakyReLU → Linear(128,64)
├── center: buffer [64] (训练集均值初始化)
└── vae_decoder: (仅 hetero/dual)
    ├── vae_mu / vae_logvar: Linear(512, 64*7*7)
    └── decoder: 5层 ConvTranspose2d → 224×224×3

FaultClassifierV5
├── encoder: BranchEncoderV5 (冻结)
├── classifier: 512→256(BN+ReLU+Drop)→128(BN+ReLU+Drop)→2
└── domain_discriminator: DomainDiscriminator (可选, DANN)
    └── GRL + Linear(512,256)→Linear(256,128)→Linear(128,2)
```

---

## 特征维度

| 特征 | 维度 | 说明 |
|------|------|------|
| Zerone 时域 | 15 | mean/rms/kurtosis/skewness 等 |
| Zerone STFT | 127 | 频带均值，去 DC |
| Zerone PSD | 1050 | 功率谱密度频带 |
| Zerone 高频 | 8 | 高频能量 |
| Zerone 合计 | **1200** | `TOTAL_FEAT_DIM` |
| 编码器输出 | 512 | `FEATURE_DIM` |
| SVDD 潜变量 | 64 | `SVDD_LATENT_DIM` |

---

## 数据集类（datasets.py）

- `CSVVibrationDataset`：从 CSV 文件加载原始振动数据（无标签训练集）
- `TransformerVibrationDataset`：从 JSONL 目录结构加载（通用）
- `LabeledVibrationDataset`：从 split_manager 的 samples 列表加载（有标签 val/test）

`__getitem__` 返回：`(data, zerone, label, sample_id)`
- GPU hetero 模式：`data` = 原始 signal [8192]
- CPU hetero 模式：`data` = 预处理好的图像 [3,224,224]
- zerone CNN 模式：`zerone` = raster 图像 [3,224,224]
- zerone MLP 模式：`zerone` = 特征向量 [1200]

---

## 配置关键参数

```python
BRANCH_MODE = 'dual'         # 'hetero' / 'zerone' / 'dual'
FUSION_MODE = 'gmu'          # 'concat' / 'attention' / 'gate' / 'gmu'
ZERONE_USE_CNN = True        # Zerone 分支架构
USE_GPU_HETERO = True        # GPU CWT/STFT 加速
SVDD_LATENT_DIM = 64         # SVDD 投影维度
QUANTILE_LOW = 0.2           # 伪标签正常阈值
QUANTILE_HIGH = 0.8          # 伪标签故障阈值
DA_MODE = 'mmd'              # 域适应：'mmd' / 'coral'
```

---

## 训练损失

### Stage 1
```
L_total = L_svdd + λ_vae * L_vae + λ_da * L_domain

L_svdd = mean(||z_svdd - center||²)
L_vae  = L1(recon, img) + β * KL
L_da   = MMD(source, target)  或  CORAL(source, target)
```

### Stage 3
```
L_total = CrossEntropy(logits, labels, smoothing=0.05)
        + λ_da * L_domain (可选)
```

---

## 8 种实验模式（run_all_modes）

1. hetero
2. zerone_cnn
3. zerone_mlp
4. dual_concat
5. dual_attention
6. dual_gate
7. dual_gmu_cnn（推荐）
8. dual_gmu_mlp

---

## 输出目录结构

```
three_stage_results_v512/
├── global_normalizer.npz          # 全局归一化参数
├── cache/                         # zerone 特征缓存
├── logs/training_*.log
├── branch_hetero/
│   ├── stage1_anomaly/
│   │   ├── training_curves/
│   │   ├── distributions/
│   │   └── feature_preview/
│   ├── stage2_pseudo/pseudo_labels.npz
│   ├── stage3_classify/
│   │   ├── tsne/
│   │   ├── confusion/
│   │   ├── roc_pr/
│   │   └── evaluation_results.json
│   └── models/
│       └── stage1/stage1_best_model.pth
└── branch_dual/fusion_gmu/   （同上结构）
```

---

## GPU 加速原理（gpu_transforms.py）

`GPUHeteroTransform`:
- 初始化时预计算 Morlet 小波核（64 尺度）和 Hann 窗
- forward: signal [B, 8192] → 3通道图像 [B, 3, 224, 224]
  - R: CWT 频谱（GPU 卷积）
  - G: STFT 幅度（torch.stft）
  - B: 时域二维化 reshape

DataLoader 在 GPU hetero 模式下返回原始 signal，training loop 内调用 `gpu_transform(signals)` 生成图像。

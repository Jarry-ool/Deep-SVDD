# Deep SVDD V5.12 — 开发参考文档

> 本文档面向开发者，记录代码内部约定、模块接口和调试要点，与 README.md 互补。

---

## 模块职责速查

| 文件 | 职责 |
|------|------|
| `main.py` | 主入口，argparse，三阶段流程调度 |
| `config.py` | `ThreeStageConfigV5` 数据类，所有超参 |
| `models.py` | 全部网络模块（编码器/融合/SVDD/分类器） |
| `datasets.py` | 数据集类，branch_mode 条件计算 |
| `features.py` | 1200维 Zerone 特征提取 |
| `training.py` | `train_stage1` / `run_stage2` / `train_stage3` |
| `gpu_transforms.py` | GPU 批量 CWT/STFT（约 100 倍加速） |
| `visualization.py` | `VisualizationManager` |
| `utils.py` | `GlobalNormalizer`、早停、检查点 |
| `data_manager.py` | `DataSplitManager`、`CHANNEL_MANAGER` |
| `evaluation.py` | 评估指标计算 |
| `data_leakage_filter.py` | 数据泄漏过滤预处理 |

---

## 模型类层次

```
AnomalyModelV5
├── encoder: BranchEncoderV5
│   ├── hetero_branch: HeteroCNN (ResNet18, 输出 512维)
│   ├── zerone_branch: ZeroneCNN (ResNet18) 或 ZeroneMLP (1200→512)
│   ├── modality_dropout: ModalityDropout (p=0.2, learnable missing token)
│   └── fusion_module: ConcatFusion / AttentionFusion / GatedFusion / GMUFusion
├── svdd_proj: Linear(512,128) → BN → LeakyReLU → Linear(128,64)
├── center: buffer [64]（训练集均值初始化）
└── vae_decoder: (仅 hetero/dual 模式)
    ├── vae_mu / vae_logvar: Linear(512, 64*7*7)
    └── decoder: 5层 ConvTranspose2d → 224×224×3

FaultClassifierV5
├── encoder: BranchEncoderV5（冻结）
├── classifier: 512→256(BN+ReLU+Drop)→128(BN+ReLU+Drop)→2
└── domain_discriminator: DomainDiscriminator（可选，DANN）
    └── GRL + Linear(512,256)→Linear(256,128)→Linear(128,2)
```

---

## 数据集类接口

| 类名 | 用途 |
|------|------|
| `CSVVibrationDataset` | 从 CSV 文件加载原始振动数据（无标签训练集） |
| `TransformerVibrationDataset` | 从 JSONL 目录结构加载（通用） |
| `LabeledVibrationDataset` | 从 `split_manager.samples` 列表加载（有标签 val/test） |

### `__getitem__` 返回格式

```python
(data, zerone, label, sample_id)
```

- **GPU hetero 模式**：`data` = 原始 signal `[8192]`，training loop 内调用 `gpu_transform(signals)` 生成图像
- **CPU hetero 模式**：`data` = 预处理好的图像 `[3, 224, 224]`
- **zerone CNN 模式**：`zerone` = raster 图像 `[3, 224, 224]`
- **zerone MLP 模式**：`zerone` = 特征向量 `[1200]`

---

## 关键数据路径约定

| 路径 | 内容 |
|------|------|
| `filtered_output/val/正常/` | 预过滤 JSONL，有标签验证集（正常） |
| `filtered_output/val/故障/` | 预过滤 JSONL，有标签验证集（故障） |
| `filtered_output/test/正常/` | 预过滤 JSONL，有标签测试集（正常） |
| `filtered_output/test/故障/` | 预过滤 JSONL，有标签测试集（故障） |
| `three_stage_results_v512/cache/` | Zerone 特征缓存（首次运行自动生成） |
| `three_stage_results_v512/global_normalizer.npz` | 全局归一化参数（所有模式共用） |

### 标签推断规则

标签由父目录名关键字自动推断：
- 含 "正常" / "normal" → label `0`
- 含 "故障" / "fault" → label `1`

---

## 输出目录结构

```
three_stage_results_v512/
├── global_normalizer.npz
├── cache/                              # zerone 特征缓存
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
└── branch_dual/fusion_gmu/             # 同上结构
```

---

## GPU 加速原理（gpu_transforms.py）

`GPUHeteroTransform.forward`：

```
输入: signal [B, 8192]
  │
  ├── CWT (Morlet 小波, 64尺度)
  │   初始化时预计算小波核，GPU 卷积
  │   → [B, 64, 8192] → 压缩 → [B, 1, 224, 224]  (R通道)
  │
  ├── STFT (torch.stft, n_fft=256, hop=64)
  │   → [B, 129, T] → 归一化 → [B, 1, 224, 224]   (G通道)
  │
  └── 时域 reshape
      → [B, 8192] → reshape → [B, 1, 224, 224]    (B通道)

输出: [B, 3, 224, 224]
```

DataLoader 在 GPU hetero 模式下返回原始 signal，training loop 内调用 `gpu_transform(signals)` 生成图像（非 Dataset 内部处理）。

---

## 优化器参数分组（Stage 1）

| 参数组 | 学习率 | 权重衰减 | 原因 |
|--------|--------|---------|------|
| `encoder` | `lr` | `1e-4` | 正常学习 |
| `svdd_proj` | `lr × 0.1` | `1e-3` | 防止投影头过快学习导致中心崩塌 |

---

## 常用运行命令

```bash
# 推荐：完整流程（dual+gmu）
python main.py --all --filter_output ./filtered_output --batch_size 128 --num_workers 0

# 所有 8 种模式对比
python main.py --run_all_modes --filter_output ./filtered_output --batch_size 128

# 从第 N 个模式续跑
python main.py --run_all_modes --start_from 5 --filter_output ./filtered_output

# 单阶段
python main.py --stage 1
python main.py --stage 2
python main.py --stage 3
```

---

## 调试与注意事项

- **Windows 多进程卡死**：设 `--num_workers 0`
- **SVDD 崩塌防护**：投影头用 LeakyReLU（非 ReLU），center 初始化时 eps 截断
- **首次运行**：自动计算并缓存 zerone 特征和归一化参数，所有模式共用，勿重复计算
- **关闭 GPU 加速**：`config.py` 中设 `USE_GPU_HETERO = False`，回退 CPU CWT/STFT
- **branch_mode 条件**：datasets.py 中 `__getitem__` 有 `branch_mode` 条件分支，修改时需同步更新
- **VAE 仅 hetero/dual 模式启用**：zerone 单分支模式无 VAE 解码器

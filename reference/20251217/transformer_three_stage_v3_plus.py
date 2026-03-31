# -*- coding: utf-8 -*-
"""
transformer_three_stage_v3_plus.py
===================================

交流变压器振动数据 三阶段渐进式故障诊断系统 V3+ (修复版)
整合 Zerone 1200维工程特征 + Hetero 三通道时频图像

【V3+ 版本改进】
    ✅ 修复1: Stage3 解冻真正生效 - 通过重建optimizer或param_group
    ✅ 修复2: VAE解耦 - 仅用h_img重构图像,切断zerone负迁移
    ✅ 修复3: Residual Gate融合 - 以zerone为主干保底,门控加权图像分支
    ✅ 修复4: 三分支模式支持 - zerone-only / hetero-only / dual
    ✅ 新增: Gate可视化、解冻自检图、阈值稳定性曲线等

【运行方式】
    python transformer_three_stage_v3_plus.py --all --branch dual
    python transformer_three_stage_v3_plus.py --stage 1 --branch zerone

Author: 基于 v3 框架修复
适用领域: 电气工程 - 变压器振动故障诊断
"""

# =============================================================================
# 第0步: 导入依赖库
# =============================================================================
import os
import sys
import json
import argparse
import warnings
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import models
from tqdm import tqdm

# 信号处理库
import pywt
import cv2
from scipy.signal import stft, welch
from scipy.stats import spearmanr

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# 评估指标
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


# =============================================================================
# 第1步: 配置类定义
# =============================================================================
@dataclass
class ThreeStageConfigV3Plus:
    """
    三阶段诊断系统配置类 V3+ (修复版)
    
    【V3+ 新增配置】
        BRANCH_MODE: 分支模式 (zerone/hetero/dual)
        FUSION_MODE: 融合方式 (concat/gate/residual_gate)
        VAE_DECOUPLE: VAE解耦开关 (True=仅用h_img重构)
    """
    
    # ================= 路径配置 =================
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v3plus"))
    
    # ================= 信号/物理参数 =================
    FS: float = 8192.0
    SIGNAL_LEN: int = 8192
    INPUT_SIZE: int = 224
    
    # ================= 模型参数 =================
    LATENT_DIM: int = 128
    LATENT_CHANNELS: int = 64
    ZERONE_DIM: int = 1200
    MLP_FEAT_DIM: int = 256
    CNN_FEAT_DIM: int = 512
    FUSED_DIM: int = 512
    
    # ================= V3+ 新增配置 =================
    BRANCH_MODE: str = "dual"            # zerone / hetero / dual
    FUSION_MODE: str = "residual_gate"   # concat / gate / residual_gate
    VAE_DECOUPLE: bool = True            # VAE解耦
    HAS_VAE: bool = True                 # 是否启用VAE
    
    # Gate相关配置
    GATE_WARMUP_EPOCHS: int = 10
    GATE_REG_LAMBDA: float = 0.01
    GATE_RESIDUAL_ALPHA: float = 0.3
    
    # ================= 训练参数 =================
    BATCH_SIZE: int = 16
    STAGE1_EPOCHS: int = 50
    STAGE3_EPOCHS: int = 30
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NU: float = 0.05
    BETA_VAE: float = 0.01
    BETA_WARMUP: int = 10
    
    # ================= 伪标签阈值 =================
    NORMAL_PERCENTILE: float = 5.0
    ANOMALY_PERCENTILE: float = 99.0
    
    # ================= Stage3解冻配置 =================
    UNFREEZE_EPOCH: int = 15
    UNFREEZE_LR_FACTOR: float = 0.1
    UNFREEZE_REBUILD_OPTIMIZER: bool = True
    
    # ================= 类别关键词 =================
    CLASS_KEYWORDS: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    })
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    VIZ_DPI: int = 300
    VIZ_LANG: str = "cn"
    
    def __post_init__(self):
        """初始化后处理"""
        self.PROJECT_ROOT = Path(self.PROJECT_ROOT)
        self.OUTPUT_ROOT = Path(self.OUTPUT_ROOT)
        
        self.TRAIN_DIR = self.PROJECT_ROOT / "train"
        self.VAL_DIR = self.PROJECT_ROOT / "val"
        self.TEST_DIR = self.PROJECT_ROOT / "test"
        
        branch_suffix = f"_{self.BRANCH_MODE}"
        self.STAGE1_DIR = self.OUTPUT_ROOT / f"stage1_unsupervised{branch_suffix}"
        self.STAGE2_DIR = self.OUTPUT_ROOT / f"stage2_pseudo_labels{branch_suffix}"
        self.STAGE3_DIR = self.OUTPUT_ROOT / f"stage3_supervised{branch_suffix}"
        self.VIZ_DIR = self.OUTPUT_ROOT / f"visualizations{branch_suffix}"
        self.MODEL_DIR = self.OUTPUT_ROOT / f"models{branch_suffix}"
        
        for d in [self.STAGE1_DIR, self.STAGE2_DIR, self.STAGE3_DIR, 
                  self.VIZ_DIR, self.MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 可视化子目录(保留原有 + 新增)
        viz_subdirs = [
            "training_curves", "score_dist", "confusion", "roc_pr",
            "tsne", "samples", "reconstruction",  # 原有
            "gate_analysis", "unfreeze_check", "error_gallery", "threshold_stability",  # 新增
        ]
        for subdir in viz_subdirs:
            (self.VIZ_DIR / subdir).mkdir(exist_ok=True)
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("三阶段故障诊断系统配置 V3+ (修复版)")
        print("="*70)
        print(f"【数据路径】 {self.PROJECT_ROOT}")
        print(f"【V3+ 配置】")
        print(f"  分支模式: {self.BRANCH_MODE} | 融合方式: {self.FUSION_MODE}")
        print(f"  VAE解耦: {self.VAE_DECOUPLE} | Gate预热: {self.GATE_WARMUP_EPOCHS}轮")
        print(f"  解冻epoch: {self.UNFREEZE_EPOCH} | 解冻lr系数: {self.UNFREEZE_LR_FACTOR}")
        print(f"【设备】 {self.DEVICE}")
        print("="*70 + "\n")


# =============================================================================
# 第2步: 可视化设置
# =============================================================================
def setup_plotting():
    """设置matplotlib绘图参数"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['font.size'] = 10

COLORS = {
    'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
    'red': '#d62728', 'purple': '#9467bd', 'gray': '#7f7f7f',
    'normal': '#2ca02c', 'fault': '#d62728', 'uncertain': '#7f7f7f'
}

LABELS = {
    'cn': {
        'normal': '正常', 'fault': '故障', 'uncertain': '不确定',
        'epoch': '训练轮次', 'loss': '损失值', 'accuracy': '准确率',
        'score': '异常得分', 'count': '样本数', 'true': '真实标签',
        'pred': '预测标签', 'precision': '精确率', 'recall': '召回率',
        'f1': 'F1分数', 'svdd_loss': 'SVDD损失', 'vae_loss': 'VAE损失',
        'gate_weight': '门控权重', 'grad_norm': '梯度范数',
    },
    'en': {
        'normal': 'Normal', 'fault': 'Fault', 'uncertain': 'Uncertain',
        'epoch': 'Epoch', 'loss': 'Loss', 'accuracy': 'Accuracy',
        'score': 'Anomaly Score', 'count': 'Count', 'true': 'True Label',
        'pred': 'Predicted Label', 'precision': 'Precision', 'recall': 'Recall',
        'f1': 'F1 Score', 'svdd_loss': 'SVDD Loss', 'vae_loss': 'VAE Loss',
        'gate_weight': 'Gate Weight', 'grad_norm': 'Gradient Norm',
    }
}


# =============================================================================
# 第3步: 数据读取与处理工具
# =============================================================================
def parse_signal_value(v: Any, target_len: int = 8192) -> Optional[np.ndarray]:
    """解析信号数据"""
    try:
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("\n", " ").replace("\r", "")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            arr = np.array([float(p) for p in parts], dtype=np.float32)
        elif isinstance(v, (list, tuple)):
            arr = np.array([float(x) for x in v], dtype=np.float32)
        else:
            return None
    except Exception:
        return None
    
    if arr.size >= target_len:
        return arr[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:arr.size] = arr
    return out


def read_jsonl_file(filepath: Path) -> List[Dict]:
    """读取JSONL格式文件"""
    records = []
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            s = line.strip()
            if s:
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[警告] 读取文件失败: {filepath}, 错误: {e}")
    return records


def read_json_file(filepath: Path) -> List[Dict]:
    """读取JSON格式文件"""
    records = []
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        data = json.loads(text)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
    except Exception as e:
        print(f"[警告] 读取文件失败: {filepath}, 错误: {e}")
    return records


def signal_to_cwt_image(sig: np.ndarray, fs: float, size: int = 224) -> np.ndarray:
    """信号转CWT图像 (单通道复制到三通道)"""
    scales = np.arange(1, 129)
    cwt_matrix, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
    cwt_abs = np.abs(cwt_matrix).astype(np.float32)
    cwt_img = cv2.resize(cwt_abs, (size, size))
    cwt_img = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    img = np.stack([cwt_img, cwt_img, cwt_img], axis=0)
    return img.astype(np.float32)


def signal_to_hetero_image(sig: np.ndarray, fs: float, size: int = 224) -> np.ndarray:
    """
    信号转Hetero三通道图像
    Ch0: CWT (Morlet小波)
    Ch1: STFT (短时频谱)
    Ch2: Context (波形折叠)
    """
    # Ch0: CWT
    scales = np.arange(1, 129)
    cwt_matrix, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
    cwt_abs = np.abs(cwt_matrix).astype(np.float32)
    cwt_img = cv2.resize(cwt_abs, (size, size))
    cwt_img = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    
    # Ch1: STFT
    try:
        nperseg = min(256, len(sig) // 4)
        f, t, Zxx = stft(sig, fs=fs, nperseg=nperseg)
        stft_abs = np.abs(Zxx).astype(np.float32)
        stft_img = cv2.resize(stft_abs, (size, size))
        stft_img = (stft_img - stft_img.min()) / (stft_img.max() - stft_img.min() + 1e-8)
    except:
        stft_img = cwt_img.copy()
    
    # Ch2: Context (波形折叠)
    try:
        n_periods = 64
        period_len = len(sig) // n_periods
        if period_len > 0:
            folded = sig[:n_periods * period_len].reshape(n_periods, period_len)
            ctx_img = cv2.resize(folded, (size, size))
            ctx_img = (ctx_img - ctx_img.min()) / (ctx_img.max() - ctx_img.min() + 1e-8)
        else:
            ctx_img = cwt_img.copy()
    except:
        ctx_img = cwt_img.copy()
    
    img = np.stack([cwt_img, stft_img, ctx_img], axis=0)
    return img.astype(np.float32)


def extract_zerone_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    提取Zerone 1200维特征 (简化版)
    包含: 时域15维 + STFT127维 + PSD1050维 + 高频8维 = 1200维
    """
    features = []
    
    # 时域特征 (15维)
    time_feat = [
        np.mean(sig),                    # 均值
        np.std(sig),                     # 标准差
        np.max(sig),                     # 最大值
        np.min(sig),                     # 最小值
        np.max(sig) - np.min(sig),       # 峰峰值
        np.sqrt(np.mean(sig**2)),        # RMS
        np.mean(np.abs(sig)),            # 平均绝对值
        np.max(np.abs(sig)) / (np.sqrt(np.mean(sig**2)) + 1e-8),  # 峰值因子
        np.max(np.abs(sig)) / (np.mean(np.abs(sig)) + 1e-8),      # 脉冲因子
        np.sqrt(np.mean(sig**2)) / (np.mean(np.abs(sig)) + 1e-8), # 波形因子
        np.sum(sig**2),                  # 能量
        np.mean(sig**3),                 # 偏度近似
        np.mean(sig**4),                 # 峰度近似
        np.sum(np.abs(np.diff(sig))),    # 一阶差分绝对和
        np.sum(np.abs(np.diff(np.diff(sig)))),  # 二阶差分绝对和
    ]
    features.extend(time_feat)
    
    # STFT特征 (127维)
    try:
        nperseg = min(256, len(sig) // 4)
        f, t, Zxx = stft(sig, fs=fs, nperseg=nperseg)
        stft_mag = np.abs(Zxx)
        stft_mean = np.mean(stft_mag, axis=1)  # 每个频率bin的平均
        # 插值到127维
        if len(stft_mean) != 127:
            stft_mean = np.interp(
                np.linspace(0, len(stft_mean)-1, 127),
                np.arange(len(stft_mean)),
                stft_mean
            )
        features.extend(stft_mean.tolist())
    except:
        features.extend([0.0] * 127)
    
    # PSD特征 (1050维)
    try:
        f_psd, psd = welch(sig, fs=fs, nperseg=min(1024, len(sig) // 2))
        # 插值到1050维
        if len(psd) != 1050:
            psd = np.interp(
                np.linspace(0, len(psd)-1, 1050),
                np.arange(len(psd)),
                psd
            )
        features.extend(psd.tolist())
    except:
        features.extend([0.0] * 1050)
    
    # 高频特征 (8维)
    try:
        f_psd, psd = welch(sig, fs=fs, nperseg=min(1024, len(sig) // 2))
        total_power = np.sum(psd) + 1e-8
        thresholds = [1000, 2000, 3000, 4000]  # Hz
        for thr in thresholds:
            idx = np.where(f_psd >= thr)[0]
            if len(idx) > 0:
                high_power = np.sum(psd[idx])
                high_amp = np.sqrt(high_power)
            else:
                high_power = 0.0
                high_amp = 0.0
            features.extend([high_amp / (np.sqrt(total_power) + 1e-8), 
                           high_power / total_power])
    except:
        features.extend([0.0] * 8)
    
    # 确保长度为1200
    features = np.array(features, dtype=np.float32)
    if len(features) < 1200:
        features = np.pad(features, (0, 1200 - len(features)))
    elif len(features) > 1200:
        features = features[:1200]
    
    return features


# =============================================================================
# 第4步: 数据集类
# =============================================================================
class DualBranchDataset(Dataset):
    """
    双分支数据集: 同时提供图像和特征
    
    【返回】
        image: (3, H, W) Hetero三通道图像
        zerone: (1200,) Zerone特征向量
        label: 标签 (-1表示无标签)
        idx: 样本索引
    """
    
    def __init__(self, root_dir: Path, cfg: ThreeStageConfigV3Plus, 
                 use_labels: bool = False, return_both: bool = True):
        self.root_dir = Path(root_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.return_both = return_both  # True=返回图像和特征, False=根据BRANCH_MODE决定
        self.samples = []  # [(filepath, time_key, sig_list, label), ...]
        
        self._load_data()
    
    def _infer_label(self, path: Path) -> Optional[int]:
        """从路径推断标签"""
        path_str = str(path).lower()
        for keyword in self.cfg.CLASS_KEYWORDS["正常"]:
            if keyword.lower() in path_str:
                return 0
        for keyword in self.cfg.CLASS_KEYWORDS["故障"]:
            if keyword.lower() in path_str:
                return 1
        return None
    
    def _load_data(self):
        """加载数据"""
        if not self.root_dir.exists():
            print(f"[警告] 目录不存在: {self.root_dir}")
            return
        
        label_counts = defaultdict(int)
        
        for subdir in self.root_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            label = self._infer_label(subdir) if self.use_labels else None
            if self.use_labels and label is not None:
                label_counts[label] += 1
            
            # 遍历文件
            for fp in subdir.rglob("*.jsonl"):
                self._parse_file(fp, label)
            for fp in subdir.rglob("*.json"):
                if not fp.name.endswith('.jsonl'):
                    self._parse_file(fp, label)
        
        print(f"[{self.root_dir.name}] 加载 {len(self.samples)} 个样本")
        if self.use_labels and label_counts:
            for lbl, cnt in label_counts.items():
                lbl_name = "正常" if lbl == 0 else "故障"
                print(f"  {lbl_name}: {cnt} 个文件夹")
    
    def _parse_file(self, fp: Path, label: Optional[int]):
        """解析文件"""
        if fp.suffix == '.jsonl':
            records = read_jsonl_file(fp)
        else:
            records = read_json_file(fp)
        
        if not records:
            return
        
        # 按时间戳分组
        groups = {}
        for rec in records:
            time_key = None
            for key in ['data_time', 'dataTime', 'timestamp', 'time']:
                if key in rec and rec[key]:
                    time_key = str(rec[key])
                    break
            if not time_key:
                continue
            
            sig = parse_signal_value(rec.get('signal_value'), self.cfg.SIGNAL_LEN)
            if sig is None:
                continue
            
            groups.setdefault(time_key, []).append(sig)
        
        for time_key, sig_list in groups.items():
            self.samples.append((fp, time_key, sig_list, label))
    
    def _aggregate_channels(self, sig_list: List[np.ndarray]) -> np.ndarray:
        """多通道信号聚合（能量加权）"""
        if len(sig_list) == 1:
            return sig_list[0]
        
        X = np.stack(sig_list, axis=1)
        E = np.mean(X ** 2, axis=0) + 1e-12
        w = E / E.sum()
        x = X @ w
        return x.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        fp, time_key, sig_list, label = self.samples[idx]
        sig = self._aggregate_channels(sig_list)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        
        # 根据分支模式决定返回内容
        branch = self.cfg.BRANCH_MODE
        
        if self.return_both or branch == "dual":
            # 返回图像和特征
            img = signal_to_hetero_image(sig, self.cfg.FS, self.cfg.INPUT_SIZE)
            zerone = extract_zerone_features(sig, self.cfg.FS)
            return (torch.from_numpy(img), 
                    torch.from_numpy(zerone),
                    label if label is not None else -1, 
                    idx)
        elif branch == "zerone":
            # 仅返回特征 (需要保持接口一致,图像用空张量)
            zerone = extract_zerone_features(sig, self.cfg.FS)
            dummy_img = torch.zeros(3, self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE)
            return (dummy_img,
                    torch.from_numpy(zerone),
                    label if label is not None else -1,
                    idx)
        else:  # hetero
            # 仅返回图像
            img = signal_to_hetero_image(sig, self.cfg.FS, self.cfg.INPUT_SIZE)
            dummy_zerone = torch.zeros(self.cfg.ZERONE_DIM)
            return (torch.from_numpy(img),
                    dummy_zerone,
                    label if label is not None else -1,
                    idx)


# =============================================================================
# 第5步: 模型定义 - 子模块
# =============================================================================
class ZeroneMLP(nn.Module):
    """Zerone特征分支 MLP编码器"""
    
    def __init__(self, input_dim: int = 1200, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroCNN(nn.Module):
    """Hetero图像分支 CNN编码器 (基于ResNet18)"""
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, 512)


class ResidualGateFusion(nn.Module):
    """
    残差门控融合模块 (V3+ 核心修复)
    
    【公式】
        h'_zr = proj_zr(h_zr)              # 256 -> 512 维度对齐
        g = sigmoid(gate_net([h_img; h'_zr]))  # 门控权重
        h_gate = g * h_img + (1-g) * h'_zr     # 门控融合
        h_fused = h_gate + alpha * h'_zr       # 残差连接(保底)
    
    【保底机制】
        即使 g 全为 0, 仍有 alpha * h'_zr 残差贡献
        保证 dual 至少不差于 zerone-only
    """
    
    def __init__(self, dim_img: int = 512, dim_zr: int = 256, 
                 out_dim: int = 512, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        
        # Zerone维度投影 (256 -> 512)
        self.proj_zr = nn.Sequential(
            nn.Linear(dim_zr, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim_img + out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Sigmoid()
        )
        
        self.out_dim = out_dim
        
        # 初始化: 使gate初始输出接近0.5
        self._init_gate()
    
    def _init_gate(self):
        """初始化门控网络,使初始输出接近0.5"""
        # 最后一层bias初始化为0,使sigmoid(0)=0.5
        with torch.no_grad():
            self.gate_net[-2].bias.zero_()
            self.gate_net[-2].weight.data.normal_(0, 0.01)
    
    def forward(self, h_img: torch.Tensor, h_zr: torch.Tensor, 
                return_gate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        参数:
            h_img: (B, 512) 图像特征
            h_zr: (B, 256) Zerone特征
            return_gate: 是否返回门控权重
        返回:
            h_fused: (B, 512) 融合特征
            g: (B, 512) 门控权重 (可选)
        """
        # 维度对齐
        h_zr_proj = self.proj_zr(h_zr)  # (B, 512)
        
        # 计算门控权重
        g = self.gate_net(torch.cat([h_img, h_zr_proj], dim=1))  # (B, 512)
        
        # 门控融合
        h_gate = g * h_img + (1 - g) * h_zr_proj
        
        # 残差连接 (保底: 即使g=0,仍有zerone贡献)
        h_fused = h_gate + self.alpha * h_zr_proj
        
        if return_gate:
            return h_fused, g
        return h_fused


class SimpleConcatFusion(nn.Module):
    """简单拼接融合 (与v3兼容)"""
    
    def __init__(self, dim_img: int = 512, dim_zr: int = 256, out_dim: int = 512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim_img + dim_zr, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.out_dim = out_dim
    
    def forward(self, h_img: torch.Tensor, h_zr: torch.Tensor,
                return_gate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_cat = torch.cat([h_img, h_zr], dim=1)
        h_fused = self.fusion(h_cat)
        if return_gate:
            # 返回均匀权重作为占位
            g = torch.ones_like(h_fused) * 0.5
            return h_fused, g
        return h_fused


class GateFusion(nn.Module):
    """标准门控融合 (软注意力)"""
    
    def __init__(self, dim_img: int = 512, dim_zr: int = 256, out_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_img + dim_zr, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.proj_img = nn.Linear(dim_img, out_dim)
        self.proj_zr = nn.Linear(dim_zr, out_dim)
        self.out_dim = out_dim
    
    def forward(self, h_img: torch.Tensor, h_zr: torch.Tensor,
                return_gate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        g = self.gate(torch.cat([h_img, h_zr], dim=1))  # (B, 2)
        h1 = self.proj_img(h_img) * g[:, 0:1]
        h2 = self.proj_zr(h_zr) * g[:, 1:2]
        h_fused = h1 + h2
        if return_gate:
            # 扩展到512维方便可视化
            g_expanded = g[:, 0:1].expand(-1, self.out_dim)
            return h_fused, g_expanded
        return h_fused


# =============================================================================
# 第6步: 模型定义 - 主模型
# =============================================================================
class BranchEncoder(nn.Module):
    """
    分支编码器: 支持 zerone-only / hetero-only / dual 三种模式
    
    【V3+ 关键修改】
        - dual模式返回 (h_fused, h_img) 用于VAE解耦
        - 支持返回gate权重用于可视化
    """
    
    def __init__(self, cfg: ThreeStageConfigV3Plus):
        super().__init__()
        self.cfg = cfg
        self.branch_mode = cfg.BRANCH_MODE
        
        # 根据模式初始化分支
        if self.branch_mode in ["zerone", "dual"]:
            self.zerone_branch = ZeroneMLP(cfg.ZERONE_DIM, cfg.MLP_FEAT_DIM)
        
        if self.branch_mode in ["hetero", "dual"]:
            self.hetero_branch = HeteroCNN(cfg.CNN_FEAT_DIM)
        
        # 融合层 (仅dual模式)
        if self.branch_mode == "dual":
            if cfg.FUSION_MODE == "residual_gate":
                self.fusion = ResidualGateFusion(
                    cfg.CNN_FEAT_DIM, cfg.MLP_FEAT_DIM, cfg.FUSED_DIM,
                    alpha=cfg.GATE_RESIDUAL_ALPHA
                )
            elif cfg.FUSION_MODE == "gate":
                self.fusion = GateFusion(cfg.CNN_FEAT_DIM, cfg.MLP_FEAT_DIM, cfg.FUSED_DIM)
            else:  # concat
                self.fusion = SimpleConcatFusion(cfg.CNN_FEAT_DIM, cfg.MLP_FEAT_DIM, cfg.FUSED_DIM)
            self.output_dim = cfg.FUSED_DIM
        elif self.branch_mode == "zerone":
            # zerone-only: 需要投影到512维
            self.proj = nn.Sequential(
                nn.Linear(cfg.MLP_FEAT_DIM, cfg.FUSED_DIM),
                nn.BatchNorm1d(cfg.FUSED_DIM),
                nn.ReLU()
            )
            self.output_dim = cfg.FUSED_DIM
        else:  # hetero
            self.output_dim = cfg.CNN_FEAT_DIM
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor,
                return_gate: bool = False, return_branches: bool = False):
        """
        参数:
            image: (B, 3, H, W)
            zerone: (B, 1200)
            return_gate: 是否返回门控权重
            return_branches: 是否返回各分支特征 (用于VAE解耦)
        
        返回:
            h_fused: (B, 512) 融合/单分支特征
            h_img: (B, 512) 图像分支特征 (仅dual+return_branches)
            g: (B, 512) 门控权重 (仅return_gate)
        """
        h_img = None
        h_zr = None
        g = None
        
        if self.branch_mode == "zerone":
            h_zr = self.zerone_branch(zerone)
            h_fused = self.proj(h_zr)
        
        elif self.branch_mode == "hetero":
            h_img = self.hetero_branch(image)
            h_fused = h_img
        
        else:  # dual
            h_img = self.hetero_branch(image)
            h_zr = self.zerone_branch(zerone)
            
            if return_gate:
                h_fused, g = self.fusion(h_img, h_zr, return_gate=True)
            else:
                h_fused = self.fusion(h_img, h_zr)
        
        # 组装返回值
        if return_branches and return_gate:
            return h_fused, h_img, g
        elif return_branches:
            return h_fused, h_img
        elif return_gate:
            return h_fused, g
        return h_fused


class AnomalyModelV3Plus(nn.Module):
    """
    混合异常检测模型 V3+ (修复版)
    
    【V3+ 关键修复】
        1. VAE解耦: 仅用h_img重构图像,切断zerone负迁移
        2. 支持gate权重输出用于可视化
        3. 支持三种分支模式
    """
    
    def __init__(self, cfg: ThreeStageConfigV3Plus):
        super().__init__()
        self.cfg = cfg
        
        # 分支编码器
        self.encoder = BranchEncoder(cfg)
        
        # SVDD投影头 (使用融合特征)
        self.svdd_proj = nn.Sequential(
            nn.Linear(cfg.FUSED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, cfg.LATENT_DIM)
        )
        self.register_buffer('center', torch.zeros(cfg.LATENT_DIM))
        
        # VAE解码器 (根据配置决定是否启用)
        self.has_vae = cfg.HAS_VAE and cfg.BRANCH_MODE in ["hetero", "dual"]
        if self.has_vae:
            # VAE输入维度: 
            # - 解耦模式: 使用h_img (512维)
            # - 非解耦模式: 使用h_fused (512维)
            vae_input_dim = cfg.CNN_FEAT_DIM if cfg.VAE_DECOUPLE else cfg.FUSED_DIM
            
            self.vae_mu = nn.Linear(vae_input_dim, cfg.LATENT_CHANNELS * 7 * 7)
            self.vae_logvar = nn.Linear(vae_input_dim, cfg.LATENT_CHANNELS * 7 * 7)
            self.vae_decoder = nn.Sequential(
                nn.ConvTranspose2d(cfg.LATENT_CHANNELS, 256, 4, 2, 1),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Sigmoid()
            )
        
        self.alpha = 0.6  # SVDD权重
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor,
                return_gate: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        【V3+ VAE解耦关键】
            - VAE使用h_img重构图像
            - SVDD使用h_fused计算异常得分
            - 梯度流向解耦,zerone分支不受VAE重构损失影响
        """
        # 编码 (返回分支特征用于VAE解耦)
        if self.cfg.BRANCH_MODE == "dual" and return_gate:
            h_fused, h_img, g = self.encoder(image, zerone, return_gate=True, return_branches=True)
        elif self.cfg.BRANCH_MODE == "dual":
            h_fused, h_img = self.encoder(image, zerone, return_branches=True)
            g = None
        else:
            h_fused = self.encoder(image, zerone)
            h_img = h_fused if self.cfg.BRANCH_MODE == "hetero" else None
            g = None
        
        # SVDD (使用融合特征)
        z_svdd = self.svdd_proj(h_fused)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        result = {
            'h_fused': h_fused,
            'svdd_z': z_svdd,
            'svdd_score': svdd_score,
        }
        
        if return_gate and g is not None:
            result['gate'] = g
        
        # VAE (使用h_img重构,实现解耦)
        if self.has_vae and h_img is not None:
            # 【关键修复】VAE仅用h_img,切断zerone梯度污染
            vae_input = h_img if self.cfg.VAE_DECOUPLE else h_fused
            
            mu = self.vae_mu(vae_input)
            logvar = self.vae_logvar(vae_input)
            
            # 重塑为空间形式
            B = mu.size(0)
            mu_spatial = mu.view(B, self.cfg.LATENT_CHANNELS, 7, 7)
            logvar_spatial = logvar.view(B, self.cfg.LATENT_CHANNELS, 7, 7)
            
            z_vae = self._reparameterize(mu_spatial, logvar_spatial)
            recon = self.vae_decoder(z_vae)
            
            # 尺寸对齐
            if recon.shape[-2:] != image.shape[-2:]:
                recon = F.interpolate(recon, size=image.shape[-2:], 
                                     mode='bilinear', align_corners=False)
            
            # 计算损失
            vae_recon_loss = F.l1_loss(recon, image, reduction='none').mean(dim=[1,2,3])
            vae_kl = -0.5 * torch.mean(1 + logvar_spatial - mu_spatial.pow(2) - logvar_spatial.exp(), 
                                       dim=[1,2,3])
            
            result.update({
                'recon': recon,
                'mu': mu_spatial,
                'logvar': logvar_spatial,
                'vae_recon_loss': vae_recon_loss,
                'vae_kl': vae_kl,
            })
        
        return result
    
    def init_center(self, dataloader: DataLoader, device: torch.device, eps: float = 0.1):
        """初始化SVDD超球中心"""
        n_samples = 0
        c = torch.zeros(self.cfg.LATENT_DIM, device=device)
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                img = batch[0].to(device)
                zerone = batch[1].to(device)
                out = self.forward(img, zerone)
                z = out['svdd_z']
                c += z.sum(dim=0)
                n_samples += z.size(0)
        c /= n_samples
        
        # 避免中心在原点
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c
    
    def anomaly_score(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """计算融合异常得分"""
        out = self.forward(image, zerone)
        svdd_score = out['svdd_score']
        
        if self.has_vae and 'vae_recon_loss' in out:
            vae_score = out['vae_recon_loss'] + 0.1 * out['vae_kl']
            return self.alpha * svdd_score + (1 - self.alpha) * vae_score
        return svdd_score


class FaultClassifierV3Plus(nn.Module):
    """
    故障分类器 V3+ (修复版)
    
    【V3+ 关键修复】
        - 支持真正的解冻生效
        - 提供参数分组功能用于rebuild optimizer
    """
    
    def __init__(self, pretrained_encoder: BranchEncoder, 
                 cfg: ThreeStageConfigV3Plus, num_classes: int = 2):
        super().__init__()
        self.encoder = pretrained_encoder
        self.cfg = cfg
        self.num_classes = num_classes
        
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(cfg.FUSED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._encoder_unfrozen = False
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor,
                return_gate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_fused = self.encoder(image, zerone)
        logits = self.classifier(h_fused)
        
        if return_gate and self.cfg.BRANCH_MODE == "dual":
            _, g = self.encoder(image, zerone, return_gate=True)
            return logits, g
        return logits
    
    def unfreeze_encoder(self, num_blocks: int = 1) -> List[nn.Parameter]:
        """
        渐进解冻编码器
        
        【返回】
            新解冻的参数列表 (用于rebuild optimizer)
        """
        params = list(self.encoder.parameters())
        n_unfreeze = min(num_blocks * 20, len(params))
        
        newly_unfrozen = []
        for p in params[-n_unfreeze:]:
            if not p.requires_grad:
                p.requires_grad = True
                newly_unfrozen.append(p)
        
        self._encoder_unfrozen = True
        print(f"  [解冻] 编码器最后 {n_unfreeze} 个参数, 新解冻 {len(newly_unfrozen)} 个")
        return newly_unfrozen
    
    def get_param_groups(self, head_lr: float, encoder_lr: float) -> List[Dict]:
        """
        获取参数分组 (用于rebuild optimizer)
        
        【返回】
            [{'params': head_params, 'lr': head_lr},
             {'params': encoder_params, 'lr': encoder_lr}]
        """
        head_params = list(self.classifier.parameters())
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        
        groups = [{'params': head_params, 'lr': head_lr, 'name': 'head'}]
        if encoder_params:
            groups.append({'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'})
        
        return groups


# =============================================================================
# 第7步: 训练函数
# =============================================================================
def train_stage1(cfg: ThreeStageConfigV3Plus) -> Tuple[AnomalyModelV3Plus, Dict]:
    """
    阶段一: 纯无监督训练
    
    【V3+ 改进】
        - 支持三分支模式
        - Gate预热机制
        - 记录gate统计信息
    """
    print("\n" + "="*70)
    print(f"【阶段一】纯无监督异常检测模型训练 (模式: {cfg.BRANCH_MODE})")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 加载数据
    print("\n[1/5] 加载训练数据...")
    train_dataset = DualBranchDataset(cfg.TRAIN_DIR, cfg, use_labels=False)
    val_dataset = DualBranchDataset(cfg.VAL_DIR, cfg, use_labels=False)
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"  合并后样本数: {len(combined_dataset)}")
    
    if len(combined_dataset) == 0:
        raise ValueError("没有加载到数据!")
    
    dataloader = DataLoader(combined_dataset, batch_size=cfg.BATCH_SIZE, 
                           shuffle=True, num_workers=0, drop_last=True)
    
    # 初始化模型
    print("\n[2/5] 初始化模型...")
    model = AnomalyModelV3Plus(cfg).to(device)
    
    # 预训练VAE (如果启用)
    if model.has_vae:
        print("\n[3/5] 预训练VAE编码器...")
        vae_params = list(model.vae_mu.parameters()) + \
                     list(model.vae_logvar.parameters()) + \
                     list(model.vae_decoder.parameters())
        if cfg.BRANCH_MODE in ["hetero", "dual"]:
            vae_params += list(model.encoder.hetero_branch.parameters())
        
        vae_opt = torch.optim.Adam(vae_params, lr=cfg.LR * 10)
        
        for epoch in range(10):
            model.train()
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"VAE预训练 {epoch+1}/10", leave=False):
                img = batch[0].to(device)
                zerone = batch[1].to(device)
                out = model(img, zerone)
                
                if 'vae_recon_loss' in out:
                    loss = out['vae_recon_loss'].mean() + 0.01 * out['vae_kl'].mean()
                    vae_opt.zero_grad()
                    loss.backward()
                    vae_opt.step()
                    total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/10 | VAE Loss: {total_loss/len(dataloader):.4f}")
    else:
        print("\n[3/5] VAE未启用,跳过预训练")
    
    # 初始化SVDD中心
    print("\n[4/5] 初始化SVDD超球中心...")
    model.init_center(dataloader, device)
    print(f"  中心向量范数: {model.center.norm().item():.4f}")
    
    # 联合训练
    print(f"\n[5/5] 联合训练 ({cfg.STAGE1_EPOCHS}轮)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.STAGE1_EPOCHS)
    
    history = {
        'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': [],
        'gate_mean': [], 'gate_std': []  # 新增: gate统计
    }
    best_loss = float('inf')
    
    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        
        # Beta预热
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch + 1) / max(cfg.BETA_WARMUP, 1))
        
        # Gate预热: 前GATE_WARMUP_EPOCHS轮固定gate (通过detach实现)
        gate_warmup = epoch < cfg.GATE_WARMUP_EPOCHS
        
        epoch_svdd, epoch_vae, epoch_total = 0, 0, 0
        gate_values = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False):
            img = batch[0].to(device)
            zerone = batch[1].to(device)
            
            # 前向传播 (返回gate用于统计)
            out = model(img, zerone, return_gate=True)
            
            # SVDD损失
            svdd_loss = out['svdd_score'].mean()
            
            # VAE损失 (如果启用)
            if 'vae_recon_loss' in out:
                vae_loss = out['vae_recon_loss'].mean() + beta * out['vae_kl'].mean()
            else:
                vae_loss = torch.tensor(0.0, device=device)
            
            # Gate正则 (如果是dual模式且非预热期)
            gate_reg = torch.tensor(0.0, device=device)
            if 'gate' in out and not gate_warmup:
                g = out['gate']
                # 熵正则: 鼓励g分布不要太极端
                entropy = -(g * torch.log(g + 1e-8) + (1-g) * torch.log(1-g + 1e-8)).mean()
                gate_reg = -cfg.GATE_REG_LAMBDA * entropy  # 最大化熵
                gate_values.append(g.detach().cpu().mean().item())
            
            # 总损失
            loss = svdd_loss + vae_loss + gate_reg
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_svdd += svdd_loss.item()
            epoch_vae += vae_loss.item()
            epoch_total += loss.item()
        
        scheduler.step()
        
        # 记录
        n_batch = len(dataloader)
        history['epoch'].append(epoch + 1)
        history['svdd_loss'].append(epoch_svdd / n_batch)
        history['vae_loss'].append(epoch_vae / n_batch)
        history['total_loss'].append(epoch_total / n_batch)
        
        if gate_values:
            history['gate_mean'].append(np.mean(gate_values))
            history['gate_std'].append(np.std(gate_values))
        else:
            history['gate_mean'].append(0.5)
            history['gate_std'].append(0.0)
        
        # 保存最佳
        if epoch_total < best_loss:
            best_loss = epoch_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.center,
                'epoch': epoch,
                'loss': best_loss
            }, cfg.MODEL_DIR / "stage1_best.pth")
        
        if (epoch + 1) % 5 == 0:
            gate_info = f" | Gate: {history['gate_mean'][-1]:.3f}±{history['gate_std'][-1]:.3f}" \
                       if cfg.BRANCH_MODE == "dual" else ""
            print(f"  [Epoch {epoch+1}] SVDD: {epoch_svdd/n_batch:.4f} | "
                  f"VAE: {epoch_vae/n_batch:.4f} | Total: {epoch_total/n_batch:.4f}{gate_info}")
    
    # 可视化
    plot_training_curves(history, cfg, "stage1")
    
    print(f"\n【阶段一完成】")
    print(f"  最佳损失: {best_loss/len(dataloader):.4f}")
    print(f"  模型保存: {cfg.MODEL_DIR / 'stage1_best.pth'}")
    
    return model, history


def run_stage2(model: AnomalyModelV3Plus, cfg: ThreeStageConfigV3Plus) -> Dict:
    """阶段二: 伪标签生成"""
    print("\n" + "="*70)
    print("【阶段二】伪标签生成")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    model.eval()
    
    # 加载所有数据计算得分
    print("\n[1/3] 计算异常得分...")
    all_dataset = DualBranchDataset(cfg.TRAIN_DIR, cfg, use_labels=False)
    val_dataset = DualBranchDataset(cfg.VAL_DIR, cfg, use_labels=False)
    combined = ConcatDataset([all_dataset, val_dataset])
    
    dataloader = DataLoader(combined, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    scores = []
    indices = []
    gate_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算得分", leave=False):
            img = batch[0].to(device)
            zerone = batch[1].to(device)
            idx = batch[3]
            
            out = model(img, zerone, return_gate=True)
            score = model.anomaly_score(img, zerone)
            
            scores.extend(score.cpu().numpy().tolist())
            indices.extend(idx.numpy().tolist())
            
            if 'gate' in out:
                gate_values.extend(out['gate'].mean(dim=1).cpu().numpy().tolist())
    
    scores = np.array(scores)
    
    # 计算阈值
    print("\n[2/3] 确定伪标签阈值...")
    t_normal = np.percentile(scores, cfg.NORMAL_PERCENTILE)
    t_anomaly = np.percentile(scores, cfg.ANOMALY_PERCENTILE)
    
    print(f"  正常阈值 (P{cfg.NORMAL_PERCENTILE}): {t_normal:.4f}")
    print(f"  异常阈值 (P{cfg.ANOMALY_PERCENTILE}): {t_anomaly:.4f}")
    
    # 分配伪标签
    pseudo_normal = np.where(scores <= t_normal)[0]
    pseudo_anomaly = np.where(scores >= t_anomaly)[0]
    pseudo_uncertain = np.where((scores > t_normal) & (scores < t_anomaly))[0]
    
    print(f"\n[3/3] 伪标签统计:")
    print(f"  高置信正常: {len(pseudo_normal)} ({100*len(pseudo_normal)/len(scores):.1f}%)")
    print(f"  高置信异常: {len(pseudo_anomaly)} ({100*len(pseudo_anomaly)/len(scores):.1f}%)")
    print(f"  不确定: {len(pseudo_uncertain)} ({100*len(pseudo_uncertain)/len(scores):.1f}%)")
    
    # 保存结果
    result = {
        'scores': scores,
        'indices': indices,
        'pseudo_normal': pseudo_normal,
        'pseudo_anomaly': pseudo_anomaly,
        'pseudo_uncertain': pseudo_uncertain,
        't_normal': t_normal,
        't_anomaly': t_anomaly,
        'gate_values': np.array(gate_values) if gate_values else None
    }
    
    np.savez(cfg.STAGE2_DIR / "pseudo_labels.npz", **{k: v for k, v in result.items() if v is not None})
    
    # 可视化
    plot_score_distribution(scores, t_normal, t_anomaly, cfg)
    
    if cfg.BRANCH_MODE == "dual" and gate_values:
        plot_gate_distribution(np.array(gate_values), cfg, split='all')
    
    print(f"\n【阶段二完成】")
    print(f"  结果保存: {cfg.STAGE2_DIR / 'pseudo_labels.npz'}")
    
    return result


def train_stage3(model: AnomalyModelV3Plus, pseudo_labels: Dict, 
                 cfg: ThreeStageConfigV3Plus) -> FaultClassifierV3Plus:
    """
    阶段三: 有监督微调
    
    【V3+ 关键修复】
        1. 解冻后重建optimizer,确保梯度更新生效
        2. 记录解冻前后的梯度范数用于自检
    """
    print("\n" + "="*70)
    print(f"【阶段三】有监督分类器训练 (模式: {cfg.BRANCH_MODE})")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 加载测试集
    print("\n[1/4] 加载测试数据...")
    test_dataset = DualBranchDataset(cfg.TEST_DIR, cfg, use_labels=True)
    
    if len(test_dataset) == 0:
        print("[警告] 测试集为空")
        return None
    
    # 统计类别
    labels = [test_dataset.samples[i][3] for i in range(len(test_dataset))]
    label_counts = Counter([l for l in labels if l is not None])
    print(f"  测试集样本数: {len(test_dataset)}")
    for lbl, cnt in label_counts.items():
        print(f"    类别 {lbl}: {cnt}")
    
    # 划分训练/验证
    n_total = len(test_dataset)
    n_train = int(n_total * 0.8)
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))
    
    train_subset = torch.utils.data.Subset(test_dataset, train_indices)
    val_subset = torch.utils.data.Subset(test_dataset, val_indices)
    
    # 类别平衡采样
    train_labels = [labels[i] for i in train_indices if labels[i] is not None]
    if train_labels:
        class_counts = np.bincount(train_labels)
        weights = 1.0 / (class_counts[train_labels] + 1e-6)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 构建分类器
    print("\n[2/4] 构建分类器...")
    classifier = FaultClassifierV3Plus(model.encoder, cfg, num_classes=2).to(device)
    
    # 初始optimizer (仅分类头)
    head_lr = 1e-3
    optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=head_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练记录
    history = {
        'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': [],
        'encoder_grad_norm': [], 'encoder_weight_diff': [],  # 新增: 解冻自检
        'gate_mean': [], 'gate_std': []  # 新增: gate统计
    }
    best_f1 = 0
    
    # 用于计算权重差异
    prev_encoder_weights = None
    
    # 解冻自检日志
    unfreeze_log = []
    
    print(f"\n[3/4] 开始训练 ({cfg.STAGE3_EPOCHS}轮)...")
    
    for epoch in range(cfg.STAGE3_EPOCHS):
        # === 训练 ===
        classifier.train()
        train_loss = 0
        gate_values = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}", leave=False):
            img = batch[0].to(device)
            zerone = batch[1].to(device)
            label = batch[2].to(device)
            
            # 跳过无效标签
            valid_mask = label >= 0
            if not valid_mask.any():
                continue
            
            if cfg.BRANCH_MODE == "dual":
                logits, g = classifier(img, zerone, return_gate=True)
                gate_values.extend(g[valid_mask].mean(dim=1).detach().cpu().numpy().tolist())
            else:
                logits = classifier(img, zerone)
            
            loss = criterion(logits[valid_mask], label[valid_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # === 验证 ===
        classifier.eval()
        val_preds, val_labels_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                img = batch[0].to(device)
                zerone = batch[1].to(device)
                label = batch[2]
                
                logits = classifier(img, zerone)
                pred = logits.argmax(dim=1)
                
                valid_mask = label >= 0
                val_preds.extend(pred[valid_mask].cpu().numpy().tolist())
                val_labels_list.extend(label[valid_mask].numpy().tolist())
        
        if val_preds:
            val_acc = accuracy_score(val_labels_list, val_preds)
            val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        else:
            val_acc, val_f1 = 0, 0
        
        scheduler.step(val_f1)
        
        # === 计算解冻自检指标 ===
        encoder_grad_norm = 0.0
        encoder_weight_diff = 0.0
        
        # 收集encoder梯度范数
        for p in classifier.encoder.parameters():
            if p.grad is not None:
                encoder_grad_norm += p.grad.norm().item() ** 2
        encoder_grad_norm = np.sqrt(encoder_grad_norm)
        
        # 计算权重变化
        current_weights = torch.cat([p.flatten() for p in classifier.encoder.parameters()]).detach().cpu()
        if prev_encoder_weights is not None:
            encoder_weight_diff = (current_weights - prev_encoder_weights).norm().item()
        prev_encoder_weights = current_weights.clone()
        
        # === 记录 ===
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / max(len(train_loader), 1))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['encoder_grad_norm'].append(encoder_grad_norm)
        history['encoder_weight_diff'].append(encoder_weight_diff)
        
        if gate_values:
            history['gate_mean'].append(np.mean(gate_values))
            history['gate_std'].append(np.std(gate_values))
        else:
            history['gate_mean'].append(0.5)
            history['gate_std'].append(0.0)
        
        # === 保存最佳 ===
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state': classifier.state_dict(),
                'epoch': epoch,
                'f1': best_f1
            }, cfg.MODEL_DIR / "stage3_best.pth")
        
        # === 解冻逻辑 (V3+ 关键修复) ===
        if epoch == cfg.UNFREEZE_EPOCH:
            print(f"\n  [Epoch {epoch+1}] 触发解冻...")
            
            # 记录解冻前状态
            pre_unfreeze_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
            
            # 解冻
            newly_unfrozen = classifier.unfreeze_encoder(num_blocks=1)
            
            # 【关键修复】重建optimizer
            if cfg.UNFREEZE_REBUILD_OPTIMIZER:
                encoder_lr = head_lr * cfg.UNFREEZE_LR_FACTOR
                param_groups = classifier.get_param_groups(head_lr, encoder_lr)
                optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
                print(f"  [修复] 重建optimizer, encoder_lr={encoder_lr:.6f}")
            else:
                # 备选: add_param_group
                if newly_unfrozen:
                    encoder_lr = head_lr * cfg.UNFREEZE_LR_FACTOR
                    optimizer.add_param_group({'params': newly_unfrozen, 'lr': encoder_lr})
                    print(f"  [修复] 添加param_group, encoder_lr={encoder_lr:.6f}")
            
            # 记录解冻后状态
            post_unfreeze_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
            
            # 自检日志
            unfreeze_log.append({
                'epoch': epoch + 1,
                'pre_params': pre_unfreeze_params,
                'post_params': post_unfreeze_params,
                'newly_unfrozen': len(newly_unfrozen),
                'optimizer_groups': len(optimizer.param_groups)
            })
            
            print(f"  [自检] 可训练参数: {pre_unfreeze_params} -> {post_unfreeze_params}")
            print(f"  [自检] optimizer组数: {len(optimizer.param_groups)}")
        
        # === 打印进度 ===
        if (epoch + 1) % 5 == 0:
            gate_info = f" | Gate: {history['gate_mean'][-1]:.3f}" if cfg.BRANCH_MODE == "dual" else ""
            grad_info = f" | EncGrad: {encoder_grad_norm:.4f}" if encoder_grad_norm > 0 else ""
            print(f"  [Epoch {epoch+1}] Loss: {train_loss/max(len(train_loader),1):.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}{gate_info}{grad_info}")
    
    # === 保存解冻自检日志 ===
    if unfreeze_log:
        with open(cfg.VIZ_DIR / "unfreeze_check" / "unfreeze_log.json", 'w', encoding='utf-8') as f:
            json.dump(unfreeze_log, f, indent=2, ensure_ascii=False)
    
    # === 最终评估 ===
    print("\n[4/4] 最终评估...")
    ckpt = torch.load(cfg.MODEL_DIR / "stage3_best.pth")
    classifier.load_state_dict(ckpt['model_state'])
    
    all_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    classifier.eval()
    all_preds, all_labels_list, all_probs = [], [], []
    all_gates = []
    
    with torch.no_grad():
        for batch in all_loader:
            img = batch[0].to(device)
            zerone = batch[1].to(device)
            label = batch[2]
            
            if cfg.BRANCH_MODE == "dual":
                logits, g = classifier(img, zerone, return_gate=True)
                all_gates.extend(g.mean(dim=1).cpu().numpy().tolist())
            else:
                logits = classifier(img, zerone)
            
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            valid_mask = label >= 0
            all_preds.extend(pred[valid_mask].cpu().numpy().tolist())
            all_labels_list.extend(label[valid_mask].numpy().tolist())
            all_probs.extend(prob[valid_mask].cpu().numpy().tolist())
    
    if all_preds:
        acc = accuracy_score(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels_list, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels_list, all_preds, average='macro', zero_division=0)
        
        print(f"\n【最终评估结果】")
        print(f"  准确率: {acc:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
    
    # === 可视化 ===
    plot_training_curves(history, cfg, "stage3")
    if all_preds and all_labels_list:
        plot_confusion_matrix(all_labels_list, all_preds, cfg)
        if all_probs:
            plot_roc_pr_curves(all_labels_list, all_probs, cfg)
    
    # 解冻自检图
    plot_unfreeze_check(history, cfg)
    
    # Gate分析 (dual模式)
    if cfg.BRANCH_MODE == "dual" and all_gates:
        plot_gate_distribution(np.array(all_gates), cfg, split='test')
        if all_preds and all_labels_list:
            plot_gate_by_class(np.array(all_gates), np.array(all_labels_list), cfg)
            # 正确/错误分样本gate对比
            correct_mask = np.array(all_preds) == np.array(all_labels_list)
            plot_gate_correct_vs_error(np.array(all_gates), correct_mask, cfg)
    
    print(f"\n【阶段三完成】")
    print(f"  最佳F1: {best_f1:.4f}")
    print(f"  模型保存: {cfg.MODEL_DIR / 'stage3_best.pth'}")
    
    return classifier


# =============================================================================
# 第8步: 可视化函数
# =============================================================================
def plot_training_curves(history: Dict, cfg: ThreeStageConfigV3Plus, stage: str):
    """绘制训练曲线"""
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    if stage == "stage1":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # SVDD + VAE损失
        ax = axes[0]
        ax.plot(history['epoch'], history['svdd_loss'], 'b-', lw=1.5, label='SVDD')
        ax.plot(history['epoch'], history['vae_loss'], 'r--', lw=1.5, label='VAE')
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['loss'])
        ax.set_title('SVDD + VAE 损失' if lang == 'cn' else 'SVDD + VAE Loss')
        ax.legend()
        ax.grid(True, ls=':', alpha=0.5)
        
        # 总损失
        ax = axes[1]
        ax.plot(history['epoch'], history['total_loss'], 'g-', lw=1.5)
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['loss'])
        ax.set_title('总损失' if lang == 'cn' else 'Total Loss')
        ax.grid(True, ls=':', alpha=0.5)
        
        # Gate统计 (如果有)
        ax = axes[2]
        if history.get('gate_mean') and any(g != 0.5 for g in history['gate_mean']):
            ax.fill_between(history['epoch'], 
                           np.array(history['gate_mean']) - np.array(history['gate_std']),
                           np.array(history['gate_mean']) + np.array(history['gate_std']),
                           alpha=0.3, color='purple')
            ax.plot(history['epoch'], history['gate_mean'], 'purple', lw=1.5)
            ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
            ax.set_ylim(0, 1)
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['gate_weight'])
        ax.set_title('Gate权重变化' if lang == 'cn' else 'Gate Weight')
        ax.grid(True, ls=':', alpha=0.5)
        
    else:  # stage3
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # 训练损失
        ax = axes[0]
        ax.plot(history['epoch'], history['train_loss'], 'b-', lw=1.5)
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['loss'])
        ax.set_title('训练损失' if lang == 'cn' else 'Training Loss')
        ax.grid(True, ls=':', alpha=0.5)
        
        # 验证指标
        ax = axes[1]
        ax.plot(history['epoch'], history['val_acc'], 'b-', lw=1.5, label=L['accuracy'])
        ax.plot(history['epoch'], history['val_f1'], 'r--', lw=1.5, label=L['f1'])
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel('Score')
        ax.set_title('验证性能' if lang == 'cn' else 'Validation Performance')
        ax.legend()
        ax.grid(True, ls=':', alpha=0.5)
        
        # Gate统计
        ax = axes[2]
        if history.get('gate_mean') and any(g != 0.5 for g in history['gate_mean']):
            ax.plot(history['epoch'], history['gate_mean'], 'purple', lw=1.5)
            ax.set_ylim(0, 1)
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['gate_weight'])
        ax.set_title('Gate权重变化' if lang == 'cn' else 'Gate Weight')
        ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "training_curves" / f"{stage}_curves.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_score_distribution(scores: np.ndarray, t_normal: float, t_anomaly: float, 
                           cfg: ThreeStageConfigV3Plus):
    """绘制异常得分分布"""
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(scores, bins=50, density=True, alpha=0.7, color=COLORS['blue'], edgecolor='white')
    ax.axvline(t_normal, color=COLORS['green'], ls='--', lw=2, label=f'{L["normal"]}阈值: {t_normal:.3f}')
    ax.axvline(t_anomaly, color=COLORS['red'], ls='--', lw=2, label=f'{L["fault"]}阈值: {t_anomaly:.3f}')
    
    ax.set_xlabel(L['score'])
    ax.set_ylabel('密度' if lang == 'cn' else 'Density')
    ax.set_title('异常得分分布' if lang == 'cn' else 'Anomaly Score Distribution')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "score_dist" / "score_distribution.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: List, y_pred: List, cfg: ThreeStageConfigV3Plus):
    """绘制混淆矩阵"""
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = [L['normal'], L['fault']]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel=L['pred'], ylabel=L['true'],
           title='混淆矩阵' if lang == 'cn' else 'Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "confusion" / "confusion_matrix.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(y_true: List, y_probs: List, cfg: ThreeStageConfigV3Plus):
    """绘制ROC和PR曲线"""
    setup_plotting()
    lang = cfg.VIZ_LANG
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    if y_probs.ndim == 2:
        y_score = y_probs[:, 1]
    else:
        y_score = y_probs
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS['blue'], lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', ls='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('FPR' if lang == 'en' else '假阳性率')
    ax.set_ylabel('TPR' if lang == 'en' else '真阳性率')
    ax.set_title('ROC Curve' if lang == 'en' else 'ROC曲线')
    ax.legend(loc='lower right')
    ax.grid(True, ls=':', alpha=0.5)
    
    # PR
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    ax = axes[1]
    ax.plot(recall_arr, precision_arr, color=COLORS['red'], lw=2, label=f'AP = {ap:.3f}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall' if lang == 'en' else '召回率')
    ax.set_ylabel('Precision' if lang == 'en' else '精确率')
    ax.set_title('PR Curve' if lang == 'en' else 'PR曲线')
    ax.legend(loc='lower left')
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "roc_pr" / "roc_pr_curves.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


# === V3+ 新增可视化 ===

def plot_gate_distribution(gate_values: np.ndarray, cfg: ThreeStageConfigV3Plus, split: str = 'all'):
    """
    绘制Gate权重分布
    
    【新增可视化 - V3+】
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图
    ax = axes[0]
    ax.hist(gate_values, bins=50, density=True, alpha=0.7, color=COLORS['purple'], edgecolor='white')
    ax.axvline(0.5, color='gray', ls='--', lw=1.5, label='均匀融合')
    ax.axvline(gate_values.mean(), color=COLORS['red'], ls='-', lw=1.5, 
              label=f'均值: {gate_values.mean():.3f}')
    ax.set_xlabel('Gate权重' if lang == 'cn' else 'Gate Weight')
    ax.set_ylabel('密度' if lang == 'cn' else 'Density')
    ax.set_title(f'Gate分布 ({split})' if lang == 'cn' else f'Gate Distribution ({split})')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    # 箱线图
    ax = axes[1]
    bp = ax.boxplot(gate_values, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['purple'])
    bp['boxes'][0].set_alpha(0.5)
    ax.axhline(0.5, color='gray', ls='--', lw=1.5)
    ax.set_ylabel('Gate权重' if lang == 'cn' else 'Gate Weight')
    ax.set_title(f'Gate箱线图 ({split})' if lang == 'cn' else f'Gate Boxplot ({split})')
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "gate_analysis" / f"gate_distribution_{split}.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_gate_by_class(gate_values: np.ndarray, labels: np.ndarray, cfg: ThreeStageConfigV3Plus):
    """
    绘制不同类别的Gate分布对比
    
    【新增可视化 - V3+】
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    normal_gates = gate_values[labels == 0]
    fault_gates = gate_values[labels == 1]
    
    ax.hist(normal_gates, bins=30, density=True, alpha=0.6, color=COLORS['green'], 
           label=f'{L["normal"]} (n={len(normal_gates)})')
    ax.hist(fault_gates, bins=30, density=True, alpha=0.6, color=COLORS['red'],
           label=f'{L["fault"]} (n={len(fault_gates)})')
    
    ax.axvline(normal_gates.mean(), color=COLORS['green'], ls='--', lw=2)
    ax.axvline(fault_gates.mean(), color=COLORS['red'], ls='--', lw=2)
    
    ax.set_xlabel('Gate权重' if lang == 'cn' else 'Gate Weight')
    ax.set_ylabel('密度' if lang == 'cn' else 'Density')
    ax.set_title('正常vs故障 Gate分布' if lang == 'cn' else 'Normal vs Fault Gate Distribution')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    # 计算统计显著性
    if len(normal_gates) > 5 and len(fault_gates) > 5:
        corr, p_value = spearmanr(
            np.concatenate([np.zeros(len(normal_gates)), np.ones(len(fault_gates))]),
            np.concatenate([normal_gates, fault_gates])
        )
        ax.text(0.02, 0.98, f'Spearman r={corr:.3f}, p={p_value:.4f}',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "gate_analysis" / "gate_by_class.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_gate_correct_vs_error(gate_values: np.ndarray, correct_mask: np.ndarray, 
                               cfg: ThreeStageConfigV3Plus):
    """
    绘制正确vs错误分类样本的Gate对比
    
    【新增可视化 - V3+】
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    correct_gates = gate_values[correct_mask]
    error_gates = gate_values[~correct_mask]
    
    if len(correct_gates) > 0:
        ax.hist(correct_gates, bins=30, density=True, alpha=0.6, color=COLORS['green'],
               label=f'正确 (n={len(correct_gates)})' if lang == 'cn' else f'Correct (n={len(correct_gates)})')
    if len(error_gates) > 0:
        ax.hist(error_gates, bins=30, density=True, alpha=0.6, color=COLORS['red'],
               label=f'错误 (n={len(error_gates)})' if lang == 'cn' else f'Error (n={len(error_gates)})')
    
    ax.set_xlabel('Gate权重' if lang == 'cn' else 'Gate Weight')
    ax.set_ylabel('密度' if lang == 'cn' else 'Density')
    ax.set_title('正确vs错误分类 Gate对比' if lang == 'cn' else 'Correct vs Error Gate Comparison')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "gate_analysis" / "gate_correct_vs_error.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_unfreeze_check(history: Dict, cfg: ThreeStageConfigV3Plus):
    """
    绘制解冻自检图
    
    【新增可视化 - V3+】
        - encoder梯度范数变化
        - encoder权重更新量变化
        - 标注解冻时间点
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = history['epoch']
    unfreeze_epoch = cfg.UNFREEZE_EPOCH + 1  # 1-indexed
    
    # 梯度范数
    ax = axes[0]
    ax.plot(epochs, history['encoder_grad_norm'], 'b-', lw=1.5)
    ax.axvline(unfreeze_epoch, color='red', ls='--', lw=1.5, label=f'解冻 @ {unfreeze_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('梯度范数' if lang == 'cn' else 'Gradient Norm')
    ax.set_title('Encoder梯度范数' if lang == 'cn' else 'Encoder Gradient Norm')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    # 权重更新量
    ax = axes[1]
    ax.plot(epochs, history['encoder_weight_diff'], 'g-', lw=1.5)
    ax.axvline(unfreeze_epoch, color='red', ls='--', lw=1.5, label=f'解冻 @ {unfreeze_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('权重更新量' if lang == 'cn' else 'Weight Update')
    ax.set_title('Encoder权重更新量' if lang == 'cn' else 'Encoder Weight Update')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "unfreeze_check" / "unfreeze_check.png", 
               dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# 第9步: 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='三阶段故障诊断系统 V3+')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--test_data', action='store_true', help='仅测试数据加载')
    parser.add_argument('--branch', type=str, default='dual', 
                       choices=['zerone', 'hetero', 'dual'], help='分支模式')
    parser.add_argument('--fusion', type=str, default='residual_gate',
                       choices=['concat', 'gate', 'residual_gate'], help='融合方式')
    parser.add_argument('--no_vae_decouple', action='store_true', help='关闭VAE解耦')
    parser.add_argument('--data_root', type=str, default=None, help='数据根目录')
    parser.add_argument('--output_root', type=str, default=None, help='输出根目录')
    
    args = parser.parse_args()
    
    # 配置
    cfg = ThreeStageConfigV3Plus()
    cfg.BRANCH_MODE = args.branch
    cfg.FUSION_MODE = args.fusion
    cfg.VAE_DECOUPLE = not args.no_vae_decouple
    
    if args.data_root:
        cfg.PROJECT_ROOT = Path(args.data_root)
    if args.output_root:
        cfg.OUTPUT_ROOT = Path(args.output_root)
    
    # 重新初始化路径
    cfg.__post_init__()
    cfg.print_config()
    
    # 测试数据加载
    if args.test_data:
        print("\n=== 测试数据加载 ===")
        for name, path in [('train', cfg.TRAIN_DIR), ('val', cfg.VAL_DIR), ('test', cfg.TEST_DIR)]:
            if path.exists():
                ds = DualBranchDataset(path, cfg, use_labels=(name=='test'))
                if len(ds) > 0:
                    sample = ds[0]
                    print(f"  {name}: {len(ds)} 样本, img={sample[0].shape}, zerone={sample[1].shape}")
        return
    
    # 运行阶段
    if args.all:
        model, _ = train_stage1(cfg)
        pseudo_labels = run_stage2(model, cfg)
        train_stage3(model, pseudo_labels, cfg)
    elif args.stage == 1:
        train_stage1(cfg)
    elif args.stage == 2:
        # 加载Stage1模型
        ckpt_path = cfg.MODEL_DIR / "stage1_best.pth"
        if not ckpt_path.exists():
            print(f"[错误] 找不到Stage1模型: {ckpt_path}")
            return
        model = AnomalyModelV3Plus(cfg)
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.center = ckpt['center']
        model.to(cfg.DEVICE)
        run_stage2(model, cfg)
    elif args.stage == 3:
        # 加载模型和伪标签
        ckpt_path = cfg.MODEL_DIR / "stage1_best.pth"
        pseudo_path = cfg.STAGE2_DIR / "pseudo_labels.npz"
        if not ckpt_path.exists() or not pseudo_path.exists():
            print("[错误] 找不到Stage1模型或Stage2伪标签")
            return
        model = AnomalyModelV3Plus(cfg)
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.center = ckpt['center']
        model.to(cfg.DEVICE)
        pseudo_labels = dict(np.load(pseudo_path, allow_pickle=True))
        train_stage3(model, pseudo_labels, cfg)
    else:
        print("请指定 --stage 或 --all")
        parser.print_help()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
transformer_three_stage_v5.py
==============================

交流变压器振动数据 三阶段渐进式故障诊断系统 V5
支持三条并行支线：Hetero-Only / Zerone-Only / Dual-Branch (融合)

【V5版本核心改进 - 大样本优化 + 融合增强】

    ✅ 1. Zerone分支重构：
       - 1200维特征 → 2D特征图 → ResNet18
       - 与Hetero分支架构对称
       
    ✅ 2. 模态Dropout：防止贪婪学习
       - 训练时随机丢弃整个模态分支
       - 可学习"缺失模态"占位符
       
    ✅ 3. 多种融合策略对比：
       - concat: 等权拼接 (baseline)
       - attention: 注意力加权
       - gate: 交叉门控
       - gmu: 门控多模态单元 (推荐)
       
    ✅ 4. 跨设备域适应：
       - MMD: 最大均值差异
       - CORAL: 协方差对齐
       - DANN: 域对抗网络 (大样本推荐)
       
    ✅ 5. 大样本优化配置：
       - 适度Dropout (0.2-0.3)
       - BatchNorm (大batch更稳定)
       - 分阶段解冻策略

【架构设计 V5】

    ┌──────────────────────────────────────────────────────────────┐
    │                        振动信号输入                           │
    │                      (8192点 @ 8192Hz)                       │
    └──────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐
    │  Hetero 图像分支          │    │  Zerone 特征分支 (V5重构) │
    │  ────────────────────    │    │  ────────────────────    │
    │  3×224×224 时频图像      │    │  1200维 → 3×20×20特征图  │
    │         ↓                │    │         ↓                │
    │    ResNet18编码器        │    │    ResNet18编码器        │
    │         ↓                │    │         ↓                │
    │      512维特征           │    │      512维特征           │
    └──────────────────────────┘    └──────────────────────────┘
            │                               │
            └───────────┬───────────────────┘
                        ▼
    ┌────────────────────────────────────────────────────────────┐
    │    ModalityDropout (V5新增)                                  │
    │    训练时随机丢弃某模态，防止贪婪学习                          │
    └────────────────────────────────────────────────────────────┘
                        ▼
    ┌────────────────────────────────────────────────────────────┐
    │    Fusion Module                                            │
    │    • concat / attention / gate / gmu                        │
    └────────────────────────────────────────────────────────────┘
                        ▼
    ┌────────────────────────────────────────────────────────────┐
    │    Domain Adaptation (跨设备)                                │
    │    • MMD + CORAL + DANN(可选)                               │
    └────────────────────────────────────────────────────────────┘

【运行方式】
    # V5推荐配置 (GMU融合 + 域适应)
    python transformer_three_stage_v5.py --branch dual --fusion_mode gmu --all
    
    # 对比实验
    python transformer_three_stage_v5.py --branch dual --fusion_mode attention --all
    python transformer_three_stage_v5.py --branch dual --fusion_mode gate --all
    
    # 消融实验
    python transformer_three_stage_v5.py --branch dual --fusion_mode gmu --no_modality_dropout --all
    python transformer_three_stage_v5.py --branch dual --fusion_mode gmu --no_domain_adapt --all

Author: V5版本 - 大样本优化 + 融合增强
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
import shutil
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import models
from tqdm import tqdm

# 信号处理库
import pywt
import cv2
from scipy import signal as sig
from scipy.stats import skew, kurtosis

# 机器学习评估
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.manifold import TSNE

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# =============================================================================
# 第1步: 全局常量与特征配置
# =============================================================================
TIME_DOMAIN_DIM = 15
STFT_BAND_DIM = 127  
PSD_BAND_DIM = 1050
HIGH_FREQ_DIM = 8
TOTAL_FEAT_DIM = TIME_DOMAIN_DIM + STFT_BAND_DIM + PSD_BAND_DIM + HIGH_FREQ_DIM  # 1200

# V5: Zerone特征图尺寸 (1200 → 3×20×20 = 1200)
ZERONE_IMG_CHANNELS = 3
ZERONE_IMG_SIZE = 20

COLORS = {
    'normal': '#2ecc71',
    'fault': '#e74c3c', 
    'uncertain': '#f39c12',
    'primary': '#3498db',
    'secondary': '#9b59b6',
}

LABELS = {
    'cn': {
        'normal': '正常', 'fault': '故障', 'uncertain': '不确定',
        'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 'f1': 'F1分数',
        'epoch': '轮次', 'loss': '损失', 'train': '训练', 'val': '验证', 'test': '测试',
    },
    'en': {
        'normal': 'Normal', 'fault': 'Fault', 'uncertain': 'Uncertain',
        'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1 Score',
        'epoch': 'Epoch', 'loss': 'Loss', 'train': 'Train', 'val': 'Validation', 'test': 'Test',
    }
}


# =============================================================================
# 第2步: 配置类定义 (V5大样本优化版)
# =============================================================================
@dataclass
class ThreeStageConfigV5:
    """
    三阶段诊断系统配置类 (V5版本 - 大样本优化)
    """
    
    # ================= 路径配置 =================
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v5"))
    
    # ================= 支线模式 =================
    BRANCH_MODE: str = "dual"  # 'hetero' / 'zerone' / 'dual'
    
    # ================= 融合策略 =================
    FUSION_MODE: str = "gmu"  # 'concat' / 'attention' / 'gate' / 'gmu'
    
    # ================= V5: Zerone分支配置 =================
    ZERONE_USE_CNN: bool = True  # True: 特征图+ResNet18, False: MLP
    ZERONE_IMG_SIZE: int = 20    # 特征图尺寸 (3×20×20=1200)
    
    # ================= V5: 模态Dropout =================
    USE_MODALITY_DROPOUT: bool = True
    MODALITY_DROPOUT_P: float = 0.2  # 大样本降低到0.2
    
    # ================= V5: 域适应配置 =================
    USE_DOMAIN_ADAPTATION: bool = True
    MMD_WEIGHT: float = 0.3          # 大样本略降
    CORAL_WEIGHT: float = 0.2        # 大样本略降
    USE_DANN: bool = True            # 大样本推荐开启
    DANN_WEIGHT: float = 1.0
    
    # ================= V5: 大样本正则化 =================
    DROPOUT_RATE: float = 0.3        # 大样本降低 (小样本0.5)
    LABEL_SMOOTHING: float = 0.05    # 大样本降低 (小样本0.1)
    USE_LAYERNORM: bool = False      # 大样本用BatchNorm
    
    # ================= V5: 分阶段解冻 =================
    UNFREEZE_EPOCH: int = 10
    PROGRESSIVE_UNFREEZE: bool = True
    
    # ================= 数据分离原则 =================
    STRICT_DATA_SEPARATION: bool = True
    
    # ================= 信号参数 =================
    FS: float = 8192.0
    SIGNAL_LEN: int = 8192
    INPUT_SIZE: int = 224
    
    # ================= 特征维度 =================
    ZERONE_DIM: int = TOTAL_FEAT_DIM
    CNN_FEAT_DIM: int = 512
    MLP_FEAT_DIM: int = 256  # 仅当ZERONE_USE_CNN=False时使用
    
    # ================= 模型参数 =================
    LATENT_DIM: int = 128
    LATENT_CHANNELS: int = 64
    
    # ================= 训练参数 =================
    BATCH_SIZE: int = 16
    STAGE1_EPOCHS: int = 50
    STAGE3_EPOCHS: int = 30
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    PATIENCE: int = 15
    
    # SVDD参数
    NU: float = 0.05
    
    # VAE参数
    BETA_VAE: float = 0.01
    BETA_WARMUP: int = 10
    
    # ================= 伪标签阈值 =================
    NORMAL_PERCENTILE: float = 5.0
    ANOMALY_PERCENTILE: float = 99.0
    
    # ================= 检查点与可视化 =================
    CHECKPOINT_EVERY: int = 5
    MAX_CHECKPOINTS: int = 5
    VIZ_EVERY: int = 3
    SAMPLE_PREVIEW_COUNT: int = 8
    
    # ================= 类别关键词 =================
    CLASS_KEYWORDS: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    })
    
    # ================= 设备 =================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ================= 可视化 =================
    VIZ_DPI: int = 300
    LANGS: Tuple[str, str] = ("cn", "en")
    
    def __post_init__(self):
        """初始化后处理"""
        self.PROJECT_ROOT = Path(self.PROJECT_ROOT)
        self.OUTPUT_ROOT = Path(self.OUTPUT_ROOT)
        
        self.TRAIN_DIR = self.PROJECT_ROOT / "train"
        self.VAL_DIR = self.PROJECT_ROOT / "val"
        self.TEST_DIR = self.PROJECT_ROOT / "test"
        
        # 输出目录命名
        if self.BRANCH_MODE == 'dual':
            suffix = f"_{self.FUSION_MODE}"
            if self.USE_MODALITY_DROPOUT:
                suffix += "_mdrop"
            if self.USE_DOMAIN_ADAPTATION:
                suffix += "_da"
            if self.USE_DANN:
                suffix += "_dann"
            self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}{suffix}"
        else:
            self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}"
        
        # 输出子目录
        self.STAGE1_DIR = self.BRANCH_DIR / "stage1_unsupervised"
        self.STAGE2_DIR = self.BRANCH_DIR / "stage2_pseudo_labels"
        self.STAGE3_DIR = self.BRANCH_DIR / "stage3_supervised"
        self.MODEL_DIR = self.BRANCH_DIR / "models"
        self.CHECKPOINT_DIR = self.BRANCH_DIR / "checkpoints"
        self.LOG_DIR = self.BRANCH_DIR / "logs"
        
        self.VIZ_DIR = self.BRANCH_DIR / "visualizations"
        self.VIZ_SUBDIRS = {
            "training_curves": self.VIZ_DIR / "training_curves",
            "score_dist": self.VIZ_DIR / "score_dist",
            "confusion": self.VIZ_DIR / "confusion",
            "roc_pr": self.VIZ_DIR / "roc_pr",
            "tsne": self.VIZ_DIR / "tsne",
            "sample_preview": self.VIZ_DIR / "sample_preview",
            "fusion_weights": self.VIZ_DIR / "fusion_weights",
            "domain_adaptation": self.VIZ_DIR / "domain_adaptation",
            "error_samples": self.VIZ_DIR / "error_samples",
        }
        
        for d in [self.STAGE1_DIR, self.STAGE2_DIR, self.STAGE3_DIR, 
                  self.MODEL_DIR, self.CHECKPOINT_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        for subdir in self.VIZ_SUBDIRS.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("三阶段故障诊断系统配置 (V5版本 - 大样本优化)")
        print("="*70)
        branch_names = {'hetero': '图像分支(Hetero)', 'zerone': '特征分支(Zerone)', 'dual': '双分支融合'}
        print(f"【支线模式】")
        print(f"  当前支线: {branch_names.get(self.BRANCH_MODE, self.BRANCH_MODE)}")
        if self.BRANCH_MODE == 'dual':
            fusion_names = {'concat': '等权拼接', 'attention': '注意力加权', 'gate': '交叉门控', 'gmu': 'GMU门控单元'}
            print(f"  融合策略: {fusion_names.get(self.FUSION_MODE, self.FUSION_MODE)}")
        print(f"【V5 Zerone分支】")
        print(f"  架构: {'特征图+ResNet18' if self.ZERONE_USE_CNN else 'MLP'}")
        if self.ZERONE_USE_CNN:
            print(f"  特征图尺寸: 3×{self.ZERONE_IMG_SIZE}×{self.ZERONE_IMG_SIZE}")
        print(f"【V5抗过拟合配置 (大样本优化)】")
        print(f"  模态Dropout: {'✅' if self.USE_MODALITY_DROPOUT else '❌'} (p={self.MODALITY_DROPOUT_P})")
        print(f"  域适应: {'✅' if self.USE_DOMAIN_ADAPTATION else '❌'}")
        if self.USE_DOMAIN_ADAPTATION:
            print(f"    MMD权重: {self.MMD_WEIGHT}, CORAL权重: {self.CORAL_WEIGHT}")
            print(f"    DANN: {'✅' if self.USE_DANN else '❌'} (权重={self.DANN_WEIGHT})")
        print(f"  Dropout率: {self.DROPOUT_RATE}")
        print(f"  标签平滑: {self.LABEL_SMOOTHING}")
        print(f"  归一化: {'LayerNorm' if self.USE_LAYERNORM else 'BatchNorm'}")
        print(f"【数据路径】")
        print(f"  项目根目录: {self.PROJECT_ROOT}")
        print(f"  输出目录: {self.BRANCH_DIR}")
        print(f"【训练参数】")
        print(f"  设备: {self.DEVICE}")
        print(f"  批大小: {self.BATCH_SIZE}")
        print(f"  阶段一轮数: {self.STAGE1_EPOCHS}")
        print(f"  阶段三轮数: {self.STAGE3_EPOCHS}")
        print("="*70 + "\n")


# =============================================================================
# 第3步: 模态Dropout模块
# =============================================================================

class ModalityDropout(nn.Module):
    """
    模态Dropout: 防止多模态网络的贪婪学习
    
    参考: Wu et al. (2022). ICML
    """
    
    def __init__(self, p: float = 0.2, use_learnable_tokens: bool = True,
                 img_dim: int = 512, feat_dim: int = 512):
        super().__init__()
        self.p = p
        self.use_learnable_tokens = use_learnable_tokens
        
        if use_learnable_tokens:
            self.img_missing_token = nn.Parameter(torch.randn(img_dim) * 0.02)
            self.feat_missing_token = nn.Parameter(torch.randn(feat_dim) * 0.02)
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        dropout_info = {'img_dropped': False, 'feat_dropped': False}
        
        if not self.training:
            return h_img, h_feat, dropout_info
        
        batch_size = h_img.size(0)
        
        drop_img = torch.rand(1).item() < self.p
        drop_feat = torch.rand(1).item() < self.p
        
        # 确保至少保留一个模态
        if drop_img and drop_feat:
            if torch.rand(1).item() > 0.5:
                drop_img = False
            else:
                drop_feat = False
        
        h_img_out = h_img
        h_feat_out = h_feat
        
        if drop_img:
            dropout_info['img_dropped'] = True
            if self.use_learnable_tokens:
                h_img_out = self.img_missing_token.unsqueeze(0).expand(batch_size, -1)
            else:
                h_img_out = torch.zeros_like(h_img)
        
        if drop_feat:
            dropout_info['feat_dropped'] = True
            if self.use_learnable_tokens:
                h_feat_out = self.feat_missing_token.unsqueeze(0).expand(batch_size, -1)
            else:
                h_feat_out = torch.zeros_like(h_feat)
        
        return h_img_out, h_feat_out, dropout_info


# =============================================================================
# 第4步: 域适应损失函数
# =============================================================================

def compute_mmd_loss(source_features: torch.Tensor, target_features: torch.Tensor, 
                     sigma: float = 1.0, max_samples: int = 1024) -> torch.Tensor:
    """MMD损失 (大样本优化：随机采样)"""
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    if n_s == 0 or n_t == 0:
        return torch.tensor(0.0, device=source_features.device)
    
    # 大样本优化：随机采样
    if n_s > max_samples:
        idx = torch.randperm(n_s)[:max_samples]
        source_features = source_features[idx]
        n_s = max_samples
    if n_t > max_samples:
        idx = torch.randperm(n_t)[:max_samples]
        target_features = target_features[idx]
        n_t = max_samples
    
    def rbf_kernel(x, y):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    K_ss = rbf_kernel(source_features, source_features)
    K_tt = rbf_kernel(target_features, target_features)
    K_st = rbf_kernel(source_features, target_features)
    
    mmd = K_ss.sum() / (n_s * n_s) + K_tt.sum() / (n_t * n_t) - 2 * K_st.sum() / (n_s * n_t)
    
    return torch.clamp(mmd, min=0)


def compute_coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """CORAL损失 (协方差对齐)"""
    d = source_features.size(1)
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    if n_s <= 1 or n_t <= 1:
        return torch.tensor(0.0, device=source_features.device)
    
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    
    source_cov = (source_centered.T @ source_centered) / (n_s - 1)
    target_cov = (target_centered.T @ target_centered) / (n_t - 1)
    
    loss = torch.sum((source_cov - target_cov) ** 2) / (4 * d * d)
    
    return loss


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 (DANN)"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """域判别器 (DANN)"""
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)  # 二分类：源域/目标域
        )
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.discriminator(reversed_features)


# =============================================================================
# 第5步: 融合模块定义
# =============================================================================

class ConcatFusion(nn.Module):
    """等权拼接融合"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        
        if use_layernorm:
            self.fusion = nn.Sequential(
                nn.Linear(img_dim + feat_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(img_dim + feat_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        self.out_dim = out_dim
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_cat = torch.cat([h_img, h_feat], dim=1)
        h_fused = self.fusion(h_cat)
        return {'h_fused': h_fused, 'weights': None}


class AttentionFusion(nn.Module):
    """注意力加权融合"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        
        self.proj_img = nn.Sequential(
            nn.Linear(img_dim, out_dim),
            norm_layer(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_feat = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            norm_layer(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gating_mlp = nn.Sequential(
            nn.Linear(img_dim + feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        concat_feat = torch.cat([h_img, h_feat], dim=1)
        scores = self.gating_mlp(concat_feat)
        weights = torch.softmax(scores, dim=1)
        
        h_img_proj = self.proj_img(h_img)
        h_feat_proj = self.proj_feat(h_feat)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        h_fused = alpha * h_img_proj + beta * h_feat_proj
        
        return {'h_fused': h_fused, 'weights': weights}


class GatedFusion(nn.Module):
    """交叉门控融合"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        
        self.gate_for_img = nn.Sequential(
            nn.Linear(feat_dim, img_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        self.gate_for_feat = nn.Sequential(
            nn.Linear(img_dim, feat_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + feat_dim, out_dim),
            norm_layer(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        g_img = self.gate_for_img(h_feat)
        g_feat = self.gate_for_feat(h_img)
        
        h_img_gated = h_img * g_img
        h_feat_gated = h_feat * g_feat
        
        h_cat = torch.cat([h_img_gated, h_feat_gated], dim=1)
        h_fused = self.fusion(h_cat)
        
        weights = {
            'g_img_mean': g_img.mean(dim=1),
            'g_feat_mean': g_feat.mean(dim=1),
        }
        
        return {'h_fused': h_fused, 'weights': weights}


class GMUFusion(nn.Module):
    """门控多模态单元 (推荐)"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        
        self.img_transform = nn.Sequential(
            nn.Linear(img_dim, out_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.feat_transform = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(img_dim + feat_dim, out_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        if use_layernorm:
            self.output_norm = nn.LayerNorm(out_dim)
        else:
            self.output_norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_img_t = self.img_transform(h_img)
        h_feat_t = self.feat_transform(h_feat)
        
        concat_feat = torch.cat([h_img, h_feat], dim=1)
        z = self.gate(concat_feat)
        
        h_fused = z * h_img_t + (1 - z) * h_feat_t
        h_fused = self.output_norm(h_fused)
        
        weights = {
            'z_mean': z.mean(dim=1),
            'z_std': z.std(dim=1),
        }
        
        return {'h_fused': h_fused, 'weights': weights}
# =============================================================================
# 第6步: 可视化工具类
# =============================================================================

class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, cfg: ThreeStageConfigV5):
        self.cfg = cfg
        self.setup_style()
        self.fusion_weights_history = []
        self.domain_adaptation_history = []
    
    def setup_style(self):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 150,
            'savefig.dpi': self.cfg.VIZ_DPI,
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
        })
    
    def get_label(self, key: str, lang: str = 'cn') -> str:
        return LABELS.get(lang, LABELS['en']).get(key, key)
    
    def record_fusion_weights(self, weights, labels=None):
        if weights is None:
            return
        if isinstance(weights, dict):
            record = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                     for k, v in weights.items()}
            if labels is not None:
                record['labels'] = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            self.fusion_weights_history.append(record)
        elif isinstance(weights, torch.Tensor):
            self.fusion_weights_history.append({
                'weights': weights.detach().cpu().numpy(),
                'labels': labels.cpu().numpy() if labels is not None else None
            })
    
    def record_domain_adaptation(self, mmd_loss, coral_loss, dann_loss, epoch):
        self.domain_adaptation_history.append({
            'epoch': epoch,
            'mmd': mmd_loss,
            'coral': coral_loss,
            'dann': dann_loss
        })
    
    def plot_sample_preview(self, images: List[np.ndarray], labels: List[int], lang: str = 'cn'):
        n = min(len(images), 8)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        title = '样本预览' if lang == 'cn' else 'Sample Preview'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for i in range(n):
            ax = axes[i // 4, i % 4]
            img = images[i]
            
            if img.ndim == 3 and img.shape[0] == 3:
                display_img = np.transpose(img, (1, 2, 0))
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
            else:
                display_img = img
            
            ax.imshow(display_img)
            label_text = self.get_label('normal' if labels[i] == 0 else 'fault', lang)
            ax.set_title(label_text, fontsize=10,
                        color=COLORS['normal'] if labels[i] == 0 else COLORS['fault'])
            ax.axis('off')
        
        for i in range(n, 8):
            axes[i // 4, i % 4].axis('off')
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["sample_preview"] / f"samples_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_training_curves(self, history: Dict, stage: str, lang: str = 'cn'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = history['epoch']
        
        ax1 = axes[0]
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'svdd_loss' in history:
            ax1.plot(epochs, history['svdd_loss'], 'r--', label='SVDD Loss', linewidth=1.5)
        if 'vae_loss' in history:
            ax1.plot(epochs, history['vae_loss'], 'g--', label='VAE Loss', linewidth=1.5)
        
        ax1.set_xlabel(self.get_label('epoch', lang))
        ax1.set_ylabel(self.get_label('loss', lang))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('训练损失' if lang == 'cn' else 'Training Loss')
        
        ax2 = axes[1]
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2)
        if 'val_f1' in history:
            ax2.plot(epochs, history['val_f1'], 'r-', label='F1 Score', linewidth=2)
        if 'val_precision' in history:
            ax2.plot(epochs, history['val_precision'], 'g--', label='Precision', linewidth=1.5)
        if 'val_recall' in history:
            ax2.plot(epochs, history['val_recall'], 'm--', label='Recall', linewidth=1.5)
        
        ax2.set_xlabel(self.get_label('epoch', lang))
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_title('验证指标' if lang == 'cn' else 'Validation Metrics')
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["training_curves"] / f"{stage}_curves_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_score_distribution(self, scores: np.ndarray, t_normal: float, t_anomaly: float, lang: str = 'cn'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(scores, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        ax.axvline(t_normal, color=COLORS['normal'], linestyle='--', linewidth=2,
                  label=f'正常阈值: {t_normal:.3f}' if lang == 'cn' else f'Normal: {t_normal:.3f}')
        ax.axvline(t_anomaly, color=COLORS['fault'], linestyle='--', linewidth=2,
                  label=f'异常阈值: {t_anomaly:.3f}' if lang == 'cn' else f'Anomaly: {t_anomaly:.3f}')
        
        ax.set_xlabel('异常得分' if lang == 'cn' else 'Anomaly Score')
        ax.set_ylabel('频数' if lang == 'cn' else 'Frequency')
        ax.set_title('异常得分分布' if lang == 'cn' else 'Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["score_dist"] / f"score_dist_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, lang: str = 'cn'):
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = [self.get_label('normal', lang), self.get_label('fault', lang)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_xlabel('预测标签' if lang == 'cn' else 'Predicted Label')
        ax.set_ylabel('真实标签' if lang == 'cn' else 'True Label')
        ax.set_title('混淆矩阵' if lang == 'cn' else 'Confusion Matrix')
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["confusion"] / f"confusion_matrix_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_roc_pr_curves(self, y_true: np.ndarray, y_scores: np.ndarray, lang: str = 'cn'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color=COLORS['primary'], linewidth=2, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('假阳率' if lang == 'cn' else 'False Positive Rate')
        axes[0].set_ylabel('真阳率' if lang == 'cn' else 'True Positive Rate')
        axes[0].set_title('ROC曲线' if lang == 'cn' else 'ROC Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        axes[1].plot(recall, precision, color=COLORS['secondary'], linewidth=2, label=f'AUC = {pr_auc:.3f}')
        axes[1].set_xlabel('召回率' if lang == 'cn' else 'Recall')
        axes[1].set_ylabel('精确率' if lang == 'cn' else 'Precision')
        axes[1].set_title('PR曲线' if lang == 'cn' else 'PR Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["roc_pr"] / f"roc_pr_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, lang: str = 'cn'):
        if len(features) > 2000:
            indices = np.random.choice(len(features), 2000, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label, name, color in [(0, 'normal', COLORS['normal']), (1, 'fault', COLORS['fault'])]:
            mask = labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=color, label=self.get_label(name, lang), alpha=0.6, s=30)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE可视化' if lang == 'cn' else 't-SNE Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["tsne"] / f"tsne_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_fusion_weights(self, lang: str = 'cn'):
        if not self.fusion_weights_history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        title = '融合权重分布 (V5)' if lang == 'cn' else 'Fusion Weight Distribution (V5)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        all_records = self.fusion_weights_history
        
        if 'z_mean' in all_records[0]:
            z_means = np.concatenate([r['z_mean'] for r in all_records])
            
            axes[0].hist(z_means, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
            axes[0].axvline(0.5, color='red', linestyle='--', label='平衡点' if lang == 'cn' else 'Balance')
            axes[0].set_xlabel('门控值 z' if lang == 'cn' else 'Gate z')
            axes[0].set_ylabel('频数' if lang == 'cn' else 'Count')
            axes[0].set_title('GMU门控分布' if lang == 'cn' else 'GMU Gate Distribution')
            axes[0].legend()
            
            if 'z_std' in all_records[0]:
                z_stds = np.concatenate([r['z_std'] for r in all_records])
                axes[1].hist(z_stds, bins=30, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
                axes[1].set_xlabel('门控标准差' if lang == 'cn' else 'Gate Std')
                axes[1].set_ylabel('频数' if lang == 'cn' else 'Count')
                axes[1].set_title('门控一致性' if lang == 'cn' else 'Gate Consistency')
        
        elif 'weights' in all_records[0]:
            weights = np.concatenate([r['weights'] for r in all_records])
            
            axes[0].hist(weights[:, 0], bins=30, color=COLORS['primary'], alpha=0.7, label='α (Hetero)')
            axes[0].hist(weights[:, 1], bins=30, color=COLORS['secondary'], alpha=0.7, label='β (Zerone)')
            axes[0].set_xlabel('权重值' if lang == 'cn' else 'Weight Value')
            axes[0].set_ylabel('频数' if lang == 'cn' else 'Count')
            axes[0].legend()
            
            axes[1].scatter(weights[:, 0], weights[:, 1], alpha=0.5, s=10)
            axes[1].plot([0, 1], [1, 0], 'r--', label='α + β = 1')
            axes[1].set_xlabel('α (Hetero)')
            axes[1].set_ylabel('β (Zerone)')
            axes[1].legend()
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["fusion_weights"] / f"fusion_weights_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def plot_domain_adaptation_loss(self, lang: str = 'cn'):
        if not self.domain_adaptation_history:
            return
        
        epochs = [r['epoch'] for r in self.domain_adaptation_history]
        mmd_losses = [r['mmd'] for r in self.domain_adaptation_history]
        coral_losses = [r['coral'] for r in self.domain_adaptation_history]
        dann_losses = [r['dann'] for r in self.domain_adaptation_history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, mmd_losses, 'b-', label='MMD Loss', linewidth=2)
        ax.plot(epochs, coral_losses, 'r-', label='CORAL Loss', linewidth=2)
        ax.plot(epochs, dann_losses, 'g-', label='DANN Loss', linewidth=2)
        
        ax.set_xlabel(self.get_label('epoch', lang))
        ax.set_ylabel('损失值' if lang == 'cn' else 'Loss Value')
        ax.set_title('域适应损失曲线' if lang == 'cn' else 'Domain Adaptation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["domain_adaptation"] / f"da_loss_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)


# =============================================================================
# 第7步: 日志和检查点管理器
# =============================================================================

class TrainingLogger:
    def __init__(self, cfg: ThreeStageConfigV5, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.log_file = cfg.LOG_DIR / f"{stage}_training_log.csv"
        self.records = []
    
    def log(self, **kwargs):
        self.records.append(kwargs)
    
    def save_csv(self):
        if not self.records:
            return
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[日志] 训练日志已保存: {self.log_file}")


class CheckpointManager:
    def __init__(self, cfg: ThreeStageConfigV5, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.ckpt_dir = cfg.CHECKPOINT_DIR / stage
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = cfg.MAX_CHECKPOINTS
    
    def save(self, model, optimizer, epoch: int, metrics: Dict, scheduler=None):
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }
        if scheduler:
            ckpt['scheduler_state'] = scheduler.state_dict()
        if hasattr(model, 'center'):
            ckpt['center'] = model.center
        
        path = self.ckpt_dir / f"checkpoint_epoch{epoch:03d}.pth"
        torch.save(ckpt, path)
        print(f"[检查点] 已保存: {path.name}")
        self._cleanup()
    
    def _cleanup(self):
        ckpts = sorted(self.ckpt_dir.glob("checkpoint_epoch*.pth"))
        while len(ckpts) > self.max_keep:
            ckpts[0].unlink()
            ckpts = ckpts[1:]
    
    def get_latest(self) -> Optional[Path]:
        ckpts = sorted(self.ckpt_dir.glob("checkpoint_epoch*.pth"))
        return ckpts[-1] if ckpts else None


# =============================================================================
# 第8步: 数据集类
# =============================================================================

class TransformerVibrationDataset(Dataset):
    """变压器振动数据集 (V5: 支持Zerone特征图)"""
    
    def __init__(self, data_dir: Path, cfg: ThreeStageConfigV5, 
                 use_labels: bool = False, split_name: str = "UNKNOWN"):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.split_name = split_name
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        if not self.data_dir.exists():
            print(f"[警告] 数据目录不存在: {self.data_dir}")
            return
        
        label_counts = Counter()
        
        for jsonl_file in self.data_dir.rglob("*.jsonl"):
            parent_name = jsonl_file.parent.name.lower()
            
            label = -1
            for class_name, keywords in self.cfg.CLASS_KEYWORDS.items():
                if any(kw in parent_name for kw in keywords):
                    label = 0 if class_name == "正常" else 1
                    break
            
            if self.use_labels:
                if label == -1:
                    continue
                label_counts[label] += 1
            
            self.samples.append({'file': jsonl_file, 'label': label})
        
        print(f"[{self.split_name}] 加载 {len(self.samples)} 个样本")
        if self.use_labels and label_counts:
            for lbl, cnt in sorted(label_counts.items()):
                name = "正常" if lbl == 0 else "故障"
                print(f"  {name}: {cnt}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        try:
            with open(sample['file'], 'r', encoding='utf-8') as f:
                data = json.loads(f.readline())
        except:
            data = {'signal': [0] * self.cfg.SIGNAL_LEN}
        
        signal = np.array(data.get('signal', [0] * self.cfg.SIGNAL_LEN), dtype=np.float32)
        
        if len(signal) < self.cfg.SIGNAL_LEN:
            signal = np.pad(signal, (0, self.cfg.SIGNAL_LEN - len(signal)))
        else:
            signal = signal[:self.cfg.SIGNAL_LEN]
        
        # 生成Hetero三通道图像
        hetero_img = self._generate_hetero_image(signal)
        
        # 生成Zerone特征 (V5: 可选特征图或向量)
        zerone_feat = self._extract_zerone_features(signal)
        
        if self.cfg.ZERONE_USE_CNN:
            # V5: 转换为特征图 (3, 20, 20)
            zerone_output = self._features_to_image(zerone_feat)
        else:
            zerone_output = zerone_feat
        
        label = sample['label'] if sample['label'] != -1 else 0
        
        return (
            torch.from_numpy(hetero_img).float(),
            torch.from_numpy(zerone_output).float(),
            label,
            idx
        )
    
    def _features_to_image(self, features: np.ndarray) -> np.ndarray:
        """
        V5新增: 将1200维特征转换为3×20×20特征图
        
        排列策略:
            - Channel 0: 时域(15) + STFT前部分(385) = 400 → 20×20
            - Channel 1: STFT后部分 + PSD前部分 = 400 → 20×20  
            - Channel 2: PSD后部分 + 高频(8) = 400 → 20×20
        """
        size = self.cfg.ZERONE_IMG_SIZE  # 20
        total_per_channel = size * size  # 400
        
        # 确保特征维度正确
        if len(features) < TOTAL_FEAT_DIM:
            features = np.pad(features, (0, TOTAL_FEAT_DIM - len(features)))
        
        # 分割成三个通道
        ch0 = features[0:total_per_channel]
        ch1 = features[total_per_channel:2*total_per_channel]
        ch2 = features[2*total_per_channel:3*total_per_channel]
        
        # Reshape成2D
        ch0 = ch0.reshape(size, size)
        ch1 = ch1.reshape(size, size)
        ch2 = ch2.reshape(size, size)
        
        # 归一化
        for ch in [ch0, ch1, ch2]:
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch[:] = (ch - ch_min) / (ch_max - ch_min)
        
        # 上采样到224×224以匹配Hetero分支
        ch0_resized = cv2.resize(ch0, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
        ch1_resized = cv2.resize(ch1, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
        ch2_resized = cv2.resize(ch2, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
        
        return np.stack([ch0_resized, ch1_resized, ch2_resized], axis=0)
    
    def _generate_hetero_image(self, signal: np.ndarray) -> np.ndarray:
        """生成三通道时频图像"""
        size = self.cfg.INPUT_SIZE
        
        # Channel 0: CWT
        try:
            scales = np.arange(1, 65)
            coeffs, _ = pywt.cwt(signal[:2048], scales, 'morl')
            cwt_img = np.abs(coeffs)
            cwt_img = cv2.resize(cwt_img, (size, size))
            cwt_img = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
        except:
            cwt_img = np.zeros((size, size))
        
        # Channel 1: STFT
        try:
            nperseg = 256
            hop = 64
            stft_matrix = []
            for i in range(0, len(signal) - nperseg, hop):
                segment = signal[i:i+nperseg]
                spectrum = np.abs(np.fft.rfft(segment * np.hanning(nperseg)))
                stft_matrix.append(spectrum)
            stft_img = np.array(stft_matrix).T
            stft_img = cv2.resize(stft_img, (size, size))
            stft_img = (stft_img - stft_img.min()) / (stft_img.max() - stft_img.min() + 1e-8)
        except:
            stft_img = np.zeros((size, size))
        
        # Channel 2: Context
        try:
            n_rows = size
            n_cols = size
            total = n_rows * n_cols
            if len(signal) >= total:
                context_img = signal[:total].reshape(n_rows, n_cols)
            else:
                padded = np.pad(signal, (0, total - len(signal)))
                context_img = padded.reshape(n_rows, n_cols)
            context_img = (context_img - context_img.min()) / (context_img.max() - context_img.min() + 1e-8)
        except:
            context_img = np.zeros((size, size))
        
        return np.stack([cwt_img, stft_img, context_img], axis=0)
    
    def _extract_zerone_features(self, signal: np.ndarray) -> np.ndarray:
        """提取1200维工程特征"""
        features = []
        
        # 时域特征 (15维)
        features.extend(self._time_domain_features(signal))
        
        # STFT特征 (127维)
        features.extend(self._stft_features(signal))
        
        # PSD特征 (1050维)
        features.extend(self._psd_features(signal))
        
        # 高频特征 (8维)
        features.extend(self._high_freq_features(signal))
        
        feat_array = np.array(features, dtype=np.float32)
        
        if len(feat_array) < TOTAL_FEAT_DIM:
            feat_array = np.pad(feat_array, (0, TOTAL_FEAT_DIM - len(feat_array)))
        else:
            feat_array = feat_array[:TOTAL_FEAT_DIM]
        
        return feat_array
    
    def _time_domain_features(self, signal: np.ndarray) -> List[float]:
        feats = []
        feats.append(np.mean(signal))
        feats.append(np.std(signal))
        feats.append(np.max(signal))
        feats.append(np.min(signal))
        feats.append(np.max(signal) - np.min(signal))
        feats.append(np.sqrt(np.mean(signal**2)))
        feats.append(np.mean(np.abs(signal)))
        feats.append(skew(signal))
        feats.append(kurtosis(signal))
        feats.append(np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-8))
        feats.append(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-8))
        feats.append(np.sqrt(np.mean(signal**2)) / (np.mean(np.abs(signal)) + 1e-8))
        feats.append(np.max(np.abs(signal)) / (np.sqrt(np.mean(np.abs(signal))) + 1e-8))
        feats.append(np.sum(signal**2))
        feats.append(np.percentile(signal, 75) - np.percentile(signal, 25))
        return feats[:TIME_DOMAIN_DIM]
    
    def _stft_features(self, signal: np.ndarray) -> List[float]:
        nperseg = 256
        hop = 128
        stft_feats = []
        for i in range(0, min(len(signal) - nperseg, nperseg * 4), hop):
            segment = signal[i:i+nperseg]
            spectrum = np.abs(np.fft.rfft(segment * np.hanning(nperseg)))
            stft_feats.extend(spectrum[:32])
        if len(stft_feats) < STFT_BAND_DIM:
            stft_feats.extend([0] * (STFT_BAND_DIM - len(stft_feats)))
        return stft_feats[:STFT_BAND_DIM]
    
    def _psd_features(self, signal: np.ndarray) -> List[float]:
        try:
            freqs, psd = sig.welch(signal, fs=self.cfg.FS, nperseg=min(2048, len(signal)))
            if len(psd) < PSD_BAND_DIM:
                psd = np.pad(psd, (0, PSD_BAND_DIM - len(psd)))
            return psd[:PSD_BAND_DIM].tolist()
        except:
            return [0] * PSD_BAND_DIM
    
    def _high_freq_features(self, signal: np.ndarray) -> List[float]:
        fft = np.abs(np.fft.rfft(signal))
        n = len(fft)
        hf_feats = []
        hf_feats.append(np.mean(fft[n//2:]))
        hf_feats.append(np.max(fft[n//2:]))
        hf_feats.append(np.std(fft[n//2:]))
        hf_feats.append(np.sum(fft[n//2:]) / (np.sum(fft) + 1e-8))
        for i in range(4):
            start = n//2 + i * n//8
            end = start + n//8
            hf_feats.append(np.mean(fft[start:end]))
        return hf_feats[:HIGH_FREQ_DIM]
# =============================================================================
# 第9步: 编码器网络 (V5: Zerone也用ResNet18)
# =============================================================================

class HeteroCNN(nn.Module):
    """Hetero图像分支 - ResNet18编码器"""
    
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ZeroneCNN(nn.Module):
    """
    V5新增: Zerone特征分支 - 特征图 + ResNet18
    
    输入: 3×224×224 特征图 (由1200维特征reshape而来)
    输出: 512维特征
    """
    
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, 3, 224, 224) 特征图
        返回:
            (B, 512) 编码特征
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ZeroneMLP(nn.Module):
    """Zerone特征分支 - MLP编码器 (兼容模式)"""
    
    def __init__(self, input_dim: int = 1200, output_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.fc(h)


class BranchEncoderV5(nn.Module):
    """
    支线编码器 (V5版本)
    
    V5改进:
        - Zerone分支可选: CNN(特征图) / MLP
        - 两分支对称架构 (都是512维输出)
        - 集成模态Dropout
        - 支持多种融合策略
    """
    
    def __init__(self, cfg: ThreeStageConfigV5):
        super().__init__()
        self.cfg = cfg
        self.branch_mode = cfg.BRANCH_MODE
        self.fusion_mode = cfg.FUSION_MODE
        
        # Hetero分支
        if self.branch_mode in ['hetero', 'dual']:
            self.hetero_branch = HeteroCNN(output_dim=cfg.CNN_FEAT_DIM)
        
        # Zerone分支 (V5: 可选CNN或MLP)
        if self.branch_mode in ['zerone', 'dual']:
            if cfg.ZERONE_USE_CNN:
                # V5新增: 特征图 + ResNet18
                self.zerone_branch = ZeroneCNN(output_dim=cfg.CNN_FEAT_DIM)
                self.zerone_out_dim = cfg.CNN_FEAT_DIM  # 512
            else:
                # 兼容模式: MLP
                self.zerone_branch = ZeroneMLP(
                    input_dim=cfg.ZERONE_DIM,
                    output_dim=cfg.CNN_FEAT_DIM,  # 也输出512维
                    use_layernorm=cfg.USE_LAYERNORM,
                    dropout=cfg.DROPOUT_RATE
                )
                self.zerone_out_dim = cfg.CNN_FEAT_DIM
        
        # 确定输出维度
        if self.branch_mode == 'hetero':
            self.output_dim = cfg.CNN_FEAT_DIM  # 512
        elif self.branch_mode == 'zerone':
            self.output_dim = cfg.CNN_FEAT_DIM  # 512
        else:  # dual
            # V5: 模态Dropout
            if cfg.USE_MODALITY_DROPOUT:
                self.modality_dropout = ModalityDropout(
                    p=cfg.MODALITY_DROPOUT_P,
                    use_learnable_tokens=True,
                    img_dim=cfg.CNN_FEAT_DIM,
                    feat_dim=cfg.CNN_FEAT_DIM  # V5: 两分支都是512
                )
            else:
                self.modality_dropout = None
            
            # 融合模块 (V5: 两分支都是512维)
            fusion_kwargs = {
                'img_dim': cfg.CNN_FEAT_DIM,    # 512
                'feat_dim': cfg.CNN_FEAT_DIM,   # 512 (V5改动)
                'out_dim': 512,
                'use_layernorm': cfg.USE_LAYERNORM,
                'dropout': cfg.DROPOUT_RATE
            }
            
            if self.fusion_mode == 'concat':
                self.fusion_module = ConcatFusion(**fusion_kwargs)
            elif self.fusion_mode == 'attention':
                self.fusion_module = AttentionFusion(**fusion_kwargs)
            elif self.fusion_mode == 'gate':
                self.fusion_module = GatedFusion(**fusion_kwargs)
            elif self.fusion_mode == 'gmu':
                self.fusion_module = GMUFusion(**fusion_kwargs)
            else:
                raise ValueError(f"未知融合模式: {self.fusion_mode}")
            
            self.output_dim = 512
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        参数:
            image: (B, 3, 224, 224) Hetero时频图像
            zerone: (B, 3, 224, 224) 或 (B, 1200) Zerone输入
        """
        result = {}
        
        if self.branch_mode == 'hetero':
            h = self.hetero_branch(image)
            result['h'] = h
            
        elif self.branch_mode == 'zerone':
            h = self.zerone_branch(zerone)
            result['h'] = h
            
        else:  # dual
            h_img = self.hetero_branch(image)
            h_feat = self.zerone_branch(zerone)
            
            # V5: 模态Dropout
            dropout_info = {'img_dropped': False, 'feat_dropped': False}
            if self.modality_dropout is not None:
                h_img, h_feat, dropout_info = self.modality_dropout(h_img, h_feat)
            
            fusion_result = self.fusion_module(h_img, h_feat)
            
            result['h'] = fusion_result['h_fused']
            result['h_img'] = h_img
            result['h_feat'] = h_feat
            result['fusion_weights'] = fusion_result['weights']
            result['dropout_info'] = dropout_info
        
        return result


# =============================================================================
# 第10步: 异常检测模型
# =============================================================================

class AnomalyModelV5(nn.Module):
    """异常检测模型 V5"""
    
    def __init__(self, cfg: ThreeStageConfigV5):
        super().__init__()
        self.cfg = cfg
        
        # 编码器
        self.encoder = BranchEncoderV5(cfg)
        
        # SVDD投影头
        norm_layer = nn.LayerNorm if cfg.USE_LAYERNORM else nn.BatchNorm1d
        self.svdd_proj = nn.Sequential(
            nn.Linear(512, 256),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(256, cfg.LATENT_DIM)
        )
        self.register_buffer('center', torch.zeros(cfg.LATENT_DIM))
        
        # VAE解码器 (仅hetero/dual模式)
        if cfg.BRANCH_MODE in ['hetero', 'dual']:
            self.vae_mu = nn.Linear(512, cfg.LATENT_CHANNELS * 7 * 7)
            self.vae_logvar = nn.Linear(512, cfg.LATENT_CHANNELS * 7 * 7)
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
            self.has_vae = True
        else:
            self.has_vae = False
        
        self.alpha = 0.6
    
    def encode(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(image, zerone)
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_result = self.encode(image, zerone)
        h = enc_result['h']
        
        z_svdd = self.svdd_proj(h)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        result = {
            'h': h,
            'z_svdd': z_svdd,
            'svdd_score': svdd_score,
        }
        
        for key in ['h_img', 'h_feat', 'fusion_weights', 'dropout_info']:
            if key in enc_result:
                result[key] = enc_result[key]
        
        if self.has_vae:
            mu = self.vae_mu(h).view(-1, self.cfg.LATENT_CHANNELS, 7, 7)
            logvar = self.vae_logvar(h).view(-1, self.cfg.LATENT_CHANNELS, 7, 7)
            
            if self.training:
                std = torch.exp(0.5 * logvar)
                z_vae = mu + std * torch.randn_like(std)
            else:
                z_vae = mu
            
            recon = self.vae_decoder(z_vae)
            if recon.shape[-1] != image.shape[-1]:
                recon = F.interpolate(recon, size=image.shape[2:], mode='bilinear', align_corners=False)
            
            vae_recon_loss = F.l1_loss(recon, image, reduction='none').mean(dim=[1,2,3])
            vae_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
            
            result.update({
                'recon': recon,
                'vae_recon_loss': vae_recon_loss,
                'vae_kl': vae_kl,
            })
        
        return result
    
    def init_center(self, dataloader: DataLoader, device: torch.device):
        """初始化SVDD中心"""
        n = 0
        c = torch.zeros(self.cfg.LATENT_DIM, device=device)
        
        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="初始化SVDD中心", leave=False):
                img, zr, _, _ = batch
                img, zr = img.to(device), zr.to(device)
                enc_result = self.encode(img, zr)
                z = self.svdd_proj(enc_result['h'])
                c += z.sum(0)
                n += z.size(0)
        
        c /= n
        
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c
        
        print(f"[SVDD] 中心初始化完成，范数: {c.norm().item():.4f}")
    
    def anomaly_score(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        out = self.forward(image, zerone)
        
        if self.has_vae:
            svdd_score = out['svdd_score']
            vae_score = out['vae_recon_loss'] + 0.01 * out['vae_kl']
            
            svdd_norm = svdd_score / (svdd_score.mean() + 1e-8)
            vae_norm = vae_score / (vae_score.mean() + 1e-8)
            
            return self.alpha * svdd_norm + (1 - self.alpha) * vae_norm
        else:
            return out['svdd_score']


# =============================================================================
# 第11步: 分类器 (V5: 分阶段解冻 + DANN)
# =============================================================================

class FaultClassifierV5(nn.Module):
    """故障分类器 V5 (分阶段解冻 + DANN)"""
    
    def __init__(self, encoder: BranchEncoderV5, cfg: ThreeStageConfigV5,
                 num_classes: int = 2, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.num_classes = num_classes
        
        # 分类头
        norm_layer = nn.LayerNorm if cfg.USE_LAYERNORM else nn.BatchNorm1d
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(256, 128),
            norm_layer(128),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )
        
        # V5: 域判别器 (DANN)
        if cfg.USE_DANN:
            self.domain_discriminator = DomainDiscriminator(feature_dim=512)
        else:
            self.domain_discriminator = None
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[V5] 编码器已冻结")
    
    def unfreeze_encoder(self, mode: str = 'all'):
        """分阶段解冻"""
        if mode == 'all':
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("[V5] 编码器已完全解冻")
        
        elif mode == 'fusion_only':
            if hasattr(self.encoder, 'fusion_module'):
                for param in self.encoder.fusion_module.parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, 'modality_dropout') and self.encoder.modality_dropout is not None:
                for param in self.encoder.modality_dropout.parameters():
                    param.requires_grad = True
            print("[V5] 融合层已解冻")
        
        elif mode == 'last_layers':
            # 解冻CNN最后几层
            if hasattr(self.encoder, 'hetero_branch'):
                for name, param in self.encoder.hetero_branch.named_parameters():
                    if 'layer4' in name or 'fc' in name:
                        param.requires_grad = True
            if hasattr(self.encoder, 'zerone_branch'):
                for name, param in self.encoder.zerone_branch.named_parameters():
                    if 'layer4' in name or 'fc' in name:
                        param.requires_grad = True
            if hasattr(self.encoder, 'fusion_module'):
                for param in self.encoder.fusion_module.parameters():
                    param.requires_grad = True
            print("[V5] 最后几层已解冻")
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor, 
                alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        enc_result = self.encoder(image, zerone)
        h = enc_result['h']
        
        logits = self.classifier(h)
        
        result = {
            'logits': logits,
            'h': h,
        }
        
        # DANN域判别
        if self.domain_discriminator is not None:
            domain_output = self.domain_discriminator(h, alpha)
            result['domain_output'] = domain_output
        
        for key in ['fusion_weights', 'h_img', 'h_feat', 'dropout_info']:
            if key in enc_result:
                result[key] = enc_result[key]
        
        return result
# =============================================================================
# 第12步: 阶段一 - 无监督训练
# =============================================================================

def train_stage1(cfg: ThreeStageConfigV5, resume_from: Path = None) -> Tuple[AnomalyModelV5, Dict]:
    """阶段一：无监督学习"""
    print("\n" + "="*70)
    print("阶段一：无监督学习 (V5)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage1")
    ckpt_mgr = CheckpointManager(cfg, "stage1")
    
    print("\n[1/4] 加载数据 (仅TRAIN)...")
    train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
    
    print(f"  训练数据集大小: {len(train_ds)}")
    
    if len(train_ds) == 0:
        print("[错误] TRAIN数据集为空!")
        return None, {}
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              num_workers=0, drop_last=True)
    
    # 样本预览
    print("\n[*] 生成样本预览...")
    preview_images, preview_labels = [], []
    for i in range(min(cfg.SAMPLE_PREVIEW_COUNT, len(train_ds))):
        img, zr, lbl, _ = train_ds[i]
        preview_images.append(img.numpy())
        preview_labels.append(lbl)
    
    for lang in cfg.LANGS:
        viz.plot_sample_preview(preview_images, preview_labels, lang=lang)
    
    # 构建模型
    print("\n[2/4] 构建模型...")
    model = AnomalyModelV5(cfg).to(device)
    print(f"  支线模式: {cfg.BRANCH_MODE}")
    print(f"  融合策略: {cfg.FUSION_MODE}")
    print(f"  Zerone架构: {'CNN(特征图)' if cfg.ZERONE_USE_CNN else 'MLP'}")
    print(f"  模态Dropout: {'✅' if cfg.USE_MODALITY_DROPOUT else '❌'}")
    print(f"  VAE启用: {model.has_vae}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE1_EPOCHS)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # 断点恢复
    if resume_from and resume_from.exists():
        print(f"\n[恢复] 从检查点加载: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        if 'center' in ckpt:
            model.center = ckpt['center']
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('metrics', {}).get('total_loss', float('inf'))
        print(f"  恢复到 epoch {start_epoch}, best_loss={best_loss:.4f}")
    
    # VAE预训练
    if model.has_vae and start_epoch == 0:
        print("\n[2.5/4] VAE预训练...")
        vae_pretrain_epochs = 5
        for epoch in range(vae_pretrain_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"VAE预训练 {epoch+1}/{vae_pretrain_epochs}", leave=False):
                img, zr, _, _ = batch
                img, zr = img.to(device), zr.to(device)
                
                out = model(img, zr)
                loss = out['vae_recon_loss'].mean() + 0.01 * out['vae_kl'].mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  VAE预训练 Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # SVDD中心初始化
    if start_epoch == 0:
        print("\n[3/4] 初始化SVDD中心...")
        model.init_center(train_loader, device)
    
    # 联合训练
    print(f"\n[4/4] 联合训练 ({cfg.STAGE1_EPOCHS}轮)...")
    
    history = {'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': []}
    
    for epoch in range(start_epoch, cfg.STAGE1_EPOCHS):
        model.train()
        epoch_svdd, epoch_vae, epoch_total = 0, 0, 0
        
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch / max(cfg.BETA_WARMUP, 1)))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False)
        for batch in pbar:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            
            svdd_loss = out['svdd_score'].mean()
            
            if model.has_vae:
                vae_loss = out['vae_recon_loss'].mean() + beta * out['vae_kl'].mean()
                total_loss = 0.5 * svdd_loss + 0.5 * vae_loss
            else:
                vae_loss = torch.tensor(0.0)
                total_loss = svdd_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_svdd += svdd_loss.item()
            epoch_vae += vae_loss.item()
            epoch_total += total_loss.item()
            
            pbar.set_postfix({'svdd': f"{svdd_loss.item():.4f}", 'vae': f"{vae_loss.item():.4f}"})
        
        scheduler.step()
        
        n_batches = len(train_loader)
        avg_svdd = epoch_svdd / n_batches
        avg_vae = epoch_vae / n_batches
        avg_total = epoch_total / n_batches
        
        history['epoch'].append(epoch + 1)
        history['svdd_loss'].append(avg_svdd)
        history['vae_loss'].append(avg_vae)
        history['total_loss'].append(avg_total)
        
        logger.log(epoch=epoch+1, svdd_loss=avg_svdd, vae_loss=avg_vae, total_loss=avg_total)
        
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.center,
                'epoch': epoch,
                'metrics': {'total_loss': best_loss}
            }, cfg.MODEL_DIR / "stage1_best.pth")
        
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(model, optimizer, epoch + 1,
                         {'svdd_loss': avg_svdd, 'vae_loss': avg_vae, 'total_loss': avg_total},
                         scheduler)
        
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage1", lang=lang)
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
    
    logger.save_csv()
    
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage1", lang=lang)
    
    print(f"\n【阶段一完成】最佳损失: {best_loss:.4f}")
    
    return model, history


# =============================================================================
# 第13步: 阶段二 - 伪标签生成
# =============================================================================

def run_stage2(model: AnomalyModelV5, cfg: ThreeStageConfigV5) -> Dict:
    """阶段二：基于异常得分生成伪标签"""
    print("\n" + "="*70)
    print("阶段二：伪标签生成")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    viz = VisualizationManager(cfg)
    
    print("\n[1/3] 加载数据 (仅TRAIN)...")
    train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
    
    loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\n[2/3] 计算异常得分...")
    model.eval()
    all_scores = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="计算得分", leave=False):
            img, zr, _, idx = batch
            img, zr = img.to(device), zr.to(device)
            scores = model.anomaly_score(img, zr)
            all_scores.extend(scores.cpu().tolist())
            all_indices.extend(idx.tolist())
    
    all_scores = np.array(all_scores)
    all_indices = np.array(all_indices)
    
    print("\n[3/3] 生成伪标签...")
    t_normal = np.percentile(all_scores, cfg.NORMAL_PERCENTILE)
    t_anomaly = np.percentile(all_scores, cfg.ANOMALY_PERCENTILE)
    
    pseudo_normal = all_indices[all_scores <= t_normal]
    pseudo_anomaly = all_indices[all_scores >= t_anomaly]
    uncertain = all_indices[(all_scores > t_normal) & (all_scores < t_anomaly)]
    
    print(f"  正常阈值 (P{cfg.NORMAL_PERCENTILE}): {t_normal:.4f}")
    print(f"  异常阈值 (P{cfg.ANOMALY_PERCENTILE}): {t_anomaly:.4f}")
    print(f"  伪标签分布:")
    print(f"    高置信正常: {len(pseudo_normal)} ({100*len(pseudo_normal)/len(all_scores):.1f}%)")
    print(f"    高置信异常: {len(pseudo_anomaly)} ({100*len(pseudo_anomaly)/len(all_scores):.1f}%)")
    print(f"    不确定: {len(uncertain)} ({100*len(uncertain)/len(all_scores):.1f}%)")
    
    pseudo_labels = {
        'all_scores': all_scores,
        'all_indices': all_indices,
        't_normal': t_normal,
        't_anomaly': t_anomaly,
        'pseudo_normal': pseudo_normal,
        'pseudo_anomaly': pseudo_anomaly,
        'uncertain': uncertain,
    }
    
    np.savez(cfg.STAGE2_DIR / "pseudo_labels.npz", **pseudo_labels)
    
    for lang in cfg.LANGS:
        viz.plot_score_distribution(all_scores, t_normal, t_anomaly, lang=lang)
    
    print(f"\n【阶段二完成】伪标签保存: {cfg.STAGE2_DIR / 'pseudo_labels.npz'}")
    
    return pseudo_labels


# =============================================================================
# 第14步: 阶段三 - 监督微调 (V5: 域适应 + DANN)
# =============================================================================

def train_stage3(model: AnomalyModelV5, pseudo_labels: Dict, cfg: ThreeStageConfigV5) -> FaultClassifierV5:
    """阶段三：有监督微调 (V5增强版)"""
    print("\n" + "="*70)
    print("阶段三：有监督微调 (V5 - 域适应 + DANN)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage3")
    ckpt_mgr = CheckpointManager(cfg, "stage3")
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    val_ds = TransformerVibrationDataset(cfg.VAL_DIR, cfg, use_labels=True, split_name="VAL")
    test_ds = TransformerVibrationDataset(cfg.TEST_DIR, cfg, use_labels=True, split_name="TEST")
    
    if len(val_ds) == 0:
        print("[警告] VAL数据集为空，无法进行监督训练")
        return None
    
    # 划分VAL为训练集和验证集 (80/20)
    n_train = int(len(val_ds) * 0.8)
    indices = list(range(len(val_ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = Subset(val_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)
    
    print(f"  VAL数据划分: 训练 {len(train_subset)} | 验证 {len(val_subset)}")
    print(f"  TEST数据: {len(test_ds)} (用于域适应和最终评估)")
    
    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 构建分类器
    print("\n[2/5] 构建分类器...")
    classifier = FaultClassifierV5(model.encoder, cfg, num_classes=2, freeze_encoder=True).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=1e-3, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # V5: Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    
    # 训练
    print(f"\n[3/5] 训练 ({cfg.STAGE3_EPOCHS}轮)...")
    print(f"  V5配置: Dropout={cfg.DROPOUT_RATE}, LabelSmoothing={cfg.LABEL_SMOOTHING}")
    print(f"  域适应: MMD={cfg.MMD_WEIGHT}, CORAL={cfg.CORAL_WEIGHT}, DANN={'✅' if cfg.USE_DANN else '❌'}")
    print(f"  分阶段解冻: Epoch {cfg.UNFREEZE_EPOCH}")
    
    history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': [], 
               'val_precision': [], 'val_recall': [], 'mmd_loss': [], 'coral_loss': [], 'dann_loss': []}
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(cfg.STAGE3_EPOCHS):
        # V5: 分阶段解冻
        if epoch == cfg.UNFREEZE_EPOCH:
            if cfg.PROGRESSIVE_UNFREEZE:
                classifier.unfreeze_encoder('fusion_only')
            else:
                classifier.unfreeze_encoder('all')
            
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, classifier.parameters()),
                lr=1e-4, weight_decay=cfg.WEIGHT_DECAY
            )
        
        if cfg.PROGRESSIVE_UNFREEZE and epoch == cfg.UNFREEZE_EPOCH + 5:
            classifier.unfreeze_encoder('last_layers')
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, classifier.parameters()),
                lr=5e-5, weight_decay=cfg.WEIGHT_DECAY
            )
        
        # DANN alpha调度
        p = epoch / cfg.STAGE3_EPOCHS
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # 训练阶段
        classifier.train()
        train_loss = 0
        epoch_mmd, epoch_coral, epoch_dann = 0, 0, 0
        
        test_iter = iter(test_loader) if cfg.USE_DOMAIN_ADAPTATION else None
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            img, zr, label, _ = batch
            img, zr, label = img.to(device), zr.to(device), label.to(device)
            
            result = classifier(img, zr, alpha=alpha)
            loss_cls = criterion(result['logits'], label)
            
            # 域适应损失
            loss_mmd = torch.tensor(0.0, device=device)
            loss_coral = torch.tensor(0.0, device=device)
            loss_dann = torch.tensor(0.0, device=device)
            
            if cfg.USE_DOMAIN_ADAPTATION and test_iter is not None:
                try:
                    test_batch = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    test_batch = next(test_iter)
                
                test_img, test_zr, _, _ = test_batch
                test_img, test_zr = test_img.to(device), test_zr.to(device)
                
                test_result = classifier(test_img, test_zr, alpha=alpha)
                
                source_feat = result['h']
                target_feat = test_result['h']
                
                loss_mmd = compute_mmd_loss(source_feat, target_feat)
                loss_coral = compute_coral_loss(source_feat, target_feat)
                
                # DANN损失
                if cfg.USE_DANN and 'domain_output' in result and 'domain_output' in test_result:
                    source_domain_label = torch.zeros(img.size(0), dtype=torch.long, device=device)
                    target_domain_label = torch.ones(test_img.size(0), dtype=torch.long, device=device)
                    
                    loss_dann = F.cross_entropy(result['domain_output'], source_domain_label) + \
                               F.cross_entropy(test_result['domain_output'], target_domain_label)
            
            # 总损失
            total_loss = loss_cls + cfg.MMD_WEIGHT * loss_mmd + cfg.CORAL_WEIGHT * loss_coral + cfg.DANN_WEIGHT * loss_dann
            
            # 记录融合权重
            if 'fusion_weights' in result and result['fusion_weights'] is not None:
                viz.record_fusion_weights(result['fusion_weights'], label)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss_cls.item()
            epoch_mmd += loss_mmd.item()
            epoch_coral += loss_coral.item()
            epoch_dann += loss_dann.item()
        
        scheduler.step()
        
        # 验证阶段
        classifier.eval()
        val_preds, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                img, zr, label, _ = batch
                img, zr = img.to(device), zr.to(device)
                result = classifier(img, zr)
                probs = F.softmax(result['logits'], dim=1)
                
                val_preds.extend(result['logits'].argmax(dim=1).cpu().tolist())
                val_labels.extend(label.tolist())
                val_probs.extend(probs[:, 1].cpu().tolist())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        
        n_batches = len(train_loader)
        avg_train_loss = train_loss / n_batches
        avg_mmd = epoch_mmd / n_batches
        avg_coral = epoch_coral / n_batches
        avg_dann = epoch_dann / n_batches
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['mmd_loss'].append(avg_mmd)
        history['coral_loss'].append(avg_coral)
        history['dann_loss'].append(avg_dann)
        
        viz.record_domain_adaptation(avg_mmd, avg_coral, avg_dann, epoch + 1)
        
        logger.log(epoch=epoch+1, train_loss=avg_train_loss, val_acc=val_acc,
                  val_f1=val_f1, val_precision=val_prec, val_recall=val_rec,
                  mmd_loss=avg_mmd, coral_loss=avg_coral, dann_loss=avg_dann)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state': classifier.state_dict(),
                'epoch': epoch,
                'f1': best_f1,
            }, cfg.MODEL_DIR / "stage3_best.pth")
        else:
            patience_counter += 1
        
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(classifier, optimizer, epoch + 1,
                         {'val_f1': val_f1, 'val_acc': val_acc}, scheduler)
        
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage3", lang=lang)
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
                  f"MMD: {avg_mmd:.4f} | CORAL: {avg_coral:.4f} | DANN: {avg_dann:.4f}")
        
        if patience_counter >= cfg.PATIENCE:
            print(f"  [早停] 连续{cfg.PATIENCE}轮无改善")
            break
    
    logger.save_csv()
    
    # 融合权重可视化
    if cfg.BRANCH_MODE == 'dual':
        print("\n[4/5] 生成融合权重可视化...")
        for lang in cfg.LANGS:
            viz.plot_fusion_weights(lang=lang)
    
    # 域适应损失可视化
    if cfg.USE_DOMAIN_ADAPTATION:
        for lang in cfg.LANGS:
            viz.plot_domain_adaptation_loss(lang=lang)
    
    # 加载最佳模型
    best_ckpt = torch.load(cfg.MODEL_DIR / "stage3_best.pth", map_location=device)
    classifier.load_state_dict(best_ckpt['model_state'])
    
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage3", lang=lang)
    
    # TEST评估
    print("\n[5/5] 在TEST上评估...")
    
    if len(test_ds) > 0:
        test_eval_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        
        classifier.eval()
        test_preds, test_labels, test_probs, test_features = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_eval_loader, desc="TEST评估", leave=False):
                img, zr, label, idx = batch
                img, zr = img.to(device), zr.to(device)
                
                result = classifier(img, zr)
                probs = F.softmax(result['logits'], dim=1)
                preds = result['logits'].argmax(dim=1)
                
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(label.tolist())
                test_probs.extend(probs[:, 1].cpu().tolist())
                test_features.append(result['h'].cpu().numpy())
        
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
        test_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)
        
        print(f"\n  【TEST评估结果 (V5)】")
        print(f"    准确率: {test_acc:.4f}")
        print(f"    F1分数: {test_f1:.4f}")
        print(f"    精确率: {test_prec:.4f}")
        print(f"    召回率: {test_rec:.4f}")
        print(f"    错误样本数: {sum(1 for p, l in zip(test_preds, test_labels) if p != l)}")
        
        # 可视化
        for lang in cfg.LANGS:
            viz.plot_confusion_matrix(np.array(test_labels), np.array(test_preds), lang=lang)
            viz.plot_roc_pr_curves(np.array(test_labels), np.array(test_probs), lang=lang)
        
        test_features = np.concatenate(test_features, axis=0)
        for lang in cfg.LANGS:
            viz.plot_tsne(test_features, np.array(test_labels), lang=lang)
        
        # 保存结果
        eval_results = {
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'n_errors': sum(1 for p, l in zip(test_preds, test_labels) if p != l),
            'config': {
                'branch_mode': cfg.BRANCH_MODE,
                'fusion_mode': cfg.FUSION_MODE,
                'zerone_use_cnn': cfg.ZERONE_USE_CNN,
                'use_modality_dropout': cfg.USE_MODALITY_DROPOUT,
                'use_domain_adaptation': cfg.USE_DOMAIN_ADAPTATION,
                'use_dann': cfg.USE_DANN,
                'dropout_rate': cfg.DROPOUT_RATE,
                'label_smoothing': cfg.LABEL_SMOOTHING,
            }
        }
        
        with open(cfg.STAGE3_DIR / "test_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n【阶段三完成】最佳验证F1: {best_f1:.4f}")
    
    return classifier


# =============================================================================
# 第15步: 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='三阶段变压器故障诊断 V5 (大样本优化)')
    
    # 基本参数
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    
    # 支线配置
    parser.add_argument('--branch', type=str, choices=['hetero', 'zerone', 'dual'], 
                       default='dual', help='支线模式')
    parser.add_argument('--fusion_mode', type=str, choices=['concat', 'attention', 'gate', 'gmu'],
                       default='gmu', help='融合策略 (推荐gmu)')
    
    # V5参数
    parser.add_argument('--zerone_mlp', action='store_true', help='Zerone使用MLP而非CNN')
    parser.add_argument('--no_modality_dropout', action='store_true', help='禁用模态Dropout')
    parser.add_argument('--no_domain_adapt', action='store_true', help='禁用域适应')
    parser.add_argument('--no_dann', action='store_true', help='禁用DANN')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='标签平滑')
    
    # 路径配置
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--output', type=str, default='./three_stage_results_v5', help='输出目录')
    
    # 功能开关
    parser.add_argument('--test_data', action='store_true', help='测试数据加载')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 初始化配置
    cfg = ThreeStageConfigV5(
        BRANCH_MODE=args.branch,
        FUSION_MODE=args.fusion_mode,
        ZERONE_USE_CNN=not args.zerone_mlp,
        USE_MODALITY_DROPOUT=not args.no_modality_dropout,
        USE_DOMAIN_ADAPTATION=not args.no_domain_adapt,
        USE_DANN=not args.no_dann,
        DROPOUT_RATE=args.dropout_rate,
        LABEL_SMOOTHING=args.label_smoothing,
    )
    
    if args.data_root:
        cfg.PROJECT_ROOT = Path(args.data_root)
    cfg.OUTPUT_ROOT = Path(args.output)
    cfg.__post_init__()
    
    cfg.print_config()
    
    # 测试数据加载
    if args.test_data:
        print("\n【测试数据加载】")
        test_ds = TransformerVibrationDataset(cfg.TEST_DIR, cfg, use_labels=True, split_name="TEST")
        if len(test_ds) > 0:
            img, zr, lbl, idx = test_ds[0]
            print(f"  支线模式: {cfg.BRANCH_MODE}")
            print(f"  融合策略: {cfg.FUSION_MODE}")
            print(f"  Hetero图像: {img.shape}")
            print(f"  Zerone输入: {zr.shape} ({'特征图' if cfg.ZERONE_USE_CNN else '向量'})")
            print(f"  标签: {lbl}")
        return
    
    # 执行训练
    device = torch.device(cfg.DEVICE)
    
    if args.all or args.stage == 1:
        resume_ckpt = None
        if args.resume:
            ckpt_mgr = CheckpointManager(cfg, "stage1")
            resume_ckpt = ckpt_mgr.get_latest()
        model, _ = train_stage1(cfg, resume_from=resume_ckpt)
    else:
        model_path = cfg.MODEL_DIR / "stage1_best.pth"
        if model_path.exists():
            model = AnomalyModelV5(cfg)
            ckpt = torch.load(model_path, map_location=cfg.DEVICE)
            model.load_state_dict(ckpt['model_state'])
            model.center = ckpt['center']
            model = model.to(cfg.DEVICE)
            print(f"[加载模型] {model_path}")
        else:
            print(f"[错误] 未找到模型: {model_path}")
            return
    
    if model is None:
        print("[错误] 阶段1模型训练失败")
        return
    
    if args.all or args.stage == 2:
        pseudo_labels = run_stage2(model, cfg)
    else:
        pseudo_path = cfg.STAGE2_DIR / "pseudo_labels.npz"
        if pseudo_path.exists():
            data = np.load(pseudo_path, allow_pickle=True)
            pseudo_labels = {k: data[k] for k in data.files}
            print(f"[加载伪标签] {pseudo_path}")
        else:
            pseudo_labels = None
    
    if args.all or args.stage == 3:
        if pseudo_labels is not None:
            train_stage3(model, pseudo_labels, cfg)
        else:
            print("[警告] 未找到伪标签，跳过阶段三")
    
    print("\n【完成】结果保存至:", cfg.BRANCH_DIR)


if __name__ == "__main__":
    main()

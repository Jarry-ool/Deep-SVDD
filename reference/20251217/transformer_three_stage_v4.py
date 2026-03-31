# -*- coding: utf-8 -*-
"""
transformer_three_stage_v4.py
==============================

交流变压器振动数据 三阶段渐进式故障诊断系统 V4
支持三条并行支线：Hetero-Only / Zerone-Only / Dual-Branch (融合)

【V4版本核心改进】
    ✅ 1. 多模态融合策略优化：
       - ConcatFusion: 等权拼接+FC (baseline)
       - AttentionFusion: 注意力加权融合 (自适应权重)
       - GatedFusion: 交叉门控融合 (互信息交换)
       
    ✅ 2. 严格的数据分离原则：
       - TRAIN: 阶段1/2中作为无标签数据（完全无监督）
       - VAL: 可读取标签，用于阶段3监督微调和阈值选择
       - TEST: 只在最终评估时读取标签，不参与任何训练
       
    ✅ 3. 集成断点续训与对比可视化功能：
       - --resume: 从检查点恢复训练
       - --compare: 生成三支线对比可视化
       - --status: 显示当前训练状态
       
    ✅ 4. 增强可视化：
       - 门控权重可视化
       - 融合特征分布对比
       - 训练曲线与评估报告

【架构设计】

    ┌──────────────────────────────────────────────────────────────┐
    │                        振动信号输入                           │
    │                      (8192点 @ 8192Hz)                       │
    └──────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐
    │  Hetero 图像分支 (CNN)    │    │  Zerone 特征分支 (MLP)    │
    │  ────────────────────    │    │  ────────────────────    │
    │  Ch0: CWT (Morlet小波)   │    │  时域: 15维              │
    │  Ch1: STFT (短时频谱)    │    │  STFT: 127维             │
    │  Ch2: Context (波形折叠) │    │  PSD: 1050维             │
    │         ↓                │    │  高频: 8维               │
    │    ResNet18编码器        │    │         ↓                │
    │         ↓                │    │    3层全连接网络          │
    │      512维特征           │    │      256维特征            │
    └──────────────────────────┘    └──────────────────────────┘
            │                               │
            └───────────┬───────────────────┘
                        ▼
    ┌────────────────────────────────────────────────────────────┐
    │         Fusion Module (可选策略)                             │
    │  ──────────────────────────────────────────────────────    │
    │  • concat:    [h_img; h_feat] → FC → 512                   │
    │  • attention: α·h_img + β·h_feat (自适应权重)               │
    │  • gate:      h_img⊙σ(W·h_feat) + h_feat⊙σ(W·h_img)       │
    └────────────────────────────────────────────────────────────┘
                        │
                        ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Hetero-Only  │  │  Dual-Branch │  │ Zerone-Only  │
    │  (支线A)     │  │   (支线C)    │  │   (支线B)    │
    └──────────────┘  └──────────────┘  └──────────────┘

【运行方式】
    # 测试数据加载
    python transformer_three_stage_v4.py --test_data
    
    # 运行单一支线 (默认concat融合)
    python transformer_three_stage_v4.py --branch dual --all
    
    # 指定融合策略
    python transformer_three_stage_v4.py --branch dual --fusion_mode attention --all
    python transformer_three_stage_v4.py --branch dual --fusion_mode gate --all
    
    # 断点续训
    python transformer_three_stage_v4.py --resume --branch dual
    
    # 生成对比可视化
    python transformer_three_stage_v4.py --compare_only
    
    # 运行全部支线对比
    python transformer_three_stage_v4.py --all_branches

Author: 基于 V3 框架优化 (V4版本)
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
from scipy.signal import stft, welch

# 可视化
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Circle
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 评估指标
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    average_precision_score, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')


# =============================================================================
# 第1步: 全局常量与特征定义
# =============================================================================

# Zerone特征维度定义
FEAT_SCHEMA = [("time", 15), ("stft", 127), ("psd", 1050), ("hf", 8)]
TOTAL_FEAT_DIM = sum(d for _, d in FEAT_SCHEMA)  # 1200

# 颜色方案 (IEEE/Nature风格)
COLORS = {
    'blue': '#0072B2',      # 深蓝
    'orange': '#E69F00',    # 橙色
    'green': '#009E73',     # 绿色
    'red': '#D55E00',       # 红色
    'purple': '#CC79A7',    # 紫色
    'cyan': '#56B4E9',      # 浅蓝
    'yellow': '#F0E442',    # 黄色
    'gray': '#999999',      # 灰色
    'normal': '#009E73',    # 正常-绿
    'fault': '#D55E00',     # 故障-红
    'uncertain': '#999999', # 不确定-灰
    'hetero': '#0072B2',    # Hetero支线-蓝
    'zerone': '#E69F00',    # Zerone支线-橙
    'dual': '#CC79A7',      # Dual支线-紫
}

# 中英文标签
LABELS = {
    'cn': {
        'normal': '正常', 'fault': '故障', 'uncertain': '不确定',
        'epoch': '训练轮次', 'loss': '损失值', 'accuracy': '准确率',
        'score': '异常得分', 'count': '样本数', 'f1': 'F1分数',
        'precision': '精确率', 'recall': '召回率',
        'svdd_loss': 'SVDD损失', 'vae_loss': 'VAE损失', 'total_loss': '总损失',
        'val_acc': '验证准确率', 'val_f1': '验证F1',
        'train_loss': '训练损失', 'recon_loss': '重构损失',
        'hetero': '图像分支', 'zerone': '特征分支', 'dual': '双分支融合',
        'feature': '特征', 'correlation': '相关性',
        'stage1': '阶段一：无监督学习', 'stage2': '阶段二：伪标签生成',
        'stage3': '阶段三：监督微调',
        'confusion_matrix': '混淆矩阵', 'roc_curve': 'ROC曲线',
        'pr_curve': 'PR曲线', 'score_dist': '得分分布',
        'tsne': 't-SNE可视化', 'feature_analysis': '特征分析',
        'fusion_weights': '融合权重分布',
    },
    'en': {
        'normal': 'Normal', 'fault': 'Fault', 'uncertain': 'Uncertain',
        'epoch': 'Epoch', 'loss': 'Loss', 'accuracy': 'Accuracy',
        'score': 'Anomaly Score', 'count': 'Count', 'f1': 'F1 Score',
        'precision': 'Precision', 'recall': 'Recall',
        'svdd_loss': 'SVDD Loss', 'vae_loss': 'VAE Loss', 'total_loss': 'Total Loss',
        'val_acc': 'Val Accuracy', 'val_f1': 'Val F1',
        'train_loss': 'Train Loss', 'recon_loss': 'Recon Loss',
        'hetero': 'Image Branch', 'zerone': 'Feature Branch', 'dual': 'Dual Branch',
        'feature': 'Feature', 'correlation': 'Correlation',
        'stage1': 'Stage 1: Unsupervised Learning', 'stage2': 'Stage 2: Pseudo-Label',
        'stage3': 'Stage 3: Supervised Fine-tuning',
        'confusion_matrix': 'Confusion Matrix', 'roc_curve': 'ROC Curve',
        'pr_curve': 'PR Curve', 'score_dist': 'Score Distribution',
        'tsne': 't-SNE Visualization', 'feature_analysis': 'Feature Analysis',
        'fusion_weights': 'Fusion Weight Distribution',
    }
}


# =============================================================================
# 第2步: 配置类定义
# =============================================================================
@dataclass
class ThreeStageConfigV4:
    """
    三阶段诊断系统配置类 (V4版本)
    
    【V4核心改进】
        FUSION_MODE: 融合策略 ('concat' / 'attention' / 'gate')
        STRICT_DATA_SEPARATION: 严格数据分离开关
    """
    
    # ================= 路径配置 =================
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v4"))
    
    # ================= 支线模式 =================
    BRANCH_MODE: str = "dual"  # 'hetero' / 'zerone' / 'dual'
    
    # ================= 融合策略 (V4新增) =================
    FUSION_MODE: str = "concat"  # 'concat' / 'attention' / 'gate'
    
    # ================= 数据分离原则 (V4优化) =================
    # True: 严格分离 - TRAIN无监督，VAL有监督，TEST仅评估
    STRICT_DATA_SEPARATION: bool = True
    
    # ================= 信号参数 =================
    FS: float = 8192.0          # 采样频率 (Hz)
    SIGNAL_LEN: int = 8192      # 信号长度
    INPUT_SIZE: int = 224       # CNN输入尺寸
    
    # ================= 特征维度 =================
    ZERONE_DIM: int = TOTAL_FEAT_DIM   # Zerone特征维度 (1200)
    CNN_FEAT_DIM: int = 512     # CNN输出维度
    MLP_FEAT_DIM: int = 256     # MLP输出维度
    
    # ================= 模型参数 =================
    LATENT_DIM: int = 128       # SVDD隐空间维度
    LATENT_CHANNELS: int = 64   # VAE空间隐变量通道数
    
    # ================= 训练参数 =================
    BATCH_SIZE: int = 16
    STAGE1_EPOCHS: int = 50
    STAGE3_EPOCHS: int = 30
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 15          # 早停耐心值
    
    # SVDD参数
    NU: float = 0.05            # 假设异常比例
    
    # VAE参数
    BETA_VAE: float = 0.01
    BETA_WARMUP: int = 10
    
    # ================= 伪标签阈值 =================
    NORMAL_PERCENTILE: float = 5.0
    ANOMALY_PERCENTILE: float = 99.0
    
    # ================= 检查点与可视化 =================
    CHECKPOINT_EVERY: int = 5       # 每N轮保存检查点
    MAX_CHECKPOINTS: int = 5        # 最多保留N个检查点
    VIZ_EVERY: int = 3              # 每N轮生成可视化
    SAMPLE_PREVIEW_COUNT: int = 8   # 样本预览数量
    
    # ================= 类别关键词 =================
    CLASS_KEYWORDS: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    })
    
    # ================= 设备 =================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ================= 可视化 =================
    VIZ_DPI: int = 300
    LANGS: Tuple[str, str] = ("cn", "en")  # 生成双语版本
    
    def __post_init__(self):
        """初始化后处理"""
        self.PROJECT_ROOT = Path(self.PROJECT_ROOT)
        self.OUTPUT_ROOT = Path(self.OUTPUT_ROOT)
        
        # 数据目录
        self.TRAIN_DIR = self.PROJECT_ROOT / "train"
        self.VAL_DIR = self.PROJECT_ROOT / "val"
        self.TEST_DIR = self.PROJECT_ROOT / "test"
        
        # 根据支线模式和融合策略设置输出子目录
        if self.BRANCH_MODE == 'dual':
            self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}_{self.FUSION_MODE}"
        else:
            self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}"
        
        # 输出子目录
        self.STAGE1_DIR = self.BRANCH_DIR / "stage1_unsupervised"
        self.STAGE2_DIR = self.BRANCH_DIR / "stage2_pseudo_labels"
        self.STAGE3_DIR = self.BRANCH_DIR / "stage3_supervised"
        self.MODEL_DIR = self.BRANCH_DIR / "models"
        self.CHECKPOINT_DIR = self.BRANCH_DIR / "checkpoints"
        self.LOG_DIR = self.BRANCH_DIR / "logs"
        
        # 可视化子目录
        self.VIZ_DIR = self.BRANCH_DIR / "visualizations"
        self.VIZ_SUBDIRS = {
            "training_curves": self.VIZ_DIR / "training_curves",
            "score_dist": self.VIZ_DIR / "score_dist",
            "confusion": self.VIZ_DIR / "confusion",
            "roc_pr": self.VIZ_DIR / "roc_pr",
            "tsne": self.VIZ_DIR / "tsne",
            "feature_analysis": self.VIZ_DIR / "feature_analysis",
            "recon": self.VIZ_DIR / "reconstruction",
            "svdd_sphere": self.VIZ_DIR / "svdd_sphere",
            "error_samples": self.VIZ_DIR / "error_samples",
            "sample_preview": self.VIZ_DIR / "sample_preview",
            "fusion_weights": self.VIZ_DIR / "fusion_weights",  # V4新增
            "reconstruction": self.VIZ_DIR / "reconstruction",
            "feature_analysis": self.VIZ_DIR / "feature_analysis",
        }
        
        # 创建必要目录
        for d in [self.STAGE1_DIR, self.STAGE2_DIR, self.STAGE3_DIR, 
                  self.MODEL_DIR, self.CHECKPOINT_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        for subdir in self.VIZ_SUBDIRS.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("三阶段故障诊断系统配置 (V4版本)")
        print("="*70)
        print(f"【支线模式】")
        branch_names = {'hetero': '图像分支(Hetero)', 'zerone': '特征分支(Zerone)', 'dual': '双分支融合'}
        print(f"  当前支线: {branch_names.get(self.BRANCH_MODE, self.BRANCH_MODE)}")
        if self.BRANCH_MODE == 'dual':
            fusion_names = {'concat': '等权拼接', 'attention': '注意力加权', 'gate': '交叉门控'}
            print(f"  融合策略: {fusion_names.get(self.FUSION_MODE, self.FUSION_MODE)}")
        print(f"【数据路径】")
        print(f"  项目根目录: {self.PROJECT_ROOT}")
        print(f"  输出目录: {self.BRANCH_DIR}")
        print(f"【数据分离规则 (V4严格模式)】")
        if self.STRICT_DATA_SEPARATION:
            print(f"  ✅ TRAIN: 完全无监督（阶段1/2）")
            print(f"  ✅ VAL: 有标签监督微调（阶段3）")
            print(f"  ✅ TEST: 仅最终评估（不参与训练）")
        else:
            print(f"  ⚠️ 宽松模式（兼容V3行为）")
        print(f"【训练参数】")
        print(f"  设备: {self.DEVICE}")
        print(f"  批大小: {self.BATCH_SIZE}")
        print(f"  阶段一轮数: {self.STAGE1_EPOCHS}")
        print(f"  阶段三轮数: {self.STAGE3_EPOCHS}")
        print(f"【监控设置】")
        print(f"  检查点间隔: 每{self.CHECKPOINT_EVERY}轮")
        print(f"  可视化间隔: 每{self.VIZ_EVERY}轮")
        print(f"  输出语言: 中文 + 英文")
        print("="*70 + "\n")


# =============================================================================
# 第3步: 融合模块定义 (V4核心新增)
# =============================================================================

class ConcatFusion(nn.Module):
    """
    等权拼接融合 (Baseline)
    
    将两个分支特征直接拼接后通过全连接层融合。
    输入: h_img (B, 512), h_feat (B, 256)
    输出: h_fused (B, 512)
    """
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.out_dim = out_dim
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        返回:
            dict: {
                'h_fused': 融合后特征,
                'weights': None (无权重)
            }
        """
        h_cat = torch.cat([h_img, h_feat], dim=1)
        h_fused = self.fusion(h_cat)
        return {'h_fused': h_fused, 'weights': None}


class AttentionFusion(nn.Module):
    """
    注意力加权融合
    
    通过注意力机制计算两个分支的动态权重，实现自适应融合。
    公式: h_fused = α·h_img' + β·h_feat'
    其中 α, β 通过 softmax(MLP([h_img; h_feat])) 得到
    
    参考文献:
        [1] Wang et al. (2023). Dual-Branch Network with Hybrid Attention
    """
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        
        # 将两个分支投影到统一维度
        self.proj_img = nn.Sequential(
            nn.Linear(img_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.proj_feat = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
        # 门控MLP: 输入拼接特征，输出2个权重分数
        # Gating MLP: input concatenated features, output 2 weight scores
        self.gating_mlp = nn.Sequential(
            nn.Linear(img_dim + feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出 [score_img, score_feat]
        )
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        返回:
            dict: {
                'h_fused': 融合后特征 (B, out_dim),
                'weights': 分支权重 (B, 2) - [α, β]
            }
        """
        # 计算融合权重
        # Calculate fusion weights via attention mechanism
        concat_feat = torch.cat([h_img, h_feat], dim=1)
        scores = self.gating_mlp(concat_feat)  # (B, 2)
        weights = torch.softmax(scores, dim=1)  # α, β ∈ [0,1], α + β = 1
        
        # 投影到统一维度
        h_img_proj = self.proj_img(h_img)    # (B, out_dim)
        h_feat_proj = self.proj_feat(h_feat)  # (B, out_dim)
        
        # 加权融合
        # Weighted fusion: h_fused = α·h_img' + β·h_feat'
        alpha = weights[:, 0:1]  # (B, 1)
        beta = weights[:, 1:2]   # (B, 1)
        h_fused = alpha * h_img_proj + beta * h_feat_proj
        
        return {'h_fused': h_fused, 'weights': weights}


class GatedFusion(nn.Module):
    """
    交叉门控融合
    
    通过交叉门控机制使两个分支特征相互调制，实现信息交换。
    公式:
        g_img = σ(W_feat · h_feat)
        g_feat = σ(W_img · h_img)
        h_img' = h_img ⊙ g_img
        h_feat' = h_feat ⊙ g_feat
        h_fused = [h_img'; h_feat'] → FC → out_dim
    
    参考文献:
        [2] Wang et al. (2025). Cross-Attention Gating Mechanism
    """
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        
        # 交叉门控网络
        # Cross-gating networks: each branch generates gate for the other
        self.gate_for_img = nn.Sequential(
            nn.Linear(feat_dim, img_dim),
            nn.Sigmoid()
        )
        self.gate_for_feat = nn.Sequential(
            nn.Linear(img_dim, feat_dim),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        返回:
            dict: {
                'h_fused': 融合后特征 (B, out_dim),
                'weights': 门控权重 dict {'g_img': (B, img_dim), 'g_feat': (B, feat_dim)}
            }
        """
        # 计算交叉门控权重
        # Calculate cross-gating weights
        g_img = self.gate_for_img(h_feat)    # 用feat信息调制img: (B, img_dim)
        g_feat = self.gate_for_feat(h_img)    # 用img信息调制feat: (B, feat_dim)
        
        # 应用门控
        # Apply gating: element-wise multiplication
        h_img_gated = h_img * g_img      # (B, img_dim)
        h_feat_gated = h_feat * g_feat   # (B, feat_dim)
        
        # 拼接并融合
        h_cat = torch.cat([h_img_gated, h_feat_gated], dim=1)
        h_fused = self.fusion(h_cat)
        
        # 返回门控统计信息用于可视化
        weights = {
            'g_img_mean': g_img.mean(dim=1),   # (B,)
            'g_feat_mean': g_feat.mean(dim=1), # (B,)
        }
        
        return {'h_fused': h_fused, 'weights': weights}


# =============================================================================
# 第4步: 可视化工具类
# =============================================================================

class VisualizationManager:
    """
    可视化管理器
    
    负责生成中英文双版本的高质量图片 (IEEE/Nature风格)
    V4新增: 融合权重可视化
    """
    
    def __init__(self, cfg: ThreeStageConfigV4):
        self.cfg = cfg
        self.setup_style()
        self.fusion_weights_history = []  # 记录融合权重
    
    def setup_style(self):
        """设置IEEE/Nature风格"""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 150,
            'savefig.dpi': self.cfg.VIZ_DPI,
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12,
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': ':',
        })
    
    def get_label(self, key: str, lang: str = 'cn') -> str:
        """获取指定语言的标签"""
        return LABELS.get(lang, LABELS['en']).get(key, key)
    
    def record_fusion_weights(self, weights: torch.Tensor, labels: torch.Tensor = None):
        """
        记录融合权重用于后续可视化
        
        参数:
            weights: (B, 2) 注意力权重 或 dict
            labels: (B,) 样本标签
        """
        if weights is None:
            return
        
        if isinstance(weights, dict):
            # GatedFusion 返回的是字典
            record = {
                'g_img_mean': weights['g_img_mean'].detach().cpu().numpy(),
                'g_feat_mean': weights['g_feat_mean'].detach().cpu().numpy(),
                'labels': labels.cpu().numpy() if labels is not None else None
            }
        else:
            # AttentionFusion 返回的是张量
            record = {
                'weights': weights.detach().cpu().numpy(),
                'labels': labels.cpu().numpy() if labels is not None else None
            }
        self.fusion_weights_history.append(record)
    
    def plot_fusion_weights(self, lang: str = 'cn'):
        """
        绘制融合权重分布图 (V4新增)
        
        展示注意力/门控权重在不同类别样本上的分布差异
        """
        if not self.fusion_weights_history:
            return None
        
        L = LABELS[lang]
        
        # 检查是注意力模式还是门控模式
        sample = self.fusion_weights_history[0]
        
        if 'weights' in sample:
            # AttentionFusion模式
            all_weights = np.concatenate([r['weights'] for r in self.fusion_weights_history], axis=0)
            all_labels = np.concatenate([r['labels'] for r in self.fusion_weights_history 
                                         if r['labels'] is not None], axis=0) if any(r['labels'] is not None for r in self.fusion_weights_history) else None
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 图1: 权重分布直方图
            ax = axes[0]
            if all_labels is not None:
                normal_weights = all_weights[all_labels == 0]
                fault_weights = all_weights[all_labels == 1]
                
                ax.hist(normal_weights[:, 0], bins=30, alpha=0.6, color=COLORS['normal'], 
                       label=f'{L["normal"]} (α_img)', density=True)
                ax.hist(fault_weights[:, 0], bins=30, alpha=0.6, color=COLORS['fault'],
                       label=f'{L["fault"]} (α_img)', density=True)
            else:
                ax.hist(all_weights[:, 0], bins=50, alpha=0.7, color=COLORS['blue'], density=True)
            
            ax.set_xlabel('Image Branch Weight (α)' if lang == 'en' else '图像分支权重 (α)')
            ax.set_ylabel('Density' if lang == 'en' else '密度')
            ax.set_title('Attention Weight Distribution' if lang == 'en' else '注意力权重分布')
            ax.legend()
            
            # 图2: 权重散点图
            ax = axes[1]
            if all_labels is not None:
                for label, color, name in [(0, COLORS['normal'], L['normal']), 
                                           (1, COLORS['fault'], L['fault'])]:
                    mask = all_labels == label
                    ax.scatter(all_weights[mask, 0], all_weights[mask, 1], 
                              c=color, label=name, alpha=0.5, s=20)
            else:
                ax.scatter(all_weights[:, 0], all_weights[:, 1], alpha=0.5, s=20)
            
            ax.set_xlabel('α (Image)' if lang == 'en' else 'α (图像)')
            ax.set_ylabel('β (Feature)' if lang == 'en' else 'β (特征)')
            ax.set_title('Weight Scatter' if lang == 'en' else '权重散点')
            ax.plot([0, 1], [1, 0], 'k--', alpha=0.3)  # α + β = 1 参考线
            ax.legend()
            
        else:
            # GatedFusion模式
            all_g_img = np.concatenate([r['g_img_mean'] for r in self.fusion_weights_history], axis=0)
            all_g_feat = np.concatenate([r['g_feat_mean'] for r in self.fusion_weights_history], axis=0)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 图1: 门控激活分布
            ax = axes[0]
            ax.hist(all_g_img, bins=50, alpha=0.6, color=COLORS['blue'], 
                   label='Gate for Image', density=True)
            ax.hist(all_g_feat, bins=50, alpha=0.6, color=COLORS['orange'],
                   label='Gate for Feature', density=True)
            ax.set_xlabel('Gate Activation' if lang == 'en' else '门控激活')
            ax.set_ylabel('Density' if lang == 'en' else '密度')
            ax.set_title('Gating Distribution' if lang == 'en' else '门控分布')
            ax.legend()
            
            # 图2: 门控散点图
            ax = axes[1]
            ax.scatter(all_g_img, all_g_feat, alpha=0.3, s=10, c=COLORS['purple'])
            ax.set_xlabel('Gate for Image' if lang == 'en' else '图像门控')
            ax.set_ylabel('Gate for Feature' if lang == 'en' else '特征门控')
            ax.set_title('Cross-Gating Scatter' if lang == 'en' else '交叉门控散点')
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["fusion_weights"] / f"fusion_weights_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return save_path
    
    # =============================================================================
# 【补充1】添加到 VisualizationManager 类中的新方法
# =============================================================================

    def plot_hetero_reconstruction(self, model, dataloader, device, n_samples: int = 6, lang: str = 'cn'):
        """
        Hetero图像分支重建可视化
        
        展示VAE对3通道图像的重建效果：
        - 第1行: 原始图像 (CWT / STFT / Context)
        - 第2行: 重建图像
        - 第3行: 残差图 (差异热力图)
        
        参数:
            model: AnomalyModelV4 模型
            dataloader: 数据加载器
            device: 计算设备
            n_samples: 展示样本数
            lang: 'cn' 或 'en'
        """
        model.eval()
        
        # 收集样本
        samples = []
        with torch.no_grad():
            for batch in dataloader:
                img, zr, label, idx = batch
                img = img.to(device)
                zr = zr.to(device)
                
                # 前向传播获取重建
                out = model(img, zr)
                
                for i in range(min(len(img), n_samples - len(samples))):
                    samples.append({
                        'original': img[i].cpu().numpy(),
                        'recon': out['recon'][i].cpu().numpy() if 'recon' in out else None,
                        'label': label[i].item(),
                        'idx': idx[i].item()
                    })
                
                if len(samples) >= n_samples:
                    break
        
        if not samples or samples[0]['recon'] is None:
            print(f"  [跳过] Hetero重建可视化 - 无VAE重建输出")
            return
        
        # 通道名称
        channel_names_cn = ['CWT (小波)', 'STFT (频谱)', 'Context (波形)']
        channel_names_en = ['CWT (Wavelet)', 'STFT (Spectrum)', 'Context (Waveform)']
        channel_names = channel_names_cn if lang == 'cn' else channel_names_en
        
        # 创建大图: 每个样本3行(原始/重建/残差) x 3列(3通道)
        fig, axes = plt.subplots(3 * n_samples, 3, figsize=(12, 4 * n_samples))
        
        title = '三通道图像重建效果' if lang == 'cn' else 'Three-Channel Image Reconstruction'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        row_labels_cn = ['原始', '重建', '残差']
        row_labels_en = ['Original', 'Reconstructed', 'Residual']
        row_labels = row_labels_cn if lang == 'cn' else row_labels_en
        
        for s_idx, sample in enumerate(samples):
            orig = sample['original']  # (3, 224, 224)
            recon = sample['recon']    # (3, 224, 224)
            label = sample['label']
            
            # 归一化到 [0, 1] 用于显示
            orig_norm = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
            recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
            residual = np.abs(orig_norm - recon_norm)
            
            base_row = s_idx * 3
            
            for ch in range(3):
                # 原始图像
                ax_orig = axes[base_row, ch]
                ax_orig.imshow(orig_norm[ch], cmap='viridis', aspect='auto')
                if s_idx == 0:
                    ax_orig.set_title(channel_names[ch], fontsize=10)
                if ch == 0:
                    label_text = '正常' if label == 0 else '故障'
                    label_text_en = 'Normal' if label == 0 else 'Fault'
                    ax_orig.set_ylabel(f"样本{s_idx+1} ({label_text if lang=='cn' else label_text_en})\n{row_labels[0]}", 
                                    fontsize=9)
                ax_orig.set_xticks([])
                ax_orig.set_yticks([])
                
                # 重建图像
                ax_recon = axes[base_row + 1, ch]
                ax_recon.imshow(recon_norm[ch], cmap='viridis', aspect='auto')
                if ch == 0:
                    ax_recon.set_ylabel(row_labels[1], fontsize=9)
                ax_recon.set_xticks([])
                ax_recon.set_yticks([])
                
                # 残差热力图
                ax_res = axes[base_row + 2, ch]
                im = ax_res.imshow(residual[ch], cmap='hot', aspect='auto', vmin=0, vmax=0.5)
                if ch == 0:
                    ax_res.set_ylabel(row_labels[2], fontsize=9)
                ax_res.set_xticks([])
                ax_res.set_yticks([])
        
        # 添加colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar_label = '重建误差' if lang == 'cn' else 'Reconstruction Error'
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        save_dir = self.cfg.VIZ_SUBDIRS.get("reconstruction", self.cfg.VIZ_DIR / "reconstruction")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"hetero_reconstruction_{lang}.png"
        fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [保存] {save_dir / filename}")


    def plot_hetero_channel_analysis(self, dataloader, device, n_samples: int = 100, lang: str = 'cn'):
        """
        Hetero三通道统计分析
        
        展示:
        - 各通道的像素强度分布 (正常 vs 故障)
        - 通道间相关性热力图
        - 空间激活热力图 (平均)
        """
        # 收集数据
        all_images = {'normal': [], 'fault': []}
        
        count = 0
        for batch in dataloader:
            img, _, label, _ = batch
            for i in range(len(img)):
                key = 'normal' if label[i].item() == 0 else 'fault'
                all_images[key].append(img[i].numpy())
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
        
        channel_names_cn = ['CWT (小波)', 'STFT (频谱)', 'Context (波形)']
        channel_names_en = ['CWT (Wavelet)', 'STFT (Spectrum)', 'Context (Waveform)']
        channel_names = channel_names_cn if lang == 'cn' else channel_names_en
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        title = '三通道图像特征分析' if lang == 'cn' else 'Three-Channel Image Feature Analysis'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        colors = {'normal': '#2ecc71', 'fault': '#e74c3c'}
        labels_cn = {'normal': '正常', 'fault': '故障'}
        labels_en = {'normal': 'Normal', 'fault': 'Fault'}
        labels = labels_cn if lang == 'cn' else labels_en
        
        # 第1行: 各通道像素强度分布
        for ch in range(3):
            ax = axes[0, ch]
            for key in ['normal', 'fault']:
                if all_images[key]:
                    data = np.array(all_images[key])[:, ch, :, :].flatten()
                    ax.hist(data, bins=50, alpha=0.6, color=colors[key], label=labels[key], density=True)
            
            ax.set_title(channel_names[ch], fontsize=11)
            ax.set_xlabel('像素强度' if lang == 'cn' else 'Pixel Intensity', fontsize=9)
            ax.set_ylabel('密度' if lang == 'cn' else 'Density', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 第2行: 空间平均激活图 (正常 vs 故障)
        for idx, key in enumerate(['normal', 'fault']):
            ax = axes[1, idx]
            if all_images[key]:
                data = np.array(all_images[key])  # (N, 3, 224, 224)
                # 计算所有通道的平均激活
                mean_activation = data.mean(axis=(0, 1))  # (224, 224)
                im = ax.imshow(mean_activation, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            title_text = f"空间平均激活 ({labels[key]})" if lang == 'cn' else f"Spatial Mean Activation ({labels[key]})"
            ax.set_title(title_text, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 第2行第3列: 通道相关性
        ax = axes[1, 2]
        if all_images['normal'] or all_images['fault']:
            all_data = all_images['normal'] + all_images['fault']
            data = np.array(all_data)  # (N, 3, 224, 224)
            # 计算通道间相关性
            ch_data = data.reshape(len(data), 3, -1)  # (N, 3, 224*224)
            ch_means = ch_data.mean(axis=2)  # (N, 3)
            corr_matrix = np.corrcoef(ch_means.T)  # (3, 3)
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            short_names = ['CWT', 'STFT', 'Ctx']
            ax.set_xticklabels(short_names, fontsize=9)
            ax.set_yticklabels(short_names, fontsize=9)
            
            # 添加数值标注
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=10)
        
        title_text = '通道相关性' if lang == 'cn' else 'Channel Correlation'
        ax.set_title(title_text, fontsize=11)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_dir = self.cfg.VIZ_SUBDIRS.get("reconstruction", self.cfg.VIZ_DIR / "reconstruction")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"hetero_channel_analysis_{lang}.png"
        fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [保存] {save_dir / filename}")


    def plot_zerone_feature_analysis(self, dataloader, device, n_samples: int = 200, lang: str = 'cn'):
        """
        Zerone特征分支分析可视化
        
        展示1200维特征的:
        - 特征段分布 (时域15 + STFT127 + PSD1050 + 高频8)
        - 正常 vs 故障的特征差异热力图
        - Top-K重要特征柱状图
        - 特征相关性矩阵 (分段)
        """
        # 收集数据
        features = {'normal': [], 'fault': []}
        
        count = 0
        for batch in dataloader:
            _, zr, label, _ = batch
            for i in range(len(zr)):
                key = 'normal' if label[i].item() == 0 else 'fault'
                features[key].append(zr[i].numpy())
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
        
        # 特征段定义
        segments = {
            'time': (0, 15, '时域特征' if lang == 'cn' else 'Time Domain'),
            'stft': (15, 142, 'STFT特征' if lang == 'cn' else 'STFT Features'),
            'psd': (142, 1192, 'PSD特征' if lang == 'cn' else 'PSD Features'),
            'hf': (1192, 1200, '高频特征' if lang == 'cn' else 'High-Freq Features')
        }
        
        fig = plt.figure(figsize=(16, 12))
        
        title = 'Zerone 1200维特征分析' if lang == 'cn' else 'Zerone 1200-D Feature Analysis'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        colors = {'normal': '#2ecc71', 'fault': '#e74c3c'}
        labels_text = {'normal': '正常' if lang == 'cn' else 'Normal', 
                    'fault': '故障' if lang == 'cn' else 'Fault'}
        
        # 子图1: 特征段均值对比 (2x2第1个)
        ax1 = fig.add_subplot(2, 2, 1)
        
        seg_names = []
        normal_means = []
        fault_means = []
        
        for seg_name, (start, end, label) in segments.items():
            seg_names.append(label)
            if features['normal']:
                normal_means.append(np.mean([f[start:end].mean() for f in features['normal']]))
            else:
                normal_means.append(0)
            if features['fault']:
                fault_means.append(np.mean([f[start:end].mean() for f in features['fault']]))
            else:
                fault_means.append(0)
        
        x = np.arange(len(seg_names))
        width = 0.35
        ax1.bar(x - width/2, normal_means, width, label=labels_text['normal'], color=colors['normal'])
        ax1.bar(x + width/2, fault_means, width, label=labels_text['fault'], color=colors['fault'])
        ax1.set_xticks(x)
        ax1.set_xticklabels(seg_names, fontsize=9)
        ax1.set_ylabel('平均值' if lang == 'cn' else 'Mean Value', fontsize=10)
        ax1.set_title('各段特征均值' if lang == 'cn' else 'Feature Segment Means', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2: 特征差异热力图 (2x2第2个)
        ax2 = fig.add_subplot(2, 2, 2)
        
        if features['normal'] and features['fault']:
            normal_mean = np.mean(features['normal'], axis=0)
            fault_mean = np.mean(features['fault'], axis=0)
            diff = fault_mean - normal_mean
            
            # 重塑为2D便于可视化 (40x30 = 1200)
            diff_2d = diff.reshape(40, 30)
            im = ax2.imshow(diff_2d, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, ax=ax2, fraction=0.046)
            
            ax2.set_xlabel('特征索引 (mod 30)' if lang == 'cn' else 'Feature Index (mod 30)', fontsize=9)
            ax2.set_ylabel('特征索引 (// 30)' if lang == 'cn' else 'Feature Index (// 30)', fontsize=9)
        
        ax2.set_title('故障-正常 特征差异' if lang == 'cn' else 'Fault-Normal Feature Difference', fontsize=11)
        
        # 子图3: Top-20 差异最大的特征 (2x2第3个)
        ax3 = fig.add_subplot(2, 2, 3)
        
        if features['normal'] and features['fault']:
            normal_mean = np.mean(features['normal'], axis=0)
            fault_mean = np.mean(features['fault'], axis=0)
            normal_std = np.std(features['normal'], axis=0) + 1e-8
            
            # 计算标准化差异
            z_diff = np.abs(fault_mean - normal_mean) / normal_std
            top_k = 20
            top_indices = np.argsort(z_diff)[-top_k:][::-1]
            top_values = z_diff[top_indices]
            
            # 标注属于哪个段
            def get_segment(idx):
                for seg_name, (start, end, _) in segments.items():
                    if start <= idx < end:
                        return seg_name
                return 'unknown'
            
            segment_colors = {'time': '#3498db', 'stft': '#9b59b6', 'psd': '#f39c12', 'hf': '#1abc9c'}
            bar_colors = [segment_colors[get_segment(i)] for i in top_indices]
            
            ax3.barh(range(top_k), top_values, color=bar_colors)
            ax3.set_yticks(range(top_k))
            ax3.set_yticklabels([f'F{i}' for i in top_indices], fontsize=8)
            ax3.set_xlabel('Z-Score 差异' if lang == 'cn' else 'Z-Score Difference', fontsize=10)
            ax3.set_title(f'Top-{top_k} 差异特征' if lang == 'cn' else f'Top-{top_k} Discriminative Features', fontsize=11)
            ax3.invert_yaxis()
            
            # 图例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=segment_colors[seg], label=segments[seg][2]) 
                            for seg in ['time', 'stft', 'psd', 'hf']]
            ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        # 子图4: 特征分布小提琴图 (选取关键段) (2x2第4个)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 选取每个段的代表性特征
        repr_indices = [0, 7, 14,  # 时域
                        20, 80, 140,  # STFT
                        200, 600, 1000,  # PSD
                        1195, 1198]  # 高频
        
        if features['normal'] and features['fault']:
            positions = []
            data_normal = []
            data_fault = []
            
            for i, idx in enumerate(repr_indices):
                if idx < 1200:
                    positions.append(i)
                    data_normal.append([f[idx] for f in features['normal']])
                    data_fault.append([f[idx] for f in features['fault']])
            
            # 绘制箱线图
            bp1 = ax4.boxplot(data_normal, positions=np.array(positions) - 0.2, widths=0.35,
                            patch_artist=True, boxprops=dict(facecolor=colors['normal'], alpha=0.7))
            bp2 = ax4.boxplot(data_fault, positions=np.array(positions) + 0.2, widths=0.35,
                            patch_artist=True, boxprops=dict(facecolor=colors['fault'], alpha=0.7))
            
            ax4.set_xticks(positions)
            ax4.set_xticklabels([f'F{i}' for i in repr_indices], fontsize=8, rotation=45)
            ax4.set_ylabel('特征值' if lang == 'cn' else 'Feature Value', fontsize=10)
            ax4.set_title('代表性特征分布' if lang == 'cn' else 'Representative Feature Distribution', fontsize=11)
            ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], 
                    [labels_text['normal'], labels_text['fault']], fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_dir = self.cfg.VIZ_SUBDIRS.get("feature_analysis", self.cfg.VIZ_DIR / "feature_analysis")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"zerone_feature_analysis_{lang}.png"
        fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [保存] {save_dir / filename}")

    def plot_dual_fusion_samples(self, model, classifier, dataloader, device, n_samples: int = 6, lang: str = 'cn'):
        """
        Dual分支融合样本可视化
        
        展示融合过程:
        - 左: Hetero 3通道图像
        - 中: Zerone 特征条形图
        - 右: 融合权重 + 预测结果
        
        参数:
            model: AnomalyModelV4 (编码器)
            classifier: FaultClassifierV4 (分类器)
            dataloader: 数据加载器
            device: 计算设备
            n_samples: 展示样本数
            lang: 'cn' 或 'en'
        """
        if self.cfg.BRANCH_MODE != 'dual':
            print(f"  [跳过] Dual融合可视化 - 当前模式: {self.cfg.BRANCH_MODE}")
            return
        
        classifier.eval()
        
        # 收集样本
        samples = []
        with torch.no_grad():
            for batch in dataloader:
                img, zr, label, idx = batch
                img, zr = img.to(device), zr.to(device)
                
                result = classifier(img, zr)
                probs = torch.softmax(result['logits'], dim=1)
                
                for i in range(min(len(img), n_samples - len(samples))):
                    sample_data = {
                        'image': img[i].cpu().numpy(),
                        'zerone': zr[i].cpu().numpy(),
                        'label': label[i].item(),
                        'pred': result['logits'][i].argmax().item(),
                        'prob': probs[i, 1].item(),  # 故障概率
                        'fusion_weights': None
                    }
                    
                    # 获取融合权重
                    if 'fusion_weights' in result and result['fusion_weights'] is not None:
                        weights = result['fusion_weights']
                        if isinstance(weights, torch.Tensor):
                            sample_data['fusion_weights'] = weights[i].cpu().numpy()
                        elif isinstance(weights, dict):
                            sample_data['fusion_weights'] = {
                                'g_img': weights['g_img_mean'][i].item() if 'g_img_mean' in weights else 0.5,
                                'g_feat': weights['g_feat_mean'][i].item() if 'g_feat_mean' in weights else 0.5
                            }
                    
                    samples.append(sample_data)
                
                if len(samples) >= n_samples:
                    break
        
        # 创建可视化
        fig = plt.figure(figsize=(18, 4 * n_samples))
        
        title = f'双分支融合样本分析 ({self.cfg.FUSION_MODE})' if lang == 'cn' else f'Dual-Branch Fusion Sample Analysis ({self.cfg.FUSION_MODE})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        channel_names_cn = ['CWT', 'STFT', 'Context']
        channel_names_en = ['CWT', 'STFT', 'Context']
        channel_names = channel_names_cn if lang == 'cn' else channel_names_en
        
        for s_idx, sample in enumerate(samples):
            # === 左侧: Hetero 3通道图像 (3个子图) ===
            for ch in range(3):
                ax_img = fig.add_subplot(n_samples, 6, s_idx * 6 + ch + 1)
                
                img_ch = sample['image'][ch]
                img_norm = (img_ch - img_ch.min()) / (img_ch.max() - img_ch.min() + 1e-8)
                ax_img.imshow(img_norm, cmap='viridis', aspect='auto')
                
                if s_idx == 0:
                    ax_img.set_title(channel_names[ch], fontsize=10)
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                
                if ch == 0:
                    label_text = '正常' if sample['label'] == 0 else '故障'
                    label_en = 'Normal' if sample['label'] == 0 else 'Fault'
                    ax_img.set_ylabel(f"样本{s_idx+1}\n({label_text if lang=='cn' else label_en})", fontsize=9)
            
            # === 中间: Zerone 特征概览 ===
            ax_zr = fig.add_subplot(n_samples, 6, s_idx * 6 + 4)
            
            zerone = sample['zerone']
            # 分段显示
            seg_means = [
                zerone[:15].mean(),      # 时域
                zerone[15:142].mean(),   # STFT
                zerone[142:1192].mean(), # PSD
                zerone[1192:].mean()     # 高频
            ]
            seg_names_cn = ['时域', 'STFT', 'PSD', '高频']
            seg_names_en = ['Time', 'STFT', 'PSD', 'HF']
            seg_names = seg_names_cn if lang == 'cn' else seg_names_en
            seg_colors = ['#3498db', '#9b59b6', '#f39c12', '#1abc9c']
            
            ax_zr.barh(seg_names, seg_means, color=seg_colors)
            ax_zr.set_xlabel('均值' if lang == 'cn' else 'Mean', fontsize=9)
            if s_idx == 0:
                ax_zr.set_title('Zerone特征' if lang == 'cn' else 'Zerone Features', fontsize=10)
            ax_zr.grid(True, alpha=0.3, axis='x')
            
            # === 右侧: 融合权重 + 预测 ===
            ax_fusion = fig.add_subplot(n_samples, 6, s_idx * 6 + 5)
            
            # 融合权重可视化
            if sample['fusion_weights'] is not None:
                if isinstance(sample['fusion_weights'], np.ndarray):
                    # Attention模式: [α, β]
                    weights = sample['fusion_weights']
                    labels_w = ['α (Hetero)', 'β (Zerone)']
                    colors_w = ['#3498db', '#e74c3c']
                    ax_fusion.bar(labels_w, weights, color=colors_w)
                    ax_fusion.set_ylim(0, 1)
                    for i, v in enumerate(weights):
                        ax_fusion.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
                elif isinstance(sample['fusion_weights'], dict):
                    # Gate模式: g_img, g_feat
                    weights = sample['fusion_weights']
                    labels_w = ['g_img', 'g_feat']
                    values = [weights.get('g_img', 0.5), weights.get('g_feat', 0.5)]
                    colors_w = ['#3498db', '#e74c3c']
                    ax_fusion.bar(labels_w, values, color=colors_w)
                    ax_fusion.set_ylim(0, 1)
                    for i, v in enumerate(values):
                        ax_fusion.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
            else:
                ax_fusion.text(0.5, 0.5, 'Concat\n(等权)', ha='center', va='center', fontsize=12,
                            transform=ax_fusion.transAxes)
                ax_fusion.set_xlim(0, 1)
                ax_fusion.set_ylim(0, 1)
            
            if s_idx == 0:
                title_w = '融合权重' if lang == 'cn' else 'Fusion Weights'
                ax_fusion.set_title(title_w, fontsize=10)
            ax_fusion.grid(True, alpha=0.3, axis='y')
            
            # === 最右侧: 预测结果 ===
            ax_pred = fig.add_subplot(n_samples, 6, s_idx * 6 + 6)
            
            pred_text = '故障' if sample['pred'] == 1 else '正常'
            pred_en = 'Fault' if sample['pred'] == 1 else 'Normal'
            true_text = '故障' if sample['label'] == 1 else '正常'
            true_en = 'Fault' if sample['label'] == 1 else 'Normal'
            
            is_correct = sample['pred'] == sample['label']
            bg_color = '#d4edda' if is_correct else '#f8d7da'
            text_color = '#155724' if is_correct else '#721c24'
            
            ax_pred.set_facecolor(bg_color)
            
            if lang == 'cn':
                text = f"预测: {pred_text}\n真实: {true_text}\n置信度: {sample['prob']:.1%}"
            else:
                text = f"Pred: {pred_en}\nTrue: {true_en}\nConf: {sample['prob']:.1%}"
            
            ax_pred.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                        color=text_color, fontweight='bold', transform=ax_pred.transAxes)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            
            if s_idx == 0:
                ax_pred.set_title('预测结果' if lang == 'cn' else 'Prediction', fontsize=10)
            
            # 添加边框
            for spine in ax_pred.spines.values():
                spine.set_color(text_color)
                spine.set_linewidth(2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_dir = self.cfg.VIZ_SUBDIRS.get("fusion_weights", self.cfg.VIZ_DIR / "fusion_weights")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"dual_fusion_samples_{self.cfg.FUSION_MODE}_{lang}.png"
        fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [保存] {save_dir / filename}")

    def plot_training_curves(self, history: Dict, stage: str, lang: str = 'cn'):
            """绘制训练曲线"""
            L = LABELS[lang]
            
            if stage == "stage1":
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                # SVDD损失
                axes[0].plot(history['epoch'], history['svdd_loss'], 
                            color=COLORS['blue'], lw=1.5, label=L['svdd_loss'])
                axes[0].set_xlabel(L['epoch'])
                axes[0].set_ylabel(L['loss'])
                axes[0].set_title(L['svdd_loss'])
                axes[0].legend()
                
                # VAE损失
                axes[1].plot(history['epoch'], history['vae_loss'],
                            color=COLORS['orange'], lw=1.5, label=L['vae_loss'])
                if 'recon_loss' in history:
                    axes[1].plot(history['epoch'], history['recon_loss'],
                                color=COLORS['green'], lw=1.5, ls='--', label=L['recon_loss'])
                axes[1].set_xlabel(L['epoch'])
                axes[1].set_ylabel(L['loss'])
                axes[1].set_title(L['vae_loss'])
                axes[1].legend()
                
                # 总损失
                axes[2].plot(history['epoch'], history['total_loss'],
                            color=COLORS['purple'], lw=1.5, label=L['total_loss'])
                axes[2].set_xlabel(L['epoch'])
                axes[2].set_ylabel(L['loss'])
                axes[2].set_title(L['total_loss'])
                axes[2].legend()
                
            else:  # stage3
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                
                # 训练损失
                axes[0].plot(history['epoch'], history['train_loss'],
                            color=COLORS['blue'], lw=1.5, label=L['train_loss'])
                axes[0].set_xlabel(L['epoch'])
                axes[0].set_ylabel(L['loss'])
                axes[0].set_title(L['train_loss'])
                axes[0].legend()
                
                # 验证指标
                axes[1].plot(history['epoch'], history['val_acc'],
                            color=COLORS['green'], lw=1.5, label=L['val_acc'])
                axes[1].plot(history['epoch'], history['val_f1'],
                            color=COLORS['orange'], lw=1.5, ls='--', label=L['val_f1'])
                axes[1].set_xlabel(L['epoch'])
                axes[1].set_ylabel(L['accuracy'])
                axes[1].set_title(f"{L['val_acc']} & {L['val_f1']}")
                axes[1].legend()
            
            plt.tight_layout()
            
            save_path = self.cfg.VIZ_SUBDIRS["training_curves"] / f"{stage}_curves_{lang}.png"
            fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
            plt.close(fig)
            return save_path
    
    def plot_score_distribution(self, scores: np.ndarray, t_normal: float, 
                                t_anomaly: float, labels: np.ndarray = None, lang: str = 'cn'):
        """绘制异常得分分布"""
        L = LABELS[lang]
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if labels is not None and len(labels) == len(scores):
            normal_scores = scores[labels == 0]
            fault_scores = scores[labels == 1]
            
            ax.hist(normal_scores, bins=50, alpha=0.6, color=COLORS['normal'], 
                   edgecolor='black', lw=0.5, label=L['normal'])
            ax.hist(fault_scores, bins=50, alpha=0.6, color=COLORS['fault'],
                   edgecolor='black', lw=0.5, label=L['fault'])
            ax.legend()
        else:
            ax.hist(scores, bins=100, alpha=0.7, color=COLORS['blue'],
                   edgecolor='black', lw=0.5)
        
        ax.axvline(t_normal, color=COLORS['green'], ls='--', lw=2,
                  label=f'{L["normal"]} ({t_normal:.3f})')
        ax.axvline(t_anomaly, color=COLORS['red'], ls='--', lw=2,
                  label=f'{L["fault"]} ({t_anomaly:.3f})')
        
        ax.set_xlabel(L['score'])
        ax.set_ylabel(L['count'])
        ax.set_title(L['score_dist'])
        ax.legend()
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["score_dist"] / f"score_distribution_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, lang: str = 'cn'):
        """绘制混淆矩阵"""
        L = LABELS[lang]
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        classes = [L['normal'], L['fault']]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Predicted' if lang == 'en' else '预测标签')
        ax.set_ylabel('True' if lang == 'en' else '真实标签')
        ax.set_title(L['confusion_matrix'])
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["confusion"] / f"confusion_matrix_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_roc_pr_curves(self, y_true: np.ndarray, y_scores: np.ndarray, lang: str = 'cn'):
        """绘制ROC和PR曲线"""
        L = LABELS[lang]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color=COLORS['blue'], lw=2,
                    label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], color=COLORS['gray'], lw=1, ls='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate' if lang == 'en' else '假阳性率')
        axes[0].set_ylabel('True Positive Rate' if lang == 'en' else '真阳性率')
        axes[0].set_title(L['roc_curve'])
        axes[0].legend(loc="lower right")
        
        # PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        axes[1].plot(recall, precision, color=COLORS['orange'], lw=2,
                    label=f'AP = {ap:.3f}')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall' if lang == 'en' else '召回率')
        axes[1].set_ylabel('Precision' if lang == 'en' else '精确率')
        axes[1].set_title(L['pr_curve'])
        axes[1].legend(loc="lower left")
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["roc_pr"] / f"roc_pr_curves_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, lang: str = 'cn'):
        """绘制t-SNE可视化"""
        L = LABELS[lang]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        for label, color, name in [(0, COLORS['normal'], L['normal']), 
                                    (1, COLORS['fault'], L['fault'])]:
            mask = labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=color, label=name, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(L['tsne'])
        ax.legend()
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["tsne"] / f"tsne_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_reconstruction(self, original: np.ndarray, recon: np.ndarray, 
                            idx: int = 0, lang: str = 'cn'):
        """绘制重构对比图"""
        L = LABELS[lang]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        channel_names = ['CWT', 'STFT', 'Context']
        
        for i, name in enumerate(channel_names):
            axes[0, i].imshow(original[i], cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'{name} - {"Original" if lang == "en" else "原始"}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon[i], cmap='viridis', aspect='auto')
            axes[1, i].set_title(f'{name} - {"Reconstructed" if lang == "en" else "重构"}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'{"Reconstruction Comparison" if lang == "en" else "重构对比"} (Sample {idx})')
        plt.tight_layout()
        
        save_path = self.cfg.VIZ_SUBDIRS["recon"] / f"recon_sample{idx}_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_svdd_sphere(self, features_2d: np.ndarray, scores: np.ndarray, 
                         center_2d: np.ndarray, lang: str = 'cn'):
        """绘制SVDD超球可视化"""
        L = LABELS[lang]
        fig, ax = plt.subplots(figsize=(8, 8))
        
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                            c=norm_scores, cmap='RdYlGn_r', alpha=0.6, s=30,
                            edgecolors='white', linewidth=0.5)
        
        ax.scatter(center_2d[0], center_2d[1], c='black', marker='X', s=200,
                  edgecolors='white', linewidth=2, label='Center' if lang == 'en' else '中心')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(L['score'])
        
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title('SVDD Feature Space' if lang == 'en' else 'SVDD特征空间')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["svdd_sphere"] / f"svdd_sphere_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_error_samples(self, error_info: List[Dict], lang: str = 'cn'):
        """绘制错误样本分析"""
        if not error_info:
            return None
        
        L = LABELS[lang]
        n_samples = min(len(error_info), 6)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, info in enumerate(error_info[:n_samples]):
            ax = axes[i]
            
            if 'image' in info and info['image'] is not None:
                img = info['image']
                if img.ndim == 3:
                    img = img[0]
                ax.imshow(img, cmap='viridis', aspect='auto')
            
            true_label = L['normal'] if info['true'] == 0 else L['fault']
            pred_label = L['normal'] if info['pred'] == 0 else L['fault']
            score = info.get('score', 0)
            
            ax.set_title(f"True: {true_label}, Pred: {pred_label}\nScore: {score:.3f}", fontsize=9)
            ax.axis('off')
        
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Error Sample Analysis' if lang == 'en' else '错误样本分析')
        plt.tight_layout()
        
        save_path = self.cfg.VIZ_SUBDIRS["error_samples"] / f"error_samples_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path
    
    def plot_sample_preview(self, images: List[np.ndarray], labels: List[int],
                            zerone_features: List[np.ndarray] = None, lang: str = 'cn'):
        """绘制样本预览"""
        L = LABELS[lang]
        n_samples = min(len(images), self.cfg.SAMPLE_PREVIEW_COUNT)
        
        fig = plt.figure(figsize=(16, 4 * ((n_samples + 3) // 4)))
        
        for i in range(n_samples):
            ax = fig.add_subplot((n_samples + 3) // 4, 4, i + 1)
            
            img = images[i]
            if img.ndim == 3 and img.shape[0] == 3:
                rgb = np.transpose(img, (1, 2, 0))
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                ax.imshow(rgb)
            else:
                ax.imshow(img[0] if img.ndim == 3 else img, cmap='viridis')
            
            label_str = L['normal'] if labels[i] == 0 else L['fault']
            ax.set_title(f"Sample {i}: {label_str}")
            ax.axis('off')
        
        plt.suptitle('Sample Preview' if lang == 'en' else '样本预览')
        plt.tight_layout()
        
        save_path = self.cfg.VIZ_SUBDIRS["sample_preview"] / f"sample_preview_{lang}.png"
        fig.savefig(save_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return save_path


# =============================================================================
# 第5步: 日志管理器
# =============================================================================

class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, cfg: ThreeStageConfigV4, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.log_file = cfg.LOG_DIR / f"{stage}_training_log.csv"
        self.history = defaultdict(list)
        self._init_csv()
    
    def _init_csv(self):
        """初始化CSV文件"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, **kwargs):
        """记录一条日志"""
        for key, value in kwargs.items():
            self.history[key].append(value)
    
    def save_csv(self):
        """保存为CSV文件"""
        if not self.history:
            return
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            headers = list(self.history.keys())
            writer.writerow(headers)
            
            n_rows = max(len(v) for v in self.history.values())
            for i in range(n_rows):
                row = [self.history[h][i] if i < len(self.history[h]) else '' for h in headers]
                writer.writerow(row)
        
        print(f"[日志] 训练日志已保存: {self.log_file}")
    
    def get_history(self) -> Dict:
        """获取历史记录"""
        return dict(self.history)


# =============================================================================
# 第6步: 检查点管理器
# =============================================================================

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, cfg: ThreeStageConfigV4, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.checkpoint_dir = cfg.CHECKPOINT_DIR / stage
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
    
    def save(self, model, optimizer, epoch: int, metrics: Dict, scheduler=None):
        """保存检查点"""
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pth"
        
        save_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            save_dict['scheduler_state'] = scheduler.state_dict()
        
        if hasattr(model, 'center'):
            save_dict['center'] = model.center
        
        torch.save(save_dict, ckpt_path)
        self.checkpoints.append(ckpt_path)
        
        while len(self.checkpoints) > self.cfg.MAX_CHECKPOINTS:
            old_ckpt = self.checkpoints.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()
        
        print(f"[检查点] 已保存: {ckpt_path.name}")
    
    def load_latest(self, model, optimizer=None, scheduler=None):
        """加载最新检查点"""
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pth"))
        if not ckpts:
            return 0, {}
        
        latest = ckpts[-1]
        ckpt = torch.load(latest, map_location=self.cfg.DEVICE)
        
        model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler is not None and 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'center' in ckpt and hasattr(model, 'center'):
            model.center = ckpt['center']
        
        print(f"[检查点] 已加载: {latest.name}")
        return ckpt['epoch'], ckpt.get('metrics', {})


# =============================================================================
# 第7步: 特征提取函数
# =============================================================================

def compute_time_features(sig: np.ndarray) -> np.ndarray:
    """
    计算时域特征 (15维)
    
    特征列表:
        均值, RMS, 方差, 标准差, 最大值, 最小值, 峰峰值,
        峭度, 偏度, 过零率, 绝对均值, 波峰因子, 脉冲因子, 裕度因子, 波形因子
    """
    x = np.asarray(sig, dtype=float).ravel()
    if len(x) == 0:
        return np.zeros(15, dtype=np.float32)
    
    mean_val = np.mean(x)
    rms_val = np.sqrt(np.mean(x**2))
    var_val = np.var(x)
    std_val = np.std(x)
    max_val = np.max(x)
    min_val = np.min(x)
    p2p_val = max_val - min_val
    
    xc = x - mean_val
    m2 = np.mean(xc**2) + 1e-12
    m4 = np.mean(xc**4)
    kurtosis = m4 / (m2**2)
    m3 = np.mean(xc**3)
    skewness = m3 / (std_val**3 + 1e-12)
    
    zero_cross = np.sum(np.abs(np.diff(np.sign(x))) > 0) / (len(x) - 1 + 1e-12)
    mean_abs = np.mean(np.abs(x))
    
    crest = max_val / (rms_val + 1e-12)
    impulse = max_val / (mean_abs + 1e-12)
    margin = max_val / (np.mean(np.abs(x)**0.5)**2 + 1e-12)
    waveform = rms_val / (mean_abs + 1e-12)
    
    return np.array([
        mean_val, rms_val, var_val, std_val, max_val, min_val, p2p_val,
        kurtosis, skewness, zero_cross, mean_abs,
        crest, impulse, margin, waveform
    ], dtype=np.float32)


def compute_stft_features(sig: np.ndarray, fs: float, nperseg: int = 128, 
                          noverlap: int = 64) -> np.ndarray:
    """计算STFT段均值特征 (127维)"""
    try:
        _, _, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx[1:, :])
        seg_means = np.mean(mag, axis=0)
        
        out = np.zeros(127, dtype=np.float32)
        L = min(len(seg_means), 127)
        out[:L] = seg_means[:L]
        return out
    except Exception:
        return np.zeros(127, dtype=np.float32)


def compute_psd_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    计算PSD特征 (1050维)
    
    1-1000Hz: 每1Hz一维 → 1000维
    1001-2000Hz: 每20Hz一维 → 50维
    """
    try:
        sig_dc = sig - np.mean(sig)
        freqs, psd = welch(sig_dc, fs=fs, nperseg=min(len(sig)//2, 4096))
        
        target_freqs = np.arange(1, 2001, 1)
        psd_interp = np.interp(target_freqs, freqs, psd)
        
        psd_low = psd_interp[:1000]
        psd_high_raw = psd_interp[1000:2000]
        
        psd_high = np.array([psd_high_raw[i:i+20].mean() for i in range(0, 1000, 20)])
        
        return np.concatenate([psd_low, psd_high]).astype(np.float32)
    except Exception:
        return np.zeros(1050, dtype=np.float32)


def compute_hf_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    计算高频特征 (8维)
    
    4个阈值 (1000, 2000, 3000, 4000 Hz) × (幅值比, 功率比)
    """
    try:
        sig_dc = sig - np.mean(sig)
        freqs, psd = welch(sig_dc, fs=fs, nperseg=min(len(sig)//2, 4096))
        
        total_power = np.sum(psd)
        hf_feat = []
        
        for thr in [1000, 2000, 3000, 4000]:
            hf_mask = freqs >= thr
            hf_power = np.sum(psd[hf_mask])
            
            amp_ratio = np.sqrt(hf_power / (total_power + 1e-12))
            pwr_ratio = hf_power / (total_power + 1e-12)
            
            hf_feat.extend([amp_ratio, pwr_ratio])
        
        return np.array(hf_feat, dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def extract_zerone_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    提取完整的Zerone 1200维特征
    
    组成:
        时域 (15维) + STFT (127维) + PSD (1050维) + 高频 (8维) = 1200维
    """
    features = []
    
    features.append(compute_time_features(sig))
    features.append(compute_stft_features(sig, fs))
    features.append(compute_psd_features(sig, fs))
    features.append(compute_hf_features(sig, fs))
    
    feat_vec = np.concatenate(features).astype(np.float32)
    
    if len(feat_vec) < TOTAL_FEAT_DIM:
        feat_vec = np.pad(feat_vec, (0, TOTAL_FEAT_DIM - len(feat_vec)))
    elif len(feat_vec) > TOTAL_FEAT_DIM:
        feat_vec = feat_vec[:TOTAL_FEAT_DIM]
    
    return feat_vec


def signal_to_hetero_image(sig: np.ndarray, fs: float, size: int = 224) -> np.ndarray:
    """
    将振动信号转换为Hetero三通道时频图像
    
    通道设计:
        Ch0: CWT (Morlet小波) - 时频局部特征
        Ch1: STFT幅度谱 - 短时频域特征
        Ch2: Context (波形折叠) - 时域细节
    """
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    
    # Ch0: CWT
    scales = np.arange(1, min(129, len(sig)//64 + 1))
    try:
        cwt_matrix, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
        cwt_abs = np.log1p(np.abs(cwt_matrix).astype(np.float32))
        c0 = cv2.resize(cwt_abs, (size, size), interpolation=cv2.INTER_LINEAR)
        c0 = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)
    except Exception:
        c0 = np.zeros((size, size), dtype=np.float32)
    
    # Ch1: STFT
    try:
        nperseg = min(256, len(sig)//4)
        _, _, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        mag = np.log1p(np.abs(Zxx).astype(np.float32))
        c1 = cv2.resize(mag, (size, size), interpolation=cv2.INTER_LINEAR)
        c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-8)
    except Exception:
        c1 = np.zeros((size, size), dtype=np.float32)
    
    # Ch2: Context (波形折叠)
    try:
        h_fold = max(1, len(sig) // size)
        if h_fold * size <= len(sig):
            mat = sig[:h_fold * size].reshape(h_fold, size)
        else:
            mat = sig.reshape(-1, 1)
        c2 = cv2.resize(mat.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
        c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-8)
    except Exception:
        c2 = np.zeros((size, size), dtype=np.float32)
    
    return np.stack([c0, c1, c2], axis=0).astype(np.float32)


# =============================================================================
# 第8步: 数据读取工具
# =============================================================================

def parse_signal_value(v: Any, target_len: int = 8192) -> Optional[np.ndarray]:
    """解析信号数据"""
    try:
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("\n", " ")
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
    """读取JSONL文件"""
    records = []
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return records


def read_json_file(filepath: Path) -> List[Dict]:
    """读取JSON文件"""
    records = []
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for key in ['data', 'records', 'list', 'items']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
    except Exception:
        pass
    return records


def get_label_from_path(filepath: Path, class_keywords: Dict) -> Optional[str]:
    """从路径推断标签"""
    for parent in filepath.parents:
        name = parent.name.lower()
        for cls, keywords in class_keywords.items():
            if any(kw.lower() in name for kw in keywords):
                return cls
    filename = filepath.name.lower()
    for cls, keywords in class_keywords.items():
        if any(kw.lower() in filename for kw in keywords):
            return cls
    return None


# =============================================================================
# 第9步: 数据集类
# =============================================================================

class TransformerVibrationDataset(Dataset):
    """
    变压器振动数据集 (V4版本)
    
    支持三种输出模式:
        - hetero: 仅输出Hetero三通道图像
        - zerone: 仅输出Zerone 1200维特征
        - dual: 同时输出两者
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        cfg: ThreeStageConfigV4,
        use_labels: bool = False,
        split_name: str = ""
    ):
        self.root_dir = Path(root_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.split_name = split_name
        self.samples: List[Tuple[Path, str, List[np.ndarray], Optional[int]]] = []
        self._build_index()
    
    def _build_index(self):
        """构建样本索引"""
        if not self.root_dir.exists():
            print(f"[警告] 目录不存在: {self.root_dir}")
            return
        
        files = list(self.root_dir.rglob("*.jsonl")) + list(self.root_dir.rglob("*.json"))
        label_counts = Counter()
        
        for fp in tqdm(files, desc=f"扫描 {self.split_name or self.root_dir.name}", leave=False):
            if fp.suffix == '.jsonl':
                records = read_jsonl_file(fp)
            else:
                records = read_json_file(fp)
            
            if not records:
                continue
            
            label = None
            if self.use_labels:
                label_str = get_label_from_path(fp, self.cfg.CLASS_KEYWORDS)
                if label_str == "正常":
                    label = 0
                elif label_str == "故障":
                    label = 1
                if label is None:
                    continue
                label_counts[label_str] += 1
            
            groups: Dict[str, List[np.ndarray]] = {}
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
        
        print(f"[{self.split_name or self.root_dir.name}] 加载 {len(self.samples)} 个样本")
        if self.use_labels and label_counts:
            for lbl, cnt in label_counts.items():
                print(f"  {lbl}: {cnt}")
    
    def _aggregate_channels(self, sig_list: List[np.ndarray]) -> np.ndarray:
        """多通道能量加权聚合"""
        if len(sig_list) == 1:
            return sig_list[0]
        X = np.stack(sig_list, axis=1)
        E = np.mean(X**2, axis=0) + 1e-12
        w = E / E.sum()
        return (X @ w).astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        返回:
            image: (3, H, W) 或 zeros
            zerone: (1200,) 或 zeros
            label: 标签 (-1表示无标签)
            idx: 索引
        """
        fp, time_key, sig_list, label = self.samples[idx]
        sig = self._aggregate_channels(sig_list)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        
        branch = self.cfg.BRANCH_MODE
        
        if branch in ['hetero', 'dual']:
            image = signal_to_hetero_image(sig, self.cfg.FS, self.cfg.INPUT_SIZE)
        else:
            image = np.zeros((3, self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE), dtype=np.float32)
        
        if branch in ['zerone', 'dual']:
            zerone = extract_zerone_features(sig, self.cfg.FS)
        else:
            zerone = np.zeros(self.cfg.ZERONE_DIM, dtype=np.float32)
        
        return (
            torch.from_numpy(image),
            torch.from_numpy(zerone),
            label if label is not None else -1,
            idx
        )
    
    def get_raw_signal(self, idx: int) -> Tuple[np.ndarray, int]:
        """获取原始信号 (用于可视化)"""
        fp, time_key, sig_list, label = self.samples[idx]
        sig = self._aggregate_channels(sig_list)
        return sig, label if label is not None else -1


# =============================================================================
# 第10步: 模型定义
# =============================================================================

class ZeroneMLP(nn.Module):
    """Zerone特征处理分支 (MLP)"""
    
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
    """Hetero图像处理分支 (CNN)"""
    
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
        
        if output_dim != 512:
            self.proj = nn.Linear(512, output_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.proj(h)


class BranchEncoderV4(nn.Module):
    """
    支线编码器 (V4版本)
    
    支持三种模式:
        - hetero: 仅CNN
        - zerone: 仅MLP
        - dual: CNN + MLP + 可选融合策略 (concat/attention/gate)
    
    V4改进:
        - 模块化融合策略
        - 返回融合权重用于可视化
    """
    
    def __init__(self, cfg: ThreeStageConfigV4):
        super().__init__()
        self.cfg = cfg
        self.branch_mode = cfg.BRANCH_MODE
        self.fusion_mode = cfg.FUSION_MODE
        
        if self.branch_mode in ['hetero', 'dual']:
            self.hetero_branch = HeteroCNN(output_dim=cfg.CNN_FEAT_DIM)
        
        if self.branch_mode in ['zerone', 'dual']:
            self.zerone_branch = ZeroneMLP(
                input_dim=cfg.ZERONE_DIM,
                output_dim=cfg.MLP_FEAT_DIM
            )
        
        # 确定输出维度和融合模块
        if self.branch_mode == 'hetero':
            self.output_dim = cfg.CNN_FEAT_DIM
        elif self.branch_mode == 'zerone':
            self.output_dim = cfg.MLP_FEAT_DIM
            self.proj = nn.Sequential(
                nn.Linear(cfg.MLP_FEAT_DIM, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.output_dim = 512
        else:  # dual
            # 根据融合模式选择融合模块
            # Select fusion module based on FUSION_MODE
            if self.fusion_mode == 'concat':
                self.fusion_module = ConcatFusion(
                    img_dim=cfg.CNN_FEAT_DIM,
                    feat_dim=cfg.MLP_FEAT_DIM,
                    out_dim=512
                )
            elif self.fusion_mode == 'attention':
                self.fusion_module = AttentionFusion(
                    img_dim=cfg.CNN_FEAT_DIM,
                    feat_dim=cfg.MLP_FEAT_DIM,
                    out_dim=512
                )
            elif self.fusion_mode == 'gate':
                self.fusion_module = GatedFusion(
                    img_dim=cfg.CNN_FEAT_DIM,
                    feat_dim=cfg.MLP_FEAT_DIM,
                    out_dim=512
                )
            else:
                raise ValueError(f"未知融合模式: {self.fusion_mode}")
            
            self.output_dim = 512
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            image: (B, 3, H, W)
            zerone: (B, 1200)
            
        返回:
            dict: {
                'h': 编码特征 (B, 512),
                'h_img': 图像分支特征 (仅dual模式),
                'h_feat': 特征分支特征 (仅dual模式),
                'fusion_weights': 融合权重 (仅dual+attention/gate模式)
            }
        """
        result = {}
        
        if self.branch_mode == 'hetero':
            h = self.hetero_branch(image)
            result['h'] = h
            
        elif self.branch_mode == 'zerone':
            h = self.zerone_branch(zerone)
            h = self.proj(h)
            result['h'] = h
            
        else:  # dual
            h_img = self.hetero_branch(image)
            h_feat = self.zerone_branch(zerone)
            
            fusion_result = self.fusion_module(h_img, h_feat)
            
            result['h'] = fusion_result['h_fused']
            result['h_img'] = h_img
            result['h_feat'] = h_feat
            result['fusion_weights'] = fusion_result['weights']
        
        return result


class AnomalyModelV4(nn.Module):
    """
    异常检测模型 V4
    
    组件:
        - 支线编码器 (根据模式和融合策略选择)
        - Deep SVDD头
        - VAE解码器 (可选，仅hetero/dual模式)
    
    V4改进:
        - 集成融合权重追踪
        - 支持多种融合策略
    """
    
    def __init__(self, cfg: ThreeStageConfigV4):
        super().__init__()
        self.cfg = cfg
        
        # 编码器
        self.encoder = BranchEncoderV4(cfg)
        
        # SVDD投影头
        self.svdd_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
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
        
        self.alpha = 0.6  # SVDD权重
    
    def encode(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码"""
        return self.encoder(image, zerone)
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        enc_result = self.encode(image, zerone)
        h = enc_result['h']
        
        # SVDD
        z_svdd = self.svdd_proj(h)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        result = {
            'h': h,
            'z_svdd': z_svdd,
            'svdd_score': svdd_score,
        }
        
        # 传递融合相关信息
        if 'h_img' in enc_result:
            result['h_img'] = enc_result['h_img']
        if 'h_feat' in enc_result:
            result['h_feat'] = enc_result['h_feat']
        if 'fusion_weights' in enc_result:
            result['fusion_weights'] = enc_result['fusion_weights']
        
        # VAE (如果启用)
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
                'mu': mu,
                'logvar': logvar
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
        """计算异常得分"""
        out = self.forward(image, zerone)
        
        if self.has_vae:
            svdd_score = out['svdd_score']
            vae_score = out['vae_recon_loss'] + 0.01 * out['vae_kl']
            
            svdd_norm = svdd_score / (svdd_score.mean() + 1e-8)
            vae_norm = vae_score / (vae_score.mean() + 1e-8)
            
            return self.alpha * svdd_norm + (1 - self.alpha) * vae_norm
        else:
            return out['svdd_score']


class FaultClassifierV4(nn.Module):
    """故障分类器 (V4版本阶段三)"""
    
    def __init__(self, encoder: BranchEncoderV4, num_classes: int = 2, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回包含融合权重的结果"""
        enc_result = self.encoder(image, zerone)
        logits = self.classifier(enc_result['h'])
        
        result = {
            'logits': logits,
            'h': enc_result['h'],
        }
        
        if 'fusion_weights' in enc_result:
            result['fusion_weights'] = enc_result['fusion_weights']
        
        return result
    
    def unfreeze_encoder(self, num_layers: int = 1):
        """解冻编码器最后几层"""
        for param in self.encoder.parameters():
            param.requires_grad = True


# =============================================================================
# 第11步: 阶段一 - 无监督训练 (V4: 严格只用TRAIN)
# =============================================================================

def train_stage1(cfg: ThreeStageConfigV4, resume_from: Path = None) -> Tuple[AnomalyModelV4, Dict]:
    """
    阶段一：无监督学习
    
    【V4严格数据分离】
        - 只使用 TRAIN 数据（无标签）
        - VAL 和 TEST 不参与阶段1训练
    
    训练流程:
        1. VAE预训练 (如果启用)
        2. SVDD中心初始化
        3. 联合训练 SVDD + VAE
    """
    print("\n" + "="*70)
    print("阶段一：无监督学习 (V4严格数据分离)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage1")
    ckpt_mgr = CheckpointManager(cfg, "stage1")
    
    # 【V4改进】只加载 TRAIN 数据 (无标签)
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
    model = AnomalyModelV4(cfg).to(device)
    print(f"  支线模式: {cfg.BRANCH_MODE}")
    print(f"  融合策略: {cfg.FUSION_MODE}")
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
    
    # VAE预训练 (如果启用且从头开始)
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
    
    # SVDD中心初始化 (如果从头开始)
    if start_epoch == 0:
        print("\n[3/4] 初始化SVDD中心...")
        model.init_center(train_loader, device)
    
    # 联合训练
    print(f"\n[4/4] 联合训练 (Epoch {start_epoch+1} -> {cfg.STAGE1_EPOCHS})...")
    
    history = {
        'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': [], 'recon_loss': []
    }
    
    for epoch in range(start_epoch, cfg.STAGE1_EPOCHS):
        model.train()
        epoch_svdd, epoch_vae, epoch_total, epoch_recon = 0, 0, 0, 0
        
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch / max(cfg.BETA_WARMUP, 1)))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False)
        for batch in pbar:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            
            svdd_loss = out['svdd_score'].mean()
            
            if model.has_vae:
                vae_loss = out['vae_recon_loss'].mean() + beta * out['vae_kl'].mean()
                total_loss = svdd_loss + vae_loss
                epoch_recon += out['vae_recon_loss'].mean().item()
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
            
            pbar.set_postfix({'svdd': f'{svdd_loss.item():.4f}', 'total': f'{total_loss.item():.4f}'})
        
        scheduler.step()
        
        n_batches = len(train_loader)
        avg_svdd = epoch_svdd / n_batches
        avg_vae = epoch_vae / n_batches
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        
        history['epoch'].append(epoch + 1)
        history['svdd_loss'].append(avg_svdd)
        history['vae_loss'].append(avg_vae)
        history['total_loss'].append(avg_total)
        history['recon_loss'].append(avg_recon)
        
        logger.log(epoch=epoch+1, svdd_loss=avg_svdd, vae_loss=avg_vae, 
                   total_loss=avg_total, recon_loss=avg_recon, lr=scheduler.get_last_lr()[0])
        
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.center,
                'epoch': epoch + 1,
                'loss': best_loss,
            }, cfg.MODEL_DIR / "stage1_best.pth")
        
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(model, optimizer, epoch + 1, 
                         {'svdd_loss': avg_svdd, 'total_loss': avg_total}, scheduler)
        
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage1", lang=lang)
            
            if model.has_vae:
                model.eval()
                with torch.no_grad():
                    sample_img, sample_zr, _, _ = next(iter(train_loader))
                    sample_img, sample_zr = sample_img.to(device), sample_zr.to(device)
                    out = model(sample_img, sample_zr)
                    
                    for lang in cfg.LANGS:
                        viz.plot_reconstruction(
                            sample_img[0].cpu().numpy(),
                            out['recon'][0].cpu().numpy(),
                            idx=epoch+1, lang=lang
                        )
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
    
    logger.save_csv()
    
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage1", lang=lang)
    
    # SVDD特征空间可视化
    print("\n[*] 生成SVDD特征空间可视化...")
    model.eval()
    all_features, all_scores = [], []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="提取特征", leave=False):
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            out = model(img, zr)
            all_features.append(out['z_svdd'].cpu().numpy())
            all_scores.append(out['svdd_score'].cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    center_2d = pca.transform(model.center.cpu().numpy().reshape(1, -1))[0]
    
    for lang in cfg.LANGS:
        viz.plot_svdd_sphere(features_2d, all_scores, center_2d, lang=lang)
    
    # V4新增: 分支可视化
    print("\n[*] 生成分支特征可视化...")

    # 创建评估用的数据加载器
    eval_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=True, split_name="TRAIN_EVAL")
    eval_loader = DataLoader(eval_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # Hetero分支可视化
    if cfg.BRANCH_MODE in ['hetero', 'dual']:
        for lang in cfg.LANGS:
            viz.plot_hetero_reconstruction(model, eval_loader, device, n_samples=6, lang=lang)
            viz.plot_hetero_channel_analysis(eval_loader, device, n_samples=100, lang=lang)

    # Zerone分支可视化
    if cfg.BRANCH_MODE in ['zerone', 'dual']:
        for lang in cfg.LANGS:
            viz.plot_zerone_feature_analysis(eval_loader, device, n_samples=200, lang=lang)

    print(f"\n【阶段一完成】最佳损失: {best_loss:.4f}")
    print(f"  模型保存: {cfg.MODEL_DIR / 'stage1_best.pth'}")
    
    return model, history


# =============================================================================
# 第12步: 阶段二 - 伪标签生成 (V4: 严格只用TRAIN)
# =============================================================================

def run_stage2(model: AnomalyModelV4, cfg: ThreeStageConfigV4) -> Dict:
    """
    阶段二：基于异常得分生成伪标签
    
    【V4严格数据分离】
        - 只使用 TRAIN 数据计算得分分布和生成伪标签
        - VAL 和 TEST 不参与
    """
    print("\n" + "="*70)
    print("阶段二：伪标签生成 (V4严格数据分离)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    viz = VisualizationManager(cfg)
    
    # 【V4改进】只加载 TRAIN 数据
    print("\n[1/3] 加载数据 (仅TRAIN)...")
    train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
    
    loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 计算异常得分
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
    
    # 计算阈值
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
    
    # 保存结果
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
    
    # 可视化
    for lang in cfg.LANGS:
        viz.plot_score_distribution(all_scores, t_normal, t_anomaly, lang=lang)
    
    # 饼图
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [len(pseudo_normal), len(uncertain), len(pseudo_anomaly)]
    labels_cn = ['高置信正常', '不确定', '高置信异常']
    labels_en = ['Confident Normal', 'Uncertain', 'Confident Anomaly']
    colors = [COLORS['normal'], COLORS['uncertain'], COLORS['fault']]
    
    ax.pie(sizes, labels=labels_cn, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title('伪标签分布')
    plt.tight_layout()
    fig.savefig(cfg.VIZ_SUBDIRS["score_dist"] / "pseudo_label_pie_cn.png",
               dpi=cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
    
    ax.clear()
    ax.pie(sizes, labels=labels_en, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title('Pseudo Label Distribution')
    plt.tight_layout()
    fig.savefig(cfg.VIZ_SUBDIRS["score_dist"] / "pseudo_label_pie_en.png",
               dpi=cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n【阶段二完成】伪标签保存: {cfg.STAGE2_DIR / 'pseudo_labels.npz'}")
    
    return pseudo_labels


# =============================================================================
# 第13步: 阶段三 - 监督微调 (V4: VAL训练, TEST仅评估)
# =============================================================================

def train_stage3(model: AnomalyModelV4, pseudo_labels: Dict, cfg: ThreeStageConfigV4) -> FaultClassifierV4:
    """
    阶段三：有监督微调
    
    【V4严格数据分离】
        - 训练: 使用 VAL 数据 (有标签)
        - 验证: 从 VAL 中划分一部分做验证
        - 测试: 仅使用 TEST 做最终评估（标签不参与训练）
    """
    print("\n" + "="*70)
    print("阶段三：有监督微调 (V4严格数据分离)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage3")
    ckpt_mgr = CheckpointManager(cfg, "stage3")
    
    # 【V4改进】只加载 VAL 数据用于训练
    print("\n[1/4] 加载数据 (VAL用于训练, TEST仅评估)...")
    val_ds = TransformerVibrationDataset(cfg.VAL_DIR, cfg, use_labels=True, split_name="VAL")
    
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
    
    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 构建分类器
    print("\n[2/4] 构建分类器...")
    classifier = FaultClassifierV4(model.encoder, num_classes=2, freeze_encoder=True).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE3_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print(f"\n[3/4] 训练 ({cfg.STAGE3_EPOCHS}轮)...")
    
    history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(cfg.STAGE3_EPOCHS):
        # 训练阶段
        classifier.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            img, zr, label, _ = batch
            img, zr, label = img.to(device), zr.to(device), label.to(device)
            
            result = classifier(img, zr)
            loss = criterion(result['logits'], label)
            
            # 记录融合权重用于可视化
            if 'fusion_weights' in result and result['fusion_weights'] is not None:
                viz.record_fusion_weights(result['fusion_weights'], label)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
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
        
        avg_train_loss = train_loss / len(train_loader)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        
        logger.log(epoch=epoch+1, train_loss=avg_train_loss, val_acc=val_acc,
                  val_f1=val_f1, val_precision=val_prec, val_recall=val_rec)
        
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
        
        if epoch == 15:
            classifier.unfreeze_encoder(1)
            print("  [*] 解冻编码器")
        
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(classifier, optimizer, epoch + 1,
                         {'val_f1': val_f1, 'val_acc': val_acc}, scheduler)
        
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage3", lang=lang)
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    logger.save_csv()
    
    # 融合权重可视化 (V4新增)
    if cfg.BRANCH_MODE == 'dual' and cfg.FUSION_MODE in ['attention', 'gate']:
        print("\n[*] 生成融合权重可视化...")
        for lang in cfg.LANGS:
            viz.plot_fusion_weights(lang=lang)
    
    # 加载最佳模型
    best_ckpt = torch.load(cfg.MODEL_DIR / "stage3_best.pth", map_location=device)
    classifier.load_state_dict(best_ckpt['model_state'])
    
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage3", lang=lang)
    
    # ==================== TEST评估 (仅评估，不训练) ====================
    print("\n[4/4] 在TEST上评估 (标签仅用于评估)...")
    test_ds = TransformerVibrationDataset(cfg.TEST_DIR, cfg, use_labels=True, split_name="TEST")
    
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
        
        classifier.eval()
        test_preds, test_labels, test_probs, test_features = [], [], [], []
        error_samples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="TEST评估", leave=False)):
                img, zr, label, idx = batch
                img, zr = img.to(device), zr.to(device)
                
                result = classifier(img, zr)
                probs = F.softmax(result['logits'], dim=1)
                preds = result['logits'].argmax(dim=1)
                
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(label.tolist())
                test_probs.extend(probs[:, 1].cpu().tolist())
                test_features.append(result['h'].cpu().numpy())
                
                for i in range(len(preds)):
                    if preds[i].item() != label[i].item():
                        error_samples.append({
                            'idx': idx[i].item(),
                            'true': label[i].item(),
                            'pred': preds[i].item(),
                            'score': probs[i, 1].item(),
                            'image': img[i].cpu().numpy() if cfg.BRANCH_MODE != 'zerone' else None,
                        })
        
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
        test_rec = recall_score(test_labels, test_preds, average='macro', zero_division=0)
        
        print(f"\n  【TEST评估结果】")
        print(f"    准确率: {test_acc:.4f}")
        print(f"    F1分数: {test_f1:.4f}")
        print(f"    精确率: {test_prec:.4f}")
        print(f"    召回率: {test_rec:.4f}")
        print(f"    错误样本数: {len(error_samples)}")
        
        # 可视化
        for lang in cfg.LANGS:
            viz.plot_confusion_matrix(np.array(test_labels), np.array(test_preds), lang=lang)
            viz.plot_roc_pr_curves(np.array(test_labels), np.array(test_probs), lang=lang)
        
        # t-SNE
        test_features = np.concatenate(test_features, axis=0)
        for lang in cfg.LANGS:
            viz.plot_tsne(test_features, np.array(test_labels), lang=lang)
        
        # 错误样本分析
        if error_samples:
            for lang in cfg.LANGS:
                viz.plot_error_samples(error_samples[:6], lang=lang)
        
        # 保存评估结果
        eval_results = {
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'n_errors': len(error_samples),
            'fusion_mode': cfg.FUSION_MODE,
            'branch_mode': cfg.BRANCH_MODE,
        }
        
        with open(cfg.STAGE3_DIR / "test_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    # V4新增: Dual融合样本可视化
    if cfg.BRANCH_MODE == 'dual':
        print("\n[*] 生成Dual融合样本可视化...")
        for lang in cfg.LANGS:
            viz.plot_dual_fusion_samples(model, classifier, test_loader, device, n_samples=6, lang=lang)

    print(f"\n【阶段三完成】最佳验证F1: {best_f1:.4f}")
    
    return classifier


# =============================================================================
# 第14步: 支线对比实验与可视化
# =============================================================================

def check_branch_status(output_root: Path) -> Dict[str, Dict]:
    """检测各支线的完成状态"""
    status = {}
    
    # 检测所有可能的支线目录
    for branch_dir in output_root.glob("branch_*"):
        branch_name = branch_dir.name.replace("branch_", "")
        
        info = {
            'completed': False,
            'stage': 0,
            'checkpoint': None,
            'has_stage1': False,
            'has_stage2': False,
            'has_stage3': False,
            'eval_result': None,
        }
        
        # 检查阶段1
        stage1_model = branch_dir / "models" / "stage1_best.pth"
        if stage1_model.exists():
            info['has_stage1'] = True
            info['stage'] = 1
        
        # 检查阶段1检查点
        stage1_ckpt_dir = branch_dir / "checkpoints" / "stage1"
        if stage1_ckpt_dir.exists():
            ckpts = sorted(stage1_ckpt_dir.glob("checkpoint_epoch*.pth"))
            if ckpts:
                info['checkpoint'] = ckpts[-1]
                try:
                    epoch_str = ckpts[-1].stem.split('epoch')[-1]
                    info['checkpoint_epoch'] = int(epoch_str)
                except:
                    info['checkpoint_epoch'] = 0
        
        # 检查阶段2
        pseudo_labels = branch_dir / "stage2_pseudo_labels" / "pseudo_labels.npz"
        if pseudo_labels.exists():
            info['has_stage2'] = True
            info['stage'] = 2
        
        # 检查阶段3
        stage3_eval = branch_dir / "stage3_supervised" / "test_evaluation.json"
        if stage3_eval.exists():
            info['has_stage3'] = True
            info['stage'] = 3
            info['completed'] = True
            try:
                with open(stage3_eval, 'r', encoding='utf-8') as f:
                    info['eval_result'] = json.load(f)
            except:
                pass
        
        status[branch_name] = info
    
    return status


def print_status(status: Dict[str, Dict]):
    """打印各支线状态"""
    print("\n" + "="*70)
    print("支线状态检测")
    print("="*70)
    
    for branch, info in status.items():
        print(f"\n【{branch.upper()}】")
        if info['completed']:
            print(f"  ✅ 已完成")
            if info['eval_result']:
                res = info['eval_result']
                print(f"     准确率: {res.get('test_acc', 0):.4f}")
                print(f"     F1分数: {res.get('test_f1', 0):.4f}")
        else:
            print(f"  ⏳ 未完成")
            print(f"     当前阶段: {info['stage']}")
            if info['checkpoint']:
                print(f"     最新检查点: {info['checkpoint'].name}")
                print(f"     检查点轮次: {info.get('checkpoint_epoch', '?')}")
    
    print("\n" + "="*70)


def generate_comparison_visualization(output_root: Path):
    """生成三支线对比可视化"""
    print("\n" + "="*70)
    print("生成支线对比可视化")
    print("="*70)
    
    status = check_branch_status(output_root)
    
    # 收集所有完成的支线结果
    results = {}
    for branch, info in status.items():
        if info.get('eval_result'):
            results[branch] = info['eval_result']
            print(f"  ✅ {branch}: Acc={info['eval_result'].get('test_acc', 0):.4f}, "
                  f"F1={info['eval_result'].get('test_f1', 0):.4f}")
    
    if len(results) < 2:
        print(f"\n[警告] 只有 {len(results)} 个支线完成，需要至少2个支线才能生成对比")
        return
    
    compare_dir = output_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in ['cn', 'en']:
        print(f"\n生成 {lang.upper()} 版本...")
        
        if lang == 'cn':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            metric_names = ['准确率', 'F1分数', '精确率', '召回率']
            title = '支线性能对比'
            ylabel = '得分'
        else:
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            title = 'Branch Performance Comparison'
            ylabel = 'Score'
        
        branches = list(results.keys())
        metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
        
        # 柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / len(branches)
        
        colors_list = ['#0072B2', '#E69F00', '#CC79A7', '#009E73', '#D55E00']
        
        for i, branch in enumerate(branches):
            values = [results[branch].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, values, width,
                         label=branch,
                         color=colors_list[i % len(colors_list)],
                         edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(branches) - 1) / 2)
        ax.set_xticklabels(metric_names)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        plt.tight_layout()
        fig.savefig(compare_dir / f"comparison_bar_{lang}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # 雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, branch in enumerate(branches):
            values = [results[branch].get(m, 0) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=branch,
                   color=colors_list[i % len(colors_list)])
            ax.fill(angles, values, alpha=0.25, color=colors_list[i % len(colors_list)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        fig.savefig(compare_dir / f"comparison_radar_{lang}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  保存: comparison_bar_{lang}.png, comparison_radar_{lang}.png")
    
    # 保存汇总
    summary = {
        'results': results, 
        'best_branch': max(results.keys(), key=lambda b: results[b].get('test_f1', 0)),
        'best_f1': max(results[b].get('test_f1', 0) for b in results.keys()),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(compare_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n【对比可视化完成】")
    print(f"  最佳支线: {summary['best_branch']} (F1={summary['best_f1']:.4f})")
    print(f"  保存目录: {compare_dir}")


def run_all_branches(cfg_base: ThreeStageConfigV4):
    """运行全部支线对比实验"""
    print("\n" + "="*70)
    print("全支线对比实验")
    print("="*70)
    
    # 定义要运行的支线配置
    branch_configs = [
        ('hetero', 'concat'),
        ('zerone', 'concat'),
        ('dual', 'concat'),
        ('dual', 'attention'),
        ('dual', 'gate'),
    ]
    
    for branch_mode, fusion_mode in branch_configs:
        print(f"\n{'='*60}")
        print(f"运行支线: {branch_mode.upper()} (融合: {fusion_mode})")
        print(f"{'='*60}")
        
        cfg = ThreeStageConfigV4(
            PROJECT_ROOT=cfg_base.PROJECT_ROOT,
            OUTPUT_ROOT=cfg_base.OUTPUT_ROOT,
            BRANCH_MODE=branch_mode,
            FUSION_MODE=fusion_mode,
            STRICT_DATA_SEPARATION=cfg_base.STRICT_DATA_SEPARATION,
        )
        cfg.__post_init__()
        cfg.print_config()
        
        # 运行三阶段
        model, _ = train_stage1(cfg)
        if model is not None:
            pseudo_labels = run_stage2(model, cfg)
            train_stage3(model, pseudo_labels, cfg)
    
    # 生成对比可视化
    generate_comparison_visualization(cfg_base.OUTPUT_ROOT)


# =============================================================================
# 第15步: 断点续训功能
# =============================================================================

def resume_training(cfg: ThreeStageConfigV4):
    """从检查点恢复训练"""
    print("\n" + "="*70)
    print("断点续训")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 检查阶段1状态
    stage1_best = cfg.MODEL_DIR / "stage1_best.pth"
    stage1_ckpt_dir = cfg.CHECKPOINT_DIR / "stage1"
    
    latest_ckpt = None
    if stage1_ckpt_dir.exists():
        ckpts = sorted(stage1_ckpt_dir.glob("checkpoint_epoch*.pth"))
        if ckpts:
            latest_ckpt = ckpts[-1]
    
    # 判断是否需要继续阶段1
    if not stage1_best.exists() or latest_ckpt:
        print("\n[继续] 阶段1训练...")
        model, _ = train_stage1(cfg, resume_from=latest_ckpt)
    else:
        print("\n[加载] 阶段1模型已存在")
        model = AnomalyModelV4(cfg).to(device)
        ckpt = torch.load(stage1_best, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.center = ckpt['center']
    
    if model is None:
        print("[错误] 模型加载失败")
        return
    
    # 阶段2
    pseudo_path = cfg.STAGE2_DIR / "pseudo_labels.npz"
    if not pseudo_path.exists():
        print("\n[继续] 阶段2伪标签生成...")
        pseudo_labels = run_stage2(model, cfg)
    else:
        print("\n[加载] 阶段2伪标签已存在")
        data = np.load(pseudo_path, allow_pickle=True)
        pseudo_labels = {k: data[k] for k in data.files}
    
    # 阶段3
    stage3_eval = cfg.STAGE3_DIR / "test_evaluation.json"
    if not stage3_eval.exists():
        print("\n[继续] 阶段3监督微调...")
        train_stage3(model, pseudo_labels, cfg)
    else:
        print("\n[完成] 阶段3已完成")
    
    print("\n【断点续训完成】")


# =============================================================================
# 第16步: 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='三阶段变压器故障诊断 V4')
    
    # 基本参数
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--all_branches', action='store_true', help='运行全部支线对比')
    
    # 支线配置
    parser.add_argument('--branch', type=str, choices=['hetero', 'zerone', 'dual'], 
                       default='dual', help='支线模式')
    parser.add_argument('--fusion_mode', type=str, choices=['concat', 'attention', 'gate'],
                       default='concat', help='融合策略 (仅dual模式有效)')
    
    # 路径配置
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--output', type=str, default='./three_stage_results_v4', help='输出目录')
    
    # 功能开关
    parser.add_argument('--test_data', action='store_true', help='测试数据加载')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--compare', action='store_true', help='生成对比可视化')
    parser.add_argument('--compare_only', action='store_true', help='只生成对比可视化')
    parser.add_argument('--status', action='store_true', help='只显示状态')
    parser.add_argument('--no_strict', action='store_true', help='禁用严格数据分离')
    
    args = parser.parse_args()
    
    # 初始化配置
    cfg = ThreeStageConfigV4(
        BRANCH_MODE=args.branch,
        FUSION_MODE=args.fusion_mode,
        STRICT_DATA_SEPARATION=not args.no_strict,
    )
    
    if args.data_root:
        cfg.PROJECT_ROOT = Path(args.data_root)
    cfg.OUTPUT_ROOT = Path(args.output)
    cfg.__post_init__()
    
    # 只显示状态
    if args.status:
        status = check_branch_status(cfg.OUTPUT_ROOT)
        print_status(status)
        return
    
    # 只生成对比
    if args.compare_only:
        generate_comparison_visualization(cfg.OUTPUT_ROOT)
        return
    
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
            print(f"  Zerone特征: {zr.shape}")
            print(f"  标签: {lbl}")
            print(f"  非零元素 (Hetero): {(img != 0).sum().item()}")
            print(f"  非零元素 (Zerone): {(zr != 0).sum().item()}")
        return
    
    # 断点续训
    if args.resume:
        resume_training(cfg)
        if args.compare:
            generate_comparison_visualization(cfg.OUTPUT_ROOT)
        return
    
    # 全支线对比
    if args.all_branches:
        run_all_branches(cfg)
        return
    
    # 单支线执行
    if args.all or args.stage == 1:
        model, _ = train_stage1(cfg)
    else:
        model_path = cfg.MODEL_DIR / "stage1_best.pth"
        if model_path.exists():
            model = AnomalyModelV4(cfg)
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
    
    # 生成对比
    if args.compare:
        generate_comparison_visualization(cfg.OUTPUT_ROOT)
    
    print("\n【完成】结果保存至:", cfg.BRANCH_DIR)


if __name__ == "__main__":
    main()

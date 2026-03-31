# -*- coding: utf-8 -*-
"""
transformer_three_stage_v3.py
==============================

交流变压器振动数据 三阶段渐进式故障诊断系统 V3
支持三条并行支线：Hetero-Only / Zerone-Only / Dual-Branch (融合)

【V3版本核心改进】
    ✅ 三条支线并行实验：
       - Hetero-Only: 仅用三通道时频图像 (CWT+STFT+Context)
       - Zerone-Only: 仅用1200维工程特征
       - Dual-Branch: 双分支融合 (V2的baseline)
    ✅ 严格的标签使用规则：
       - TRAIN + TEST: 阶段1/2中作为无标签数据
       - VAL: 唯一可用于阈值选择/监督微调的带标签数据
       - TEST标签: 只用于最终评估，不反向影响模型
    ✅ 丰富的可视化与监控：
       - 定期检查点 (每5轮，最多保留5个)
       - 丰富可视化 (每3轮生成)
       - 完整CSV训练日志
       - 错误样本溯源与可视化
       - 特征相关性分析
       - 中英文双版本图片 (IEEE/Nature风格)

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
            │        │                    │        │
            │        └────────┬───────────┘        │
            ▼                 ▼                    ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Hetero-Only  │  │  Dual-Branch │  │ Zerone-Only  │
    │  (支线A)     │  │   (支线C)    │  │   (支线B)    │
    └──────────────┘  └──────────────┘  └──────────────┘

【运行方式】
    # 测试数据加载
    python transformer_three_stage_v3.py --test_data
    
    # 运行单一支线
    python transformer_three_stage_v3.py --branch hetero --all
    python transformer_three_stage_v3.py --branch zerone --all
    python transformer_three_stage_v3.py --branch dual --all
    
    # 运行全部支线对比
    python transformer_three_stage_v3.py --all_branches
    
    # 分阶段运行
    python transformer_three_stage_v3.py --branch dual --stage 1
    python transformer_three_stage_v3.py --branch dual --stage 2
    python transformer_three_stage_v3.py --branch dual --stage 3

Author: 基于 zerone + hetero 代码框架整合 (V3)
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
    }
}


# =============================================================================
# 第2步: 配置类定义
# =============================================================================
@dataclass
class ThreeStageConfigV3:
    """
    三阶段诊断系统配置类 (V3版本)
    
    【核心改进】
        BRANCH_MODE: 支线模式 ('hetero' / 'zerone' / 'dual')
        STRICT_LABEL_RULE: 严格标签使用规则开关
    """
    
    # ================= 路径配置 =================
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v3"))
    
    # ================= 支线模式 (V3新增) =================
    BRANCH_MODE: str = "dual"  # 'hetero' / 'zerone' / 'dual'
    
    # ================= 严格标签规则 (V3新增) =================
    STRICT_LABEL_RULE: bool = True  # True: VAL为唯一有标签数据源
    
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
    FUSION_MODE: str = "concat" # 融合方式: concat/attention/gate
    
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
    
    # ================= 检查点与可视化 (V3新增) =================
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
        
        # 根据支线模式设置输出子目录
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
        print("三阶段故障诊断系统配置 (V3版本)")
        print("="*70)
        print(f"【支线模式】")
        branch_names = {'hetero': '图像分支(Hetero)', 'zerone': '特征分支(Zerone)', 'dual': '双分支融合'}
        print(f"  当前支线: {branch_names.get(self.BRANCH_MODE, self.BRANCH_MODE)}")
        print(f"【数据路径】")
        print(f"  项目根目录: {self.PROJECT_ROOT}")
        print(f"  输出目录: {self.BRANCH_DIR}")
        print(f"【标签规则】")
        print(f"  严格模式: {'✅ 启用 (VAL为唯一标签源)' if self.STRICT_LABEL_RULE else '❌ 禁用'}")
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
# 第3步: 可视化工具类
# =============================================================================

class VisualizationManager:
    """
    可视化管理器
    
    负责生成中英文双版本的高质量图片 (IEEE/Nature风格)
    """
    
    def __init__(self, cfg: ThreeStageConfigV3):
        self.cfg = cfg
        self.setup_style()
    
    def setup_style(self):
        """设置IEEE/Nature风格"""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SimHei', 'Microsoft YaHei','Arial', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei'],
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
    
    def save_dual_lang(self, fig, base_path: Path, close: bool = True):
        """保存中英文双版本图片"""
        # 中文版
        cn_path = base_path.with_name(base_path.stem + "_cn" + base_path.suffix)
        fig.savefig(cn_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # 英文版 (需要重新绘制)
        en_path = base_path.with_name(base_path.stem + "_en" + base_path.suffix)
        fig.savefig(en_path, dpi=self.cfg.VIZ_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        if close:
            plt.close(fig)
        
        return cn_path, en_path
    
    def plot_training_curves(self, history: Dict, stage: str, lang: str = 'cn'):
        """
        绘制训练曲线
        
        参数:
            history: 训练历史字典
            stage: 阶段名称 ('stage1' / 'stage3')
            lang: 语言 ('cn' / 'en')
        """
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
        """
        绘制异常得分分布
        
        参数:
            scores: 异常得分数组
            t_normal: 正常阈值
            t_anomaly: 异常阈值
            labels: 真实标签 (可选，用于着色)
            lang: 语言
        """
        L = LABELS[lang]
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if labels is not None and len(labels) == len(scores):
            # 按标签分开绘制
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
        
        # 阈值线
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
        
        # 添加颜色条
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        # 标签
        classes = [L['normal'], L['fault']]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        
        # 数值标注
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
        
        # 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 按标签绘制
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
            # 原始
            axes[0, i].imshow(original[i], cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'{name} - {"Original" if lang == "en" else "原始"}')
            axes[0, i].axis('off')
            
            # 重构
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
        
        # 归一化得分用于颜色映射
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # 绘制散点
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                            c=norm_scores, cmap='RdYlGn_r', alpha=0.6, s=30,
                            edgecolors='white', linewidth=0.5)
        
        # 绘制中心
        ax.scatter(center_2d[0], center_2d[1], c='black', marker='X', s=200,
                  edgecolors='white', linewidth=2, label='Center' if lang == 'en' else '中心')
        
        # 颜色条
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
    
    def plot_feature_correlation(self, features: np.ndarray, feature_names: List[str],
                                 lang: str = 'cn'):
        """绘制特征相关性热力图"""
        L = LABELS[lang]
        
        # 计算相关矩阵
        corr_matrix = np.corrcoef(features.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # 颜色条
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        
        # 标签
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        
        ax.set_title(L['feature_analysis'] + ' - ' + L['correlation'])
        
        plt.tight_layout()
        save_path = self.cfg.VIZ_SUBDIRS["feature_analysis"] / f"feature_correlation_{lang}.png"
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
                # 显示图像 (取第一个通道)
                img = info['image']
                if img.ndim == 3:
                    img = img[0]
                ax.imshow(img, cmap='viridis', aspect='auto')
            
            true_label = L['normal'] if info['true'] == 0 else L['fault']
            pred_label = L['normal'] if info['pred'] == 0 else L['fault']
            score = info.get('score', 0)
            
            ax.set_title(f"True: {true_label}, Pred: {pred_label}\nScore: {score:.3f}", fontsize=9)
            ax.axis('off')
        
        # 隐藏多余的子图
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
            
            # 显示三通道合成或第一通道
            img = images[i]
            if img.ndim == 3 and img.shape[0] == 3:
                # RGB合成
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
# 第4步: 日志管理器
# =============================================================================

class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, cfg: ThreeStageConfigV3, stage: str):
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
            
            # 表头
            headers = list(self.history.keys())
            writer.writerow(headers)
            
            # 数据行
            n_rows = max(len(v) for v in self.history.values())
            for i in range(n_rows):
                row = [self.history[h][i] if i < len(self.history[h]) else '' for h in headers]
                writer.writerow(row)
        
        print(f"[日志] 训练日志已保存: {self.log_file}")
    
    def get_history(self) -> Dict:
        """获取历史记录"""
        return dict(self.history)


# =============================================================================
# 第5步: 检查点管理器
# =============================================================================

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, cfg: ThreeStageConfigV3, stage: str):
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
        
        # 保留最新的N个检查点
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
# 第6步: 特征提取函数
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
    """
    计算STFT段均值特征 (127维)
    """
    try:
        _, _, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx[1:, :])  # 去DC
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
        
        # 插值到1Hz分辨率
        target_freqs = np.arange(1, 2001, 1)
        psd_interp = np.interp(target_freqs, freqs, psd)
        
        # 分段
        psd_low = psd_interp[:1000]  # 1-1000Hz
        psd_high_raw = psd_interp[1000:2000]  # 1001-2000Hz
        
        # 高频段每20Hz聚合
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
            
            # 幅值比 (近似)
            amp_ratio = np.sqrt(hf_power / (total_power + 1e-12))
            # 功率比
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
    
    # 时域特征
    features.append(compute_time_features(sig))
    
    # STFT特征
    features.append(compute_stft_features(sig, fs))
    
    # PSD特征
    features.append(compute_psd_features(sig, fs))
    
    # 高频特征
    features.append(compute_hf_features(sig, fs))
    
    feat_vec = np.concatenate(features).astype(np.float32)
    
    # 确保维度正确
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
    # Z-score归一化
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    
    # ----- Ch0: CWT -----
    scales = np.arange(1, min(129, len(sig)//64 + 1))
    try:
        cwt_matrix, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
        cwt_abs = np.log1p(np.abs(cwt_matrix).astype(np.float32))
        c0 = cv2.resize(cwt_abs, (size, size), interpolation=cv2.INTER_LINEAR)
        c0 = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)
    except Exception:
        c0 = np.zeros((size, size), dtype=np.float32)
    
    # ----- Ch1: STFT -----
    try:
        nperseg = min(256, len(sig)//4)
        _, _, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        mag = np.log1p(np.abs(Zxx).astype(np.float32))
        c1 = cv2.resize(mag, (size, size), interpolation=cv2.INTER_LINEAR)
        c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-8)
    except Exception:
        c1 = np.zeros((size, size), dtype=np.float32)
    
    # ----- Ch2: Context (波形折叠) -----
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
# 第7步: 数据读取工具
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
# 第8步: 数据集类
# =============================================================================

class TransformerVibrationDataset(Dataset):
    """
    变压器振动数据集 (V3版本)
    
    支持三种输出模式:
        - hetero: 仅输出Hetero三通道图像
        - zerone: 仅输出Zerone 1200维特征
        - dual: 同时输出两者
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        cfg: ThreeStageConfigV3,
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
            
            # 获取标签
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
            
            # 按时间戳分组
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
            image: (3, H, W) 或 zeros (根据支线模式)
            zerone: (1200,) 或 zeros (根据支线模式)
            label: 标签 (-1表示无标签)
            idx: 索引
        """
        fp, time_key, sig_list, label = self.samples[idx]
        sig = self._aggregate_channels(sig_list)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        
        # 根据支线模式决定输出
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
# 第9步: 模型定义
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


class BranchEncoder(nn.Module):
    """
    支线编码器 (V3版本)
    
    支持三种模式:
        - hetero: 仅CNN
        - zerone: 仅MLP
        - dual: CNN + MLP + 融合
    """
    
    def __init__(self, cfg: ThreeStageConfigV3):
        super().__init__()
        self.cfg = cfg
        self.branch_mode = cfg.BRANCH_MODE
        
        if self.branch_mode in ['hetero', 'dual']:
            self.hetero_branch = HeteroCNN(output_dim=cfg.CNN_FEAT_DIM)
        
        if self.branch_mode in ['zerone', 'dual']:
            self.zerone_branch = ZeroneMLP(
                input_dim=cfg.ZERONE_DIM,
                output_dim=cfg.MLP_FEAT_DIM
            )
        
        # 确定输出维度
        if self.branch_mode == 'hetero':
            self.output_dim = cfg.CNN_FEAT_DIM  # 512
        elif self.branch_mode == 'zerone':
            self.output_dim = cfg.MLP_FEAT_DIM  # 256
            # 添加投影层使输出维度一致
            self.proj = nn.Sequential(
                nn.Linear(cfg.MLP_FEAT_DIM, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.output_dim = 512
        else:  # dual
            fused_dim = cfg.CNN_FEAT_DIM + cfg.MLP_FEAT_DIM  # 768
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.output_dim = 512
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            image: (B, 3, H, W)
            zerone: (B, 1200)
        返回:
            (B, 512) 编码特征
        """
        if self.branch_mode == 'hetero':
            return self.hetero_branch(image)
        
        elif self.branch_mode == 'zerone':
            h = self.zerone_branch(zerone)
            return self.proj(h)
        
        else:  # dual
            h_img = self.hetero_branch(image)
            h_zr = self.zerone_branch(zerone)
            h_cat = torch.cat([h_img, h_zr], dim=1)
            return self.fusion(h_cat)


class AnomalyModelV3(nn.Module):
    """
    异常检测模型 V3
    
    组件:
        - 支线编码器 (根据模式选择)
        - Deep SVDD头
        - VAE解码器 (可选，仅hetero/dual模式)
    """
    
    def __init__(self, cfg: ThreeStageConfigV3):
        super().__init__()
        self.cfg = cfg
        
        # 编码器
        self.encoder = BranchEncoder(cfg)
        
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
    
    def encode(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(image, zerone)
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        h = self.encode(image, zerone)
        
        # SVDD
        z_svdd = self.svdd_proj(h)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        result = {
            'h': h,
            'z_svdd': z_svdd,
            'svdd_score': svdd_score,
        }
        
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
                h = self.encode(img, zr)
                z = self.svdd_proj(h)
                c += z.sum(0)
                n += z.size(0)
        
        c /= n
        
        # 避免中心在原点附近
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c
        
        print(f"[SVDD] 中心初始化完成，范数: {c.norm().item():.4f}")
    
    def anomaly_score(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """计算异常得分"""
        out = self.forward(image, zerone)
        
        if self.has_vae:
            # 综合得分
            svdd_score = out['svdd_score']
            vae_score = out['vae_recon_loss'] + 0.01 * out['vae_kl']
            
            # 归一化
            svdd_norm = svdd_score / (svdd_score.mean() + 1e-8)
            vae_norm = vae_score / (vae_score.mean() + 1e-8)
            
            return self.alpha * svdd_norm + (1 - self.alpha) * vae_norm
        else:
            return out['svdd_score']


class FaultClassifier(nn.Module):
    """故障分类器 (阶段三)"""
    
    def __init__(self, encoder: BranchEncoder, num_classes: int = 2, freeze_encoder: bool = True):
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
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        h = self.encoder(image, zerone)
        return self.classifier(h)
    
    def unfreeze_encoder(self, num_layers: int = 1):
        """解冻编码器最后几层"""
        # 简化版：全部解冻
        for param in self.encoder.parameters():
            param.requires_grad = True


# =============================================================================
# 第10步: 阶段一 - 无监督训练
# =============================================================================

def train_stage1(cfg: ThreeStageConfigV3) -> Tuple[AnomalyModelV3, Dict]:
    """
    阶段一：无监督学习
    
    数据使用:
        - TRAIN + VAL (但不使用标签，当作无标签数据)
        - 如果 STRICT_LABEL_RULE=True，也将 TEST 作为无标签数据参与训练
    
    训练流程:
        1. VAE预训练 (如果启用)
        2. SVDD中心初始化
        3. 联合训练 SVDD + VAE
    """
    print("\n" + "="*70)
    print("阶段一：无监督学习")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 初始化工具
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage1")
    ckpt_mgr = CheckpointManager(cfg, "stage1")
    
    # 加载数据 (无标签)
    print("\n[1/4] 加载数据...")
    train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
    val_ds = TransformerVibrationDataset(cfg.VAL_DIR, cfg, use_labels=False, split_name="VAL")
    
    # 合并数据集
    combined_ds = ConcatDataset([train_ds, val_ds])
    print(f"  合并数据集大小: {len(combined_ds)}")
    
    train_loader = DataLoader(combined_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
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
    model = AnomalyModelV3(cfg).to(device)
    print(f"  支线模式: {cfg.BRANCH_MODE}")
    print(f"  VAE启用: {model.has_vae}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE1_EPOCHS)
    
    # VAE预训练 (如果启用)
    if model.has_vae:
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
    print("\n[3/4] 初始化SVDD中心...")
    model.init_center(train_loader, device)
    
    # 联合训练
    print(f"\n[4/4] 联合训练 ({cfg.STAGE1_EPOCHS}轮)...")
    
    history = {
        'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': [], 'recon_loss': []
    }
    best_loss = float('inf')
    
    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        epoch_svdd, epoch_vae, epoch_total, epoch_recon = 0, 0, 0, 0
        
        # Beta warmup
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch / max(cfg.BETA_WARMUP, 1)))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False)
        for batch in pbar:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            
            # SVDD损失
            svdd_loss = out['svdd_score'].mean()
            
            # VAE损失
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
        
        # 日志记录
        logger.log(epoch=epoch+1, svdd_loss=avg_svdd, vae_loss=avg_vae, 
                   total_loss=avg_total, recon_loss=avg_recon, lr=scheduler.get_last_lr()[0])
        
        # 保存最佳模型
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.center,
                'epoch': epoch,
                'loss': best_loss,
            }, cfg.MODEL_DIR / "stage1_best.pth")
        
        # 定期检查点
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(model, optimizer, epoch + 1, 
                         {'svdd_loss': avg_svdd, 'total_loss': avg_total}, scheduler)
        
        # 定期可视化
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage1", lang=lang)
            
            # 重构可视化 (如果启用VAE)
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
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
    
    # 保存日志
    logger.save_csv()
    
    # 最终可视化
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
    
    # PCA降维
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    center_2d = pca.transform(model.center.cpu().numpy().reshape(1, -1))[0]
    
    for lang in cfg.LANGS:
        viz.plot_svdd_sphere(features_2d, all_scores, center_2d, lang=lang)
    
    print(f"\n【阶段一完成】最佳损失: {best_loss:.4f}")
    print(f"  模型保存: {cfg.MODEL_DIR / 'stage1_best.pth'}")
    
    return model, history


# =============================================================================
# 第11步: 阶段二 - 伪标签生成
# =============================================================================

def run_stage2(model: AnomalyModelV3, cfg: ThreeStageConfigV3) -> Dict:
    """
    阶段二：基于异常得分生成伪标签
    
    数据使用:
        - TRAIN + VAL (无标签，用于计算得分分布)
    
    输出:
        - 伪标签集合 (pseudo_normal, pseudo_anomaly, uncertain)
        - 异常得分分布可视化
    """
    print("\n" + "="*70)
    print("阶段二：伪标签生成")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    viz = VisualizationManager(cfg)
    
    # 加载数据 (无标签)
    print("\n[1/3] 加载数据...")
    train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
    val_ds = TransformerVibrationDataset(cfg.VAL_DIR, cfg, use_labels=False, split_name="VAL")
    combined_ds = ConcatDataset([train_ds, val_ds])
    
    loader = DataLoader(combined_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
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
# 第12步: 阶段三 - 监督微调 (严格标签规则)
# =============================================================================

def train_stage3(model: AnomalyModelV3, pseudo_labels: Dict, cfg: ThreeStageConfigV3) -> FaultClassifier:
    """
    阶段三：有监督微调
    
    【严格标签规则】
        - 训练数据: VAL (有标签)
        - 评估数据: TEST (有标签，但只用于评估，不反向影响模型)
    
    训练流程:
        1. 使用VAL数据训练分类器
        2. 在VAL上做交叉验证
        3. 最终在TEST上评估
    """
    print("\n" + "="*70)
    print("阶段三：有监督微调 (严格标签规则)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage3")
    ckpt_mgr = CheckpointManager(cfg, "stage3")
    
    # 加载VAL数据 (有标签，用于训练)
    print("\n[1/4] 加载数据...")
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
    classifier = FaultClassifier(model.encoder, num_classes=2, freeze_encoder=True).to(device)
    
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
            
            logits = classifier(img, zr)
            loss = criterion(logits, label)
            
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
                logits = classifier(img, zr)
                probs = F.softmax(logits, dim=1)
                
                val_preds.extend(logits.argmax(dim=1).cpu().tolist())
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
        
        # 日志记录
        logger.log(epoch=epoch+1, train_loss=avg_train_loss, val_acc=val_acc,
                  val_f1=val_f1, val_precision=val_prec, val_recall=val_rec)
        
        # 保存最佳模型
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
        
        # # 早停
        # if patience_counter >= cfg.PATIENCE:
        #     print(f"\n  [早停] 验证F1连续{cfg.PATIENCE}轮未提升")
        #     break
        
        # 解冻编码器
        if epoch == 15:
            classifier.unfreeze_encoder(1)
            print("  [*] 解冻编码器")
        
        # 定期检查点
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(classifier, optimizer, epoch + 1,
                         {'val_f1': val_f1, 'val_acc': val_acc}, scheduler)
        
        # 定期可视化
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage3", lang=lang)
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    # 保存日志
    logger.save_csv()
    
    # 加载最佳模型
    best_ckpt = torch.load(cfg.MODEL_DIR / "stage3_best.pth", map_location=device)
    classifier.load_state_dict(best_ckpt['model_state'])
    
    # 最终可视化
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage3", lang=lang)
    
    # ==================== TEST评估 (仅评估，不训练) ====================
    print("\n[4/4] 在TEST上评估...")
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
                
                h = classifier.encoder(img, zr)
                logits = classifier.classifier(h)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(label.tolist())
                test_probs.extend(probs[:, 1].cpu().tolist())
                test_features.append(h.cpu().numpy())
                
                # 收集错误样本
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
        }
        
        with open(cfg.STAGE3_DIR / "test_evaluation.json", 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n【阶段三完成】最佳验证F1: {best_f1:.4f}")
    
    return classifier


# =============================================================================
# 第13步: 支线对比实验
# =============================================================================

def run_all_branches(cfg_base: ThreeStageConfigV3):
    """
    运行全部三条支线并对比
    """
    print("\n" + "="*70)
    print("全支线对比实验")
    print("="*70)
    
    branches = ['hetero', 'zerone', 'dual']
    results = {}
    
    for branch in branches:
        print(f"\n{'='*60}")
        print(f"运行支线: {branch.upper()}")
        print(f"{'='*60}")
        
        # 创建该支线的配置
        cfg = ThreeStageConfigV3(
            PROJECT_ROOT=cfg_base.PROJECT_ROOT,
            OUTPUT_ROOT=cfg_base.OUTPUT_ROOT,
            BRANCH_MODE=branch,
            STRICT_LABEL_RULE=cfg_base.STRICT_LABEL_RULE,
        )
        cfg.__post_init__()
        cfg.print_config()
        
        # 运行三阶段
        model, _ = train_stage1(cfg)
        pseudo_labels = run_stage2(model, cfg)
        classifier = train_stage3(model, pseudo_labels, cfg)
        
        # 读取评估结果
        eval_file = cfg.STAGE3_DIR / "test_evaluation.json"
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                results[branch] = json.load(f)
        else:
            results[branch] = {'test_f1': 0, 'test_acc': 0}
    
    # 生成对比报告
    print("\n" + "="*70)
    print("支线对比结果")
    print("="*70)
    print(f"{'支线':<15} {'准确率':<12} {'F1分数':<12} {'精确率':<12} {'召回率':<12}")
    print("-"*63)
    for branch, res in results.items():
        print(f"{branch:<15} {res.get('test_acc', 0):<12.4f} {res.get('test_f1', 0):<12.4f} "
              f"{res.get('test_precision', 0):<12.4f} {res.get('test_recall', 0):<12.4f}")
    
    # 保存对比结果
    comparison_path = cfg_base.OUTPUT_ROOT / "branch_comparison.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 绘制对比图
    viz = VisualizationManager(cfg_base)
    
    for lang in ['cn', 'en']:
        L = LABELS[lang]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(branches))
        width = 0.2
        
        metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
        metric_names = [L['accuracy'], L['f1'], L['precision'], L['recall']]
        colors_list = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]
        
        for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors_list)):
            values = [results[b].get(metric, 0) for b in branches]
            ax.bar(x + i * width, values, width, label=name, color=color)
        
        branch_names = [L.get(b, b) for b in branches]
        ax.set_xlabel('Branch' if lang == 'en' else '支线')
        ax.set_ylabel('Score' if lang == 'en' else '得分')
        ax.set_title('Branch Comparison' if lang == 'en' else '支线对比')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(branch_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(cfg_base.OUTPUT_ROOT / f"branch_comparison_{lang}.png",
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print(f"\n【对比实验完成】结果保存: {comparison_path}")
    
    return results


# =============================================================================
# 第14步: 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='三阶段变压器故障诊断 V3')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--all_branches', action='store_true', help='运行全部支线对比')
    parser.add_argument('--branch', type=str, choices=['hetero', 'zerone', 'dual'], 
                       default='dual', help='支线模式')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--output', type=str, default='./three_stage_results_v3', help='输出目录')
    parser.add_argument('--test_data', action='store_true', help='测试数据加载')
    parser.add_argument('--no_strict', action='store_true', help='禁用严格标签规则')
    
    args = parser.parse_args()
    
    # 初始化配置
    cfg = ThreeStageConfigV3(
        BRANCH_MODE=args.branch,
        STRICT_LABEL_RULE=not args.no_strict,
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
            print(f"  Hetero图像: {img.shape}")
            print(f"  Zerone特征: {zr.shape}")
            print(f"  标签: {lbl}")
            print(f"  非零元素 (Hetero): {(img != 0).sum().item()}")
            print(f"  非零元素 (Zerone): {(zr != 0).sum().item()}")
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
            model = AnomalyModelV3(cfg)
            ckpt = torch.load(model_path, map_location=cfg.DEVICE)
            model.load_state_dict(ckpt['model_state'])
            model.center = ckpt['center']
            model = model.to(cfg.DEVICE)
            print(f"[加载模型] {model_path}")
        else:
            print(f"[错误] 未找到模型: {model_path}")
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

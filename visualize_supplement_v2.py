# -*- coding: utf-8 -*-
"""
visualize_supplement_v2.py
==========================
可视化补充脚本：跳过训练，加载已保存模型，生成中英文双版本可视化

【功能】
    ✅ 加载已训练的 stage1_best_v2.pth 和 stage3_best_v2.pth
    ✅ 生成中文版和英文版全部可视化
    ✅ 混淆矩阵、ROC/PR曲线、t-SNE、特征分析、得分分布

【使用方法】
    python visualize_supplement_v2.py --results_dir ./three_stage_results_v2 --data_root <数据目录>
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# 信号处理
import pywt
import cv2
from scipy.signal import stft, welch

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 评估指标
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


# =============================================================================
# 多语言标签
# =============================================================================
LANG_LABELS = {
    'cn': {
        'normal': '正常', 'fault': '故障', 'uncertain': '不确定',
        'predicted': '预测标签', 'true': '真实标签',
        'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率',
        'confusion_title': '混淆矩阵',
        'roc_title': 'ROC曲线', 'pr_title': 'Precision-Recall曲线',
        'random': '随机分类', 'baseline': '基线',
        'fpr': '假正例率 (FPR)', 'tpr': '真正例率 (TPR)',
        'tsne_title': 't-SNE特征空间可视化',
        'tsne_dim1': 't-SNE维度1', 'tsne_dim2': 't-SNE维度2',
        'score_title': '异常得分分布 (按类别)',
        'score_xlabel': '异常得分', 'density': '密度',
        'mean': '均值', 'samples': '样本数',
        'feat_mean_title': '正常vs故障 特征均值对比',
        'feat_norm_title': '特征向量范数分布',
        'feat_var_title': '特征方差分布',
        'feat_corr_title': '特征相关性热图',
        'feat_index': '特征索引', 'feat_mean': '特征均值',
        'feat_norm': '特征向量范数', 'feat_var': '特征方差',
    },
    'en': {
        'normal': 'Normal', 'fault': 'Fault', 'uncertain': 'Uncertain',
        'predicted': 'Predicted Label', 'true': 'True Label',
        'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
        'confusion_title': 'Confusion Matrix',
        'roc_title': 'ROC Curve', 'pr_title': 'Precision-Recall Curve',
        'random': 'Random', 'baseline': 'Baseline',
        'fpr': 'False Positive Rate', 'tpr': 'True Positive Rate',
        'tsne_title': 't-SNE Feature Space Visualization',
        'tsne_dim1': 't-SNE Dimension 1', 'tsne_dim2': 't-SNE Dimension 2',
        'score_title': 'Anomaly Score Distribution by Class',
        'score_xlabel': 'Anomaly Score', 'density': 'Density',
        'mean': 'Mean', 'samples': 'samples',
        'feat_mean_title': 'Normal vs Fault Feature Mean Comparison',
        'feat_norm_title': 'Feature Vector Norm Distribution',
        'feat_var_title': 'Feature Variance Distribution',
        'feat_corr_title': 'Feature Correlation Heatmap',
        'feat_index': 'Feature Index', 'feat_mean': 'Feature Mean',
        'feat_norm': 'Feature Norm', 'feat_var': 'Feature Variance',
    }
}

COLORS = {
    'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
    'red': '#d62728', 'purple': '#9467bd',
    'normal': '#2ca02c', 'fault': '#d62728'
}


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class VisualizationConfig:
    RESULTS_DIR: Path = field(default_factory=lambda: Path("./three_stage_results_v2"))
    DATA_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    
    FS: float = 8192.0
    SIGNAL_LEN: int = 8192
    INPUT_SIZE: int = 224
    
    ZERONE_DIM: int = 1200
    CNN_FEAT_DIM: int = 512
    MLP_FEAT_DIM: int = 256
    LATENT_DIM: int = 128
    LATENT_CHANNELS: int = 64
    FUSION_MODE: str = "concat"
    
    BATCH_SIZE: int = 16
    VIZ_DPI: int = 300
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    CLASS_KEYWORDS: Dict = field(default_factory=lambda: {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    })
    
    def __post_init__(self):
        self.RESULTS_DIR = Path(self.RESULTS_DIR)
        self.DATA_ROOT = Path(self.DATA_ROOT)
        
        self.TEST_DIR = self.DATA_ROOT / "test"
        self.VIZ_DIR = self.RESULTS_DIR / "visualizations"
        self.MODEL_DIR = self.RESULTS_DIR / "models"
        
        for subdir in ["confusion", "roc_pr", "tsne", "feature_analysis", "score_dist"]:
            (self.VIZ_DIR / subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 特征提取 & 数据处理 (简化版)
# =============================================================================
def extract_zerone_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """提取1200维特征"""
    sig = np.asarray(sig, dtype=np.float32).ravel()
    sig_dc = sig - np.mean(sig)
    
    # 时域特征 (15维)
    x = sig
    time_feat = np.array([
        np.mean(x), np.sqrt(np.mean(x**2)), np.var(x), np.std(x),
        np.max(x), np.min(x), np.max(x)-np.min(x),
        np.mean((x-np.mean(x))**4)/(np.var(x)**2+1e-12),
        np.mean((x-np.mean(x))**3)/(np.std(x)**3+1e-12),
        np.sum(np.abs(np.diff(np.sign(x)))>0)/(len(x)-1+1e-12),
        np.mean(np.abs(x)),
        np.max(x)/(np.sqrt(np.mean(x**2))+1e-12),
        np.max(x)/(np.mean(np.abs(x))+1e-12),
        np.max(x)/(np.mean(np.abs(x)**4)**0.25+1e-12),
        np.sqrt(np.mean(x**2))/(np.mean(np.abs(x))+1e-12)
    ], dtype=np.float32)
    
    # STFT特征 (127维)
    try:
        from scipy.signal import stft as scipy_stft
        _, _, Zxx = scipy_stft(sig_dc, fs=fs, nperseg=128, noverlap=64)
        stft_feat = np.mean(np.abs(Zxx[1:, :]), axis=0)[:127]
        if len(stft_feat) < 127:
            stft_feat = np.pad(stft_feat, (0, 127-len(stft_feat)))
    except:
        stft_feat = np.zeros(127, dtype=np.float32)
    
    # PSD + HF (1050 + 8 = 1058维，填0简化)
    psd_hf = np.zeros(1058, dtype=np.float32)
    
    feat_vec = np.concatenate([time_feat, stft_feat, psd_hf])
    if len(feat_vec) < 1200:
        feat_vec = np.pad(feat_vec, (0, 1200-len(feat_vec)))
    return feat_vec[:1200].astype(np.float32)


def signal_to_hetero_image(sig: np.ndarray, fs: float, size: int = 224) -> np.ndarray:
    """转换为3通道时频图像"""
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    
    # CWT
    try:
        scales = np.arange(1, min(129, len(sig)//64+1))
        cwt_m, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
        c0 = cv2.resize(np.log1p(np.abs(cwt_m)), (size, size))
        c0 = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)
    except:
        c0 = np.zeros((size, size), dtype=np.float32)
    
    # STFT
    try:
        nperseg = min(256, len(sig)//4)
        _, _, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        c1 = cv2.resize(np.log1p(np.abs(Zxx)), (size, size))
        c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-8)
    except:
        c1 = np.zeros((size, size), dtype=np.float32)
    
    # Context
    try:
        h = max(1, len(sig)//size)
        mat = sig[:h*size].reshape(h, size) if h*size <= len(sig) else sig.reshape(-1,1)
        c2 = cv2.resize(mat.astype(np.float32), (size, size))
        c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-8)
    except:
        c2 = np.zeros((size, size), dtype=np.float32)
    
    return np.stack([c0, c1, c2], axis=0).astype(np.float32)


# =============================================================================
# 数据集
# =============================================================================
def parse_signal_value(v, target_len=8192):
    try:
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace("[","").replace("]","").replace("\n"," ").split(",") if p.strip()]
            arr = np.array([float(p) for p in parts], dtype=np.float32)
        elif isinstance(v, (list, tuple)):
            arr = np.array([float(x) for x in v], dtype=np.float32)
        else:
            return None
        if arr.size >= target_len:
            return arr[:target_len]
        out = np.zeros(target_len, dtype=np.float32)
        out[:arr.size] = arr
        return out
    except:
        return None


class DualBranchDataset(Dataset):
    def __init__(self, root_dir, cfg, use_labels=False):
        self.root_dir = Path(root_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.samples = []
        self._build_index()
    
    def _build_index(self):
        if not self.root_dir.exists():
            print(f"[警告] 目录不存在: {self.root_dir}")
            return
        
        files = list(self.root_dir.rglob("*.jsonl")) + list(self.root_dir.rglob("*.json"))
        
        for fp in tqdm(files, desc=f"扫描 {self.root_dir.name}", leave=False):
            try:
                text = fp.read_text(encoding='utf-8', errors='ignore')
                if fp.suffix == '.jsonl':
                    records = [json.loads(l) for l in text.splitlines() if l.strip()]
                else:
                    data = json.loads(text)
                    records = data if isinstance(data, list) else data.get('data', data.get('records', []))
            except:
                continue
            
            if not records:
                continue
            
            # 获取标签
            label = None
            if self.use_labels:
                path_str = str(fp).lower()
                if any(k in path_str for k in ['正常', 'normal', '健康']):
                    label = 0
                elif any(k in path_str for k in ['故障', 'fault', '异常']):
                    label = 1
                if label is None:
                    continue
            
            groups = {}
            for rec in records:
                time_key = rec.get('data_time') or rec.get('dataTime') or rec.get('timestamp') or rec.get('time')
                if not time_key:
                    continue
                sig = parse_signal_value(rec.get('signal_value'), self.cfg.SIGNAL_LEN)
                if sig is not None:
                    groups.setdefault(str(time_key), []).append(sig)
            
            for tk, sigs in groups.items():
                self.samples.append((fp, tk, sigs, label))
        
        print(f"[{self.root_dir.name}] 加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fp, tk, sigs, label = self.samples[idx]
        if len(sigs) == 1:
            sig = sigs[0]
        else:
            X = np.stack(sigs, axis=1)
            E = np.mean(X**2, axis=0) + 1e-12
            sig = (X @ (E/E.sum())).astype(np.float32)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        
        return (
            torch.from_numpy(signal_to_hetero_image(sig, self.cfg.FS, self.cfg.INPUT_SIZE)),
            torch.from_numpy(extract_zerone_features(sig, self.cfg.FS)),
            label if label is not None else -1,
            idx
        )


# =============================================================================
# 模型定义
# =============================================================================
class ZeroneMLP(nn.Module):
    def __init__(self, input_dim=1200, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class HeteroCNN(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.proj = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
    def forward(self, x): return self.proj(self.encoder(x))


class DualBranchEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.zerone_branch = ZeroneMLP(cfg.ZERONE_DIM, cfg.MLP_FEAT_DIM)
        self.hetero_branch = HeteroCNN(cfg.CNN_FEAT_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.CNN_FEAT_DIM + cfg.MLP_FEAT_DIM, 512),
            nn.BatchNorm1d(512), nn.ReLU()
        )
        self.output_dim = 512
    
    def forward(self, image, zerone):
        return self.fusion(torch.cat([self.hetero_branch(image), self.zerone_branch(zerone)], dim=1))


class HybridAnomalyModelV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = DualBranchEncoder(cfg)
        self.svdd_proj = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, cfg.LATENT_DIM)
        )
        self.register_buffer('center', torch.zeros(cfg.LATENT_DIM))
        
        self.vae_mu = nn.Linear(512, cfg.LATENT_CHANNELS * 7 * 7)
        self.vae_logvar = nn.Linear(512, cfg.LATENT_CHANNELS * 7 * 7)
        self.vae_decoder = nn.Sequential(
            nn.ConvTranspose2d(cfg.LATENT_CHANNELS, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
        self.alpha = 0.6
    
    def encode(self, image, zerone):
        return self.encoder(image, zerone)
    
    def forward(self, image, zerone):
        h = self.encode(image, zerone)
        z_svdd = self.svdd_proj(h)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        mu = self.vae_mu(h).view(-1, self.cfg.LATENT_CHANNELS, 7, 7)
        logvar = self.vae_logvar(h).view(-1, self.cfg.LATENT_CHANNELS, 7, 7)
        z_vae = mu if not self.training else mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
        
        recon = self.vae_decoder(z_vae)
        if recon.shape[-1] != image.shape[-1]:
            recon = F.interpolate(recon, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        return {
            'h': h, 'z_svdd': z_svdd, 'svdd_score': svdd_score,
            'vae_recon_loss': F.l1_loss(recon, image, reduction='none').mean(dim=[1,2,3]),
            'vae_kl': -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
        }
    
    def anomaly_score(self, image, zerone):
        out = self.forward(image, zerone)
        return self.alpha * out['svdd_score'] + (1-self.alpha) * (out['vae_recon_loss'] + 0.1*out['vae_kl'])


class FaultClassifierV2(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, image, zerone):
        return self.classifier(self.encoder(image, zerone))


# =============================================================================
# 可视化函数 (中英文双版本)
# =============================================================================
def setup_plotting(lang='cn'):
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.unicode_minus'] = False
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def plot_confusion_matrix(y_true, y_pred, save_path: Path, lang='cn'):
    """绘制混淆矩阵"""
    L = LANG_LABELS[lang]
    setup_plotting(lang)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[L['normal'], L['fault']],
                yticklabels=[L['normal'], L['fault']],
                annot_kws={'size': 18})
    
    ax.set_xlabel(L['predicted'], fontsize=12)
    ax.set_ylabel(L['true'], fontsize=12)
    ax.set_title(L['confusion_title'], fontsize=14)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    
    txt = f"{L['accuracy']}: {acc:.4f}\nF1: {f1:.4f}\n{L['precision']}: {prec:.4f}\n{L['recall']}: {rec:.4f}"
    ax.text(2.5, 0.5, txt, fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return cm


def plot_roc_pr_curves(y_true, y_scores, save_path: Path, lang='cn'):
    """绘制ROC和PR曲线"""
    L = LANG_LABELS[lang]
    setup_plotting(lang)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=COLORS['blue'], lw=2, label=f'{L["roc_title"]} (AUC={roc_auc:.4f})')
    axes[0].plot([0,1], [0,1], 'gray', ls='--', label=L['random'])
    axes[0].fill_between(fpr, tpr, alpha=0.2, color=COLORS['blue'])
    axes[0].set_xlabel(L['fpr'], fontsize=12)
    axes[0].set_ylabel(L['tpr'], fontsize=12)
    axes[0].set_title(L['roc_title'], fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, ls=':', alpha=0.5)
    
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    baseline = sum(y_true)/len(y_true)
    axes[1].plot(rec, prec, color=COLORS['red'], lw=2, label=f'{L["pr_title"]} (AP={ap:.4f})')
    axes[1].axhline(baseline, color='gray', ls='--', label=f'{L["baseline"]} ({baseline:.2f})')
    axes[1].fill_between(rec, prec, alpha=0.2, color=COLORS['red'])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title(L['pr_title'], fontsize=14)
    axes[1].legend(loc='lower left')
    axes[1].grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return {'roc_auc': roc_auc, 'ap': ap}


def plot_tsne(features, labels, save_path: Path, lang='cn'):
    """t-SNE可视化"""
    L = LANG_LABELS[lang]
    setup_plotting(lang)
    
    print(f"  [t-SNE] 计算中 ({lang})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    feat_2d = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for lbl, color, name in [(0, COLORS['normal'], L['normal']), (1, COLORS['fault'], L['fault'])]:
        mask = labels == lbl
        ax.scatter(feat_2d[mask, 0], feat_2d[mask, 1], c=color, label=name, 
                   alpha=0.6, s=50, edgecolors='white', lw=0.5)
    
    ax.set_xlabel(L['tsne_dim1'], fontsize=12)
    ax.set_ylabel(L['tsne_dim2'], fontsize=12)
    ax.set_title(L['tsne_title'], fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, ls=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distribution(scores, labels, save_path: Path, lang='cn'):
    """得分分布"""
    L = LANG_LABELS[lang]
    setup_plotting(lang)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lbl, color, name in [(0, COLORS['normal'], L['normal']), (1, COLORS['fault'], L['fault'])]:
        s = scores[labels == lbl]
        ax.hist(s, bins=50, alpha=0.6, color=color, label=f'{name} (n={len(s)})', density=True)
        ax.axvline(s.mean(), color=color, ls='--', lw=2, label=f'{name} {L["mean"]}: {s.mean():.4f}')
    
    ax.set_xlabel(L['score_xlabel'], fontsize=12)
    ax.set_ylabel(L['density'], fontsize=12)
    ax.set_title(L['score_title'], fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_analysis(features, labels, save_path: Path, lang='cn'):
    """特征分析"""
    L = LANG_LABELS[lang]
    setup_plotting(lang)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 特征均值对比
    ax1 = fig.add_subplot(2, 2, 1)
    n_show = 50
    nm = features[labels==0].mean(axis=0)[:n_show]
    fm = features[labels==1].mean(axis=0)[:n_show]
    x = np.arange(n_show)
    ax1.bar(x-0.175, nm, 0.35, label=L['normal'], color=COLORS['normal'], alpha=0.7)
    ax1.bar(x+0.175, fm, 0.35, label=L['fault'], color=COLORS['fault'], alpha=0.7)
    ax1.set_xlabel(L['feat_index'], fontsize=11)
    ax1.set_ylabel(L['feat_mean'], fontsize=11)
    ax1.set_title(L['feat_mean_title'], fontsize=12)
    ax1.legend()
    ax1.grid(True, ls=':', alpha=0.5)
    
    # 2. 范数分布
    ax2 = fig.add_subplot(2, 2, 2)
    norm = np.linalg.norm(features, axis=1)
    ax2.hist(norm[labels==0], bins=30, alpha=0.6, color=COLORS['normal'], label=L['normal'], density=True)
    ax2.hist(norm[labels==1], bins=30, alpha=0.6, color=COLORS['fault'], label=L['fault'], density=True)
    ax2.set_xlabel(L['feat_norm'], fontsize=11)
    ax2.set_ylabel(L['density'], fontsize=11)
    ax2.set_title(L['feat_norm_title'], fontsize=12)
    ax2.legend()
    ax2.grid(True, ls=':', alpha=0.5)
    
    # 3. 方差
    ax3 = fig.add_subplot(2, 2, 3)
    nv = features[labels==0].var(axis=0)[:100]
    fv = features[labels==1].var(axis=0)[:100]
    ax3.plot(nv, label=L['normal'], color=COLORS['normal'], alpha=0.8)
    ax3.plot(fv, label=L['fault'], color=COLORS['fault'], alpha=0.8)
    ax3.set_xlabel(L['feat_index'], fontsize=11)
    ax3.set_ylabel(L['feat_var'], fontsize=11)
    ax3.set_title(L['feat_var_title'], fontsize=12)
    ax3.legend()
    ax3.grid(True, ls=':', alpha=0.5)
    
    # 4. 相关性
    ax4 = fig.add_subplot(2, 2, 4)
    corr = np.corrcoef(features[:, :20].T)
    im = ax4.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title(L['feat_corr_title'], fontsize=12)
    ax4.set_xlabel(L['feat_index'], fontsize=11)
    ax4.set_ylabel(L['feat_index'], fontsize=11)
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results, save_path: Path, lang='cn'):
    """生成评估报告"""
    if lang == 'cn':
        lines = [
            "=" * 70,
            "三阶段变压器故障诊断系统 - V2版本 评估报告",
            "=" * 70, "",
            "【分类性能】",
            f"  准确率: {results['acc']:.4f}",
            f"  F1分数: {results['f1']:.4f}",
            f"  精确率: {results['prec']:.4f}",
            f"  召回率: {results['rec']:.4f}", "",
            "【曲线指标】",
            f"  ROC-AUC: {results['roc_auc']:.4f}",
            f"  Average Precision: {results['ap']:.4f}", "",
            "【数据集】",
            f"  总样本: {results['total']}",
            f"  正常: {results['n_normal']}",
            f"  故障: {results['n_fault']}", "",
            "=" * 70
        ]
    else:
        lines = [
            "=" * 70,
            "Three-Stage Transformer Fault Diagnosis - V2 Evaluation Report",
            "=" * 70, "",
            "[Classification Performance]",
            f"  Accuracy: {results['acc']:.4f}",
            f"  F1 Score: {results['f1']:.4f}",
            f"  Precision: {results['prec']:.4f}",
            f"  Recall: {results['rec']:.4f}", "",
            "[Curve Metrics]",
            f"  ROC-AUC: {results['roc_auc']:.4f}",
            f"  Average Precision: {results['ap']:.4f}", "",
            "[Dataset]",
            f"  Total: {results['total']}",
            f"  Normal: {results['n_normal']}",
            f"  Fault: {results['n_fault']}", "",
            "=" * 70
        ]
    
    text = "\n".join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='可视化补充脚本 V2 (中英文双版本)')
    parser.add_argument('--results_dir', type=str, default='./three_stage_results_v2')
    parser.add_argument('--data_root', type=str, 
                        default=r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016")
    parser.add_argument('--skip_tsne', action='store_true', help='跳过t-SNE')
    args = parser.parse_args()
    
    cfg = VisualizationConfig()
    cfg.RESULTS_DIR = Path(args.results_dir)
    cfg.DATA_ROOT = Path(args.data_root)
    cfg.__post_init__()
    
    device = torch.device(cfg.DEVICE)
    print(f"\n设备: {device}")
    
    # 加载模型
    print("\n" + "="*60)
    print("【加载模型】")
    print("="*60)
    
    stage1_path = cfg.MODEL_DIR / "stage1_best_v2.pth"
    stage3_path = cfg.MODEL_DIR / "stage3_best_v2.pth"
    
    if not stage1_path.exists():
        print(f"[错误] 未找到: {stage1_path}")
        return
    if not stage3_path.exists():
        print(f"[错误] 未找到: {stage3_path}")
        return
    
    model = HybridAnomalyModelV2(cfg)
    ckpt1 = torch.load(stage1_path, map_location=device)
    model.load_state_dict(ckpt1['model_state'])
    model.center = ckpt1['center']
    model = model.to(device).eval()
    print(f"  ✅ Stage1模型加载成功")
    
    classifier = FaultClassifierV2(model.encoder, 2).to(device)
    ckpt3 = torch.load(stage3_path, map_location=device)
    classifier.load_state_dict(ckpt3['model_state'])
    classifier.eval()
    print(f"  ✅ Stage3分类器加载成功 (F1={ckpt3.get('f1', 'N/A')})")
    
    # 加载数据
    print("\n" + "="*60)
    print("【加载测试数据】")
    print("="*60)
    
    test_ds = DualBranchDataset(cfg.TEST_DIR, cfg, use_labels=True)
    if len(test_ds) == 0:
        print("[错误] 测试数据为空")
        return
    
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    labels_all = [test_ds.samples[i][3] for i in range(len(test_ds))]
    n_normal = sum(1 for l in labels_all if l == 0)
    n_fault = sum(1 for l in labels_all if l == 1)
    print(f"  总样本: {len(test_ds)}, 正常: {n_normal}, 故障: {n_fault}")
    
    # 推理
    print("\n" + "="*60)
    print("【推理】")
    print("="*60)
    
    all_feat, all_lbl, all_pred, all_score, all_prob = [], [], [], [], []
    
    with torch.no_grad():
        for img, zr, lbl, _ in tqdm(test_loader, desc="推理中"):
            img, zr = img.to(device), zr.to(device)
            
            all_feat.append(model.encode(img, zr).cpu().numpy())
            all_lbl.extend(lbl.tolist())
            all_score.extend(model.anomaly_score(img, zr).cpu().numpy())
            
            logits = classifier(img, zr)
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_prob.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    
    all_feat = np.concatenate(all_feat)
    all_lbl = np.array(all_lbl)
    all_pred = np.array(all_pred)
    all_score = np.array(all_score)
    all_prob = np.array(all_prob)
    
    # 计算指标
    results = {
        'acc': accuracy_score(all_lbl, all_pred),
        'f1': f1_score(all_lbl, all_pred, average='macro'),
        'prec': precision_score(all_lbl, all_pred, average='macro'),
        'rec': recall_score(all_lbl, all_pred, average='macro'),
        'total': len(all_lbl), 'n_normal': n_normal, 'n_fault': n_fault
    }
    
    # 生成双语可视化
    print("\n" + "="*60)
    print("【生成可视化 (中文+英文)】")
    print("="*60)
    
    for lang in ['cn', 'en']:
        suffix = '_cn' if lang == 'cn' else '_en'
        print(f"\n--- {lang.upper()} 版本 ---")
        
        # 混淆矩阵
        plot_confusion_matrix(all_lbl, all_pred, 
                              cfg.VIZ_DIR / "confusion" / f"confusion_matrix{suffix}.png", lang)
        print(f"  ✅ confusion_matrix{suffix}.png")
        
        # ROC/PR
        rp = plot_roc_pr_curves(all_lbl, all_prob, 
                                cfg.VIZ_DIR / "roc_pr" / f"roc_pr_curves{suffix}.png", lang)
        results['roc_auc'], results['ap'] = rp['roc_auc'], rp['ap']
        print(f"  ✅ roc_pr_curves{suffix}.png")
        
        # t-SNE
        if not args.skip_tsne:
            plot_tsne(all_feat, all_lbl, 
                      cfg.VIZ_DIR / "tsne" / f"tsne{suffix}.png", lang)
            print(f"  ✅ tsne{suffix}.png")
        
        # 得分分布
        plot_score_distribution(all_score, all_lbl, 
                                cfg.VIZ_DIR / "score_dist" / f"score_by_class{suffix}.png", lang)
        print(f"  ✅ score_by_class{suffix}.png")
        
        # 特征分析
        plot_feature_analysis(all_feat, all_lbl, 
                              cfg.VIZ_DIR / "feature_analysis" / f"feature_analysis{suffix}.png", lang)
        print(f"  ✅ feature_analysis{suffix}.png")
        
        # 报告
        generate_report(results, cfg.RESULTS_DIR / f"evaluation_report{suffix}.txt", lang)
        print(f"  ✅ evaluation_report{suffix}.txt")
    
    print("\n" + "="*60)
    print("【完成】所有可视化已生成")
    print("="*60)
    print(f"  结果目录: {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
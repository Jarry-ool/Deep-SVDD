# -*- coding: utf-8 -*-
"""
transformer_three_stage_v2.py
==============================

交流变压器振动数据 三阶段渐进式故障诊断系统 
整合 Zerone 1200维工程特征 + Hetero 三通道时频图像

【V2版本改进】
    ✅ 保留 Hetero 的三通道图像 (CWT+STFT+Context) → CNN分支
    ✅ 新增 Zerone 的1200维特征 → MLP分支  
    ✅ 双分支特征融合，发挥各自优势
    ✅ 完整复用 zerone_features.py 的特征提取函数

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
                    │                           │
                    └───────────┬───────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │      特征融合层           │
                    │   512 + 256 → 512维      │
                    └──────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Deep SVDD   │    │  VAE 重构    │    │  分类头      │
    │  (异常检测)  │    │  (可解释性)  │    │  (阶段三)    │
    └──────────────┘    └──────────────┘    └──────────────┘

【运行方式】
    # 测试数据加载
    python transformer_three_stage_v2.py --test_data
    
    # 运行全部阶段
    python transformer_three_stage_v2.py --all
    
    # 分阶段运行
    python transformer_three_stage_v2.py --stage 1
    python transformer_three_stage_v2.py --stage 2
    python transformer_three_stage_v2.py --stage 3

Author: 基于 zerone + hetero 代码框架整合
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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter

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

# 可视化
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 评估指标
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    average_precision_score
)
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


# =============================================================================
# 第1步: 尝试导入项目现有模块（Zerone特征提取）
# =============================================================================
# 添加项目路径
PROJECT_PATH = Path("/mnt/project")
if PROJECT_PATH.exists():
    sys.path.insert(0, str(PROJECT_PATH))

# 尝试导入 zerone_features
try:
    from zerone_features import (
        compute_time_features,      # 15维时域特征
        stft_segment_means,         # 127维STFT特征
        compute_psd,                # PSD计算
        compute_high_frequency_ratios,  # 高频指标
        FEAT_SCHEMA,                # 特征维度定义
        TOTAL_FEAT_DIM,             # 总特征维度 (1200)
    )
    ZERONE_AVAILABLE = True
    print("[✓] 成功导入 zerone_features 模块")
except ImportError as e:
    ZERONE_AVAILABLE = False
    print(f"[!] 未能导入 zerone_features: {e}")
    print("    将使用内置的简化版特征提取")
    
    # 定义简化版特征提取（后备方案）
    FEAT_SCHEMA = [("time", 15), ("stft", 127), ("psd", 1050), ("hf", 8)]
    TOTAL_FEAT_DIM = sum(d for _, d in FEAT_SCHEMA)


# =============================================================================
# 第2步: 配置类定义
# =============================================================================
@dataclass
class ThreeStageConfigV2:
    """
    三阶段诊断系统配置类 (V2版本)
    
    【新增配置】
        USE_ZERONE_FEATURES: 是否使用Zerone的1200维特征
        USE_HETERO_IMAGE: 是否使用Hetero的三通道图像
        FUSION_MODE: 特征融合方式 (concat/attention/gate)
    """
    
    # ================= 路径配置 =================
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v2"))
    
    # ================= 信号参数 =================
    FS: float = 8192.0          # 采样频率 (Hz)
    SIGNAL_LEN: int = 8192      # 信号长度
    INPUT_SIZE: int = 224       # CNN输入尺寸
    
    # ================= 特征配置（新增）=================
    USE_ZERONE_FEATURES: bool = True   # 是否使用Zerone 1200维特征
    USE_HETERO_IMAGE: bool = True      # 是否使用Hetero三通道图像
    ZERONE_DIM: int = TOTAL_FEAT_DIM   # Zerone特征维度 (1200)
    FUSION_MODE: str = "concat"        # 融合方式: concat/attention/gate
    
    # ================= 模型参数 =================
    LATENT_DIM: int = 128       # SVDD隐空间维度
    LATENT_CHANNELS: int = 64   # VAE空间隐变量通道数
    CNN_FEAT_DIM: int = 512     # CNN输出维度
    MLP_FEAT_DIM: int = 256     # MLP输出维度
    
    # ================= 训练参数 =================
    BATCH_SIZE: int = 16
    STAGE1_EPOCHS: int = 50
    STAGE3_EPOCHS: int = 30
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    
    # SVDD参数
    NU: float = 0.05            # 假设异常比例
    
    # VAE参数
    BETA_VAE: float = 0.01
    BETA_WARMUP: int = 10
    
    # ================= 伪标签阈值 =================
    NORMAL_PERCENTILE: float = 5.0
    ANOMALY_PERCENTILE: float = 99.0
    
    # ================= 类别关键词 =================
    CLASS_KEYWORDS: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    })
    
    # ================= 设备 =================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ================= 可视化 =================
    VIZ_DPI: int = 300
    VIZ_LANG: str = "cn"
    
    def __post_init__(self):
        """初始化后处理"""
        self.PROJECT_ROOT = Path(self.PROJECT_ROOT)
        self.OUTPUT_ROOT = Path(self.OUTPUT_ROOT)
        
        # 数据目录
        self.TRAIN_DIR = self.PROJECT_ROOT / "train"
        self.VAL_DIR = self.PROJECT_ROOT / "val"
        self.TEST_DIR = self.PROJECT_ROOT / "test"
        
        # 输出子目录
        self.STAGE1_DIR = self.OUTPUT_ROOT / "stage1_unsupervised"
        self.STAGE2_DIR = self.OUTPUT_ROOT / "stage2_pseudo_labels"
        self.STAGE3_DIR = self.OUTPUT_ROOT / "stage3_supervised"
        self.VIZ_DIR = self.OUTPUT_ROOT / "visualizations"
        self.MODEL_DIR = self.OUTPUT_ROOT / "models"
        
        # 创建目录
        for d in [self.STAGE1_DIR, self.STAGE2_DIR, self.STAGE3_DIR, 
                  self.VIZ_DIR, self.MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 可视化子目录
        for subdir in ["training_curves", "score_dist", "confusion", 
                       "roc_pr", "tsne", "feature_analysis"]:
            (self.VIZ_DIR / subdir).mkdir(exist_ok=True)
        
        # 检查特征配置
        if not ZERONE_AVAILABLE and self.USE_ZERONE_FEATURES:
            print("[警告] zerone_features 不可用，将使用内置简化版")
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("三阶段故障诊断系统配置 (V2版本)")
        print("="*70)
        print(f"【数据路径】")
        print(f"  项目根目录: {self.PROJECT_ROOT}")
        print(f"【特征配置】")
        print(f"  Zerone特征 (1200维): {'✅ 启用' if self.USE_ZERONE_FEATURES else '❌ 禁用'}")
        print(f"  Hetero图像 (3通道): {'✅ 启用' if self.USE_HETERO_IMAGE else '❌ 禁用'}")
        print(f"  融合方式: {self.FUSION_MODE}")
        print(f"【模型参数】")
        print(f"  CNN特征: {self.CNN_FEAT_DIM}维")
        print(f"  MLP特征: {self.MLP_FEAT_DIM}维")
        print(f"  融合后: {self.CNN_FEAT_DIM + self.MLP_FEAT_DIM}维")
        print(f"【训练参数】")
        print(f"  设备: {self.DEVICE}")
        print(f"  批大小: {self.BATCH_SIZE}")
        print("="*70 + "\n")


# =============================================================================
# 第3步: 可视化设置
# =============================================================================
def setup_plotting():
    """设置matplotlib绘图参数"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10

COLORS = {
    'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
    'red': '#d62728', 'purple': '#9467bd',
    'normal': '#2ca02c', 'fault': '#d62728', 'uncertain': '#7f7f7f'
}

LABELS = {
    'cn': {
        'normal': '正常', 'fault': '故障', 'uncertain': '不确定',
        'epoch': '训练轮次', 'loss': '损失值', 'accuracy': '准确率',
        'score': '异常得分', 'count': '样本数',
    },
    'en': {
        'normal': 'Normal', 'fault': 'Fault', 'uncertain': 'Uncertain',
        'epoch': 'Epoch', 'loss': 'Loss', 'accuracy': 'Accuracy',
        'score': 'Anomaly Score', 'count': 'Count',
    }
}


# =============================================================================
# 第4步: 特征提取函数
# =============================================================================

def extract_zerone_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    提取Zerone 1200维工程特征
    
    【特征组成】
        时域 (15维): 均值、RMS、峭度、波形因子等
        STFT (127维): 短时傅里叶变换段均值
        PSD (1050维): 1-1000Hz逐Hz + 1001-2000Hz每20Hz
        高频 (8维): 4个阈值 × (幅值比+功率比)
        
    【参数】
        sig: 一维振动信号 (N,)
        fs: 采样频率 (Hz)
        
    【返回】
        np.ndarray: 1200维特征向量
    """
    sig = np.asarray(sig, dtype=np.float32).ravel()
    
    # 去直流
    sig_dc = sig - np.mean(sig)
    
    features = []
    
    # ----- 1. 时域特征 (15维) -----
    if ZERONE_AVAILABLE:
        time_feat = compute_time_features(sig)
    else:
        # 简化版时域特征
        time_feat = _compute_time_features_simple(sig)
    features.append(time_feat)
    
    # ----- 2. STFT特征 (127维) -----
    if ZERONE_AVAILABLE:
        stft_feat = stft_segment_means(sig_dc, fs, nperseg=128, noverlap=64)
    else:
        stft_feat = _compute_stft_simple(sig_dc, fs)
    features.append(stft_feat)
    
    # ----- 3. PSD特征 (1050维) -----
    #   1-1000Hz: 每1Hz一维 → 1000维
    #   1001-2000Hz: 每20Hz一维 → 50维
    if ZERONE_AVAILABLE:
        psd_full = compute_psd(sig_dc, fs, fmin=1, fmax=2000, df=1, nperseg=len(sig)//2)
        # 分段处理
        psd_low = psd_full[:1000] if len(psd_full) >= 1000 else np.pad(psd_full, (0, 1000-len(psd_full)))
        psd_high_raw = psd_full[1000:2000] if len(psd_full) >= 2000 else np.zeros(1000)
        # 1001-2000Hz 每20Hz聚合
        psd_high = np.array([psd_high_raw[i:i+20].mean() for i in range(0, 1000, 20)])
    else:
        psd_low = np.zeros(1000)
        psd_high = np.zeros(50)
    
    psd_feat = np.concatenate([psd_low, psd_high])  # 1050维
    features.append(psd_feat)
    
    # ----- 4. 高频特征 (8维) -----
    hf_feat = []
    thresholds = [1000, 2000, 3000, 4000]
    for thr in thresholds:
        if ZERONE_AVAILABLE:
            amp_ratio, pwr_ratio = compute_high_frequency_ratios(
                psd_full if 'psd_full' in dir() else np.zeros(2000), 
                fs, threshold_hz=thr, fmin=1, fmax=4000
            )
        else:
            amp_ratio, pwr_ratio = 0.0, 0.0
        hf_feat.extend([amp_ratio, pwr_ratio])
    features.append(np.array(hf_feat))
    
    # 拼接所有特征
    feat_vec = np.concatenate(features).astype(np.float32)
    
    # 确保维度正确 (1200维)
    if len(feat_vec) < TOTAL_FEAT_DIM:
        feat_vec = np.pad(feat_vec, (0, TOTAL_FEAT_DIM - len(feat_vec)))
    elif len(feat_vec) > TOTAL_FEAT_DIM:
        feat_vec = feat_vec[:TOTAL_FEAT_DIM]
    
    return feat_vec


def _compute_time_features_simple(sig: np.ndarray) -> np.ndarray:
    """简化版时域特征提取 (15维)"""
    x = np.asarray(sig, dtype=float).ravel()
    if len(x) == 0:
        return np.zeros(15)
    
    mean_val = np.mean(x)
    rms_val = np.sqrt(np.mean(x**2))
    var_val = np.var(x)
    std_val = np.std(x)
    max_val = np.max(x)
    min_val = np.min(x)
    p2p_val = max_val - min_val
    
    xc = x - mean_val
    m2 = np.mean(xc**2)
    m4 = np.mean(xc**4)
    kurtosis = m4 / (m2**2 + 1e-12)
    m3 = np.mean(xc**3)
    skewness = m3 / (std_val**3 + 1e-12)
    
    zero_cross = np.sum(np.abs(np.diff(np.sign(x))) > 0) / (len(x) - 1 + 1e-12)
    mean_abs = np.mean(np.abs(x))
    
    crest = max_val / (rms_val + 1e-12)
    impulse = max_val / (mean_abs + 1e-12)
    margin = max_val / (np.mean(np.abs(x)**4)**0.25 + 1e-12)
    waveform = rms_val / (mean_abs + 1e-12)
    
    return np.array([
        mean_val, rms_val, var_val, std_val, max_val, min_val, p2p_val,
        kurtosis, skewness, zero_cross, mean_abs,
        crest, impulse, margin, waveform
    ], dtype=np.float32)


def _compute_stft_simple(sig: np.ndarray, fs: float) -> np.ndarray:
    """简化版STFT特征提取 (127维)"""
    try:
        from scipy.signal import stft as scipy_stft
        _, _, Zxx = scipy_stft(sig, fs=fs, nperseg=128, noverlap=64)
        mag = np.abs(Zxx[1:, :])  # 去DC
        seg_means = np.mean(mag, axis=0)
        out = np.zeros(127, dtype=np.float32)
        L = min(len(seg_means), 127)
        out[:L] = seg_means[:L]
        return out
    except Exception:
        return np.zeros(127, dtype=np.float32)


def signal_to_hetero_image(sig: np.ndarray, fs: float, size: int = 224) -> np.ndarray:
    """
    将振动信号转换为Hetero三通道时频图像
    
    【通道设计】（来自hetero_model.py）
        Ch0: CWT (Morlet小波) - 捕捉时频局部特征
        Ch1: STFT幅度谱 - 捕捉短时频域特征
        Ch2: Context (波形折叠) - 保留原始时域细节
        
    【参数】
        sig: 一维振动信号 (N,)
        fs: 采样频率 (Hz)
        size: 输出图像边长 (默认224)
        
    【返回】
        np.ndarray: 形状为 (3, size, size) 的图像张量
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
# 第5步: 数据读取工具
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
# 第6步: 数据集类（双分支输出）
# =============================================================================

class DualBranchDataset(Dataset):
    """
    双分支数据集：同时输出 Zerone特征 + Hetero图像
    
    【输出】
        image: (3, 224, 224) Hetero三通道图像
        zerone: (1200,) Zerone工程特征
        label: 标签 (-1表示无标签)
        idx: 样本索引
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        cfg: ThreeStageConfigV2,
        use_labels: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.samples: List[Tuple[Path, str, List[np.ndarray], Optional[int]]] = []
        self._build_index()
    
    def _build_index(self):
        """构建样本索引"""
        if not self.root_dir.exists():
            print(f"[警告] 目录不存在: {self.root_dir}")
            return
        
        files = list(self.root_dir.rglob("*.jsonl")) + list(self.root_dir.rglob("*.json"))
        label_counts = Counter()
        
        for fp in tqdm(files, desc=f"扫描 {self.root_dir.name}", leave=False):
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
        
        print(f"[{self.root_dir.name}] 加载 {len(self.samples)} 个样本")
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
            image: (3, H, W) Hetero图像
            zerone: (1200,) Zerone特征
            label: 标签
            idx: 索引
        """
        fp, time_key, sig_list, label = self.samples[idx]
        sig = self._aggregate_channels(sig_list)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        
        # Hetero图像
        if self.cfg.USE_HETERO_IMAGE:
            image = signal_to_hetero_image(sig, self.cfg.FS, self.cfg.INPUT_SIZE)
        else:
            image = np.zeros((3, self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE), dtype=np.float32)
        
        # Zerone特征
        if self.cfg.USE_ZERONE_FEATURES:
            zerone = extract_zerone_features(sig, self.cfg.FS)
        else:
            zerone = np.zeros(self.cfg.ZERONE_DIM, dtype=np.float32)
        
        return (
            torch.from_numpy(image),
            torch.from_numpy(zerone),
            label if label is not None else -1,
            idx
        )


# =============================================================================
# 第7步: 模型定义（双分支架构）
# =============================================================================

class ZeroneMLP(nn.Module):
    """
    Zerone特征处理分支 (MLP)
    
    【架构】
        1200维输入 → 512 → 256 → 256维输出
        
    【设计考量】
        - 使用BatchNorm + Dropout防止过拟合
        - 保留中间维度以捕捉特征交互
    """
    
    def __init__(self, input_dim: int = 1200, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层：降维
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # 第二层：特征学习
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 第三层：输出
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroCNN(nn.Module):
    """
    Hetero图像处理分支 (CNN)
    
    【架构】
        基于ResNet18，输出512维特征
        
    【来源】
        直接复用 hetero_model.py 的编码器设计
    """
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )  # 输出 (B, 512)
        
        if output_dim != 512:
            self.proj = nn.Linear(512, output_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.proj(h)


class DualBranchEncoder(nn.Module):
    """
    双分支编码器：融合 Zerone + Hetero
    
    【融合策略】
        concat: 简单拼接 (默认)
        attention: 注意力加权融合
        gate: 门控融合
    """
    
    def __init__(self, cfg: ThreeStageConfigV2):
        super().__init__()
        self.cfg = cfg
        
        # Zerone分支
        self.zerone_branch = ZeroneMLP(
            input_dim=cfg.ZERONE_DIM,
            output_dim=cfg.MLP_FEAT_DIM
        )
        
        # Hetero分支
        self.hetero_branch = HeteroCNN(output_dim=cfg.CNN_FEAT_DIM)
        
        # 融合层
        fused_dim = cfg.CNN_FEAT_DIM + cfg.MLP_FEAT_DIM  # 512 + 256 = 768
        
        if cfg.FUSION_MODE == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
        elif cfg.FUSION_MODE == "attention":
            self.fusion = AttentionFusion(cfg.CNN_FEAT_DIM, cfg.MLP_FEAT_DIM, 512)
        else:  # gate
            self.fusion = GateFusion(cfg.CNN_FEAT_DIM, cfg.MLP_FEAT_DIM, 512)
        
        self.output_dim = 512
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """
        参数:
            image: (B, 3, H, W)
            zerone: (B, 1200)
        返回:
            (B, 512) 融合特征
        """
        h_img = self.hetero_branch(image)    # (B, 512)
        h_zr = self.zerone_branch(zerone)    # (B, 256)
        
        if self.cfg.FUSION_MODE == "concat":
            h_cat = torch.cat([h_img, h_zr], dim=1)  # (B, 768)
            return self.fusion(h_cat)
        else:
            return self.fusion(h_img, h_zr)


class AttentionFusion(nn.Module):
    """注意力融合模块"""
    def __init__(self, dim1: int, dim2: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim1, 1)
        self.w2 = nn.Linear(dim2, 1)
        self.proj = nn.Linear(dim1 + dim2, out_dim)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        a1 = torch.sigmoid(self.w1(x1))
        a2 = torch.sigmoid(self.w2(x2))
        h = torch.cat([a1 * x1, a2 * x2], dim=1)
        return self.proj(h)


class GateFusion(nn.Module):
    """门控融合模块"""
    def __init__(self, dim1: int, dim2: int, out_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim1 + dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.proj1 = nn.Linear(dim1, out_dim)
        self.proj2 = nn.Linear(dim2, out_dim)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([x1, x2], dim=1))
        h1 = self.proj1(x1) * g[:, 0:1]
        h2 = self.proj2(x2) * g[:, 1:2]
        return h1 + h2


class HybridAnomalyModelV2(nn.Module):
    """
    混合异常检测模型 V2 (双分支)
    
    【组件】
        - 双分支编码器 (Zerone MLP + Hetero CNN)
        - Deep SVDD头 (异常检测)
        - VAE解码器 (可解释性)
    """
    
    def __init__(self, cfg: ThreeStageConfigV2):
        super().__init__()
        self.cfg = cfg
        
        # 双分支编码器
        self.encoder = DualBranchEncoder(cfg)
        
        # SVDD投影头
        self.svdd_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, cfg.LATENT_DIM)
        )
        self.register_buffer('center', torch.zeros(cfg.LATENT_DIM))
        
        # VAE解码器 (简化版，用于可解释性)
        self.vae_mu = nn.Linear(512, cfg.LATENT_CHANNELS * 7 * 7)
        self.vae_logvar = nn.Linear(512, cfg.LATENT_CHANNELS * 7 * 7)
        self.vae_decoder = nn.Sequential(
            nn.ConvTranspose2d(cfg.LATENT_CHANNELS, 256, 4, 2, 1),  # 7→14
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14→28
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28→56
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56→112
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 112→224
            nn.Sigmoid()
        )
        
        self.alpha = 0.6  # SVDD权重
    
    def encode(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(image, zerone)
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        h = self.encode(image, zerone)  # (B, 512)
        
        # SVDD
        z_svdd = self.svdd_proj(h)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        # VAE
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
        
        return {
            'h': h,
            'z_svdd': z_svdd,
            'svdd_score': svdd_score,
            'recon': recon,
            'vae_recon_loss': vae_recon_loss,
            'vae_kl': vae_kl,
            'mu': mu,
            'logvar': logvar
        }
    
    def init_center(self, dataloader: DataLoader, device: torch.device):
        """初始化SVDD中心"""
        n = 0
        c = torch.zeros(self.cfg.LATENT_DIM, device=device)
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                img, zr, _, _ = batch
                img, zr = img.to(device), zr.to(device)
                h = self.encode(img, zr)
                z = self.svdd_proj(h)
                c += z.sum(0)
                n += z.size(0)
        c /= n
        
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c
    
    def anomaly_score(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        """计算融合异常得分"""
        out = self.forward(image, zerone)
        svdd_score = out['svdd_score']
        vae_score = out['vae_recon_loss'] + 0.1 * out['vae_kl']
        return self.alpha * svdd_score + (1 - self.alpha) * vae_score


class FaultClassifierV2(nn.Module):
    """故障分类器 V2 (基于双分支编码器)"""
    
    def __init__(self, pretrained_encoder: DualBranchEncoder, num_classes: int = 2):
        super().__init__()
        self.encoder = pretrained_encoder
        
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        h = self.encoder(image, zerone)
        return self.classifier(h)
    
    def unfreeze_encoder(self, num_blocks: int = 1):
        """渐进解冻"""
        params = list(self.encoder.parameters())
        n = min(num_blocks * 20, len(params))
        for p in params[-n:]:
            p.requires_grad = True
        print(f"  [解冻] 编码器最后 {n} 个参数")


# =============================================================================
# 第8步: 训练函数
# =============================================================================

def train_stage1_v2(cfg: ThreeStageConfigV2) -> Tuple[HybridAnomalyModelV2, Dict]:
    """阶段一：无监督训练"""
    print("\n" + "="*70)
    print("【阶段一】无监督异常检测训练 (V2双分支)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    train_ds = DualBranchDataset(cfg.TRAIN_DIR, cfg, use_labels=False)
    val_ds = DualBranchDataset(cfg.VAL_DIR, cfg, use_labels=False)
    combined = ConcatDataset([train_ds, val_ds])
    
    if len(combined) == 0:
        raise ValueError("没有加载到数据！")
    
    print(f"  总样本数: {len(combined)}")
    
    loader = DataLoader(combined, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                       num_workers=0, drop_last=True)
    
    # 初始化模型
    print("\n[2/5] 初始化模型...")
    model = HybridAnomalyModelV2(cfg).to(device)
    
    # 预训练VAE
    print("\n[3/5] 预训练VAE (10轮)...")
    vae_params = list(model.vae_mu.parameters()) + list(model.vae_logvar.parameters()) + \
                 list(model.vae_decoder.parameters()) + list(model.encoder.parameters())
    vae_opt = torch.optim.Adam(vae_params, lr=cfg.LR * 10)
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"VAE预训练 {epoch+1}/10", leave=False):
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            out = model(img, zr)
            
            loss = out['vae_recon_loss'].mean() + 0.01 * out['vae_kl'].mean()
            
            vae_opt.zero_grad()
            loss.backward()
            vae_opt.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}/10 | VAE Loss: {total_loss/len(loader):.4f}")
    
    # 初始化SVDD中心
    print("\n[4/5] 初始化SVDD中心...")
    model.init_center(loader, device)
    print(f"  中心范数: {model.center.norm().item():.4f}")
    
    # 联合训练
    print(f"\n[5/5] 联合训练 ({cfg.STAGE1_EPOCHS}轮)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.STAGE1_EPOCHS)
    
    history = {'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': []}
    best_loss = float('inf')
    
    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch + 1) / max(cfg.BETA_WARMUP, 1))
        
        epoch_svdd, epoch_vae = 0, 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}")
        for batch in pbar:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            out = model(img, zr)
            
            svdd_loss = out['svdd_score'].mean()
            vae_loss = out['vae_recon_loss'].mean() + beta * out['vae_kl'].mean()
            total_loss = svdd_loss + vae_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_svdd += svdd_loss.item()
            epoch_vae += vae_loss.item()
            
            pbar.set_postfix({'SVDD': f'{svdd_loss.item():.3f}', 'VAE': f'{vae_loss.item():.3f}'})
        
        scheduler.step()
        
        avg_svdd = epoch_svdd / len(loader)
        avg_vae = epoch_vae / len(loader)
        avg_total = avg_svdd + avg_vae
        
        history['epoch'].append(epoch + 1)
        history['svdd_loss'].append(avg_svdd)
        history['vae_loss'].append(avg_vae)
        history['total_loss'].append(avg_total)
        
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.center,
                'epoch': epoch,
                'loss': best_loss
            }, cfg.MODEL_DIR / "stage1_best_v2.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f}")
    
    # 保存训练曲线
    plot_training_curves_v2(history, cfg, "stage1")
    
    print(f"\n【阶段一完成】最佳Loss: {best_loss:.4f}")
    return model, history


def run_stage2_v2(model: HybridAnomalyModelV2, cfg: ThreeStageConfigV2) -> Dict:
    """阶段二：伪标签生成"""
    print("\n" + "="*70)
    print("【阶段二】伪标签生成")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    model.eval()
    
    # 加载数据
    train_ds = DualBranchDataset(cfg.TRAIN_DIR, cfg, use_labels=False)
    val_ds = DualBranchDataset(cfg.VAL_DIR, cfg, use_labels=False)
    combined = ConcatDataset([train_ds, val_ds])
    
    loader = DataLoader(combined, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 计算异常得分
    print("\n[1/2] 计算异常得分...")
    all_scores = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理中"):
            img, zr, _, idx = batch
            img, zr = img.to(device), zr.to(device)
            score = model.anomaly_score(img, zr)
            all_scores.extend(score.cpu().numpy().tolist())
            all_indices.extend(idx.numpy().tolist())
    
    scores = np.array(all_scores)
    indices = np.array(all_indices)
    
    # 生成伪标签
    print("\n[2/2] 生成伪标签...")
    t_normal = np.percentile(scores, cfg.NORMAL_PERCENTILE)
    t_anomaly = np.percentile(scores, cfg.ANOMALY_PERCENTILE)
    
    pseudo_normal = indices[scores <= t_normal].tolist()
    pseudo_anomaly = indices[scores >= t_anomaly].tolist()
    uncertain = indices[(scores > t_normal) & (scores < t_anomaly)].tolist()
    
    print(f"\n【伪标签分布】")
    print(f"  正常阈值: {t_normal:.4f}")
    print(f"  异常阈值: {t_anomaly:.4f}")
    print(f"  高置信正常: {len(pseudo_normal)} ({100*len(pseudo_normal)/len(scores):.1f}%)")
    print(f"  高置信异常: {len(pseudo_anomaly)} ({100*len(pseudo_anomaly)/len(scores):.1f}%)")
    print(f"  不确定: {len(uncertain)} ({100*len(uncertain)/len(scores):.1f}%)")
    
    pseudo_labels = {
        'scores': scores, 'indices': indices,
        'pseudo_normal': pseudo_normal, 'pseudo_anomaly': pseudo_anomaly,
        'uncertain': uncertain, 't_normal': t_normal, 't_anomaly': t_anomaly
    }
    
    np.savez(cfg.STAGE2_DIR / "pseudo_labels_v2.npz", **pseudo_labels)
    
    # 可视化
    plot_score_distribution_v2(scores, t_normal, t_anomaly, cfg)
    
    return pseudo_labels


def train_stage3_v2(model: HybridAnomalyModelV2, pseudo_labels: Dict, 
                    cfg: ThreeStageConfigV2) -> FaultClassifierV2:
    """阶段三：有监督微调"""
    print("\n" + "="*70)
    print("【阶段三】有监督分类器训练")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 加载测试集
    print("\n[1/3] 加载测试数据...")
    test_ds = DualBranchDataset(cfg.TEST_DIR, cfg, use_labels=True)
    
    if len(test_ds) == 0:
        print("[警告] 测试集为空")
        return None
    
    # 划分
    n_train = int(len(test_ds) * 0.8)
    train_subset = torch.utils.data.Subset(test_ds, range(n_train))
    val_subset = torch.utils.data.Subset(test_ds, range(n_train, len(test_ds)))
    
    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 构建分类器
    print("\n[2/3] 构建分类器...")
    classifier = FaultClassifierV2(model.encoder, num_classes=2).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print(f"\n[3/3] 训练 ({cfg.STAGE3_EPOCHS}轮)...")
    history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = 0
    
    for epoch in range(cfg.STAGE3_EPOCHS):
        classifier.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            img, zr, label, _ = batch
            img, zr, label = img.to(device), zr.to(device), label.to(device)
            
            logits = classifier(img, zr)
            loss = criterion(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        classifier.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                img, zr, label, _ = batch
                img, zr = img.to(device), zr.to(device)
                pred = classifier(img, zr).argmax(dim=1)
                val_preds.extend(pred.cpu().tolist())
                val_labels.extend(label.tolist())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state': classifier.state_dict(),
                'epoch': epoch, 'f1': best_f1
            }, cfg.MODEL_DIR / "stage3_best_v2.pth")
        
        if epoch == 15:
            classifier.unfreeze_encoder(1)
        
        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1}] Loss: {train_loss/len(train_loader):.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    plot_training_curves_v2(history, cfg, "stage3")
    print(f"\n【阶段三完成】最佳F1: {best_f1:.4f}")
    
    return classifier


# =============================================================================
# 第9步: 可视化函数
# =============================================================================

def plot_training_curves_v2(history: Dict, cfg: ThreeStageConfigV2, stage: str):
    """绘制训练曲线"""
    setup_plotting()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    if stage == "stage1":
        axes[0].plot(history['epoch'], history['svdd_loss'], 'b-', lw=1.5, label='SVDD损失')
        axes[0].plot(history['epoch'], history['vae_loss'], 'r--', lw=1.5, label='VAE损失')
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失值')
        axes[0].set_title('阶段一: SVDD + VAE 损失')
        axes[0].legend()
        axes[0].grid(True, ls=':', alpha=0.5)
        
        axes[1].plot(history['epoch'], history['total_loss'], 'g-', lw=1.5)
        axes[1].set_xlabel('训练轮次')
        axes[1].set_ylabel('损失值')
        axes[1].set_title('总损失')
        axes[1].grid(True, ls=':', alpha=0.5)
    else:
        axes[0].plot(history['epoch'], history['train_loss'], 'b-', lw=1.5)
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失值')
        axes[0].set_title('阶段三: 训练损失')
        axes[0].grid(True, ls=':', alpha=0.5)
        
        axes[1].plot(history['epoch'], history['val_acc'], 'b-', lw=1.5, label='准确率')
        axes[1].plot(history['epoch'], history['val_f1'], 'r--', lw=1.5, label='F1分数')
        axes[1].set_xlabel('训练轮次')
        axes[1].set_ylabel('指标值')
        axes[1].set_title('验证集性能')
        axes[1].legend()
        axes[1].grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "training_curves" / f"{stage}_curves_v2.png", 
                dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_score_distribution_v2(scores: np.ndarray, t_normal: float, t_anomaly: float,
                               cfg: ThreeStageConfigV2):
    """绘制异常得分分布"""
    setup_plotting()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(scores, bins=100, alpha=0.7, color=COLORS['blue'], edgecolor='black', lw=0.5)
    ax.axvline(t_normal, color=COLORS['green'], ls='--', lw=2, label=f'正常阈值 ({t_normal:.3f})')
    ax.axvline(t_anomaly, color=COLORS['red'], ls='--', lw=2, label=f'异常阈值 ({t_anomaly:.3f})')
    
    ax.set_xlabel('异常得分')
    ax.set_ylabel('样本数')
    ax.set_title('异常得分分布 (V2双分支)')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "score_dist" / "score_distribution_v2.png",
                dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# 第10步: 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='三阶段变压器故障诊断 V2')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--output', type=str, default='./three_stage_results_v2')
    parser.add_argument('--test_data', action='store_true', help='测试数据加载')
    parser.add_argument('--no_zerone', action='store_true', help='禁用Zerone特征')
    parser.add_argument('--no_hetero', action='store_true', help='禁用Hetero图像')
    
    args = parser.parse_args()
    
    # 初始化配置
    cfg = ThreeStageConfigV2()
    if args.data_root:
        cfg.PROJECT_ROOT = Path(args.data_root)
    cfg.OUTPUT_ROOT = Path(args.output)
    cfg.USE_ZERONE_FEATURES = not args.no_zerone
    cfg.USE_HETERO_IMAGE = not args.no_hetero
    cfg.__post_init__()
    cfg.print_config()
    
    # 测试数据加载
    if args.test_data:
        print("\n【测试数据加载】")
        test_ds = DualBranchDataset(cfg.TEST_DIR, cfg, use_labels=True)
        if len(test_ds) > 0:
            img, zr, lbl, idx = test_ds[0]
            print(f"  Hetero图像: {img.shape}")
            print(f"  Zerone特征: {zr.shape}")
            print(f"  标签: {lbl}")
        return
    
    # 执行阶段
    if args.all or args.stage == 1:
        model, _ = train_stage1_v2(cfg)
    else:
        model_path = cfg.MODEL_DIR / "stage1_best_v2.pth"
        if model_path.exists():
            model = HybridAnomalyModelV2(cfg)
            ckpt = torch.load(model_path, map_location=cfg.DEVICE)
            model.load_state_dict(ckpt['model_state'])
            model.center = ckpt['center']
            model = model.to(cfg.DEVICE)
        else:
            print(f"[错误] 未找到模型: {model_path}")
            return
    
    if args.all or args.stage == 2:
        pseudo_labels = run_stage2_v2(model, cfg)
    else:
        pseudo_path = cfg.STAGE2_DIR / "pseudo_labels_v2.npz"
        if pseudo_path.exists():
            data = np.load(pseudo_path, allow_pickle=True)
            pseudo_labels = {k: data[k] for k in data.files}
        else:
            pseudo_labels = None
    
    if args.all or args.stage == 3:
        if pseudo_labels:
            train_stage3_v2(model, pseudo_labels, cfg)
    
    print("\n【完成】结果保存至:", cfg.OUTPUT_ROOT)


if __name__ == "__main__":
    main()

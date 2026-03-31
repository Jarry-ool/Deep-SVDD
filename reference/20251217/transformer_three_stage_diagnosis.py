# -*- coding: utf-8 -*-
"""
transformer_three_stage_diagnosis.py
=====================================

交流变压器振动数据 三阶段渐进式故障诊断系统
适配现有项目结构（config.py / hetero_config.py / zerone_config.py）

【核心思路】
    阶段一 (Stage 1): 纯无监督学习
        - 输入: train + val 目录下的所有数据（忽略正常/故障标签）
        - 方法: Deep SVDD + VAE 融合模型
        - 输出: 每个样本的异常得分
        
    阶段二 (Stage 2): 伪标签生成
        - 输入: 阶段一的异常得分
        - 方法: 基于分位数/MAD的阈值筛选
        - 输出: 高置信正常/高置信异常/不确定 三类伪标签
        
    阶段三 (Stage 3): 有监督微调
        - 输入: 伪标签 + 真实标签（test目录）
        - 方法: 迁移学习分类器
        - 输出: 最终故障诊断模型

【数据目录结构】
    20251016/
    ├── train/              # 训练数据（作为无标签使用）
    │   ├── 114--故障--交流变压器/
    │   ├── 120--正常--交流变压器/
    │   └── ...
    ├── val/                # 验证数据（作为无标签使用）
    │   └── ...
    └── test/               # 测试数据（用于最终评估）
        └── ...

【运行方式】
    python transformer_three_stage_diagnosis.py --stage 1  # 仅运行阶段一
    python transformer_three_stage_diagnosis.py --stage 2  # 仅运行阶段二
    python transformer_three_stage_diagnosis.py --stage 3  # 仅运行阶段三
    python transformer_three_stage_diagnosis.py --all      # 运行全部阶段

Author: 基于 zerone/hetero 代码框架改进
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
matplotlib.use('Agg')  # 非交互式后端，适合服务器
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
class ThreeStageConfig:
    """
    三阶段诊断系统配置类
    
    【参数说明】
        PROJECT_ROOT: 数据根目录，包含 train/val/test 三个子文件夹
        FS: 采样频率 (Hz)，变压器振动数据通常为 8192 Hz
        SIGNAL_LEN: 单条信号长度，与采样频率对应
        INPUT_SIZE: CNN输入图像尺寸，ResNet推荐224
        
    【使用方法】
        cfg = ThreeStageConfig()
        cfg.PROJECT_ROOT = Path("你的数据路径")
    """
    
    # ================= 路径配置 =================
    # 请根据实际情况修改此路径
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(
        r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
    ))
    
    # 输出目录（自动创建）
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results"))
    
    # ================= 信号/物理参数 =================
    FS: float = 8192.0          # 采样频率 (Hz)
    SIGNAL_LEN: int = 8192      # 信号长度（采样点数）
    INPUT_SIZE: int = 224       # CNN输入图像尺寸
    
    # ================= 模型参数 =================
    LATENT_DIM: int = 128       # SVDD隐空间维度
    LATENT_CHANNELS: int = 64   # VAE空间隐变量通道数
    
    # ================= 训练参数 =================
    BATCH_SIZE: int = 16        # 批大小（根据显存调整）
    STAGE1_EPOCHS: int = 50     # 阶段一训练轮数
    STAGE3_EPOCHS: int = 30     # 阶段三训练轮数
    LR: float = 1e-4            # 学习率
    WEIGHT_DECAY: float = 1e-5  # 权重衰减
    
    # SVDD参数
    NU: float = 0.05            # 假设异常比例（5%）
    
    # VAE参数
    BETA_VAE: float = 0.01      # KL散度权重
    BETA_WARMUP: int = 10       # Beta预热轮数
    
    # ================= 伪标签阈值 =================
    NORMAL_PERCENTILE: float = 5.0    # 低于此分位数 = 高置信正常
    ANOMALY_PERCENTILE: float = 99.0  # 高于此分位数 = 高置信异常
    
    # ================= 类别关键词 =================
    # 用于从目录名判断类别
    CLASS_KEYWORDS: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    })
    
    # ================= 设备 =================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ================= 可视化参数 =================
    VIZ_DPI: int = 300          # 图像DPI
    VIZ_LANG: str = "cn"        # 默认语言: cn/en
    
    def __post_init__(self):
        """初始化后处理：创建必要目录"""
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
        
        # 创建可视化子目录
        for subdir in ["training_curves", "score_dist", "confusion", 
                       "roc_pr", "tsne", "samples", "reconstruction"]:
            (self.VIZ_DIR / subdir).mkdir(exist_ok=True)
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("三阶段故障诊断系统配置")
        print("="*60)
        print(f"【数据路径】")
        print(f"  项目根目录: {self.PROJECT_ROOT}")
        print(f"  训练数据: {self.TRAIN_DIR}")
        print(f"  验证数据: {self.VAL_DIR}")
        print(f"  测试数据: {self.TEST_DIR}")
        print(f"【输出路径】")
        print(f"  结果目录: {self.OUTPUT_ROOT}")
        print(f"【信号参数】")
        print(f"  采样频率: {self.FS} Hz")
        print(f"  信号长度: {self.SIGNAL_LEN} 点")
        print(f"【模型参数】")
        print(f"  SVDD隐空间: {self.LATENT_DIM} 维")
        print(f"  VAE隐通道: {self.LATENT_CHANNELS} 通道")
        print(f"【训练参数】")
        print(f"  批大小: {self.BATCH_SIZE}")
        print(f"  学习率: {self.LR}")
        print(f"  设备: {self.DEVICE}")
        print("="*60 + "\n")


# =============================================================================
# 第2步: 可视化设置（符合 IEEE/Nature 期刊标准）
# =============================================================================
def setup_plotting():
    """
    设置matplotlib绘图参数，符合学术期刊标准
    
    【标准参考】
        - IEEE: 单栏宽度 3.5 inch, 双栏 7 inch
        - Nature: 单栏 88mm, 双栏 180mm
        - 分辨率: 300 DPI 以上
    """
    plt.rcParams['font.family'] = 'sans-serif'
    # 优先使用中文字体
    plt.rcParams['font.sans-serif'] = [
        'SimHei',           # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'STHeiti',          # macOS 黑体
        'Arial',            # 通用英文
        'DejaVu Sans'       # Linux 默认
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    plt.rcParams['figure.dpi'] = 150            # 显示DPI
    plt.rcParams['savefig.dpi'] = 300           # 保存DPI
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

# 颜色方案（色盲友好）
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'normal': '#2ca02c',   # 正常-绿色
    'fault': '#d62728',    # 故障-红色
    'uncertain': '#7f7f7f' # 不确定-灰色
}

# 中英文标签映射
LABELS = {
    'cn': {
        'normal': '正常', 'fault': '故障', 'uncertain': '不确定',
        'epoch': '训练轮次', 'loss': '损失值', 'accuracy': '准确率',
        'score': '异常得分', 'count': '样本数', 'true': '真实标签',
        'pred': '预测标签', 'fpr': '假阳性率', 'tpr': '真阳性率',
        'precision': '精确率', 'recall': '召回率', 'f1': 'F1分数',
        'train_loss': '训练损失', 'val_loss': '验证损失',
        'svdd_loss': 'SVDD损失', 'vae_loss': 'VAE损失',
    },
    'en': {
        'normal': 'Normal', 'fault': 'Fault', 'uncertain': 'Uncertain',
        'epoch': 'Epoch', 'loss': 'Loss', 'accuracy': 'Accuracy',
        'score': 'Anomaly Score', 'count': 'Count', 'true': 'True Label',
        'pred': 'Predicted Label', 'fpr': 'False Positive Rate', 
        'tpr': 'True Positive Rate', 'precision': 'Precision', 
        'recall': 'Recall', 'f1': 'F1 Score',
        'train_loss': 'Train Loss', 'val_loss': 'Val Loss',
        'svdd_loss': 'SVDD Loss', 'vae_loss': 'VAE Loss',
    }
}


# =============================================================================
# 第3步: 数据读取与处理工具
# =============================================================================
def parse_signal_value(v: Any, target_len: int = 8192) -> Optional[np.ndarray]:
    """
    解析信号数据，支持字符串和列表格式
    
    【参数】
        v: 原始信号值，可能是逗号分隔的字符串或数字列表
        target_len: 目标信号长度
        
    【返回】
        np.ndarray: 长度为 target_len 的一维数组，失败返回 None
        
    【示例】
        >>> sig = parse_signal_value("0.1,0.2,0.3", target_len=5)
        >>> print(sig)  # [0.1, 0.2, 0.3, 0.0, 0.0]
    """
    try:
        if isinstance(v, str):
            # 清理字符串：去除括号、换行
            s = v.replace("[", "").replace("]", "").replace("\n", " ").replace("\r", "")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            arr = np.array([float(p) for p in parts], dtype=np.float32)
        elif isinstance(v, (list, tuple)):
            arr = np.array([float(x) for x in v], dtype=np.float32)
        else:
            return None
    except Exception:
        return None
    
    # 长度对齐：截断或零填充
    if arr.size >= target_len:
        return arr[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:arr.size] = arr
    return out


def read_jsonl_file(filepath: Path) -> List[Dict]:
    """
    读取JSONL格式文件
    
    【参数】
        filepath: JSONL文件路径
        
    【返回】
        List[Dict]: 解析后的记录列表
    """
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
    """
    读取JSON格式文件（支持多种格式）
    
    【参数】
        filepath: JSON文件路径
        
    【返回】
        List[Dict]: 解析后的记录列表
    """
    records = []
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        data = json.loads(text)
        
        # 格式1: 直接是列表
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        
        # 格式2: 字典包含列表
        if isinstance(data, dict):
            # 检查常见的列表键
            for key in ['data', 'records', 'list', 'items']:
                if key in data and isinstance(data[key], list):
                    return [d for d in data[key] if isinstance(d, dict)]
            
            # 格式3: 字典的值本身就是记录列表
            for v in data.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return v
    except Exception as e:
        print(f"[警告] 读取文件失败: {filepath}, 错误: {e}")
    return records


def signal_to_cwt_image(
    sig: np.ndarray, 
    fs: float, 
    size: int = 224
) -> np.ndarray:
    """
    将一维振动信号转换为三通道CWT时频图像
    
    【通道设计】
        Ch0 (CWT): Morlet小波变换，捕捉时频特征
        Ch1 (STFT): 短时傅里叶变换，捕捉短时频域特征
        Ch2 (Context): 原始信号折叠，保留时域细节
        
    【参数】
        sig: 一维振动信号 (N,)
        fs: 采样频率 (Hz)
        size: 输出图像边长
        
    【返回】
        np.ndarray: 形状为 (3, size, size) 的图像张量
    """
    # 第一步: Z-score归一化
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    
    # 第二步: CWT通道
    #   使用Morlet小波，尺度范围1-128
    #   对数压缩增强对比度
    scales = np.arange(1, min(129, len(sig) // 64 + 1))
    try:
        cwt_matrix, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0/fs)
        cwt_abs = np.abs(cwt_matrix).astype(np.float32)
        cwt_abs = np.log1p(cwt_abs)  # 对数压缩
        c0 = cv2.resize(cwt_abs, (size, size), interpolation=cv2.INTER_LINEAR)
        c0 = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)
    except Exception:
        c0 = np.zeros((size, size), dtype=np.float32)
    
    # 第三步: STFT通道
    try:
        nperseg = min(256, len(sig) // 4)
        noverlap = nperseg // 2
        _, _, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
        mag = np.abs(Zxx).astype(np.float32)
        mag = np.log1p(mag)  # 对数压缩
        c1 = cv2.resize(mag, (size, size), interpolation=cv2.INTER_LINEAR)
        c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-8)
    except Exception:
        c1 = np.zeros((size, size), dtype=np.float32)
    
    # 第四步: Context通道（信号折叠）
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
    
    # 第五步: 堆叠三通道
    img = np.stack([c0, c1, c2], axis=0)
    return img.astype(np.float32)


def get_label_from_path(filepath: Path, class_keywords: Dict) -> Optional[str]:
    """
    从文件路径推断类别标签
    
    【参数】
        filepath: 文件路径
        class_keywords: 类别关键词字典 {"正常": ("正常", "normal"), ...}
        
    【返回】
        str: 类别名称，无法判断时返回 None
    """
    # 检查所有父目录名
    for parent in filepath.parents:
        name = parent.name.lower()
        for cls, keywords in class_keywords.items():
            if any(kw.lower() in name for kw in keywords):
                return cls
    
    # 检查文件名本身
    filename = filepath.name.lower()
    for cls, keywords in class_keywords.items():
        if any(kw.lower() in filename for kw in keywords):
            return cls
    
    return None


# =============================================================================
# 第4步: 数据集类定义
# =============================================================================
class VibrationDataset(Dataset):
    """
    振动信号数据集（通用基类）
    
    【功能】
        - 递归扫描目录下所有 .json/.jsonl 文件
        - 支持按时间戳聚合多传感器数据
        - 支持有标签/无标签两种模式
        
    【参数】
        root_dir: 数据根目录
        cfg: 配置对象
        use_labels: 是否使用标签（False=无监督模式）
        
    【样本定义】
        一个样本 = 同一时间戳下的多传感器数据聚合
        多通道信号通过能量加权合并为单通道
    """
    
    def __init__(
        self, 
        root_dir: Union[str, Path], 
        cfg: ThreeStageConfig,
        use_labels: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        
        # 样本索引: [(文件路径, 时间戳, [信号列表], 标签), ...]
        self.samples: List[Tuple[Path, str, List[np.ndarray], Optional[int]]] = []
        
        # 构建索引
        self._build_index()
    
    def _build_index(self):
        """扫描目录并构建样本索引"""
        if not self.root_dir.exists():
            print(f"[警告] 目录不存在: {self.root_dir}")
            return
        
        # 收集所有数据文件
        files = list(self.root_dir.rglob("*.jsonl")) + list(self.root_dir.rglob("*.json"))
        
        label_counts = Counter()
        
        for fp in tqdm(files, desc=f"扫描 {self.root_dir.name}", leave=False):
            # 读取记录
            if fp.suffix == '.jsonl':
                records = read_jsonl_file(fp)
            else:
                records = read_json_file(fp)
            
            if not records:
                continue
            
            # 获取标签（如果需要）
            label = None
            if self.use_labels:
                label_str = get_label_from_path(fp, self.cfg.CLASS_KEYWORDS)
                if label_str == "正常":
                    label = 0
                elif label_str == "故障":
                    label = 1
                # 无法判断标签的文件跳过
                if label is None:
                    continue
                label_counts[label_str] += 1
            
            # 按时间戳分组
            groups: Dict[str, List[np.ndarray]] = {}
            for rec in records:
                # 获取时间戳
                time_key = None
                for key in ['data_time', 'dataTime', 'timestamp', 'time']:
                    if key in rec and rec[key]:
                        time_key = str(rec[key])
                        break
                if not time_key:
                    continue
                
                # 解析信号
                sig = parse_signal_value(rec.get('signal_value'), self.cfg.SIGNAL_LEN)
                if sig is None:
                    continue
                
                groups.setdefault(time_key, []).append(sig)
            
            # 添加样本
            for time_key, sig_list in groups.items():
                self.samples.append((fp, time_key, sig_list, label))
        
        print(f"[{self.root_dir.name}] 加载 {len(self.samples)} 个样本")
        if self.use_labels and label_counts:
            for lbl, cnt in label_counts.items():
                print(f"  {lbl}: {cnt} 个文件")
    
    def _aggregate_channels(self, sig_list: List[np.ndarray]) -> np.ndarray:
        """
        多通道信号聚合（能量加权）
        
        【原理】
            w_i = E_i / sum(E_j)  其中 E_i = mean(sig_i^2)
            x = sum(w_i * sig_i)
        """
        if len(sig_list) == 1:
            return sig_list[0]
        
        # 堆叠并计算能量
        X = np.stack(sig_list, axis=1)  # (T, U)
        E = np.mean(X ** 2, axis=0) + 1e-12  # (U,)
        w = E / E.sum()  # 归一化权重
        x = X @ w  # 加权平均
        return x.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        获取样本
        
        【返回】
            img: 三通道图像张量 (3, H, W)
            label: 标签（无标签模式返回-1）
            idx: 样本索引
        """
        fp, time_key, sig_list, label = self.samples[idx]
        
        # 聚合多通道
        sig = self._aggregate_channels(sig_list)
        
        # Z-score归一化
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        
        # 转图像
        img = signal_to_cwt_image(sig, self.cfg.FS, self.cfg.INPUT_SIZE)
        
        return torch.from_numpy(img), label if label is not None else -1, idx


# =============================================================================
# 第5步: 模型定义
# =============================================================================
class DeepSVDD(nn.Module):
    """
    Deep Support Vector Data Description
    
    【原理】
        将正常样本映射到隐空间的紧凑超球内
        异常样本偏离超球中心，得分较高
        
    【参考文献】
        Ruff et al., "Deep One-Class Classification", ICML 2018
    """
    
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        
        # 使用ResNet18作为编码器骨干
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )  # 输出: (B, 512)
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )  # 输出: (B, latent_dim)
        
        # 超球中心（训练后固定）
        self.register_buffer('center', torch.zeros(latent_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        h = self.encoder(x)
        z = self.projection(h)
        return z
    
    def init_center(self, dataloader: DataLoader, device: torch.device, eps: float = 0.1):
        """
        使用数据均值初始化超球中心
        
        【注意】
            中心不能太靠近原点，否则会导致trivial solution
        """
        n_samples = 0
        c = torch.zeros(self.projection[-1].out_features, device=device)
        
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                z = self.forward(x)
                c += z.sum(dim=0)
                n_samples += z.size(0)
        c /= n_samples
        
        # 避免中心在原点
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """计算异常得分: 到中心的距离"""
        z = self.forward(x)
        return torch.sum((z - self.center) ** 2, dim=1)


class SpatialResNetVAE(nn.Module):
    """
    空间隐变量VAE（保留7x7空间结构）
    
    【特点】
        - 不将特征压成一维向量，保留空间结构
        - 适合捕捉局部故障特征（如局部放电）
        
    【参考】
        基于 hetero_model.py 的设计
    """
    
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        
        # 编码器: ResNet18骨干
        resnet = models.resnet18(weights=None)
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,  # -> 64 x 56 x 56
            resnet.layer2,  # -> 128 x 28 x 28
            resnet.layer3,  # -> 256 x 14 x 14
            resnet.layer4   # -> 512 x 7 x 7
        )
        
        # 空间隐变量投影
        self.mu_conv = nn.Conv2d(512, latent_channels, 1)
        self.logvar_conv = nn.Conv2d(512, latent_channels, 1)
        
        # 解码器
        self.decoder_input = nn.Conv2d(latent_channels, 512, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 7->14
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 14->28
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 28->56
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),    # 56->112
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 112->224
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        h = self.encoder_stem(x)
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h)
        z = self.reparameterize(mu, logvar)
        z_dec = self.decoder_input(z)
        recon = self.decoder(z_dec)
        
        # 尺寸对齐
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return recon, mu, logvar
    
    def anomaly_score(self, x: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
        """基于重构误差+KL散度的异常得分"""
        recon, mu, logvar = self.forward(x)
        recon_loss = F.l1_loss(recon, x, reduction='none').mean(dim=[1,2,3])
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
        return recon_loss + beta * kl_loss


class HybridAnomalyModel(nn.Module):
    """
    混合异常检测模型（SVDD + VAE融合）
    
    【融合策略】
        final_score = α × SVDD_score + (1-α) × VAE_score
        α 默认为 0.6（SVDD更擅长异常检测）
    """
    
    def __init__(self, cfg: ThreeStageConfig):
        super().__init__()
        self.svdd = DeepSVDD(cfg.LATENT_DIM)
        self.vae = SpatialResNetVAE(cfg.LATENT_CHANNELS)
        self.alpha = 0.6  # SVDD权重
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # SVDD分支
        svdd_z = self.svdd(x)
        svdd_score = torch.sum((svdd_z - self.svdd.center) ** 2, dim=1)
        
        # VAE分支
        recon, mu, logvar = self.vae(x)
        vae_recon_loss = F.l1_loss(recon, x, reduction='none').mean(dim=[1,2,3])
        vae_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
        
        return {
            'svdd_z': svdd_z,
            'svdd_score': svdd_score,
            'recon': recon,
            'vae_recon_loss': vae_recon_loss,
            'vae_kl': vae_kl,
            'mu': mu,
            'logvar': logvar
        }
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """计算融合异常得分"""
        out = self.forward(x)
        svdd_score = out['svdd_score']
        vae_score = out['vae_recon_loss'] + 0.1 * out['vae_kl']
        return self.alpha * svdd_score + (1 - self.alpha) * vae_score


class FaultClassifier(nn.Module):
    """
    故障分类器（阶段三使用）
    
    【特点】
        - 复用阶段一训练的编码器
        - 渐进解冻策略
    """
    
    def __init__(self, pretrained_encoder: nn.Module, num_classes: int = 2):
        super().__init__()
        self.encoder = pretrained_encoder
        
        # 冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 分类头
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if feat.dim() > 2:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.classifier(feat)
    
    def unfreeze_encoder(self, num_blocks: int = 1):
        """渐进解冻编码器"""
        params = list(self.encoder.parameters())
        # 解冻最后 num_blocks * 10 个参数
        n_unfreeze = min(num_blocks * 20, len(params))
        for p in params[-n_unfreeze:]:
            p.requires_grad = True
        print(f"  [解冻] 编码器最后 {n_unfreeze} 个参数")


# =============================================================================
# 第6步: 训练与评估函数
# =============================================================================

def train_stage1(cfg: ThreeStageConfig) -> Tuple[HybridAnomalyModel, Dict]:
    """
    阶段一: 纯无监督训练
    
    【流程】
        1. 加载 train + val 目录下的所有数据（忽略标签）
        2. 预训练VAE编码器
        3. 初始化SVDD超球中心
        4. 联合训练SVDD + VAE
        
    【返回】
        model: 训练好的混合模型
        history: 训练历史
    """
    print("\n" + "="*70)
    print("【阶段一】纯无监督异常检测模型训练")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # ========== 第1步: 加载数据（无标签模式） ==========
    print("\n[1/5] 加载训练数据（无标签模式）...")
    
    train_dataset = VibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False)
    val_dataset = VibrationDataset(cfg.VAL_DIR, cfg, use_labels=False)
    
    # 合并数据集
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"  合并后样本总数: {len(combined_dataset)}")
    
    if len(combined_dataset) == 0:
        raise ValueError("没有加载到任何数据！请检查数据路径。")
    
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    # ========== 第2步: 初始化模型 ==========
    print("\n[2/5] 初始化混合模型...")
    model = HybridAnomalyModel(cfg).to(device)
    
    # ========== 第3步: 预训练VAE ==========
    print("\n[3/5] 预训练VAE编码器（10轮）...")
    vae_opt = torch.optim.Adam(model.vae.parameters(), lr=cfg.LR * 10)
    
    for epoch in range(10):
        model.vae.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"VAE预训练 {epoch+1}/10", leave=False):
            x = batch[0].to(device)
            recon, mu, logvar = model.vae(x)
            
            recon_loss = F.l1_loss(recon, x, reduction='sum') / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + 0.01 * kl_loss
            
            vae_opt.zero_grad()
            loss.backward()
            vae_opt.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}/10 | VAE Loss: {total_loss/len(dataloader):.4f}")
    
    # ========== 第4步: 初始化SVDD中心 ==========
    print("\n[4/5] 初始化SVDD超球中心...")
    model.svdd.init_center(dataloader, device)
    print(f"  中心向量范数: {model.svdd.center.norm().item():.4f}")
    
    # ========== 第5步: 联合训练 ==========
    print(f"\n[5/5] 联合训练SVDD + VAE（{cfg.STAGE1_EPOCHS}轮）...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.STAGE1_EPOCHS)
    
    history = {'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': []}
    best_loss = float('inf')
    
    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        
        # Beta预热
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch + 1) / max(cfg.BETA_WARMUP, 1))
        
        epoch_svdd = 0
        epoch_vae = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}")
        for batch in pbar:
            x = batch[0].to(device)
            out = model(x)
            
            # 损失计算
            svdd_loss = out['svdd_score'].mean()
            vae_loss = out['vae_recon_loss'].mean() + beta * out['vae_kl'].mean()
            total_loss = svdd_loss + vae_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_svdd += svdd_loss.item()
            epoch_vae += vae_loss.item()
            
            pbar.set_postfix({
                'SVDD': f'{svdd_loss.item():.3f}',
                'VAE': f'{vae_loss.item():.3f}',
                'β': f'{beta:.3f}'
            })
        
        scheduler.step()
        
        # 记录历史
        avg_svdd = epoch_svdd / len(dataloader)
        avg_vae = epoch_vae / len(dataloader)
        avg_total = avg_svdd + avg_vae
        
        history['epoch'].append(epoch + 1)
        history['svdd_loss'].append(avg_svdd)
        history['vae_loss'].append(avg_vae)
        history['total_loss'].append(avg_total)
        
        # 保存最佳模型
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.svdd.center,
                'epoch': epoch,
                'loss': best_loss
            }, cfg.MODEL_DIR / "stage1_best.pth")
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
    
    # 保存训练曲线
    plot_training_curves(history, cfg, "stage1")
    
    print(f"\n【阶段一完成】最佳Loss: {best_loss:.4f}")
    print(f"  模型保存至: {cfg.MODEL_DIR / 'stage1_best.pth'}")
    
    return model, history


def run_stage2(model: HybridAnomalyModel, cfg: ThreeStageConfig) -> Dict:
    """
    阶段二: 伪标签生成
    
    【流程】
        1. 对所有样本计算异常得分
        2. 基于分位数确定阈值
        3. 生成伪标签
        
    【返回】
        pseudo_labels: 包含得分、阈值、伪标签索引的字典
    """
    print("\n" + "="*70)
    print("【阶段二】伪标签生成")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    model.eval()
    
    # ========== 第1步: 加载所有数据 ==========
    print("\n[1/3] 加载数据集...")
    train_dataset = VibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False)
    val_dataset = VibrationDataset(cfg.VAL_DIR, cfg, use_labels=False)
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    dataloader = DataLoader(combined_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ========== 第2步: 计算异常得分 ==========
    print("\n[2/3] 计算异常得分...")
    all_scores = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="推理中"):
            x, _, idx = batch
            x = x.to(device)
            score = model.anomaly_score(x)
            all_scores.extend(score.cpu().numpy().tolist())
            all_indices.extend(idx.numpy().tolist())
    
    scores = np.array(all_scores)
    indices = np.array(all_indices)
    
    # ========== 第3步: 确定阈值并生成伪标签 ==========
    print("\n[3/3] 生成伪标签...")
    
    # 使用分位数确定阈值
    t_normal = np.percentile(scores, cfg.NORMAL_PERCENTILE)
    t_anomaly = np.percentile(scores, cfg.ANOMALY_PERCENTILE)
    
    print(f"\n【阈值确定】")
    print(f"  正常阈值 (P{cfg.NORMAL_PERCENTILE}): {t_normal:.4f}")
    print(f"  异常阈值 (P{cfg.ANOMALY_PERCENTILE}): {t_anomaly:.4f}")
    
    # 分配伪标签
    pseudo_normal = indices[scores <= t_normal].tolist()
    pseudo_anomaly = indices[scores >= t_anomaly].tolist()
    uncertain = indices[(scores > t_normal) & (scores < t_anomaly)].tolist()
    
    print(f"\n【伪标签分布】")
    print(f"  高置信正常: {len(pseudo_normal):5d} ({100*len(pseudo_normal)/len(scores):.1f}%)")
    print(f"  高置信异常: {len(pseudo_anomaly):5d} ({100*len(pseudo_anomaly)/len(scores):.1f}%)")
    print(f"  不确定区域: {len(uncertain):5d} ({100*len(uncertain)/len(scores):.1f}%)")
    
    # 保存结果
    pseudo_labels = {
        'scores': scores,
        'indices': indices,
        'pseudo_normal': pseudo_normal,
        'pseudo_anomaly': pseudo_anomaly,
        'uncertain': uncertain,
        't_normal': t_normal,
        't_anomaly': t_anomaly
    }
    
    np.savez(cfg.STAGE2_DIR / "pseudo_labels.npz", **pseudo_labels)
    
    # 可视化
    plot_score_distribution(scores, t_normal, t_anomaly, cfg)
    plot_pseudo_label_pie(pseudo_labels, cfg)
    
    print(f"\n【阶段二完成】")
    print(f"  结果保存至: {cfg.STAGE2_DIR}")
    
    return pseudo_labels


def train_stage3(
    model: HybridAnomalyModel, 
    pseudo_labels: Dict,
    cfg: ThreeStageConfig
) -> FaultClassifier:
    """
    阶段三: 有监督微调
    
    【流程】
        1. 加载测试集（有真实标签）
        2. 构建迁移学习分类器
        3. 渐进解冻训练
        
    【返回】
        classifier: 训练好的分类器
    """
    print("\n" + "="*70)
    print("【阶段三】有监督分类器训练")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # ========== 第1步: 加载测试集（有标签） ==========
    print("\n[1/4] 加载测试数据（有标签模式）...")
    test_dataset = VibrationDataset(cfg.TEST_DIR, cfg, use_labels=True)
    
    if len(test_dataset) == 0:
        print("[警告] 测试集为空，跳过阶段三")
        return None
    
    # 统计类别分布
    labels = [test_dataset.samples[i][3] for i in range(len(test_dataset))]
    label_counts = Counter(labels)
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
    train_labels = [labels[i] for i in train_indices]
    class_counts = np.bincount([l for l in train_labels if l is not None])
    weights = 1.0 / (class_counts[[l for l in train_labels if l is not None]] + 1e-6)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ========== 第2步: 构建分类器 ==========
    print("\n[2/4] 构建迁移学习分类器...")
    classifier = FaultClassifier(model.svdd.encoder, num_classes=2).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # ========== 第3步: 训练 ==========
    print(f"\n[3/4] 开始训练（{cfg.STAGE3_EPOCHS}轮）...")
    
    history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = 0
    
    for epoch in range(cfg.STAGE3_EPOCHS):
        # 训练
        classifier.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}", leave=False):
            x, label, _ = batch
            x = x.to(device)
            label = label.to(device)
            
            logits = classifier(x)
            loss = criterion(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        classifier.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, label, _ = batch
                x = x.to(device)
                logits = classifier(x)
                pred = logits.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy().tolist())
                val_labels.extend(label.numpy().tolist())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        scheduler.step(val_f1)
        
        # 记录
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # 保存最佳
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state': classifier.state_dict(),
                'epoch': epoch,
                'f1': best_f1
            }, cfg.MODEL_DIR / "stage3_best.pth")
        
        # 渐进解冻
        if epoch == 15:
            classifier.unfreeze_encoder(1)
        if epoch == 25:
            classifier.unfreeze_encoder(2)
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    # ========== 第4步: 最终评估 ==========
    print("\n[4/4] 最终评估...")
    
    # 加载最佳模型
    ckpt = torch.load(cfg.MODEL_DIR / "stage3_best.pth")
    classifier.load_state_dict(ckpt['model_state'])
    
    # 在完整测试集上评估
    all_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    classifier.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in all_loader:
            x, label, _ = batch
            x = x.to(device)
            logits = classifier(x)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(label.numpy().tolist())
            all_probs.extend(prob.cpu().numpy().tolist())
    
    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\n【最终评估结果】")
    print(f"  准确率 (Accuracy): {acc:.4f}")
    print(f"  F1分数 (Macro F1): {f1:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    
    # 可视化
    plot_confusion_matrix(all_labels, all_preds, cfg)
    plot_roc_pr_curves(all_labels, all_probs, cfg)
    plot_training_curves(history, cfg, "stage3")
    
    print(f"\n【阶段三完成】")
    print(f"  最佳F1: {best_f1:.4f}")
    print(f"  模型保存至: {cfg.MODEL_DIR / 'stage3_best.pth'}")
    
    return classifier


# =============================================================================
# 第7步: 可视化函数
# =============================================================================

def plot_training_curves(history: Dict, cfg: ThreeStageConfig, stage: str):
    """
    绘制训练曲线
    
    【参数】
        history: 训练历史字典
        cfg: 配置对象
        stage: 阶段名称 (stage1/stage3)
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    if stage == "stage1":
        # SVDD + VAE 损失
        ax = axes[0]
        ax.plot(history['epoch'], history['svdd_loss'], 'b-', linewidth=1.5, label=L['svdd_loss'])
        ax.plot(history['epoch'], history['vae_loss'], 'r--', linewidth=1.5, label=L['vae_loss'])
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['loss'])
        ax.set_title('阶段一: SVDD + VAE 损失曲线' if lang == 'cn' else 'Stage 1: SVDD + VAE Loss')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 总损失
        ax = axes[1]
        ax.plot(history['epoch'], history['total_loss'], 'g-', linewidth=1.5)
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['loss'])
        ax.set_title('总损失曲线' if lang == 'cn' else 'Total Loss')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    else:  # stage3
        # 训练损失
        ax = axes[0]
        ax.plot(history['epoch'], history['train_loss'], 'b-', linewidth=1.5)
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel(L['loss'])
        ax.set_title('阶段三: 训练损失' if lang == 'cn' else 'Stage 3: Training Loss')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 验证指标
        ax = axes[1]
        ax.plot(history['epoch'], history['val_acc'], 'b-', linewidth=1.5, label=L['accuracy'])
        ax.plot(history['epoch'], history['val_f1'], 'r--', linewidth=1.5, label=L['f1'])
        ax.set_xlabel(L['epoch'])
        ax.set_ylabel('指标值' if lang == 'cn' else 'Score')
        ax.set_title('验证集性能' if lang == 'cn' else 'Validation Performance')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "training_curves" / f"{stage}_curves.png", dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_score_distribution(scores: np.ndarray, t_normal: float, t_anomaly: float, cfg: ThreeStageConfig):
    """
    绘制异常得分分布图
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 直方图
    ax.hist(scores, bins=100, alpha=0.7, color=COLORS['blue'], edgecolor='black', linewidth=0.5)
    
    # 阈值线
    ax.axvline(t_normal, color=COLORS['green'], linestyle='--', linewidth=2,
               label=f'{L["normal"]}阈值 ({t_normal:.3f})' if lang == 'cn' else f'Normal threshold ({t_normal:.3f})')
    ax.axvline(t_anomaly, color=COLORS['red'], linestyle='--', linewidth=2,
               label=f'{L["fault"]}阈值 ({t_anomaly:.3f})' if lang == 'cn' else f'Fault threshold ({t_anomaly:.3f})')
    
    ax.set_xlabel(L['score'])
    ax.set_ylabel(L['count'])
    ax.set_title('异常得分分布 & 伪标签阈值' if lang == 'cn' else 'Anomaly Score Distribution & Pseudo-label Thresholds')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "score_dist" / "score_distribution.png", dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_pseudo_label_pie(pseudo_labels: Dict, cfg: ThreeStageConfig):
    """
    绘制伪标签分布饼图
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    counts = [
        len(pseudo_labels['pseudo_normal']),
        len(pseudo_labels['uncertain']),
        len(pseudo_labels['pseudo_anomaly'])
    ]
    labels = [L['normal'], L['uncertain'], L['fault']]
    colors = [COLORS['normal'], COLORS['uncertain'], COLORS['fault']]
    
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=(0.02, 0, 0.02),
        textprops={'fontsize': 12}
    )
    
    ax.set_title('伪标签分布' if lang == 'cn' else 'Pseudo-label Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "score_dist" / "pseudo_label_pie.png", dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: List, y_pred: List, cfg: ThreeStageConfig):
    """
    绘制混淆矩阵
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    classes = [L['normal'], L['fault']]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel=L['pred'],
        ylabel=L['true'],
        title='混淆矩阵' if lang == 'cn' else 'Confusion Matrix'
    )
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "confusion" / "confusion_matrix.png", dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(y_true: List, y_probs: List, cfg: ThreeStageConfig):
    """
    绘制ROC和PR曲线
    """
    setup_plotting()
    lang = cfg.VIZ_LANG
    L = LABELS[lang]
    
    y_score = np.array(y_probs)[:, 1]  # 故障类的概率
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC曲线
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS['blue'], linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel(L['fpr'])
    ax.set_ylabel(L['tpr'])
    ax.set_title('ROC曲线' if lang == 'cn' else 'ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # PR曲线
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(recall, precision, color=COLORS['red'], linewidth=2, label=f'AP = {ap:.3f}')
    ax.set_xlabel(L['recall'])
    ax.set_ylabel(L['precision'])
    ax.set_title('PR曲线' if lang == 'cn' else 'Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(cfg.VIZ_DIR / "roc_pr" / "roc_pr_curves.png", dpi=cfg.VIZ_DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# 第8步: 主函数
# =============================================================================

def main():
    """
    主函数：解析命令行参数并执行相应阶段
    
    【使用方法】
        python transformer_three_stage_diagnosis.py --stage 1
        python transformer_three_stage_diagnosis.py --stage 2
        python transformer_three_stage_diagnosis.py --stage 3
        python transformer_three_stage_diagnosis.py --all
    """
    parser = argparse.ArgumentParser(description='三阶段变压器故障诊断系统')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--output', type=str, default='./three_stage_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化配置
    cfg = ThreeStageConfig()
    if args.data_root:
        cfg.PROJECT_ROOT = Path(args.data_root)
        cfg.__post_init__()
    cfg.OUTPUT_ROOT = Path(args.output)
    cfg.__post_init__()
    
    cfg.print_config()
    
    # 检查数据目录
    if not cfg.PROJECT_ROOT.exists():
        print(f"[错误] 数据目录不存在: {cfg.PROJECT_ROOT}")
        print("请修改配置文件中的 PROJECT_ROOT 或使用 --data_root 参数指定")
        return
    
    # 执行阶段
    if args.all or args.stage == 1:
        model, _ = train_stage1(cfg)
    else:
        # 加载已有模型
        model_path = cfg.MODEL_DIR / "stage1_best.pth"
        if model_path.exists():
            print(f"加载阶段一模型: {model_path}")
            model = HybridAnomalyModel(cfg)
            ckpt = torch.load(model_path, map_location=cfg.DEVICE)
            model.load_state_dict(ckpt['model_state'])
            model.svdd.center = ckpt['center']
            model = model.to(cfg.DEVICE)
        else:
            print(f"[错误] 未找到阶段一模型: {model_path}")
            print("请先运行 --stage 1")
            return
    
    if args.all or args.stage == 2:
        pseudo_labels = run_stage2(model, cfg)
    else:
        # 加载已有伪标签
        pseudo_path = cfg.STAGE2_DIR / "pseudo_labels.npz"
        if pseudo_path.exists():
            print(f"加载伪标签: {pseudo_path}")
            data = np.load(pseudo_path, allow_pickle=True)
            pseudo_labels = {k: data[k] for k in data.files}
        else:
            pseudo_labels = None
    
    if args.all or args.stage == 3:
        if pseudo_labels is None:
            print("[警告] 未找到伪标签，跳过阶段三")
        else:
            train_stage3(model, pseudo_labels, cfg)
    
    print("\n" + "="*70)
    print("【全部完成】")
    print(f"结果保存至: {cfg.OUTPUT_ROOT}")
    print("="*70)


if __name__ == "__main__":
    main()

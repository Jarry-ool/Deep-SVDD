# -*- coding: utf-8 -*-
"""
config.py - 配置类和全局常量
============================

V5.12 更新:
- 新增 A40 台式机配置 (BATCH_SIZE=1026, LR=2e-4)
- 新增 VAL_TEST_SPLIT 参数
- 新增 CSV 原始数据路径配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import torch


# =============================================================================
# 全局常量与特征配置
# =============================================================================
TIME_DOMAIN_DIM = 15
STFT_BAND_DIM = 127  
PSD_BAND_DIM = 1050   # 1000 (1-1000Hz@1Hz) + 50 (1001-2000Hz@20Hz聚合)
HIGH_FREQ_DIM = 8
TOTAL_FEAT_DIM = TIME_DOMAIN_DIM + STFT_BAND_DIM + PSD_BAND_DIM + HIGH_FREQ_DIM  # 1200

# 特征Schema
FEAT_SCHEMA = [
    ("time", TIME_DOMAIN_DIM),
    ("stft", STFT_BAND_DIM),
    ("psd", PSD_BAND_DIM),
    ("hf", HIGH_FREQ_DIM),
]

# 颜色配置
COLORS = {
    'normal': '#2ecc71',
    'fault': '#e74c3c', 
    'uncertain': '#f39c12',
    'primary': '#3498db',
    'secondary': '#9b59b6',
}

# 标签文本
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
# 配置类定义
# =============================================================================

@dataclass
class ThreeStageConfigV5:
    """
    三阶段诊断系统配置类 V5.12
    
    V5.12 更新:
    - A40 台式机配置 (大批量 + 高学习率)
    - CSV 原始数据读取支持
    - Val/Test 自动划分
    """
    
    # === 设备配置 (V5.12: A40台式机) ===
    DEVICE: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 训练超参数 (V5.12: A40优化) ===
    # 笔记本配置: BATCH_SIZE=16, LR=1e-4
    # A40配置:    BATCH_SIZE=1026, LR=2e-4
    BATCH_SIZE: int = 1026      # V5.12: A40台式机大批量
    LR: float = 2e-4            # V5.12: 配合大批量的学习率
    WEIGHT_DECAY: float = 1e-4
    STAGE1_EPOCHS: int = 50
    STAGE3_EPOCHS: int = 100
    PATIENCE: int = 15
    
    # === 信号参数 ===
    SIGNAL_LEN: int = 8192
    FS: float = 8192.0
    
    # === 模型架构参数 ===
    INPUT_SIZE: int = 224
    FEATURE_DIM: int = 512
    
    # === 支线模式 ===
    BRANCH_MODE: str = 'dual'     # 'hetero', 'zerone', 'dual'
    FUSION_MODE: str = 'gmu'      # 'concat', 'attention', 'gate', 'gmu'
    ZERONE_USE_CNN: bool = True   # Zerone分支使用CNN还是MLP
    
    # === V5.1 正则化特性 ===
    USE_MODALITY_DROPOUT: bool = True
    MODALITY_DROPOUT_RATE: float = 0.2
    USE_DOMAIN_ADAPTATION: bool = True
    DA_MODE: str = 'mmd'          # 'mmd', 'coral', 'dann'
    USE_DANN: bool = False
    DA_WEIGHT: float = 0.1
    DROPOUT_RATE: float = 0.3
    LABEL_SMOOTHING: float = 0.05
    
    # === V5.1 全局归一化 ===
    USE_GLOBAL_NORMALIZATION: bool = True
    
    # === V5.11 标签翻转检测 ===
    ENABLE_LABEL_FLIP_DETECTION: bool = True
    LABEL_FLIP_THRESHOLD: float = 0.15
    
    # === V5.12 数据划分 ===
    VAL_TEST_SPLIT: float = 0.5   # Val:Test = 50:50
    
    # === 可视化参数 ===
    SAMPLE_PREVIEW_COUNT: int = 8   # 样本预览数量
    VIZ_DPI: int = 150              # 图像DPI
    LANGS: tuple = ('cn', 'en')     # 输出语言版本
    
    # === 阶段一参数 ===
    ANOMALY_TYPE: str = 'svdd'    # 'svdd' or 'vae'
    SVDD_NU: float = 0.1
    VAE_LATENT_DIM: int = 32
    
    # === 阶段二参数 ===
    QUANTILE_LOW: float = 0.2
    QUANTILE_HIGH: float = 0.8
    
    # === 数据路径 ===
    PROJECT_ROOT: Path = field(default_factory=lambda: Path("E:/CODE/DATA/vibration_data_2022_"))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v5"))
    LABELED_DATA_DIR: Path = None  # 已标注数据目录
    
    # 类别关键词
    CLASS_KEYWORDS: Dict = field(default_factory=lambda: {
        "正常": ["正常", "normal", "good", "健康"],
        "故障": ["故障", "异常", "fault", "abnormal", "defect", "error"]
    })
    
    # 自动生成的路径
    RAW_DATA_DIR: Path = None
    TRAIN_DIR: Path = None    # JSONL训练数据目录
    VAL_DIR: Path = None      # JSONL验证数据目录
    TEST_DIR: Path = None     # JSONL测试数据目录
    STAGE1_DIR: Path = None
    STAGE2_DIR: Path = None
    STAGE3_DIR: Path = None
    MODEL_DIR: Path = None
    BRANCH_DIR: Path = None
    VIZ_SUBDIRS: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化衍生路径"""
        # 原始CSV数据目录
        self.RAW_DATA_DIR = self.PROJECT_ROOT / "交流站" / "00 振动原始数据"
        
        # JSONL数据目录 (兼容V5.11)
        self.TRAIN_DIR = self.PROJECT_ROOT / "TRAIN"
        self.VAL_DIR = self.PROJECT_ROOT / "VAL"
        self.TEST_DIR = self.PROJECT_ROOT / "TEST"
        
        # 已标注数据目录
        if self.LABELED_DATA_DIR is None:
            self.LABELED_DATA_DIR = self.PROJECT_ROOT
        
        # 输出目录结构
        self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}"
        if self.BRANCH_MODE == 'dual':
            self.BRANCH_DIR = self.BRANCH_DIR / f"fusion_{self.FUSION_MODE}"
        
        self.STAGE1_DIR = self.BRANCH_DIR / "stage1_anomaly"
        self.STAGE2_DIR = self.BRANCH_DIR / "stage2_pseudo"
        self.STAGE3_DIR = self.BRANCH_DIR / "stage3_classify"
        self.MODEL_DIR = self.BRANCH_DIR / "models"
        
        # 创建目录
        for d in [self.STAGE1_DIR, self.STAGE2_DIR, self.STAGE3_DIR, self.MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 可视化子目录
        self.VIZ_SUBDIRS = {
            "training_curves": self.STAGE1_DIR / "training_curves",
            "distributions": self.STAGE1_DIR / "distributions",
            "feature_preview": self.STAGE1_DIR / "feature_preview",
            "tsne": self.STAGE3_DIR / "tsne",
            "confusion": self.STAGE3_DIR / "confusion",
            "roc_pr": self.STAGE3_DIR / "roc_pr",
            "misclassified": self.STAGE3_DIR / "misclassified",
            "channel_info": self.STAGE1_DIR / "channel_info",
            "data_split": self.STAGE1_DIR / "data_split",
        }
        for v_dir in self.VIZ_SUBDIRS.values():
            v_dir.mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("【三阶段诊断系统配置 V5.12】")
        print("="*60)
        print(f"  设备: {self.DEVICE}")
        print(f"  支线模式: {self.BRANCH_MODE}")
        if self.BRANCH_MODE == 'dual':
            print(f"  融合策略: {self.FUSION_MODE}")
        print(f"  Zerone分支: {'CNN' if self.ZERONE_USE_CNN else 'MLP'}")
        print("-"*60)
        print(f"  批量大小: {self.BATCH_SIZE}")
        print(f"  学习率: {self.LR}")
        print(f"  信号长度: {self.SIGNAL_LEN}")
        print(f"  采样率: {self.FS} Hz")
        print("-"*60)
        print(f"  模态Dropout: {self.USE_MODALITY_DROPOUT}")
        print(f"  域适应: {self.USE_DOMAIN_ADAPTATION} ({self.DA_MODE})")
        print(f"  全局归一化: {self.USE_GLOBAL_NORMALIZATION}")
        print(f"  标签翻转检测: {self.ENABLE_LABEL_FLIP_DETECTION}")
        print("-"*60)
        print(f"  Val/Test划分: {int(self.VAL_TEST_SPLIT*100)}:{int((1-self.VAL_TEST_SPLIT)*100)}")
        print(f"  原始数据: {self.RAW_DATA_DIR}")
        print(f"  输出目录: {self.BRANCH_DIR}")
        print("="*60 + "\n")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'version': '5.12',
            'device': self.DEVICE,
            'batch_size': self.BATCH_SIZE,
            'lr': self.LR,
            'branch_mode': self.BRANCH_MODE,
            'fusion_mode': self.FUSION_MODE,
            'zerone_use_cnn': self.ZERONE_USE_CNN,
            'signal_len': self.SIGNAL_LEN,
            'fs': self.FS,
            'use_global_normalization': self.USE_GLOBAL_NORMALIZATION,
            'val_test_split': self.VAL_TEST_SPLIT,
            'anomaly_type': self.ANOMALY_TYPE,
            'output_root': str(self.OUTPUT_ROOT),
        }

# -*- coding: utf-8 -*-
"""
config.py - 配置类 (V5.12 精简版)
=================================

使用方法:
    # 基础配置
    cfg = ThreeStageConfigV5()
    
    # 自定义配置
    cfg = ThreeStageConfigV5(
        PROJECT_ROOT=Path("E:/DATA"),
        BATCH_SIZE=128,
        NUM_WORKERS=4,
        BRANCH_MODE='hetero',  # 'hetero', 'zerone', 'dual'
    )
    
    # 命令行传参
    python main.py --batch_size 128 --num_workers 4 --branch hetero
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import torch


# =============================================================================
# 全局常量 (1200维特征体系)
# =============================================================================
TIME_DOMAIN_DIM = 15
STFT_BAND_DIM = 127  
PSD_BAND_DIM = 1050
HIGH_FREQ_DIM = 8
TOTAL_FEAT_DIM = TIME_DOMAIN_DIM + STFT_BAND_DIM + PSD_BAND_DIM + HIGH_FREQ_DIM  # 1200

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
# 配置类
# =============================================================================

@dataclass
class ThreeStageConfigV5:
    """
    三阶段诊断系统配置类 V5.12
    
    核心参数:
        BATCH_SIZE: 批量大小 (A40建议128-256, 笔记本16-32)
        NUM_WORKERS: DataLoader并行数 (Windows建议2-4, 卡死用0)
        BRANCH_MODE: 支线模式 ('hetero'/'zerone'/'dual')
    
    性能优化:
        - BRANCH_MODE='hetero': 只计算CWT/STFT图像，省去zerone计算
        - BRANCH_MODE='zerone': 只计算1200维特征，省去图像生成
        - 特征缓存: 首次计算后自动缓存，后续直接加载
    """
    
    # === 设备 ===
    DEVICE: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 训练参数 (核心) ===
    BATCH_SIZE: int = 128           # 批量大小
    LR: float = 1e-4                # 学习率（Deep SVDD官方推荐）
    WEIGHT_DECAY: float = 1e-4
    STAGE1_EPOCHS: int = 50         # Stage1训练轮数
    STAGE3_EPOCHS: int = 100        # Stage3训练轮数
    PATIENCE: int = 15              # 早停耐心值
    NUM_WORKERS: int = 4            # DataLoader workers (Windows建议2-4)
    
    # === 信号参数 ===
    SIGNAL_LEN: int = 8192
    FS: float = 8192.0
    
    # === 模型架构 ===
    INPUT_SIZE: int = 224
    FEATURE_DIM: int = 512
    
    # === 支线模式 (影响__getitem__计算量) ===
    BRANCH_MODE: str = 'dual'       # 'hetero': 只用图像, 'zerone': 只用特征, 'dual': 都用
    FUSION_MODE: str = 'gmu'        # 'concat', 'attention', 'gate', 'gmu'
    ZERONE_USE_CNN: bool = True     # Zerone分支用CNN(图像)还是MLP(向量)
    USE_GPU_HETERO: bool = True     # GPU加速hetero图像生成 (推荐开启)
    
    # === 正则化 ===
    USE_MODALITY_DROPOUT: bool = True
    MODALITY_DROPOUT_RATE: float = 0.2
    USE_DOMAIN_ADAPTATION: bool = True
    DA_MODE: str = 'mmd'
    USE_DANN: bool = False
    DA_WEIGHT: float = 0.1
    DROPOUT_RATE: float = 0.3
    LABEL_SMOOTHING: float = 0.05
    
    # === 其他 ===
    USE_GLOBAL_NORMALIZATION: bool = True
    ENABLE_LABEL_FLIP_DETECTION: bool = True
    LABEL_FLIP_THRESHOLD: float = 0.15
    VAL_TEST_SPLIT: float = 0.5
    
    # === 可视化 ===
    SAMPLE_PREVIEW_COUNT: int = 8
    VIZ_DPI: int = 150
    LANGS: tuple = ('cn', 'en')
    
    # === 阶段参数 ===
    ANOMALY_TYPE: str = 'svdd'
    SVDD_NU: float = 0.1
    VAE_LATENT_DIM: int = 32
    QUANTILE_LOW: float = 0.2
    QUANTILE_HIGH: float = 0.8
    
    # === SVDD防崩塌参数 ===
    SVDD_LATENT_DIM: int = 64           # SVDD潜在维度（原128，降低容量）
    SVDD_LR_SCALE: float = 0.1          # SVDD投影头学习率缩放（相对于主学习率）
    SVDD_WEIGHT_DECAY: float = 1e-3     # SVDD投影头权重衰减（增加正则）
    SVDD_CENTER_EPS: float = 0.1        # 中心初始化时的epsilon
    
    # === 数据路径 ===
    PROJECT_ROOT: Path = field(default_factory=lambda: Path("E:/CODE/DATA/vibration_data"))
    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("./three_stage_results_v512"))
    LABELED_DATA_DIR: Path = None
    
    CLASS_KEYWORDS: Dict = field(default_factory=lambda: {
        "正常": ["正常", "normal", "good", "健康"],
        "故障": ["故障", "异常", "fault", "abnormal", "defect", "error"]
    })
    
    # 自动生成的路径
    RAW_DATA_DIR: Path = None
    TRAIN_DIR: Path = None
    VAL_DIR: Path = None
    TEST_DIR: Path = None
    STAGE1_DIR: Path = None
    STAGE2_DIR: Path = None
    STAGE3_DIR: Path = None
    MODEL_DIR: Path = None
    BRANCH_DIR: Path = None
    VIZ_SUBDIRS: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化衍生路径"""
        # CSV数据目录 (智能判断)
        potential_raw_dir = self.PROJECT_ROOT / "交流站" / "00 振动原始数据"
        if potential_raw_dir.exists():
            self.RAW_DATA_DIR = potential_raw_dir
        else:
            self.RAW_DATA_DIR = self.PROJECT_ROOT
        
        # JSONL目录
        self.TRAIN_DIR = self.PROJECT_ROOT / "TRAIN"
        self.VAL_DIR = self.PROJECT_ROOT / "VAL"
        self.TEST_DIR = self.PROJECT_ROOT / "TEST"
        
        if self.LABELED_DATA_DIR is None:
            self.LABELED_DATA_DIR = self.PROJECT_ROOT
        
        # 输出目录
        self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}"
        if self.BRANCH_MODE == 'dual':
            self.BRANCH_DIR = self.BRANCH_DIR / f"fusion_{self.FUSION_MODE}"
            if not self.ZERONE_USE_CNN:
                # dual模式下MLP变体
                self.BRANCH_DIR = self.OUTPUT_ROOT / f"branch_{self.BRANCH_MODE}" / f"fusion_{self.FUSION_MODE}_mlp"
        elif self.BRANCH_MODE == 'zerone' and not self.ZERONE_USE_CNN:
            # zerone_mlp 单独目录
            self.BRANCH_DIR = self.OUTPUT_ROOT / "branch_zerone_mlp"
        
        self.STAGE1_DIR = self.BRANCH_DIR / "stage1_anomaly"
        self.STAGE2_DIR = self.BRANCH_DIR / "stage2_pseudo"
        self.STAGE3_DIR = self.BRANCH_DIR / "stage3_classify"
        self.MODEL_DIR = self.BRANCH_DIR / "models"
        
        for d in [self.STAGE1_DIR, self.STAGE2_DIR, self.STAGE3_DIR, self.MODEL_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
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
        print(f"  DataLoader workers: {self.NUM_WORKERS}")
        print("-"*60)
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
            'num_workers': self.NUM_WORKERS,
            'branch_mode': self.BRANCH_MODE,
            'fusion_mode': self.FUSION_MODE,
            'zerone_use_cnn': self.ZERONE_USE_CNN,
            'signal_len': self.SIGNAL_LEN,
            'fs': self.FS,
        }

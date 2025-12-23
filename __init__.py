# -*- coding: utf-8 -*-
"""
V5.12 模块化变压器振动诊断系统
================================

三阶段渐进式故障诊断系统，支持:
- CSV原始数据读取
- 通道名称映射
- 数据泄露防止
- A40 GPU优化

模块结构:
- config.py       : 配置类和全局常量
- features.py     : 特征提取函数 (1200维Zerone特征)
- utils.py        : 工具函数（归一化、日志、检查点）
- visualization.py: 可视化工具
- datasets.py     : 数据集类 (JSONL/CSV/Labeled)
- models.py       : 模型定义 (编码器、融合、SVDD/VAE、分类器)
- data_manager.py : 数据管理（CSV读取、通道映射、数据划分）
- training.py     : 训练函数（Stage 1/2/3）
- main.py         : 主入口

使用示例:
    # 完整流程
    python -m v5_12_modular.main --all --data_root "path/to/data"
    
    # 仅准备数据
    python -m v5_12_modular.main --prepare_data --labeled_dir "path/to/labeled"

Author: PhD Candidate (Electrical Eng.)
Version: 5.12 Modular
"""

__version__ = "5.12"
__author__ = "PhD Candidate"

# 配置
from .config import ThreeStageConfigV5, COLORS, LABELS, TOTAL_FEAT_DIM

# 特征提取
from .features import extract_zerone_features, split_feature_vector

# 工具类
from .utils import (
    GlobalNormalizer, GLOBAL_NORMALIZER,
    TrainingLogger, CheckpointManager, EarlyStopping,
    set_seed, count_parameters, save_json, load_json
)

# 数据管理
from .data_manager import (
    ChannelNameManager, DataSplitManager, CHANNEL_MANAGER,
    read_vibration_csv, scan_csv_files, generate_sample_id
)

# 数据集
from .datasets import (
    TransformerVibrationDataset,
    CSVVibrationDataset,
    LabeledVibrationDataset,
    generate_hetero_image,
    vector_to_image_raster
)

# 模型
from .models import (
    ModalityDropout,
    ConcatFusion, AttentionFusion, GatedFusion, GMUFusion,
    HeteroCNN, ZeroneCNN, ZeroneMLP, BranchEncoderV5,
    AnomalyModelV5, FaultClassifierV5,
    DomainDiscriminator,
    compute_mmd_loss, compute_coral_loss
)

# 训练
from .training import (
    train_stage1, run_stage2, train_stage3,
    run_full_pipeline
)

# 可视化
from .visualization import VisualizationManager

# 主入口
from .main import prepare_datasets

# -*- coding: utf-8 -*-
"""
main.py - V5.12 模块化变压器振动诊断系统主入口
==============================================

使用方法:

1. 完整流程 (从CSV原始数据):
   python -m v5_12_modular.main --all \
       --data_root "E:/CODE/DATA/vibration_data_2022_" \
       --labeled_dir "E:/CODE/DATA/20251016" \
       --output_root "./results_v512"

2. 仅数据准备:
   python -m v5_12_modular.main --prepare_data \
       --data_root "E:/CODE/DATA/vibration_data_2022_" \
       --labeled_dir "E:/CODE/DATA/20251016"

3. 使用预过滤结果:
   python -m v5_12_modular.main --all \
       --filter_output "./filtered_data"

4. 单独运行某阶段:
   python -m v5_12_modular.main --stage 1
   python -m v5_12_modular.main --stage 2
   python -m v5_12_modular.main --stage 3

Author: PhD Candidate (Electrical Eng.)
Version: 5.12 Modular
"""

import argparse
import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from config import ThreeStageConfigV5
from utils import GlobalNormalizer, GLOBAL_NORMALIZER, set_seed, save_json
from data_manager import (
    DataSplitManager, ChannelNameManager, CHANNEL_MANAGER,
    scan_csv_files, read_vibration_csv
)
from datasets import (
    TransformerVibrationDataset, CSVVibrationDataset, LabeledVibrationDataset
)
from training import train_stage1, run_stage2, train_stage3, run_full_pipeline
from visualization import VisualizationManager


def prepare_datasets(cfg: ThreeStageConfigV5, 
                     filter_output_dir: Path = None,
                     labeled_dir: Path = None,
                     val_ratio: float = 0.5):
    """
    准备数据集的统一入口
    
    优先级:
    1. filter_output_dir: 使用预过滤脚本的输出
    2. labeled_dir: 自动划分已标注数据
    3. 默认: 从cfg.PROJECT_ROOT加载
    
    返回:
        (train_ds, val_ds, test_ds, split_manager)
    """
    print("\n" + "="*60)
    print("【数据准备】")
    print("="*60)
    
    split_manager = DataSplitManager(cfg)
    
    # 方式1: 使用预过滤结果
    if filter_output_dir and Path(filter_output_dir).exists():
        print(f"\n[方式1] 使用预过滤结果: {filter_output_dir}")
        split_manager.load_from_filter_output(filter_output_dir)
        
        # 加载通道映射
        mapping_path = Path(filter_output_dir) / "channel_mapping.json"
        if mapping_path.exists():
            CHANNEL_MANAGER.load_mapping(mapping_path)
        
        # 创建数据集
        val_ds = LabeledVibrationDataset(
            split_manager.val_samples, cfg, 
            split_name="VAL", normalizer=None
        )
        test_ds = LabeledVibrationDataset(
            split_manager.test_samples, cfg,
            split_name="TEST", normalizer=None
        )
        
        # 扫描训练数据
        csv_files = scan_csv_files(cfg.RAW_DATA_DIR)
        if csv_files:
            train_ds = CSVVibrationDataset(
                csv_files, cfg, use_labels=False, split_name="TRAIN",
                normalizer=None, excluded_ids=split_manager.excluded_ids
            )
        else:
            train_ds = TransformerVibrationDataset(
                cfg.PROJECT_ROOT, cfg, use_labels=False,
                split_name="TRAIN", normalizer=None
            )
    
    # 方式2: 自动划分已标注数据
    elif labeled_dir and Path(labeled_dir).exists():
        print(f"\n[方式2] 自动划分已标注数据: {labeled_dir}")
        
        # 划分val/test
        split_manager.auto_split_labeled_data(
            labeled_dir, val_ratio=val_ratio,
            class_keywords=cfg.CLASS_KEYWORDS
        )
        
        # 生成排除ID
        split_manager.generate_excluded_ids_from_labeled()
        
        # 创建数据集
        val_ds = LabeledVibrationDataset(
            split_manager.val_samples, cfg,
            split_name="VAL", normalizer=None
        )
        test_ds = LabeledVibrationDataset(
            split_manager.test_samples, cfg,
            split_name="TEST", normalizer=None
        )
        
        # 扫描训练数据 (排除已标注)
        csv_files = scan_csv_files(cfg.RAW_DATA_DIR)
        if csv_files:
            train_ds = CSVVibrationDataset(
                csv_files, cfg, use_labels=False, split_name="TRAIN",
                normalizer=None, excluded_ids=split_manager.excluded_ids
            )
        else:
            train_ds = TransformerVibrationDataset(
                cfg.PROJECT_ROOT, cfg, use_labels=False,
                split_name="TRAIN", normalizer=None
            )
        
        # 保存划分结果
        split_manager.save_split(cfg.VIZ_SUBDIRS['data_split'])
        CHANNEL_MANAGER.save_mapping(cfg.VIZ_SUBDIRS['channel_info'] / "channel_mapping.json")
    
    # 方式3: 默认加载
    else:

        root = Path(cfg.PROJECT_ROOT)
        print(f"\n[方式3] 默认加载: {cfg.PROJECT_ROOT}")
        
        train_dir = root / "train"
        val_dir   = root / "val"
        test_dir  = root / "test"

        if train_dir.exists() and val_dir.exists() and test_dir.exists():
            print(f"[方式3-A] 检测到划分目录 -> train/val/test")
            train_ds = TransformerVibrationDataset(train_dir, cfg, use_labels=False, split_name="TRAIN", normalizer=None)
            val_ds   = TransformerVibrationDataset(val_dir,   cfg, use_labels=True,  split_name="VAL",   normalizer=None)
            test_ds  = TransformerVibrationDataset(test_dir,  cfg, use_labels=True,  split_name="TEST",  normalizer=None)
        else:
            print(f"[方式3-B] 未检测到划分目录，使用整体数据进行")
            train_ds = TransformerVibrationDataset(
                cfg.PROJECT_ROOT, cfg, use_labels=False,
                split_name="TRAIN", normalizer=None
            )
            val_ds = TransformerVibrationDataset(
                cfg.PROJECT_ROOT, cfg, use_labels=True,
                split_name="VAL", normalizer=None
            )
            test_ds = TransformerVibrationDataset(
                cfg.PROJECT_ROOT, cfg, use_labels=True,
                split_name="TEST", normalizer=None
            )
    
    # 计算全局归一化参数
    print("\n[归一化] 计算全局归一化参数...")
    if hasattr(train_ds, 'get_all_features_for_normalization'):
        all_features = train_ds.get_all_features_for_normalization()
        # 限制采样数量
        if len(all_features) > 5000:
            indices = random.sample(range(len(all_features)), 5000)
            all_features = [all_features[i] for i in indices]
        
        GLOBAL_NORMALIZER.fit(all_features)
        GLOBAL_NORMALIZER.save(cfg.MODEL_DIR / "global_normalizer.npz")
    
    # 更新数据集的归一化器
    for ds in [train_ds, val_ds, test_ds]:
        if hasattr(ds, 'normalizer'):
            ds.normalizer = GLOBAL_NORMALIZER
    
    # 打印摘要
    split_manager.print_summary()
    CHANNEL_MANAGER.print_summary()
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_ds)}")
    print(f"  验证集: {len(val_ds)}")
    print(f"  测试集: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds, split_manager


def main():
    parser = argparse.ArgumentParser(description="V5.12 变压器振动诊断系统")
    
    # 运行模式
    parser.add_argument('--all', action='store_true', help='运行完整流程')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='运行指定阶段')
    parser.add_argument('--prepare_data', action='store_true', help='仅准备数据')
    parser.add_argument('--test_data', action='store_true', help='测试数据读取')
    
    # 数据路径
    parser.add_argument('--data_root', type=str, default="E:/CODE/DATA/vibration_data_2022_",
                        help='数据根目录')
    parser.add_argument('--labeled_dir', type=str, default=None,
                        help='已标注数据目录')
    parser.add_argument('--filter_output', type=str, default=None,
                        help='预过滤脚本输出目录')
    parser.add_argument('--output_root', type=str, default="./three_stage_results_v512",
                        help='输出根目录')
    
    # 模型配置
    parser.add_argument('--branch', type=str, default='dual',
                        choices=['hetero', 'zerone', 'dual'], help='支线模式')
    parser.add_argument('--fusion_mode', type=str, default='gmu',
                        choices=['concat', 'attention', 'gate', 'gmu'], help='融合模式')
    parser.add_argument('--zerone_mlp', action='store_true', help='Zerone使用MLP而非CNN')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1026, help='批量大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--epochs1', type=int, default=50, help='Stage1训练轮数')
    parser.add_argument('--epochs3', type=int, default=100, help='Stage3训练轮数')
    parser.add_argument('--val_ratio', type=float, default=0.5, help='验证集比例')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    cfg = ThreeStageConfigV5(
        PROJECT_ROOT=Path(args.data_root),
        OUTPUT_ROOT=Path(args.output_root),
        BATCH_SIZE=args.batch_size,
        LR=args.lr,
        STAGE1_EPOCHS=args.epochs1,
        STAGE3_EPOCHS=args.epochs3,
        BRANCH_MODE=args.branch,
        FUSION_MODE=args.fusion_mode,
        ZERONE_USE_CNN=not args.zerone_mlp,
        VAL_TEST_SPLIT=args.val_ratio,
    )
    
    if args.device:
        cfg.DEVICE = args.device
    
    if args.labeled_dir:
        cfg.LABELED_DATA_DIR = Path(args.labeled_dir)
    
    # 打印配置
    cfg.print_config()
    
    # ========== 测试数据读取 ==========
    if args.test_data:
        print("\n【测试数据读取】")
        csv_files = scan_csv_files(cfg.RAW_DATA_DIR)
        if csv_files:
            print(f"找到 {len(csv_files)} 个CSV文件")
            # 测试读取第一个文件
            if len(csv_files) > 0:
                csv_data = read_vibration_csv(csv_files[0], cfg)
                if csv_data:
                    print(f"  文件: {csv_files[0].name}")
                    print(f"  数据形状: {csv_data['data'].shape}")
                    print(f"  通道数: {csv_data['channel_count']}")
                    print(f"  通道名: {csv_data['channel_names'][:3]}...")
        
        CHANNEL_MANAGER.print_summary()
        return
    
    # ========== 准备数据 ==========
    train_ds, val_ds, test_ds, split_manager = prepare_datasets(
        cfg,
        filter_output_dir=args.filter_output,
        labeled_dir=args.labeled_dir,
        val_ratio=args.val_ratio
    )
    
    if args.prepare_data:
        print("\n【数据准备完成】")
        return
    
    # ========== 运行训练 ==========
    if args.all:
        print("\n【运行完整流程】")
        results = run_full_pipeline(
            cfg, train_ds, val_ds, test_ds,
            resume_stage1=Path(args.resume) if args.resume else None
        )
        
        # 保存最终结果
        final_results = {
            'config': cfg.to_dict(),
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'test_samples': len(test_ds),
            'timestamp': datetime.now().isoformat(),
        }
        save_json(final_results, cfg.OUTPUT_ROOT / "run_summary.json")
        
        print("\n" + "="*60)
        print("【完整流程结束】")
        print(f"输出目录: {cfg.OUTPUT_ROOT}")
        print("="*60)
    
    elif args.stage == 1:
        print("\n【运行阶段一】")
        model, history = train_stage1(cfg, train_ds, 
                                       resume_from=Path(args.resume) if args.resume else None)
    
    elif args.stage == 2:
        print("\n【运行阶段二】")
        # 加载Stage1模型
        model_path = cfg.MODEL_DIR / "stage1" / "stage1_best_model.pth"
        if not model_path.exists():
            print(f"[错误] 未找到Stage1模型: {model_path}")
            return
        
        from models import AnomalyModelV5
        model = AnomalyModelV5(
            branch_mode=cfg.BRANCH_MODE,
            fusion_mode=cfg.FUSION_MODE,
            zerone_use_cnn=cfg.ZERONE_USE_CNN
        ).to(cfg.DEVICE)
        
        ckpt = torch.load(model_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state'])
        if 'center' in ckpt:
            model.center = ckpt['center']
        
        pseudo_labels = run_stage2(model, cfg, train_ds)
    
    elif args.stage == 3:
        print("\n【运行阶段三】")
        # 加载Stage1模型
        model_path = cfg.MODEL_DIR / "stage1" / "stage1_best_model.pth"
        pseudo_path = cfg.STAGE2_DIR / "pseudo_labels.npz"
        
        if not model_path.exists():
            print(f"[错误] 未找到Stage1模型: {model_path}")
            return
        if not pseudo_path.exists():
            print(f"[错误] 未找到伪标签: {pseudo_path}")
            return
        
        from models import AnomalyModelV5
        model = AnomalyModelV5(
            branch_mode=cfg.BRANCH_MODE,
            fusion_mode=cfg.FUSION_MODE,
            zerone_use_cnn=cfg.ZERONE_USE_CNN
        ).to(cfg.DEVICE)
        
        ckpt = torch.load(model_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state'])
        
        pseudo_labels = dict(np.load(pseudo_path, allow_pickle=True))
        
        classifier = train_stage3(model, pseudo_labels, cfg, val_ds, test_ds, train_ds)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

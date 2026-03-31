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

# Windows多进程兼容性: 限制底层库的线程数，避免多进程卡死
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import json
import random
import logging
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


def setup_logging(output_dir: Path, log_name: str = "run"):
    """
    设置日志系统，同时输出到控制台和文件
    
    Args:
        output_dir: 日志输出目录
        log_name: 日志文件名前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{log_name}_{timestamp}.log"
    
    # 创建文件日志handler (只写文件，不写控制台)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    
    # 重定向print：控制台输出 + 写日志文件
    class LoggerWriter:
        def __init__(self, logger, original_stdout):
            self.logger = logger
            self.original_stdout = original_stdout
            
        def write(self, message):
            # 先输出到控制台
            self.original_stdout.write(message)
            
            # 过滤掉tqdm的进度条更新，只记录有意义的内容到日志
            if message and message.strip() and '\r' not in message:
                self.logger.info(message.rstrip())
            
        def flush(self):
            self.original_stdout.flush()
    
    sys.stdout = LoggerWriter(logger, sys.__stdout__)
    
    print(f"[日志] 日志文件: {log_file}")
    return log_file


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
        print(f"\n[方式3] 默认加载: {cfg.PROJECT_ROOT}")
        
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
    
    # 计算全局归一化参数 (保存到公共位置，所有模式共用)
    normalizer_path = cfg.OUTPUT_ROOT / "global_normalizer.npz"
    
    if normalizer_path.exists():
        print("\n[归一化] 加载已有参数...")
        GLOBAL_NORMALIZER.load(normalizer_path)
    elif hasattr(train_ds, 'get_all_features_for_normalization'):
        print("\n[归一化] 计算全局归一化参数...")
        all_features = train_ds.get_all_features_for_normalization()
        # 限制采样数量
        if len(all_features) > 5000:
            indices = random.sample(range(len(all_features)), 5000)
            all_features = [all_features[i] for i in indices]
        
        GLOBAL_NORMALIZER.fit(all_features)
        GLOBAL_NORMALIZER.save(normalizer_path)
        print(f"[归一化] 参数已保存: {normalizer_path}")
    
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
    parser.add_argument('--run_all_modes', action='store_true', 
                        help='运行所有8种模式组合并生成对比图')
    parser.add_argument('--start_from', type=int, default=1,
                        help='从第几个模式开始 (1-8), 用于断点续跑')
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
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers数 (Windows建议0-4)')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    
    args = parser.parse_args()
    
    # 初始化日志系统
    output_root = Path(args.output_root)
    log_file = setup_logging(output_root / "logs", "training")
    
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
        NUM_WORKERS=args.num_workers,
    )
    
    if args.device:
        cfg.DEVICE = args.device
    
    if args.labeled_dir:
        cfg.LABELED_DATA_DIR = Path(args.labeled_dir)
    
    # 打印配置
    cfg.print_config()
    
    # ========== 运行所有模式实验 ==========
    if args.run_all_modes:
        run_all_modes_experiment(args)
        return
    
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
        # model, history = train_stage1(cfg, train_ds, 
        #                                resume_from=Path(args.resume) if args.resume else None)
        print("\n【运行阶段一】")
        model, history = train_stage1(
            cfg,
            train_ds=train_ds,
            val_ds=val_ds,   # ✅ 关键：把已加载的验证集传进去
            resume_from=Path(args.resume) if args.resume else None
        )
    
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


def run_all_modes_experiment(args):
    """
    运行所有8种模式组合并生成对比图
    
    优化: 只加载一次数据，8种模式复用
    
    模式组合:
    1. hetero
    2. zerone (CNN)
    3. zerone (MLP)
    4. dual + concat
    5. dual + attention
    6. dual + gate
    7. dual + gmu (CNN)
    8. dual + gmu (MLP)
    """
    from config import ThreeStageConfigV5
    from training import run_full_pipeline
    from utils import set_seed, save_json
    
    print("\n" + "="*70)
    print("【运行所有模式组合实验】")
    print("="*70)
    
    # 定义所有模式组合
    MODE_CONFIGS = [
        {'name': 'hetero', 'branch': 'hetero', 'fusion': 'concat', 'zerone_cnn': True},
        {'name': 'zerone_cnn', 'branch': 'zerone', 'fusion': 'concat', 'zerone_cnn': True},
        {'name': 'zerone_mlp', 'branch': 'zerone', 'fusion': 'concat', 'zerone_cnn': False},
        {'name': 'dual_concat', 'branch': 'dual', 'fusion': 'concat', 'zerone_cnn': True},
        {'name': 'dual_attention', 'branch': 'dual', 'fusion': 'attention', 'zerone_cnn': True},
        {'name': 'dual_gate', 'branch': 'dual', 'fusion': 'gate', 'zerone_cnn': True},
        {'name': 'dual_gmu_cnn', 'branch': 'dual', 'fusion': 'gmu', 'zerone_cnn': True},
        {'name': 'dual_gmu_mlp', 'branch': 'dual', 'fusion': 'gmu', 'zerone_cnn': False},
    ]
    
    output_root = Path(args.output_root)
    compare_dir = output_root / "mode_comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 只加载一次数据 ==========
    print("\n[数据准备] 加载数据集 (只加载一次，所有模式复用)...")
    
    # 用第一个模式的配置加载数据
    base_cfg = ThreeStageConfigV5(
        PROJECT_ROOT=Path(args.data_root),
        OUTPUT_ROOT=output_root,
        BATCH_SIZE=args.batch_size,
        LR=args.lr,
        STAGE1_EPOCHS=args.epochs1,
        STAGE3_EPOCHS=args.epochs3,
        BRANCH_MODE='dual',  # 用dual模式加载
        FUSION_MODE='gmu',
        ZERONE_USE_CNN=True,
        VAL_TEST_SPLIT=args.val_ratio,
        NUM_WORKERS=args.num_workers,
    )
    
    if args.device:
        base_cfg.DEVICE = args.device
    if args.labeled_dir:
        base_cfg.LABELED_DATA_DIR = Path(args.labeled_dir)
    
    # 一次性加载数据
    train_ds, val_ds, test_ds, split_manager = prepare_datasets(
        base_cfg,
        filter_output_dir=args.filter_output,
        labeled_dir=args.labeled_dir,
        val_ratio=args.val_ratio
    )
    
    print(f"\n[数据准备完成] 训练集: {len(train_ds)}, 验证集: {len(val_ds)}, 测试集: {len(test_ds)}")
    
    # ========== 运行所有模式 ==========
    all_results = []
    start_idx = args.start_from - 1  # 转为0-indexed
    
    if start_idx > 0:
        print(f"\n[跳过] 从模式 {args.start_from} 开始，跳过前 {start_idx} 个模式")
    
    for i, mode_cfg in enumerate(MODE_CONFIGS):
        # 跳过已完成的模式
        if i < start_idx:
            print(f"\n[跳过] 模式 {i+1}/8: {mode_cfg['name']}")
            continue
        
        print(f"\n{'='*70}")
        print(f"【模式 {i+1}/8】{mode_cfg['name']}")
        print(f"  branch={mode_cfg['branch']}, fusion={mode_cfg['fusion']}, zerone_cnn={mode_cfg['zerone_cnn']}")
        print("="*70)
        
        set_seed(args.seed)
        
        # 创建本模式的配置 (复用数据)
        cfg = ThreeStageConfigV5(
            PROJECT_ROOT=Path(args.data_root),
            OUTPUT_ROOT=output_root,
            BATCH_SIZE=args.batch_size,
            LR=args.lr,
            STAGE1_EPOCHS=args.epochs1,
            STAGE3_EPOCHS=args.epochs3,
            BRANCH_MODE=mode_cfg['branch'],
            FUSION_MODE=mode_cfg['fusion'],
            ZERONE_USE_CNN=mode_cfg['zerone_cnn'],
            VAL_TEST_SPLIT=args.val_ratio,
            NUM_WORKERS=args.num_workers,
        )
        
        if args.device:
            cfg.DEVICE = args.device
        if args.labeled_dir:
            cfg.LABELED_DATA_DIR = Path(args.labeled_dir)
        
        # ★ 关键：更新数据集的cfg引用，让branch_mode生效
        for ds in [train_ds, val_ds, test_ds]:
            if hasattr(ds, 'cfg'):
                ds.cfg = cfg
        
        # 运行完整流程 (复用已加载的数据)
        try:
            results = run_full_pipeline(cfg, train_ds, val_ds, test_ds)
            
            # 读取评估结果
            eval_path = cfg.STAGE3_DIR / "evaluation_results.json"
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    eval_results = json.load(f)
            else:
                eval_results = {}
            
            result_entry = {
                'name': mode_cfg['name'],
                'branch': mode_cfg['branch'],
                'fusion': mode_cfg['fusion'],
                'zerone_cnn': mode_cfg['zerone_cnn'],
                'test_accuracy': eval_results.get('test_accuracy', 0),
                'test_f1': eval_results.get('test_f1', 0),
                'test_precision': eval_results.get('test_precision', 0),
                'test_recall': eval_results.get('test_recall', 0),
                'best_val_f1': eval_results.get('best_val_f1', 0),
            }
            all_results.append(result_entry)
            
            print(f"\n  ✅ {mode_cfg['name']} 完成: Acc={result_entry['test_accuracy']:.4f}, F1={result_entry['test_f1']:.4f}, Prec={result_entry['test_precision']:.4f}, Rec={result_entry['test_recall']:.4f}")
            
        except Exception as e:
            print(f"\n  ❌ {mode_cfg['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'name': mode_cfg['name'],
                'branch': mode_cfg['branch'],
                'fusion': mode_cfg['fusion'],
                'zerone_cnn': mode_cfg['zerone_cnn'],
                'test_accuracy': 0,
                'test_f1': 0,
                'test_precision': 0,
                'test_recall': 0,
                'best_val_f1': 0,
                'error': str(e),
            })
        
        # 清理GPU显存，避免累积导致OOM
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    # 保存所有结果
    save_json(all_results, compare_dir / "all_results.json")
    
    # 生成对比图
    print("\n" + "="*70)
    print("【生成对比图】")
    print("="*70)
    
    generate_comparison_plots(all_results, compare_dir)
    
    print(f"\n对比结果已保存: {compare_dir}")
    
    return all_results


def generate_comparison_plots(results: list, output_dir: Path):
    """
    生成多维度对比图 (中英文各一套)
    
    图表:
    1. 总体对比 (8种模式)
    2. 支线模式对比 (hetero vs zerone vs dual)
    3. 融合模式对比 (concat vs attention vs gate vs gmu)
    4. Zerone架构对比 (CNN vs MLP)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 语言配置
    LANG_CONFIG = {
        'cn': {
            'title_overall': '所有模式性能对比',
            'title_branch': '支线模式性能对比',
            'title_fusion': '融合模式性能对比',
            'title_zerone': 'Zerone架构性能对比 (CNN vs MLP)',
            'title_radar': '多维度性能雷达图',
            'title_heatmap': '模式×指标热力图',
            'accuracy': '准确率',
            'f1': 'F1分数',
            'precision': '精确率',
            'recall': '召回率',
            'mode': '模式',
            'branch_names': {'hetero': 'Hetero\n(图像)', 'zerone': 'Zerone\n(特征)', 'dual': 'Dual\n(双分支)'},
            'fusion_names': {'concat': '拼接', 'attention': '注意力', 'gate': '门控', 'gmu': 'GMU'},
            'zerone_names': {'cnn': 'CNN\n(图像)', 'mlp': 'MLP\n(向量)'},
        },
        'en': {
            'title_overall': 'Overall Mode Performance Comparison',
            'title_branch': 'Branch Mode Performance Comparison',
            'title_fusion': 'Fusion Mode Performance Comparison',
            'title_zerone': 'Zerone Architecture Comparison (CNN vs MLP)',
            'title_radar': 'Multi-dimensional Performance Radar',
            'title_heatmap': 'Mode × Metric Heatmap',
            'accuracy': 'Accuracy',
            'f1': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall',
            'mode': 'Mode',
            'branch_names': {'hetero': 'Hetero\n(Image)', 'zerone': 'Zerone\n(Feature)', 'dual': 'Dual\n(Both)'},
            'fusion_names': {'concat': 'Concat', 'attention': 'Attention', 'gate': 'Gate', 'gmu': 'GMU'},
            'zerone_names': {'cnn': 'CNN\n(Image)', 'mlp': 'MLP\n(Vector)'},
        }
    }
    
    for lang, L in LANG_CONFIG.items():
        # 设置字体
        if lang == 'cn':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = True
        
        # ========== 1. 总体对比 ==========
        fig, ax = plt.subplots(figsize=(16, 6))
        
        names = [r['name'] for r in results]
        accs = [r['test_accuracy'] * 100 for r in results]
        f1s = [r['test_f1'] * 100 for r in results]
        precs = [r.get('test_precision', 0) * 100 for r in results]
        recs = [r.get('test_recall', 0) * 100 for r in results]
        
        x = np.arange(len(names))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, accs, width, label=L['accuracy'], color='#3498db', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, f1s, width, label=L['f1'], color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, precs, width, label=L['precision'], color='#2ecc71', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, recs, width, label=L['recall'], color='#9b59b6', alpha=0.8)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(L['title_overall'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_overall_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 2. 支线模式对比 ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按支线模式分组取最佳结果
        branch_results = {}
        for r in results:
            branch = r['branch']
            if branch not in branch_results or r['test_f1'] > branch_results[branch]['test_f1']:
                branch_results[branch] = r
        
        branches = ['hetero', 'zerone', 'dual']
        branch_names = [L['branch_names'].get(b, b) for b in branches]
        branch_accs = [branch_results.get(b, {}).get('test_accuracy', 0) * 100 for b in branches]
        branch_f1s = [branch_results.get(b, {}).get('test_f1', 0) * 100 for b in branches]
        branch_precs = [branch_results.get(b, {}).get('test_precision', 0) * 100 for b in branches]
        branch_recs = [branch_results.get(b, {}).get('test_recall', 0) * 100 for b in branches]
        
        x = np.arange(len(branches))
        width = 0.2
        
        bars1 = ax.bar(x - 1.5*width, branch_accs, width, label=L['accuracy'], color='#3498db', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, branch_f1s, width, label=L['f1'], color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, branch_precs, width, label=L['precision'], color='#2ecc71', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, branch_recs, width, label=L['recall'], color='#9b59b6', alpha=0.8)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(L['title_branch'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(branch_names, fontsize=11)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_branch_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 3. 融合模式对比 ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 只看dual模式下的不同融合
        fusion_results = {r['fusion']: r for r in results if r['branch'] == 'dual' and r['zerone_cnn']}
        
        fusions = ['concat', 'attention', 'gate', 'gmu']
        fusion_names = [L['fusion_names'].get(f, f) for f in fusions]
        fusion_accs = [fusion_results.get(f, {}).get('test_accuracy', 0) * 100 for f in fusions]
        fusion_f1s = [fusion_results.get(f, {}).get('test_f1', 0) * 100 for f in fusions]
        
        x = np.arange(len(fusions))
        
        bars1 = ax.bar(x - width/2, fusion_accs, width, label=L['accuracy'], color='#f39c12', alpha=0.8)
        bars2 = ax.bar(x + width/2, fusion_f1s, width, label=L['f1'], color='#1abc9c', alpha=0.8)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(L['title_fusion'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(fusion_names, fontsize=11)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_fusion_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 4. Zerone架构对比 (CNN vs MLP) ==========
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # zerone分支: CNN vs MLP
        zerone_cnn = next((r for r in results if r['name'] == 'zerone_cnn'), {})
        zerone_mlp = next((r for r in results if r['name'] == 'zerone_mlp'), {})
        # dual+gmu: CNN vs MLP
        dual_cnn = next((r for r in results if r['name'] == 'dual_gmu_cnn'), {})
        dual_mlp = next((r for r in results if r['name'] == 'dual_gmu_mlp'), {})
        
        categories = [f"Zerone\n{L['zerone_names']['cnn']}", f"Zerone\n{L['zerone_names']['mlp']}",
                     f"Dual+GMU\n{L['zerone_names']['cnn']}", f"Dual+GMU\n{L['zerone_names']['mlp']}"]
        cnn_mlp_accs = [
            zerone_cnn.get('test_accuracy', 0) * 100,
            zerone_mlp.get('test_accuracy', 0) * 100,
            dual_cnn.get('test_accuracy', 0) * 100,
            dual_mlp.get('test_accuracy', 0) * 100,
        ]
        cnn_mlp_f1s = [
            zerone_cnn.get('test_f1', 0) * 100,
            zerone_mlp.get('test_f1', 0) * 100,
            dual_cnn.get('test_f1', 0) * 100,
            dual_mlp.get('test_f1', 0) * 100,
        ]
        
        x = np.arange(len(categories))
        colors_acc = ['#3498db', '#85c1e9', '#3498db', '#85c1e9']
        colors_f1 = ['#e74c3c', '#f1948a', '#e74c3c', '#f1948a']
        
        bars1 = ax.bar(x - width/2, cnn_mlp_accs, width, label=L['accuracy'], color=colors_acc, alpha=0.8)
        bars2 = ax.bar(x + width/2, cnn_mlp_f1s, width, label=L['f1'], color=colors_f1, alpha=0.8)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(L['title_zerone'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_zerone_arch_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 5. 雷达图 ==========
        metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
        metric_labels = [L['accuracy'], L['f1'], L['precision'], L['recall']]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, r in enumerate(results):
            values = [r.get(m, 0) * 100 for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=r['name'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title(L['title_radar'], fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_radar_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 6. 热力图 ==========
        import seaborn as sns
        
        names = [r['name'] for r in results]
        data = np.array([[r.get(m, 0) * 100 for m in metrics] for r in results])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                    xticklabels=metric_labels, yticklabels=names,
                    vmin=0, vmax=100, cbar_kws={'label': '%'})
        
        ax.set_title(L['title_heatmap'], fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_heatmap_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print(f"  ✅ 对比图已生成 (中英文各6张)")
    print(f"     - compare_overall_*.png     总体对比")
    print(f"     - compare_branch_*.png      支线模式对比")
    print(f"     - compare_fusion_*.png      融合模式对比")
    print(f"     - compare_zerone_arch_*.png Zerone架构对比")
    print(f"     - compare_radar_*.png       雷达图")
    print(f"     - compare_heatmap_*.png     热力图")


if __name__ == "__main__":
    main()

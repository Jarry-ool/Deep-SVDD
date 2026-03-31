# -*- coding: utf-8 -*-
"""
单独运行t-SNE可视化 (用于补跑)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from config import ThreeStageConfigV5
from models import FaultClassifierV5, AnomalyModelV5
from datasets import LabeledVibrationDataset
from visualization import VisualizationManager
from gpu_transforms import GPUHeteroTransform
from data_manager import DataSplitManager


def run_tsne_for_branch(branch_mode='hetero'):
    """为指定branch重新生成t-SNE"""
    
    output_root = Path("./three_stage_results_v512")
    filter_output = Path("./filtered_output")
    
    cfg = ThreeStageConfigV5(
        PROJECT_ROOT=Path("E:/CODE/code/trans_data/00 振动原始数据"),
        OUTPUT_ROOT=output_root,
        BRANCH_MODE=branch_mode,
        FUSION_MODE='concat',
        ZERONE_USE_CNN=True,
        BATCH_SIZE=256,
    )
    
    device = torch.device(cfg.DEVICE)
    branch_dir = output_root / f"branch_{branch_mode}"
    model_dir = branch_dir / "models"
    
    # 加载测试数据（使用和main.py一样的方式）
    print("[1/5] 加载测试数据...")
    split_manager = DataSplitManager(cfg)
    split_manager.load_from_filter_output(filter_output)
    
    # 检查是否成功加载
    print(f"  验证集样本: {len(split_manager.val_samples)}")
    print(f"  测试集样本: {len(split_manager.test_samples)}")
    
    if len(split_manager.test_samples) == 0:
        print("  ❌ 没有加载到测试数据!")
        print(f"  请检查 {filter_output} 目录结构")
        return
    
    # 创建测试数据集
    test_ds = LabeledVibrationDataset(
        split_manager.test_samples, cfg, 
        split_name="TEST", normalizer=None
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"  测试数据集大小: {len(test_ds)}")
    
    # GPU变换
    use_gpu_hetero = cfg.USE_GPU_HETERO and branch_mode in ('hetero', 'dual')
    gpu_transform = None
    if use_gpu_hetero:
        gpu_transform = GPUHeteroTransform(size=cfg.INPUT_SIZE, device=device)
        print("  [GPU加速] 已启用")
    
    # Step 1: 加载Stage1模型 (AnomalyModelV5)
    print("\n[2/5] 加载Stage1模型...")
    stage1_model = AnomalyModelV5(
        branch_mode=branch_mode,
        fusion_mode='concat',
        zerone_use_cnn=True,
    ).to(device)
    
    stage1_path = model_dir / "stage1" / "stage1_best_model.pth"
    if stage1_path.exists():
        ckpt = torch.load(stage1_path, map_location=device)
        stage1_model.load_state_dict(ckpt['model_state'])
        print(f"  ✓ 已加载: {stage1_path}")
    else:
        print(f"  ❌ 找不到: {stage1_path}")
        return
    
    # Step 2: 创建分类器，使用Stage1的encoder
    print("\n[3/5] 创建分类器...")
    classifier = FaultClassifierV5(
        encoder=stage1_model.encoder,
        num_classes=2,
        freeze_encoder=True,
    ).to(device)
    
    # Step 3: 加载Stage3分类器权重
    stage3_path = model_dir / "stage3" / "stage3_best_model.pth"
    if stage3_path.exists():
        ckpt = torch.load(stage3_path, map_location=device)
        classifier.load_state_dict(ckpt['model_state'])
        print(f"  ✓ 已加载: {stage3_path}")
    else:
        print(f"  ❌ 找不到: {stage3_path}")
        return
    
    # 提取特征
    print("\n[4/5] 提取特征...")
    classifier.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            data, zr, labels, _ = batch
            
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = classifier(img, zr)
            if 'h' in out:
                all_features.extend(out['h'].cpu().numpy())
                all_labels.extend(labels.tolist())
    
    print(f"  提取特征数: {len(all_features)}")
    
    if len(all_features) == 0:
        print("  ❌ 没有提取到特征!")
        return
    
    # 生成t-SNE
    print("\n[5/5] 生成t-SNE...")
    viz_dir = branch_dir / "stage3_classify"
    viz = VisualizationManager(viz_dir)
    
    for lang in ('cn', 'en'):
        viz.plot_tsne(np.array(all_features), np.array(all_labels), lang=lang)
    
    print(f"\n✅ t-SNE已保存到: {viz_dir / 'tsne'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', default='hetero', help='branch模式: hetero/zerone/dual')
    args = parser.parse_args()
    
    run_tsne_for_branch(args.branch)

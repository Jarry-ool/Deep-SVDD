# -*- coding: utf-8 -*-
"""
regenerate_cache.py - 重新生成zerone特征缓存
"""
import os
import sys

# 设置环境变量避免OpenMP警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, '.')

from pathlib import Path

def main():
    print("=" * 50)
    print("重建Zerone特征缓存")
    print("=" * 50)
    
    # 原始CSV数据目录（训练数据来源）
    raw_data_dir = Path("E:/CODE/code/trans_data/00 振动原始数据")
    
    print(f"\n[1] 检查原始数据目录: {raw_data_dir}")
    
    if not raw_data_dir.exists():
        print(f"  ❌ 目录不存在!")
        return
    
    print(f"  ✓ 目录存在")
    
    # 导入模块
    print("\n[2] 导入模块...")
    from config import ThreeStageConfigV5
    from datasets import CSVVibrationDataset
    from data_manager import scan_csv_files, DataSplitManager
    print("  ✓ 模块导入成功")
    
    # 配置
    print("\n[3] 创建配置...")
    cfg = ThreeStageConfigV5(
        PROJECT_ROOT=raw_data_dir,
        OUTPUT_ROOT=Path("./three_stage_results_v512"),
    )
    print(f"  OUTPUT_ROOT: {cfg.OUTPUT_ROOT}")
    print(f"  缓存目录: {cfg.OUTPUT_ROOT / 'cache'}")
    
    # 加载排除ID（从filtered_output）
    print("\n[4] 加载排除ID...")
    filter_output = Path("./filtered_output")
    excluded_ids = set()
    
    if filter_output.exists():
        split_manager = DataSplitManager(filter_output)
        excluded_ids = split_manager.excluded_ids
        print(f"  排除ID数: {len(excluded_ids)}")
    else:
        print(f"  [警告] filtered_output不存在，不排除任何样本")
    
    # 扫描原始CSV文件
    print("\n[5] 扫描原始CSV文件...")
    csv_files = scan_csv_files(raw_data_dir)
    print(f"  找到 {len(csv_files)} 个CSV文件")
    
    if len(csv_files) == 0:
        print("  ❌ 没找到CSV文件!")
        return
    
    # 创建数据集
    print("\n[6] 创建数据集 (会切分样本)...")
    train_ds = CSVVibrationDataset(
        csv_files, cfg, 
        split_name="TRAIN",
        excluded_ids=excluded_ids
    )
    print(f"  样本数: {len(train_ds)}")
    
    # 提取特征
    print("\n[7] 提取特征并保存缓存 (需要几十分钟)...")
    features = train_ds.get_all_features_for_normalization()
    
    print(f"\n{'=' * 50}")
    print(f"✅ 缓存重建完成!")
    print(f"  特征数量: {len(features)}")
    print(f"  保存位置: {cfg.OUTPUT_ROOT / 'cache'}")
    print("=" * 50)

if __name__ == "__main__":
    main()

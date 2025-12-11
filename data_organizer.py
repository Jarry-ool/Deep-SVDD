# -*- coding: utf-8 -*-
"""
data_organizer.py
==================
数据整理工具：帮助你把大量无标签数据分配到 train/val 目录

【使用场景】
    你有：
    - 几万条无标签的JSONL文件（新采集的）
    - 几千条有标签的数据（现有的，目录名带"正常/故障"）
    
【工具功能】
    1. 扫描无标签数据，自动分配到 train/val
    2. 保持有标签数据在 test/ 目录
    3. 生成数据统计报告

【使用方法】
    # 查看使用指南
    python data_organizer.py --guide
    
    # 检查现有有标签数据
    python data_organizer.py --check "E:\\...\\20251016\\test"
    
    # 整理无标签数据到 train/val
    python data_organizer.py --source "D:\\无标签数据" --target "E:\\...\\20251016"

Author: 电气工程变压器故障诊断项目
"""

import os
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from collections import Counter
from datetime import datetime


def count_samples_in_file(filepath: Path) -> int:
    """
    统计单个JSONL/JSON文件中的样本数
    
    【说明】
        一个样本 = 一条振动信号记录
        JSONL文件：每行一个样本
        JSON文件：可能是列表或嵌套结构
    """
    count = 0
    try:
        text = filepath.read_text(encoding='utf-8', errors='ignore')
        
        if filepath.suffix == '.jsonl':
            # JSONL: 每行一个记录
            for line in text.splitlines():
                if line.strip():
                    count += 1
        else:
            # JSON: 尝试解析
            data = json.loads(text)
            if isinstance(data, list):
                count = len(data)
            elif isinstance(data, dict):
                # 检查常见的列表字段
                for key in ['data', 'records', 'list', 'items']:
                    if key in data and isinstance(data[key], list):
                        count = len(data[key])
                        break
                # 检查字典的值是否为列表
                if count == 0:
                    for v in data.values():
                        if isinstance(v, list):
                            count = len(v)
                            break
    except Exception:
        pass
    
    return max(count, 1)  # 至少算1个


def organize_unlabeled_data(
    source_dir: str,
    target_root: str,
    train_ratio: float = 0.7,
    seed: int = 42,
    copy_mode: str = 'copy'  # 'copy' 或 'move'
):
    """
    将无标签数据整理到 train/val 目录
    
    【参数】
        source_dir: 无标签数据源目录
        target_root: 目标根目录（包含 train/val/test）
        train_ratio: 训练集比例（默认70%）
        seed: 随机种子
        copy_mode: 'copy'=复制文件, 'move'=移动文件
        
    【注意】
        - 不会修改 test/ 目录下的任何内容
        - 无标签数据会放入 train/unlabeled/ 和 val/unlabeled/
    """
    random.seed(seed)
    
    source = Path(source_dir)
    target = Path(target_root)
    
    if not source.exists():
        print(f"[错误] 源目录不存在: {source}")
        return
    
    # 创建目录
    train_dir = target / "train" / "unlabeled"
    val_dir = target / "val" / "unlabeled"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有数据文件
    print(f"\n[1/3] 扫描数据文件...")
    files = list(source.rglob("*.jsonl")) + list(source.rglob("*.json"))
    
    if not files:
        print(f"[警告] 未找到任何 .jsonl 或 .json 文件")
        return
    
    # 统计样本数
    print(f"[2/3] 统计样本数量...")
    total_samples = 0
    file_samples = []
    for fp in files:
        n = count_samples_in_file(fp)
        file_samples.append((fp, n))
        total_samples += n
    
    # 随机打乱
    random.shuffle(file_samples)
    
    n_train = int(len(file_samples) * train_ratio)
    train_files = file_samples[:n_train]
    val_files = file_samples[n_train:]
    
    train_samples = sum(n for _, n in train_files)
    val_samples = sum(n for _, n in val_files)
    
    print(f"\n【数据整理计划】")
    print(f"  源目录: {source}")
    print(f"  目标目录: {target}")
    print(f"  ─────────────────────────────────")
    print(f"  总文件数: {len(files):,}")
    print(f"  总样本数: {total_samples:,}")
    print(f"  ─────────────────────────────────")
    print(f"  训练集: {len(train_files):,} 文件 / {train_samples:,} 样本 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_files):,} 文件 / {val_samples:,} 样本 ({(1-train_ratio)*100:.0f}%)")
    print(f"  ─────────────────────────────────")
    print(f"  操作模式: {'复制' if copy_mode == 'copy' else '移动'}")
    
    # 确认
    confirm = input("\n确认执行? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    # 复制/移动文件
    print(f"\n[3/3] {'复制' if copy_mode == 'copy' else '移动'}文件...")
    
    op_func = shutil.copy2 if copy_mode == 'copy' else shutil.move
    
    for i, (fp, _) in enumerate(train_files):
        # 保留原始文件名，加上序号前缀避免重名
        dst = train_dir / f"{i:05d}_{fp.name}"
        op_func(str(fp), str(dst))
        if (i + 1) % 500 == 0:
            print(f"  训练集进度: {i+1}/{len(train_files)}")
    
    for i, (fp, _) in enumerate(val_files):
        dst = val_dir / f"{i:05d}_{fp.name}"
        op_func(str(fp), str(dst))
        if (i + 1) % 500 == 0:
            print(f"  验证集进度: {i+1}/{len(val_files)}")
    
    print(f"\n【完成】")
    print(f"  train/unlabeled/: {len(train_files):,} 文件")
    print(f"  val/unlabeled/: {len(val_files):,} 文件")
    
    # 生成报告
    report_path = target / "data_organization_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"数据整理报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"源目录: {source}\n")
        f.write(f"目标目录: {target}\n\n")
        f.write(f"训练集:\n")
        f.write(f"  文件数: {len(train_files):,}\n")
        f.write(f"  样本数: {train_samples:,}\n")
        f.write(f"  位置: train/unlabeled/\n\n")
        f.write(f"验证集:\n")
        f.write(f"  文件数: {len(val_files):,}\n")
        f.write(f"  样本数: {val_samples:,}\n")
        f.write(f"  位置: val/unlabeled/\n")
    
    print(f"  报告保存至: {report_path}")


def check_existing_labeled_data(test_dir: str):
    """
    检查现有有标签数据的分布
    
    【参数】
        test_dir: test目录路径
    """
    test = Path(test_dir)
    
    if not test.exists():
        print(f"[错误] 目录不存在: {test}")
        return
    
    # 类别关键词
    class_keywords = {
        "正常": ("正常", "normal", "健康", "healthy"),
        "故障": ("故障", "异常", "fault", "abnormal", "error"),
    }
    
    # 统计
    stats = {cls: [] for cls in class_keywords}
    unknown = []
    
    for subdir in test.iterdir():
        if not subdir.is_dir():
            continue
        
        name = subdir.name.lower()
        files = list(subdir.rglob("*.jsonl")) + list(subdir.rglob("*.json"))
        samples = sum(count_samples_in_file(f) for f in files)
        
        assigned = False
        for cls, keywords in class_keywords.items():
            if any(kw.lower() in name for kw in keywords):
                stats[cls].append({
                    'name': subdir.name,
                    'files': len(files),
                    'samples': samples
                })
                assigned = True
                break
        
        if not assigned:
            unknown.append({
                'name': subdir.name,
                'files': len(files),
                'samples': samples
            })
    
    # 打印报告
    print(f"\n{'='*60}")
    print(f"有标签数据统计报告")
    print(f"{'='*60}")
    print(f"目录: {test}\n")
    
    total_files = 0
    total_samples = 0
    
    for cls, items in stats.items():
        cls_files = sum(d['files'] for d in items)
        cls_samples = sum(d['samples'] for d in items)
        total_files += cls_files
        total_samples += cls_samples
        
        print(f"【{cls}】 {len(items)} 个子目录")
        print(f"    文件数: {cls_files:,}")
        print(f"    样本数: {cls_samples:,}")
        print(f"    ─────────────────────────────────")
        
        # 按样本数排序显示前5个
        sorted_items = sorted(items, key=lambda x: x['samples'], reverse=True)
        for item in sorted_items[:5]:
            print(f"    {item['name']}")
            print(f"      └─ {item['files']} 文件, {item['samples']:,} 样本")
        
        if len(items) > 5:
            remaining = len(items) - 5
            remaining_samples = sum(d['samples'] for d in sorted_items[5:])
            print(f"    ... 还有 {remaining} 个目录 ({remaining_samples:,} 样本)")
        print()
    
    if unknown:
        unk_files = sum(d['files'] for d in unknown)
        unk_samples = sum(d['samples'] for d in unknown)
        total_files += unk_files
        total_samples += unk_samples
        
        print(f"【未识别】 {len(unknown)} 个子目录")
        print(f"    文件数: {unk_files:,}")
        print(f"    样本数: {unk_samples:,}")
        print(f"    ─────────────────────────────────")
        for item in unknown[:3]:
            print(f"    {item['name']}")
            print(f"      └─ {item['files']} 文件, {item['samples']:,} 样本")
        print()
    
    print(f"{'='*60}")
    print(f"总计: {total_files:,} 文件, {total_samples:,} 样本")
    print(f"{'='*60}")
    
    # 检查类别平衡
    normal_samples = sum(d['samples'] for d in stats.get('正常', []))
    fault_samples = sum(d['samples'] for d in stats.get('故障', []))
    
    if normal_samples > 0 and fault_samples > 0:
        ratio = normal_samples / fault_samples
        print(f"\n类别比例: 正常:故障 = {ratio:.2f}:1")
        if ratio > 3 or ratio < 0.33:
            print("  ⚠️ 类别不平衡较严重，训练时会自动进行平衡采样")
    
    return stats


def check_directory_structure(root_dir: str):
    """
    检查并显示目录结构
    """
    root = Path(root_dir)
    
    if not root.exists():
        print(f"[错误] 目录不存在: {root}")
        return
    
    print(f"\n{'='*60}")
    print(f"目录结构检查")
    print(f"{'='*60}")
    print(f"根目录: {root}\n")
    
    expected = ['train', 'val', 'test']
    
    for subdir in expected:
        path = root / subdir
        if path.exists():
            files = list(path.rglob("*.jsonl")) + list(path.rglob("*.json"))
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            print(f"✅ {subdir}/")
            print(f"   └─ {len(subdirs)} 个子目录, {len(files)} 个数据文件")
        else:
            print(f"❌ {subdir}/ (不存在)")
    
    print()


def print_usage_guide():
    """打印使用指南"""
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         数据组织最佳实践指南                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  【你的数据情况】                                                              ║
║    ✅ 有标签数据：几千条（目录名带"正常/故障"）→ 代码自动识别标签                 ║
║    ❌ 无标签数据：几万条（新采集的）→ 用于无监督学习                             ║
║                                                                              ║
║  【三阶段训练流程】                                                            ║
║                                                                              ║
║    ┌──────────────────────────────────────────────────────────────────┐      ║
║    │  阶段一 (无监督)                                                  │      ║
║    │    输入: train/ + val/ 目录下的所有数据                           │      ║
║    │    方法: Deep SVDD + VAE                                         │      ║
║    │    输出: 每个样本的异常得分                                       │      ║
║    └──────────────────────────────────────────────────────────────────┘      ║
║                                    ↓                                         ║
║    ┌──────────────────────────────────────────────────────────────────┐      ║
║    │  阶段二 (伪标签生成)                                              │      ║
║    │    输入: 阶段一的异常得分                                         │      ║
║    │    方法: 分位数阈值筛选                                           │      ║
║    │    输出: 高置信正常/高置信异常/不确定                               │      ║
║    └──────────────────────────────────────────────────────────────────┘      ║
║                                    ↓                                         ║
║    ┌──────────────────────────────────────────────────────────────────┐      ║
║    │  阶段三 (有监督)                                                  │      ║
║    │    输入: test/ 目录（有真实标签）+ 伪标签数据                       │      ║
║    │    方法: 迁移学习分类器                                           │      ║
║    │    输出: 最终故障诊断模型                                          │      ║
║    └──────────────────────────────────────────────────────────────────┘      ║
║                                                                              ║
║  【推荐数据放置方式】                                                          ║
║                                                                              ║
║    20251016/                                                                 ║
║    ├── train/                     ← 无标签数据 (70%)                         ║
║    │   └── unlabeled/                                                        ║
║    │       ├── 00000_data.jsonl                                              ║
║    │       └── ...                约 21,000 个文件                            ║
║    │                                                                         ║
║    ├── val/                       ← 无标签数据 (30%)                         ║
║    │   └── unlabeled/                                                        ║
║    │       └── ...                约 9,000 个文件                             ║
║    │                                                                         ║
║    └── test/                      ← 有标签数据 (现有的几千条)                  ║
║        ├── 114--故障--交流变压器/                                             ║
║        │   └── *.jsonl                                                       ║
║        ├── 120--正常--交流变压器/                                             ║
║        │   └── *.jsonl                                                       ║
║        └── ...                                                               ║
║                                                                              ║
║  【操作步骤】                                                                  ║
║                                                                              ║
║    步骤1: 保持现有有标签数据在 test/ 目录（不用动）                            ║
║                                                                              ║
║    步骤2: 整理无标签数据                                                       ║
║           python data_organizer.py --source "D:\\无标签数据" \\               ║
║                                    --target "E:\\...\\20251016"              ║
║                                                                              ║
║    步骤3: 检查数据分布                                                         ║
║           python data_organizer.py --check "E:\\...\\20251016\\test"         ║
║                                                                              ║
║    步骤4: 运行三阶段训练                                                       ║
║           python transformer_three_stage_diagnosis.py --all                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


def main():
    parser = argparse.ArgumentParser(
        description='数据整理工具 - 变压器故障诊断项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看使用指南
  python data_organizer.py --guide
  
  # 检查现有有标签数据
  python data_organizer.py --check "E:\\...\\20251016\\test"
  
  # 检查目录结构
  python data_organizer.py --structure "E:\\...\\20251016"
  
  # 整理无标签数据到 train/val
  python data_organizer.py --source "D:\\新数据" --target "E:\\...\\20251016"
  
  # 移动而非复制（节省空间）
  python data_organizer.py --source "D:\\新数据" --target "E:\\...\\20251016" --move
        """
    )
    
    parser.add_argument('--source', type=str, help='无标签数据源目录')
    parser.add_argument('--target', type=str, help='目标根目录（含train/val/test）')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例 (默认0.7)')
    parser.add_argument('--check', type=str, help='检查有标签数据目录')
    parser.add_argument('--structure', type=str, help='检查目录结构')
    parser.add_argument('--guide', action='store_true', help='显示使用指南')
    parser.add_argument('--move', action='store_true', help='移动文件而非复制')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    if args.guide:
        print_usage_guide()
    elif args.structure:
        check_directory_structure(args.structure)
    elif args.check:
        check_existing_labeled_data(args.check)
    elif args.source and args.target:
        copy_mode = 'move' if args.move else 'copy'
        organize_unlabeled_data(
            args.source, args.target, 
            args.train_ratio, args.seed, 
            copy_mode
        )
    else:
        print_usage_guide()


if __name__ == "__main__":
    main()

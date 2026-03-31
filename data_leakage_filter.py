# -*- coding: utf-8 -*-
"""
data_leakage_filter.py - 数据泄露筛选脚本 v2.0
=================================================

核心逻辑:
- 只用信号前N个值生成指纹（不依赖datetime、传感器名）
- 信号值完全一致 = 同一条数据 = 应从训练集排除

功能:
1. 从标注数据提取信号指纹
2. 在原始CSV中匹配，找到泄露的样本
3. 将标注数据划分为 val/test 并保存到目录
4. 生成排除ID列表，用于训练时过滤

使用方法:
    python data_leakage_filter.py \
        --labeled_dir "E:/CODE/DATA/20251016" \
        --raw_dir "E:/CODE/DATA/trans_data/00 振动原始数据" \
        --output "./filtered_output" \
        --val_ratio 0.5 \
        --signal_check_points 20

Author: PhD Candidate
Version: 2.0
"""

import os
import sys
import json
import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
from tqdm import tqdm


class SignatureExtractor:
    """
    信号指纹提取器
    
    指纹 = hash(信号前N个值)  -- 只用信号值，最准确
    """
    
    def __init__(self, check_points: int = 20):
        """
        参数:
            check_points: 用于生成指纹的信号前N个点（越多越准确，推荐20）
        """
        self.check_points = check_points
    
    def _compute_fingerprint(self, signal_head: List[float]) -> str:
        """
        计算信号指纹
        
        只用信号前N个值，四舍五入到4位小数（避免浮点精度问题）
        """
        # 四舍五入到4位小数，转字符串
        signal_str = ','.join([f'{v:.4f}' for v in signal_head])
        return hashlib.md5(signal_str.encode()).hexdigest()
    
    def extract_from_jsonl(self, jsonl_path: Path) -> List[Dict]:
        """
        从jsonl文件提取指纹
        
        返回:
            [{'fingerprint': str, 'signal_head': list, 'label': int, 
              'file': Path, 'line': int}, ...]
        """
        signatures = []
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # 获取信号值
                    signal_raw = data.get('signal_value', data.get('signal', None))
                    if signal_raw is None:
                        continue
                    
                    if isinstance(signal_raw, str):
                        try:
                            signal = [float(x) for x in signal_raw.split(',')[:self.check_points]]
                        except:
                            continue
                    elif isinstance(signal_raw, list):
                        signal = [float(x) for x in signal_raw[:self.check_points]]
                    else:
                        continue
                    
                    if len(signal) < self.check_points:
                        continue
                    
                    # 推断标签
                    parent_name = jsonl_path.parent.name.lower()
                    if any(kw in parent_name for kw in ['正常', 'normal', 'good', '健康']):
                        label = 0
                    elif any(kw in parent_name for kw in ['故障', '异常', 'fault', 'abnormal']):
                        label = 1
                    else:
                        label = -1  # 未知
                    
                    # 生成指纹（只用信号值）
                    fingerprint = self._compute_fingerprint(signal)
                    
                    signatures.append({
                        'fingerprint': fingerprint,
                        'signal_head': signal,
                        'label': label,
                        'file': jsonl_path,
                        'line': line_num,
                    })
        
        except Exception as e:
            print(f"  [警告] 读取失败 {jsonl_path.name}: {e}")
        
        return signatures
    
    def extract_from_csv(self, csv_path: Path, signal_len: int = 8192) -> List[Dict]:
        """
        从CSV文件提取指纹
        
        返回:
            [{'fingerprint': str, 'signal_head': list, 
              'file': Path, 'channel': str, 'start_idx': int}, ...]
        """
        signatures = []
        
        try:
            import pandas as pd
            
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
            df = None
            
            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    break
                except:
                    continue
            
            if df is None or df.empty:
                return signatures
            
            # 查找时间列（排除它）
            time_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in ['time', '时间', 'date', 'unnamed']):
                    time_col = col
                    break
            
            # 获取数据列（排除时间列）
            data_cols = [c for c in df.columns if c != time_col and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
            
            if not data_cols:
                return signatures
            
            # 对每个通道，滑动窗口提取指纹
            for col in data_cols:
                values = df[col].dropna().values
                
                if len(values) < signal_len:
                    continue
                
                # 滑动窗口切分
                for start in range(0, len(values) - signal_len + 1, signal_len):
                    # 获取信号前N个点
                    signal_head = values[start:start + self.check_points].tolist()
                    
                    if len(signal_head) < self.check_points:
                        continue
                    
                    # 生成指纹（只用信号值）
                    fingerprint = self._compute_fingerprint(signal_head)
                    
                    signatures.append({
                        'fingerprint': fingerprint,
                        'signal_head': signal_head,
                        'file': csv_path,
                        'channel': str(col),
                        'start_idx': start,
                    })
        
        except Exception as e:
            print(f"  [警告] CSV读取失败 {csv_path.name}: {e}")
        
        return signatures


class LeakageDetector:
    """数据泄露检测器"""
    
    def __init__(self, extractor: SignatureExtractor):
        self.extractor = extractor
        self.labeled_sigs: Dict[str, List[Dict]] = defaultdict(list)  # fingerprint -> [sig_info, ...]
        self.raw_sigs: Dict[str, List[Dict]] = defaultdict(list)      # fingerprint -> [sig_info, ...]
    
    def load_labeled_data(self, labeled_dir: Path) -> int:
        """加载标注数据"""
        print("\n[步骤1/5] 加载已标注数据...")
        
        jsonl_files = list(labeled_dir.rglob("*.jsonl"))
        print(f"  扫描到 {len(jsonl_files)} 个标注文件")
        
        total_sigs = 0
        for f in tqdm(jsonl_files, desc="提取标注指纹"):
            sigs = self.extractor.extract_from_jsonl(f)
            for sig in sigs:
                self.labeled_sigs[sig['fingerprint']].append(sig)
            total_sigs += len(sigs)
        
        unique_fps = len(self.labeled_sigs)
        print(f"  提取指纹: {total_sigs} 个")
        print(f"  唯一指纹: {unique_fps} 个")
        print(f"  来自文件: {len(jsonl_files)} 个")
        
        return total_sigs
    
    def load_raw_data(self, raw_dir: Path) -> int:
        """加载原始CSV数据"""
        print("\n[步骤2/5] 加载原始CSV数据...")
        
        csv_files = list(raw_dir.rglob("*.csv"))
        print(f"  扫描到 {len(csv_files)} 个CSV文件")
        
        total_sigs = 0
        for f in tqdm(csv_files, desc="提取CSV指纹"):
            sigs = self.extractor.extract_from_csv(f)
            for sig in sigs:
                self.raw_sigs[sig['fingerprint']].append(sig)
            total_sigs += len(sigs)
        
        unique_fps = len(self.raw_sigs)
        print(f"  提取指纹: {total_sigs} 个")
        print(f"  唯一指纹: {unique_fps} 个")
        print(f"  来自文件: {len(csv_files)} 个")
        
        return total_sigs
    
    def detect_leakage(self) -> Tuple[Set[Path], Set[str], Dict]:
        """
        检测数据泄露
        
        返回:
            (匹配到的标注文件集合, 排除ID集合, 统计信息)
        """
        print("\n[步骤3/5] 检测数据泄露...")
        
        # 找交集
        labeled_fps = set(self.labeled_sigs.keys())
        raw_fps = set(self.raw_sigs.keys())
        matched_fps = labeled_fps & raw_fps
        
        print(f"  标注数据指纹: {len(labeled_fps)}")
        print(f"  原始CSV指纹:  {len(raw_fps)}")
        print(f"  匹配指纹数:   {len(matched_fps)}")
        
        # 收集匹配到的标注文件
        matched_labeled_files = set()
        matched_labeled_count = 0
        
        for fp in matched_fps:
            for sig in self.labeled_sigs[fp]:
                matched_labeled_files.add(sig['file'])
                matched_labeled_count += 1
        
        # 生成排除ID（用于从训练集排除）
        excluded_ids = set()
        for fp in matched_fps:
            for sig in self.raw_sigs[fp]:
                # ID格式: filename_ch{channel}_s{start}
                csv_name = sig['file'].stem
                channel = sig['channel']
                start = sig['start_idx']
                excluded_id = f"{csv_name}_ch{channel}_s{start}"
                excluded_ids.add(excluded_id)
        
        stats = {
            'labeled_fingerprints': len(labeled_fps),
            'raw_fingerprints': len(raw_fps),
            'matched_fingerprints': len(matched_fps),
            'matched_labeled_samples': matched_labeled_count,
            'matched_labeled_files': len(matched_labeled_files),
            'excluded_ids': len(excluded_ids),
        }
        
        print(f"\n  【泄露检测结果】")
        print(f"  匹配到的标注样本: {matched_labeled_count}")
        print(f"  涉及标注文件: {len(matched_labeled_files)}")
        print(f"  生成排除ID: {len(excluded_ids)}")
        
        if len(labeled_fps) > 0:
            match_rate = 100 * len(matched_fps) / len(labeled_fps)
            print(f"  匹配率: {match_rate:.1f}%")
        
        return matched_labeled_files, excluded_ids, stats


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, labeled_dir: Path, output_dir: Path, val_ratio: float = 0.5):
        self.labeled_dir = labeled_dir
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        
        # 创建输出目录
        self.val_dir = output_dir / "val"
        self.test_dir = output_dir / "test"
        
        for d in [self.val_dir, self.test_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def split_and_save(self) -> Dict:
        """
        划分标注数据为val/test并保存
        
        返回:
            划分统计
        """
        print("\n[步骤4/5] 划分标注数据...")
        
        # 收集所有标注文件
        all_files = list(self.labeled_dir.rglob("*.jsonl"))
        
        # 按类别分组
        normal_files = []
        fault_files = []
        
        for f in all_files:
            parent_name = f.parent.name.lower()
            if any(kw in parent_name for kw in ['正常', 'normal', 'good', '健康']):
                normal_files.append(f)
            elif any(kw in parent_name for kw in ['故障', '异常', 'fault', 'abnormal']):
                fault_files.append(f)
        
        print(f"  总文件数: {len(all_files)}")
        print(f"  正常样本: {len(normal_files)}")
        print(f"  故障样本: {len(fault_files)}")
        
        # 随机打乱
        np.random.seed(42)
        np.random.shuffle(normal_files)
        np.random.shuffle(fault_files)
        
        # 分层划分
        n_val_normal = int(len(normal_files) * self.val_ratio)
        n_val_fault = int(len(fault_files) * self.val_ratio)
        
        val_normal = normal_files[:n_val_normal]
        val_fault = fault_files[:n_val_fault]
        test_normal = normal_files[n_val_normal:]
        test_fault = fault_files[n_val_fault:]
        
        val_files = val_normal + val_fault
        test_files = test_normal + test_fault
        
        print(f"\n  划分结果:")
        print(f"  验证集: {len(val_files)} ({len(val_normal)}正常 + {len(val_fault)}故障)")
        print(f"  测试集: {len(test_files)} ({len(test_normal)}正常 + {len(test_fault)}故障)")
        
        # 复制文件
        print("\n[步骤5/5] 保存划分结果...")
        
        val_stats = self._copy_files_with_structure(val_files, self.val_dir, "验证集")
        test_stats = self._copy_files_with_structure(test_files, self.test_dir, "测试集")
        
        # 保存统计信息
        split_info = {
            'total_files': len(all_files),
            'normal_files': len(normal_files),
            'fault_files': len(fault_files),
            'val_count': len(val_files),
            'val_normal': len(val_normal),
            'val_fault': len(val_fault),
            'test_count': len(test_files),
            'test_normal': len(test_normal),
            'test_fault': len(test_fault),
            'val_ratio': self.val_ratio,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.output_dir / "split_info.json", 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        # 保存文件列表
        self._save_file_list(val_files, self.output_dir / "val_files.txt")
        self._save_file_list(test_files, self.output_dir / "test_files.txt")
        
        return split_info
    
    def _copy_files_with_structure(self, files: List[Path], dest_dir: Path, desc: str) -> Dict:
        """复制文件到目标目录，保持类别子目录结构"""
        stats = {'正常': 0, '故障': 0}
        
        for f in tqdm(files, desc=f"复制{desc}"):
            # 确定类别
            parent_name = f.parent.name.lower()
            if any(kw in parent_name for kw in ['正常', 'normal', 'good', '健康']):
                category = "正常"
            else:
                category = "故障"
            
            # 创建类别子目录
            category_dir = dest_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # 复制文件（保留原文件名）
            dest_file = category_dir / f.name
            
            # 如果文件名冲突，加后缀
            if dest_file.exists():
                stem = f.stem
                suffix = f.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = category_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(f, dest_file)
            stats[category] += 1
        
        return stats
    
    def _save_file_list(self, files: List[Path], output_path: Path):
        """保存文件列表"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(str(file.absolute()) + '\n')


def save_excluded_ids(excluded_ids: Set[str], output_dir: Path):
    """保存排除ID列表"""
    output_path = output_dir / "excluded_sample_ids.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid in sorted(excluded_ids):
            f.write(eid + '\n')
    
    print(f"\n  排除ID已保存: {output_path}")


def save_leakage_stats(stats: Dict, output_dir: Path):
    """保存泄露检测统计"""
    output_path = output_dir / "leakage_stats.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="数据泄露筛选脚本 v2.0 - 只用信号值匹配，最准确",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程（泄露检测 + 划分）
  python data_leakage_filter_v2.py \\
      --labeled_dir "E:/CODE/DATA/20251016" \\
      --raw_dir "E:/CODE/DATA/trans_data/00 振动原始数据" \\
      --output "./filtered_output"

  # 仅划分（跳过泄露检测）
  python data_leakage_filter_v2.py \\
      --labeled_dir "E:/CODE/DATA/20251016" \\
      --output "./filtered_output" \\
      --skip_leakage_check
        """
    )
    
    parser.add_argument('--labeled_dir', type=str, required=True,
                        help='已标注数据目录 (包含jsonl文件)')
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='原始CSV数据目录 (用于泄露检测)')
    parser.add_argument('--output', type=str, default='./filtered_output',
                        help='输出目录 (默认: ./filtered_output)')
    parser.add_argument('--val_ratio', type=float, default=0.5,
                        help='验证集比例 (默认: 0.5)')
    parser.add_argument('--signal_check_points', type=int, default=20,
                        help='用于匹配的信号前N个点 (默认: 20，越多越准确)')
    parser.add_argument('--skip_leakage_check', action='store_true',
                        help='跳过泄露检测，仅划分数据')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("数据泄露筛选脚本 v2.0")
    print("="*60)
    print(f"已标注数据目录: {args.labeled_dir}")
    print(f"原始数据目录:   {args.raw_dir or '(跳过泄露检测)'}")
    print(f"输出目录:       {args.output}")
    print(f"验证集比例:     {args.val_ratio}")
    print(f"信号匹配点数:   {args.signal_check_points}")
    print("="*60)
    print("匹配逻辑: 只用信号前N个值，不依赖datetime/传感器名")
    print("="*60)
    
    labeled_dir = Path(args.labeled_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建提取器
    extractor = SignatureExtractor(check_points=args.signal_check_points)
    
    # 泄露检测
    excluded_ids = set()
    leakage_stats = {}
    
    if args.raw_dir and not args.skip_leakage_check:
        raw_dir = Path(args.raw_dir)
        if raw_dir.exists():
            detector = LeakageDetector(extractor)
            detector.load_labeled_data(labeled_dir)
            detector.load_raw_data(raw_dir)
            _, excluded_ids, leakage_stats = detector.detect_leakage()
            
            # 保存结果
            save_excluded_ids(excluded_ids, output_dir)
            save_leakage_stats(leakage_stats, output_dir)
        else:
            print(f"\n[警告] 原始数据目录不存在: {raw_dir}")
            print("  将跳过泄露检测，仅进行数据划分")
    else:
        if not args.skip_leakage_check and not args.raw_dir:
            print("\n[提示] 未指定--raw_dir，跳过泄露检测")
    
    # 数据划分
    splitter = DatasetSplitter(labeled_dir, output_dir, args.val_ratio)
    split_info = splitter.split_and_save()
    
    # 打印摘要
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    print(f"\n输出目录: {output_dir.absolute()}")
    print(f"  │")
    print(f"  ├── val/                      验证集")
    print(f"  │   ├── 正常/                 {split_info.get('val_normal', 0)} 文件")
    print(f"  │   └── 故障/                 {split_info.get('val_fault', 0)} 文件")
    print(f"  │")
    print(f"  ├── test/                     测试集")
    print(f"  │   ├── 正常/                 {split_info.get('test_normal', 0)} 文件")
    print(f"  │   └── 故障/                 {split_info.get('test_fault', 0)} 文件")
    print(f"  │")
    print(f"  ├── excluded_sample_ids.txt   排除ID ({len(excluded_ids)} 个)")
    print(f"  ├── split_info.json           划分统计")
    print(f"  ├── leakage_stats.json        泄露检测统计")
    print(f"  ├── val_files.txt             验证集原始路径")
    print(f"  └── test_files.txt            测试集原始路径")
    
    print("\n" + "="*60)
    print("【下一步】使用v5_12_modular训练:")
    print("="*60)
    print(f"python -m v5_12_modular.main --all \\")
    print(f"    --data_root \"你的原始CSV数据目录\" \\")
    print(f"    --filter_output \"{output_dir.absolute()}\" \\")
    print(f"    --batch_size 24 --lr 2e-4")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

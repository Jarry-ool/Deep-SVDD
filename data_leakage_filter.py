# -*- coding: utf-8 -*-
"""
data_leakage_filter.py
======================

数据泄露筛选脚本

功能：
1. 溯源已标注的jsonl/json数据，找到对应的原始CSV样本
2. 生成排除列表，确保训练数据不包含已标注样本
3. 将已标注数据划分为 val/test 集

使用方法：
    python data_leakage_filter.py --labeled_dir ./labeled_data --raw_dir ./raw_data --output ./filtered_output

Author: PhD Candidate (Electrical Eng.)
"""

import os
import json
import hashlib
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import random


# =============================================================================
# 第1步: 数据指纹生成
# =============================================================================

def compute_signal_fingerprint(signal: np.ndarray, precision: int = 4) -> str:
    """
    计算信号的指纹（用于匹配）
    
    使用信号的统计特征生成指纹，而非全量数据哈希
    这样可以在信号略有截断时仍然匹配
    
    参数:
        signal: 信号数组
        precision: 小数精度
    返回:
        指纹字符串
    """
    if len(signal) == 0:
        return "empty"
    
    # 计算关键统计量
    stats = {
        'len': len(signal),
        'mean': round(float(np.mean(signal)), precision),
        'std': round(float(np.std(signal)), precision),
        'min': round(float(np.min(signal)), precision),
        'max': round(float(np.max(signal)), precision),
        'first5': [round(float(x), precision) for x in signal[:5]],
        'last5': [round(float(x), precision) for x in signal[-5:]],
    }
    
    # 生成哈希
    stats_str = json.dumps(stats, sort_keys=True)
    return hashlib.md5(stats_str.encode()).hexdigest()


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    从文件名提取元数据
    
    文件名格式示例：
    - 广蓄A厂__3主变_20231125002106.csv
    - 梅州局新铺站__1主变_20240111000651.csv
    
    返回:
        {
            'station': 站点名,
            'transformer': 变压器编号,
            'timestamp': 时间戳,
        }
    """
    # 去掉扩展名
    stem = Path(filename).stem
    
    # 尝试多种匹配模式
    metadata = {
        'station': '',
        'transformer': '',
        'timestamp': '',
        'original': stem,
    }
    
    # 模式1: 站点__变压器_时间戳
    pattern1 = r'^(.+?)__(\d+主变)_(\d+)$'
    match = re.match(pattern1, stem)
    if match:
        metadata['station'] = match.group(1)
        metadata['transformer'] = match.group(2)
        metadata['timestamp'] = match.group(3)
        return metadata
    
    # 模式2: 站点_变压器_时间戳 (单下划线)
    pattern2 = r'^(.+?)_(\d+主变)_(\d+)$'
    match = re.match(pattern2, stem)
    if match:
        metadata['station'] = match.group(1)
        metadata['transformer'] = match.group(2)
        metadata['timestamp'] = match.group(3)
        return metadata
    
    # 模式3: 任意格式，提取时间戳
    pattern3 = r'(\d{14}|\d{12}|\d{8})'
    match = re.search(pattern3, stem)
    if match:
        metadata['timestamp'] = match.group(1)
    
    return metadata


# =============================================================================
# 第2步: 标注数据加载器
# =============================================================================

class LabeledDataLoader:
    """已标注数据加载器"""
    
    def __init__(self, labeled_dir: Path):
        self.labeled_dir = Path(labeled_dir)
        self.samples = []
        self.fingerprints = {}  # {fingerprint: sample_info}
    
    def load(self) -> int:
        """
        加载所有标注数据
        
        返回:
            加载的样本数量
        """
        if not self.labeled_dir.exists():
            print(f"[警告] 标注数据目录不存在: {self.labeled_dir}")
            return 0
        
        # 扫描jsonl和json文件
        files = list(self.labeled_dir.rglob("*.jsonl")) + list(self.labeled_dir.rglob("*.json"))
        
        print(f"[LabeledDataLoader] 扫描到 {len(files)} 个标注文件")
        
        for file_path in tqdm(files, desc="加载标注数据"):
            self._load_file(file_path)
        
        print(f"[LabeledDataLoader] 共加载 {len(self.samples)} 个标注样本")
        print(f"[LabeledDataLoader] 生成 {len(self.fingerprints)} 个唯一指纹")
        
        return len(self.samples)
    
    def _load_file(self, file_path: Path):
        """加载单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # jsonl格式：每行一个json
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        self._process_sample(data, file_path, line_num)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[警告] 无法读取文件 {file_path}: {e}")
    
    def _process_sample(self, data: Dict, file_path: Path, line_num: int):
        """处理单个样本"""
        # 提取信号数据
        raw_signal = data.get('signal_value', data.get('signal', None))
        if raw_signal is None:
            return
        
        # 转换为numpy数组
        if isinstance(raw_signal, str):
            try:
                signal = np.array([float(x) for x in raw_signal.split(',')])
            except:
                return
        else:
            signal = np.array(raw_signal)
        
        # 计算指纹
        fingerprint = compute_signal_fingerprint(signal)
        
        # 提取元数据
        source_file = data.get('source_file', data.get('file', ''))
        metadata = extract_metadata_from_filename(source_file)
        
        # 推断标签
        label = -1
        parent_name = file_path.parent.name.lower()
        if '正常' in parent_name or 'normal' in parent_name:
            label = 0
        elif '故障' in parent_name or 'fault' in parent_name or '异常' in parent_name:
            label = 1
        
        # 也可以从数据中获取标签
        if 'label' in data:
            label = int(data['label'])
        
        sample_info = {
            'file_path': str(file_path),
            'line_num': line_num,
            'fingerprint': fingerprint,
            'source_file': source_file,
            'metadata': metadata,
            'label': label,
            'signal_length': len(signal),
        }
        
        self.samples.append(sample_info)
        
        # 记录指纹映射
        if fingerprint not in self.fingerprints:
            self.fingerprints[fingerprint] = []
        self.fingerprints[fingerprint].append(sample_info)
    
    def get_fingerprints(self) -> Set[str]:
        """获取所有指纹"""
        return set(self.fingerprints.keys())
    
    def get_labeled_samples(self) -> List[Dict]:
        """获取有标签的样本"""
        return [s for s in self.samples if s['label'] != -1]


# =============================================================================
# 第3步: 原始数据扫描器
# =============================================================================

class RawDataScanner:
    """原始CSV数据扫描器"""
    
    def __init__(self, raw_dir: Path, signal_len: int = 8192):
        self.raw_dir = Path(raw_dir)
        self.signal_len = signal_len
        self.csv_files = []
        self.sample_fingerprints = {}  # {fingerprint: [(csv_path, channel, start_idx), ...]}
    
    def scan(self) -> int:
        """
        扫描原始数据目录
        
        返回:
            扫描的CSV文件数量
        """
        if not self.raw_dir.exists():
            print(f"[警告] 原始数据目录不存在: {self.raw_dir}")
            return 0
        
        self.csv_files = list(self.raw_dir.rglob("*.csv"))
        print(f"[RawDataScanner] 扫描到 {len(self.csv_files)} 个CSV文件")
        
        return len(self.csv_files)
    
    def build_fingerprint_index(self, sample_rate: float = 1.0) -> int:
        """
        构建指纹索引（用于快速匹配）
        
        参数:
            sample_rate: 采样率（0-1），用于加速大数据集处理
        返回:
            索引的样本数量
        """
        total_samples = 0
        
        for csv_path in tqdm(self.csv_files, desc="构建指纹索引"):
            try:
                # 读取CSV
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='gbk')
                
                data = df.values.astype(np.float32)
                n_samples, n_channels = data.shape
                
                # 对每个通道切分样本
                for ch_idx in range(n_channels):
                    channel_data = data[:, ch_idx]
                    
                    # 滑动窗口
                    for start_idx in range(0, len(channel_data) - self.signal_len + 1, 
                                          self.signal_len // 2):
                        # 采样
                        if random.random() > sample_rate:
                            continue
                        
                        signal = channel_data[start_idx:start_idx + self.signal_len]
                        fingerprint = compute_signal_fingerprint(signal)
                        
                        if fingerprint not in self.sample_fingerprints:
                            self.sample_fingerprints[fingerprint] = []
                        self.sample_fingerprints[fingerprint].append({
                            'csv_path': str(csv_path),
                            'channel_idx': ch_idx,
                            'start_idx': start_idx,
                        })
                        total_samples += 1
            
            except Exception as e:
                print(f"[警告] 无法处理CSV: {csv_path}, 错误: {e}")
                continue
        
        print(f"[RawDataScanner] 索引了 {total_samples} 个样本片段")
        print(f"[RawDataScanner] 生成 {len(self.sample_fingerprints)} 个唯一指纹")
        
        return total_samples
    
    def match_fingerprint(self, fingerprint: str) -> List[Dict]:
        """
        匹配指纹
        
        返回:
            匹配到的样本位置列表
        """
        return self.sample_fingerprints.get(fingerprint, [])


# =============================================================================
# 第4步: 数据泄露检测器
# =============================================================================

class LeakageDetector:
    """数据泄露检测器"""
    
    def __init__(self, labeled_loader: LabeledDataLoader, raw_scanner: RawDataScanner):
        self.labeled_loader = labeled_loader
        self.raw_scanner = raw_scanner
        self.matches = []  # 匹配到的样本
        self.unmatched = []  # 未匹配的样本
    
    def detect(self) -> Dict[str, int]:
        """
        执行泄露检测
        
        返回:
            检测统计
        """
        labeled_fingerprints = self.labeled_loader.get_fingerprints()
        
        print(f"\n[LeakageDetector] 开始匹配 {len(labeled_fingerprints)} 个标注样本指纹...")
        
        matched_count = 0
        unmatched_count = 0
        
        for fingerprint in tqdm(labeled_fingerprints, desc="匹配指纹"):
            matches = self.raw_scanner.match_fingerprint(fingerprint)
            
            if matches:
                matched_count += 1
                for labeled_sample in self.labeled_loader.fingerprints[fingerprint]:
                    self.matches.append({
                        'labeled': labeled_sample,
                        'raw_matches': matches,
                    })
            else:
                unmatched_count += 1
                for labeled_sample in self.labeled_loader.fingerprints[fingerprint]:
                    self.unmatched.append(labeled_sample)
        
        stats = {
            'total_labeled_fingerprints': len(labeled_fingerprints),
            'matched': matched_count,
            'unmatched': unmatched_count,
            'match_rate': matched_count / len(labeled_fingerprints) if labeled_fingerprints else 0,
        }
        
        print(f"\n[LeakageDetector] 检测完成:")
        print(f"  总指纹数: {stats['total_labeled_fingerprints']}")
        print(f"  匹配成功: {stats['matched']} ({stats['match_rate']:.1%})")
        print(f"  未匹配: {stats['unmatched']}")
        
        return stats
    
    def get_excluded_locations(self) -> Set[Tuple[str, int, int]]:
        """
        获取需要排除的原始数据位置
        
        返回:
            {(csv_path, channel_idx, start_idx), ...}
        """
        excluded = set()
        for match in self.matches:
            for raw_match in match['raw_matches']:
                excluded.add((
                    raw_match['csv_path'],
                    raw_match['channel_idx'],
                    raw_match['start_idx'],
                ))
        return excluded
    
    def generate_excluded_ids(self) -> Set[str]:
        """
        生成排除样本ID集合
        
        返回:
            样本ID集合
        """
        excluded_ids = set()
        for match in self.matches:
            for raw_match in match['raw_matches']:
                csv_path = Path(raw_match['csv_path'])
                sample_id = f"{csv_path.stem}_ch{raw_match['channel_idx']}_s{raw_match['start_idx']}"
                excluded_ids.add(sample_id)
        return excluded_ids


# =============================================================================
# 第5步: 数据集划分器
# =============================================================================

class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, val_ratio: float = 0.5, random_seed: int = 42):
        """
        参数:
            val_ratio: 验证集比例 (在标注数据中)
            random_seed: 随机种子
        """
        self.val_ratio = val_ratio
        self.random_seed = random_seed
    
    def split_labeled(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        划分已标注数据为 val/test
        
        参数:
            samples: 已标注样本列表
        返回:
            (val_samples, test_samples)
        """
        random.seed(self.random_seed)
        
        # 按标签分层划分
        label_groups = defaultdict(list)
        for sample in samples:
            label_groups[sample['label']].append(sample)
        
        val_samples = []
        test_samples = []
        
        for label, group in label_groups.items():
            random.shuffle(group)
            split_idx = int(len(group) * self.val_ratio)
            val_samples.extend(group[:split_idx])
            test_samples.extend(group[split_idx:])
        
        # 打乱顺序
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        print(f"\n[DatasetSplitter] 划分结果:")
        print(f"  验证集: {len(val_samples)} 个样本")
        print(f"  测试集: {len(test_samples)} 个样本")
        
        # 统计各类别
        for name, dataset in [("验证集", val_samples), ("测试集", test_samples)]:
            label_counts = defaultdict(int)
            for s in dataset:
                label_counts[s['label']] += 1
            print(f"  {name}分布: {dict(label_counts)}")
        
        return val_samples, test_samples


# =============================================================================
# 第6步: 报告生成器
# =============================================================================

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, 
                 detector: LeakageDetector,
                 val_samples: List[Dict],
                 test_samples: List[Dict],
                 excluded_ids: Set[str]):
        """生成完整报告"""
        
        # 1. 保存排除ID列表
        excluded_path = self.output_dir / "excluded_sample_ids.txt"
        with open(excluded_path, 'w', encoding='utf-8') as f:
            for sample_id in sorted(excluded_ids):
                f.write(f"{sample_id}\n")
        print(f"[Report] 排除ID列表已保存: {excluded_path}")
        
        # 2. 保存验证集信息
        val_path = self.output_dir / "validation_samples.json"
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        print(f"[Report] 验证集信息已保存: {val_path}")
        
        # 3. 保存测试集信息
        test_path = self.output_dir / "test_samples.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        print(f"[Report] 测试集信息已保存: {test_path}")
        
        # 4. 生成汇总报告
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': {
                'total_labeled_samples': len(val_samples) + len(test_samples),
                'validation_samples': len(val_samples),
                'test_samples': len(test_samples),
                'excluded_raw_samples': len(excluded_ids),
                'matched_rate': len(detector.matches) / (len(detector.matches) + len(detector.unmatched)) 
                               if (detector.matches or detector.unmatched) else 0,
            },
            'validation_label_distribution': self._count_labels(val_samples),
            'test_label_distribution': self._count_labels(test_samples),
        }
        
        report_path = self.output_dir / "filter_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Report] 汇总报告已保存: {report_path}")
        
        # 5. 打印摘要
        self._print_summary(report)
    
    def _count_labels(self, samples: List[Dict]) -> Dict[str, int]:
        """统计标签分布"""
        counts = defaultdict(int)
        for s in samples:
            label_name = "正常" if s['label'] == 0 else ("故障" if s['label'] == 1 else "未知")
            counts[label_name] += 1
        return dict(counts)
    
    def _print_summary(self, report: Dict):
        """打印摘要"""
        print("\n" + "="*60)
        print("【数据泄露筛选报告】")
        print("="*60)
        print(f"生成时间: {report['generated_at']}")
        print(f"\n【统计信息】")
        for key, value in report['statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        print(f"\n【验证集标签分布】")
        for label, count in report['validation_label_distribution'].items():
            print(f"  {label}: {count}")
        print(f"\n【测试集标签分布】")
        for label, count in report['test_label_distribution'].items():
            print(f"  {label}: {count}")
        print("="*60)


# =============================================================================
# 第7步: 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='数据泄露筛选脚本')
    
    parser.add_argument('--labeled_dir', type=str, required=True,
                       help='已标注数据目录 (包含jsonl/json文件)')
    parser.add_argument('--raw_dir', type=str, required=True,
                       help='原始CSV数据目录')
    parser.add_argument('--output', type=str, default='./filtered_output',
                       help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.5,
                       help='验证集比例 (默认0.5，即val:test=50:50)')
    parser.add_argument('--signal_len', type=int, default=8192,
                       help='信号长度 (默认8192)')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                       help='原始数据采样率 (默认1.0，即全量)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("数据泄露筛选脚本 v1.0")
    print("="*60)
    print(f"已标注数据目录: {args.labeled_dir}")
    print(f"原始数据目录: {args.raw_dir}")
    print(f"输出目录: {args.output}")
    print(f"验证集比例: {args.val_ratio}")
    print("="*60 + "\n")
    
    # 1. 加载已标注数据
    print("[步骤1/5] 加载已标注数据...")
    labeled_loader = LabeledDataLoader(Path(args.labeled_dir))
    labeled_loader.load()
    
    # 2. 扫描原始数据
    print("\n[步骤2/5] 扫描原始CSV数据...")
    raw_scanner = RawDataScanner(Path(args.raw_dir), args.signal_len)
    raw_scanner.scan()
    
    # 3. 构建指纹索引
    print("\n[步骤3/5] 构建原始数据指纹索引...")
    raw_scanner.build_fingerprint_index(args.sample_rate)
    
    # 4. 执行泄露检测
    print("\n[步骤4/5] 执行数据泄露检测...")
    detector = LeakageDetector(labeled_loader, raw_scanner)
    detector.detect()
    
    # 5. 划分数据集
    print("\n[步骤5/5] 划分val/test数据集...")
    splitter = DatasetSplitter(args.val_ratio, args.seed)
    labeled_samples = labeled_loader.get_labeled_samples()
    val_samples, test_samples = splitter.split_labeled(labeled_samples)
    
    # 6. 生成报告
    print("\n[生成报告]")
    excluded_ids = detector.generate_excluded_ids()
    report_gen = ReportGenerator(Path(args.output))
    report_gen.generate(detector, val_samples, test_samples, excluded_ids)
    
    print("\n【完成】数据筛选完成！")
    print(f"  - 排除ID列表: {args.output}/excluded_sample_ids.txt")
    print(f"  - 验证集: {args.output}/validation_samples.json")
    print(f"  - 测试集: {args.output}/test_samples.json")


if __name__ == "__main__":
    main()

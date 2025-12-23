# -*- coding: utf-8 -*-
"""
data_manager.py - 数据管理
==========================

V5.12新增模块，包含:
- ChannelNameManager: 通道名称映射管理
- DataSplitManager: 数据划分管理
- CSV读取函数
- 样本ID生成
"""

import json
import hashlib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter, defaultdict
from tqdm import tqdm

from config import ThreeStageConfigV5


# =============================================================================
# 通道名称管理器
# =============================================================================

class ChannelNameManager:
    """
    通道名称映射管理器
    
    功能:
    - 将中文通道名映射为伪名称 (channel1, channel2, ...)
    - 统计传感器数量
    - 保留原始名称用于可视化
    """
    
    def __init__(self):
        self.original_to_pseudo: Dict[str, str] = {}  # 原始名 -> 伪名
        self.pseudo_to_original: Dict[str, str] = {}  # 伪名 -> 原始名
        self.file_mappings: Dict[str, Dict] = {}      # 文件级别映射
        self._counter = 0
    
    def register_channels(self, file_path: Path, channel_names: List[str]) -> Dict[str, str]:
        """
        注册一个文件的通道名称
        
        返回: {pseudo_name: original_name} 的映射
        """
        file_key = str(file_path)
        mapping = {}
        
        for idx, name in enumerate(channel_names):
            if name not in self.original_to_pseudo:
                self._counter += 1
                pseudo = f"channel{self._counter}"
                self.original_to_pseudo[name] = pseudo
                self.pseudo_to_original[pseudo] = name
            
            pseudo = self.original_to_pseudo[name]
            mapping[pseudo] = name
        
        self.file_mappings[file_key] = {
            'original_names': channel_names,
            'mapping': mapping,
            'channel_count': len(channel_names)
        }
        
        return mapping
    
    def get_pseudo_name(self, original_name: str) -> str:
        """获取伪名称"""
        return self.original_to_pseudo.get(original_name, original_name)
    
    def get_original_name(self, pseudo_name: str) -> str:
        """获取原始名称"""
        return self.pseudo_to_original.get(pseudo_name, pseudo_name)
    
    def get_sensor_count(self) -> int:
        """获取唯一传感器数量"""
        return len(self.original_to_pseudo)
    
    def print_summary(self):
        """打印通道信息摘要"""
        print("\n" + "="*50)
        print("【通道信息摘要】")
        print("="*50)
        print(f"  唯一传感器数: {self.get_sensor_count()}")
        print(f"  文件数: {len(self.file_mappings)}")
        
        if self.pseudo_to_original:
            print("\n  通道映射:")
            for pseudo, original in sorted(self.pseudo_to_original.items(), 
                                           key=lambda x: int(x[0].replace('channel', ''))):
                print(f"    {pseudo} → {original}")
        print("="*50 + "\n")
    
    def save_mapping(self, path: Path):
        """保存映射到JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'sensor_count': self.get_sensor_count(),
            'original_to_pseudo': self.original_to_pseudo,
            'pseudo_to_original': self.pseudo_to_original,
            'file_mappings': self.file_mappings
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[ChannelManager] 映射已保存: {path}")
    
    def load_mapping(self, path: Path):
        """从JSON加载映射"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.original_to_pseudo = data['original_to_pseudo']
        self.pseudo_to_original = data['pseudo_to_original']
        self.file_mappings = data.get('file_mappings', {})
        self._counter = len(self.original_to_pseudo)
        
        print(f"[ChannelManager] 映射已加载: {path}")


# 全局实例
CHANNEL_MANAGER = ChannelNameManager()


# =============================================================================
# CSV读取函数
# =============================================================================

def read_vibration_csv(csv_path: Path, cfg: ThreeStageConfigV5 = None) -> Optional[Dict]:
    """
    读取振动CSV文件
    
    参数:
        csv_path: CSV文件路径
        cfg: 配置对象
    
    返回:
        {
            'data': np.ndarray (samples, channels),
            'channel_names': List[str],
            'pseudo_names': List[str],
            'mapping': Dict[str, str],
            'channel_count': int
        }
    """
    csv_path = Path(csv_path)
    
    # 尝试不同编码读取
    for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        print(f"[警告] 无法读取CSV文件: {csv_path}")
        return None
    
    # 获取通道名称
    channel_names = df.columns.tolist()
    
    # 注册通道映射
    mapping = CHANNEL_MANAGER.register_channels(csv_path, channel_names)
    pseudo_names = [CHANNEL_MANAGER.get_pseudo_name(name) for name in channel_names]
    
    # 转换为numpy数组
    data = df.values.astype(np.float32)
    
    return {
        'data': data,
        'channel_names': channel_names,
        'pseudo_names': pseudo_names,
        'mapping': mapping,
        'channel_count': len(channel_names),
        'file_path': str(csv_path)
    }


def scan_csv_files(data_dir: Path, pattern: str = "*.csv") -> List[Path]:
    """
    扫描目录下所有CSV文件
    
    参数:
        data_dir: 数据目录
        pattern: 文件匹配模式
    
    返回:
        CSV文件路径列表
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[警告] 数据目录不存在: {data_dir}")
        return []
    
    csv_files = list(data_dir.rglob(pattern))
    print(f"[scan_csv_files] 找到 {len(csv_files)} 个CSV文件")
    
    return csv_files


def generate_sample_id(csv_path: Path, channel_idx: int, start_idx: int) -> str:
    """
    生成唯一的样本ID
    
    格式: {filename}_ch{channel_idx}_s{start_idx}
    """
    stem = Path(csv_path).stem
    return f"{stem}_ch{channel_idx}_s{start_idx}"


def compute_signal_fingerprint(signal: np.ndarray) -> str:
    """
    计算信号指纹 (用于数据泄露检测)
    
    使用统计特征生成哈希，对轻微变化更鲁棒
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    
    # 统计特征
    features = [
        float(np.mean(signal)),
        float(np.std(signal)),
        float(np.min(signal)),
        float(np.max(signal)),
    ]
    
    # 头尾采样
    n_sample = min(5, len(signal))
    features.extend(signal[:n_sample].tolist())
    features.extend(signal[-n_sample:].tolist())
    
    # 生成哈希
    feat_str = ",".join(f"{x:.6f}" for x in features)
    return hashlib.md5(feat_str.encode()).hexdigest()[:16]


# =============================================================================
# 数据划分管理器
# =============================================================================

class DataSplitManager:
    """
    数据划分管理器
    
    功能:
    1. 从原始CSV扫描所有样本
    2. 加载已标注数据作为val/test
    3. 自动排除已标注样本，剩余作为train
    4. 支持保存和加载划分结果
    """
    
    def __init__(self, cfg: ThreeStageConfigV5 = None):
        self.cfg = cfg
        self.train_samples = []      # 训练样本
        self.val_samples = []        # 验证样本
        self.test_samples = []       # 测试样本
        self.excluded_ids = set()    # 需要排除的样本ID
        self.split_info = {}         # 划分统计信息
    
    def load_from_filter_output(self, filter_output_dir: Path) -> bool:
        """
        从data_leakage_filter.py的输出加载划分结果
        """
        filter_dir = Path(filter_output_dir)
        
        # 加载排除ID列表
        excluded_path = filter_dir / "excluded_sample_ids.txt"
        if excluded_path.exists():
            with open(excluded_path, 'r', encoding='utf-8') as f:
                self.excluded_ids = {line.strip() for line in f if line.strip()}
            print(f"[DataSplitManager] 加载 {len(self.excluded_ids)} 个排除ID")
        
        # 加载验证集信息
        val_path = filter_dir / "validation_samples.json"
        if val_path.exists():
            with open(val_path, 'r', encoding='utf-8') as f:
                self.val_samples = json.load(f)
            print(f"[DataSplitManager] 加载 {len(self.val_samples)} 个验证样本")
        
        # 加载测试集信息
        test_path = filter_dir / "test_samples.json"
        if test_path.exists():
            with open(test_path, 'r', encoding='utf-8') as f:
                self.test_samples = json.load(f)
            print(f"[DataSplitManager] 加载 {len(self.test_samples)} 个测试样本")
        
        return len(self.val_samples) > 0 or len(self.test_samples) > 0
    
    def auto_split_labeled_data(self, labeled_dir: Path, val_ratio: float = 0.5,
                                 seed: int = 42, class_keywords: Dict = None) -> Tuple[List[Dict], List[Dict]]:
        """
        自动划分已标注数据为val/test
        
        参数:
            labeled_dir: 已标注数据目录
            val_ratio: 验证集比例
            seed: 随机种子
            class_keywords: 类别关键词
        """
        random.seed(seed)
        np.random.seed(seed)
        
        labeled_dir = Path(labeled_dir)
        if not labeled_dir.exists():
            print(f"[警告] 已标注数据目录不存在: {labeled_dir}")
            return [], []
        
        if class_keywords is None:
            class_keywords = {
                "正常": ["正常", "normal", "good", "健康"],
                "故障": ["故障", "异常", "fault", "abnormal", "defect", "error"]
            }
        
        # 收集所有已标注样本
        all_samples = []
        label_counts = Counter()
        
        # 扫描jsonl和json文件
        for ext in ["*.jsonl", "*.json"]:
            for data_file in labeled_dir.rglob(ext):
                parent_name = data_file.parent.name.lower()
                
                # 推断标签
                label = -1
                for class_name, keywords in class_keywords.items():
                    if any(kw in parent_name for kw in keywords):
                        label = 0 if class_name == "正常" else 1
                        break
                
                if label == -1:
                    continue
                
                # 读取文件
                try:
                    if ext == "*.jsonl":
                        with open(data_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    all_samples.append({
                                        'file_path': str(data_file),
                                        'line_num': line_num,
                                        'label': label,
                                        'data': data,
                                    })
                                    label_counts[label] += 1
                                except json.JSONDecodeError:
                                    continue
                    else:  # json
                        with open(data_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for idx, item in enumerate(data):
                                    all_samples.append({
                                        'file_path': str(data_file),
                                        'line_num': idx,
                                        'label': label,
                                        'data': item,
                                    })
                                    label_counts[label] += 1
                            else:
                                all_samples.append({
                                    'file_path': str(data_file),
                                    'line_num': 0,
                                    'label': label,
                                    'data': data,
                                })
                                label_counts[label] += 1
                except Exception as e:
                    print(f"[警告] 读取文件失败: {data_file}, 错误: {e}")
        
        print(f"[DataSplitManager] 共找到 {len(all_samples)} 个已标注样本")
        for lbl, cnt in sorted(label_counts.items()):
            name = "正常" if lbl == 0 else "故障"
            print(f"  {name}: {cnt}")
        
        # 按标签分层划分
        label_groups = defaultdict(list)
        for sample in all_samples:
            label_groups[sample['label']].append(sample)
        
        val_samples = []
        test_samples = []
        
        for label, group in label_groups.items():
            random.shuffle(group)
            split_idx = int(len(group) * val_ratio)
            val_samples.extend(group[:split_idx])
            test_samples.extend(group[split_idx:])
        
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        self.val_samples = val_samples
        self.test_samples = test_samples
        
        print(f"\n[DataSplitManager] 划分结果:")
        print(f"  验证集: {len(val_samples)} 个样本")
        print(f"  测试集: {len(test_samples)} 个样本")
        
        return val_samples, test_samples
    
    def scan_train_data(self, csv_files: List[Path], signal_len: int = 8192,
                        overlap: float = 0.5) -> int:
        """
        扫描CSV文件生成训练样本（排除已标注）
        
        参数:
            csv_files: CSV文件列表
            signal_len: 信号长度
            overlap: 滑动窗口重叠比例
        """
        self.train_samples = []
        excluded_count = 0
        hop = int(signal_len * (1 - overlap))
        
        for csv_path in tqdm(csv_files, desc="扫描训练数据"):
            csv_data = read_vibration_csv(csv_path, self.cfg)
            if csv_data is None:
                continue
            
            data = csv_data['data']
            n_samples, n_channels = data.shape
            
            for ch_idx in range(n_channels):
                channel_data = data[:, ch_idx]
                
                # 滑动窗口切分
                for start_idx in range(0, len(channel_data) - signal_len + 1, hop):
                    sample_id = generate_sample_id(csv_path, ch_idx, start_idx)
                    
                    # 检查是否需要排除
                    if sample_id in self.excluded_ids:
                        excluded_count += 1
                        continue
                    
                    signal = channel_data[start_idx:start_idx + signal_len]
                    
                    self.train_samples.append({
                        'signal': signal.copy(),
                        'sample_id': sample_id,
                        'channel_name': csv_data['channel_names'][ch_idx],
                        'pseudo_name': csv_data['pseudo_names'][ch_idx],
                        'file_path': str(csv_path),
                        'label': -1,  # 无标签
                    })
        
        print(f"\n[DataSplitManager] 训练集扫描完成:")
        print(f"  训练样本: {len(self.train_samples)}")
        print(f"  排除样本: {excluded_count}")
        
        return len(self.train_samples)
    
    def generate_excluded_ids_from_labeled(self) -> Set[str]:
        """
        从已标注数据生成排除ID
        """
        excluded = set()
        
        for sample in self.val_samples + self.test_samples:
            data = sample.get('data', {})
            source_file = data.get('source_file', data.get('file', ''))
            
            if source_file:
                stem = Path(source_file).stem
                excluded.add(stem)
        
        self.excluded_ids = excluded
        return excluded
    
    def save_split(self, output_dir: Path):
        """保存划分结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存排除ID
        with open(output_dir / "excluded_sample_ids.txt", 'w', encoding='utf-8') as f:
            for sample_id in sorted(self.excluded_ids):
                f.write(f"{sample_id}\n")
        
        # 保存val样本信息（简化版）
        val_info = [{'file_path': s['file_path'], 'line_num': s['line_num'], 'label': s['label']} 
                    for s in self.val_samples]
        with open(output_dir / "validation_samples.json", 'w', encoding='utf-8') as f:
            json.dump(val_info, f, ensure_ascii=False, indent=2)
        
        # 保存test样本信息
        test_info = [{'file_path': s['file_path'], 'line_num': s['line_num'], 'label': s['label']} 
                     for s in self.test_samples]
        with open(output_dir / "test_samples.json", 'w', encoding='utf-8') as f:
            json.dump(test_info, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            'train_samples': len(self.train_samples),
            'val_samples': len(self.val_samples),
            'test_samples': len(self.test_samples),
            'excluded_ids': len(self.excluded_ids),
        }
        with open(output_dir / "split_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"[DataSplitManager] 划分结果已保存至: {output_dir}")
    
    def print_summary(self):
        """打印划分摘要"""
        print("\n" + "="*60)
        print("【数据划分摘要】")
        print("="*60)
        print(f"  训练集 (Train): {len(self.train_samples)} 样本 (无标签)")
        print(f"  验证集 (Val):   {len(self.val_samples)} 样本 (有标签)")
        print(f"  测试集 (Test):  {len(self.test_samples)} 样本 (有标签)")
        print(f"  排除ID数:       {len(self.excluded_ids)}")
        
        # 统计val/test的标签分布
        for name, samples in [("验证集", self.val_samples), ("测试集", self.test_samples)]:
            if samples:
                label_counts = Counter(s['label'] for s in samples)
                dist_str = ", ".join(f"{'正常' if l==0 else '故障'}:{c}" 
                                    for l, c in sorted(label_counts.items()))
                print(f"  {name}分布: {dist_str}")
        
        print("="*60 + "\n")

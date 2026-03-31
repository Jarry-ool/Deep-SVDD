# -*- coding: utf-8 -*-
"""
datasets.py - 数据集类 (V5.12 性能优化版)
==========================================

V5.12 优化:
1. 按branch_mode条件计算 - hetero/zerone模式跳过不需要的分支
2. zerone特征缓存复用 - __init__时加载缓存到内存，__getitem__直接查表
3. 多进程并行加载

性能提升:
- hetero模式: 省去zerone特征计算 (~40%)
- zerone模式: 省去CWT/STFT图像生成 (~60%)  
- dual模式+缓存: 省去zerone特征计算 (~35%)

使用方法:
    cfg = ThreeStageConfigV5(BRANCH_MODE='hetero')  # 或 'zerone', 'dual'
    dataset = CSVVibrationDataset(files, cfg, normalizer=normalizer)
    # __getitem__会自动根据branch_mode优化计算
"""

import json
import hashlib
import numpy as np
import cv2
import pywt
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from config import (
    ThreeStageConfigV5, TOTAL_FEAT_DIM,
    TIME_DOMAIN_DIM, STFT_BAND_DIM, PSD_BAND_DIM, HIGH_FREQ_DIM
)
from features import extract_zerone_features, split_feature_vector
from utils import GlobalNormalizer


# =============================================================================
# 全局配置
# =============================================================================
NUM_WORKERS = min(16, cpu_count())


# =============================================================================
# 并行处理辅助函数 (模块顶层定义，支持pickle)
# =============================================================================

def _load_signal_from_jsonl(file_path: str) -> Tuple[str, np.ndarray, int]:
    """从JSONL文件加载信号"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
        
        raw_signal = data.get('signal_value', data.get('signal', None))
        if raw_signal is None:
            signal = np.zeros(8192, dtype=np.float32)
        elif isinstance(raw_signal, str):
            signal = np.array([float(x) for x in raw_signal.split(',')], dtype=np.float32)
        else:
            signal = np.array(raw_signal, dtype=np.float32)
        
        return (file_path, signal, -1)
    except:
        return (file_path, np.zeros(8192, dtype=np.float32), -1)


def _load_signal_from_jsonl_with_label(args: Tuple) -> Dict:
    """从JSONL文件加载信号和标签"""
    file_path, signal_len, class_keywords = args
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
        
        raw_signal = data.get('signal_value', data.get('signal', None))
        if raw_signal is None:
            signal = np.zeros(signal_len, dtype=np.float32)
        elif isinstance(raw_signal, str):
            signal = np.array([float(x) for x in raw_signal.split(',')], dtype=np.float32)
        else:
            signal = np.array(raw_signal, dtype=np.float32)
        
        if len(signal) < signal_len:
            signal = np.pad(signal, (0, signal_len - len(signal)))
        else:
            signal = signal[:signal_len]
        
        # 推断标签
        parent_name = Path(file_path).parent.name.lower()
        label = -1
        if class_keywords:
            for class_name, keywords in class_keywords.items():
                if any(kw in parent_name for kw in keywords):
                    label = 0 if class_name == "正常" else 1
                    break
        
        return {'file': file_path, 'signal': signal, 'label': label}
    except:
        return {'file': file_path, 'signal': np.zeros(signal_len, dtype=np.float32), 'label': -1}


def _extract_feature_worker(args: Tuple) -> np.ndarray:
    """特征提取worker"""
    signal, fs = args
    try:
        from features import extract_zerone_features
        return extract_zerone_features(signal, fs=fs)
    except:
        return np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)


def _read_and_segment_csv(args: Tuple) -> List[Dict]:
    """读取CSV并切分为多个样本"""
    csv_path, signal_len = args
    samples = []
    try:
        import pandas as pd
        
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except:
                continue
        
        if df is None or df.empty:
            return samples
        
        # 找时间列
        time_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['time', '时间', 'date', 'unnamed']):
                time_col = col
                break
        
        # 数据列
        data_cols = [c for c in df.columns if c != time_col]
        numeric_cols = []
        for c in data_cols:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if df[c].notna().sum() > 0:
                    numeric_cols.append(c)
            except:
                pass
        
        if not numeric_cols:
            return samples
        
        data_matrix = df[numeric_cols].values.astype(np.float32)
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)
        
        n_rows, n_channels = data_matrix.shape
        n_segments = n_rows // signal_len
        
        for ch_idx, ch_name in enumerate(numeric_cols):
            channel_data = data_matrix[:, ch_idx]
            
            for seg_idx in range(n_segments):
                start = seg_idx * signal_len
                end = start + signal_len
                signal = channel_data[start:end].copy()
                
                sample_id = f"{Path(csv_path).stem}_{ch_name}_{seg_idx}"
                samples.append({
                    'sample_id': sample_id,
                    'file': str(csv_path),
                    'channel': ch_name,
                    'segment': seg_idx,
                    'signal': signal,
                    'label': -1,
                })
    except:
        pass
    return samples


# =============================================================================
# 特征缓存管理器
# =============================================================================

class FeatureCache:
    """特征缓存管理器"""
    
    def __init__(self, cache_dir: Path, dataset_name: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
    
    def _get_cache_path(self, hash_val: str, suffix: str = 'npy') -> Path:
        return self.cache_dir / f"cache_{self.dataset_name}_{hash_val}.{suffix}"
    
    def _compute_hash(self, items: List) -> str:
        content = str(sorted([str(p) for p in items[:500]]))
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def load_features(self, keys: List) -> Optional[np.ndarray]:
        """加载特征缓存"""
        hash_val = self._compute_hash(keys)
        cache_path = self._get_cache_path(hash_val, 'npy')
        if cache_path.exists():
            print(f"[缓存] 加载特征: {cache_path.name}")
            return np.load(cache_path)
        return None
    
    def save_features(self, keys: List, features: np.ndarray):
        """保存特征缓存"""
        hash_val = self._compute_hash(keys)
        cache_path = self._get_cache_path(hash_val, 'npy')
        np.save(cache_path, features)
        print(f"[缓存] 特征已保存: {cache_path.name} ({len(features)} 样本)")


# =============================================================================
# Zerone图像生成
# =============================================================================

def _render_panel_raster(vec: np.ndarray, wrap: bool, tile_width: int, 
                         W_UNIT: int = 3, H_UNIT: int = 2) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = v.size
    if n == 0:
        return np.zeros((H_UNIT, W_UNIT), dtype=np.float32)
    
    if not wrap:
        rows, cols = 1, n
    else:
        cols = max(1, int(tile_width))
        rows = int(np.ceil(n / float(cols)))
    
    H = rows * H_UNIT
    W = cols * W_UNIT
    canvas = np.zeros((H, W), dtype=np.float32)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x0, y0 = c * W_UNIT, r * H_UNIT
            canvas[y0:y0 + H_UNIT, x0:x0 + W_UNIT] = v[idx]
            idx += 1
    
    return np.clip(canvas, 0.0, 1.0)


def vector_to_image_raster(vec_norm01: np.ndarray, target_size: int = 224) -> np.ndarray:
    """将归一化后的1200维特征向量转换为3通道图像"""
    W_UNIT, H_UNIT = 3, 2
    tile_widths = {"time": 16, "stft": 16, "psd": 32, "hf": 8}
    panel_order = ["time", "stft", "psd", "hf"]
    GAP_ROWS = 1
    
    segs = split_feature_vector(np.asarray(vec_norm01, dtype=np.float32))
    name2seg = {
        "time": segs.get("time", np.zeros(TIME_DOMAIN_DIM)),
        "stft": segs.get("stft", np.zeros(STFT_BAND_DIM)),
        "psd": segs.get("psd", np.zeros(PSD_BAND_DIM)),
        "hf": segs.get("hf", np.zeros(HIGH_FREQ_DIM)),
    }
    
    panels = []
    for name in panel_order:
        tw = tile_widths.get(name, 16)
        wrap = (name != "time")
        pan = _render_panel_raster(name2seg[name], wrap=wrap, tile_width=tw)
        panels.append(pan)
    
    max_w = max(p.shape[1] for p in panels)
    normalized_panels = []
    for p in panels:
        if p.shape[1] < max_w:
            pad_w = max_w - p.shape[1]
            p = np.pad(p, ((0, 0), (0, pad_w)), mode='constant', constant_values=0)
        normalized_panels.append(p)
    
    gap = np.zeros((GAP_ROWS, max_w), dtype=np.float32)
    rows_list = []
    for i, p in enumerate(normalized_panels):
        rows_list.append(p)
        if i < len(normalized_panels) - 1:
            rows_list.append(gap)
    
    if rows_list:
        combined = np.vstack(rows_list)
    else:
        combined = np.zeros((target_size, target_size), dtype=np.float32)
    
    combined = cv2.resize(combined, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = np.stack([combined, combined, combined], axis=0)
    
    return img.astype(np.float32)


def generate_hetero_image(signal: np.ndarray, size: int = 224) -> np.ndarray:
    """
    生成Hetero三通道图像 (CWT + STFT + 时域)
    
    通道:
    - R: CWT时频图
    - G: STFT时频图
    - B: 时域reshape
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    
    # CWT
    try:
        scales = np.arange(1, 65)
        coeffs, _ = pywt.cwt(signal, scales, 'morl')
        cwt_img = np.abs(coeffs)
        cwt_img = cv2.resize(cwt_img, (size, size))
        cmin, cmax = cwt_img.min(), cwt_img.max()
        cwt_img = (cwt_img - cmin) / (cmax - cmin + 1e-8) if cmax - cmin > 1e-8 else np.full((size, size), 0.5)
    except:
        cwt_img = np.full((size, size), 0.5)
    
    # STFT
    try:
        stft_matrix = []
        for i in range(0, len(signal) - 256, 64):
            seg = signal[i:i+256]
            spec = np.abs(np.fft.rfft(seg * np.hanning(256)))
            stft_matrix.append(spec)
        stft_img = np.array(stft_matrix).T if stft_matrix else np.zeros((129, 1))
        stft_img = cv2.resize(stft_img, (size, size))
        smin, smax = stft_img.min(), stft_img.max()
        stft_img = (stft_img - smin) / (smax - smin + 1e-8) if smax - smin > 1e-8 else np.full((size, size), 0.5)
    except:
        stft_img = np.full((size, size), 0.5)
    
    # Context (时域reshape)
    try:
        total = size * size
        ctx = signal[:total] if len(signal) >= total else np.pad(signal, (0, total - len(signal)))
        ctx_img = ctx.reshape(size, size)
        ctmin, ctmax = ctx_img.min(), ctx_img.max()
        ctx_img = (ctx_img - ctmin) / (ctmax - ctmin + 1e-8) if ctmax - ctmin > 1e-8 else np.full((size, size), 0.5)
    except:
        ctx_img = np.full((size, size), 0.5)
    
    return np.stack([cwt_img, stft_img, ctx_img], axis=0).astype(np.float32)


# =============================================================================
# 基类：优化的__getitem__逻辑
# =============================================================================

class OptimizedDatasetMixin:
    """
    优化的Dataset混入类
    
    提供:
    1. 按branch_mode条件计算
    2. zerone特征缓存复用
    """
    
    def _init_feature_cache_memory(self):
        """
        尝试将特征缓存加载到内存
        在Dataset.__init__末尾调用
        
        注意: 必须使用和get_all_features_for_normalization相同的key格式
        """
        self._feature_cache_memory = None
        
        if not hasattr(self, 'samples') or not self.samples:
            return
        
        # 使用和get_all_features_for_normalization相同的key格式
        # CSVVibrationDataset: sample_id
        # TransformerVibrationDataset: file_paths
        # LabeledVibrationDataset: file
        if hasattr(self, 'file_paths') and self.file_paths:
            # TransformerVibrationDataset
            cache_keys = self.file_paths
        elif self.samples and 'sample_id' in self.samples[0]:
            # CSVVibrationDataset
            cache_keys = [s['sample_id'] for s in self.samples]
        else:
            # LabeledVibrationDataset
            cache_keys = [s.get('file', f'sample_{i}') for i, s in enumerate(self.samples)]
        
        # 尝试加载缓存
        if hasattr(self, 'cache'):
            cached = self.cache.load_features(cache_keys)
            if cached is not None and len(cached) == len(self.samples):
                self._feature_cache_memory = cached
                print(f"[{self.split_name}] 特征缓存已加载到内存 ({len(cached)} 样本)")
    
    def _get_zerone_feat(self, idx: int, signal: np.ndarray) -> np.ndarray:
        """
        获取zerone特征 (优先用缓存)
        
        Args:
            idx: 样本索引
            signal: 原始信号
        
        Returns:
            1200维特征向量
        """
        # 优先用内存缓存
        if self._feature_cache_memory is not None:
            return self._feature_cache_memory[idx]
        
        # 否则实时计算
        return extract_zerone_features(signal, fs=self.cfg.FS)
    
    def _optimized_getitem(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        优化的__getitem__实现
        
        模式:
        - GPU_HETERO模式: 返回signal，由训练循环在GPU上批量计算hetero
        - hetero: CPU计算hetero图像，zerone给零占位
        - zerone: hetero给零占位，CPU计算zerone特征/图像
        - dual: 都计算（zerone优先用缓存）
        
        Returns:
            (hetero_or_signal, zerone_out, label, idx)
        """
        sample = self.samples[idx]
        signal = sample.get('signal', np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32))
        
        # 确保信号长度
        if len(signal) < self.cfg.SIGNAL_LEN:
            signal = np.pad(signal, (0, self.cfg.SIGNAL_LEN - len(signal)))
        else:
            signal = signal[:self.cfg.SIGNAL_LEN]
        
        branch_mode = self.cfg.BRANCH_MODE
        size = self.cfg.INPUT_SIZE
        use_gpu_hetero = getattr(self.cfg, 'USE_GPU_HETERO', False)
        
        # === GPU模式: 返回signal，让训练循环在GPU上批量计算 ===
        if use_gpu_hetero and branch_mode in ('hetero', 'dual'):
            # 返回signal而不是hetero_img
            signal_tensor = torch.from_numpy(signal).float()
            
            if branch_mode == 'hetero':
                # zerone给零占位
                if self.cfg.ZERONE_USE_CNN:
                    zerone_out = np.zeros((3, size, size), dtype=np.float32)
                else:
                    zerone_out = np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)
            else:  # dual
                zerone_feat = self._get_zerone_feat(idx, signal)
                if self.normalizer is not None and self.normalizer.is_fitted:
                    zerone_feat_norm = self.normalizer.transform(zerone_feat)
                else:
                    zerone_feat_norm = np.clip(zerone_feat, 0, 1)
                
                if self.cfg.ZERONE_USE_CNN:
                    zerone_out = vector_to_image_raster(zerone_feat_norm, target_size=size)
                else:
                    zerone_out = zerone_feat_norm
            
            label = sample.get('label', 0)
            if label == -1:
                label = 0
            
            return (
                signal_tensor,  # 返回signal，不是hetero_img
                torch.from_numpy(zerone_out).float(),
                label,
                idx
            )
        
        # === CPU模式: 原有逻辑 ===
        if branch_mode == 'hetero':
            hetero_img = generate_hetero_image(signal, size)
            if self.cfg.ZERONE_USE_CNN:
                zerone_out = np.zeros((3, size, size), dtype=np.float32)
            else:
                zerone_out = np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)
        
        elif branch_mode == 'zerone':
            hetero_img = np.zeros((3, size, size), dtype=np.float32)
            zerone_feat = self._get_zerone_feat(idx, signal)
            if self.normalizer is not None and self.normalizer.is_fitted:
                zerone_feat_norm = self.normalizer.transform(zerone_feat)
            else:
                zerone_feat_norm = np.clip(zerone_feat, 0, 1)
            
            if self.cfg.ZERONE_USE_CNN:
                zerone_out = vector_to_image_raster(zerone_feat_norm, target_size=size)
            else:
                zerone_out = zerone_feat_norm
        
        else:  # dual
            hetero_img = generate_hetero_image(signal, size)
            zerone_feat = self._get_zerone_feat(idx, signal)
            
            if self.normalizer is not None and self.normalizer.is_fitted:
                zerone_feat_norm = self.normalizer.transform(zerone_feat)
            else:
                zerone_feat_norm = np.clip(zerone_feat, 0, 1)
            
            if self.cfg.ZERONE_USE_CNN:
                zerone_out = vector_to_image_raster(zerone_feat_norm, target_size=size)
            else:
                zerone_out = zerone_feat_norm
        
        label = sample.get('label', 0)
        if label == -1:
            label = 0
        
        return (
            torch.from_numpy(hetero_img).float(),
            torch.from_numpy(zerone_out).float(),
            label,
            idx
        )


# =============================================================================
# JSONL数据集
# =============================================================================

class TransformerVibrationDataset(Dataset, OptimizedDatasetMixin):
    """JSONL格式变压器振动数据集"""
    
    def __init__(self, data_dir: Path, cfg: ThreeStageConfigV5, 
                 use_labels: bool = False, split_name: str = "UNKNOWN",
                 normalizer: GlobalNormalizer = None,
                 num_workers: int = None):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.split_name = split_name
        self.normalizer = normalizer
        self.num_workers = num_workers or NUM_WORKERS
        
        self.cache = FeatureCache(cfg.OUTPUT_ROOT / "cache", f"{split_name}_jsonl")
        self.samples = []
        self.file_paths = []
        
        self._load_samples_parallel()
        self._init_feature_cache_memory()  # 加载特征缓存到内存
    
    def _load_samples_parallel(self):
        """并行加载所有样本"""
        if not self.data_dir.exists():
            print(f"[警告] 目录不存在: {self.data_dir}")
            return
        
        print(f"[{self.split_name}] 扫描文件...")
        self.file_paths = list(self.data_dir.rglob("*.jsonl"))
        
        if not self.file_paths:
            print(f"[{self.split_name}] 未找到jsonl文件")
            return
        
        print(f"[{self.split_name}] 并行加载 {len(self.file_paths)} 个信号...")
        
        args_list = [
            (str(f), self.cfg.SIGNAL_LEN, dict(self.cfg.CLASS_KEYWORDS))
            for f in self.file_paths
        ]
        
        with Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_load_signal_from_jsonl_with_label, args_list),
                total=len(args_list),
                desc=f"{self.split_name}加载",
                ncols=80, leave=True, dynamic_ncols=False
            ))
        
        for r in results:
            if self.use_labels and r['label'] == -1:
                continue
            self.samples.append(r)
        
        print(f"[{self.split_name}] 加载完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        return self._optimized_getitem(idx)
    
    def get_all_features_for_normalization(self) -> List[np.ndarray]:
        """并行提取所有特征 (带缓存)"""
        cached = self.cache.load_features(self.file_paths)
        if cached is not None:
            return list(cached)
        
        print(f"[{self.split_name}] 并行提取特征 (workers={self.num_workers})...")
        
        args_list = [(s['signal'], self.cfg.FS) for s in self.samples]
        
        with Pool(self.num_workers) as pool:
            features = list(tqdm(
                pool.imap(_extract_feature_worker, args_list),
                total=len(args_list),
                desc="提取特征",
                ncols=80, leave=True, dynamic_ncols=False
            ))
        
        self.cache.save_features(self.file_paths, np.array(features))
        return features


# =============================================================================
# CSV数据集
# =============================================================================

class CSVVibrationDataset(Dataset, OptimizedDatasetMixin):
    """CSV格式振动数据集"""
    
    def __init__(self, csv_files: List[Path], cfg: ThreeStageConfigV5,
                 use_labels: bool = False, split_name: str = "TRAIN",
                 normalizer: GlobalNormalizer = None,
                 excluded_ids: Set[str] = None,
                 num_workers: int = None):
        self.csv_files = [Path(f) for f in csv_files]
        self.cfg = cfg
        self.use_labels = use_labels
        self.split_name = split_name
        self.normalizer = normalizer
        self.excluded_ids = excluded_ids or set()
        self.num_workers = num_workers or NUM_WORKERS
        
        self.cache = FeatureCache(cfg.OUTPUT_ROOT / "cache", f"{split_name}_csv")
        self.samples = []
        
        self._load_csv_parallel()
        self._init_feature_cache_memory()  # 加载特征缓存到内存
    
    def _load_csv_parallel(self):
        """并行加载CSV文件并切分"""
        if not self.csv_files:
            print(f"[{self.split_name}] 无CSV文件")
            return
        
        if not isinstance(self.excluded_ids, set):
            self.excluded_ids = set(self.excluded_ids)
        
        print(f"[{self.split_name}] 并行读取+切分 {len(self.csv_files)} 个CSV (workers={self.num_workers})...")
        
        args_list = [(str(f), self.cfg.SIGNAL_LEN) for f in self.csv_files]
        
        with Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_read_and_segment_csv, args_list),
                total=len(args_list),
                desc=f"{self.split_name}读取",
                ncols=80, leave=True, dynamic_ncols=False
            ))
        
        all_samples = []
        for sample_list in results:
            all_samples.extend(sample_list)
        
        print(f"[{self.split_name}] 切分完成，共 {len(all_samples)} 个样本")
        
        if self.excluded_ids:
            self.samples = [s for s in all_samples if s['sample_id'] not in self.excluded_ids]
            excluded_count = len(all_samples) - len(self.samples)
            print(f"[{self.split_name}] 最终 {len(self.samples)} 个样本, 排除 {excluded_count} 个")
        else:
            self.samples = all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        return self._optimized_getitem(idx)
    
    def get_all_features_for_normalization(self) -> List[np.ndarray]:
        """并行提取所有特征"""
        sample_ids = [s['sample_id'] for s in self.samples]
        
        cached = self.cache.load_features(sample_ids)
        if cached is not None and len(cached) == len(self.samples):
            return list(cached)
        
        print(f"[{self.split_name}] 并行提取特征 (workers={self.num_workers})...")
        
        args_list = [(s['signal'], self.cfg.FS) for s in self.samples]
        
        with Pool(self.num_workers) as pool:
            features = list(tqdm(
                pool.imap(_extract_feature_worker, args_list),
                total=len(args_list),
                desc="提取特征",
                ncols=80, leave=True, dynamic_ncols=False
            ))
        
        self.cache.save_features(sample_ids, np.array(features))
        return features


# =============================================================================
# 已标注数据集
# =============================================================================

class LabeledVibrationDataset(Dataset, OptimizedDatasetMixin):
    """已标注振动数据集"""
    
    def __init__(self, samples: List[Dict], cfg: ThreeStageConfigV5,
                 split_name: str = "UNKNOWN", normalizer: GlobalNormalizer = None,
                 num_workers: int = None):
        self.cfg = cfg
        self.split_name = split_name
        self.normalizer = normalizer
        self.num_workers = num_workers or NUM_WORKERS
        
        self.cache = FeatureCache(cfg.OUTPUT_ROOT / "cache", f"{split_name}_labeled")
        self.samples = samples
        
        self._preload_signals_parallel()
        self._init_feature_cache_memory()  # 加载特征缓存到内存
    
    def _preload_signals_parallel(self):
        """并行预加载信号"""
        if not self.samples:
            return
        
        need_load = [s for s in self.samples if 'signal' not in s]
        if not need_load:
            print(f"[{self.split_name}] {len(self.samples)} 个样本已预加载")
            return
        
        print(f"[{self.split_name}] 并行加载 {len(need_load)} 个信号...")
        
        def load_single(sample):
            try:
                file_path = sample.get('file_path', sample.get('file', ''))
                line_num = sample.get('line_num', 0)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if line_num < len(lines):
                        data = json.loads(lines[line_num])
                        raw_signal = data.get('signal_value', data.get('signal', None))
                        
                        if raw_signal is None:
                            return np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
                        elif isinstance(raw_signal, str):
                            signal = np.array([float(x) for x in raw_signal.split(',')], dtype=np.float32)
                        else:
                            signal = np.array(raw_signal, dtype=np.float32)
                        
                        if len(signal) < self.cfg.SIGNAL_LEN:
                            signal = np.pad(signal, (0, self.cfg.SIGNAL_LEN - len(signal)))
                        else:
                            signal = signal[:self.cfg.SIGNAL_LEN]
                        return signal
            except:
                pass
            return np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            signals = list(tqdm(
                executor.map(load_single, need_load),
                total=len(need_load),
                desc=f"加载{self.split_name}",
                ncols=80, leave=True, dynamic_ncols=False
            ))
        
        for sample, signal in zip(need_load, signals):
            sample['signal'] = signal
        
        print(f"[{self.split_name}] 加载完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        return self._optimized_getitem(idx)
    
    def get_all_features_for_normalization(self) -> List[np.ndarray]:
        """并行提取特征"""
        file_keys = [s.get('file', f'sample_{i}') for i, s in enumerate(self.samples)]
        
        cached = self.cache.load_features(file_keys)
        if cached is not None and len(cached) == len(self.samples):
            return list(cached)
        
        print(f"[{self.split_name}] 并行提取特征 (workers={self.num_workers})...")
        
        args_list = [(s.get('signal', np.zeros(self.cfg.SIGNAL_LEN)), self.cfg.FS) 
                     for s in self.samples]
        
        with Pool(self.num_workers) as pool:
            features = list(tqdm(
                pool.imap(_extract_feature_worker, args_list),
                total=len(args_list),
                desc="提取特征",
                ncols=80, leave=True, dynamic_ncols=False
            ))
        
        self.cache.save_features(file_keys, np.array(features))
        return features

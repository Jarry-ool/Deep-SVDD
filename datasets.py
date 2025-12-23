# -*- coding: utf-8 -*-
"""
datasets.py - 数据集类
=====================

包含:
- Zerone图像生成函数
- TransformerVibrationDataset: JSONL格式数据集
- CSVVibrationDataset: CSV格式数据集 (V5.12新增)
- LabeledVibrationDataset: 已标注数据集 (用于val/test)
"""

import json
import numpy as np
import cv2
import pywt
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from tqdm import tqdm

from config import (
    ThreeStageConfigV5, TOTAL_FEAT_DIM,
    TIME_DOMAIN_DIM, STFT_BAND_DIM, PSD_BAND_DIM, HIGH_FREQ_DIM
)
from features import extract_zerone_features, split_feature_vector
from utils import GlobalNormalizer
from data_manager import read_vibration_csv, generate_sample_id


# =============================================================================
# Zerone图像生成
# =============================================================================

def _render_panel_raster(vec: np.ndarray, wrap: bool, tile_width: int, 
                         W_UNIT: int = 3, H_UNIT: int = 2) -> np.ndarray:
    """
    渲染单个面板为栅格图
    """
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
    """
    将归一化后的1200维特征向量转换为3通道图像 (raster-stripe布局)
    
    返回:
        (3, target_size, target_size) float32 [0,1]
    """
    W_UNIT, H_UNIT = 3, 2
    tile_widths = {"time": 16, "stft": 16, "psd": 32, "hf": 8}
    panel_order = ["time", "stft", "psd", "hf"]
    GAP_ROWS = 1
    
    # 切分特征
    segs = split_feature_vector(np.asarray(vec_norm01, dtype=np.float32))
    name2seg = {
        "time": segs.get("time", np.zeros(TIME_DOMAIN_DIM)),
        "stft": segs.get("stft", np.zeros(STFT_BAND_DIM)),
        "psd": segs.get("psd", np.zeros(PSD_BAND_DIM)),
        "hf": segs.get("hf", np.zeros(HIGH_FREQ_DIM)),
    }
    
    # 渲染每个面板
    panel_arrays = []
    panel_widths = []
    
    for name in panel_order:
        seg = name2seg[name]
        tile_width = tile_widths.get(name, 16)
        arr = _render_panel_raster(seg, wrap=True, tile_width=tile_width, 
                                   W_UNIT=W_UNIT, H_UNIT=H_UNIT)
        panel_arrays.append(arr)
        panel_widths.append(arr.shape[1])
    
    # 统一宽度
    max_w = max(panel_widths) if panel_widths else 1
    
    # 垂直拼接
    rows_list = []
    for idx, arr in enumerate(panel_arrays):
        if arr.shape[1] < max_w:
            pad_w = max_w - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode='constant', constant_values=0)
        rows_list.append(arr)
        if idx < len(panel_arrays) - 1 and GAP_ROWS > 0:
            gap = np.zeros((GAP_ROWS, max_w), dtype=np.float32)
            rows_list.append(gap)
    
    canvas = np.vstack(rows_list)
    
    # 应用colormap生成3通道
    try:
        from matplotlib import colormaps as cmaps
        cmap = cmaps.get_cmap('jet')
    except:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('jet')
    
    rgb = cmap(canvas)[:, :, :3]
    rgb_resized = cv2.resize(rgb.astype(np.float32), (target_size, target_size), 
                             interpolation=cv2.INTER_NEAREST)
    
    return np.transpose(rgb_resized, (2, 0, 1)).astype(np.float32)


def generate_hetero_image(signal: np.ndarray, size: int = 224) -> np.ndarray:
    """
    生成Hetero三通道时频图像 (CWT + STFT + Context)
    
    返回:
        (3, size, size) float32 [0,1]
    """
    # Channel 0: CWT
    try:
        scales = np.arange(1, 65)
        coeffs, _ = pywt.cwt(signal[:2048], scales, 'morl')
        cwt_img = np.abs(coeffs)
        cwt_img = cv2.resize(cwt_img, (size, size))
        cwt_min, cwt_max = cwt_img.min(), cwt_img.max()
        if cwt_max - cwt_min > 1e-8:
            cwt_img = (cwt_img - cwt_min) / (cwt_max - cwt_min + 1e-8)
        else:
            cwt_img = np.full((size, size), 0.5)
    except:
        cwt_img = np.full((size, size), 0.5)
    
    # Channel 1: STFT
    try:
        nperseg = 256
        hop = 64
        stft_matrix = []
        for i in range(0, len(signal) - nperseg, hop):
            segment = signal[i:i+nperseg]
            spectrum = np.abs(np.fft.rfft(segment * np.hanning(nperseg)))
            stft_matrix.append(spectrum)
        if stft_matrix:
            stft_img = np.array(stft_matrix).T
            stft_img = cv2.resize(stft_img, (size, size))
            stft_min, stft_max = stft_img.min(), stft_img.max()
            if stft_max - stft_min > 1e-8:
                stft_img = (stft_img - stft_min) / (stft_max - stft_min + 1e-8)
            else:
                stft_img = np.full((size, size), 0.5)
        else:
            stft_img = np.full((size, size), 0.5)
    except:
        stft_img = np.full((size, size), 0.5)
    
    # Channel 2: Context
    try:
        total = size * size
        if len(signal) >= total:
            context_img = signal[:total].reshape(size, size)
        else:
            padded = np.pad(signal, (0, total - len(signal)))
            context_img = padded.reshape(size, size)
        ctx_min, ctx_max = context_img.min(), context_img.max()
        if ctx_max - ctx_min > 1e-8:
            context_img = (context_img - ctx_min) / (ctx_max - ctx_min + 1e-8)
        else:
            context_img = np.full((size, size), 0.5)
    except:
        context_img = np.full((size, size), 0.5)
    
    return np.stack([cwt_img, stft_img, context_img], axis=0).astype(np.float32)


# =============================================================================
# JSONL格式数据集 (V5.11兼容)
# =============================================================================

class TransformerVibrationDataset(Dataset):
    """变压器振动数据集 (JSONL格式)"""
    
    def __init__(self, data_dir: Path, cfg: ThreeStageConfigV5, 
                 use_labels: bool = False, split_name: str = "UNKNOWN",
                 normalizer: GlobalNormalizer = None):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.use_labels = use_labels
        self.split_name = split_name
        self.normalizer = normalizer
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        if not self.data_dir.exists():
            print(f"[警告] 数据目录不存在: {self.data_dir}")
            return
        
        label_counts = Counter()
        
        for jsonl_file in self.data_dir.rglob("*.jsonl"):
            parent_name = jsonl_file.parent.name.lower()
            
            label = -1
            for class_name, keywords in self.cfg.CLASS_KEYWORDS.items():
                if any(kw in parent_name for kw in keywords):
                    label = 0 if class_name == "正常" else 1
                    break
            
            if self.use_labels:
                if label == -1:
                    continue
                label_counts[label] += 1
            
            self.samples.append({'file': jsonl_file, 'label': label})
        
        print(f"[{self.split_name}] 加载 {len(self.samples)} 个样本")
        if self.use_labels and label_counts:
            for lbl, cnt in sorted(label_counts.items()):
                name = "正常" if lbl == 0 else "故障"
                print(f"  {name}: {cnt}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        try:
            with open(sample['file'], 'r', encoding='utf-8') as f:
                data = json.loads(f.readline())
        except:
            data = {}
        
        # 读取signal_value
        raw_signal = data.get('signal_value', data.get('signal', None))
        if raw_signal is None:
            signal = np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
        elif isinstance(raw_signal, str):
            signal = np.array([float(x) for x in raw_signal.split(',')], dtype=np.float32)
        else:
            signal = np.array(raw_signal, dtype=np.float32)
        
        if len(signal) < self.cfg.SIGNAL_LEN:
            signal = np.pad(signal, (0, self.cfg.SIGNAL_LEN - len(signal)))
        else:
            signal = signal[:self.cfg.SIGNAL_LEN]
        
        # Hetero图像
        hetero_img = generate_hetero_image(signal, self.cfg.INPUT_SIZE)
        
        # Zerone特征
        zerone_feat = extract_zerone_features(signal, fs=self.cfg.FS)
        
        if self.normalizer is not None and self.normalizer.is_fitted:
            zerone_feat_norm = self.normalizer.transform(zerone_feat)
        else:
            zerone_feat_norm = np.clip(zerone_feat, 0, 1)
        
        if self.cfg.ZERONE_USE_CNN:
            zerone_img = vector_to_image_raster(zerone_feat_norm, target_size=self.cfg.INPUT_SIZE)
        else:
            zerone_img = zerone_feat_norm
        
        label = sample['label'] if sample['label'] != -1 else 0
        
        return (
            torch.from_numpy(hetero_img).float(),
            torch.from_numpy(zerone_img).float(),
            label,
            idx
        )
    
    def get_all_features_for_normalization(self) -> List[np.ndarray]:
        """提取所有样本的原始特征（用于计算全局归一化参数）"""
        features_list = []
        print(f"[{self.split_name}] 预扫描特征用于全局归一化...")
        
        for sample in tqdm(self.samples, desc="提取特征", leave=False):
            try:
                with open(sample['file'], 'r', encoding='utf-8') as f:
                    data = json.loads(f.readline())
                
                raw_signal = data.get('signal_value', data.get('signal', None))
                if raw_signal is None:
                    signal = np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
                elif isinstance(raw_signal, str):
                    signal = np.array([float(x) for x in raw_signal.split(',')], dtype=np.float32)
                else:
                    signal = np.array(raw_signal, dtype=np.float32)
                
                if len(signal) < self.cfg.SIGNAL_LEN:
                    signal = np.pad(signal, (0, self.cfg.SIGNAL_LEN - len(signal)))
                else:
                    signal = signal[:self.cfg.SIGNAL_LEN]
                
                feat = extract_zerone_features(signal, fs=self.cfg.FS)
                features_list.append(feat)
            except:
                features_list.append(np.zeros(TOTAL_FEAT_DIM, dtype=np.float32))
        
        return features_list


# =============================================================================
# CSV格式数据集 (V5.12新增)
# =============================================================================

class CSVVibrationDataset(Dataset):
    """
    CSV振动数据集 (V5.12新增)
    
    从原始CSV文件读取振动数据，支持:
    - 滑动窗口切分
    - 排除已标注样本
    - 通道名称映射
    """
    
    def __init__(self, csv_files: List[Path], cfg: ThreeStageConfigV5,
                 use_labels: bool = False, split_name: str = "TRAIN",
                 normalizer: GlobalNormalizer = None,
                 excluded_ids: Set[str] = None,
                 overlap: float = 0.5):
        self.cfg = cfg
        self.use_labels = use_labels
        self.split_name = split_name
        self.normalizer = normalizer
        self.excluded_ids = excluded_ids or set()
        self.overlap = overlap
        
        self.samples = []
        self._load_samples(csv_files)
    
    def _load_samples(self, csv_files: List[Path]):
        """从CSV文件加载样本"""
        hop = int(self.cfg.SIGNAL_LEN * (1 - self.overlap))
        excluded_count = 0
        
        for csv_path in tqdm(csv_files, desc=f"加载{self.split_name}数据"):
            csv_data = read_vibration_csv(csv_path, self.cfg)
            if csv_data is None:
                continue
            
            data = csv_data['data']
            n_samples, n_channels = data.shape
            
            for ch_idx in range(n_channels):
                channel_data = data[:, ch_idx]
                
                for start_idx in range(0, len(channel_data) - self.cfg.SIGNAL_LEN + 1, hop):
                    sample_id = generate_sample_id(csv_path, ch_idx, start_idx)
                    
                    if sample_id in self.excluded_ids:
                        excluded_count += 1
                        continue
                    
                    self.samples.append({
                        'signal': channel_data[start_idx:start_idx + self.cfg.SIGNAL_LEN].copy(),
                        'sample_id': sample_id,
                        'channel_name': csv_data['channel_names'][ch_idx],
                        'pseudo_name': csv_data['pseudo_names'][ch_idx],
                        'file_path': str(csv_path),
                        'label': -1,
                    })
        
        print(f"[{self.split_name}] 加载 {len(self.samples)} 个样本, 排除 {excluded_count} 个")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        signal = sample['signal']
        
        # Hetero图像
        hetero_img = generate_hetero_image(signal, self.cfg.INPUT_SIZE)
        
        # Zerone特征
        zerone_feat = extract_zerone_features(signal, fs=self.cfg.FS)
        
        if self.normalizer is not None and self.normalizer.is_fitted:
            zerone_feat_norm = self.normalizer.transform(zerone_feat)
        else:
            zerone_feat_norm = np.clip(zerone_feat, 0, 1)
        
        if self.cfg.ZERONE_USE_CNN:
            zerone_img = vector_to_image_raster(zerone_feat_norm, target_size=self.cfg.INPUT_SIZE)
        else:
            zerone_img = zerone_feat_norm
        
        label = sample['label'] if sample['label'] != -1 else 0
        
        return (
            torch.from_numpy(hetero_img).float(),
            torch.from_numpy(zerone_img).float(),
            label,
            idx
        )
    
    def get_all_features_for_normalization(self) -> List[np.ndarray]:
        """提取所有样本特征用于归一化"""
        features_list = []
        for sample in tqdm(self.samples, desc="提取特征", leave=False):
            feat = extract_zerone_features(sample['signal'], fs=self.cfg.FS)
            features_list.append(feat)
        return features_list


# =============================================================================
# 已标注数据集 (用于val/test)
# =============================================================================

class LabeledVibrationDataset(Dataset):
    """
    已标注振动数据集 (用于val/test)
    
    从DataSplitManager的样本列表加载
    """
    
    def __init__(self, samples: List[Dict], cfg: ThreeStageConfigV5,
                 split_name: str = "UNKNOWN", normalizer: GlobalNormalizer = None):
        self.samples = samples
        self.cfg = cfg
        self.split_name = split_name
        self.normalizer = normalizer
        
        self._preload_signals()
    
    def _preload_signals(self):
        """预加载信号数据"""
        print(f"[{self.split_name}] 预加载 {len(self.samples)} 个样本...")
        
        for sample in tqdm(self.samples, desc=f"加载{self.split_name}", leave=False):
            if 'signal' not in sample:
                try:
                    with open(sample['file_path'], 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if sample['line_num'] < len(lines):
                            data = json.loads(lines[sample['line_num']])
                            raw_signal = data.get('signal_value', data.get('signal', None))
                            
                            if raw_signal is None:
                                signal = np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
                            elif isinstance(raw_signal, str):
                                signal = np.array([float(x) for x in raw_signal.split(',')], dtype=np.float32)
                            else:
                                signal = np.array(raw_signal, dtype=np.float32)
                            
                            if len(signal) < self.cfg.SIGNAL_LEN:
                                signal = np.pad(signal, (0, self.cfg.SIGNAL_LEN - len(signal)))
                            else:
                                signal = signal[:self.cfg.SIGNAL_LEN]
                            
                            sample['signal'] = signal
                        else:
                            sample['signal'] = np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
                except Exception as e:
                    sample['signal'] = np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.samples[idx]
        signal = sample.get('signal', np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32))
        
        # Hetero图像
        hetero_img = generate_hetero_image(signal, self.cfg.INPUT_SIZE)
        
        # Zerone特征
        zerone_feat = extract_zerone_features(signal, fs=self.cfg.FS)
        
        if self.normalizer is not None and self.normalizer.is_fitted:
            zerone_feat_norm = self.normalizer.transform(zerone_feat)
        else:
            zerone_feat_norm = np.clip(zerone_feat, 0, 1)
        
        if self.cfg.ZERONE_USE_CNN:
            zerone_img = vector_to_image_raster(zerone_feat_norm, target_size=self.cfg.INPUT_SIZE)
        else:
            zerone_img = zerone_feat_norm
        
        label = sample['label']
        
        return (
            torch.from_numpy(hetero_img).float(),
            torch.from_numpy(zerone_img).float(),
            label,
            idx
        )
    
    def get_all_features_for_normalization(self) -> List[np.ndarray]:
        """提取所有样本特征用于归一化"""
        features_list = []
        for sample in tqdm(self.samples, desc="提取特征", leave=False):
            signal = sample.get('signal', np.zeros(self.cfg.SIGNAL_LEN, dtype=np.float32))
            feat = extract_zerone_features(signal, fs=self.cfg.FS)
            features_list.append(feat)
        return features_list

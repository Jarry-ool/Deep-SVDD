# -*- coding: utf-8 -*-
"""
features.py - 特征提取函数
==========================

1200维Zerone特征体系:
- 时域特征: 15维
- STFT特征: 127维  
- PSD特征: 1050维
- 高频特征: 8维
"""

import numpy as np
from scipy import signal as sig
from typing import Dict

from config import (
    TIME_DOMAIN_DIM, STFT_BAND_DIM, PSD_BAND_DIM, 
    HIGH_FREQ_DIM, TOTAL_FEAT_DIM, FEAT_SCHEMA
)


def compute_time_features(signal: np.ndarray) -> np.ndarray:
    """
    15维时域统计特征
    
    返回: [mean, rms, var, std, max, min, p2p, 
           kurtosis, skewness, zero_cross_rate, mean_abs,
           crest_factor, impulse_factor, margin_factor, waveform_factor]
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    N = x.size
    if N == 0:
        return np.zeros(TIME_DOMAIN_DIM, dtype=np.float32)
    
    # 基础统计
    mean_val = float(np.mean(x))
    rms_val = float(np.sqrt(np.mean(x ** 2)))
    var_val = float(np.var(x, ddof=0))
    std_val = float(np.std(x, ddof=0))
    max_val = float(np.max(x))
    min_val = float(np.min(x))
    p2p_val = max_val - min_val
    
    # 高阶矩
    xc = x - mean_val
    m2 = np.mean(xc ** 2)
    m3 = np.mean(xc ** 3)
    m4 = np.mean(xc ** 4)
    kurtosis_val = float(m4 / (m2 ** 2)) if m2 > 1e-12 else 0.0
    skewness_val = float(m3 / (std_val ** 3)) if std_val > 1e-12 else 0.0
    
    # 过零率
    sign_changes = np.sum(np.signbit(x[1:]) != np.signbit(x[:-1]))
    zero_cross_rate = float(sign_changes) / (N - 1) if N > 1 else 0.0
    
    # 其他指标
    mean_abs = float(np.mean(np.abs(x)))
    root4_mean = float(np.mean(np.abs(x) ** 4) ** 0.25) if np.any(x != 0) else 0.0
    
    crest_factor = max_val / rms_val if rms_val > 1e-12 else 0.0
    impulse_factor = max_val / mean_abs if mean_abs > 1e-12 else 0.0
    margin_factor = max_val / root4_mean if root4_mean > 1e-12 else 0.0
    waveform_factor = rms_val / mean_abs if mean_abs > 1e-12 else 0.0
    
    feats = np.array([
        mean_val, rms_val, var_val, std_val, max_val, min_val, p2p_val,
        kurtosis_val, skewness_val, zero_cross_rate, mean_abs,
        crest_factor, impulse_factor, margin_factor, waveform_factor
    ], dtype=np.float32)
    
    return feats[:TIME_DOMAIN_DIM]


def compute_stft_features(signal: np.ndarray, fs: float, 
                          nperseg: int = 128, noverlap: int = 64) -> np.ndarray:
    """
    127维STFT段均值特征 (去DC)
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    x_dc = x - np.mean(x)  # 去直流
    
    try:
        f, t, Zxx = sig.stft(x_dc, fs=fs, window='hann', 
                             nperseg=nperseg, noverlap=noverlap, boundary=None)
        mag = np.abs(Zxx[1:, :]) if Zxx.shape[0] > 1 else np.abs(Zxx)  # 去DC bin
        seg_means = np.mean(mag, axis=0)
        
        out = np.zeros(STFT_BAND_DIM, dtype=np.float32)
        L = min(seg_means.size, STFT_BAND_DIM)
        if L > 0:
            out[:L] = seg_means[:L]
        return out
    except Exception:
        return np.zeros(STFT_BAND_DIM, dtype=np.float32)


def compute_psd_features(signal: np.ndarray, fs: float, fmax: int = 4000) -> np.ndarray:
    """
    1050维PSD特征
    - 1-1000Hz: 1Hz栅格 (1000维)
    - 1001-2000Hz: 20Hz聚合 (50维)
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    x_dc = x - np.mean(x)  # 去直流
    
    try:
        nperseg = min(4096, len(x_dc) // 2) if len(x_dc) >= 2 else 1
        f_raw, Pxx_raw = sig.welch(
            x_dc, fs=fs, window='hann', nperseg=nperseg, 
            noverlap=nperseg // 2, nfft=len(x_dc), detrend=False, scaling='density'
        )
        
        # 插值到1Hz栅格 (1-fmax Hz)
        freqs_target = np.arange(1, fmax + 1, 1)
        mask = (f_raw >= 1) & (f_raw <= fmax)
        f_valid, P_valid = f_raw[mask], Pxx_raw[mask]
        
        if f_valid.size < 2:
            psd_full = np.zeros(fmax, dtype=np.float32)
        else:
            psd_full = np.interp(freqs_target, f_valid, P_valid).astype(np.float32)
        
        # 低频 1-1000Hz (1Hz栅格, 1000维)
        psd_low = psd_full[:1000] if psd_full.size >= 1000 else np.pad(psd_full, (0, 1000 - psd_full.size))[:1000]
        
        # 中频 1001-2000Hz 聚合为50段 (每20Hz一段)
        psd_mid = np.zeros(50, dtype=np.float32)
        for i in range(50):
            l = 1000 + i * 20
            r = 1000 + (i + 1) * 20
            if l < psd_full.size:
                seg = psd_full[l:min(r, psd_full.size)]
                psd_mid[i] = float(np.mean(seg)) if seg.size > 0 else 0.0
        
        return np.concatenate([psd_low, psd_mid], axis=0).astype(np.float32)
    
    except Exception:
        return np.zeros(PSD_BAND_DIM, dtype=np.float32)


def compute_high_freq_features(signal: np.ndarray, fs: float, fmax: int = 4000) -> np.ndarray:
    """
    8维高频特征
    4个阈值 × (幅值比 + 功率比)
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    
    try:
        # 计算完整PSD
        nperseg = min(4096, len(x) // 2) if len(x) >= 2 else 1
        f_raw, Pxx_raw = sig.welch(x - np.mean(x), fs=fs, window='hann', 
                                    nperseg=nperseg, noverlap=nperseg // 2)
        
        freqs_target = np.arange(1, fmax + 1, 1)
        mask = (f_raw >= 1) & (f_raw <= fmax)
        f_valid, P_valid = f_raw[mask], Pxx_raw[mask]
        
        if f_valid.size < 2:
            return np.zeros(HIGH_FREQ_DIM, dtype=np.float32)
        
        psd = np.interp(freqs_target, f_valid, P_valid)
        freqs = freqs_target
        
        hf_feats = []
        for threshold_hz in [1000.0, 2000.0, 3000.0, 4000.0]:
            low_mask = freqs < threshold_hz
            high_mask = freqs >= threshold_hz
            
            if np.any(low_mask) and np.any(high_mask):
                low_vals = psd[low_mask]
                high_vals = psd[high_mask]
                
                # 幅值比
                amp_ratio = float(np.mean(high_vals) / (np.mean(low_vals) + 1e-12))
                
                # 功率比
                total_power = float(np.trapz(psd, freqs))
                hf_power = float(np.trapz(high_vals, freqs[high_mask]))
                power_ratio = float(hf_power / (total_power + 1e-12))
            else:
                amp_ratio, power_ratio = 0.0, 0.0
            
            hf_feats.extend([amp_ratio, power_ratio])
        
        return np.array(hf_feats, dtype=np.float32)[:HIGH_FREQ_DIM]
    
    except Exception:
        return np.zeros(HIGH_FREQ_DIM, dtype=np.float32)


def extract_zerone_features(signal: np.ndarray, fs: float = 8192.0) -> np.ndarray:
    """
    提取完整1200维Zerone特征
    
    参数:
        signal: 振动信号
        fs: 采样率
    
    返回:
        (1200,) 特征向量
    """
    # 时域特征 (15维)
    time_feat = compute_time_features(signal)
    
    # STFT特征 (127维)
    stft_feat = compute_stft_features(signal, fs)
    
    # PSD特征 (1050维)
    psd_feat = compute_psd_features(signal, fs)
    
    # 高频特征 (8维)
    hf_feat = compute_high_freq_features(signal, fs)
    
    # 拼接
    features = np.concatenate([time_feat, stft_feat, psd_feat, hf_feat], axis=0)
    
    # 确保维度正确
    if features.size < TOTAL_FEAT_DIM:
        features = np.pad(features, (0, TOTAL_FEAT_DIM - features.size))
    else:
        features = features[:TOTAL_FEAT_DIM]
    
    # 修复nan/inf
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return features.astype(np.float32)


def split_feature_vector(vec: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将1200维特征向量按schema切分
    
    返回:
        {'time': (15,), 'stft': (127,), 'psd': (1050,), 'hf': (8,)}
    """
    v = np.asarray(vec).ravel()
    out = {}
    start = 0
    for name, length in FEAT_SCHEMA:
        end = start + int(length)
        if end > v.size:
            out[name] = v[start:].copy()
            break
        out[name] = v[start:end].copy()
        start = end
    return out

# -*- coding: utf-8 -*-
"""
gpu_transforms.py - GPU加速的信号变换 (V5.12)
=============================================

使用PyTorch在GPU上批量计算CWT和STFT，大幅提升hetero图像生成速度。

性能对比:
    CPU单样本: ~50ms (CWT 30ms + STFT 10ms + resize 10ms)
    GPU批量128: ~0.5ms/样本 (提速100倍)

使用方法:
    transform = GPUHeteroTransform(size=224, device='cuda')
    
    # 在训练循环中
    for signals, zerone_feat, labels, _ in dataloader:
        signals = signals.to(device)  # [B, L]
        hetero_img = transform(signals)  # [B, 3, 224, 224]
        output = model(hetero_img, zerone_feat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GPUHeteroTransform(nn.Module):
    """
    GPU批量生成Hetero三通道图像
    
    通道:
    - R: CWT时频图 (Morlet小波)
    - G: STFT时频图
    - B: 时域reshape
    
    Args:
        size: 输出图像大小 (默认224)
        n_scales: CWT尺度数 (默认64)
        signal_len: 信号长度 (默认8192)
        device: 计算设备
    """
    
    def __init__(self, size: int = 224, n_scales: int = 64, 
                 signal_len: int = 8192, device: str = 'cuda'):
        super().__init__()
        self.size = size
        self.n_scales = n_scales
        self.signal_len = signal_len
        self.device = torch.device(device)
        
        # STFT参数
        self.n_fft = 256
        self.hop_length = 64
        self.window = nn.Parameter(
            torch.hann_window(self.n_fft), requires_grad=False
        )
        
        # 预计算Morlet小波核
        self._init_wavelet_kernels()
    
    def _init_wavelet_kernels(self):
        """
        预计算Morlet小波核用于conv1d
        
        Morlet小波: ψ(t) = exp(-t²/2) * exp(i*ω₀*t)
        """
        scales = np.arange(1, self.n_scales + 1)
        
        # 小波参数
        omega0 = 6.0  # 中心频率
        
        kernels_real = []
        kernels_imag = []
        
        for scale in scales:
            # 核长度随尺度变化
            kernel_len = min(int(10 * scale), self.signal_len // 2)
            if kernel_len % 2 == 0:
                kernel_len += 1
            
            t = np.arange(-kernel_len // 2, kernel_len // 2 + 1) / scale
            
            # Morlet小波
            gaussian = np.exp(-t**2 / 2)
            wavelet_real = gaussian * np.cos(omega0 * t)
            wavelet_imag = gaussian * np.sin(omega0 * t)
            
            # 归一化
            norm = np.sqrt(scale)
            wavelet_real = wavelet_real / norm
            wavelet_imag = wavelet_imag / norm
            
            kernels_real.append(torch.tensor(wavelet_real, dtype=torch.float32))
            kernels_imag.append(torch.tensor(wavelet_imag, dtype=torch.float32))
        
        self.wavelet_kernels_real = kernels_real
        self.wavelet_kernels_imag = kernels_imag
        self.scales = scales
    
    def cwt_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        批量CWT计算 (GPU)
        
        Args:
            signals: [B, L] 输入信号
        
        Returns:
            [B, size, size] CWT时频图
        """
        B, L = signals.shape
        device = signals.device
        
        # 存储所有尺度的结果
        cwt_results = []
        
        for i, (kernel_r, kernel_i) in enumerate(zip(
            self.wavelet_kernels_real, self.wavelet_kernels_imag
        )):
            kernel_r = kernel_r.to(device)
            kernel_i = kernel_i.to(device)
            
            # 准备卷积核 [out_channels=1, in_channels=1, kernel_size]
            kernel_r = kernel_r.view(1, 1, -1)
            kernel_i = kernel_i.view(1, 1, -1)
            
            # 卷积计算
            signals_3d = signals.unsqueeze(1)  # [B, 1, L]
            
            # padding保持长度
            pad = kernel_r.shape[-1] // 2
            
            conv_r = F.conv1d(signals_3d, kernel_r, padding=pad)
            conv_i = F.conv1d(signals_3d, kernel_i, padding=pad)
            
            # 取模
            magnitude = torch.sqrt(conv_r**2 + conv_i**2 + 1e-8)
            
            # 裁剪到原始长度
            magnitude = magnitude[:, 0, :L]  # [B, L]
            
            cwt_results.append(magnitude)
        
        # 堆叠: [B, n_scales, L]
        cwt_matrix = torch.stack(cwt_results, dim=1)
        
        # resize到目标大小: [B, n_scales, L] -> [B, size, size]
        cwt_matrix = cwt_matrix.unsqueeze(1)  # [B, 1, n_scales, L]
        cwt_img = F.interpolate(cwt_matrix, size=(self.size, self.size), 
                                mode='bilinear', align_corners=False)
        cwt_img = cwt_img.squeeze(1)  # [B, size, size]
        
        # 归一化到[0,1]
        cwt_img = self._normalize_batch(cwt_img)
        
        return cwt_img
    
    def stft_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        批量STFT计算 (GPU)
        
        Args:
            signals: [B, L] 输入信号
        
        Returns:
            [B, size, size] STFT时频图
        """
        B, L = signals.shape
        device = signals.device
        
        # 确保window在正确设备上
        window = self.window.to(device)
        
        # torch.stft返回复数
        stft_complex = torch.stft(
            signals, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # 取幅度: [B, n_fft//2+1, time_frames]
        stft_mag = torch.abs(stft_complex)
        
        # resize到目标大小
        stft_mag = stft_mag.unsqueeze(1)  # [B, 1, freq, time]
        stft_img = F.interpolate(stft_mag, size=(self.size, self.size),
                                 mode='bilinear', align_corners=False)
        stft_img = stft_img.squeeze(1)  # [B, size, size]
        
        # 归一化到[0,1]
        stft_img = self._normalize_batch(stft_img)
        
        return stft_img
    
    def ctx_batch(self, signals: torch.Tensor) -> torch.Tensor:
        """
        批量时域reshape (GPU)
        
        Args:
            signals: [B, L] 输入信号
        
        Returns:
            [B, size, size] 时域图
        """
        B, L = signals.shape
        total_pixels = self.size * self.size
        
        if L >= total_pixels:
            ctx = signals[:, :total_pixels]
        else:
            # padding
            ctx = F.pad(signals, (0, total_pixels - L))
        
        ctx_img = ctx.view(B, self.size, self.size)
        
        # 归一化到[0,1]
        ctx_img = self._normalize_batch(ctx_img)
        
        return ctx_img
    
    def _normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量归一化到[0,1]，处理nan/inf
        
        Args:
            x: [B, H, W]
        
        Returns:
            归一化后的tensor
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)
        
        # 处理nan和inf
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=1.0, neginf=0.0)
        
        x_min = x_flat.min(dim=1, keepdim=True)[0]
        x_max = x_flat.max(dim=1, keepdim=True)[0]
        
        # 避免除零
        x_range = x_max - x_min
        x_range = torch.where(x_range < 1e-8, torch.ones_like(x_range), x_range)
        
        x_norm = (x_flat - x_min) / x_range
        
        # 再次确保输出在[0,1]范围内
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        return x_norm.view(B, self.size, self.size)
    
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        批量生成Hetero三通道图像
        
        Args:
            signals: [B, L] 输入信号 (已在GPU上)
        
        Returns:
            [B, 3, size, size] Hetero图像 (R=CWT, G=STFT, B=Context)
        """
        # 输入检查：处理nan/inf
        signals = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        
        cwt_img = self.cwt_batch(signals)    # [B, size, size]
        stft_img = self.stft_batch(signals)  # [B, size, size]
        ctx_img = self.ctx_batch(signals)    # [B, size, size]
        
        # 堆叠为3通道
        hetero_img = torch.stack([cwt_img, stft_img, ctx_img], dim=1)
        
        # 最终检查：确保无nan/inf
        hetero_img = torch.nan_to_num(hetero_img, nan=0.5, posinf=1.0, neginf=0.0)
        
        return hetero_img  # [B, 3, size, size]
    
    @torch.no_grad()
    def transform_numpy(self, signals: np.ndarray) -> np.ndarray:
        """
        NumPy接口 (用于可视化等场景)
        
        Args:
            signals: [B, L] 或 [L] NumPy数组
        
        Returns:
            [B, 3, size, size] 或 [3, size, size] NumPy数组
        """
        single = False
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
            single = True
        
        signals_tensor = torch.from_numpy(signals.astype(np.float32)).to(self.device)
        hetero_img = self.forward(signals_tensor)
        result = hetero_img.cpu().numpy()
        
        if single:
            result = result[0]
        
        return result


class GPUZeroneTransform(nn.Module):
    """
    GPU批量计算Zerone特征 (可选，如果需要GPU加速zerone)
    
    注意: 通常zerone特征用缓存即可，不需要GPU加速
    """
    pass  # 预留接口


# =============================================================================
# 便捷函数
# =============================================================================

_GPU_TRANSFORM = None

def get_gpu_transform(size: int = 224, device: str = 'cuda') -> GPUHeteroTransform:
    """
    获取GPU变换器单例
    """
    global _GPU_TRANSFORM
    if _GPU_TRANSFORM is None or _GPU_TRANSFORM.device != torch.device(device):
        _GPU_TRANSFORM = GPUHeteroTransform(size=size, device=device)
    return _GPU_TRANSFORM


def generate_hetero_image_gpu(signals: torch.Tensor, size: int = 224) -> torch.Tensor:
    """
    GPU生成Hetero图像 (便捷函数)
    
    Args:
        signals: [B, L] GPU tensor
        size: 输出大小
    
    Returns:
        [B, 3, size, size] GPU tensor
    """
    transform = get_gpu_transform(size=size, device=signals.device.type)
    return transform(signals)

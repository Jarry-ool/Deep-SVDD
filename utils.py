# -*- coding: utf-8 -*-
"""
utils.py - 工具函数
==================

包含:
- GlobalNormalizer: 全局归一化管理器
- TrainingLogger: 训练日志记录器
- CheckpointManager: 检查点管理器
"""

import csv
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from config import TOTAL_FEAT_DIM


# =============================================================================
# 全局归一化管理器
# =============================================================================

class GlobalNormalizer:
    """
    全局归一化管理器
    
    在Stage1开始前预扫描train集，计算：
    - Zerone特征的全局min/max (1200维)
    
    所有样本用这个统计量归一化，避免逐样本归一化导致的信息丢失
    """
    
    def __init__(self):
        self.zerone_min = None  # (1200,)
        self.zerone_max = None  # (1200,)
        self.is_fitted = False
    
    def fit(self, features_list: List[np.ndarray]):
        """
        从特征列表计算全局统计量
        
        参数:
            features_list: List of (1200,) 特征向量
        """
        if len(features_list) == 0:
            print("[GlobalNormalizer] 警告: 空特征列表，使用默认值")
            self.zerone_min = np.zeros(TOTAL_FEAT_DIM)
            self.zerone_max = np.ones(TOTAL_FEAT_DIM)
            self.is_fitted = True
            return
        
        # 堆叠为矩阵 (N, 1200)
        feat_matrix = np.stack(features_list, axis=0)
        
        # 处理nan/inf
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 按列计算min/max
        self.zerone_min = np.min(feat_matrix, axis=0)
        self.zerone_max = np.max(feat_matrix, axis=0)
        
        # 处理常数列 (max == min)
        constant_mask = (self.zerone_max - self.zerone_min) < 1e-8
        self.zerone_max[constant_mask] = self.zerone_min[constant_mask] + 1.0
        
        self.is_fitted = True
        print(f"[GlobalNormalizer] 已拟合，特征范围: "
              f"min={self.zerone_min.min():.4f}~{self.zerone_min.max():.4f}, "
              f"max={self.zerone_max.min():.4f}~{self.zerone_max.max():.4f}")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征向量到[0,1]
        
        参数:
            features: (1200,) 或 (N, 1200)
        返回:
            归一化后的特征
        """
        if not self.is_fitted:
            raise RuntimeError("GlobalNormalizer未拟合，请先调用fit()")
        
        features = np.asarray(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        is_1d = (features.ndim == 1)
        if is_1d:
            features = features.reshape(1, -1)
        
        # 归一化
        rng = self.zerone_max - self.zerone_min + 1e-8
        normalized = (features - self.zerone_min) / rng
        normalized = np.clip(normalized, 0.0, 1.0)
        
        if is_1d:
            return normalized.squeeze(0)
        return normalized
    
    def save(self, path: Path):
        """保存归一化参数"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, 
                 zerone_min=self.zerone_min, 
                 zerone_max=self.zerone_max,
                 is_fitted=self.is_fitted)
        print(f"[GlobalNormalizer] 参数已保存: {path}")
    
    def load(self, path: Path):
        """加载归一化参数"""
        path = Path(path)
        data = np.load(path)
        self.zerone_min = data['zerone_min']
        self.zerone_max = data['zerone_max']
        self.is_fitted = bool(data['is_fitted'])
        print(f"[GlobalNormalizer] 参数已加载: {path}")


# 全局实例
GLOBAL_NORMALIZER = GlobalNormalizer()


# =============================================================================
# 训练日志记录器
# =============================================================================

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: Path, stage: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.log_file = self.log_dir / f"{stage}_training_log.csv"
        self.records = []
    
    def log(self, **kwargs):
        """记录一条训练日志"""
        kwargs['timestamp'] = datetime.now().isoformat()
        self.records.append(kwargs)
    
    def save_csv(self):
        """保存为CSV文件"""
        if not self.records:
            return
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[日志] 训练日志已保存: {self.log_file}")
    
    def get_history(self, key: str) -> List[Any]:
        """获取某个指标的历史值"""
        return [r.get(key) for r in self.records if key in r]


# =============================================================================
# 检查点管理器
# =============================================================================

class CheckpointManager:
    """模型检查点管理器"""
    
    def __init__(self, ckpt_dir: Path, stage: str, max_keep: int = 5):
        self.ckpt_dir = Path(ckpt_dir) / stage
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.max_keep = max_keep
    
    def save(self, model, optimizer, epoch: int, metrics: Dict, 
             scheduler=None, extra: Dict = None):
        """保存检查点"""
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }
        if scheduler:
            ckpt['scheduler_state'] = scheduler.state_dict()
        if hasattr(model, 'center') and model.center is not None:
            ckpt['center'] = model.center.cpu() if isinstance(model.center, torch.Tensor) else model.center
        if extra:
            ckpt.update(extra)
        
        path = self.ckpt_dir / f"checkpoint_epoch{epoch:03d}.pth"
        torch.save(ckpt, path)
        print(f"[检查点] 已保存: {path.name}")
        self._cleanup()
        return path
    
    def save_best(self, model, metrics: Dict, name: str = "best"):
        """保存最佳模型"""
        path = self.ckpt_dir / f"{name}_model.pth"
        ckpt = {
            'model_state': model.state_dict(),
            'metrics': metrics,
        }
        if hasattr(model, 'center') and model.center is not None:
            ckpt['center'] = model.center.cpu() if isinstance(model.center, torch.Tensor) else model.center
        torch.save(ckpt, path)
        print(f"[检查点] 最佳模型已保存: {path.name}")
        return path
    
    def _cleanup(self):
        """清理旧的检查点"""
        ckpts = sorted(self.ckpt_dir.glob("checkpoint_epoch*.pth"))
        while len(ckpts) > self.max_keep:
            ckpts[0].unlink()
            ckpts = ckpts[1:]
    
    def get_latest(self) -> Optional[Path]:
        """获取最新的检查点"""
        ckpts = sorted(self.ckpt_dir.glob("checkpoint_epoch*.pth"))
        return ckpts[-1] if ckpts else None
    
    def load(self, path: Path, model, optimizer=None, scheduler=None) -> Dict:
        """加载检查点"""
        path = Path(path)
        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        if optimizer and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler and 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if hasattr(model, 'center') and 'center' in ckpt:
            model.center = ckpt['center']
        print(f"[检查点] 已加载: {path.name}, epoch={ckpt.get('epoch', 'N/A')}")
        return ckpt


# =============================================================================
# 辅助函数
# =============================================================================

def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict, path: Path):
    """保存JSON文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Dict:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False

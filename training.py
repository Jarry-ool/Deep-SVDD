# -*- coding: utf-8 -*-
"""
training.py - 训练函数
=====================

包含三阶段训练的完整实现:
- train_stage1: 无监督学习 (SVDD + VAE)
- run_stage2: 伪标签生成
- train_stage3: 有监督微调
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)

from config import ThreeStageConfigV5, TOTAL_FEAT_DIM
from features import extract_zerone_features
from utils import (
    GlobalNormalizer, GLOBAL_NORMALIZER, 
    TrainingLogger, CheckpointManager, EarlyStopping
)
from models import (
    AnomalyModelV5, FaultClassifierV5,
    compute_mmd_loss, compute_coral_loss
)
from datasets import (
    TransformerVibrationDataset, CSVVibrationDataset, LabeledVibrationDataset,
    generate_hetero_image, vector_to_image_raster
)
from data_manager import DataSplitManager, CHANNEL_MANAGER, scan_csv_files
from visualization import VisualizationManager


# =============================================================================
# 可视化辅助函数
# =============================================================================

def _visualize_vae_reconstruction(model, dataloader, device, viz: VisualizationManager, 
                                   cfg: ThreeStageConfigV5):
    """VAE重建可视化"""
    model.eval()
    originals, recons = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            out = model(img, zr)
            
            if 'recon' in out:
                originals.extend(img.cpu().numpy())
                recons.extend(out['recon'].cpu().numpy())
            
            if len(originals) >= 4:
                break
    
    if originals and recons:
        for lang in cfg.LANGS:
            viz.plot_reconstruction(np.array(originals[:4]), np.array(recons[:4]), 
                                   n_samples=4, lang=lang)


def _generate_sample_previews(train_ds: Dataset, cfg: ThreeStageConfigV5,
                               viz: VisualizationManager):
    """生成样本预览（Hetero + Zerone双分支）"""
    print("\n[*] 生成样本预览 (Hetero + Zerone)...")
    
    preview_hetero, preview_zerone, preview_labels = [], [], []
    n_preview = min(cfg.SAMPLE_PREVIEW_COUNT, len(train_ds))
    
    for i in range(n_preview):
        hetero, zerone, lbl, _ = train_ds[i]
        preview_hetero.append(hetero.numpy() if hasattr(hetero, 'numpy') else hetero)
        preview_zerone.append(zerone.numpy() if hasattr(zerone, 'numpy') else zerone)
        preview_labels.append(lbl)
    
    for lang in cfg.LANGS:
        # Hetero预览
        viz.plot_sample_preview(preview_hetero, preview_labels, lang=lang, prefix="hetero_samples")
        # Zerone预览
        viz.plot_zerone_preview(preview_zerone, preview_labels, lang=lang)


def _generate_compare_preview(cfg: ThreeStageConfigV5, normalizer: GlobalNormalizer,
                               viz: VisualizationManager):
    """
    V5.1: 从VAL目录生成正常vs故障对比预览图（中英文各7张）
    """
    import pywt
    import cv2
    
    print("\n[*] 生成正常vs故障对比预览...")
    print(f"  数据源: {cfg.VAL_DIR}")
    
    # 收集正常和故障样本文件
    normal_files = []
    fault_files = []
    
    if not cfg.VAL_DIR.exists():
        print("  [警告] VAL目录不存在，跳过对比预览")
        return
    
    for jsonl_file in cfg.VAL_DIR.rglob("*.jsonl"):
        parent_name = jsonl_file.parent.name.lower()
        if any(kw in parent_name for kw in ["正常", "normal", "健康"]):
            normal_files.append(jsonl_file)
        elif any(kw in parent_name for kw in ["故障", "异常", "fault", "abnormal"]):
            fault_files.append(jsonl_file)
    
    print(f"  找到正常样本: {len(normal_files)}")
    print(f"  找到故障样本: {len(fault_files)}")
    
    if len(normal_files) == 0 or len(fault_files) == 0:
        print("  [警告] VAL目录中没有找到足够的正常/故障样本，跳过对比预览")
        return
    
    # 随机抽取
    np.random.seed(42)
    n_samples = 4
    normal_selected = list(np.random.choice(normal_files, min(n_samples, len(normal_files)), replace=False))
    fault_selected = list(np.random.choice(fault_files, min(n_samples, len(fault_files)), replace=False))
    
    # 辅助函数：加载信号
    def _load_signal(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
        raw = data.get('signal_value', data.get('signal', None))
        if raw is None:
            return np.zeros(cfg.SIGNAL_LEN, dtype=np.float32)
        elif isinstance(raw, str):
            return np.array([float(x) for x in raw.split(',')], dtype=np.float32)
        return np.array(raw, dtype=np.float32)
    
    # 辅助函数：生成Hetero图像
    def _gen_hetero(signal, size=224):
        try:
            scales = np.arange(1, 65)
            coeffs, _ = pywt.cwt(signal[:2048], scales, 'morl')
            cwt_img = np.abs(coeffs)
            cwt_img = cv2.resize(cwt_img, (size, size))
            cmin, cmax = cwt_img.min(), cwt_img.max()
            cwt_img = (cwt_img - cmin) / (cmax - cmin + 1e-8) if cmax - cmin > 1e-8 else np.full((size, size), 0.5)
        except:
            cwt_img = np.full((size, size), 0.5)
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
        try:
            total = size * size
            ctx = signal[:total] if len(signal) >= total else np.pad(signal, (0, total - len(signal)))
            ctx_img = ctx.reshape(size, size)
            ctmin, ctmax = ctx_img.min(), ctx_img.max()
            ctx_img = (ctx_img - ctmin) / (ctmax - ctmin + 1e-8) if ctmax - ctmin > 1e-8 else np.full((size, size), 0.5)
        except:
            ctx_img = np.full((size, size), 0.5)
        return np.stack([cwt_img, stft_img, ctx_img], axis=0).astype(np.float32)
    
    # 预加载所有图像数据
    normal_zerone_imgs, normal_hetero_imgs = [], []
    fault_zerone_imgs, fault_hetero_imgs = [], []
    normal_feats, fault_feats = [], []
    
    for f in normal_selected:
        signal = _load_signal(f)
        feat = extract_zerone_features(signal, fs=cfg.FS)
        feat_norm = normalizer.transform(feat)
        normal_feats.append(feat)
        normal_zerone_imgs.append(vector_to_image_raster(feat_norm, target_size=cfg.INPUT_SIZE))
        normal_hetero_imgs.append(_gen_hetero(signal, size=cfg.INPUT_SIZE))
    
    for f in fault_selected:
        signal = _load_signal(f)
        feat = extract_zerone_features(signal, fs=cfg.FS)
        feat_norm = normalizer.transform(feat)
        fault_feats.append(feat)
        fault_zerone_imgs.append(vector_to_image_raster(feat_norm, target_size=cfg.INPUT_SIZE))
        fault_hetero_imgs.append(_gen_hetero(signal, size=cfg.INPUT_SIZE))
    
    # 生成对比预览（中英文各7张）
    for lang in cfg.LANGS:
        viz.plot_normal_vs_fault_compare(
            normal_zerone_imgs, fault_zerone_imgs,
            normal_hetero_imgs, fault_hetero_imgs,
            np.array(normal_feats) if normal_feats else None,
            np.array(fault_feats) if fault_feats else None,
            lang=lang
        )
    
    print(f"  ✅ 对比预览生成完成！共14张图 (中英文各7张)")


# =============================================================================
# 阶段一：无监督学习
# =============================================================================

def train_stage1(cfg: ThreeStageConfigV5, 
                 train_ds: Dataset = None,
                 resume_from: Path = None) -> Tuple[AnomalyModelV5, Dict]:
    """
    阶段一：无监督学习 (SVDD + VAE)
    
    参数:
        cfg: 配置对象
        train_ds: 训练数据集 (可选，如果不提供则从cfg.TRAIN_DIR加载)
        resume_from: 恢复检查点路径
    
    返回:
        (model, history)
    """
    print("\n" + "="*70)
    print("阶段一：无监督学习 (V5.12)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    logger = TrainingLogger(cfg.STAGE1_DIR, "stage1")
    ckpt_mgr = CheckpointManager(cfg.MODEL_DIR, "stage1")
    
    # ========== 准备数据集 ==========
    if train_ds is None:
        print("\n[1/5] 计算全局归一化参数...")
        temp_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=False, 
            split_name="TRAIN", normalizer=None
        )
        
        if len(temp_ds) == 0:
            print("[错误] TRAIN数据集为空!")
            return None, {}
        
        all_features = temp_ds.get_all_features_for_normalization()
        GLOBAL_NORMALIZER.fit(all_features)
        GLOBAL_NORMALIZER.save(cfg.MODEL_DIR / "global_normalizer.npz")
        
        train_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=False,
            split_name="TRAIN", normalizer=GLOBAL_NORMALIZER
        )
    else:
        # 使用提供的数据集，尝试计算归一化参数
        if hasattr(train_ds, 'get_all_features_for_normalization'):
            print("\n[1/5] 计算全局归一化参数...")
            all_features = train_ds.get_all_features_for_normalization()
            if len(all_features) > 0:
                GLOBAL_NORMALIZER.fit(all_features[:5000])  # 限制采样数量
                GLOBAL_NORMALIZER.save(cfg.MODEL_DIR / "global_normalizer.npz")
    
    print(f"\n[2/5] 训练数据集大小: {len(train_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )
    
    # ========== 创建可视化管理器 ==========
    viz = VisualizationManager(cfg.STAGE1_DIR)
    
    # ========== 样本预览（Hetero + Zerone双分支）==========
    _generate_sample_previews(train_ds, cfg, viz)
    
    # ========== V5.1: 正常vs故障对比预览（从VAL目录）==========
    _generate_compare_preview(cfg, GLOBAL_NORMALIZER, viz)
    
    # ========== 构建模型 ==========
    print("\n[3/5] 构建模型...")
    model = AnomalyModelV5(
        branch_mode=cfg.BRANCH_MODE,
        fusion_mode=cfg.FUSION_MODE,
        zerone_use_cnn=cfg.ZERONE_USE_CNN,
        use_modality_dropout=cfg.USE_MODALITY_DROPOUT,
        modality_dropout_p=cfg.MODALITY_DROPOUT_RATE,
        dropout_rate=cfg.DROPOUT_RATE
    ).to(device)
    
    print(f"  支线模式: {cfg.BRANCH_MODE}")
    print(f"  融合策略: {cfg.FUSION_MODE}")
    print(f"  Zerone架构: {'CNN' if cfg.ZERONE_USE_CNN else 'MLP'}")
    print(f"  模态Dropout: {'✅' if cfg.USE_MODALITY_DROPOUT else '❌'}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE1_EPOCHS)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # 恢复训练
    if resume_from and Path(resume_from).exists():
        print(f"\n[恢复] 从检查点加载: {resume_from}")
        ckpt = ckpt_mgr.load(resume_from, model, optimizer, scheduler)
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('metrics', {}).get('total_loss', float('inf'))
    
    # VAE预训练
    if model.has_vae and start_epoch == 0:
        print("\n[4/5] VAE预训练 (5轮)...")
        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"VAE预训练 {epoch+1}/5", leave=False):
                img, zr, _, _ = batch
                img, zr = img.to(device), zr.to(device)
                
                out = model(img, zr)
                loss = out['vae_recon_loss'].mean() + 0.01 * out['vae_kl'].mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  VAE预训练 Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # VAE重建可视化
        _visualize_vae_reconstruction(model, train_loader, device, viz, cfg)
    
    # SVDD中心初始化
    if start_epoch == 0:
        print("\n[*] 初始化SVDD中心...")
        model.init_center(train_loader, device)
    
    # ========== 联合训练 ==========
    print(f"\n[5/5] 联合训练 ({cfg.STAGE1_EPOCHS}轮)...")
    
    history = {'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': []}
    
    for epoch in range(start_epoch, cfg.STAGE1_EPOCHS):
        model.train()
        epoch_svdd, epoch_vae, epoch_total = 0, 0, 0
        
        beta = min(1.0, 1.0 * (epoch / max(10, 1)))  # KL权重warmup
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False)
        for batch in pbar:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            
            svdd_loss = out['svdd_score'].mean()
            
            if model.has_vae:
                vae_loss = out['vae_recon_loss'].mean() + beta * 0.01 * out['vae_kl'].mean()
                total_loss = 0.5 * svdd_loss + 0.5 * vae_loss
            else:
                vae_loss = torch.tensor(0.0)
                total_loss = svdd_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_svdd += svdd_loss.item()
            epoch_vae += vae_loss.item()
            epoch_total += total_loss.item()
            
            pbar.set_postfix({'svdd': f"{svdd_loss.item():.4f}", 'vae': f"{vae_loss.item():.4f}"})
        
        scheduler.step()
        
        n_batches = len(train_loader)
        avg_svdd = epoch_svdd / n_batches
        avg_vae = epoch_vae / n_batches
        avg_total = epoch_total / n_batches
        
        history['epoch'].append(epoch + 1)
        history['svdd_loss'].append(avg_svdd)
        history['vae_loss'].append(avg_vae)
        history['total_loss'].append(avg_total)
        
        logger.log(epoch=epoch+1, svdd_loss=avg_svdd, vae_loss=avg_vae, total_loss=avg_total)
        
        if avg_total < best_loss:
            best_loss = avg_total
            ckpt_mgr.save_best(model, {'total_loss': best_loss}, name="stage1_best")
        
        if (epoch + 1) % 5 == 0:
            ckpt_mgr.save(model, optimizer, epoch + 1,
                         {'svdd_loss': avg_svdd, 'vae_loss': avg_vae, 'total_loss': avg_total},
                         scheduler)
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
    
    logger.save_csv()
    
    # ========== 绘制训练曲线 ==========
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage1", lang=lang)
    
    print(f"\n【阶段一完成】最佳损失: {best_loss:.4f}")
    
    return model, history


# =============================================================================
# 阶段二：伪标签生成
# =============================================================================

def run_stage2(model: AnomalyModelV5, cfg: ThreeStageConfigV5,
               train_ds: Dataset = None) -> Dict:
    """
    阶段二：基于异常得分生成伪标签
    
    参数:
        model: 阶段一训练的模型
        cfg: 配置对象
        train_ds: 训练数据集
    
    返回:
        pseudo_labels字典
    """
    print("\n" + "="*70)
    print("阶段二：伪标签生成")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    # 加载归一化器
    normalizer_path = cfg.MODEL_DIR / "global_normalizer.npz"
    if normalizer_path.exists():
        GLOBAL_NORMALIZER.load(normalizer_path)
    
    # 准备数据
    if train_ds is None:
        train_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=False,
            split_name="TRAIN", normalizer=GLOBAL_NORMALIZER
        )
    
    loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\n[1/2] 计算异常得分...")
    model.eval()
    all_scores = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="计算得分", leave=False):
            img, zr, _, idx = batch
            img, zr = img.to(device), zr.to(device)
            scores = model.anomaly_score(img, zr)
            all_scores.extend(scores.cpu().tolist())
            all_indices.extend(idx.tolist())
    
    all_scores = np.array(all_scores)
    all_indices = np.array(all_indices)
    
    print("\n[2/2] 生成伪标签...")
    # 使用分位数确定阈值
    normal_percentile = cfg.QUANTILE_LOW * 100
    anomaly_percentile = cfg.QUANTILE_HIGH * 100
    
    t_normal = np.percentile(all_scores, normal_percentile)
    t_anomaly = np.percentile(all_scores, anomaly_percentile)
    
    pseudo_normal = all_indices[all_scores <= t_normal]
    pseudo_anomaly = all_indices[all_scores >= t_anomaly]
    uncertain = all_indices[(all_scores > t_normal) & (all_scores < t_anomaly)]
    
    print(f"  正常阈值 (P{normal_percentile:.0f}): {t_normal:.4f}")
    print(f"  异常阈值 (P{anomaly_percentile:.0f}): {t_anomaly:.4f}")
    print(f"  伪标签分布:")
    print(f"    高置信正常: {len(pseudo_normal)} ({100*len(pseudo_normal)/len(all_scores):.1f}%)")
    print(f"    高置信异常: {len(pseudo_anomaly)} ({100*len(pseudo_anomaly)/len(all_scores):.1f}%)")
    print(f"    不确定: {len(uncertain)} ({100*len(uncertain)/len(all_scores):.1f}%)")
    
    pseudo_labels = {
        'all_scores': all_scores,
        'all_indices': all_indices,
        't_normal': t_normal,
        't_anomaly': t_anomaly,
        'pseudo_normal': pseudo_normal,
        'pseudo_anomaly': pseudo_anomaly,
        'uncertain': uncertain,
    }
    
    np.savez(cfg.STAGE2_DIR / "pseudo_labels.npz", **pseudo_labels)
    
    # ========== 绘制得分分布 ==========
    viz = VisualizationManager(cfg.STAGE2_DIR)
    for lang in cfg.LANGS:
        viz.plot_score_distribution(all_scores, t_normal, t_anomaly, lang=lang)
    
    print(f"\n【阶段二完成】伪标签保存: {cfg.STAGE2_DIR / 'pseudo_labels.npz'}")
    
    return pseudo_labels


# =============================================================================
# 阶段三：有监督微调
# =============================================================================

def train_stage3(model: AnomalyModelV5, pseudo_labels: Dict, cfg: ThreeStageConfigV5,
                 val_ds: Dataset = None, test_ds: Dataset = None,
                 train_ds: Dataset = None) -> FaultClassifierV5:
    """
    阶段三：有监督微调
    
    参数:
        model: 阶段一训练的模型
        pseudo_labels: 阶段二生成的伪标签
        cfg: 配置对象
        val_ds: 验证数据集
        test_ds: 测试数据集
        train_ds: 原始训练数据集 (用于构建伪标签数据)
    
    返回:
        classifier模型
    """
    print("\n" + "="*70)
    print("阶段三：有监督微调 (V5.12)")
    print("="*70)
    
    device = torch.device(cfg.DEVICE)
    
    logger = TrainingLogger(cfg.STAGE3_DIR, "stage3")
    ckpt_mgr = CheckpointManager(cfg.MODEL_DIR, "stage3")
    
    # 加载归一化器
    normalizer_path = cfg.MODEL_DIR / "global_normalizer.npz"
    if normalizer_path.exists():
        GLOBAL_NORMALIZER.load(normalizer_path)
    
    # ========== 准备数据集 ==========
    print("\n[1/4] 准备数据...")
    
    # 从伪标签构建训练数据
    pseudo_normal_idx = pseudo_labels.get('pseudo_normal', np.array([]))
    pseudo_anomaly_idx = pseudo_labels.get('pseudo_anomaly', np.array([]))
    
    # 使用Subset构建伪标签数据集
    if train_ds is not None:
        # 创建带伪标签的子集
        all_pseudo_idx = list(pseudo_normal_idx) + list(pseudo_anomaly_idx)
        all_pseudo_labels = [0] * len(pseudo_normal_idx) + [1] * len(pseudo_anomaly_idx)
        
        pseudo_ds = _PseudoLabelSubset(train_ds, all_pseudo_idx, all_pseudo_labels)
        print(f"  伪标签训练数据: {len(pseudo_ds)} 样本")
    else:
        pseudo_ds = None
    
    # 验证和测试数据
    if val_ds is None:
        val_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=True,
            split_name="VAL", normalizer=GLOBAL_NORMALIZER
        )
    
    if test_ds is None:
        test_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=True,
            split_name="TEST", normalizer=GLOBAL_NORMALIZER
        )
    
    print(f"  验证数据: {len(val_ds)} 样本")
    print(f"  测试数据: {len(test_ds)} 样本")
    
    # 数据加载器
    if pseudo_ds is not None and len(pseudo_ds) > 0:
        train_loader = DataLoader(pseudo_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    else:
        print("[警告] 无伪标签数据，使用验证集训练")
        train_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ========== 构建分类器 ==========
    print("\n[2/4] 构建分类器...")
    
    classifier = FaultClassifierV5(
        encoder=model.encoder,
        num_classes=2,
        freeze_encoder=True,
        use_layernorm=False,
        dropout_rate=cfg.DROPOUT_RATE,
        use_dann=cfg.USE_DANN
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE3_EPOCHS)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    
    early_stop = EarlyStopping(patience=cfg.PATIENCE, mode='max')
    
    # ========== 训练循环 ==========
    print(f"\n[3/4] 训练分类器 ({cfg.STAGE3_EPOCHS}轮)...")
    
    history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0
    
    for epoch in range(cfg.STAGE3_EPOCHS):
        # 训练
        classifier.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}", leave=False)
        for batch in pbar:
            img, zr, labels, _ = batch
            img, zr = img.to(device), zr.to(device)
            labels = labels.to(device)
            
            out = classifier(img, zr)
            loss = criterion(out['logits'], labels)
            
            # 域适应损失
            if cfg.USE_DOMAIN_ADAPTATION and 'h' in out:
                if cfg.DA_MODE == 'mmd':
                    # 使用批次内的正常/异常样本作为源/目标
                    normal_mask = labels == 0
                    fault_mask = labels == 1
                    if normal_mask.sum() > 0 and fault_mask.sum() > 0:
                        da_loss = compute_mmd_loss(out['h'][normal_mask], out['h'][fault_mask])
                        loss = loss + cfg.DA_WEIGHT * da_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # 验证
        val_acc, val_f1, _, _ = _evaluate(classifier, val_loader, device)
        
        avg_loss = epoch_loss / len(train_loader)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        logger.log(epoch=epoch+1, train_loss=avg_loss, val_acc=val_acc, val_f1=val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_mgr.save_best(classifier, {'val_f1': val_f1}, name="stage3_best")
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if early_stop(val_f1):
            print(f"\n[早停] 在第 {epoch+1} 轮停止训练")
            break
    
    logger.save_csv()
    
    # ========== 创建可视化管理器 ==========
    viz = VisualizationManager(cfg.STAGE3_DIR)
    
    # ========== 绘制训练曲线 ==========
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage3", lang=lang)
    
    # ========== 最终评估 ==========
    print("\n[4/4] 最终评估...")
    
    # 加载最佳模型
    best_path = cfg.MODEL_DIR / "stage3" / "stage3_best_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        classifier.load_state_dict(ckpt['model_state'])
    
    test_acc, test_f1, test_preds, test_labels = _evaluate(classifier, test_loader, device)
    
    print(f"\n【阶段三完成】")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  测试F1分数: {test_f1:.4f}")
    
    # ========== 绘制混淆矩阵 ==========
    for lang in cfg.LANGS:
        viz.plot_confusion_matrix(test_labels, test_preds, lang=lang)
    
    # ========== 计算ROC/PR曲线数据 ==========
    classifier.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            img, zr, _, _ = batch
            img, zr = img.to(device), zr.to(device)
            out = classifier(img, zr)
            probs = F.softmax(out['logits'], dim=1)[:, 1]  # 故障类的概率
            all_probs.extend(probs.cpu().tolist())
    
    for lang in cfg.LANGS:
        viz.plot_roc_pr_curves(test_labels, all_probs, lang=lang)
    
    # ========== t-SNE可视化 ==========
    print("\n[*] 生成t-SNE可视化...")
    classifier.eval()
    all_features = []
    all_labels_for_tsne = []
    with torch.no_grad():
        for batch in test_loader:
            img, zr, labels, _ = batch
            img, zr = img.to(device), zr.to(device)
            out = classifier(img, zr)
            if 'h' in out:
                all_features.extend(out['h'].cpu().numpy())
                all_labels_for_tsne.extend(labels.tolist())
    
    if all_features:
        for lang in cfg.LANGS:
            viz.plot_tsne(np.array(all_features), np.array(all_labels_for_tsne), 
                         title="Feature Space (t-SNE)", lang=lang)
    
    # 保存评估结果
    results = {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_val_f1': best_val_f1,
    }
    
    import json
    with open(cfg.STAGE3_DIR / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return classifier


class _PseudoLabelSubset(Dataset):
    """伪标签子集数据集"""
    
    def __init__(self, dataset: Dataset, indices: List[int], labels: List[int]):
        self.dataset = dataset
        self.indices = indices
        self.labels = labels
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = self.dataset[real_idx]
        # 替换标签
        return (data[0], data[1], self.labels[idx], idx)


def _evaluate(model, loader, device) -> Tuple[float, float, List[int], List[int]]:
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            img, zr, labels, _ = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            preds = out['logits'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return acc, f1, all_preds, all_labels


# =============================================================================
# 完整流程
# =============================================================================

def run_full_pipeline(cfg: ThreeStageConfigV5,
                      train_ds: Dataset = None,
                      val_ds: Dataset = None,
                      test_ds: Dataset = None,
                      resume_stage1: Path = None) -> Dict:
    """
    运行完整的三阶段流程
    
    返回:
        包含所有结果的字典
    """
    results = {}
    
    # 阶段一
    model, history1 = train_stage1(cfg, train_ds, resume_stage1)
    results['stage1'] = {'model': model, 'history': history1}
    
    if model is None:
        print("[错误] 阶段一失败")
        return results
    
    # 阶段二
    pseudo_labels = run_stage2(model, cfg, train_ds)
    results['stage2'] = pseudo_labels
    
    # 阶段三
    classifier = train_stage3(model, pseudo_labels, cfg, val_ds, test_ds, train_ds)
    results['stage3'] = {'classifier': classifier}
    
    return results

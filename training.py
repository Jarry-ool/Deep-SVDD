# -*- coding: utf-8 -*-
"""
training.py - 训练函数 (V5.12 优化版)
=====================================

性能优化:
1. DataLoader: num_workers + pin_memory
2. 与datasets.py配合的branch_mode条件计算

使用方法:
    cfg = ThreeStageConfigV5(NUM_WORKERS=4, BRANCH_MODE='hetero')
    model, history = train_stage1(cfg, train_ds, val_ds)
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
from gpu_transforms import GPUHeteroTransform


# =============================================================================
# 可视化辅助函数
# =============================================================================

def _visualize_vae_reconstruction(model, dataloader, device, viz: VisualizationManager, 
                                   cfg: ThreeStageConfigV5, gpu_transform=None):
    """VAE重建可视化"""
    model.eval()
    originals, recons = [], []
    
    use_gpu_hetero = gpu_transform is not None
    
    with torch.no_grad():
        for batch in dataloader:
            data, zr, _, _ = batch
            
            # GPU模式: data是signal，需要转换
            if use_gpu_hetero:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
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
    """
    生成样本预览（Hetero + Zerone双分支）
    
    注意: 预览始终用CPU生成，不受USE_GPU_HETERO影响
    """
    print("\n[*] 生成样本预览 (Hetero + Zerone)...")
    
    preview_hetero, preview_zerone, preview_labels = [], [], []
    n_preview = min(cfg.SAMPLE_PREVIEW_COUNT, len(train_ds))
    
    # 预览始终用CPU生成hetero图像
    for i in range(n_preview):
        sample = train_ds.samples[i]
        signal = sample.get('signal', np.zeros(cfg.SIGNAL_LEN, dtype=np.float32))
        label = sample.get('label', 0)
        if label == -1:
            label = 0
        
        # CPU生成hetero图像
        hetero_img = generate_hetero_image(signal, cfg.INPUT_SIZE)
        preview_hetero.append(hetero_img)
        
        # zerone特征/图像
        zerone_feat = extract_zerone_features(signal, fs=cfg.FS)
        if hasattr(train_ds, 'normalizer') and train_ds.normalizer is not None:
            zerone_feat_norm = train_ds.normalizer.transform(zerone_feat)
        else:
            zerone_feat_norm = np.clip(zerone_feat, 0, 1)
        
        if cfg.ZERONE_USE_CNN:
            zerone_out = vector_to_image_raster(zerone_feat_norm, target_size=cfg.INPUT_SIZE)
        else:
            zerone_out = zerone_feat_norm
        
        preview_zerone.append(zerone_out)
        preview_labels.append(label)
    
    for lang in cfg.LANGS:
        # Hetero预览
        viz.plot_sample_preview(preview_hetero, preview_labels, lang=lang, prefix="hetero_samples")
        # Zerone预览
        viz.plot_zerone_preview(preview_zerone, preview_labels, lang=lang)


def _generate_compare_preview(cfg: ThreeStageConfigV5, normalizer: GlobalNormalizer,
                               viz: VisualizationManager, val_ds=None):
    """
    V5.1: 生成正常vs故障对比预览图（中英文各7张）
    
    优化: 
    - 优先使用已加载的val_ds
    - 并行处理
    """
    import pywt
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    
    print("\n[*] 生成正常vs故障对比预览...")
    
    normal_samples = []
    fault_samples = []
    
    # 方式1: 使用已加载的val_ds
    if val_ds is not None and hasattr(val_ds, 'samples') and len(val_ds.samples) > 0:
        print(f"  数据源: 已加载的验证集 ({len(val_ds.samples)} 样本)")
        
        for sample in val_ds.samples:
            label = sample.get('label', -1)
            if label == 0:  # 正常
                normal_samples.append(sample)
            elif label == 1:  # 故障
                fault_samples.append(sample)
        
        print(f"  找到正常样本: {len(normal_samples)}")
        print(f"  找到故障样本: {len(fault_samples)}")
        
        if len(normal_samples) == 0 or len(fault_samples) == 0:
            print("  [警告] 验证集中没有足够的正常/故障样本，跳过对比预览")
            return
        
        # 随机抽取
        np.random.seed(42)
        n_samples = 4
        normal_selected = list(np.random.choice(range(len(normal_samples)), min(n_samples, len(normal_samples)), replace=False))
        fault_selected = list(np.random.choice(range(len(fault_samples)), min(n_samples, len(fault_samples)), replace=False))
        
        # 处理样本
        def _process_sample(sample):
            """处理单个样本"""
            try:
                signal = sample.get('signal', None)
                if signal is None:
                    return None
                signal = np.array(signal, dtype=np.float32)
                
                # 特征提取
                feat = extract_zerone_features(signal, fs=cfg.FS)
                feat_norm = normalizer.transform(feat) if normalizer.is_fitted else feat
                zerone_img = vector_to_image_raster(feat_norm, target_size=cfg.INPUT_SIZE)
                
                # Hetero图像
                hetero_img = generate_hetero_image(signal, cfg.INPUT_SIZE)
                
                return {'feat': feat, 'zerone_img': zerone_img, 'hetero_img': hetero_img}
            except Exception as e:
                return None
        
        # 处理选中的样本
        normal_results = [_process_sample(normal_samples[i]) for i in normal_selected]
        fault_results = [_process_sample(fault_samples[i]) for i in fault_selected]
        
        normal_results = [r for r in normal_results if r is not None]
        fault_results = [r for r in fault_results if r is not None]
        
        if not normal_results or not fault_results:
            print("  [警告] 处理样本失败，跳过对比预览")
            return
        
        normal_zerone_imgs = [r['zerone_img'] for r in normal_results]
        normal_hetero_imgs = [r['hetero_img'] for r in normal_results]
        normal_feats = [r['feat'] for r in normal_results]
        
        fault_zerone_imgs = [r['zerone_img'] for r in fault_results]
        fault_hetero_imgs = [r['hetero_img'] for r in fault_results]
        fault_feats = [r['feat'] for r in fault_results]
        
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
        return
    
    # 方式2: 从VAL目录读取JSONL (兼容旧逻辑)
    print(f"  数据源: {cfg.VAL_DIR}")
    
    if not cfg.VAL_DIR.exists():
        print("  [跳过] VAL目录不存在，且无已加载验证集")
        return
    
    normal_files = []
    fault_files = []
    
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
    
    # 辅助函数：加载信号并处理（用于并行）
    def _process_single_file(fpath):
        """加载信号并生成特征和图像"""
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.loads(f.readline())
            raw = data.get('signal_value', data.get('signal', None))
            if raw is None:
                signal = np.zeros(cfg.SIGNAL_LEN, dtype=np.float32)
            elif isinstance(raw, str):
                signal = np.array([float(x) for x in raw.split(',')], dtype=np.float32)
            else:
                signal = np.array(raw, dtype=np.float32)
            
            # 特征提取
            feat = extract_zerone_features(signal, fs=cfg.FS)
            feat_norm = normalizer.transform(feat)
            zerone_img = vector_to_image_raster(feat_norm, target_size=cfg.INPUT_SIZE)
            
            # Hetero图像
            size = cfg.INPUT_SIZE
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
            hetero_img = np.stack([cwt_img, stft_img, ctx_img], axis=0).astype(np.float32)
            
            return {'feat': feat, 'zerone_img': zerone_img, 'hetero_img': hetero_img}
        except Exception as e:
            return None
    
    # 并行处理所有样本
    all_files = normal_selected + fault_selected
    print(f"  并行处理 {len(all_files)} 个样本...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_process_single_file, all_files))
    
    # 分离结果
    n_normal = len(normal_selected)
    normal_results = [r for r in results[:n_normal] if r is not None]
    fault_results = [r for r in results[n_normal:] if r is not None]
    
    normal_zerone_imgs = [r['zerone_img'] for r in normal_results]
    normal_hetero_imgs = [r['hetero_img'] for r in normal_results]
    normal_feats = [r['feat'] for r in normal_results]
    
    fault_zerone_imgs = [r['zerone_img'] for r in fault_results]
    fault_hetero_imgs = [r['hetero_img'] for r in fault_results]
    fault_feats = [r['feat'] for r in fault_results]
    
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
    
    # ========== Zerone特征高级可视化 (仅zerone/dual模式) ==========
    if cfg.BRANCH_MODE in ('zerone', 'dual') and normal_feats and fault_feats:
        print(f"\n[*] 生成Zerone特征高级分析...")
        all_feats = np.vstack([np.array(normal_feats), np.array(fault_feats)])
        all_labels = np.array([0]*len(normal_feats) + [1]*len(fault_feats))
        
        for lang in cfg.LANGS:
            # STFT段均值统计
            viz.plot_zerone_stft_stats(all_feats, all_labels, lang=lang)
            # 时域特征相关性
            viz.plot_time_domain_correlation(all_feats, all_labels, split_name='val', lang=lang)
            # PSD瀑布图
            viz.plot_psd_waterfall(all_feats, all_labels, split_name='val', lang=lang)
        
        print(f"  ✅ Zerone高级分析完成！")


# =============================================================================
# 阶段一：无监督学习
# =============================================================================

def train_stage1(cfg: ThreeStageConfigV5, 
                 train_ds: Dataset = None,
                 val_ds: Dataset = None,
                 resume_from: Path = None) -> Tuple[AnomalyModelV5, Dict]:
    """
    阶段一：无监督学习 (SVDD + VAE)
    
    参数:
        cfg: 配置对象
        train_ds: 训练数据集 (可选，如果不提供则从cfg.TRAIN_DIR加载)
        val_ds: 验证数据集 (用于生成对比预览)
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
        GLOBAL_NORMALIZER.save(cfg.OUTPUT_ROOT / "global_normalizer.npz")
        
        train_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=False,
            split_name="TRAIN", normalizer=GLOBAL_NORMALIZER
        )
    else:
        # 使用提供的数据集
        # 优先加载已有的归一化参数 (公共位置)
        normalizer_path = cfg.OUTPUT_ROOT / "global_normalizer.npz"
        if normalizer_path.exists() and not GLOBAL_NORMALIZER.is_fitted:
            print("\n[1/5] 加载已有归一化参数...")
            GLOBAL_NORMALIZER.load(normalizer_path)
        elif hasattr(train_ds, 'get_all_features_for_normalization') and not GLOBAL_NORMALIZER.is_fitted:
            print("\n[1/5] 计算全局归一化参数...")
            all_features = train_ds.get_all_features_for_normalization()
            if len(all_features) > 0:
                GLOBAL_NORMALIZER.fit(all_features[:5000])
                GLOBAL_NORMALIZER.save(normalizer_path)
    
    print(f"\n[2/5] 训练数据集大小: {len(train_ds)}")
    
    # === DataLoader配置 ===
    # num_workers: Windows建议2-4，卡死用0
    # pin_memory: 加速CPU->GPU传输
    num_workers = cfg.NUM_WORKERS if hasattr(cfg, 'NUM_WORKERS') else 4
    batch_size = cfg.BATCH_SIZE
    
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs.update(num_workers=num_workers, prefetch_factor=2)
    else:
        loader_kwargs.update(num_workers=0)
    
    train_loader = DataLoader(train_ds, **loader_kwargs)
    print(f"  DataLoader: batch={batch_size}, workers={num_workers}, pin_memory=True")
    
    # ========== 创建可视化管理器 ==========
    viz = VisualizationManager(cfg.STAGE1_DIR)
    
    # ========== 样本预览（Hetero + Zerone双分支）==========
    _generate_sample_previews(train_ds, cfg, viz)
    
    # ========== V5.1: 正常vs故障对比预览 ==========
    _generate_compare_preview(cfg, GLOBAL_NORMALIZER, viz, val_ds)
    
    # ========== 构建模型 ==========
    print("\n[3/5] 构建模型...")
    model = AnomalyModelV5(
        branch_mode=cfg.BRANCH_MODE,
        fusion_mode=cfg.FUSION_MODE,
        zerone_use_cnn=cfg.ZERONE_USE_CNN,
        use_modality_dropout=cfg.USE_MODALITY_DROPOUT,
        modality_dropout_p=cfg.MODALITY_DROPOUT_RATE,
        dropout_rate=cfg.DROPOUT_RATE,
        latent_dim=getattr(cfg, 'SVDD_LATENT_DIM', 64),  # 使用配置的SVDD维度
    ).to(device)
    
    print(f"  支线模式: {cfg.BRANCH_MODE}")
    print(f"  融合策略: {cfg.FUSION_MODE}")
    print(f"  Zerone架构: {'CNN' if cfg.ZERONE_USE_CNN else 'MLP'}")
    print(f"  模态Dropout: {'✅' if cfg.USE_MODALITY_DROPOUT else '❌'}")
    
    # 分组参数：SVDD投影头使用更小的学习率和更大的权重衰减
    svdd_lr_scale = getattr(cfg, 'SVDD_LR_SCALE', 0.1)
    svdd_weight_decay = getattr(cfg, 'SVDD_WEIGHT_DECAY', 1e-3)
    
    param_groups = [
        # 编码器参数：正常学习率
        {'params': model.encoder.parameters(), 'lr': cfg.LR, 'weight_decay': cfg.WEIGHT_DECAY},
        # SVDD投影头：较小学习率 + 较大权重衰减（防止崩塌）
        {'params': model.svdd_proj.parameters(), 'lr': cfg.LR * svdd_lr_scale, 'weight_decay': svdd_weight_decay},
    ]
    
    # VAE参数（如果有）
    if model.has_vae:
        param_groups.extend([
            {'params': model.vae_mu.parameters(), 'lr': cfg.LR, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': model.vae_logvar.parameters(), 'lr': cfg.LR, 'weight_decay': cfg.WEIGHT_DECAY},
            {'params': model.vae_decoder.parameters(), 'lr': cfg.LR, 'weight_decay': cfg.WEIGHT_DECAY},
        ])
    
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.STAGE1_EPOCHS)
    
    print(f"  SVDD学习率: {cfg.LR * svdd_lr_scale:.6f} (主LR的{svdd_lr_scale}倍)")
    print(f"  SVDD权重衰减: {svdd_weight_decay}")
    
    start_epoch = 0
    best_loss = float('inf')
    
    # 恢复训练
    if resume_from and Path(resume_from).exists():
        print(f"\n[恢复] 从检查点加载: {resume_from}")
        ckpt = ckpt_mgr.load(resume_from, model, optimizer, scheduler)
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('metrics', {}).get('total_loss', float('inf'))
    
    # GPU Hetero变换器（提前初始化，避免作用域问题）
    use_gpu_hetero = getattr(cfg, 'USE_GPU_HETERO', False) and cfg.BRANCH_MODE in ('hetero', 'dual')
    print(f"  [DEBUG] USE_GPU_HETERO={getattr(cfg, 'USE_GPU_HETERO', 'NOT_SET')}, BRANCH_MODE={cfg.BRANCH_MODE}, use_gpu_hetero={use_gpu_hetero}")
    gpu_transform = None
    if use_gpu_hetero:
        gpu_transform = GPUHeteroTransform(size=cfg.INPUT_SIZE, device=device)
        print(f"  [GPU加速] Hetero图像在GPU上批量计算")
    
    # VAE预训练
    if model.has_vae and start_epoch == 0:
        print("\n[4/5] VAE预训练 (5轮)...")
        
        for epoch in range(5):
            model.train()
            total_loss = 0
            nan_caught = False
            for i, batch in enumerate(tqdm(train_loader, desc=f"VAE预训练 {epoch+1}/5", leave=False, ncols=80)):
                data, zr, _, _ = batch
                
                # GPU模式: data是signal [B, L]，需要在GPU上计算hetero
                if use_gpu_hetero and gpu_transform is not None:
                    signals = data.to(device)
                    img = gpu_transform(signals)  # [B, 3, 224, 224]
                else:
                    img = data.to(device)
                    signals = data  # 用于debug
                zr = zr.to(device)
                
                out = model(img, zr)
                loss = out['vae_recon_loss'].mean() + 0.01 * out['vae_kl'].mean()
                
                # === NaN Catch: 抓第一个坏batch ===
                if not torch.isfinite(loss):
                    print(f"\n[NaN CATCH] epoch={epoch} batch={i}")
                    print(f"  recon_loss finite: {torch.isfinite(out['vae_recon_loss']).all().item()}")
                    print(f"  kl finite:         {torch.isfinite(out['vae_kl']).all().item()}")
                    print(f"  img: min={img.min():.4f}, max={img.max():.4f}, has_nan={torch.isnan(img).any()}")
                    
                    torch.save({
                        "epoch": epoch,
                        "batch_idx": i,
                        "signals": signals.detach().cpu(),
                        "img": img.detach().cpu(),
                        "zr": zr.detach().cpu(),
                        "vae_recon_loss": out['vae_recon_loss'].detach().cpu(),
                        "vae_kl": out['vae_kl'].detach().cpu(),
                    }, cfg.OUTPUT_ROOT / "nan_batch_vae_pretrain.pth")
                    print(f"  已保存到: {cfg.OUTPUT_ROOT / 'nan_batch_vae_pretrain.pth'}")
                    nan_caught = True
                    break
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if nan_caught:
                print(f"  [!] VAE预训练在epoch={epoch}提前终止，请检查nan_batch_vae_pretrain.pth")
                break
            
            print(f"  VAE预训练 Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # VAE重建可视化
        _visualize_vae_reconstruction(model, train_loader, device, viz, cfg, gpu_transform=gpu_transform)
    
    # SVDD中心初始化
    if start_epoch == 0:
        print("\n[*] 初始化SVDD中心...")
        model.init_center(train_loader, device, gpu_transform=gpu_transform)
    
    # ========== 联合训练 ==========
    print(f"\n[5/5] 联合训练 ({cfg.STAGE1_EPOCHS}轮)...")
    
    history = {'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': []}
    early_stop = EarlyStopping(patience=10, mode='min')  # Stage1早停
    
    for epoch in range(start_epoch, cfg.STAGE1_EPOCHS):
        model.train()
        epoch_svdd, epoch_vae, epoch_total = 0, 0, 0
        
        beta = min(1.0, 1.0 * (epoch / max(10, 1)))  # KL权重warmup
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False, ncols=80)
        for batch in pbar:
            data, zr, _, _ = batch
            
            # GPU模式: data是signal，需要在GPU上计算hetero
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = model(img, zr)
            
            # SVDD Loss with soft-boundary防崩塌
            svdd_scores = out['svdd_score']  # (batch,) 每个样本到center的距离²
            svdd_loss = svdd_scores.mean()
            
            # === Soft-Boundary SVDD 防崩塌 ===
            # 思路：不仅最小化距离，还要保持一定的"半径"
            # 1. 计算当前batch的"半径"（最大距离的某个分位数）
            # 2. 如果半径太小，添加惩罚
            with torch.no_grad():
                radius_quantile = torch.quantile(svdd_scores, 0.9)  # 90%分位数作为半径
            
            # 半径正则：鼓励半径保持在合理范围
            # 如果半径接近0，给予惩罚；用log防止数值问题
            radius_reg = -torch.log(radius_quantile + 1e-6)
            
            # 对比正则：鼓励样本之间有差异（使用z_svdd的方差）
            z_svdd = out['z_svdd']  # (batch, latent_dim)
            z_variance = z_svdd.var(dim=0).mean()  # 每个维度的方差，取平均
            diversity_reg = 1.0 / (z_variance + 1e-6)
            
            # 总SVDD Loss — 系数通过 cfg 控制以支持消融实验
            _lam_r = getattr(cfg, 'LAMBDA_RADIUS', 0.01)
            _lam_d = getattr(cfg, 'LAMBDA_DIV',    0.001)
            _lam_v = getattr(cfg, 'LAMBDA_VAE',    0.5)
            svdd_loss_total = svdd_loss + _lam_r * radius_reg + _lam_d * diversity_reg

            if model.has_vae:
                vae_loss = out['vae_recon_loss'].mean() + beta * 0.01 * out['vae_kl'].mean()
                total_loss = _lam_v * svdd_loss_total + (1.0 - _lam_v) * vae_loss
            else:
                vae_loss = torch.tensor(0.0)
                total_loss = svdd_loss_total
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_svdd += svdd_loss.item()  # 记录原始svdd_loss，不含正则项
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
        
        # 早停检查
        if early_stop(avg_total):
            print(f"\n[早停] Stage1在第 {epoch+1} 轮停止训练")
            break
    
    logger.save_csv()
    
    # ========== 绘制训练曲线 ==========
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage1", lang=lang)
    
    print(f"\n【阶段一完成】最佳损失: {best_loss:.4f}")
    
    # 清理GPU显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
    
    # 加载归一化器 (公共位置)
    normalizer_path = cfg.OUTPUT_ROOT / "global_normalizer.npz"
    if normalizer_path.exists() and not GLOBAL_NORMALIZER.is_fitted:
        GLOBAL_NORMALIZER.load(normalizer_path)
    
    # 准备数据
    if train_ds is None:
        train_ds = TransformerVibrationDataset(
            cfg.PROJECT_ROOT, cfg, use_labels=False,
            split_name="TRAIN", normalizer=GLOBAL_NORMALIZER
        )
    
    # === DataLoader配置 ===
    num_workers = cfg.NUM_WORKERS if hasattr(cfg, 'NUM_WORKERS') else 4
    batch_size = cfg.BATCH_SIZE
    
    loader_kwargs = dict(batch_size=batch_size, shuffle=False, pin_memory=True)
    if num_workers > 0:
        loader_kwargs.update(num_workers=num_workers, prefetch_factor=2)
    else:
        loader_kwargs.update(num_workers=0)
    
    loader = DataLoader(train_ds, **loader_kwargs)
    
    # GPU Hetero变换器
    use_gpu_hetero = getattr(cfg, 'USE_GPU_HETERO', False) and cfg.BRANCH_MODE in ('hetero', 'dual')
    gpu_transform = None
    if use_gpu_hetero:
        gpu_transform = GPUHeteroTransform(size=cfg.INPUT_SIZE, device=device)
        print(f"  [GPU加速] Hetero图像在GPU上批量计算")
    
    print("\n[1/2] 计算异常得分...")
    model.eval()
    all_scores = []
    all_indices = []
    all_features = []  # 收集特征用于可视化
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="计算得分", leave=False, ncols=80):
            data, zr, _, idx = batch
            
            # GPU模式: data是signal，需要转换
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = model(img, zr)
            scores = out['svdd_score']
            all_scores.extend(scores.cpu().tolist())
            all_indices.extend(idx.tolist())
            
            # 收集部分特征用于可视化 (每100个batch采样一次)
            if len(all_features) < 10000 and np.random.rand() < 0.01:
                if 'h' in out:
                    all_features.extend(out['h'].cpu().numpy()[:10])  # 每次取10个
    
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
    
    # ========== SVDD特征空间可视化 ==========
    if len(all_features) > 0:
        print("\n[*] 生成SVDD特征空间可视化...")
        all_features = np.array(all_features)
        viz_stage1 = VisualizationManager(cfg.STAGE1_DIR)
        # 随机采样对应的scores
        n_vis = min(len(all_features), 5000)
        vis_indices = np.random.choice(len(all_scores), n_vis, replace=False)
        vis_scores = all_scores[vis_indices]
        # 特征数量可能不匹配scores，所以分别采样
        feat_indices = np.random.choice(len(all_features), min(n_vis, len(all_features)), replace=False)
        for lang in cfg.LANGS:
            viz_stage1.plot_svdd_feature_space(all_features[feat_indices], vis_scores[:len(feat_indices)], lang=lang)
    
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
    
    # 加载归一化器 (公共位置)
    normalizer_path = cfg.OUTPUT_ROOT / "global_normalizer.npz"
    if normalizer_path.exists() and not GLOBAL_NORMALIZER.is_fitted:
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
    
    # === DataLoader配置 ===
    num_workers = cfg.NUM_WORKERS if hasattr(cfg, 'NUM_WORKERS') else 4
    batch_size = cfg.BATCH_SIZE
    
    loader_kwargs = dict(batch_size=batch_size, pin_memory=True)
    if num_workers > 0:
        loader_kwargs.update(num_workers=num_workers)
    else:
        loader_kwargs.update(num_workers=0)
    
    if pseudo_ds is not None and len(pseudo_ds) > 0:
        train_loader = DataLoader(pseudo_ds, shuffle=True, **loader_kwargs)
    else:
        print("[警告] 无伪标签数据，使用验证集训练")
        train_loader = DataLoader(val_ds, shuffle=True, **loader_kwargs)
    
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    # GPU Hetero变换器
    use_gpu_hetero = getattr(cfg, 'USE_GPU_HETERO', False) and cfg.BRANCH_MODE in ('hetero', 'dual')
    gpu_transform = None
    if use_gpu_hetero:
        gpu_transform = GPUHeteroTransform(size=cfg.INPUT_SIZE, device=device)
        print(f"  [GPU加速] Hetero图像在GPU上批量计算")
    
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}", leave=False, ncols=80)
        for batch in pbar:
            data, zr, labels, _ = batch
            
            # GPU模式: data是signal，需要转换
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
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
        val_acc, val_f1, _, _, _, _ = _evaluate(classifier, val_loader, device, gpu_transform=gpu_transform)
        
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
            ckpt_mgr.save(classifier, optimizer, epoch + 1,
                         {'train_loss': avg_loss, 'val_acc': val_acc, 'val_f1': val_f1},
                         scheduler)
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
    
    test_acc, test_f1, test_prec, test_rec, test_preds, test_labels = _evaluate(classifier, test_loader, device, gpu_transform=gpu_transform)
    
    print(f"\n【阶段三完成】")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  测试F1分数: {test_f1:.4f}")
    print(f"  测试精确率: {test_prec:.4f}")
    print(f"  测试召回率: {test_rec:.4f}")
    
    # ========== 绘制混淆矩阵 ==========
    for lang in cfg.LANGS:
        viz.plot_confusion_matrix(test_labels, test_preds, lang=lang)
    
    # ========== 计算ROC/PR曲线数据 ==========
    classifier.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            data, zr, _, _ = batch
            
            # GPU模式: data是signal，需要转换
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
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
            data, zr, labels, _ = batch
            
            # GPU模式: data是signal，需要转换
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = classifier(img, zr)
            if 'h' in out:
                all_features.extend(out['h'].cpu().numpy())
                all_labels_for_tsne.extend(labels.tolist())
    
    if all_features:
        for lang in cfg.LANGS:
            viz.plot_tsne(np.array(all_features), np.array(all_labels_for_tsne), lang=lang)
    
    # ========== 错误分析可视化 ==========
    print("\n[*] 生成错误分析可视化...")
    
    # 收集错误样本信息
    error_info = []
    classifier.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, zr, labels, _ = batch
            
            if use_gpu_hetero and gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = classifier(img, zr)
            preds = out['logits'].argmax(dim=1)
            
            # 找错误样本
            for i in range(len(preds)):
                if preds[i].item() != labels[i].item():
                    error_info.append({
                        'image': img[i].cpu(),
                        'true': labels[i].item(),
                        'pred': preds[i].item()
                    })
                    if len(error_info) >= 16:  # 最多收集16个
                        break
            if len(error_info) >= 16:
                break
    
    # 绘制错误分析图
    if error_info:
        for lang in cfg.LANGS:
            viz.plot_error_analysis(error_info, test_preds, test_labels, all_probs, lang=lang)
    
    # ========== 特征可分性分析 ==========
    if all_features:
        print("\n[*] 生成特征可分性分析...")
        for lang in cfg.LANGS:
            viz.plot_feature_separability(np.array(all_features), np.array(all_labels_for_tsne), lang=lang)
    
    # ========== 双分支/门控分析 (仅dual模式) ==========
    if cfg.BRANCH_MODE == 'dual':
        print("\n[*] 生成双分支融合分析...")
        
        # 收集双分支特征和门控权重
        hetero_feats_list = []
        zerone_feats_list = []
        gate_weights_list = []
        labels_list = []
        
        classifier.eval()
        with torch.no_grad():
            for batch in test_loader:
                data, zr, labels_batch, _ = batch
                
                if use_gpu_hetero and gpu_transform is not None:
                    signals = data.to(device)
                    img = gpu_transform(signals)
                else:
                    img = data.to(device)
                zr = zr.to(device)
                
                out = classifier(img, zr)
                
                # 收集分支特征
                if 'h_hetero' in out:
                    hetero_feats_list.extend(out['h_hetero'].cpu().numpy())
                if 'h_zerone' in out:
                    zerone_feats_list.extend(out['h_zerone'].cpu().numpy())
                if 'gate_weights' in out:
                    gate_weights_list.extend(out['gate_weights'].cpu().numpy())
                labels_list.extend(labels_batch.tolist())
                
                if len(labels_list) >= 500:  # 限制样本数
                    break
        
        labels_arr = np.array(labels_list)
        
        # 双分支特征空间对比
        if hetero_feats_list and zerone_feats_list:
            for lang in cfg.LANGS:
                viz.plot_dual_branch_analysis(
                    np.array(hetero_feats_list), 
                    np.array(zerone_feats_list),
                    labels_arr, lang=lang
                )
        
        # 门控权重分析
        if gate_weights_list:
            for lang in cfg.LANGS:
                viz.plot_gate_distribution(np.array(gate_weights_list), labels_arr, lang=lang)
    
    # 保存评估结果
    results = {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'best_val_f1': best_val_f1,
    }
    
    import json
    with open(cfg.STAGE3_DIR / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 清理GPU显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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


def _evaluate(model, loader, device, gpu_transform=None) -> Tuple[float, float, float, float, List[int], List[int]]:
    """评估模型，返回 (acc, f1, precision, recall, preds, labels)"""
    model.eval()
    all_preds = []
    all_labels = []
    
    use_gpu_hetero = gpu_transform is not None
    
    with torch.no_grad():
        for batch in loader:
            data, zr, labels, _ = batch
            
            # GPU模式: data是signal，需要转换
            if use_gpu_hetero:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = model(img, zr)
            preds = out['logits'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return acc, f1, prec, rec, all_preds, all_labels


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
    
    # 阶段一 (传入val_ds用于生成对比预览)
    model, history1 = train_stage1(cfg, train_ds, val_ds, resume_stage1)
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

# -*- coding: utf-8 -*-
"""
continue_dual_training.py
==========================

专门用于继续 branch_dual 训练的脚本

你的情况：
- branch_hetero: ✅ 已完成
- branch_zerone: ✅ 已完成  
- branch_dual: ⏳ 训练到第12轮中断

使用方法：
    python continue_dual_training.py

这个脚本会：
1. 检测 branch_dual 的检查点
2. 从最新检查点恢复训练
3. 完成后生成三支线对比可视化
"""

import os
import sys
from pathlib import Path

# 确保能导入主模块
sys.path.insert(0, str(Path(__file__).parent))

def main():
    # 导入主模块
    try:
        from transformer_three_stage_v3 import (
            ThreeStageConfigV3, 
            AnomalyModelV3,
            train_stage1, 
            run_stage2, 
            train_stage3,
            TransformerVibrationDataset,
            VisualizationManager,
            TrainingLogger,
            CheckpointManager,
        )
        from torch.utils.data import DataLoader, ConcatDataset
        import torch
        import numpy as np
        from tqdm import tqdm
    except ImportError as e:
        print(f"[错误] 无法导入必要模块: {e}")
        print("请确保 transformer_three_stage_v3.py 在同一目录下")
        return
    
    # ============ 配置 ============
    # 请根据你的实际情况修改这两个路径
    DATA_ROOT = Path(r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016")
    OUTPUT_ROOT = Path("./three_stage_results_v3")
    
    print("="*70)
    print("继续 branch_dual 训练")
    print("="*70)
    print(f"数据目录: {DATA_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    
    # 检测检查点
    dual_dir = OUTPUT_ROOT / "branch_dual"
    ckpt_dir = dual_dir / "checkpoints" / "stage1"
    model_dir = dual_dir / "models"
    
    # 查找最新检查点
    latest_ckpt = None
    start_epoch = 0
    
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pth"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            try:
                start_epoch = int(latest_ckpt.stem.split('epoch')[-1])
            except:
                start_epoch = 0
            print(f"\n[检测到检查点] {latest_ckpt.name}")
            print(f"  将从第 {start_epoch} 轮恢复训练")
    
    # 检查阶段1是否已完成
    stage1_best = model_dir / "stage1_best.pth"
    stage1_done = stage1_best.exists()
    
    # 检查阶段2是否已完成
    pseudo_labels_file = dual_dir / "stage2_pseudo_labels" / "pseudo_labels.npz"
    stage2_done = pseudo_labels_file.exists()
    
    # 检查阶段3是否已完成
    stage3_eval = dual_dir / "stage3_supervised" / "test_evaluation.json"
    stage3_done = stage3_eval.exists()
    
    print(f"\n当前状态:")
    print(f"  阶段1 (无监督学习): {'✅ 已完成' if stage1_done else '⏳ 未完成'}")
    print(f"  阶段2 (伪标签生成): {'✅ 已完成' if stage2_done else '⏳ 未完成'}")
    print(f"  阶段3 (监督微调):   {'✅ 已完成' if stage3_done else '⏳ 未完成'}")
    
    if stage3_done:
        print("\n[跳过] branch_dual 已完全完成!")
        generate_comparison(OUTPUT_ROOT)
        return
    
    # 创建配置
    cfg = ThreeStageConfigV3(
        PROJECT_ROOT=DATA_ROOT,
        OUTPUT_ROOT=OUTPUT_ROOT,
        BRANCH_MODE='dual',
    )
    cfg.__post_init__()
    
    device = torch.device(cfg.DEVICE)
    print(f"\n使用设备: {device}")
    
    # ============ 阶段1：从检查点恢复或使用已有模型 ============
    if not stage1_done:
        if latest_ckpt:
            print(f"\n{'='*60}")
            print(f"阶段1：从检查点恢复 (第{start_epoch+1}轮 -> 第{cfg.STAGE1_EPOCHS}轮)")
            print(f"{'='*60}")
            
            # 加载检查点
            ckpt = torch.load(latest_ckpt, map_location=device)
            
            # 创建模型
            model = AnomalyModelV3(cfg).to(device)
            model.load_state_dict(ckpt['model_state'])
            if 'center' in ckpt:
                model.center = ckpt['center']
            
            # 创建优化器
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
            if 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            
            # 创建调度器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.STAGE1_EPOCHS
            )
            if 'scheduler_state' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            
            # 加载数据
            print("\n加载数据...")
            train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
            val_ds = TransformerVibrationDataset(cfg.VAL_DIR, cfg, use_labels=False, split_name="VAL")
            combined_ds = ConcatDataset([train_ds, val_ds])
            train_loader = DataLoader(combined_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                                      num_workers=0, drop_last=True)
            
            # 工具
            viz = VisualizationManager(cfg)
            logger = TrainingLogger(cfg, "stage1")
            ckpt_mgr = CheckpointManager(cfg, "stage1")
            
            # 恢复历史
            history = {'epoch': [], 'svdd_loss': [], 'vae_loss': [], 'total_loss': [], 'recon_loss': []}
            best_loss = ckpt.get('metrics', {}).get('total_loss', float('inf'))
            
            # 继续训练
            print(f"\n开始训练...")
            for epoch in range(start_epoch, cfg.STAGE1_EPOCHS):
                model.train()
                epoch_svdd, epoch_vae, epoch_total, epoch_recon = 0, 0, 0, 0
                
                beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch / max(cfg.BETA_WARMUP, 1)))
                
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=True)
                for batch in pbar:
                    img, zr, _, _ = batch
                    img, zr = img.to(device), zr.to(device)
                    
                    out = model(img, zr)
                    
                    svdd_loss = out['svdd_score'].mean()
                    
                    if model.has_vae:
                        vae_loss = out['vae_recon_loss'].mean() + beta * out['vae_kl'].mean()
                        total_loss = svdd_loss + vae_loss
                        epoch_recon += out['vae_recon_loss'].mean().item()
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
                    
                    pbar.set_postfix({'svdd': f'{svdd_loss.item():.4f}', 'total': f'{total_loss.item():.4f}'})
                
                scheduler.step()
                
                n_batches = len(train_loader)
                avg_svdd = epoch_svdd / n_batches
                avg_vae = epoch_vae / n_batches
                avg_total = epoch_total / n_batches
                avg_recon = epoch_recon / n_batches
                
                history['epoch'].append(epoch + 1)
                history['svdd_loss'].append(avg_svdd)
                history['vae_loss'].append(avg_vae)
                history['total_loss'].append(avg_total)
                history['recon_loss'].append(avg_recon)
                
                logger.log(epoch=epoch+1, svdd_loss=avg_svdd, vae_loss=avg_vae,
                           total_loss=avg_total, recon_loss=avg_recon)
                
                # 保存最佳模型
                if avg_total < best_loss:
                    best_loss = avg_total
                    torch.save({
                        'model_state': model.state_dict(),
                        'center': model.center,
                        'epoch': epoch,
                        'loss': best_loss,
                    }, cfg.MODEL_DIR / "stage1_best.pth")
                    print(f"  [保存] 最佳模型 (loss={best_loss:.4f})")
                
                # 检查点
                if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
                    ckpt_mgr.save(model, optimizer, epoch + 1,
                                 {'svdd_loss': avg_svdd, 'total_loss': avg_total}, scheduler)
                
                # 可视化
                if (epoch + 1) % cfg.VIZ_EVERY == 0:
                    for lang in cfg.LANGS:
                        viz.plot_training_curves(history, "stage1", lang=lang)
                
                print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
            
            logger.save_csv()
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage1", lang=lang)
            
            print(f"\n【阶段1完成】最佳损失: {best_loss:.4f}")
        else:
            # 没有检查点，完整训练阶段1
            print("\n[警告] 没有找到检查点，将完整训练阶段1")
            model, _ = train_stage1(cfg)
    else:
        # 加载已完成的阶段1模型
        print(f"\n加载已完成的阶段1模型...")
        model = AnomalyModelV3(cfg).to(device)
        ckpt = torch.load(stage1_best, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.center = ckpt['center']
        print(f"  模型加载成功 (epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.4f})")
    
    # ============ 阶段2：伪标签生成 ============
    if not stage2_done:
        print(f"\n{'='*60}")
        print("阶段2：伪标签生成")
        print(f"{'='*60}")
        pseudo_labels = run_stage2(model, cfg)
    else:
        print(f"\n加载已有的伪标签...")
        data = np.load(pseudo_labels_file, allow_pickle=True)
        pseudo_labels = {k: data[k] for k in data.files}
        print(f"  伪标签加载成功")
    
    # ============ 阶段3：监督微调 ============
    if not stage3_done:
        print(f"\n{'='*60}")
        print("阶段3：监督微调")
        print(f"{'='*60}")
        train_stage3(model, pseudo_labels, cfg)
    
    print(f"\n{'='*70}")
    print("branch_dual 训练完成!")
    print(f"{'='*70}")
    
    # 生成对比可视化
    generate_comparison(OUTPUT_ROOT)


def generate_comparison(output_root: Path):
    """生成三支线对比可视化"""
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"\n{'='*70}")
    print("生成三支线对比可视化")
    print(f"{'='*70}")
    
    # 收集结果
    results = {}
    for branch in ['hetero', 'zerone', 'dual']:
        eval_file = output_root / f"branch_{branch}" / "stage3_supervised" / "test_evaluation.json"
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                results[branch] = json.load(f)
                print(f"  ✅ {branch}: F1={results[branch].get('test_f1', 0):.4f}")
        else:
            print(f"  ❌ {branch}: 未完成")
    
    if len(results) < 2:
        print("\n[警告] 需要至少2个支线完成才能生成对比")
        return
    
    # 创建输出目录
    compare_dir = output_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in ['cn', 'en']:
        print(f"\n生成 {lang.upper()} 版本...")
        
        if lang == 'cn':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            metric_names = ['准确率', 'F1分数', '精确率', '召回率']
            branch_names = {'hetero': '图像分支', 'zerone': '特征分支', 'dual': '双分支融合'}
            title = '三支线性能对比'
        else:
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            branch_names = {'hetero': 'Image Branch', 'zerone': 'Feature Branch', 'dual': 'Dual Branch'}
            title = 'Three-Branch Performance Comparison'
        
        branches = list(results.keys())
        metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
        
        # 柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.25
        colors_list = ['#0072B2', '#E69F00', '#CC79A7']
        
        for i, branch in enumerate(branches):
            values = [results[branch].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, values, width,
                         label=branch_names.get(branch, branch),
                         color=colors_list[i % len(colors_list)],
                         edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Score' if lang == 'en' else '得分')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_names)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        plt.tight_layout()
        fig.savefig(compare_dir / f"comparison_bar_{lang}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # 雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, branch in enumerate(branches):
            values = [results[branch].get(m, 0) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=branch_names.get(branch, branch),
                   color=colors_list[i % len(colors_list)])
            ax.fill(angles, values, alpha=0.25, color=colors_list[i % len(colors_list)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        fig.savefig(compare_dir / f"comparison_radar_{lang}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  保存: comparison_bar_{lang}.png, comparison_radar_{lang}.png")
    
    # 保存汇总
    summary = {'results': results, 'best_branch': max(results.keys(), key=lambda b: results[b].get('test_f1', 0))}
    with open(compare_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n【对比可视化完成】")
    print(f"  最佳支线: {summary['best_branch']} (F1={results[summary['best_branch']].get('test_f1', 0):.4f})")
    print(f"  保存目录: {compare_dir}")


if __name__ == "__main__":
    main()

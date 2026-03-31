# -*- coding: utf-8 -*-
"""
resume_and_compare.py
======================

断点续训 + 三支线对比可视化脚本

功能：
1. 自动检测已完成的支线，跳过不重复训练
2. 从检查点恢复中断的训练
3. 生成三支线对比可视化（中英文）

使用方法：
    # 继续训练 branch_dual (从检查点恢复)
    python resume_and_compare.py --resume
    
    # 只生成对比可视化 (如果三个支线都有结果)
    python resume_and_compare.py --compare_only
    
    # 完整流程：续训 + 对比
    python resume_and_compare.py --resume --compare

Author: V3断点续训扩展
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 尝试导入V3主模块
try:
    from transformer_three_stage_v3 import (
        ThreeStageConfigV3,
        AnomalyModelV3,
        FaultClassifier,
        TransformerVibrationDataset,
        VisualizationManager,
        TrainingLogger,
        CheckpointManager,
        train_stage1,
        run_stage2,
        train_stage3,
        COLORS,
        LABELS,
    )
    V3_AVAILABLE = True
except ImportError as e:
    print(f"[警告] 无法导入transformer_three_stage_v3: {e}")
    V3_AVAILABLE = False


# =============================================================================
# 颜色和标签定义 (备用)
# =============================================================================
if not V3_AVAILABLE:
    COLORS = {
        'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
        'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
        'hetero': '#0072B2', 'zerone': '#E69F00', 'dual': '#CC79A7',
    }


# =============================================================================
# 检测支线状态
# =============================================================================
def check_branch_status(output_root: Path) -> Dict[str, Dict]:
    """
    检测各支线的完成状态
    
    返回:
        {
            'hetero': {'completed': True/False, 'stage': 1/2/3, 'checkpoint': Path or None},
            'zerone': {...},
            'dual': {...}
        }
    """
    status = {}
    
    for branch in ['hetero', 'zerone', 'dual']:
        branch_dir = output_root / f"branch_{branch}"
        
        info = {
            'completed': False,
            'stage': 0,
            'checkpoint': None,
            'has_stage1': False,
            'has_stage2': False,
            'has_stage3': False,
            'eval_result': None,
        }
        
        if not branch_dir.exists():
            status[branch] = info
            continue
        
        # 检查阶段1
        stage1_model = branch_dir / "models" / "stage1_best.pth"
        if stage1_model.exists():
            info['has_stage1'] = True
            info['stage'] = 1
        
        # 检查阶段1检查点
        stage1_ckpt_dir = branch_dir / "checkpoints" / "stage1"
        if stage1_ckpt_dir.exists():
            ckpts = sorted(stage1_ckpt_dir.glob("checkpoint_epoch*.pth"))
            if ckpts:
                info['checkpoint'] = ckpts[-1]
                # 提取epoch数
                try:
                    epoch_str = ckpts[-1].stem.split('epoch')[-1]
                    info['checkpoint_epoch'] = int(epoch_str)
                except:
                    info['checkpoint_epoch'] = 0
        
        # 检查阶段2
        pseudo_labels = branch_dir / "stage2_pseudo_labels" / "pseudo_labels.npz"
        if pseudo_labels.exists():
            info['has_stage2'] = True
            info['stage'] = 2
        
        # 检查阶段3
        stage3_model = branch_dir / "models" / "stage3_best.pth"
        stage3_eval = branch_dir / "stage3_supervised" / "test_evaluation.json"
        if stage3_model.exists():
            info['has_stage3'] = True
            info['stage'] = 3
        
        if stage3_eval.exists():
            info['completed'] = True
            try:
                with open(stage3_eval, 'r', encoding='utf-8') as f:
                    info['eval_result'] = json.load(f)
            except:
                pass
        
        status[branch] = info
    
    return status


def print_status(status: Dict[str, Dict]):
    """打印各支线状态"""
    print("\n" + "="*70)
    print("支线状态检测")
    print("="*70)
    
    for branch, info in status.items():
        print(f"\n【{branch.upper()}】")
        if info['completed']:
            print(f"  ✅ 已完成")
            if info['eval_result']:
                res = info['eval_result']
                print(f"     准确率: {res.get('test_acc', 0):.4f}")
                print(f"     F1分数: {res.get('test_f1', 0):.4f}")
        else:
            print(f"  ⏳ 未完成")
            print(f"     当前阶段: {info['stage']}")
            if info['checkpoint']:
                print(f"     最新检查点: {info['checkpoint'].name}")
                print(f"     检查点轮次: {info.get('checkpoint_epoch', '?')}")
    
    print("\n" + "="*70)


# =============================================================================
# 断点续训函数
# =============================================================================
def resume_training(output_root: Path, data_root: Path, target_branch: str = 'dual'):
    """
    从检查点恢复训练
    
    参数:
        output_root: 输出根目录
        data_root: 数据根目录
        target_branch: 要恢复的支线
    """
    if not V3_AVAILABLE:
        print("[错误] 无法导入transformer_three_stage_v3模块")
        return
    
    status = check_branch_status(output_root)
    print_status(status)
    
    branch_info = status.get(target_branch, {})
    
    if branch_info.get('completed'):
        print(f"\n[跳过] {target_branch} 已完成训练")
        return
    
    # 创建配置
    cfg = ThreeStageConfigV3(
        PROJECT_ROOT=data_root,
        OUTPUT_ROOT=output_root,
        BRANCH_MODE=target_branch,
    )
    cfg.__post_init__()
    
    device = torch.device(cfg.DEVICE)
    
    # 根据当前阶段决定从哪里恢复
    current_stage = branch_info.get('stage', 0)
    
    if current_stage < 1 or not branch_info.get('has_stage1'):
        # 需要训练阶段1
        print(f"\n[恢复] 从阶段1继续训练...")
        
        if branch_info.get('checkpoint'):
            # 有检查点，从检查点恢复
            print(f"  从检查点恢复: {branch_info['checkpoint'].name}")
            model, history = resume_stage1_from_checkpoint(cfg, branch_info['checkpoint'])
        else:
            # 没有检查点，重新开始阶段1
            print(f"  没有检查点，重新开始阶段1")
            model, history = train_stage1(cfg)
    else:
        # 阶段1已完成，加载模型
        print(f"\n[加载] 阶段1模型已存在")
        model = AnomalyModelV3(cfg)
        ckpt = torch.load(cfg.MODEL_DIR / "stage1_best.pth", map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.center = ckpt['center']
        model = model.to(device)
    
    # 阶段2
    if current_stage < 2 or not branch_info.get('has_stage2'):
        print(f"\n[继续] 运行阶段2...")
        pseudo_labels = run_stage2(model, cfg)
    else:
        print(f"\n[加载] 阶段2结果已存在")
        pseudo_path = cfg.STAGE2_DIR / "pseudo_labels.npz"
        data = np.load(pseudo_path, allow_pickle=True)
        pseudo_labels = {k: data[k] for k in data.files}
    
    # 阶段3
    if not branch_info.get('has_stage3') or not branch_info.get('completed'):
        print(f"\n[继续] 运行阶段3...")
        train_stage3(model, pseudo_labels, cfg)
    else:
        print(f"\n[跳过] 阶段3已完成")
    
    print(f"\n[完成] {target_branch} 支线训练完成!")


def resume_stage1_from_checkpoint(cfg: ThreeStageConfigV3, checkpoint_path: Path):
    """
    从检查点恢复阶段1训练
    """
    from torch.utils.data import DataLoader, ConcatDataset
    from tqdm import tqdm
    
    device = torch.device(cfg.DEVICE)
    
    # 加载检查点
    ckpt = torch.load(checkpoint_path, map_location=device)
    start_epoch = ckpt['epoch']
    
    print(f"  从第 {start_epoch} 轮恢复...")
    
    # 重建模型
    model = AnomalyModelV3(cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    if 'center' in ckpt:
        model.center = ckpt['center']
    
    # 重建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    if 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    
    # 重建调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.STAGE1_EPOCHS, last_epoch=start_epoch-1
    )
    if 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    
    # 加载数据
    train_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=False, split_name="TRAIN")
    val_ds = TransformerVibrationDataset(cfg.VAL_DIR, cfg, use_labels=False, split_name="VAL")
    combined_ds = ConcatDataset([train_ds, val_ds])
    train_loader = DataLoader(combined_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              num_workers=0, drop_last=True)
    
    # 初始化工具
    viz = VisualizationManager(cfg)
    logger = TrainingLogger(cfg, "stage1")
    ckpt_mgr = CheckpointManager(cfg, "stage1")
    
    # 恢复历史记录
    history = {
        'epoch': list(range(1, start_epoch + 1)),
        'svdd_loss': ckpt.get('metrics', {}).get('svdd_history', [0]*start_epoch),
        'vae_loss': [0] * start_epoch,
        'total_loss': [ckpt.get('metrics', {}).get('total_loss', 0)] * start_epoch,
        'recon_loss': [0] * start_epoch,
    }
    
    best_loss = ckpt.get('metrics', {}).get('total_loss', float('inf'))
    
    # 继续训练
    print(f"\n  继续训练 (第{start_epoch+1}轮 -> 第{cfg.STAGE1_EPOCHS}轮)...")
    
    for epoch in range(start_epoch, cfg.STAGE1_EPOCHS):
        model.train()
        epoch_svdd, epoch_vae, epoch_total, epoch_recon = 0, 0, 0, 0
        
        beta = min(cfg.BETA_VAE, cfg.BETA_VAE * (epoch / max(cfg.BETA_WARMUP, 1)))
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}", leave=False)
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
                   total_loss=avg_total, recon_loss=avg_recon, lr=scheduler.get_last_lr()[0])
        
        # 保存最佳模型
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'model_state': model.state_dict(),
                'center': model.center,
                'epoch': epoch,
                'loss': best_loss,
            }, cfg.MODEL_DIR / "stage1_best.pth")
        
        # 定期检查点
        if (epoch + 1) % cfg.CHECKPOINT_EVERY == 0:
            ckpt_mgr.save(model, optimizer, epoch + 1,
                         {'svdd_loss': avg_svdd, 'total_loss': avg_total}, scheduler)
        
        # 定期可视化
        if (epoch + 1) % cfg.VIZ_EVERY == 0:
            for lang in cfg.LANGS:
                viz.plot_training_curves(history, "stage1", lang=lang)
        
        if (epoch + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] SVDD: {avg_svdd:.4f} | VAE: {avg_vae:.4f} | Total: {avg_total:.4f}")
    
    # 保存日志
    logger.save_csv()
    
    # 最终可视化
    for lang in cfg.LANGS:
        viz.plot_training_curves(history, "stage1", lang=lang)
    
    print(f"\n  【阶段一恢复完成】最佳损失: {best_loss:.4f}")
    
    return model, history


# =============================================================================
# 三支线对比可视化
# =============================================================================
def generate_comparison_visualization(output_root: Path, lang: str = 'cn'):
    """
    生成三支线对比可视化
    
    参数:
        output_root: 输出根目录
        lang: 语言 ('cn' / 'en')
    """
    # 设置字体
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    status = check_branch_status(output_root)
    
    # 收集结果
    results = {}
    for branch in ['hetero', 'zerone', 'dual']:
        if status[branch].get('eval_result'):
            results[branch] = status[branch]['eval_result']
    
    if len(results) < 2:
        print(f"[警告] 只有 {len(results)} 个支线有结果，无法生成对比")
        return None
    
    # 创建输出目录
    compare_dir = output_root / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 图1: 柱状图对比 ====================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    branches = list(results.keys())
    metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
    
    if lang == 'cn':
        metric_names = ['准确率', 'F1分数', '精确率', '召回率']
        branch_names = {'hetero': '图像分支', 'zerone': '特征分支', 'dual': '双分支融合'}
        title = '三支线性能对比'
        ylabel = '得分'
    else:
        metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        branch_names = {'hetero': 'Image Branch', 'zerone': 'Feature Branch', 'dual': 'Dual Branch'}
        title = 'Three-Branch Performance Comparison'
        ylabel = 'Score'
    
    x = np.arange(len(metrics))
    width = 0.25
    
    colors_list = ['#0072B2', '#E69F00', '#CC79A7']  # 蓝、橙、紫
    
    for i, branch in enumerate(branches):
        values = [results[branch].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, 
                     label=branch_names.get(branch, branch),
                     color=colors_list[i % len(colors_list)],
                     edgecolor='black', linewidth=0.5)
        
        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metrics' if lang == 'en' else '评价指标')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # 添加基线
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Baseline 0.9')
    
    plt.tight_layout()
    save_path = compare_dir / f"branch_comparison_bar_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  保存: {save_path}")
    
    # ==================== 图2: 雷达图 ====================
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, branch in enumerate(branches):
        values = [results[branch].get(m, 0) for m in metrics]
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=branch_names.get(branch, branch),
               color=colors_list[i % len(colors_list)])
        ax.fill(angles, values, alpha=0.25, color=colors_list[i % len(colors_list)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    save_path = compare_dir / f"branch_comparison_radar_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  保存: {save_path}")
    
    # ==================== 图3: 热力图 ====================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_matrix = []
    for branch in branches:
        row = [results[branch].get(m, 0) for m in metrics]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix)
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 标签
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(branches)))
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels([branch_names.get(b, b) for b in branches])
    
    # 数值标注
    for i in range(len(branches)):
        for j in range(len(metrics)):
            val = data_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                   color=color, fontsize=11, fontweight='bold')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score' if lang == 'en' else '得分')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = compare_dir / f"branch_comparison_heatmap_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  保存: {save_path}")
    
    # ==================== 保存JSON汇总 ====================
    summary = {
        'branches': branches,
        'metrics': metrics,
        'results': results,
        'best_branch': max(results.keys(), key=lambda b: results[b].get('test_f1', 0)),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    summary_path = compare_dir / "comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  保存: {summary_path}")
    
    return summary


def generate_all_comparisons(output_root: Path):
    """生成中英文双版本对比可视化"""
    print("\n" + "="*70)
    print("生成三支线对比可视化")
    print("="*70)
    
    for lang in ['cn', 'en']:
        print(f"\n【{lang.upper()}版本】")
        generate_comparison_visualization(output_root, lang=lang)
    
    print("\n[完成] 对比可视化已生成!")


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='断点续训 + 三支线对比')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--compare', action='store_true', help='生成对比可视化')
    parser.add_argument('--compare_only', action='store_true', help='只生成对比可视化')
    parser.add_argument('--branch', type=str, default='dual', 
                       choices=['hetero', 'zerone', 'dual'], help='要恢复的支线')
    parser.add_argument('--output', type=str, default='./three_stage_results_v3', help='输出目录')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--status', action='store_true', help='只显示状态')
    
    args = parser.parse_args()
    
    output_root = Path(args.output)
    
    # 自动检测数据目录
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        # 尝试从现有配置推断
        possible_roots = [
            Path(r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"),
            Path("./data/20251016"),
            Path("./20251016"),
        ]
        data_root = None
        for p in possible_roots:
            if p.exists():
                data_root = p
                break
        
        if data_root is None:
            print("[警告] 未找到数据目录，请使用 --data_root 指定")
            data_root = Path("./data")
    
    print(f"\n输出目录: {output_root}")
    print(f"数据目录: {data_root}")
    
    # 检测状态
    status = check_branch_status(output_root)
    print_status(status)
    
    if args.status:
        return
    
    # 只生成对比
    if args.compare_only:
        generate_all_comparisons(output_root)
        return
    
    # 恢复训练
    if args.resume:
        resume_training(output_root, data_root, args.branch)
    
    # 生成对比
    if args.compare or args.resume:
        # 重新检测状态
        status = check_branch_status(output_root)
        completed_count = sum(1 for s in status.values() if s.get('completed'))
        
        if completed_count >= 2:
            generate_all_comparisons(output_root)
        else:
            print(f"\n[提示] 只有 {completed_count} 个支线完成，需要至少2个支线才能生成对比")


if __name__ == "__main__":
    main()

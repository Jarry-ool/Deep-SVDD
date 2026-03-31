# -*- coding: utf-8 -*-
"""
补充评估脚本：为已完成训练的模型计算完整指标并生成对比图

功能：
1. 加载已训练的Stage3模型
2. 计算4个指标：Accuracy, F1, Precision, Recall
3. 更新evaluation_results.json
4. 生成完整对比图（柱状图、雷达图、热力图）

用法：
    python run_supplementary_eval.py
    python run_supplementary_eval.py --output_root ./three_stage_results_v512
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, '.')

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ThreeStageConfigV5
from models import AnomalyModelV5, FaultClassifierV5
from datasets import LabeledVibrationDataset
from data_manager import DataSplitManager
from gpu_transforms import GPUHeteroTransform


# 语言配置
LANG_CONFIG = {
    'cn': {
        'accuracy': '准确率', 'f1': 'F1分数', 'precision': '精确率', 'recall': '召回率',
        'title_overall': '所有模式性能对比',
        'title_branch': '支线模式对比（最佳结果）',
        'title_fusion': '融合策略对比（Dual模式）',
        'title_zerone': 'Zerone架构对比',
        'title_radar': '多维度性能雷达图',
        'title_heatmap': '模式×指标热力图',
        'branch_names': {'hetero': '异构图像', 'zerone': 'Zerone特征', 'dual': '双分支融合'},
    },
    'en': {
        'accuracy': 'Accuracy', 'f1': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall',
        'title_overall': 'Overall Performance Comparison',
        'title_branch': 'Branch Mode Comparison (Best)',
        'title_fusion': 'Fusion Strategy Comparison (Dual)',
        'title_zerone': 'Zerone Architecture Comparison',
        'title_radar': 'Multi-dimensional Performance Radar',
        'title_heatmap': 'Mode × Metric Heatmap',
        'branch_names': {'hetero': 'Hetero Image', 'zerone': 'Zerone Feature', 'dual': 'Dual Branch'},
    }
}

# 8种模式配置
MODE_CONFIGS = [
    {'name': 'hetero', 'branch': 'hetero', 'fusion': 'concat', 'zerone_cnn': True},
    {'name': 'zerone_cnn', 'branch': 'zerone', 'fusion': 'concat', 'zerone_cnn': True},
    {'name': 'zerone_mlp', 'branch': 'zerone', 'fusion': 'concat', 'zerone_cnn': False},
    {'name': 'dual_concat', 'branch': 'dual', 'fusion': 'concat', 'zerone_cnn': True},
    {'name': 'dual_attention', 'branch': 'dual', 'fusion': 'attention', 'zerone_cnn': True},
    {'name': 'dual_gate', 'branch': 'dual', 'fusion': 'gate', 'zerone_cnn': True},
    {'name': 'dual_gmu_cnn', 'branch': 'dual', 'fusion': 'gmu', 'zerone_cnn': True},
    {'name': 'dual_gmu_mlp', 'branch': 'dual', 'fusion': 'gmu', 'zerone_cnn': False},
]


def evaluate_model(classifier, test_loader, device, gpu_transform=None):
    """评估模型，返回4个指标"""
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            data, zr, labels, _ = batch
            
            if gpu_transform is not None:
                signals = data.to(device)
                img = gpu_transform(signals)
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = classifier(img, zr)
            preds = out['logits'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'test_accuracy': acc,
        'test_f1': f1,
        'test_precision': prec,
        'test_recall': rec,
    }


def load_and_evaluate_mode(mode_cfg, output_root, test_loader, device):
    """加载指定模式的模型并评估"""
    branch = mode_cfg['branch']
    fusion = mode_cfg['fusion']
    zerone_cnn = mode_cfg['zerone_cnn']
    
    # 确定模型路径
    if branch == 'dual':
        branch_dir = output_root / f"branch_{branch}" / f"fusion_{fusion}"
        if not zerone_cnn:
            branch_dir = output_root / f"branch_{branch}" / f"fusion_{fusion}_mlp"
    else:
        branch_dir = output_root / f"branch_{branch}"
        if branch == 'zerone' and not zerone_cnn:
            branch_dir = output_root / "branch_zerone_mlp"
    
    model_dir = branch_dir / "models"
    stage1_path = model_dir / "stage1" / "stage1_best_model.pth"
    stage3_path = model_dir / "stage3" / "stage3_best_model.pth"
    
    if not stage1_path.exists() or not stage3_path.exists():
        print(f"  ⚠ 模型不存在: {mode_cfg['name']}")
        return None
    
    # 创建配置
    cfg = ThreeStageConfigV5(
        PROJECT_ROOT=Path("."),
        OUTPUT_ROOT=output_root,
        BRANCH_MODE=branch,
        FUSION_MODE=fusion,
        ZERONE_USE_CNN=zerone_cnn,
    )
    
    # GPU变换
    use_gpu_hetero = cfg.USE_GPU_HETERO and branch in ('hetero', 'dual')
    gpu_transform = None
    if use_gpu_hetero:
        gpu_transform = GPUHeteroTransform(size=cfg.INPUT_SIZE, device=device)
    
    # 加载Stage1模型
    stage1_model = AnomalyModelV5(
        branch_mode=branch,
        fusion_mode=fusion,
        zerone_use_cnn=zerone_cnn,
    ).to(device)
    
    ckpt = torch.load(stage1_path, map_location=device)
    stage1_model.load_state_dict(ckpt['model_state'])
    
    # 创建分类器
    classifier = FaultClassifierV5(
        encoder=stage1_model.encoder,
        num_classes=2,
        freeze_encoder=True,
    ).to(device)
    
    # 加载Stage3权重
    ckpt = torch.load(stage3_path, map_location=device)
    classifier.load_state_dict(ckpt['model_state'])
    
    # 评估
    results = evaluate_model(classifier, test_loader, device, gpu_transform)
    
    return results


def generate_radar_chart(results, output_dir, lang='cn'):
    """生成雷达图"""
    L = LANG_CONFIG[lang]
    
    # 设置字体
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    metric_labels = [L['accuracy'], L['f1'], L['precision'], L['recall']]
    
    # 准备数据
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, r in enumerate(results):
        values = [r.get(m, 0) * 100 for m in metrics]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, label=r['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title(L['title_radar'], fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    fig.savefig(output_dir / f"compare_radar_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 雷达图已保存: compare_radar_{lang}.png")


def generate_heatmap(results, output_dir, lang='cn'):
    """生成热力图"""
    L = LANG_CONFIG[lang]
    
    # 设置字体
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    metric_labels = [L['accuracy'], L['f1'], L['precision'], L['recall']]
    
    # 构建矩阵
    names = [r['name'] for r in results]
    data = np.array([[r.get(m, 0) * 100 for m in metrics] for r in results])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                xticklabels=metric_labels, yticklabels=names,
                vmin=0, vmax=100, cbar_kws={'label': '%'})
    
    ax.set_title(L['title_heatmap'], fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    fig.savefig(output_dir / f"compare_heatmap_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 热力图已保存: compare_heatmap_{lang}.png")


def generate_bar_charts(results, output_dir, lang='cn'):
    """生成柱状图（4个指标）"""
    L = LANG_CONFIG[lang]
    
    # 设置字体
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # ========== 1. 总体对比 ==========
    fig, ax = plt.subplots(figsize=(16, 6))
    
    names = [r['name'] for r in results]
    accs = [r.get('test_accuracy', 0) * 100 for r in results]
    f1s = [r.get('test_f1', 0) * 100 for r in results]
    precs = [r.get('test_precision', 0) * 100 for r in results]
    recs = [r.get('test_recall', 0) * 100 for r in results]
    
    x = np.arange(len(names))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, accs, width, label=L['accuracy'], color='#3498db', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, f1s, width, label=L['f1'], color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, precs, width, label=L['precision'], color='#2ecc71', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, recs, width, label=L['recall'], color='#9b59b6', alpha=0.8)
    
    ax.set_ylabel('%', fontsize=12)
    ax.set_title(L['title_overall'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
    
    plt.tight_layout()
    fig.savefig(output_dir / f"compare_overall_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 总体对比图已保存: compare_overall_{lang}.png")
    
    # ========== 2. 支线模式对比 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    branch_results = {}
    for r in results:
        branch = r.get('branch', 'unknown')
        if branch not in branch_results or r.get('test_f1', 0) > branch_results[branch].get('test_f1', 0):
            branch_results[branch] = r
    
    branches = ['hetero', 'zerone', 'dual']
    branch_names = [L['branch_names'].get(b, b) for b in branches]
    branch_accs = [branch_results.get(b, {}).get('test_accuracy', 0) * 100 for b in branches]
    branch_f1s = [branch_results.get(b, {}).get('test_f1', 0) * 100 for b in branches]
    branch_precs = [branch_results.get(b, {}).get('test_precision', 0) * 100 for b in branches]
    branch_recs = [branch_results.get(b, {}).get('test_recall', 0) * 100 for b in branches]
    
    x = np.arange(len(branches))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, branch_accs, width, label=L['accuracy'], color='#3498db', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, branch_f1s, width, label=L['f1'], color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, branch_precs, width, label=L['precision'], color='#2ecc71', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, branch_recs, width, label=L['recall'], color='#9b59b6', alpha=0.8)
    
    ax.set_ylabel('%', fontsize=12)
    ax.set_title(L['title_branch'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(branch_names, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / f"compare_branch_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 支线对比图已保存: compare_branch_{lang}.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='补充评估脚本')
    parser.add_argument('--output_root', default='./three_stage_results_v512', help='输出目录')
    parser.add_argument('--filter_output', default='./filtered_output', help='过滤结果目录')
    parser.add_argument('--data_root', default='E:/CODE/code/trans_data/00 振动原始数据', help='原始数据目录')
    parser.add_argument('--skip_eval', action='store_true', help='跳过评估，只生成图表')
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    filter_output = Path(args.filter_output)
    compare_dir = output_root / "mode_comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 尝试加载已有结果
    results_path = compare_dir / "all_results.json"
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"已加载 {len(all_results)} 个模式结果")
    else:
        all_results = []
    
    # 检查是否需要重新评估
    need_eval = not args.skip_eval and (
        len(all_results) == 0 or 
        'test_precision' not in all_results[0]
    )
    
    if need_eval:
        print("\n[1/3] 加载测试数据...")
        
        # 创建配置
        cfg = ThreeStageConfigV5(
            PROJECT_ROOT=Path(args.data_root),
            OUTPUT_ROOT=output_root,
        )
        
        # 加载测试数据
        split_manager = DataSplitManager(cfg)
        split_manager.load_from_filter_output(filter_output)
        
        test_ds = LabeledVibrationDataset(
            split_manager.test_samples, cfg,
            split_name="TEST", normalizer=None
        )
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
        print(f"  测试集: {len(test_ds)} 样本")
        
        print("\n[2/3] 评估所有模式...")
        all_results = []
        
        for mode_cfg in MODE_CONFIGS:
            print(f"\n  评估: {mode_cfg['name']}")
            
            results = load_and_evaluate_mode(mode_cfg, output_root, test_loader, device)
            
            if results:
                results['name'] = mode_cfg['name']
                results['branch'] = mode_cfg['branch']
                results['fusion'] = mode_cfg['fusion']
                results['zerone_cnn'] = mode_cfg['zerone_cnn']
                all_results.append(results)
                print(f"    Acc={results['test_accuracy']:.4f}, F1={results['test_f1']:.4f}, "
                      f"Prec={results['test_precision']:.4f}, Rec={results['test_recall']:.4f}")
            else:
                all_results.append({
                    'name': mode_cfg['name'],
                    'branch': mode_cfg['branch'],
                    'fusion': mode_cfg['fusion'],
                    'zerone_cnn': mode_cfg['zerone_cnn'],
                    'test_accuracy': 0,
                    'test_f1': 0,
                    'test_precision': 0,
                    'test_recall': 0,
                })
        
        # 保存结果
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {results_path}")
    
    # 生成图表
    print("\n[3/3] 生成对比图表...")
    
    valid_results = [r for r in all_results if r.get('test_accuracy', 0) > 0]
    
    if len(valid_results) == 0:
        print("  ⚠ 没有有效结果，跳过图表生成")
        return
    
    for lang in ['cn', 'en']:
        print(f"\n  [{lang.upper()}]")
        generate_bar_charts(valid_results, compare_dir, lang)
        generate_radar_chart(valid_results, compare_dir, lang)
        generate_heatmap(valid_results, compare_dir, lang)
    
    print(f"\n✅ 完成！所有图表已保存到: {compare_dir}")
    print("   生成的图表:")
    print("     - compare_overall_*.png   总体对比柱状图")
    print("     - compare_branch_*.png    支线模式对比柱状图")
    print("     - compare_radar_*.png     雷达图")
    print("     - compare_heatmap_*.png   热力图")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
compare_experiments.py
=======================

跨实验对比可视化脚本

【功能】
    1. 自动扫描多个实验目录
    2. 提取训练日志和评估结果
    3. 生成对比图表：
       - 训练曲线对比
       - TEST指标对比（柱状图）
       - 融合策略对比（雷达图）
       - 域适应效果对比
       - 消融实验汇总表

【使用方法】
    # 对比所有实验
    python compare_experiments.py --results_dir ./three_stage_results_v5
    
    # 只对比特定实验
    python compare_experiments.py --results_dir ./three_stage_results_v5 \
        --experiments branch_dual_gmu_mdrop_da_dann branch_dual_attention_mdrop_da
    
    # 指定输出目录
    python compare_experiments.py --results_dir ./three_stage_results_v5 \
        --output ./comparison_results

Author: V5实验对比工具
"""

import os
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# 颜色方案
COLORS = [
    '#3498db',  # 蓝
    '#e74c3c',  # 红
    '#2ecc71',  # 绿
    '#9b59b6',  # 紫
    '#f39c12',  # 橙
    '#1abc9c',  # 青
    '#e91e63',  # 粉
    '#00bcd4',  # 浅蓝
]

# 实验名称映射（美化显示）
NAME_MAPPING = {
    'branch_dual_gmu_mdrop_da_dann': 'Dual+GMU+DA+DANN',
    'branch_dual_gmu_mdrop_da': 'Dual+GMU+DA',
    'branch_dual_gmu_mdrop': 'Dual+GMU+MDrop',
    'branch_dual_gmu': 'Dual+GMU',
    'branch_dual_attention_mdrop_da_dann': 'Dual+Attn+DA+DANN',
    'branch_dual_attention_mdrop_da': 'Dual+Attn+DA',
    'branch_dual_attention': 'Dual+Attention',
    'branch_dual_gate_mdrop_da_dann': 'Dual+Gate+DA+DANN',
    'branch_dual_gate_mdrop_da': 'Dual+Gate+DA',
    'branch_dual_gate': 'Dual+Gate',
    'branch_dual_concat_mdrop_da_dann': 'Dual+Concat+DA+DANN',
    'branch_dual_concat_mdrop_da': 'Dual+Concat+DA',
    'branch_dual_concat': 'Dual+Concat',
    'branch_hetero': 'Hetero-Only',
    'branch_zerone': 'Zerone-Only',
}


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    name: str
    display_name: str
    path: Path
    
    # Stage3训练历史
    train_history: Dict[str, List[float]] = None
    
    # TEST评估结果
    test_acc: float = 0.0
    test_f1: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    n_errors: int = 0
    
    # 配置信息
    config: Dict = None
    
    def __post_init__(self):
        if self.train_history is None:
            self.train_history = {}
        if self.config is None:
            self.config = {}


def load_experiment(exp_dir: Path) -> Optional[ExperimentResult]:
    """加载单个实验结果"""
    exp_name = exp_dir.name
    display_name = NAME_MAPPING.get(exp_name, exp_name)
    
    result = ExperimentResult(
        name=exp_name,
        display_name=display_name,
        path=exp_dir
    )
    
    # 加载训练日志
    log_file = exp_dir / "logs" / "stage3_training_log.csv"
    if log_file.exists():
        result.train_history = load_training_log(log_file)
    
    # 加载TEST评估结果
    eval_file = exp_dir / "stage3_supervised" / "test_evaluation.json"
    if eval_file.exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            result.test_acc = eval_data.get('test_acc', 0)
            result.test_f1 = eval_data.get('test_f1', 0)
            result.test_precision = eval_data.get('test_precision', 0)
            result.test_recall = eval_data.get('test_recall', 0)
            result.n_errors = eval_data.get('n_errors', 0)
            result.config = eval_data.get('config', {})
    
    # 检查是否有有效数据
    if result.test_f1 > 0 or result.train_history:
        return result
    
    return None


def load_training_log(log_file: Path) -> Dict[str, List[float]]:
    """加载训练日志CSV"""
    history = defaultdict(list)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                try:
                    history[key].append(float(value))
                except (ValueError, TypeError):
                    pass
    
    return dict(history)


def scan_experiments(results_dir: Path, experiment_names: List[str] = None) -> List[ExperimentResult]:
    """扫描所有实验目录"""
    experiments = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        if experiment_names and exp_dir.name not in experiment_names:
            continue
        
        exp = load_experiment(exp_dir)
        if exp:
            experiments.append(exp)
            print(f"  ✓ 加载: {exp.display_name} (F1={exp.test_f1:.4f})")
    
    # 按F1分数排序
    experiments.sort(key=lambda x: x.test_f1, reverse=True)
    
    return experiments


# =============================================================================
# 可视化函数
# =============================================================================

def plot_training_curves_comparison(experiments: List[ExperimentResult], 
                                    output_dir: Path, lang: str = 'cn'):
    """对比训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    title = '训练曲线对比' if lang == 'cn' else 'Training Curves Comparison'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    metrics = [
        ('train_loss', '训练损失' if lang == 'cn' else 'Train Loss'),
        ('val_f1', '验证F1' if lang == 'cn' else 'Validation F1'),
        ('val_acc', '验证准确率' if lang == 'cn' else 'Validation Accuracy'),
        ('mmd_loss', 'MMD损失' if lang == 'cn' else 'MMD Loss'),
    ]
    
    for idx, (metric, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for i, exp in enumerate(experiments):
            if metric in exp.train_history and exp.train_history[metric]:
                epochs = exp.train_history.get('epoch', list(range(1, len(exp.train_history[metric]) + 1)))
                values = exp.train_history[metric]
                ax.plot(epochs, values, color=COLORS[i % len(COLORS)], 
                       label=exp.display_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"training_curves_comparison_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {save_path.name}")


def plot_test_metrics_bar(experiments: List[ExperimentResult], 
                          output_dir: Path, lang: str = 'cn'):
    """TEST指标柱状图对比"""
    fig, ax = plt.subplots(figsize=(max(10, len(experiments) * 1.5), 7))
    
    title = 'TEST集评估指标对比' if lang == 'cn' else 'TEST Set Evaluation Comparison'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
    metric_names = {
        'test_acc': '准确率' if lang == 'cn' else 'Accuracy',
        'test_f1': 'F1分数' if lang == 'cn' else 'F1 Score',
        'test_precision': '精确率' if lang == 'cn' else 'Precision',
        'test_recall': '召回率' if lang == 'cn' else 'Recall',
    }
    
    x = np.arange(len(experiments))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [getattr(exp, metric, 0) for exp in experiments]
        bars = ax.bar(x + i * width, values, width, label=metric_names[metric], 
                     color=COLORS[i], alpha=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_ylabel('分数' if lang == 'cn' else 'Score')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([exp.display_name for exp in experiments], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / f"test_metrics_comparison_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {save_path.name}")


def plot_radar_chart(experiments: List[ExperimentResult], 
                     output_dir: Path, lang: str = 'cn'):
    """雷达图对比"""
    # 最多显示6个实验
    experiments = experiments[:6]
    
    metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
    metric_names = {
        'test_acc': '准确率' if lang == 'cn' else 'Acc',
        'test_f1': 'F1' if lang == 'cn' else 'F1',
        'test_precision': '精确率' if lang == 'cn' else 'Prec',
        'test_recall': '召回率' if lang == 'cn' else 'Recall',
    }
    
    labels = [metric_names[m] for m in metrics]
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    title = '多维度性能对比' if lang == 'cn' else 'Multi-dimensional Performance'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    for i, exp in enumerate(experiments):
        values = [getattr(exp, m, 0) for m in metrics]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, color=COLORS[i % len(COLORS)], linewidth=2, 
               label=exp.display_name, alpha=0.8)
        ax.fill(angles, values, color=COLORS[i % len(COLORS)], alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"radar_comparison_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {save_path.name}")


def plot_domain_adaptation_comparison(experiments: List[ExperimentResult], 
                                      output_dir: Path, lang: str = 'cn'):
    """域适应效果对比"""
    # 筛选有域适应数据的实验
    da_experiments = [exp for exp in experiments 
                      if 'mmd_loss' in exp.train_history and exp.train_history['mmd_loss']]
    
    if not da_experiments:
        print("  ⚠ 没有域适应数据，跳过此图")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    title = '域适应损失对比' if lang == 'cn' else 'Domain Adaptation Loss Comparison'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    da_metrics = [
        ('mmd_loss', 'MMD Loss'),
        ('coral_loss', 'CORAL Loss'),
        ('dann_loss', 'DANN Loss'),
    ]
    
    for idx, (metric, metric_name) in enumerate(da_metrics):
        ax = axes[idx]
        
        for i, exp in enumerate(da_experiments):
            if metric in exp.train_history and exp.train_history[metric]:
                epochs = exp.train_history.get('epoch', list(range(1, len(exp.train_history[metric]) + 1)))
                values = exp.train_history[metric]
                ax.plot(epochs, values, color=COLORS[i % len(COLORS)], 
                       label=exp.display_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"domain_adaptation_comparison_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {save_path.name}")


def plot_ablation_heatmap(experiments: List[ExperimentResult], 
                          output_dir: Path, lang: str = 'cn'):
    """消融实验热力图"""
    # 提取配置信息
    configs = []
    for exp in experiments:
        config = {
            'name': exp.display_name,
            'F1': exp.test_f1,
            'GMU': 'gmu' in exp.name.lower(),
            'Attention': 'attention' in exp.name.lower(),
            'Gate': 'gate' in exp.name.lower(),
            'Concat': 'concat' in exp.name.lower(),
            'MDrop': 'mdrop' in exp.name.lower(),
            'DA': '_da' in exp.name.lower(),
            'DANN': 'dann' in exp.name.lower(),
        }
        configs.append(config)
    
    if not configs:
        return
    
    # 构建矩阵
    feature_cols = ['GMU', 'Attention', 'Gate', 'Concat', 'MDrop', 'DA', 'DANN']
    matrix = []
    row_labels = []
    f1_scores = []
    
    for config in configs:
        row = [1 if config.get(col, False) else 0 for col in feature_cols]
        matrix.append(row)
        row_labels.append(config['name'])
        f1_scores.append(config['F1'])
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(configs) * 0.5)))
    
    title = '消融实验配置与F1对比' if lang == 'cn' else 'Ablation Study: Config vs F1'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 绘制热力图
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=feature_cols, yticklabels=row_labels,
                ax=ax, cbar_kws={'label': '启用'})
    
    # 添加F1分数
    for i, f1 in enumerate(f1_scores):
        ax.text(len(feature_cols) + 0.5, i + 0.5, f'F1={f1:.3f}', 
               va='center', fontsize=10, fontweight='bold',
               color='green' if f1 > 0.7 else ('orange' if f1 > 0.5 else 'red'))
    
    ax.set_xlabel('配置项' if lang == 'cn' else 'Configuration')
    ax.set_ylabel('实验' if lang == 'cn' else 'Experiment')
    
    plt.tight_layout()
    save_path = output_dir / f"ablation_heatmap_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {save_path.name}")


def generate_summary_table(experiments: List[ExperimentResult], 
                           output_dir: Path, lang: str = 'cn'):
    """生成汇总表格"""
    
    # Markdown表格
    md_lines = []
    if lang == 'cn':
        md_lines.append("# 实验结果汇总\n")
        md_lines.append("| 排名 | 实验名称 | 准确率 | F1分数 | 精确率 | 召回率 | 错误数 |")
        md_lines.append("|:----:|:--------|:------:|:------:|:------:|:------:|:------:|")
    else:
        md_lines.append("# Experiment Results Summary\n")
        md_lines.append("| Rank | Experiment | Accuracy | F1 Score | Precision | Recall | Errors |")
        md_lines.append("|:----:|:-----------|:--------:|:--------:|:---------:|:------:|:------:|")
    
    for i, exp in enumerate(experiments, 1):
        md_lines.append(
            f"| {i} | {exp.display_name} | {exp.test_acc:.4f} | {exp.test_f1:.4f} | "
            f"{exp.test_precision:.4f} | {exp.test_recall:.4f} | {exp.n_errors} |"
        )
    
    md_content = "\n".join(md_lines)
    
    md_path = output_dir / f"summary_table_{lang}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  ✓ 保存: {md_path.name}")
    
    # CSV表格
    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Experiment', 'Accuracy', 'F1', 'Precision', 'Recall', 'Errors'])
        for i, exp in enumerate(experiments, 1):
            writer.writerow([i, exp.display_name, exp.test_acc, exp.test_f1, 
                           exp.test_precision, exp.test_recall, exp.n_errors])
    print(f"  ✓ 保存: {csv_path.name}")
    
    # 打印到控制台
    print("\n" + "="*80)
    print(md_content)
    print("="*80)


def plot_f1_ranking(experiments: List[ExperimentResult], 
                    output_dir: Path, lang: str = 'cn'):
    """F1分数排名图"""
    fig, ax = plt.subplots(figsize=(10, max(6, len(experiments) * 0.4)))
    
    title = 'F1分数排名' if lang == 'cn' else 'F1 Score Ranking'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    names = [exp.display_name for exp in experiments]
    f1_scores = [exp.test_f1 for exp in experiments]
    
    # 颜色根据分数变化
    colors = []
    for f1 in f1_scores:
        if f1 >= 0.8:
            colors.append('#2ecc71')  # 绿
        elif f1 >= 0.6:
            colors.append('#3498db')  # 蓝
        elif f1 >= 0.4:
            colors.append('#f39c12')  # 橙
        else:
            colors.append('#e74c3c')  # 红
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, f1_scores, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{f1:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('F1 Score')
    ax.set_xlim(0, 1.1)
    ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='优秀 (0.8)')
    ax.axvline(0.6, color='blue', linestyle='--', alpha=0.5, label='良好 (0.6)')
    ax.axvline(0.4, color='orange', linestyle='--', alpha=0.5, label='及格 (0.4)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = output_dir / f"f1_ranking_{lang}.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 保存: {save_path.name}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='跨实验对比可视化工具')
    parser.add_argument('--results_dir', type=str, default='./three_stage_results_v5',
                       help='实验结果根目录')
    parser.add_argument('--experiments', type=str, nargs='*', default=None,
                       help='指定要对比的实验名称（目录名）')
    parser.add_argument('--output', type=str, default=None,
                       help='对比结果输出目录')
    parser.add_argument('--lang', type=str, choices=['cn', 'en'], default='cn',
                       help='可视化语言')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[错误] 结果目录不存在: {results_dir}")
        return
    
    output_dir = Path(args.output) if args.output else results_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("跨实验对比可视化")
    print("="*60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"语言: {args.lang}")
    
    # 扫描实验
    print("\n[1/8] 扫描实验目录...")
    experiments = scan_experiments(results_dir, args.experiments)
    
    if not experiments:
        print("[错误] 没有找到有效的实验结果")
        return
    
    print(f"\n共找到 {len(experiments)} 个有效实验")
    
    # 生成可视化
    print("\n[2/8] 生成训练曲线对比...")
    plot_training_curves_comparison(experiments, output_dir, args.lang)
    
    print("\n[3/8] 生成TEST指标柱状图...")
    plot_test_metrics_bar(experiments, output_dir, args.lang)
    
    print("\n[4/8] 生成雷达图...")
    plot_radar_chart(experiments, output_dir, args.lang)
    
    print("\n[5/8] 生成域适应对比...")
    plot_domain_adaptation_comparison(experiments, output_dir, args.lang)
    
    print("\n[6/8] 生成消融实验热力图...")
    plot_ablation_heatmap(experiments, output_dir, args.lang)
    
    print("\n[7/8] 生成F1排名图...")
    plot_f1_ranking(experiments, output_dir, args.lang)
    
    print("\n[8/8] 生成汇总表格...")
    generate_summary_table(experiments, output_dir, args.lang)
    
    print(f"\n【完成】对比结果保存至: {output_dir}")
    print("\n生成的文件:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

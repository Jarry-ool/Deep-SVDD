# -*- coding: utf-8 -*-
"""
通用补跑可视化脚本：为所有已完成训练的模式生成完整可视化

功能：
1. 自动检测已完成训练的模式
2. 根据模式类型(hetero/zerone/dual)生成对应的可视化
3. 生成统一的对比图表（雷达图、热力图等）

用法：
    python run_all_visualizations.py                    # 处理所有已完成模式
    python run_all_visualizations.py --mode hetero      # 只处理指定模式
    python run_all_visualizations.py --skip_eval        # 跳过评估，只用已有结果生成对比图
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, '.')

import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from tqdm import tqdm

from config import ThreeStageConfigV5
from models import AnomalyModelV5, FaultClassifierV5
from datasets import LabeledVibrationDataset
from data_manager import DataSplitManager
from gpu_transforms import GPUHeteroTransform
from visualization import VisualizationManager
from utils import GLOBAL_NORMALIZER


# ============================================================
# 模式配置
# ============================================================
MODE_CONFIGS = {
    'hetero': {'branch': 'hetero', 'fusion': 'concat', 'zerone_cnn': True},
    'zerone_cnn': {'branch': 'zerone', 'fusion': 'concat', 'zerone_cnn': True},
    'zerone_mlp': {'branch': 'zerone', 'fusion': 'concat', 'zerone_cnn': False},
    'dual_concat': {'branch': 'dual', 'fusion': 'concat', 'zerone_cnn': True},
    'dual_attention': {'branch': 'dual', 'fusion': 'attention', 'zerone_cnn': True},
    'dual_gate': {'branch': 'dual', 'fusion': 'gate', 'zerone_cnn': True},
    'dual_gmu_cnn': {'branch': 'dual', 'fusion': 'gmu', 'zerone_cnn': True},
    'dual_gmu_mlp': {'branch': 'dual', 'fusion': 'gmu', 'zerone_cnn': False},
}

# 语言配置
LANG_CONFIG = {
    'cn': {
        'accuracy': '准确率', 'f1': 'F1分数', 'precision': '精确率', 'recall': '召回率',
        'title_radar': '多维度性能雷达图',
        'title_heatmap': '模式×指标热力图',
        'title_overall': '所有模式性能对比',
        'branch_names': {'hetero': 'Hetero图像', 'zerone': 'Zerone特征', 'dual': '双分支融合'},
    },
    'en': {
        'accuracy': 'Accuracy', 'f1': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall',
        'title_radar': 'Multi-dimensional Performance Radar',
        'title_heatmap': 'Mode × Metric Heatmap',
        'title_overall': 'Overall Performance Comparison',
        'branch_names': {'hetero': 'Hetero Image', 'zerone': 'Zerone Feature', 'dual': 'Dual Branch'},
    }
}


# ============================================================
# 工具函数
# ============================================================
def get_mode_dir(output_root, mode_name):
    """获取模式对应的目录"""
    cfg = MODE_CONFIGS[mode_name]
    branch = cfg['branch']
    fusion = cfg['fusion']
    zerone_cnn = cfg['zerone_cnn']
    
    if branch == 'dual':
        branch_dir = output_root / f"branch_{branch}" / f"fusion_{fusion}"
        if not zerone_cnn:
            branch_dir = output_root / f"branch_{branch}" / f"fusion_{fusion}_mlp"
    else:
        branch_dir = output_root / f"branch_{branch}"
        if branch == 'zerone' and not zerone_cnn:
            branch_dir = output_root / "branch_zerone_mlp"
    
    return branch_dir


def check_mode_completed(output_root, mode_name):
    """检查模式是否已完成训练"""
    branch_dir = get_mode_dir(output_root, mode_name)
    stage3_path = branch_dir / "models" / "stage3" / "stage3_best_model.pth"
    return stage3_path.exists()


def load_models(mode_name, output_root, device):
    """加载Stage1和Stage3模型"""
    cfg_dict = MODE_CONFIGS[mode_name]
    branch_dir = get_mode_dir(output_root, mode_name)
    
    model_dir = branch_dir / "models"
    stage1_path = model_dir / "stage1" / "stage1_best_model.pth"
    stage3_path = model_dir / "stage3" / "stage3_best_model.pth"
    
    if not stage1_path.exists() or not stage3_path.exists():
        return None, None, None
    
    # 创建配置
    cfg = ThreeStageConfigV5(
        PROJECT_ROOT=Path("."),
        OUTPUT_ROOT=output_root,
        BRANCH_MODE=cfg_dict['branch'],
        FUSION_MODE=cfg_dict['fusion'],
        ZERONE_USE_CNN=cfg_dict['zerone_cnn'],
    )
    
    # 加载Stage1模型
    stage1_model = AnomalyModelV5(
        branch_mode=cfg_dict['branch'],
        fusion_mode=cfg_dict['fusion'],
        zerone_use_cnn=cfg_dict['zerone_cnn'],
    ).to(device)
    
    ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
    stage1_model.load_state_dict(ckpt['model_state'])
    stage1_model.eval()
    
    # 创建分类器
    classifier = FaultClassifierV5(
        encoder=stage1_model.encoder,
        num_classes=2,
        freeze_encoder=True,
    ).to(device)
    
    # 加载Stage3权重
    ckpt = torch.load(stage3_path, map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt['model_state'])
    classifier.eval()
    
    return stage1_model, classifier, cfg


# ============================================================
# 单模式可视化
# ============================================================
def process_single_mode(mode_name, output_root, test_loader, val_loader, device):
    """处理单个模式的可视化"""
    print(f"\n{'='*60}")
    print(f"处理模式: {mode_name}")
    print(f"{'='*60}")
    
    cfg_dict = MODE_CONFIGS[mode_name]
    branch = cfg_dict['branch']
    
    # 加载模型
    stage1_model, classifier, cfg = load_models(mode_name, output_root, device)
    if classifier is None:
        print(f"  ⚠ 模型未找到，跳过")
        return None
    
    branch_dir = get_mode_dir(output_root, mode_name)
    
    # GPU变换
    use_gpu_hetero = cfg.USE_GPU_HETERO and branch in ('hetero', 'dual')
    gpu_transform = None
    if use_gpu_hetero:
        gpu_transform = GPUHeteroTransform(size=cfg.INPUT_SIZE, device=device)
    
    # ========== 1. 收集预测结果 ==========
    print("\n  [1/4] 收集预测结果...")
    all_preds, all_labels, all_probs = [], [], []
    all_features = []
    error_info = []
    
    # 双分支特征 (仅dual模式)
    hetero_feats_list, zerone_feats_list, gate_weights_list = [], [], []
    
    classifier.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="    评估", leave=False, ncols=80):
            data, zr, labels, _ = batch
            
            if use_gpu_hetero and gpu_transform is not None:
                img = gpu_transform(data.to(device))
            else:
                img = data.to(device)
            zr = zr.to(device)
            
            out = classifier(img, zr)
            preds = out['logits'].argmax(dim=1)
            probs = F.softmax(out['logits'], dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            
            if 'h' in out:
                all_features.extend(out['h'].cpu().numpy())
            
            # 双分支特征 (限制数量)
            if branch == 'dual' and len(hetero_feats_list) < 500:
                if 'h_hetero' in out:
                    hetero_feats_list.extend(out['h_hetero'].cpu().numpy())
                if 'h_zerone' in out:
                    zerone_feats_list.extend(out['h_zerone'].cpu().numpy())
                if 'gate_weights' in out:
                    gate_weights_list.extend(out['gate_weights'].cpu().numpy())
            
            # 错误样本
            for i in range(len(preds)):
                if preds[i].item() != labels[i].item() and len(error_info) < 16:
                    error_info.append({
                        'image': img[i].cpu(),
                        'true': labels[i].item(),
                        'pred': preds[i].item()
                    })
    
    # 计算指标
    metrics = {
        'name': mode_name,
        'branch': branch,
        'fusion': cfg_dict['fusion'],
        'zerone_cnn': cfg_dict['zerone_cnn'],
        'test_accuracy': accuracy_score(all_labels, all_preds),
        'test_f1': f1_score(all_labels, all_preds, average='weighted'),
        'test_precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'test_recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
    }
    print(f"    Acc={metrics['test_accuracy']:.4f}, F1={metrics['test_f1']:.4f}, "
          f"Prec={metrics['test_precision']:.4f}, Rec={metrics['test_recall']:.4f}")
    
    all_features = np.array(all_features) if all_features else None
    all_labels_arr = np.array(all_labels)
    
    # ========== 2. Stage3可视化 ==========
    print("\n  [2/4] 生成Stage3可视化...")
    viz = VisualizationManager(branch_dir / "stage3_classify")
    
    for lang in ['cn', 'en']:
        # 错误分析
        if error_info:
            viz.plot_error_analysis(error_info, all_preds, all_labels, all_probs, lang=lang)
        
        # 特征可分性
        if all_features is not None and len(all_features) > 0:
            viz.plot_feature_separability(all_features, all_labels_arr, lang=lang)
    
    # ========== 3. 双分支/门控分析 (仅dual模式) ==========
    if branch == 'dual':
        print("\n  [3/4] 生成双分支/门控分析...")
        labels_sub = all_labels_arr[:len(hetero_feats_list)] if hetero_feats_list else all_labels_arr[:500]
        
        for lang in ['cn', 'en']:
            if hetero_feats_list and zerone_feats_list:
                viz.plot_dual_branch_analysis(
                    np.array(hetero_feats_list),
                    np.array(zerone_feats_list),
                    labels_sub, lang=lang
                )
            if gate_weights_list:
                viz.plot_gate_distribution(np.array(gate_weights_list), labels_sub, lang=lang)
    else:
        print("\n  [3/4] 跳过双分支分析 (非dual模式)")
    
    # ========== 4. Zerone特征分析 (仅zerone_mlp或dual_mlp模式，需要1D向量) ==========
    zerone_use_cnn = cfg_dict.get('zerone_cnn', True)
    if branch in ('zerone', 'dual') and not zerone_use_cnn:
        print("\n  [4/4] 生成Zerone特征分析 (MLP模式)...")
        
        # 收集zerone特征 (1D向量)
        zerone_feats, zerone_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                _, zr, labels, _ = batch
                zerone_feats.extend(zr.numpy())
                zerone_labels.extend(labels.tolist())
                if len(zerone_feats) >= 2000:
                    break
        
        zerone_feats = np.array(zerone_feats)
        zerone_labels = np.array(zerone_labels)
        
        # 确认是1D向量格式
        if zerone_feats.ndim == 2 and zerone_feats.shape[1] >= 1200:
            viz_preview = VisualizationManager(branch_dir / "stage1_anomaly")
            for lang in ['cn', 'en']:
                viz_preview.plot_zerone_stft_stats(zerone_feats, zerone_labels, lang=lang)
                viz_preview.plot_time_domain_correlation(zerone_feats, zerone_labels, 'val', lang=lang)
                viz_preview.plot_psd_waterfall(zerone_feats, zerone_labels, 'val', lang=lang)
        else:
            print(f"    ⚠ Zerone特征格式不符合预期: {zerone_feats.shape}，跳过")
    elif branch in ('zerone', 'dual') and zerone_use_cnn:
        print("\n  [4/4] 跳过Zerone特征分析 (CNN模式用图像，非1D向量)")
    else:
        print("\n  [4/4] 跳过Zerone分析 (非zerone/dual模式)")
    
    print(f"\n  ✅ {mode_name} 完成!")
    return metrics


# ============================================================
# 综合对比图表
# ============================================================
def generate_comparison_charts(all_results, output_dir):
    """生成综合对比图表"""
    print(f"\n{'='*60}")
    print("生成综合对比图表")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤有效结果
    valid_results = [r for r in all_results if r.get('test_accuracy', 0) > 0]
    if not valid_results:
        print("  ⚠ 没有有效结果")
        return
    
    for lang in ['cn', 'en']:
        L = LANG_CONFIG[lang]
        
        # 设置字体
        if lang == 'cn':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        # ========== 1. 柱状图 (4指标) ==========
        print(f"\n  [{lang.upper()}] 总体对比柱状图...")
        fig, ax = plt.subplots(figsize=(16, 6))
        
        names = [r['name'] for r in valid_results]
        metrics_keys = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
        metrics_labels = [L['accuracy'], L['f1'], L['precision'], L['recall']]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        x = np.arange(len(names))
        width = 0.2
        
        for i, (key, label, color) in enumerate(zip(metrics_keys, metrics_labels, colors)):
            values = [r.get(key, 0) * 100 for r in valid_results]
            bars = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color, alpha=0.8)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(L['title_overall'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_overall_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 2. 雷达图 ==========
        print(f"  [{lang.upper()}] 雷达图...")
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
        angles += angles[:1]
        
        colors_radar = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
        
        for i, r in enumerate(valid_results):
            values = [r.get(k, 0) * 100 for k in metrics_keys]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=r['name'], color=colors_radar[i])
            ax.fill(angles, values, alpha=0.1, color=colors_radar[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title(L['title_radar'], fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_radar_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 3. 热力图 ==========
        print(f"  [{lang.upper()}] 热力图...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        data = np.array([[r.get(k, 0) * 100 for k in metrics_keys] for r in valid_results])
        
        sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                    xticklabels=metrics_labels, yticklabels=names,
                    vmin=0, vmax=100, cbar_kws={'label': '%'})
        
        ax.set_title(L['title_heatmap'], fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_heatmap_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 4. 分支模式对比 ==========
        print(f"  [{lang.upper()}] 分支对比柱状图...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        branch_best = {}
        for r in valid_results:
            b = r.get('branch', 'unknown')
            if b not in branch_best or r.get('test_f1', 0) > branch_best[b].get('test_f1', 0):
                branch_best[b] = r
        
        branches = ['hetero', 'zerone', 'dual']
        branch_names_display = [L['branch_names'].get(b, b) for b in branches]
        
        x = np.arange(len(branches))
        width = 0.2
        
        for i, (key, label, color) in enumerate(zip(metrics_keys, metrics_labels, colors)):
            values = [branch_best.get(b, {}).get(key, 0) * 100 for b in branches]
            ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color, alpha=0.8)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(f"{L['branch_names']['hetero']} vs {L['branch_names']['zerone']} vs {L['branch_names']['dual']}", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(branch_names_display, fontsize=11)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_branch_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ========== 5. Zerone架构对比 (CNN vs MLP) ==========
        print(f"  [{lang.upper()}] Zerone架构对比...")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 按name查找结果
        zerone_cnn = next((r for r in valid_results if r.get('name') == 'zerone_cnn'), {})
        zerone_mlp = next((r for r in valid_results if r.get('name') == 'zerone_mlp'), {})
        dual_cnn = next((r for r in valid_results if r.get('name') == 'dual_gmu_cnn'), {})
        dual_mlp = next((r for r in valid_results if r.get('name') == 'dual_gmu_mlp'), {})
        
        zerone_names = {'cnn': '(图像)', 'mlp': '(向量)'} if lang == 'cn' else {'cnn': '(Image)', 'mlp': '(Vector)'}
        title_zerone = 'Zerone架构性能对比 (CNN vs MLP)' if lang == 'cn' else 'Zerone Architecture Comparison (CNN vs MLP)'
        
        categories = [f"Zerone\nCNN\n{zerone_names['cnn']}", f"Zerone\nMLP\n{zerone_names['mlp']}",
                     f"Dual+GMU\nCNN\n{zerone_names['cnn']}", f"Dual+GMU\nMLP\n{zerone_names['mlp']}"]
        cnn_mlp_accs = [
            zerone_cnn.get('test_accuracy', 0) * 100,
            zerone_mlp.get('test_accuracy', 0) * 100,
            dual_cnn.get('test_accuracy', 0) * 100,
            dual_mlp.get('test_accuracy', 0) * 100,
        ]
        cnn_mlp_f1s = [
            zerone_cnn.get('test_f1', 0) * 100,
            zerone_mlp.get('test_f1', 0) * 100,
            dual_cnn.get('test_f1', 0) * 100,
            dual_mlp.get('test_f1', 0) * 100,
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cnn_mlp_accs, width, label=L['accuracy'], color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, cnn_mlp_f1s, width, label=L['f1'], color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('%', fontsize=12)
        ax.set_title(title_zerone, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"compare_zerone_arch_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    # 保存结果JSON
    with open(output_dir / "all_results.json", 'w', encoding='utf-8') as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 对比图表已保存到: {output_dir}")
    print("   - compare_overall_*.png   总体对比柱状图")
    print("   - compare_radar_*.png     雷达图")
    print("   - compare_heatmap_*.png   热力图")
    print("   - compare_branch_*.png    分支对比柱状图")
    print("   - compare_zerone_arch_*.png Zerone架构对比")
    print("   - all_results.json        完整结果数据")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='通用补跑可视化脚本')
    parser.add_argument('--mode', type=str, default=None, 
                        choices=list(MODE_CONFIGS.keys()),
                        help='指定单个模式 (不指定则处理所有)')
    parser.add_argument('--skip_eval', action='store_true', 
                        help='跳过评估，只用已有结果生成对比图')
    parser.add_argument('--output_root', default='./three_stage_results_v512', help='输出目录')
    parser.add_argument('--filter_output', default='./filtered_output', help='过滤结果目录')
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    filter_output = Path(args.filter_output)
    compare_dir = output_root / "mode_comparison"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"输出目录: {output_root}")
    
    # 检测已完成的模式
    completed_modes = []
    for mode_name in MODE_CONFIGS.keys():
        if check_mode_completed(output_root, mode_name):
            completed_modes.append(mode_name)
    
    print(f"\n已完成的模式 ({len(completed_modes)}/{len(MODE_CONFIGS)}): {completed_modes}")
    
    if not completed_modes:
        print("⚠ 没有找到已完成的模式")
        return
    
    # 确定要处理的模式
    if args.mode:
        if args.mode not in completed_modes:
            print(f"⚠ 模式 {args.mode} 未完成训练")
            return
        modes_to_process = [args.mode]
    else:
        modes_to_process = completed_modes
    
    all_results = []
    
    if not args.skip_eval:
        # 加载归一化器
        normalizer_path = output_root / "global_normalizer.npz"
        if normalizer_path.exists():
            GLOBAL_NORMALIZER.load(normalizer_path)
            print(f"归一化器已加载: {normalizer_path}")
        
        # 加载数据
        print("\n加载数据集...")
        
        # 创建一个临时cfg用于加载split信息
        temp_cfg = ThreeStageConfigV5(
            PROJECT_ROOT=Path("."),
            OUTPUT_ROOT=output_root,
        )
        
        split_manager = DataSplitManager(temp_cfg)
        split_manager.load_from_filter_output(filter_output)
        
        # 处理每个模式 - 为每个模式创建对应的数据集
        for mode_name in modes_to_process:
            # 创建该模式专用的cfg
            mode_cfg_dict = MODE_CONFIGS[mode_name]
            mode_cfg = ThreeStageConfigV5(
                PROJECT_ROOT=Path("."),
                OUTPUT_ROOT=output_root,
                BRANCH_MODE=mode_cfg_dict['branch'],
                FUSION_MODE=mode_cfg_dict['fusion'],
                ZERONE_USE_CNN=mode_cfg_dict['zerone_cnn'],
            )
            
            # 创建该模式专用的数据集
            test_ds = LabeledVibrationDataset(
                split_manager.test_samples, mode_cfg,
                split_name="TEST", normalizer=GLOBAL_NORMALIZER
            )
            val_ds = LabeledVibrationDataset(
                split_manager.val_samples, mode_cfg,
                split_name="VAL", normalizer=GLOBAL_NORMALIZER
            )
            
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
            
            if mode_name == modes_to_process[0]:
                print(f"  测试集: {len(test_ds)} 样本")
                print(f"  验证集: {len(val_ds)} 样本")
            
            result = process_single_mode(mode_name, output_root, test_loader, val_loader, device)
            if result:
                all_results.append(result)
    else:
        # 从已有文件加载结果
        results_path = compare_dir / "all_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"已加载 {len(all_results)} 个模式结果")
        else:
            print(f"⚠ 未找到已有结果: {results_path}")
            return
    
    # 生成综合对比图表
    if all_results:
        generate_comparison_charts(all_results, compare_dir)
    
    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

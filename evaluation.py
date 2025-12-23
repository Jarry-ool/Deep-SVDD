# -*- coding: utf-8 -*-
"""
evaluation.py - 评估工具
========================

包含:
- 模型评估函数
- 指标计算
- 错误分析
- 评估报告生成
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)

from visualization import VisualizationManager


def evaluate_classifier(model, dataloader: DataLoader, device: torch.device,
                       return_features: bool = False) -> Dict:
    """
    评估分类器模型
    
    参数:
        model: 分类器模型
        dataloader: 数据加载器
        device: 设备
        return_features: 是否返回特征向量
    
    返回:
        评估结果字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = [] if return_features else None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中", leave=False):
            img, zr, labels, _ = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            logits = out['logits']
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # 正类概率
            
            if return_features and 'h' in out:
                all_features.extend(out['h'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    results = compute_classification_metrics(all_labels, all_preds, all_probs)
    
    if return_features:
        results['features'] = np.array(all_features)
    
    results['predictions'] = all_preds
    results['labels'] = all_labels
    results['probabilities'] = all_probs
    
    return results


def evaluate_anomaly_detector(model, dataloader: DataLoader, device: torch.device,
                             labels: np.ndarray = None) -> Dict:
    """
    评估异常检测模型
    
    参数:
        model: 异常检测模型
        dataloader: 数据加载器
        device: 设备
        labels: 真实标签（可选）
    
    返回:
        评估结果字典
    """
    model.eval()
    
    all_scores = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算异常得分", leave=False):
            img, zr, _, idx = batch
            img, zr = img.to(device), zr.to(device)
            
            scores = model.anomaly_score(img, zr)
            all_scores.extend(scores.cpu().tolist())
            all_indices.extend(idx.tolist())
    
    all_scores = np.array(all_scores)
    all_indices = np.array(all_indices)
    
    results = {
        'scores': all_scores,
        'indices': all_indices,
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'min_score': float(np.min(all_scores)),
        'max_score': float(np.max(all_scores)),
    }
    
    # 如果有标签，计算分类性能（使用中位数作为阈值）
    if labels is not None and len(labels) == len(all_scores):
        threshold = np.median(all_scores)
        preds = (all_scores > threshold).astype(int)
        
        results['threshold'] = float(threshold)
        results['accuracy'] = float(accuracy_score(labels, preds))
        results['f1'] = float(f1_score(labels, preds, average='weighted'))
        
        # ROC-AUC
        if len(np.unique(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, all_scores)
            results['auc'] = float(auc(fpr, tpr))
    
    return results


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray = None) -> Dict:
    """
    计算分类指标
    
    返回:
        指标字典
    """
    results = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # 按类别指标
    for cls in [0, 1]:
        mask = y_true == cls
        if mask.sum() > 0:
            cls_preds = y_pred[mask]
            results[f'class_{cls}_accuracy'] = float((cls_preds == cls).mean())
    
    # 如果有概率，计算AUC
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            results['roc_auc'] = float(auc(fpr, tpr))
            results['ap'] = float(average_precision_score(y_true, y_prob))
        except:
            pass
    
    return results


def analyze_errors(model, dataloader: DataLoader, device: torch.device,
                  max_errors: int = 20) -> List[Dict]:
    """
    分析错误分类样本
    
    返回:
        错误样本信息列表
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            img, zr, labels, idx = batch
            img, zr = img.to(device), zr.to(device)
            
            out = model(img, zr)
            preds = out['logits'].argmax(dim=1)
            probs = F.softmax(out['logits'], dim=1)
            
            # 找错误样本
            error_mask = preds.cpu() != labels
            error_indices = torch.where(error_mask)[0]
            
            for ei in error_indices:
                i = ei.item()
                errors.append({
                    'index': idx[i].item() if isinstance(idx[i], torch.Tensor) else idx[i],
                    'true': labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i],
                    'pred': preds[i].item(),
                    'prob': probs[i].cpu().numpy().tolist(),
                    'image': img[i].cpu().numpy(),
                    'confidence': float(probs[i].max().item()),
                })
                
                if len(errors) >= max_errors:
                    return errors
    
    return errors


def generate_evaluation_report(results: Dict, output_path: Path, lang: str = 'cn'):
    """
    生成评估报告
    
    参数:
        results: 评估结果字典
        output_path: 输出路径
        lang: 语言
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 60)
    lines.append("评估报告" if lang == 'cn' else "Evaluation Report")
    lines.append("=" * 60)
    lines.append("")
    
    # 主要指标
    lines.append("【主要指标】" if lang == 'cn' else "[Main Metrics]")
    lines.append("-" * 40)
    
    metric_names = {
        'cn': {'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率', 
               'f1': 'F1分数', 'roc_auc': 'ROC-AUC', 'ap': '平均精度'},
        'en': {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
               'f1': 'F1 Score', 'roc_auc': 'ROC-AUC', 'ap': 'Avg Precision'}
    }
    
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap']:
        if key in results:
            name = metric_names.get(lang, metric_names['en']).get(key, key)
            lines.append(f"  {name}: {results[key]:.4f}")
    
    lines.append("")
    
    # 混淆矩阵
    if 'confusion_matrix' in results:
        lines.append("【混淆矩阵】" if lang == 'cn' else "[Confusion Matrix]")
        lines.append("-" * 40)
        cm = results['confusion_matrix']
        labels = ['正常', '故障'] if lang == 'cn' else ['Normal', 'Fault']
        lines.append(f"             预测{labels[0]}  预测{labels[1]}" if lang == 'cn' 
                    else f"             Pred {labels[0]}  Pred {labels[1]}")
        lines.append(f"  真实{labels[0]}:     {cm[0][0]:5d}     {cm[0][1]:5d}" if lang == 'cn'
                    else f"  True {labels[0]}:    {cm[0][0]:5d}     {cm[0][1]:5d}")
        lines.append(f"  真实{labels[1]}:     {cm[1][0]:5d}     {cm[1][1]:5d}" if lang == 'cn'
                    else f"  True {labels[1]}:    {cm[1][0]:5d}     {cm[1][1]:5d}")
        lines.append("")
    
    # 类别准确率
    lines.append("【类别准确率】" if lang == 'cn' else "[Per-Class Accuracy]")
    lines.append("-" * 40)
    for cls, name in [(0, '正常' if lang == 'cn' else 'Normal'), 
                      (1, '故障' if lang == 'cn' else 'Fault')]:
        key = f'class_{cls}_accuracy'
        if key in results:
            lines.append(f"  {name}: {results[key]:.4f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    # 写入文件
    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 同时保存JSON格式
    json_path = output_path.with_suffix('.json')
    json_results = {k: v for k, v in results.items() 
                   if not isinstance(v, np.ndarray)}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"[评估] 报告已保存: {output_path}")
    return report_text


def run_full_evaluation(classifier, test_loader: DataLoader, device: torch.device,
                       output_dir: Path, lang: str = 'cn') -> Dict:
    """
    运行完整评估流程
    
    包含:
    - 基本指标计算
    - 可视化生成
    - 错误分析
    - 报告生成
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("【完整评估流程】" if lang == 'cn' else "[Full Evaluation]")
    print("="*60)
    
    # 1. 基本评估
    print("\n[1/4] 计算评估指标...")
    results = evaluate_classifier(classifier, test_loader, device, return_features=True)
    
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  F1分数: {results['f1']:.4f}")
    if 'roc_auc' in results:
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    
    # 2. 可视化
    print("\n[2/4] 生成可视化...")
    viz = VisualizationManager(output_dir)
    
    # 混淆矩阵
    viz.plot_confusion_matrix(results['labels'], results['predictions'], lang=lang)
    
    # ROC/PR曲线
    if results['probabilities'] is not None:
        viz.plot_roc_pr_curves(results['labels'], results['probabilities'], lang=lang)
    
    # t-SNE
    if 'features' in results and len(results['features']) > 10:
        viz.plot_tsne(results['features'], results['labels'], lang=lang)
    
    # 3. 错误分析
    print("\n[3/4] 分析错误样本...")
    errors = analyze_errors(classifier, test_loader, device, max_errors=20)
    results['error_count'] = len(errors)
    
    if errors:
        viz.plot_error_samples(errors, lang=lang)
        
        # 保存错误样本信息
        error_info = [{'index': e['index'], 'true': e['true'], 'pred': e['pred'], 
                      'confidence': e['confidence']} for e in errors]
        with open(output_dir / 'error_samples.json', 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
    
    # 4. 生成报告
    print("\n[4/4] 生成评估报告...")
    generate_evaluation_report(results, output_dir / 'evaluation_report.txt', lang=lang)
    
    print(f"\n【评估完成】结果保存于: {output_dir}")
    
    return results


class LabelFlipDetector:
    """
    标签翻转检测器 (V5.11功能)
    
    检测可能的标签错误
    """
    
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.suspicious_samples = []
    
    def detect(self, model, dataloader: DataLoader, device: torch.device) -> List[Dict]:
        """
        检测可能标签错误的样本
        
        返回:
            可疑样本列表
        """
        model.eval()
        self.suspicious_samples = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="标签翻转检测", leave=False):
                img, zr, labels, idx = batch
                img, zr = img.to(device), zr.to(device)
                
                out = model(img, zr)
                probs = F.softmax(out['logits'], dim=1)
                preds = out['logits'].argmax(dim=1)
                
                # 找到预测与标签不一致且置信度高的样本
                for i in range(len(labels)):
                    pred = preds[i].item()
                    label = labels[i].item()
                    conf = probs[i, pred].item()
                    
                    if pred != label and conf > (1 - self.threshold):
                        self.suspicious_samples.append({
                            'index': idx[i].item() if isinstance(idx[i], torch.Tensor) else idx[i],
                            'original_label': label,
                            'predicted_label': pred,
                            'confidence': conf,
                            'suggestion': f"可能应为类别{pred}"
                        })
        
        print(f"[标签检测] 发现 {len(self.suspicious_samples)} 个可疑样本")
        return self.suspicious_samples
    
    def save_report(self, output_path: Path):
        """保存检测报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.suspicious_samples, f, ensure_ascii=False, indent=2)
        
        print(f"[标签检测] 报告已保存: {output_path}")

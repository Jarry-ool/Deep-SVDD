# -*- coding: utf-8 -*-
"""
visualization.py - 可视化工具
=============================

包含:
- VisualizationManager: 完整可视化管理器
- 训练曲线、分布图、t-SNE、混淆矩阵、ROC/PR等
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from config import COLORS, LABELS


class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, output_dir: Path, viz_dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.viz_dpi = viz_dpi
        
        # 创建子目录
        self.subdirs = {
            "training_curves": self.output_dir / "training_curves",
            "distributions": self.output_dir / "distributions",
            "feature_preview": self.output_dir / "feature_preview",
            "tsne": self.output_dir / "tsne",
            "confusion": self.output_dir / "confusion",
            "roc_pr": self.output_dir / "roc_pr",
            "misclassified": self.output_dir / "misclassified",
            "reconstruction": self.output_dir / "reconstruction",
        }
        for d in self.subdirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        self.setup_style()
        self.fusion_weights_history = []
        self.domain_adaptation_history = []
    
    def setup_style(self):
        """设置matplotlib样式"""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 150,
            'savefig.dpi': self.viz_dpi,
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
        })
    
    def get_label(self, key: str, lang: str = 'cn') -> str:
        """获取标签文本"""
        return LABELS.get(lang, LABELS['en']).get(key, key)
    
    def record_fusion_weights(self, weights, labels=None):
        """记录融合权重"""
        if weights is None:
            return
        if isinstance(weights, dict):
            record = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                     for k, v in weights.items()}
            if labels is not None:
                record['labels'] = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            self.fusion_weights_history.append(record)
        elif isinstance(weights, torch.Tensor):
            self.fusion_weights_history.append({
                'weights': weights.detach().cpu().numpy(),
                'labels': labels.cpu().numpy() if labels is not None else None
            })
    
    def record_domain_adaptation(self, mmd_loss, coral_loss, dann_loss, epoch):
        """记录域适应损失"""
        self.domain_adaptation_history.append({
            'epoch': epoch,
            'mmd': mmd_loss,
            'coral': coral_loss,
            'dann': dann_loss
        })
    
    def plot_sample_preview(self, images: List[np.ndarray], labels: List[int], 
                           lang: str = 'cn', prefix: str = "samples"):
        """样本预览"""
        n = min(len(images), 8)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        title = '样本预览' if lang == 'cn' else 'Sample Preview'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for i in range(n):
            ax = axes[i // 4, i % 4]
            img = images[i].copy()
            
            if img.ndim == 3 and img.shape[0] == 3:
                display_img = np.transpose(img, (1, 2, 0))
            else:
                display_img = img
            
            # 稳健归一化
            img_min, img_max = np.nanmin(display_img), np.nanmax(display_img)
            if np.isfinite(img_min) and np.isfinite(img_max) and (img_max - img_min) > 1e-8:
                display_img = (display_img - img_min) / (img_max - img_min + 1e-8)
            else:
                display_img = np.full_like(display_img, 0.5)
            display_img = np.clip(display_img, 0, 1)
            
            ax.imshow(display_img)
            label_text = self.get_label('normal' if labels[i] == 0 else 'fault', lang)
            ax.set_title(label_text, fontsize=10,
                        color=COLORS['normal'] if labels[i] == 0 else COLORS['fault'])
            ax.axis('off')
        
        for i in range(n, 8):
            axes[i // 4, i % 4].axis('off')
        
        plt.tight_layout()
        save_path = self.subdirs["feature_preview"] / f"{prefix}_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 样本预览已保存: {save_path}")
    
    def plot_training_curves(self, history: Dict, stage: str, lang: str = 'cn'):
        """训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = history.get('epoch', [])
        if not epochs:
            plt.close(fig)
            return
        
        ax1 = axes[0]
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'svdd_loss' in history:
            ax1.plot(epochs, history['svdd_loss'], 'r--', label='SVDD Loss', linewidth=1.5)
        if 'vae_loss' in history:
            ax1.plot(epochs, history['vae_loss'], 'g--', label='VAE Loss', linewidth=1.5)
        if 'total_loss' in history:
            ax1.plot(epochs, history['total_loss'], 'm-', label='Total Loss', linewidth=2)
        
        ax1.set_xlabel(self.get_label('epoch', lang))
        ax1.set_ylabel(self.get_label('loss', lang))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('训练损失' if lang == 'cn' else 'Training Loss')
        
        ax2 = axes[1]
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'], 'b-', label='Accuracy', linewidth=2)
        if 'val_f1' in history:
            ax2.plot(epochs, history['val_f1'], 'r-', label='F1 Score', linewidth=2)
        if 'val_precision' in history:
            ax2.plot(epochs, history['val_precision'], 'g--', label='Precision', linewidth=1.5)
        if 'val_recall' in history:
            ax2.plot(epochs, history['val_recall'], 'm--', label='Recall', linewidth=1.5)
        
        ax2.set_xlabel(self.get_label('epoch', lang))
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_title('验证指标' if lang == 'cn' else 'Validation Metrics')
        
        plt.tight_layout()
        save_path = self.subdirs["training_curves"] / f"{stage}_curves_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 训练曲线已保存: {save_path}")
    
    def plot_score_distribution(self, scores: np.ndarray, t_normal: float, 
                                t_anomaly: float, lang: str = 'cn'):
        """异常得分分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(scores, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        ax.axvline(t_normal, color=COLORS['normal'], linestyle='--', linewidth=2,
                  label=f'正常阈值: {t_normal:.3f}' if lang == 'cn' else f'Normal: {t_normal:.3f}')
        ax.axvline(t_anomaly, color=COLORS['fault'], linestyle='--', linewidth=2,
                  label=f'异常阈值: {t_anomaly:.3f}' if lang == 'cn' else f'Anomaly: {t_anomaly:.3f}')
        
        ax.set_xlabel('异常得分' if lang == 'cn' else 'Anomaly Score')
        ax.set_ylabel('频数' if lang == 'cn' else 'Frequency')
        ax.set_title('异常得分分布' if lang == 'cn' else 'Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.subdirs["distributions"] / f"score_dist_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 得分分布已保存: {save_path}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, lang: str = 'cn'):
        """混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = [self.get_label('normal', lang), self.get_label('fault', lang)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_xlabel('预测标签' if lang == 'cn' else 'Predicted Label')
        ax.set_ylabel('真实标签' if lang == 'cn' else 'True Label')
        ax.set_title('混淆矩阵' if lang == 'cn' else 'Confusion Matrix')
        
        plt.tight_layout()
        save_path = self.subdirs["confusion"] / f"confusion_matrix_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 混淆矩阵已保存: {save_path}")
    
    def plot_roc_pr_curves(self, y_true: np.ndarray, y_scores: np.ndarray, lang: str = 'cn'):
        """ROC和PR曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color=COLORS['primary'], linewidth=2, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('假阳率' if lang == 'cn' else 'False Positive Rate')
        axes[0].set_ylabel('真阳率' if lang == 'cn' else 'True Positive Rate')
        axes[0].set_title('ROC曲线' if lang == 'cn' else 'ROC Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        axes[1].plot(recall, precision, color=COLORS['secondary'], linewidth=2, label=f'AUC = {pr_auc:.3f}')
        axes[1].set_xlabel('召回率' if lang == 'cn' else 'Recall')
        axes[1].set_ylabel('精确率' if lang == 'cn' else 'Precision')
        axes[1].set_title('PR曲线' if lang == 'cn' else 'PR Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.subdirs["roc_pr"] / f"roc_pr_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] ROC/PR曲线已保存: {save_path}")
    
    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, lang: str = 'cn'):
        """t-SNE可视化"""
        if len(features) > 2000:
            indices = np.random.choice(len(features), 2000, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        perplexity = min(30, len(features) - 1)
        if perplexity < 5:
            print("[警告] 样本太少，跳过t-SNE可视化")
            return
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label, name, color in [(0, 'normal', COLORS['normal']), (1, 'fault', COLORS['fault'])]:
            mask = labels == label
            if mask.sum() > 0:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=color, label=self.get_label(name, lang), alpha=0.6, s=30)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE可视化' if lang == 'cn' else 't-SNE Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.subdirs["tsne"] / f"tsne_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] t-SNE已保存: {save_path}")
    
    def plot_fusion_weights(self, lang: str = 'cn'):
        """融合权重分布"""
        if not self.fusion_weights_history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        title = '融合权重分布' if lang == 'cn' else 'Fusion Weight Distribution'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        all_records = self.fusion_weights_history
        
        if 'z_mean' in all_records[0]:
            z_means = np.concatenate([r['z_mean'] for r in all_records])
            
            axes[0].hist(z_means, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
            axes[0].axvline(0.5, color='red', linestyle='--', label='平衡点' if lang == 'cn' else 'Balance')
            axes[0].set_xlabel('门控值 z' if lang == 'cn' else 'Gate z')
            axes[0].set_ylabel('频数' if lang == 'cn' else 'Count')
            axes[0].set_title('GMU门控分布' if lang == 'cn' else 'GMU Gate Distribution')
            axes[0].legend()
            
            if 'z_std' in all_records[0]:
                z_stds = np.concatenate([r['z_std'] for r in all_records])
                axes[1].hist(z_stds, bins=30, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
                axes[1].set_xlabel('门控标准差' if lang == 'cn' else 'Gate Std')
                axes[1].set_ylabel('频数' if lang == 'cn' else 'Count')
                axes[1].set_title('门控一致性' if lang == 'cn' else 'Gate Consistency')
        
        elif 'weights' in all_records[0]:
            weights = np.concatenate([r['weights'] for r in all_records])
            
            axes[0].hist(weights[:, 0], bins=30, color=COLORS['primary'], alpha=0.7, label='α (Hetero)')
            axes[0].hist(weights[:, 1], bins=30, color=COLORS['secondary'], alpha=0.7, label='β (Zerone)')
            axes[0].set_xlabel('权重值' if lang == 'cn' else 'Weight Value')
            axes[0].set_ylabel('频数' if lang == 'cn' else 'Count')
            axes[0].legend()
            
            axes[1].scatter(weights[:, 0], weights[:, 1], alpha=0.5, s=10)
            axes[1].plot([0, 1], [1, 0], 'r--', label='α + β = 1')
            axes[1].set_xlabel('α (Hetero)')
            axes[1].set_ylabel('β (Zerone)')
            axes[1].legend()
        
        plt.tight_layout()
        save_path = self.subdirs["distributions"] / f"fusion_weights_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 融合权重已保存: {save_path}")
    
    def plot_reconstruction(self, original: np.ndarray, recon: np.ndarray, 
                           n_samples: int = 4, lang: str = 'cn'):
        """VAE重建效果可视化"""
        fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
        
        title = 'VAE重建效果' if lang == 'cn' else 'VAE Reconstruction'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for i in range(min(n_samples, len(original))):
            # 原图
            orig_img = original[i].copy()
            if orig_img.ndim == 3 and orig_img.shape[0] == 3:
                orig_img = np.transpose(orig_img, (1, 2, 0))
            orig_min, orig_max = np.nanmin(orig_img), np.nanmax(orig_img)
            if np.isfinite(orig_min) and np.isfinite(orig_max) and (orig_max - orig_min) > 1e-8:
                orig_img = (orig_img - orig_min) / (orig_max - orig_min + 1e-8)
            else:
                orig_img = np.full_like(orig_img, 0.5)
            orig_img = np.clip(orig_img, 0, 1)
            
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title('原图' if lang == 'cn' else 'Original')
            axes[0, i].axis('off')
            
            # 重建
            recon_img = recon[i].copy()
            if recon_img.ndim == 3 and recon_img.shape[0] == 3:
                recon_img = np.transpose(recon_img, (1, 2, 0))
            recon_min, recon_max = np.nanmin(recon_img), np.nanmax(recon_img)
            if np.isfinite(recon_min) and np.isfinite(recon_max) and (recon_max - recon_min) > 1e-8:
                recon_img = (recon_img - recon_min) / (recon_max - recon_min + 1e-8)
            else:
                recon_img = np.full_like(recon_img, 0.5)
            recon_img = np.clip(recon_img, 0, 1)
            
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title('重建' if lang == 'cn' else 'Recon')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        save_path = self.subdirs["reconstruction"] / f"reconstruction_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] VAE重建已保存: {save_path}")
    
    def plot_error_samples(self, error_info: List[Dict], lang: str = 'cn'):
        """错误样本可视化"""
        n = min(8, len(error_info))
        if n == 0:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        title = '错误分类样本' if lang == 'cn' else 'Misclassified Samples'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for i in range(n):
            ax = axes[i // 4, i % 4]
            info = error_info[i]
            
            img = info.get('image', np.zeros((224, 224, 3)))
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            img_min, img_max = np.nanmin(img), np.nanmax(img)
            if np.isfinite(img_min) and np.isfinite(img_max) and (img_max - img_min) > 1e-8:
                img = (img - img_min) / (img_max - img_min + 1e-8)
            else:
                img = np.full_like(img, 0.5)
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            true_label = '正常' if info.get('true', 0) == 0 else '故障'
            pred_label = '正常' if info.get('pred', 0) == 0 else '故障'
            if lang == 'en':
                true_label = 'Normal' if info.get('true', 0) == 0 else 'Fault'
                pred_label = 'Normal' if info.get('pred', 0) == 0 else 'Fault'
            
            ax.set_title(f'真:{true_label} 预:{pred_label}', fontsize=9, color='red')
            ax.axis('off')
        
        for i in range(n, 8):
            axes[i // 4, i % 4].axis('off')
        
        plt.tight_layout()
        save_path = self.subdirs["misclassified"] / f"error_samples_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 错误样本已保存: {save_path}")
    
    def plot_feature_comparison(self, normal_feats: np.ndarray, fault_feats: np.ndarray,
                                feature_names: List[str] = None, lang: str = 'cn'):
        """正常vs故障特征对比"""
        n_features = min(20, normal_feats.shape[1] if normal_feats.ndim > 1 else len(normal_feats))
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        title = '正常 vs 故障 特征分布' if lang == 'cn' else 'Normal vs Fault Feature Distribution'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for i in range(n_features):
            ax = axes[i // 5, i % 5]
            
            if normal_feats.ndim > 1:
                normal_vals = normal_feats[:, i]
                fault_vals = fault_feats[:, i] if fault_feats.ndim > 1 else fault_feats
            else:
                normal_vals = [normal_feats[i]]
                fault_vals = [fault_feats[i]] if fault_feats.ndim > 1 else fault_feats
            
            ax.hist(normal_vals, bins=20, alpha=0.7, color=COLORS['normal'], 
                   label='正常' if lang == 'cn' else 'Normal')
            ax.hist(fault_vals, bins=20, alpha=0.7, color=COLORS['fault'],
                   label='故障' if lang == 'cn' else 'Fault')
            
            fname = feature_names[i] if feature_names and i < len(feature_names) else f'F{i}'
            ax.set_title(fname, fontsize=9)
            ax.legend(fontsize=7)
        
        for i in range(n_features, 20):
            axes[i // 5, i % 5].axis('off')
        
        plt.tight_layout()
        save_path = self.subdirs["distributions"] / f"feature_comparison_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] 特征对比已保存: {save_path}")
    
    def plot_zerone_preview(self, images: List[np.ndarray], labels: List[int], 
                            lang: str = 'cn'):
        """Zerone分支专属样本预览"""
        n = min(len(images), 8)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        title = 'Zerone样本预览 (Raster-Stripe)' if lang == 'cn' else 'Zerone Sample Preview (Raster-Stripe)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        for i in range(n):
            ax = axes[i // 4, i % 4]
            img = images[i].copy()
            
            if img.ndim == 3 and img.shape[0] == 3:
                display_img = np.transpose(img, (1, 2, 0))
            else:
                display_img = img
            
            display_img = np.clip(display_img, 0, 1)
            
            ax.imshow(display_img)
            label_text = '正常' if labels[i] == 0 else '故障'
            if lang == 'en':
                label_text = 'Normal' if labels[i] == 0 else 'Fault'
            ax.set_title(label_text, fontsize=10,
                        color=COLORS['normal'] if labels[i] == 0 else COLORS['fault'])
            ax.axis('off')
        
        for i in range(n, 8):
            axes[i // 4, i % 4].axis('off')
        
        plt.tight_layout()
        save_path = self.subdirs["feature_preview"] / f"zerone_samples_{lang}.png"
        fig.savefig(save_path, dpi=self.viz_dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"[可视化] Zerone预览已保存: {save_path}")
    
    def plot_normal_vs_fault_compare(self, normal_zerone_imgs: List[np.ndarray],
                                      fault_zerone_imgs: List[np.ndarray],
                                      normal_hetero_imgs: List[np.ndarray],
                                      fault_hetero_imgs: List[np.ndarray],
                                      normal_feats: np.ndarray,
                                      fault_feats: np.ndarray,
                                      lang: str = 'cn'):
        """
        V5.1: 正常vs故障对比预览 (生成7张图)
        
        参数:
            normal_zerone_imgs: 正常样本的Zerone图像列表
            fault_zerone_imgs: 故障样本的Zerone图像列表
            normal_hetero_imgs: 正常样本的Hetero图像列表
            fault_hetero_imgs: 故障样本的Hetero图像列表
            normal_feats: 正常样本特征 (N, 1200)
            fault_feats: 故障样本特征 (N, 1200)
            lang: 语言 'cn' 或 'en'
        """
        n_samples = min(4, len(normal_zerone_imgs), len(fault_zerone_imgs))
        if n_samples == 0:
            print(f"  [警告] 没有足够的正常/故障样本，跳过对比预览")
            return
        
        # 语言配置
        L = {
            'cn': {
                'normal': '正常', 'fault': '故障',
                'zerone_compare_title': 'Zerone样本对比 (正常 vs 故障)',
                'hetero_compare_title': 'Hetero样本对比 (正常 vs 故障)',
                'zerone_normal_title': 'Zerone正常样本预览',
                'zerone_fault_title': 'Zerone故障样本预览',
                'hetero_normal_title': 'Hetero正常样本预览',
                'hetero_fault_title': 'Hetero故障样本预览',
                'feature_title': '特征分布对比 (正常 vs 故障)',
                'time_title': '时域特征 (15维)', 'stft_title': 'STFT特征 (127维)',
                'psd_title': 'PSD特征 (前200维, 1-200Hz)', 'hf_title': '高频特征 (8维)',
                'feat_idx': '特征索引', 'feat_val': '特征值', 'freq': '频率 (Hz)', 'psd_label': '功率谱密度',
                'hf_labels': ['1k幅', '1k功', '2k幅', '2k功', '3k幅', '3k功', '4k幅', '4k功'],
            },
            'en': {
                'normal': 'Normal', 'fault': 'Fault',
                'zerone_compare_title': 'Zerone Sample Comparison (Normal vs Fault)',
                'hetero_compare_title': 'Hetero Sample Comparison (Normal vs Fault)',
                'zerone_normal_title': 'Zerone Normal Sample Preview',
                'zerone_fault_title': 'Zerone Fault Sample Preview',
                'hetero_normal_title': 'Hetero Normal Sample Preview',
                'hetero_fault_title': 'Hetero Fault Sample Preview',
                'feature_title': 'Feature Distribution Comparison (Normal vs Fault)',
                'time_title': 'Time-Domain Features (15D)', 'stft_title': 'STFT Features (127D)',
                'psd_title': 'PSD Features (1-200Hz)', 'hf_title': 'High-Freq Features (8D)',
                'feat_idx': 'Feature Index', 'feat_val': 'Feature Value', 'freq': 'Frequency (Hz)', 'psd_label': 'PSD',
                'hf_labels': ['1kA', '1kP', '2kA', '2kP', '3kA', '3kP', '4kA', '4kP'],
            }
        }.get(lang, 'en')
        
        save_dir = self.subdirs["feature_preview"]
        
        # 1. Zerone对比图
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        fig.suptitle(L['zerone_compare_title'], fontsize=16, fontweight='bold')
        for i in range(n_samples):
            img = normal_zerone_imgs[i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            axes[0, i].imshow(np.clip(img, 0, 1))
            axes[0, i].set_title(f"{L['normal']} #{i+1}", fontsize=12, color='green', fontweight='bold')
            axes[0, i].axis('off')
            
            img = fault_zerone_imgs[i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            axes[1, i].imshow(np.clip(img, 0, 1))
            axes[1, i].set_title(f"{L['fault']} #{i+1}", fontsize=12, color='red', fontweight='bold')
            axes[1, i].axis('off')
        plt.tight_layout()
        fig.savefig(save_dir / f"zerone_compare_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # 2. Hetero对比图
        if normal_hetero_imgs and fault_hetero_imgs:
            fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
            fig.suptitle(L['hetero_compare_title'], fontsize=16, fontweight='bold')
            for i in range(n_samples):
                img = normal_hetero_imgs[i]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                axes[0, i].imshow(np.clip(img, 0, 1))
                axes[0, i].set_title(f"{L['normal']} #{i+1}", fontsize=12, color='green', fontweight='bold')
                axes[0, i].axis('off')
                
                img = fault_hetero_imgs[i]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                axes[1, i].imshow(np.clip(img, 0, 1))
                axes[1, i].set_title(f"{L['fault']} #{i+1}", fontsize=12, color='red', fontweight='bold')
                axes[1, i].axis('off')
            plt.tight_layout()
            fig.savefig(save_dir / f"hetero_compare_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # 3. Zerone正常样本
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(L['zerone_normal_title'], fontsize=16, fontweight='bold')
        for i in range(min(8, len(normal_zerone_imgs))):
            ax = axes[i // 4, i % 4]
            img = normal_zerone_imgs[i % len(normal_zerone_imgs)]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"{L['normal']} #{i+1}", fontsize=11, color='green', fontweight='bold')
            ax.axis('off')
        for i in range(len(normal_zerone_imgs), 8):
            axes[i // 4, i % 4].axis('off')
        plt.tight_layout()
        fig.savefig(save_dir / f"zerone_normal_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # 4. Zerone故障样本
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(L['zerone_fault_title'], fontsize=16, fontweight='bold')
        for i in range(min(8, len(fault_zerone_imgs))):
            ax = axes[i // 4, i % 4]
            img = fault_zerone_imgs[i % len(fault_zerone_imgs)]
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"{L['fault']} #{i+1}", fontsize=11, color='red', fontweight='bold')
            ax.axis('off')
        for i in range(len(fault_zerone_imgs), 8):
            axes[i // 4, i % 4].axis('off')
        plt.tight_layout()
        fig.savefig(save_dir / f"zerone_fault_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # 5. Hetero正常样本
        if normal_hetero_imgs:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(L['hetero_normal_title'], fontsize=16, fontweight='bold')
            for i in range(min(8, len(normal_hetero_imgs))):
                ax = axes[i // 4, i % 4]
                img = normal_hetero_imgs[i % len(normal_hetero_imgs)]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(f"{L['normal']} #{i+1}", fontsize=11, color='green', fontweight='bold')
                ax.axis('off')
            for i in range(len(normal_hetero_imgs), 8):
                axes[i // 4, i % 4].axis('off')
            plt.tight_layout()
            fig.savefig(save_dir / f"hetero_normal_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # 6. Hetero故障样本
        if fault_hetero_imgs:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(L['hetero_fault_title'], fontsize=16, fontweight='bold')
            for i in range(min(8, len(fault_hetero_imgs))):
                ax = axes[i // 4, i % 4]
                img = fault_hetero_imgs[i % len(fault_hetero_imgs)]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(f"{L['fault']} #{i+1}", fontsize=11, color='red', fontweight='bold')
                ax.axis('off')
            for i in range(len(fault_hetero_imgs), 8):
                axes[i // 4, i % 4].axis('off')
            plt.tight_layout()
            fig.savefig(save_dir / f"hetero_fault_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        # 7. 特征分布对比图
        if normal_feats is not None and fault_feats is not None and len(normal_feats) > 0 and len(fault_feats) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(L['feature_title'], fontsize=16, fontweight='bold')
            
            # 时域特征 (0-15)
            ax = axes[0, 0]
            ax.plot(normal_feats[:, :15].mean(axis=0), 'g-o', label=L['normal'], linewidth=2, markersize=6)
            ax.plot(fault_feats[:, :15].mean(axis=0), 'r-s', label=L['fault'], linewidth=2, markersize=6)
            ax.fill_between(range(15), normal_feats[:, :15].min(axis=0), normal_feats[:, :15].max(axis=0), alpha=0.2, color='green')
            ax.fill_between(range(15), fault_feats[:, :15].min(axis=0), fault_feats[:, :15].max(axis=0), alpha=0.2, color='red')
            ax.set_title(L['time_title'], fontsize=12)
            ax.set_xlabel(L['feat_idx'])
            ax.set_ylabel(L['feat_val'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # STFT特征 (15-142)
            ax = axes[0, 1]
            ax.plot(normal_feats[:, 15:142].mean(axis=0), 'g-', label=L['normal'], linewidth=1.5, alpha=0.8)
            ax.plot(fault_feats[:, 15:142].mean(axis=0), 'r-', label=L['fault'], linewidth=1.5, alpha=0.8)
            ax.set_title(L['stft_title'], fontsize=12)
            ax.set_xlabel(L['feat_idx'])
            ax.set_ylabel(L['feat_val'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # PSD特征 (142-342, 即1-200Hz)
            ax = axes[1, 0]
            ax.plot(normal_feats[:, 142:342].mean(axis=0), 'g-', label=L['normal'], linewidth=1.5, alpha=0.8)
            ax.plot(fault_feats[:, 142:342].mean(axis=0), 'r-', label=L['fault'], linewidth=1.5, alpha=0.8)
            ax.set_title(L['psd_title'], fontsize=12)
            ax.set_xlabel(L['freq'])
            ax.set_ylabel(L['psd_label'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 高频特征 (1192-1200)
            ax = axes[1, 1]
            x = np.arange(8)
            width = 0.35
            ax.bar(x - width/2, normal_feats[:, 1192:1200].mean(axis=0), width, label=L['normal'], color='green', alpha=0.7)
            ax.bar(x + width/2, fault_feats[:, 1192:1200].mean(axis=0), width, label=L['fault'], color='red', alpha=0.7)
            ax.set_title(L['hf_title'], fontsize=12)
            ax.set_xlabel(L['feat_idx'])
            ax.set_ylabel(L['feat_val'])
            ax.set_xticks(x)
            ax.set_xticklabels(L['hf_labels'], fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(save_dir / f"feature_distribution_{lang}.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        
        print(f"[可视化] 正常vs故障对比预览已保存 ({lang}): {save_dir}")

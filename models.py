# -*- coding: utf-8 -*-
"""
models.py - 模型定义
====================

包含:
- ModalityDropout: 模态Dropout
- 融合模块: ConcatFusion, AttentionFusion, GatedFusion, GMUFusion
- 编码器: HeteroCNN, ZeroneCNN, ZeroneMLP, BranchEncoderV5
- AnomalyModelV5: 异常检测模型
- FaultClassifierV5: 故障分类器
- DomainDiscriminator: 域判别器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm


# =============================================================================
# 模态Dropout模块
# =============================================================================

class ModalityDropout(nn.Module):
    """模态Dropout: 防止多模态网络的贪婪学习"""
    
    def __init__(self, p: float = 0.2, use_learnable_tokens: bool = True,
                 img_dim: int = 512, feat_dim: int = 512):
        super().__init__()
        self.p = p
        self.use_learnable_tokens = use_learnable_tokens
        
        if use_learnable_tokens:
            self.img_missing_token = nn.Parameter(torch.randn(img_dim) * 0.02)
            self.feat_missing_token = nn.Parameter(torch.randn(feat_dim) * 0.02)
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        dropout_info = {'img_dropped': False, 'feat_dropped': False}
        
        if not self.training:
            return h_img, h_feat, dropout_info
        
        batch_size = h_img.size(0)
        
        drop_img = torch.rand(1).item() < self.p
        drop_feat = torch.rand(1).item() < self.p
        
        # 至少保留一个模态
        if drop_img and drop_feat:
            if torch.rand(1).item() > 0.5:
                drop_img = False
            else:
                drop_feat = False
        
        h_img_out = h_img
        h_feat_out = h_feat
        
        if drop_img:
            dropout_info['img_dropped'] = True
            if self.use_learnable_tokens:
                h_img_out = self.img_missing_token.unsqueeze(0).expand(batch_size, -1)
            else:
                h_img_out = torch.zeros_like(h_img)
        
        if drop_feat:
            dropout_info['feat_dropped'] = True
            if self.use_learnable_tokens:
                h_feat_out = self.feat_missing_token.unsqueeze(0).expand(batch_size, -1)
            else:
                h_feat_out = torch.zeros_like(h_feat)
        
        return h_img_out, h_feat_out, dropout_info


# =============================================================================
# 域适应损失函数
# =============================================================================

def compute_mmd_loss(source_features: torch.Tensor, target_features: torch.Tensor, 
                     sigma: float = 1.0, max_samples: int = 1024) -> torch.Tensor:
    """MMD损失"""
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    if n_s == 0 or n_t == 0:
        return torch.tensor(0.0, device=source_features.device)
    
    if n_s > max_samples:
        idx = torch.randperm(n_s)[:max_samples]
        source_features = source_features[idx]
        n_s = max_samples
    if n_t > max_samples:
        idx = torch.randperm(n_t)[:max_samples]
        target_features = target_features[idx]
        n_t = max_samples
    
    def rbf_kernel(x, y):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    K_ss = rbf_kernel(source_features, source_features)
    K_tt = rbf_kernel(target_features, target_features)
    K_st = rbf_kernel(source_features, target_features)
    
    mmd = K_ss.sum() / (n_s * n_s) + K_tt.sum() / (n_t * n_t) - 2 * K_st.sum() / (n_s * n_t)
    
    return torch.clamp(mmd, min=0)


def compute_coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """CORAL损失"""
    d = source_features.size(1)
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    if n_s <= 1 or n_t <= 1:
        return torch.tensor(0.0, device=source_features.device)
    
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    
    source_cov = (source_centered.T @ source_centered) / (n_s - 1)
    target_cov = (target_centered.T @ target_centered) / (n_t - 1)
    
    loss = torch.sum((source_cov - target_cov) ** 2) / (4 * d * d)
    
    return loss


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层 (DANN)"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """域判别器 (DANN)"""
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.discriminator(reversed_features)


# =============================================================================
# 融合模块
# =============================================================================

class ConcatFusion(nn.Module):
    """等权拼接融合"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        
        if use_layernorm:
            self.fusion = nn.Sequential(
                nn.Linear(img_dim + feat_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(img_dim + feat_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        self.out_dim = out_dim
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_cat = torch.cat([h_img, h_feat], dim=1)
        h_fused = self.fusion(h_cat)
        return {'h_fused': h_fused, 'weights': None}


class AttentionFusion(nn.Module):
    """注意力加权融合"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        
        self.proj_img = nn.Sequential(
            nn.Linear(img_dim, out_dim),
            norm_layer(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_feat = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            norm_layer(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gating_mlp = nn.Sequential(
            nn.Linear(img_dim + feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        concat_feat = torch.cat([h_img, h_feat], dim=1)
        scores = self.gating_mlp(concat_feat)
        weights = torch.softmax(scores, dim=1)
        
        h_img_proj = self.proj_img(h_img)
        h_feat_proj = self.proj_feat(h_feat)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        h_fused = alpha * h_img_proj + beta * h_feat_proj
        
        return {'h_fused': h_fused, 'weights': weights}


class GatedFusion(nn.Module):
    """交叉门控融合"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        
        self.gate_for_img = nn.Sequential(
            nn.Linear(feat_dim, img_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        self.gate_for_feat = nn.Sequential(
            nn.Linear(img_dim, feat_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + feat_dim, out_dim),
            norm_layer(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        g_img = self.gate_for_img(h_feat)
        g_feat = self.gate_for_feat(h_img)
        
        h_img_gated = h_img * g_img
        h_feat_gated = h_feat * g_feat
        
        h_cat = torch.cat([h_img_gated, h_feat_gated], dim=1)
        h_fused = self.fusion(h_cat)
        
        weights = {
            'g_img_mean': g_img.mean(dim=1),
            'g_feat_mean': g_feat.mean(dim=1),
        }
        
        return {'h_fused': h_fused, 'weights': weights}


class GMUFusion(nn.Module):
    """门控多模态单元 (推荐)"""
    
    def __init__(self, img_dim: int = 512, feat_dim: int = 512, out_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        
        self.img_transform = nn.Sequential(
            nn.Linear(img_dim, out_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.feat_transform = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(img_dim + feat_dim, out_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
        if use_layernorm:
            self.output_norm = nn.LayerNorm(out_dim)
        else:
            self.output_norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, h_img: torch.Tensor, h_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_img_t = self.img_transform(h_img)
        h_feat_t = self.feat_transform(h_feat)
        
        concat_feat = torch.cat([h_img, h_feat], dim=1)
        z = self.gate(concat_feat)
        
        h_fused = z * h_img_t + (1 - z) * h_feat_t
        h_fused = self.output_norm(h_fused)
        
        weights = {
            'z_mean': z.mean(dim=1),
            'z_std': z.std(dim=1),
        }
        
        return {'h_fused': h_fused, 'weights': weights}


# =============================================================================
# 编码器网络
# =============================================================================

class HeteroCNN(nn.Module):
    """Hetero图像分支 - ResNet18编码器"""
    
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ZeroneCNN(nn.Module):
    """Zerone图像分支 - ResNet18编码器 (处理raster-stripe图像)"""
    
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ZeroneMLP(nn.Module):
    """Zerone特征分支 - MLP编码器 (兼容模式)"""
    
    def __init__(self, input_dim: int = 1200, output_dim: int = 512,
                 use_layernorm: bool = False, dropout: float = 0.3):
        super().__init__()
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.fc(h)


class BranchEncoderV5(nn.Module):
    """支线编码器 (V5.1版本)"""
    
    def __init__(self, branch_mode: str = 'dual', fusion_mode: str = 'gmu',
                 zerone_use_cnn: bool = True, use_modality_dropout: bool = True,
                 modality_dropout_p: float = 0.2, use_layernorm: bool = False,
                 dropout_rate: float = 0.3, cnn_feat_dim: int = 512,
                 zerone_dim: int = 1200):
        super().__init__()
        self.branch_mode = branch_mode
        self.fusion_mode = fusion_mode
        
        # Hetero分支
        if self.branch_mode in ['hetero', 'dual']:
            self.hetero_branch = HeteroCNN(output_dim=cnn_feat_dim)
        
        # Zerone分支
        if self.branch_mode in ['zerone', 'dual']:
            if zerone_use_cnn:
                self.zerone_branch = ZeroneCNN(output_dim=cnn_feat_dim)
            else:
                self.zerone_branch = ZeroneMLP(
                    input_dim=zerone_dim,
                    output_dim=cnn_feat_dim,
                    use_layernorm=use_layernorm,
                    dropout=dropout_rate
                )
        
        # 确定输出维度
        if self.branch_mode == 'hetero':
            self.output_dim = cnn_feat_dim
        elif self.branch_mode == 'zerone':
            self.output_dim = cnn_feat_dim
        else:  # dual
            # 模态Dropout
            if use_modality_dropout:
                self.modality_dropout = ModalityDropout(
                    p=modality_dropout_p,
                    use_learnable_tokens=True,
                    img_dim=cnn_feat_dim,
                    feat_dim=cnn_feat_dim
                )
            else:
                self.modality_dropout = None
            
            # 融合模块
            fusion_kwargs = {
                'img_dim': cnn_feat_dim,
                'feat_dim': cnn_feat_dim,
                'out_dim': 512,
                'use_layernorm': use_layernorm,
                'dropout': dropout_rate
            }
            
            if self.fusion_mode == 'concat':
                self.fusion_module = ConcatFusion(**fusion_kwargs)
            elif self.fusion_mode == 'attention':
                self.fusion_module = AttentionFusion(**fusion_kwargs)
            elif self.fusion_mode == 'gate':
                self.fusion_module = GatedFusion(**fusion_kwargs)
            elif self.fusion_mode == 'gmu':
                self.fusion_module = GMUFusion(**fusion_kwargs)
            else:
                raise ValueError(f"未知融合模式: {self.fusion_mode}")
            
            self.output_dim = 512
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = {}
        
        if self.branch_mode == 'hetero':
            h = self.hetero_branch(image)
            result['h'] = h
            
        elif self.branch_mode == 'zerone':
            h = self.zerone_branch(zerone)
            result['h'] = h
            
        else:  # dual
            h_img = self.hetero_branch(image)
            h_feat = self.zerone_branch(zerone)
            
            # 模态Dropout
            dropout_info = {'img_dropped': False, 'feat_dropped': False}
            if self.modality_dropout is not None:
                h_img, h_feat, dropout_info = self.modality_dropout(h_img, h_feat)
            
            fusion_result = self.fusion_module(h_img, h_feat)
            
            result['h'] = fusion_result['h_fused']
            result['h_img'] = h_img
            result['h_feat'] = h_feat
            result['fusion_weights'] = fusion_result['weights']
            result['dropout_info'] = dropout_info
        
        return result


# =============================================================================
# 异常检测模型
# =============================================================================

class AnomalyModelV5(nn.Module):
    """异常检测模型 V5.1"""
    
    def __init__(self, branch_mode: str = 'dual', fusion_mode: str = 'gmu',
                 zerone_use_cnn: bool = True, use_modality_dropout: bool = True,
                 modality_dropout_p: float = 0.2, use_layernorm: bool = False,
                 dropout_rate: float = 0.3, latent_dim: int = 64,
                 latent_channels: int = 64):
        super().__init__()
        
        self.branch_mode = branch_mode
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        
        self.encoder = BranchEncoderV5(
            branch_mode=branch_mode,
            fusion_mode=fusion_mode,
            zerone_use_cnn=zerone_use_cnn,
            use_modality_dropout=use_modality_dropout,
            modality_dropout_p=modality_dropout_p,
            use_layernorm=use_layernorm,
            dropout_rate=dropout_rate
        )
        
        # SVDD投影头
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        self.svdd_proj = nn.Sequential(
            nn.Linear(512, 256),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        self.register_buffer('center', torch.zeros(latent_dim))
        
        # VAE解码器
        if branch_mode in ['hetero', 'dual']:
            self.vae_mu = nn.Linear(512, latent_channels * 7 * 7)
            self.vae_logvar = nn.Linear(512, latent_channels * 7 * 7)
            self.vae_decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_channels, 256, 4, 2, 1),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Sigmoid()
            )
            self.has_vae = True
        else:
            self.has_vae = False
        
        self.alpha = 0.6  # SVDD与VAE的权重
    
    def encode(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(image, zerone)
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_result = self.encode(image, zerone)
        h = enc_result['h']
        
        z_svdd = self.svdd_proj(h)
        svdd_score = torch.sum((z_svdd - self.center) ** 2, dim=1)
        
        result = {
            'h': h,
            'z_svdd': z_svdd,
            'svdd_score': svdd_score,
        }
        
        for key in ['h_img', 'h_feat', 'fusion_weights', 'dropout_info']:
            if key in enc_result:
                result[key] = enc_result[key]
        
        if self.has_vae:
            mu = self.vae_mu(h).view(-1, self.latent_channels, 7, 7)
            logvar = self.vae_logvar(h).view(-1, self.latent_channels, 7, 7)
            
            if self.training:
                std = torch.exp(0.5 * logvar)
                z_vae = mu + std * torch.randn_like(std)
            else:
                z_vae = mu
            
            recon = self.vae_decoder(z_vae)
            if recon.shape[-1] != image.shape[-1]:
                recon = F.interpolate(recon, size=image.shape[2:], mode='bilinear', align_corners=False)
            
            vae_recon_loss = F.l1_loss(recon, image, reduction='none').mean(dim=[1,2,3])
            vae_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
            
            result.update({
                'recon': recon,
                'vae_recon_loss': vae_recon_loss,
                'vae_kl': vae_kl,
            })
        
        return result
    
    def init_center(self, dataloader: DataLoader, device: torch.device):
        """初始化SVDD中心"""
        n = 0
        c = torch.zeros(self.latent_dim, device=device)
        
        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="初始化SVDD中心", leave=False):
                img, zr, _, _ = batch
                img, zr = img.to(device), zr.to(device)
                enc_result = self.encode(img, zr)
                z = self.svdd_proj(enc_result['h'])
                c += z.sum(0)
                n += z.size(0)
        
        c /= n
        
        # 避免中心太接近零
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.center = c
        
        print(f"[SVDD] 中心初始化完成，范数: {c.norm().item():.4f}")
    
    def anomaly_score(self, image: torch.Tensor, zerone: torch.Tensor) -> torch.Tensor:
        out = self.forward(image, zerone)
        
        if self.has_vae:
            svdd_score = out['svdd_score']
            vae_score = out['vae_recon_loss'] + 0.01 * out['vae_kl']
            
            svdd_norm = svdd_score / (svdd_score.mean() + 1e-8)
            vae_norm = vae_score / (vae_score.mean() + 1e-8)
            
            return self.alpha * svdd_norm + (1 - self.alpha) * vae_norm
        else:
            return out['svdd_score']


# =============================================================================
# 故障分类器
# =============================================================================

class FaultClassifierV5(nn.Module):
    """故障分类器 V5.1"""
    
    def __init__(self, encoder: BranchEncoderV5, num_classes: int = 2,
                 freeze_encoder: bool = True, use_layernorm: bool = False,
                 dropout_rate: float = 0.3, use_dann: bool = False):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            norm_layer(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        if use_dann:
            self.domain_discriminator = DomainDiscriminator(feature_dim=512)
        else:
            self.domain_discriminator = None
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[分类器] 编码器已冻结")
    
    def unfreeze_encoder(self, mode: str = 'all'):
        if mode == 'all':
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("[分类器] 编码器已完全解冻")
        
        elif mode == 'fusion_only':
            if hasattr(self.encoder, 'fusion_module'):
                for param in self.encoder.fusion_module.parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, 'modality_dropout') and self.encoder.modality_dropout is not None:
                for param in self.encoder.modality_dropout.parameters():
                    param.requires_grad = True
            print("[分类器] 融合层已解冻")
        
        elif mode == 'last_layers':
            if hasattr(self.encoder, 'hetero_branch'):
                for name, param in self.encoder.hetero_branch.named_parameters():
                    if 'layer4' in name or 'fc' in name:
                        param.requires_grad = True
            if hasattr(self.encoder, 'zerone_branch'):
                for name, param in self.encoder.zerone_branch.named_parameters():
                    if 'layer4' in name or 'fc' in name:
                        param.requires_grad = True
            if hasattr(self.encoder, 'fusion_module'):
                for param in self.encoder.fusion_module.parameters():
                    param.requires_grad = True
            print("[分类器] 最后几层已解冻")
    
    def forward(self, image: torch.Tensor, zerone: torch.Tensor, 
                alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        enc_result = self.encoder(image, zerone)
        h = enc_result['h']
        
        logits = self.classifier(h)
        
        result = {
            'logits': logits,
            'h': h,
        }
        
        if self.domain_discriminator is not None:
            domain_output = self.domain_discriminator(h, alpha)
            result['domain_output'] = domain_output
        
        for key in ['fusion_weights', 'h_img', 'h_feat', 'dropout_info']:
            if key in enc_result:
                result[key] = enc_result[key]
        
        return result

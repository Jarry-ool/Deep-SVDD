# -*- coding: utf-8 -*-
"""
V4 可视化补充模块
==================

【插入位置说明】
1. 将下面的方法添加到 VisualizationManager 类中 (约第545行之后)
2. 在 train_stage1 末尾调用 Hetero/Zerone 可视化
3. 在 train_stage3 末尾调用 Dual Fusion 可视化

【具体插入点】
- VisualizationManager类: 第545-750行区间，在 plot_fusion_weights 方法之后添加
- train_stage1调用: 约第2095行，在 "阶段一完成" 打印之前
- train_stage3调用: 约第2445行，在 "阶段三完成" 打印之前
"""

# =============================================================================
# 【补充1】添加到 VisualizationManager 类中的新方法
# =============================================================================

def plot_hetero_reconstruction(self, model, dataloader, device, n_samples: int = 6, lang: str = 'cn'):
    """
    Hetero图像分支重建可视化
    
    展示VAE对3通道图像的重建效果：
    - 第1行: 原始图像 (CWT / STFT / Context)
    - 第2行: 重建图像
    - 第3行: 残差图 (差异热力图)
    
    参数:
        model: AnomalyModelV4 模型
        dataloader: 数据加载器
        device: 计算设备
        n_samples: 展示样本数
        lang: 'cn' 或 'en'
    """
    model.eval()
    
    # 收集样本
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            img, zr, label, idx = batch
            img = img.to(device)
            zr = zr.to(device)
            
            # 前向传播获取重建
            out = model(img, zr)
            
            for i in range(min(len(img), n_samples - len(samples))):
                samples.append({
                    'original': img[i].cpu().numpy(),
                    'recon': out['recon'][i].cpu().numpy() if 'recon' in out else None,
                    'label': label[i].item(),
                    'idx': idx[i].item()
                })
            
            if len(samples) >= n_samples:
                break
    
    if not samples or samples[0]['recon'] is None:
        print(f"  [跳过] Hetero重建可视化 - 无VAE重建输出")
        return
    
    # 通道名称
    channel_names_cn = ['CWT (小波)', 'STFT (频谱)', 'Context (波形)']
    channel_names_en = ['CWT (Wavelet)', 'STFT (Spectrum)', 'Context (Waveform)']
    channel_names = channel_names_cn if lang == 'cn' else channel_names_en
    
    # 创建大图: 每个样本3行(原始/重建/残差) x 3列(3通道)
    fig, axes = plt.subplots(3 * n_samples, 3, figsize=(12, 4 * n_samples))
    
    title = '三通道图像重建效果' if lang == 'cn' else 'Three-Channel Image Reconstruction'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    row_labels_cn = ['原始', '重建', '残差']
    row_labels_en = ['Original', 'Reconstructed', 'Residual']
    row_labels = row_labels_cn if lang == 'cn' else row_labels_en
    
    for s_idx, sample in enumerate(samples):
        orig = sample['original']  # (3, 224, 224)
        recon = sample['recon']    # (3, 224, 224)
        label = sample['label']
        
        # 归一化到 [0, 1] 用于显示
        orig_norm = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
        residual = np.abs(orig_norm - recon_norm)
        
        base_row = s_idx * 3
        
        for ch in range(3):
            # 原始图像
            ax_orig = axes[base_row, ch]
            ax_orig.imshow(orig_norm[ch], cmap='viridis', aspect='auto')
            if s_idx == 0:
                ax_orig.set_title(channel_names[ch], fontsize=10)
            if ch == 0:
                label_text = '正常' if label == 0 else '故障'
                label_text_en = 'Normal' if label == 0 else 'Fault'
                ax_orig.set_ylabel(f"样本{s_idx+1} ({label_text if lang=='cn' else label_text_en})\n{row_labels[0]}", 
                                  fontsize=9)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            
            # 重建图像
            ax_recon = axes[base_row + 1, ch]
            ax_recon.imshow(recon_norm[ch], cmap='viridis', aspect='auto')
            if ch == 0:
                ax_recon.set_ylabel(row_labels[1], fontsize=9)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
            
            # 残差热力图
            ax_res = axes[base_row + 2, ch]
            im = ax_res.imshow(residual[ch], cmap='hot', aspect='auto', vmin=0, vmax=0.5)
            if ch == 0:
                ax_res.set_ylabel(row_labels[2], fontsize=9)
            ax_res.set_xticks([])
            ax_res.set_yticks([])
    
    # 添加colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar_label = '重建误差' if lang == 'cn' else 'Reconstruction Error'
    fig.colorbar(im, cax=cbar_ax, label=cbar_label)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    save_dir = self.cfg.VIZ_SUBDIRS.get("reconstruction", self.cfg.VIZ_DIR / "reconstruction")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"hetero_reconstruction_{lang}.png"
    fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [保存] {save_dir / filename}")


def plot_hetero_channel_analysis(self, dataloader, device, n_samples: int = 100, lang: str = 'cn'):
    """
    Hetero三通道统计分析
    
    展示:
    - 各通道的像素强度分布 (正常 vs 故障)
    - 通道间相关性热力图
    - 空间激活热力图 (平均)
    """
    # 收集数据
    all_images = {'normal': [], 'fault': []}
    
    count = 0
    for batch in dataloader:
        img, _, label, _ = batch
        for i in range(len(img)):
            key = 'normal' if label[i].item() == 0 else 'fault'
            all_images[key].append(img[i].numpy())
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    
    channel_names_cn = ['CWT (小波)', 'STFT (频谱)', 'Context (波形)']
    channel_names_en = ['CWT (Wavelet)', 'STFT (Spectrum)', 'Context (Waveform)']
    channel_names = channel_names_cn if lang == 'cn' else channel_names_en
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    title = '三通道图像特征分析' if lang == 'cn' else 'Three-Channel Image Feature Analysis'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = {'normal': '#2ecc71', 'fault': '#e74c3c'}
    labels_cn = {'normal': '正常', 'fault': '故障'}
    labels_en = {'normal': 'Normal', 'fault': 'Fault'}
    labels = labels_cn if lang == 'cn' else labels_en
    
    # 第1行: 各通道像素强度分布
    for ch in range(3):
        ax = axes[0, ch]
        for key in ['normal', 'fault']:
            if all_images[key]:
                data = np.array(all_images[key])[:, ch, :, :].flatten()
                ax.hist(data, bins=50, alpha=0.6, color=colors[key], label=labels[key], density=True)
        
        ax.set_title(channel_names[ch], fontsize=11)
        ax.set_xlabel('像素强度' if lang == 'cn' else 'Pixel Intensity', fontsize=9)
        ax.set_ylabel('密度' if lang == 'cn' else 'Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 第2行: 空间平均激活图 (正常 vs 故障)
    for idx, key in enumerate(['normal', 'fault']):
        ax = axes[1, idx]
        if all_images[key]:
            data = np.array(all_images[key])  # (N, 3, 224, 224)
            # 计算所有通道的平均激活
            mean_activation = data.mean(axis=(0, 1))  # (224, 224)
            im = ax.imshow(mean_activation, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        title_text = f"空间平均激活 ({labels[key]})" if lang == 'cn' else f"Spatial Mean Activation ({labels[key]})"
        ax.set_title(title_text, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 第2行第3列: 通道相关性
    ax = axes[1, 2]
    if all_images['normal'] or all_images['fault']:
        all_data = all_images['normal'] + all_images['fault']
        data = np.array(all_data)  # (N, 3, 224, 224)
        # 计算通道间相关性
        ch_data = data.reshape(len(data), 3, -1)  # (N, 3, 224*224)
        ch_means = ch_data.mean(axis=2)  # (N, 3)
        corr_matrix = np.corrcoef(ch_means.T)  # (3, 3)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        short_names = ['CWT', 'STFT', 'Ctx']
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)
        
        # 添加数值标注
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=10)
    
    title_text = '通道相关性' if lang == 'cn' else 'Channel Correlation'
    ax.set_title(title_text, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_dir = self.cfg.VIZ_SUBDIRS.get("reconstruction", self.cfg.VIZ_DIR / "reconstruction")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"hetero_channel_analysis_{lang}.png"
    fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [保存] {save_dir / filename}")


def plot_zerone_feature_analysis(self, dataloader, device, n_samples: int = 200, lang: str = 'cn'):
    """
    Zerone特征分支分析可视化
    
    展示1200维特征的:
    - 特征段分布 (时域15 + STFT127 + PSD1050 + 高频8)
    - 正常 vs 故障的特征差异热力图
    - Top-K重要特征柱状图
    - 特征相关性矩阵 (分段)
    """
    # 收集数据
    features = {'normal': [], 'fault': []}
    
    count = 0
    for batch in dataloader:
        _, zr, label, _ = batch
        for i in range(len(zr)):
            key = 'normal' if label[i].item() == 0 else 'fault'
            features[key].append(zr[i].numpy())
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break
    
    # 特征段定义
    segments = {
        'time': (0, 15, '时域特征' if lang == 'cn' else 'Time Domain'),
        'stft': (15, 142, 'STFT特征' if lang == 'cn' else 'STFT Features'),
        'psd': (142, 1192, 'PSD特征' if lang == 'cn' else 'PSD Features'),
        'hf': (1192, 1200, '高频特征' if lang == 'cn' else 'High-Freq Features')
    }
    
    fig = plt.figure(figsize=(16, 12))
    
    title = 'Zerone 1200维特征分析' if lang == 'cn' else 'Zerone 1200-D Feature Analysis'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = {'normal': '#2ecc71', 'fault': '#e74c3c'}
    labels_text = {'normal': '正常' if lang == 'cn' else 'Normal', 
                   'fault': '故障' if lang == 'cn' else 'Fault'}
    
    # 子图1: 特征段均值对比 (2x2第1个)
    ax1 = fig.add_subplot(2, 2, 1)
    
    seg_names = []
    normal_means = []
    fault_means = []
    
    for seg_name, (start, end, label) in segments.items():
        seg_names.append(label)
        if features['normal']:
            normal_means.append(np.mean([f[start:end].mean() for f in features['normal']]))
        else:
            normal_means.append(0)
        if features['fault']:
            fault_means.append(np.mean([f[start:end].mean() for f in features['fault']]))
        else:
            fault_means.append(0)
    
    x = np.arange(len(seg_names))
    width = 0.35
    ax1.bar(x - width/2, normal_means, width, label=labels_text['normal'], color=colors['normal'])
    ax1.bar(x + width/2, fault_means, width, label=labels_text['fault'], color=colors['fault'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_names, fontsize=9)
    ax1.set_ylabel('平均值' if lang == 'cn' else 'Mean Value', fontsize=10)
    ax1.set_title('各段特征均值' if lang == 'cn' else 'Feature Segment Means', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 特征差异热力图 (2x2第2个)
    ax2 = fig.add_subplot(2, 2, 2)
    
    if features['normal'] and features['fault']:
        normal_mean = np.mean(features['normal'], axis=0)
        fault_mean = np.mean(features['fault'], axis=0)
        diff = fault_mean - normal_mean
        
        # 重塑为2D便于可视化 (40x30 = 1200)
        diff_2d = diff.reshape(40, 30)
        im = ax2.imshow(diff_2d, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        ax2.set_xlabel('特征索引 (mod 30)' if lang == 'cn' else 'Feature Index (mod 30)', fontsize=9)
        ax2.set_ylabel('特征索引 (// 30)' if lang == 'cn' else 'Feature Index (// 30)', fontsize=9)
    
    ax2.set_title('故障-正常 特征差异' if lang == 'cn' else 'Fault-Normal Feature Difference', fontsize=11)
    
    # 子图3: Top-20 差异最大的特征 (2x2第3个)
    ax3 = fig.add_subplot(2, 2, 3)
    
    if features['normal'] and features['fault']:
        normal_mean = np.mean(features['normal'], axis=0)
        fault_mean = np.mean(features['fault'], axis=0)
        normal_std = np.std(features['normal'], axis=0) + 1e-8
        
        # 计算标准化差异
        z_diff = np.abs(fault_mean - normal_mean) / normal_std
        top_k = 20
        top_indices = np.argsort(z_diff)[-top_k:][::-1]
        top_values = z_diff[top_indices]
        
        # 标注属于哪个段
        def get_segment(idx):
            for seg_name, (start, end, _) in segments.items():
                if start <= idx < end:
                    return seg_name
            return 'unknown'
        
        segment_colors = {'time': '#3498db', 'stft': '#9b59b6', 'psd': '#f39c12', 'hf': '#1abc9c'}
        bar_colors = [segment_colors[get_segment(i)] for i in top_indices]
        
        ax3.barh(range(top_k), top_values, color=bar_colors)
        ax3.set_yticks(range(top_k))
        ax3.set_yticklabels([f'F{i}' for i in top_indices], fontsize=8)
        ax3.set_xlabel('Z-Score 差异' if lang == 'cn' else 'Z-Score Difference', fontsize=10)
        ax3.set_title(f'Top-{top_k} 差异特征' if lang == 'cn' else f'Top-{top_k} Discriminative Features', fontsize=11)
        ax3.invert_yaxis()
        
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=segment_colors[seg], label=segments[seg][2]) 
                          for seg in ['time', 'stft', 'psd', 'hf']]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # 子图4: 特征分布小提琴图 (选取关键段) (2x2第4个)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # 选取每个段的代表性特征
    repr_indices = [0, 7, 14,  # 时域
                    20, 80, 140,  # STFT
                    200, 600, 1000,  # PSD
                    1195, 1198]  # 高频
    
    if features['normal'] and features['fault']:
        positions = []
        data_normal = []
        data_fault = []
        
        for i, idx in enumerate(repr_indices):
            if idx < 1200:
                positions.append(i)
                data_normal.append([f[idx] for f in features['normal']])
                data_fault.append([f[idx] for f in features['fault']])
        
        # 绘制箱线图
        bp1 = ax4.boxplot(data_normal, positions=np.array(positions) - 0.2, widths=0.35,
                         patch_artist=True, boxprops=dict(facecolor=colors['normal'], alpha=0.7))
        bp2 = ax4.boxplot(data_fault, positions=np.array(positions) + 0.2, widths=0.35,
                         patch_artist=True, boxprops=dict(facecolor=colors['fault'], alpha=0.7))
        
        ax4.set_xticks(positions)
        ax4.set_xticklabels([f'F{i}' for i in repr_indices], fontsize=8, rotation=45)
        ax4.set_ylabel('特征值' if lang == 'cn' else 'Feature Value', fontsize=10)
        ax4.set_title('代表性特征分布' if lang == 'cn' else 'Representative Feature Distribution', fontsize=11)
        ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], 
                  [labels_text['normal'], labels_text['fault']], fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_dir = self.cfg.VIZ_SUBDIRS.get("feature_analysis", self.cfg.VIZ_DIR / "feature_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"zerone_feature_analysis_{lang}.png"
    fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [保存] {save_dir / filename}")


def plot_dual_fusion_samples(self, model, classifier, dataloader, device, n_samples: int = 6, lang: str = 'cn'):
    """
    Dual分支融合样本可视化
    
    展示融合过程:
    - 左: Hetero 3通道图像
    - 中: Zerone 特征条形图
    - 右: 融合权重 + 预测结果
    
    参数:
        model: AnomalyModelV4 (编码器)
        classifier: FaultClassifierV4 (分类器)
        dataloader: 数据加载器
        device: 计算设备
        n_samples: 展示样本数
        lang: 'cn' 或 'en'
    """
    if self.cfg.BRANCH_MODE != 'dual':
        print(f"  [跳过] Dual融合可视化 - 当前模式: {self.cfg.BRANCH_MODE}")
        return
    
    classifier.eval()
    
    # 收集样本
    samples = []
    with torch.no_grad():
        for batch in dataloader:
            img, zr, label, idx = batch
            img, zr = img.to(device), zr.to(device)
            
            result = classifier(img, zr)
            probs = torch.softmax(result['logits'], dim=1)
            
            for i in range(min(len(img), n_samples - len(samples))):
                sample_data = {
                    'image': img[i].cpu().numpy(),
                    'zerone': zr[i].cpu().numpy(),
                    'label': label[i].item(),
                    'pred': result['logits'][i].argmax().item(),
                    'prob': probs[i, 1].item(),  # 故障概率
                    'fusion_weights': None
                }
                
                # 获取融合权重
                if 'fusion_weights' in result and result['fusion_weights'] is not None:
                    weights = result['fusion_weights']
                    if isinstance(weights, torch.Tensor):
                        sample_data['fusion_weights'] = weights[i].cpu().numpy()
                    elif isinstance(weights, dict):
                        sample_data['fusion_weights'] = {
                            'g_img': weights['g_img_mean'][i].item() if 'g_img_mean' in weights else 0.5,
                            'g_feat': weights['g_feat_mean'][i].item() if 'g_feat_mean' in weights else 0.5
                        }
                
                samples.append(sample_data)
            
            if len(samples) >= n_samples:
                break
    
    # 创建可视化
    fig = plt.figure(figsize=(18, 4 * n_samples))
    
    title = f'双分支融合样本分析 ({self.cfg.FUSION_MODE})' if lang == 'cn' else f'Dual-Branch Fusion Sample Analysis ({self.cfg.FUSION_MODE})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    channel_names_cn = ['CWT', 'STFT', 'Context']
    channel_names_en = ['CWT', 'STFT', 'Context']
    channel_names = channel_names_cn if lang == 'cn' else channel_names_en
    
    for s_idx, sample in enumerate(samples):
        # === 左侧: Hetero 3通道图像 (3个子图) ===
        for ch in range(3):
            ax_img = fig.add_subplot(n_samples, 6, s_idx * 6 + ch + 1)
            
            img_ch = sample['image'][ch]
            img_norm = (img_ch - img_ch.min()) / (img_ch.max() - img_ch.min() + 1e-8)
            ax_img.imshow(img_norm, cmap='viridis', aspect='auto')
            
            if s_idx == 0:
                ax_img.set_title(channel_names[ch], fontsize=10)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            if ch == 0:
                label_text = '正常' if sample['label'] == 0 else '故障'
                label_en = 'Normal' if sample['label'] == 0 else 'Fault'
                ax_img.set_ylabel(f"样本{s_idx+1}\n({label_text if lang=='cn' else label_en})", fontsize=9)
        
        # === 中间: Zerone 特征概览 ===
        ax_zr = fig.add_subplot(n_samples, 6, s_idx * 6 + 4)
        
        zerone = sample['zerone']
        # 分段显示
        seg_means = [
            zerone[:15].mean(),      # 时域
            zerone[15:142].mean(),   # STFT
            zerone[142:1192].mean(), # PSD
            zerone[1192:].mean()     # 高频
        ]
        seg_names_cn = ['时域', 'STFT', 'PSD', '高频']
        seg_names_en = ['Time', 'STFT', 'PSD', 'HF']
        seg_names = seg_names_cn if lang == 'cn' else seg_names_en
        seg_colors = ['#3498db', '#9b59b6', '#f39c12', '#1abc9c']
        
        ax_zr.barh(seg_names, seg_means, color=seg_colors)
        ax_zr.set_xlabel('均值' if lang == 'cn' else 'Mean', fontsize=9)
        if s_idx == 0:
            ax_zr.set_title('Zerone特征' if lang == 'cn' else 'Zerone Features', fontsize=10)
        ax_zr.grid(True, alpha=0.3, axis='x')
        
        # === 右侧: 融合权重 + 预测 ===
        ax_fusion = fig.add_subplot(n_samples, 6, s_idx * 6 + 5)
        
        # 融合权重可视化
        if sample['fusion_weights'] is not None:
            if isinstance(sample['fusion_weights'], np.ndarray):
                # Attention模式: [α, β]
                weights = sample['fusion_weights']
                labels_w = ['α (Hetero)', 'β (Zerone)']
                colors_w = ['#3498db', '#e74c3c']
                ax_fusion.bar(labels_w, weights, color=colors_w)
                ax_fusion.set_ylim(0, 1)
                for i, v in enumerate(weights):
                    ax_fusion.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
            elif isinstance(sample['fusion_weights'], dict):
                # Gate模式: g_img, g_feat
                weights = sample['fusion_weights']
                labels_w = ['g_img', 'g_feat']
                values = [weights.get('g_img', 0.5), weights.get('g_feat', 0.5)]
                colors_w = ['#3498db', '#e74c3c']
                ax_fusion.bar(labels_w, values, color=colors_w)
                ax_fusion.set_ylim(0, 1)
                for i, v in enumerate(values):
                    ax_fusion.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        else:
            ax_fusion.text(0.5, 0.5, 'Concat\n(等权)', ha='center', va='center', fontsize=12,
                          transform=ax_fusion.transAxes)
            ax_fusion.set_xlim(0, 1)
            ax_fusion.set_ylim(0, 1)
        
        if s_idx == 0:
            title_w = '融合权重' if lang == 'cn' else 'Fusion Weights'
            ax_fusion.set_title(title_w, fontsize=10)
        ax_fusion.grid(True, alpha=0.3, axis='y')
        
        # === 最右侧: 预测结果 ===
        ax_pred = fig.add_subplot(n_samples, 6, s_idx * 6 + 6)
        
        pred_text = '故障' if sample['pred'] == 1 else '正常'
        pred_en = 'Fault' if sample['pred'] == 1 else 'Normal'
        true_text = '故障' if sample['label'] == 1 else '正常'
        true_en = 'Fault' if sample['label'] == 1 else 'Normal'
        
        is_correct = sample['pred'] == sample['label']
        bg_color = '#d4edda' if is_correct else '#f8d7da'
        text_color = '#155724' if is_correct else '#721c24'
        
        ax_pred.set_facecolor(bg_color)
        
        if lang == 'cn':
            text = f"预测: {pred_text}\n真实: {true_text}\n置信度: {sample['prob']:.1%}"
        else:
            text = f"Pred: {pred_en}\nTrue: {true_en}\nConf: {sample['prob']:.1%}"
        
        ax_pred.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                    color=text_color, fontweight='bold', transform=ax_pred.transAxes)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        
        if s_idx == 0:
            ax_pred.set_title('预测结果' if lang == 'cn' else 'Prediction', fontsize=10)
        
        # 添加边框
        for spine in ax_pred.spines.values():
            spine.set_color(text_color)
            spine.set_linewidth(2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_dir = self.cfg.VIZ_SUBDIRS.get("fusion_weights", self.cfg.VIZ_DIR / "fusion_weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"dual_fusion_samples_{self.cfg.FUSION_MODE}_{lang}.png"
    fig.savefig(save_dir / filename, dpi=self.cfg.VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [保存] {save_dir / filename}")


# =============================================================================
# 【补充2】在 ThreeStageConfigV4.__post_init__ 中添加新的可视化子目录
# =============================================================================
"""
在 self.VIZ_SUBDIRS 字典中添加:

    "reconstruction": self.VIZ_DIR / "reconstruction",
    "feature_analysis": self.VIZ_DIR / "feature_analysis",

完整应为:
    self.VIZ_SUBDIRS = {
        "training_curves": self.VIZ_DIR / "training_curves",
        "score_dist": self.VIZ_DIR / "score_distribution",
        "roc_pr": self.VIZ_DIR / "roc_pr_curves",
        "confusion": self.VIZ_DIR / "confusion_matrix",
        "tsne": self.VIZ_DIR / "tsne",
        "samples": self.VIZ_DIR / "sample_preview",
        "error_analysis": self.VIZ_DIR / "error_analysis",
        "fusion_weights": self.VIZ_DIR / "fusion_weights",
        "reconstruction": self.VIZ_DIR / "reconstruction",      # 新增
        "feature_analysis": self.VIZ_DIR / "feature_analysis",  # 新增
    }
"""


# =============================================================================
# 【补充3】在 train_stage1 末尾添加调用代码 (约第2095行前)
# =============================================================================
"""
在 train_stage1 函数的 "阶段一完成" 打印之前添加:

    # ==================== V4新增: 分支可视化 ====================
    print("\\n[*] 生成分支特征可视化...")
    
    # 创建评估用的数据加载器
    eval_ds = TransformerVibrationDataset(cfg.TRAIN_DIR, cfg, use_labels=True, split_name="TRAIN_EVAL")
    eval_loader = DataLoader(eval_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Hetero分支可视化
    if cfg.BRANCH_MODE in ['hetero', 'dual']:
        print("  [Hetero] 图像重建与通道分析...")
        for lang in cfg.LANGS:
            viz.plot_hetero_reconstruction(model, eval_loader, device, n_samples=6, lang=lang)
            viz.plot_hetero_channel_analysis(eval_loader, device, n_samples=100, lang=lang)
    
    # Zerone分支可视化
    if cfg.BRANCH_MODE in ['zerone', 'dual']:
        print("  [Zerone] 特征分析...")
        for lang in cfg.LANGS:
            viz.plot_zerone_feature_analysis(eval_loader, device, n_samples=200, lang=lang)
"""


# =============================================================================
# 【补充4】在 train_stage3 末尾添加调用代码 (约第2445行前)
# =============================================================================
"""
在 train_stage3 函数的 "阶段三完成" 打印之前添加:

    # ==================== V4新增: Dual融合样本可视化 ====================
    if cfg.BRANCH_MODE == 'dual':
        print("\\n[*] 生成Dual融合样本可视化...")
        for lang in cfg.LANGS:
            viz.plot_dual_fusion_samples(model, classifier, test_loader, device, n_samples=6, lang=lang)
"""

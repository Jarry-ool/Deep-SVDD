# -*- coding: utf-8 -*-
"""
unit_ranking_analysis.py
========================
使用已训练的 v512 Hetero Stage-1 模型，在 filtered_output 标注集上
计算**变压器单元级**异常评分与真实故障状态的相关性。

运行方式 (在 Deep SVDD 目录下):
    python unit_ranking_analysis.py

输出:
    - 控制台打印每台变压器的平均异常分、真实故障占比、排名
    - 保存 unit_ranking_results.json
    - 保存 unit_ranking_plot.png (排名可视化)
"""

import os, json, sys
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "three_stage_results_v512/branch_hetero/models/stage1/stage1_best_model.pth"
FILT_DIR   = BASE_DIR / "filtered_output"
NORM_PATH  = BASE_DIR / "three_stage_results_v512/global_normalizer.npz"
OUT_JSON   = BASE_DIR / "unit_ranking_results.json"
OUT_PNG    = BASE_DIR / "unit_ranking_plot.png"

# ---------- 检查文件 ----------
if not MODEL_PATH.exists():
    sys.exit(f"[错误] 未找到模型: {MODEL_PATH}")

sys.path.insert(0, str(BASE_DIR))


def load_jsonl(path: Path):
    """加载单个 JSONL 样本，返回 numpy signal array."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vals = np.array([float(v) for v in obj["signal_value"].split(",")],
                             dtype=np.float32)
            records.append(vals)
    if not records:
        return None
    # 多通道取均值
    sig = np.stack(records, axis=0).mean(axis=0)  # (8192,)
    return sig


def collect_samples(split: str = "test"):
    """
    收集 filtered_output/<split>/故障|正常 下所有 JSONL 文件。
    返回 list of dict: {unit_id, label, path}
    """
    samples = []
    for label_dir, label in [("故障", 1), ("正常", 0)]:
        d = FILT_DIR / split / label_dir
        if not d.exists():
            print(f"[警告] 目录不存在: {d}")
            continue
        for p in sorted(d.glob("*.jsonl")):
            # 文件名格式: <unit_id>_<timestamp>.jsonl
            unit_id = p.stem.split("_")[0]
            samples.append({"unit_id": unit_id, "label": label, "path": p})
    return samples


_gpu_transform = None  # 全局缓存，避免重复构建卷积核

def build_hetero_image(sig: np.ndarray, device="cpu"):
    """
    构建 Hetero 3通道图像 (CWT/STFT/raw)，224×224。
    使用 GPUHeteroTransform（支持 CPU/CUDA）。
    """
    global _gpu_transform
    if _gpu_transform is None:
        from gpu_transforms import GPUHeteroTransform
        _gpu_transform = GPUHeteroTransform(size=224, n_scales=64,
                                            signal_len=8192, device=device)
        _gpu_transform.eval()

    sig_t = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)  # (1, 8192)
    with torch.no_grad():
        img = _gpu_transform(sig_t)  # (1, 3, 224, 224)
    return img


def load_model(device="cpu"):
    """加载 Stage-1 AnomalyModelV5 (hetero branch)."""
    from models import AnomalyModelV5

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # 支持多种 checkpoint 格式
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # 直接就是 state_dict

    model = AnomalyModelV5(
        branch_mode="hetero",
        fusion_mode="concat",
        zerone_use_cnn=True,
        use_modality_dropout=False,  # 推理时关闭
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    # center 作为 buffer 已经保存在 state_dict 中
    return model


@torch.no_grad()
def compute_svdd_score(model, img_tensor):
    """计算单样本 SVDD 分数 (使用模型内置 center buffer)."""
    out = model(img_tensor, None)   # hetero branch: zerone=None
    # forward 返回 dict，包含 svdd_score
    if isinstance(out, dict):
        score = float(out["svdd_score"].item())
    elif isinstance(out, (tuple, list)):
        score = float(out[0].item())
    else:
        score = float(out.item())
    return score


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")

    # 加载模型
    print("[模型] 加载 Stage-1 Hetero 模型 (AnomalyModelV5)...")
    try:
        model = load_model(device)
        center_norm = model.center.norm().item()
        print(f"  超球心 norm = {center_norm:.4f}")
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        print("  请确认已安装 PyTorch 并在 Deep SVDD 目录下运行此脚本")
        raise

    # 收集样本
    print("[数据] 收集 test split 样本...")
    samples = collect_samples("test")
    if not samples:
        samples = collect_samples("val")
        print("  test 为空，使用 val split")
    print(f"  共 {len(samples)} 个样本，来自 {len(set(s['unit_id'] for s in samples))} 台变压器")

    # 按单元整理
    unit_scores  = defaultdict(list)   # unit_id -> [svdd_scores]
    unit_labels  = defaultdict(list)   # unit_id -> [0/1]

    for i, s in enumerate(samples):
        if i % 100 == 0:
            print(f"  推理进度: {i}/{len(samples)}", end="\r")
        sig = load_jsonl(s["path"])
        if sig is None:
            continue
        try:
            img = build_hetero_image(sig, device)
            score = compute_svdd_score(model, img)
            unit_scores[s["unit_id"]].append(score)
            unit_labels[s["unit_id"]].append(s["label"])
        except Exception as e:
            print(f"\n  [警告] 样本处理失败 {s['path'].name}: {e}")

    print(f"\n[完成] 处理了 {sum(len(v) for v in unit_scores.values())} 个有效样本")

    # 每台变压器统计
    units = sorted(unit_scores.keys())
    rows = []
    for uid in units:
        scores = unit_scores[uid]
        labels = unit_labels[uid]
        mean_score = float(np.mean(scores))
        fault_ratio = float(np.mean(labels))
        n_fault = int(np.sum(labels))
        n_total = len(labels)
        rows.append({
            "unit_id": uid,
            "mean_svdd_score": mean_score,
            "fault_ratio": fault_ratio,
            "n_fault": n_fault,
            "n_normal": n_total - n_fault,
            "n_total": n_total,
        })

    # 按 SVDD 分排名 (高分=异常)
    rows.sort(key=lambda x: x["mean_svdd_score"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank_by_score"] = rank

    # Spearman 相关系数
    from scipy.stats import spearmanr, kendalltau
    scores_arr      = np.array([r["mean_svdd_score"] for r in rows])
    fault_ratio_arr = np.array([r["fault_ratio"] for r in rows])
    rho, p_spearman = spearmanr(scores_arr, fault_ratio_arr)
    tau, p_kendall  = kendalltau(scores_arr, fault_ratio_arr)

    print("\n" + "=" * 65)
    print(f"{'Unit ID':<16} {'Mean Score':>12} {'Fault%':>8} {'#F':>4} {'#N':>4} {'Rank':>5}")
    print("-" * 65)
    for r in rows:
        print(f"{r['unit_id']:<16} {r['mean_svdd_score']:>12.4f} "
              f"{r['fault_ratio']*100:>7.1f}% {r['n_fault']:>4} {r['n_normal']:>4} "
              f"{r['rank_by_score']:>5}")
    print("=" * 65)
    print(f"Spearman ρ = {rho:.4f}  (p = {p_spearman:.4f})")
    print(f"Kendall  τ = {tau:.4f}  (p = {p_kendall:.4f})")

    # 保存 JSON
    result = {
        "units": rows,
        "spearman_rho": rho,
        "spearman_p": p_spearman,
        "kendall_tau": tau,
        "kendall_p": p_kendall,
        "n_units": len(rows),
        "device": device,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_JSON}")

    # 可视化
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图: 单元排名条形图
        ax = axes[0]
        colors = ["#d62728" if r["fault_ratio"] > 0 else "#1f77b4" for r in rows]
        ypos = np.arange(len(rows))
        ax.barh(ypos, [r["mean_svdd_score"] for r in rows], color=colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"Unit {r['unit_id'][-4:]}" for r in rows], fontsize=7)
        ax.set_xlabel("Mean SVDD Score (higher = more anomalous)")
        ax.set_title(f"Unit-Level Risk Ranking (Hetero Stage-1)\n"
                     f"Spearman ρ={rho:.3f}, Kendall τ={tau:.3f}")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#d62728", label="Fault unit"),
                            Patch(color="#1f77b4", label="Normal unit")])

        # 右图: scatter score vs fault_ratio
        ax = axes[1]
        ax.scatter([r["mean_svdd_score"] for r in rows],
                   [r["fault_ratio"] * 100 for r in rows],
                   c=colors, s=80, edgecolors="k", linewidths=0.5)
        ax.set_xlabel("Mean SVDD Score")
        ax.set_ylabel("Fault Sample Ratio (%)")
        ax.set_title("SVDD Score vs Ground-Truth Fault Ratio")
        for r in rows:
            ax.annotate(r["unit_id"][-4:],
                        (r["mean_svdd_score"], r["fault_ratio"] * 100),
                        fontsize=6, ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {OUT_PNG}")
    except Exception as e:
        print(f"[警告] 可视化失败: {e}")


if __name__ == "__main__":
    run()

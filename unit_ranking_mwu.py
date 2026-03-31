# -*- coding: utf-8 -*-
"""
unit_ranking_mwu.py
===================
对已有的单元排名结果做更合适的统计检验：
  - 合并 val + test 两个 split，增加统计效力
  - 用 Mann-Whitney U 单尾检验（故障单元分数 > 正常单元分数）
  - 用置换检验 (permutation test) 作为无分布假设的补充
  - 报告 precision@K 及随机基线

注意：这不是重新跑模型，而是对已有的 unit_ranking_results.json 做补充分析，
并在 val split 上额外推理（如果需要）。

如果 unit_ranking_results.json 只包含 test split，
本脚本会自动读取 filtered_output/val/ 并重用已加载的模型权重。

运行:
    cd "3-code/diagnosis/test/Deep SVDD"
    python unit_ranking_mwu.py
"""

import os, json, sys
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from collections import defaultdict

BASE_DIR     = Path(__file__).parent
JSON_PATH    = BASE_DIR / "unit_ranking_results.json"
FILT_DIR     = BASE_DIR / "filtered_output"
MODEL_PATH   = BASE_DIR / "three_stage_results_v512/branch_hetero/models/stage1/stage1_best_model.pth"
OUT_JSON     = BASE_DIR / "unit_ranking_mwu_results.json"


# ── 1. 读取已有 test-split 结果 ──────────────────────────────────────
if not JSON_PATH.exists():
    sys.exit("[错误] 未找到 unit_ranking_results.json，请先运行 unit_ranking_analysis.py")

with open(JSON_PATH) as f:
    prev = json.load(f)

test_units = {u["unit_id"]: u for u in prev["units"]}
print(f"[已有] test-split 单元数: {len(test_units)}")


# ── 2. 推理 val split（获取更多单元）─────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            vals = np.array([float(v) for v in obj["signal_value"].split(",")],
                             dtype=np.float32)
            records.append(vals)
    if not records: return None
    return np.stack(records).mean(axis=0)


def collect_split(split):
    samples = []
    for label_dir, label in [("故障", 1), ("正常", 0)]:
        d = FILT_DIR / split / label_dir
        if not d.exists(): continue
        for p in sorted(d.glob("*.jsonl")):
            uid = p.stem.split("_")[0]
            samples.append({"unit_id": uid, "label": label, "path": p})
    return samples


import torch

def load_model(device="cpu"):
    sys.path.insert(0, str(BASE_DIR))
    from models import AnomalyModelV5
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model = AnomalyModelV5(branch_mode="hetero", fusion_mode="concat",
                           zerone_use_cnn=True, use_modality_dropout=False).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

_gpu_transform = None
def build_img(sig, device):
    global _gpu_transform
    if _gpu_transform is None:
        from gpu_transforms import GPUHeteroTransform
        _gpu_transform = GPUHeteroTransform(size=224, n_scales=64,
                                             signal_len=8192, device=device)
        _gpu_transform.eval()
    sig_t = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return _gpu_transform(sig_t)

@torch.no_grad()
def score_sample(model, img):
    out = model(img, None)
    return float(out["svdd_score"].item()) if isinstance(out, dict) else float(out.item())


print("[推理] 加载 val split 样本...")
device = "cuda" if torch.cuda.is_available() else "cpu"
val_samples = collect_split("val")
print(f"  val 样本数: {len(val_samples)}")

val_unit_scores  = defaultdict(list)
val_unit_labels  = defaultdict(list)

if val_samples:
    model = load_model(device)
    for i, s in enumerate(val_samples):
        if i % 50 == 0: print(f"  进度 {i}/{len(val_samples)}", end="\r")
        sig = load_jsonl(s["path"])
        if sig is None: continue
        try:
            img = build_img(sig, device)
            sc  = score_sample(model, img)
            val_unit_scores[s["unit_id"]].append(sc)
            val_unit_labels[s["unit_id"]].append(s["label"])
        except Exception as e:
            pass
    print(f"\n  val 有效单元: {len(val_unit_scores)}")
else:
    print("  val 目录为空，仅使用 test split")


# ── 3. 合并 val + test 单元 ──────────────────────────────────────────
all_units = {}

# test units（已有结果）
for uid, u in test_units.items():
    all_units[uid] = {
        "mean_score":   u["mean_svdd_score"],
        "fault_ratio":  u["fault_ratio"],
        "n_fault":      u["n_fault"],
        "n_normal":     u["n_normal"],
        "split":        "test",
    }

# val units（新推理）
for uid in val_unit_scores:
    sc_arr = val_unit_scores[uid]
    lb_arr = val_unit_labels[uid]
    if uid in all_units:
        # 同一单元跨split：合并
        old = all_units[uid]
        n_old = old["n_fault"] + old["n_normal"]
        n_new = len(sc_arr)
        merged_score = (old["mean_score"] * n_old + np.mean(sc_arr) * n_new) / (n_old + n_new)
        all_units[uid]["mean_score"]  = merged_score
        all_units[uid]["n_fault"]    += int(np.sum(lb_arr))
        all_units[uid]["n_normal"]   += int(np.sum([1-l for l in lb_arr]))
        all_units[uid]["split"]       = "both"
    else:
        all_units[uid] = {
            "mean_score":  float(np.mean(sc_arr)),
            "fault_ratio": float(np.mean(lb_arr)),
            "n_fault":     int(np.sum(lb_arr)),
            "n_normal":    int(np.sum([1-l for l in lb_arr])),
            "split":       "val",
        }

# 只保留纯故障 (fault_ratio=1) 或纯正常 (fault_ratio=0) 的单元
clean_units = {uid: u for uid, u in all_units.items()
               if u["fault_ratio"] in (0.0, 1.0)}

print(f"\n[合并] 总单元: {len(all_units)}  纯标签单元: {len(clean_units)}")

fault_scores  = np.array([u["mean_score"] for u in clean_units.values() if u["fault_ratio"]==1.0])
normal_scores = np.array([u["mean_score"] for u in clean_units.values() if u["fault_ratio"]==0.0])
print(f"  故障单元: {len(fault_scores)}  正常单元: {len(normal_scores)}")
print(f"  故障均分: {fault_scores.mean():.4f}  正常均分: {normal_scores.mean():.4f}")


# ── 4. Mann-Whitney U 单尾检验 ────────────────────────────────────────
# H1: 故障单元 SVDD 分数 > 正常单元（单尾）
stat, p_mwu = mannwhitneyu(fault_scores, normal_scores, alternative="greater")
n_f, n_n = len(fault_scores), len(normal_scores)
auc_mwu  = stat / (n_f * n_n)  # rank-AUC = P(fault > normal)

print(f"\n── Mann-Whitney U 单尾检验 ──────────────────────")
print(f"  U = {stat:.1f},  p = {p_mwu:.4f}")
print(f"  Rank-AUC (P(fault>normal)) = {auc_mwu:.4f}")
if p_mwu < 0.05:
    print("  *** 显著 (p < 0.05) ***")
elif p_mwu < 0.10:
    print("  趋势显著 (p < 0.10)")
else:
    print("  未达显著性")


# ── 5. 置换检验 ───────────────────────────────────────────────────────
np.random.seed(42)
all_scores   = np.concatenate([fault_scores, normal_scores])
n_perm       = 10000
obs_diff     = fault_scores.mean() - normal_scores.mean()
perm_diffs   = []
for _ in range(n_perm):
    perm = np.random.permutation(all_scores)
    perm_diffs.append(perm[:n_f].mean() - perm[n_f:].mean())
p_perm = np.mean(np.array(perm_diffs) >= obs_diff)
print(f"\n── 置换检验 (n={n_perm}) ─────────────────────────")
print(f"  观测差值: {obs_diff:.4f}")
print(f"  p_perm = {p_perm:.4f}")


# ── 6. Precision@K ────────────────────────────────────────────────────
sorted_units = sorted(clean_units.items(), key=lambda x: x[1]["mean_score"], reverse=True)
n_total_fault = len(fault_scores)
random_prec   = n_total_fault / len(clean_units)

print(f"\n── Precision@K ─────────────────────────────────")
print(f"  随机基线: {random_prec*100:.1f}%")
for K in [5, 10, 15]:
    top_K = sorted_units[:K]
    prec  = sum(1 for _, u in top_K if u["fault_ratio"]==1.0) / K
    print(f"  P@{K:2d} = {prec*100:.1f}%")


# ── 7. 保存结果 ───────────────────────────────────────────────────────
result = {
    "n_units_total":       len(clean_units),
    "n_fault_units":       int(n_f),
    "n_normal_units":      int(n_n),
    "fault_mean_score":    float(fault_scores.mean()),
    "normal_mean_score":   float(normal_scores.mean()),
    "mwu_statistic":       float(stat),
    "mwu_p_onetail":       float(p_mwu),
    "rank_auc":            float(auc_mwu),
    "perm_p_onetail":      float(p_perm),
    "precision_at_5":      sum(1 for _, u in sorted_units[:5]  if u["fault_ratio"]==1.0)/5,
    "precision_at_10":     sum(1 for _, u in sorted_units[:10] if u["fault_ratio"]==1.0)/10,
    "precision_at_15":     sum(1 for _, u in sorted_units[:15] if u["fault_ratio"]==1.0)/15,
    "random_baseline_prec":float(random_prec),
}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n结果已保存: {OUT_JSON}")
print("\n请将 unit_ranking_mwu_results.json 内容告知，以便更新论文。")

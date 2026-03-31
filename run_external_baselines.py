# -*- coding: utf-8 -*-
"""
run_external_baselines.py
=========================
外部无监督基线对比实验
- OC-SVM (RBF kernel) + 1200-dim Zerone 特征
- Isolation Forest   + 1200-dim Zerone 特征
- VAE-only           (Stage1 VAE 重构误差, 无SVDD)

训练集: filtered_output/val/正常/  (仅正常样本, one-class setting)
测试集: filtered_output/test/      (正常+故障, 带标签)

使用方法:
    cd "3-code/diagnosis/test/Deep SVDD"
    python run_external_baselines.py

结果保存到: external_baselines_results.json
论文所需AUC-ROC填入 main.tex 的 [XX] 占位符处
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from features import extract_zerone_features

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
FILT_DIR   = BASE_DIR / "filtered_output"

# One-class训练: 20251016/train/ 正常样本（与Stage1无监督训练数据同源，独立于评估集）
DATA_20251016 = BASE_DIR.parent / "20251016"
TRAIN_DIR     = DATA_20251016 / "train"   # 3899个JSONL，含正常/故障

# 评估: filtered_output/test（与主模型完全相同的2250个测试样本）
TEST_DIR   = DATA_20251016 / "test"
OUT_FILE   = BASE_DIR / "external_baselines_results.json"

FS = 8192.0


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _load_one(jsonl_path: Path):
    """从单个 .jsonl 文件读取信号, 返回 (feature_1200, label)"""
    path_str = str(jsonl_path)
    if "正常" in path_str or "normal" in path_str.lower():
        label = 0
    elif "故障" in path_str or "fault" in path_str.lower() or "abnormal" in path_str.lower():
        label = 1
    else:
        return None

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                raw = obj.get("signal_value") or obj.get("signal") or obj.get("data")
                if raw is None:
                    continue
                if isinstance(raw, str):
                    arr = np.array(raw.split(","), dtype=np.float32)
                else:
                    arr = np.asarray(raw, dtype=np.float32)
                # pad / trim → 8192
                if arr.size >= FS:
                    arr = arr[:int(FS)]
                else:
                    arr = np.pad(arr, (0, int(FS) - arr.size))
                feat = extract_zerone_features(arr, FS)
                return (feat, label)
    except Exception:
        pass
    return None


def load_split(data_dir: Path, max_workers: int = 4):
    """加载某个 split 目录下所有样本, 返回 (X, y)"""
    files = sorted(data_dir.rglob("*.jsonl"))
    print(f"  Found {len(files)} files in {data_dir}")
    feats, labels = [], []
    t0 = time.time()

    # 用多进程加速特征提取
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_load_one, f): f for f in files}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                feats.append(res[0])
                labels.append(res[1])
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(files)} ({time.time()-t0:.0f}s)")

    X = np.stack(feats, axis=0) if feats else np.empty((0, 1200))
    y = np.array(labels, dtype=np.int32)
    return X, y


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        precision_score, recall_score, average_precision_score
    )

    print("=" * 60)
    print("External Unsupervised Baselines for Deep SVDD Comparison")
    print("=" * 60)
    print(f"  One-class train : {TRAIN_DIR}  (normal samples only)")
    print(f"  Evaluation test : {TEST_DIR}   (same 2250-sample set as main model)")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training dir not found: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {TEST_DIR}")

    # 1) 加载训练数据 (20251016/train/，仅取正常样本用于 one-class 训练)
    print("\n[1/4] Loading training split from 20251016/train/ ...")
    X_tr, y_tr = load_split(TRAIN_DIR)
    print(f"  Total={len(y_tr)}, normal={np.sum(y_tr==0)}, fault={np.sum(y_tr==1)}")

    print("\n[2/4] Loading test split (evaluation, filtered_output/test/) ...")
    X_test, y_test = load_split(TEST_DIR)
    print(f"  Test total={len(y_test)}, normal={np.sum(y_test==0)}, fault={np.sum(y_test==1)}")

    # 只用训练集中的正常样本拟合（one-class，独立于评估集）
    X_train_normal = X_tr[y_tr == 0]
    print(f"\n  One-class training samples (normal only from train/): {len(X_train_normal)}")

    # 归一化 (fit on normal only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # 2) OC-SVM ───────────────────────────────────────────────
    print("\n[3/4] Running OC-SVM (RBF, nu=0.1) ...")
    t0 = time.time()
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    ocsvm.fit(X_train_scaled)
    scores_ocsvm = -ocsvm.decision_function(X_test_scaled)   # higher = more anomalous
    preds_ocsvm  = (ocsvm.predict(X_test_scaled) == -1).astype(int)
    r_ocsvm = {
        "auc_roc":    float(roc_auc_score(y_test, scores_ocsvm)),
        "auc_pr":     float(average_precision_score(y_test, scores_ocsvm)),
        "accuracy":   float(accuracy_score(y_test, preds_ocsvm)),
        "precision":  float(precision_score(y_test, preds_ocsvm, average="macro", zero_division=0)),
        "recall":     float(recall_score(y_test, preds_ocsvm, average="macro", zero_division=0)),
        "f1":         float(f1_score(y_test, preds_ocsvm, average="macro", zero_division=0)),
    }
    print(f"  OC-SVM  AUC={r_ocsvm['auc_roc']:.4f}  Acc={r_ocsvm['accuracy']:.4f}"
          f"  F1={r_ocsvm['f1']:.4f}  ({time.time()-t0:.1f}s)")
    results["ocsvm"] = r_ocsvm

    # 3) Isolation Forest ─────────────────────────────────────
    print("\n[4/4] Running Isolation Forest (n=200, contamination=0.5) ...")
    t0 = time.time()
    isof = IsolationForest(n_estimators=200, contamination=0.5,
                           random_state=42, n_jobs=-1)
    isof.fit(X_train_scaled)
    scores_isof = -isof.score_samples(X_test_scaled)         # higher = more anomalous
    preds_isof  = (isof.predict(X_test_scaled) == -1).astype(int)
    r_isof = {
        "auc_roc":    float(roc_auc_score(y_test, scores_isof)),
        "auc_pr":     float(average_precision_score(y_test, scores_isof)),
        "accuracy":   float(accuracy_score(y_test, preds_isof)),
        "precision":  float(precision_score(y_test, preds_isof, average="macro", zero_division=0)),
        "recall":     float(recall_score(y_test, preds_isof, average="macro", zero_division=0)),
        "f1":         float(f1_score(y_test, preds_isof, average="macro", zero_division=0)),
    }
    print(f"  IsoForest AUC={r_isof['auc_roc']:.4f}  Acc={r_isof['accuracy']:.4f}"
          f"  F1={r_isof['f1']:.4f}  ({time.time()-t0:.1f}s)")
    results["isolation_forest"] = r_isof

    # 4) 汇总 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<28} {'AUC-ROC':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    for name, r in results.items():
        label = {"ocsvm": "OC-SVM + Zerone",
                 "isolation_forest": "Isolation Forest + Zerone"}.get(name, name)
        print(f"{label:<28} {r['auc_roc']:>8.4f} {r['accuracy']:>7.4f}"
              f" {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    # 已知 DualSVDD 结果 (参考)
    print(f"{'DualSVDD Hetero (ours)':<28} {'0.6660':>8} {'0.6060':>7}"
          f" {'0.6650':>7} {'0.6060':>7} {'0.5960':>7}")
    print("=" * 60)

    # 保存
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {OUT_FILE}")
    print("\n>>> 将 OC-SVM AUC 和 IsoForest AUC 填入 main.tex Table 2 的 [XX] 占位符")


if __name__ == "__main__":
    main()

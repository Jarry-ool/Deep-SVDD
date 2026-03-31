# -*- coding: utf-8 -*-
"""
run_ablations.py
================
运行消融实验，获取 paper_2/main.tex Table 3 中 [TBD] 的精确数值。

变体 (仅改 Stage1 损失系数，其余配置与 v512 Hetero 完全一致):
  --variant no_diversity  : 去掉 L_div (令 LAMBDA_DIV=0)
  --variant no_vae        : 去掉 L_vae (令 LAMBDA_VAE=0 → 纯SVDD)
  --variant no_mmd        : Stage3 关闭 MMD (DA_WEIGHT=0)

使用方法:
    cd "3-code/diagnosis/test/Deep SVDD"
    python run_ablations.py --variant no_diversity
    python run_ablations.py --variant no_vae
    python run_ablations.py --variant no_mmd
    python run_ablations.py --summarize

结果目录: ablation_results/<variant>/
汇总文件: ablation_summary.json

依赖: training.py 已通过 getattr(cfg, 'LAMBDA_*', default) 支持损失系数控制。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR     = Path(__file__).parent
FILT_DIR     = BASE_DIR / "filtered_output"
ABLATION_DIR = BASE_DIR / "ablation_results"

# 参考基线 (full model, v512 Hetero, 2250-sample fleet test)
FULL_RESULT = {
    "variant":        "full_model (Hetero v512)",
    "test_accuracy":  0.6058,
    "test_f1":        0.5959,
    "test_precision": 0.6650,
    "test_recall":    0.6058,
}


def build_cfg(variant: str):
    from config import ThreeStageConfigV5
    cfg = ThreeStageConfigV5(
        BRANCH_MODE              = "hetero",
        FUSION_MODE              = "concat",
        ZERONE_USE_CNN           = True,
        USE_MODALITY_DROPOUT     = False,
        USE_GLOBAL_NORMALIZATION = True,
        OUTPUT_ROOT              = Path(f"./ablation_results/{variant}"),
    )
    if variant == "no_diversity":
        cfg.LAMBDA_DIV = 0.0        # training.py 使用 getattr(cfg, 'LAMBDA_DIV', 0.001)
    elif variant == "no_vae":
        cfg.LAMBDA_VAE = 0.0        # 纯SVDD，无VAE辅助
    elif variant == "no_mmd":
        cfg.USE_DOMAIN_ADAPTATION = False
        cfg.DA_WEIGHT = 0.0
    return cfg


def run_variant(variant: str):
    from main import prepare_datasets
    from training import run_full_pipeline

    print(f"\n{'='*60}\nAblation: {variant}\n{'='*60}")

    cfg = build_cfg(variant)
    out_dir = ABLATION_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集（使用 filtered_output 预过滤结果，与主实验完全一致）
    print("[数据] 加载数据集 ...")
    train_ds, val_ds, test_ds, _ = prepare_datasets(cfg, filter_output_dir=FILT_DIR)

    # 运行完整三阶段流程
    run_full_pipeline(cfg, train_ds, val_ds, test_ds)

    # 读取 train_stage3 保存的评估结果
    eval_json = cfg.STAGE3_DIR / "evaluation_results.json"
    if eval_json.exists():
        with open(eval_json) as f:
            metrics = json.load(f)
        out = {
            "variant":        variant,
            "test_accuracy":  metrics.get("test_accuracy"),
            "test_f1":        metrics.get("test_f1"),
            "test_precision": metrics.get("test_precision"),
            "test_recall":    metrics.get("test_recall"),
        }
    else:
        print(f"[警告] 未找到评估结果: {eval_json}")
        out = {"variant": variant, "test_accuracy": None, "test_f1": None,
               "note": "evaluation_results.json not found"}

    with open(out_dir / "test_evaluation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f">>> Saved → {out_dir / 'test_evaluation.json'}")
    return out


def summarize():
    variants = ["no_diversity", "no_vae", "no_mmd"]
    rows = []
    for v in variants:
        p = ABLATION_DIR / v / "test_evaluation.json"
        if p.exists():
            with open(p) as f:
                rows.append(json.load(f))
        else:
            rows.append({"variant": v, "test_accuracy": None,
                         "test_f1": None, "note": "not run yet"})
    rows.append(FULL_RESULT)

    print("\n" + "=" * 55)
    print("Table 3 — Ablation Summary (Hetero branch, 2250-sample fleet test)")
    print(f"{'Variant':<28} {'Acc':>8} {'F1':>8}")
    print("-" * 55)
    for r in rows:
        acc = f"{r['test_accuracy']:.4f}" if r.get("test_accuracy") is not None else "[TBD]"
        f1  = f"{r['test_f1']:.4f}"       if r.get("test_f1")       is not None else "[TBD]"
        print(f"{r['variant']:<28} {acc:>8} {f1:>8}")
    print("=" * 55)

    with open(BASE_DIR / "ablation_summary.json", "w") as f:
        json.dump({r["variant"]: r for r in rows}, f, indent=2)
    print(">>> Saved → ablation_summary.json")
    print(">>> Replace [TBD] in main.tex Table 3 with these numbers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["no_diversity", "no_vae", "no_mmd"])
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    elif args.variant:
        run_variant(args.variant)
    else:
        ap.print_help()

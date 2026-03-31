# -*- coding: utf-8 -*-
"""
speed_profile_v5.py  (精简版：只测速，不做全局归一化/并行提特征)
================================================================
输出每 step 三段耗时：
  1) fetch_time: next(it) 取 batch 时间（数据读取/生成/预处理）
  2) h2d_time  : batch 搬到 GPU 时间
  3) gpu_time  : forward + loss + backward + step 时间

用法示例：
  python speed_profile_v5.py --labeled_dir "E:/CODE/code/20251016" --batch_size 128 --num_workers 0 --steps 50
  python speed_profile_v5.py --filter_output_dir "./filtered_output" --labeled_dir "E:/CODE/code/20251016" --batch_size 128 --num_workers 2 --prefetch_factor 1 --persistent_workers --steps 50
  python speed_profile_v5.py --filter_output_dir "./filtered_output" --amp --steps 50
"""

import os, time, argparse
from pathlib import Path
import numpy as np
import torch

# ---- 避免线程爆炸（尤其 FFT/CWT/STFT）----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---- 项目内导入（注意：不再 import main.prepare_datasets）----
from config import ThreeStageConfigV5
from data_manager import DataSplitManager, CHANNEL_MANAGER, scan_csv_files
from datasets import CSVVibrationDataset, LabeledVibrationDataset, TransformerVibrationDataset
from models import AnomalyModelV5


def _worker_init_fn(worker_id: int):
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def make_loader(ds, batch_size: int, num_workers: int, pin_memory: bool,
                prefetch_factor: int, persistent_workers: bool, timeout: int):
    from torch.utils.data import DataLoader
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs.update(dict(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            timeout=timeout,
            worker_init_fn=_worker_init_fn,
        ))
    return DataLoader(ds, **kwargs)


def to_device(batch, device: torch.device, non_blocking: bool):
    """
    适配你数据集返回 (hetero_img, zerone_img_or_feat, label, idx/metadata)
    """
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise RuntimeError(f"Unexpected batch type/len: {type(batch)} {getattr(batch, '__len__', None)}")

    x_img = batch[0].to(device, non_blocking=non_blocking)
    x_zero = batch[1].to(device, non_blocking=non_blocking)
    return x_img, x_zero


@torch.no_grad()
def quick_dataset_probe(ds, n=3):
    print("\n" + "="*70)
    print("[Probe] __getitem__ 单样本耗时（取 ds[0] 多次）")
    if len(ds) == 0:
        print("  [跳过] 数据集为空")
        print("="*70)
        return
    for k in range(n):
        t0 = time.perf_counter()
        _ = ds[0]
        dt = time.perf_counter() - t0
        print(f"  ds[0] 第{k+1}次: {dt:.4f}s")
    print("="*70)


def prepare_datasets_fast(cfg: ThreeStageConfigV5,
                         filter_output_dir: Path = None,
                         labeled_dir: Path = None,
                         val_ratio: float = 0.5):
    """
    精简的数据准备：只构建 train/val/test，不做：
      - 全局归一化统计（不会出现 TRAIN 并行提特征）
      - 保存 split / 映射文件
    """
    print("\n" + "="*60)
    print("【数据准备(精简)】")
    print("="*60)

    split_manager = DataSplitManager(cfg)

    # 方式1：预过滤输出
    if filter_output_dir and Path(filter_output_dir).exists():
        print(f"\n[方式1] 使用预过滤结果: {filter_output_dir}")
        split_manager.load_from_filter_output(filter_output_dir)

        mapping_path = Path(filter_output_dir) / "channel_mapping.json"
        if mapping_path.exists():
            CHANNEL_MANAGER.load_mapping(mapping_path)

        val_ds = LabeledVibrationDataset(split_manager.val_samples, cfg, split_name="VAL", normalizer=None)
        test_ds = LabeledVibrationDataset(split_manager.test_samples, cfg, split_name="TEST", normalizer=None)

        csv_files = scan_csv_files(cfg.RAW_DATA_DIR)
        if csv_files:
            train_ds = CSVVibrationDataset(
                csv_files, cfg, use_labels=False, split_name="TRAIN",
                normalizer=None, excluded_ids=split_manager.excluded_ids
            )
        else:
            train_ds = TransformerVibrationDataset(cfg.PROJECT_ROOT, cfg, use_labels=False, split_name="TRAIN", normalizer=None)

    # 方式2：只有 labeled_dir
    elif labeled_dir and Path(labeled_dir).exists():
        print(f"\n[方式2] 自动划分已标注数据: {labeled_dir}")

        split_manager.auto_split_labeled_data(
            labeled_dir, val_ratio=val_ratio,
            class_keywords=cfg.CLASS_KEYWORDS
        )
        split_manager.generate_excluded_ids_from_labeled()

        val_ds = LabeledVibrationDataset(split_manager.val_samples, cfg, split_name="VAL", normalizer=None)
        test_ds = LabeledVibrationDataset(split_manager.test_samples, cfg, split_name="TEST", normalizer=None)

        csv_files = scan_csv_files(cfg.RAW_DATA_DIR)
        if csv_files:
            train_ds = CSVVibrationDataset(
                csv_files, cfg, use_labels=False, split_name="TRAIN",
                normalizer=None, excluded_ids=split_manager.excluded_ids
            )
        else:
            train_ds = TransformerVibrationDataset(cfg.PROJECT_ROOT, cfg, use_labels=False, split_name="TRAIN", normalizer=None)

    # 方式3：默认
    else:
        print(f"\n[方式3] 默认加载: {cfg.PROJECT_ROOT}")
        train_ds = TransformerVibrationDataset(cfg.PROJECT_ROOT, cfg, use_labels=False, split_name="TRAIN", normalizer=None)
        val_ds = TransformerVibrationDataset(cfg.PROJECT_ROOT, cfg, use_labels=True, split_name="VAL", normalizer=None)
        test_ds = TransformerVibrationDataset(cfg.PROJECT_ROOT, cfg, use_labels=True, split_name="TEST", normalizer=None)

    print("\n[Dataset sizes]")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_ds, val_ds, test_ds, split_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_output_dir", type=str, default="", help="预过滤输出目录（优先使用）")
    parser.add_argument("--labeled_dir", type=str, default="", help="已标注数据目录（用于自动划分val/test）")
    parser.add_argument("--val_ratio", type=float, default=0.5)

    parser.add_argument("--branch_mode", type=str, default="dual", choices=["hetero", "zerone", "dual"])
    parser.add_argument("--fusion_mode", type=str, default="gmu", choices=["concat", "attention", "gate", "gmu"])
    parser.add_argument("--zerone_use_cnn", action="store_true", help="Zerone 用 CNN 输入（默认按cfg）")
    parser.add_argument("--zerone_use_mlp", action="store_true", help="Zerone 用 MLP 输入（覆盖cfg）")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)

    parser.add_argument("--amp", action="store_true", help="启用混合精度（推荐对比）")
    parser.add_argument("--compile", action="store_true", help="torch.compile（PyTorch2）")
    parser.add_argument("--non_blocking", action="store_true", help="to(device, non_blocking=True)")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # ---- 共享内存策略（Linux / Windows 都可）----
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- cfg ----
    cfg = ThreeStageConfigV5()
    cfg.BRANCH_MODE = args.branch_mode
    cfg.FUSION_MODE = args.fusion_mode
    if args.zerone_use_mlp:
        cfg.ZERONE_USE_CNN = False
    elif args.zerone_use_cnn:
        cfg.ZERONE_USE_CNN = True

    print("\n" + "="*70)
    print("[Config]")
    print(f"  device={device}")
    print(f"  branch_mode={cfg.BRANCH_MODE}  fusion_mode={cfg.FUSION_MODE}  ZERONE_USE_CNN={cfg.ZERONE_USE_CNN}")
    print(f"  batch_size={args.batch_size}  num_workers={args.num_workers}  pin_memory={args.pin_memory}")
    if args.num_workers > 0:
        print(f"  prefetch_factor={args.prefetch_factor}  persistent_workers={args.persistent_workers}  timeout={args.timeout}")
    print(f"  steps={args.steps}  warmup={args.warmup}  amp={args.amp}  compile={args.compile}")
    print("="*70)

    # ---- datasets（精简版，不会并行提特征）----
    train_ds, val_ds, test_ds, _ = prepare_datasets_fast(
        cfg,
        filter_output_dir=Path(args.filter_output_dir) if args.filter_output_dir else None,
        labeled_dir=Path(args.labeled_dir) if args.labeled_dir else None,
        val_ratio=args.val_ratio
    )

    # 选 profiling 数据集：优先 train，没有就用 val（避免 train=0 直接崩）
    profile_ds = train_ds if len(train_ds) > 0 else val_ds
    if len(profile_ds) == 0:
        raise RuntimeError("train/val 都是空的：请检查 --labeled_dir 或 --filter_output_dir 是否正确")

    quick_dataset_probe(profile_ds, n=3)

    loader = make_loader(
        profile_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        timeout=args.timeout
    )

    # ---- model ----
    model = AnomalyModelV5(
        branch_mode=cfg.BRANCH_MODE,
        fusion_mode=cfg.FUSION_MODE,
        zerone_use_cnn=cfg.ZERONE_USE_CNN,
        use_modality_dropout=getattr(cfg, "USE_MODALITY_DROPOUT", True),
        modality_dropout_p=getattr(cfg, "MODALITY_DROPOUT_RATE", 0.2),
        use_layernorm=getattr(cfg, "USE_LAYERNORM", False),
        dropout_rate=getattr(cfg, "DROPOUT_RATE", 0.3),
        latent_dim=getattr(cfg, "LATENT_DIM", 64),
        latent_channels=getattr(cfg, "LATENT_CHANNELS", 16),
        has_vae=True,
    ).to(device)
    model.train()

    if args.compile:
        try:
            model = torch.compile(model)
            print("[OK] torch.compile enabled")
        except Exception as e:
            print("[WARN] torch.compile failed:", repr(e))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    it = iter(loader)

    def step_once():
        nonlocal it
        # 1) fetch
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t1 = time.perf_counter()

        # 2) H2D
        x_img, x_zero = to_device(batch, device, non_blocking=args.non_blocking)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # 3) GPU compute
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            out = model(x_img, x_zero)
            # 对齐你“VAE预训练”阶段：只算 VAE 重构 + KL
            loss = out["vae_recon_loss"].mean() + 0.01 * out["vae_kl"].mean()

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        return (t1 - t0), (t2 - t1), (t3 - t2), float(loss.detach().cpu().item())

    # warmup
    print("\n[Warmup] ...")
    for _ in range(max(0, args.warmup)):
        step_once()

    # measure
    fetch_list, h2d_list, gpu_list, loss_list = [], [], [], []
    print("\n[Measure] ...")
    for k in range(args.steps):
        ft, ht, gt, lv = step_once()
        fetch_list.append(ft); h2d_list.append(ht); gpu_list.append(gt); loss_list.append(lv)
        if (k + 1) % 10 == 0:
            print(f"  step {k+1:4d}/{args.steps} | fetch={np.mean(fetch_list):.3f}s | h2d={np.mean(h2d_list):.3f}s | gpu={np.mean(gpu_list):.3f}s | loss~{np.mean(loss_list):.4f}")

    fetch = float(np.mean(fetch_list))
    h2d = float(np.mean(h2d_list))
    gpu = float(np.mean(gpu_list))
    total = fetch + h2d + gpu
    it_per_s = 1.0 / max(total, 1e-9)

    print("\n" + "="*70)
    print("[RESULT]")
    print(f"  fetch_time(avg) = {fetch:.4f}s  (数据读取/生成/预处理)")
    print(f"  h2d_time(avg)   = {h2d:.4f}s  (CPU->GPU搬运)")
    print(f"  gpu_time(avg)   = {gpu:.4f}s  (forward+backward+step)")
    print(f"  total(avg)      = {total:.4f}s  => {it_per_s:.3f} it/s")
    print(f"  loss(avg)       = {float(np.mean(loss_list)):.6f}")
    print("="*70)

    print("\n[如何判读]")
    print("  - fetch_time 大：瓶颈在 Dataset/__getitem__/磁盘IO（优先缓存/减少构图/线程池）")
    print("  - gpu_time   大：瓶颈在模型训练（优先 AMP/compile/channels_last/更大batch+accum）")
    print("  - h2d_time   大：pin_memory=True + non_blocking=True 往往有帮助（但先确保不死机）")


if __name__ == "__main__":
    main()

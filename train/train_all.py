#!/usr/bin/env python3
"""Unified training script — trains all (or selected) denoising models on a dataset.

Each model gets its own W&B run (when --wandb-project is set).
A Markdown + JSON report is generated in <dataset>/weights/ at the end.

Usage:
    python train/train_all.py \
        --dataset data_generation/datasets/deep_space_..._39075e4f \
        --noise-type non_gaussian \
        --models all \
        --epochs 50 \
        --wandb-project sd-science
"""

import argparse
import gc
import json
import os
import sys
import time
import uuid as _uuid_mod
from datetime import datetime
from pathlib import Path

# Allow MPS to fall back to CPU for unsupported ops (e.g. complex tensor operations).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aws_scripts.download_dataset_from_s3 import download_dataset_from_s3
from aws_scripts.push_runs_to_s3 import push_runs_to_s3
from train.compare_report import run_compare_report

import wandb

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

try:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        WANDB_OK = True
    else:
        WANDB_OK = False
except Exception as e:
    print(f"Warning: W&B login failed: {e}")
    WANDB_OK = False

CLOUD_TRAINING = os.getenv("CLOUD_TRAINING", "False") == "True"
# Transformer first — largest VRAM consumer (O(T²) attention), trains safely
# before GPU memory gets fragmented by smaller models.
ALL_MODELS = ["transformer", "unet", "vae", "resnet", "hybrid", "wavelet"]
from train.device_utils import get_device, empty_cache, get_batch_size_multiplier


# Per-model batch sizes measured on ~8 GiB GPU with signal_len=1024, nperseg=128.
# Values calibrated from actual vram= logs; target ≤ 6.5 GiB peak (fwd+bwd).
# --batch-size overrides all of these when explicitly provided.
MODEL_BATCH_SIZES = {
    "transformer": 128,   # O(T²) attention at T=1024; measured 4.99 GB
    "unet":        1024,  # torch.stft on GPU; measured 6.47 GB — ~1 GB margin on 8 GB card
    "vae":         8192,  # torch.stft on GPU; measured 3.91 GB
    "resnet":      2048,  # torch.stft on GPU; measured 4.82 GB
    "hybrid":      4096,  # torch.stft 4-ch on GPU; measured 5.16 GB
    "wavelet":     512,   # CPU-only
}

# Per-model default learning rates.
# Transformer needs a 10× higher LR — at 1e-4 it stalls at epoch 1 (loss ≈ E[x²],
# meaning it outputs near-zero and never escapes that plateau).
# All other models converge well but reach the fine-tuning plateau only after many
# epochs at 1e-4; 3e-4 shortens the initial descent without causing instability.
# --lr overrides all of these when explicitly provided.
MODEL_LEARNING_RATES = {
    "transformer": 1e-3,  # stuck at 1e-4; 10× increase needed to escape init plateau
    "unet":        1e-3,
    "vae":         6e-4,
    "resnet":      6e-4,
    "hybrid":      3e-4,
    "wavelet":     None,  # not applicable (grid search)
}

# Allow overriding via environment variables (e.g., from RunPod/GitHub Action)
# Expects JSON string: MODEL_BATCH_SIZES='{"unet": 512, "vae": 1024}'
env_batch_sizes = os.getenv("MODEL_BATCH_SIZES")
if env_batch_sizes:
    try:
        overrides = json.loads(env_batch_sizes)
        MODEL_BATCH_SIZES.update(overrides)
        print(f"INFO: Overriding MODEL_BATCH_SIZES from env: {overrides}")
    except Exception as e:
        print(f"Warning: Failed to parse MODEL_BATCH_SIZES env var: {e}")

def _resolve_batch_size(args, model_key: str) -> int:
    """Return explicit --batch-size if given, else default × device multiplier."""
    if args.batch_size is not None:
        return args.batch_size
    env_batch_sizes = os.getenv("MODEL_BATCH_SIZES")
    if env_batch_sizes:
        try:
            overrides = json.loads(env_batch_sizes)
            MODEL_BATCH_SIZES.update(overrides)
            print(f"INFO: Overriding MODEL_BATCH_SIZES from env: {overrides}")
            return MODEL_BATCH_SIZES.get(model_key, 4)
        except Exception as e:
            print(f"Warning: Failed to parse MODEL_BATCH_SIZES env var: {e}")
    else:
        base = MODEL_BATCH_SIZES[model_key]
        mult = getattr(args, '_bs_mult', 1.0)
        # Transformer has O(T²) memory in attention — don't scale batch with device multiplier.
        if model_key == "transformer":
            mult = 1.0
        return int(base * mult)


env_lrs = os.getenv("MODEL_LEARNING_RATES")
if env_lrs:
    try:
        overrides = json.loads(env_lrs)
        MODEL_LEARNING_RATES.update(overrides)
        print(f"INFO: Overriding MODEL_LEARNING_RATES from env: {overrides}")
    except Exception as e:
        print(f"Warning: Failed to parse MODEL_LEARNING_RATES env var: {e}")


# ── model runners ─────────────────────────────────────────────────────────────

def run_unet(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_uae import UnetAutoencoderTrainer
    print("\n" + "=" * 60)
    print("=== UNet (Mask + STFT, MSELoss) ===")
    print("=" * 60)
    bs = _resolve_batch_size(args, "unet")
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["unet"]
    return UnetAutoencoderTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        noverlap=args.nperseg * 3 // 4,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
        run_id=args.run_id,
        device=args.device,
    ).train()


def run_resnet(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_resnet import ResNetAutoencoderTrainer
    print("\n" + "=" * 60)
    print("=== ResNet (STFT autoencoder, MSELoss) ===")
    print("=" * 60)
    bs = _resolve_batch_size(args, "resnet")
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["resnet"]
    return ResNetAutoencoderTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
        run_id=args.run_id,
        device=args.device,
    ).train()


def run_vae(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_vae import VAETrainer
    print("\n" + "=" * 60)
    print("=== VAE (SpectrogramVAE, MSELoss + KL) ===")
    print("=" * 60)
    bs = _resolve_batch_size(args, "vae")
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["vae"]
    return VAETrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
        run_id=args.run_id,
        device=args.device,
    ).train()


def run_transformer(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_transformer import TransformerTrainer
    print("\n" + "=" * 60)
    print("=== Transformer (time-domain, MSELoss) ===")
    print("=" * 60)
    bs = _resolve_batch_size(args, "transformer")
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["transformer"]
    return TransformerTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
        run_id=args.run_id,
        device=args.device,
    ).train()


def run_wavelet(dataset_dir: Path, cfg: dict, args) -> dict | None:
    from train.wavelet_grid_search import grid_search_wavelet
    import numpy as np
    print("\n" + "=" * 60)
    print("=== Wavelet (grid search) ===")
    print("=" * 60)
    noisy = np.load(dataset_dir / "train" / f"{args.noise_type}_signals.npy")
    clean = np.load(dataset_dir / "train" / "clean_signals.npy")
    if args.partial_train < 1.0:
        n = max(1, int(len(noisy) * args.partial_train))
        noisy, clean = noisy[:n], clean[:n]
    best_params, val_mse, test_mse = grid_search_wavelet(noisy, clean, random_state=args.seed)
    print(f"  Best params: {best_params}")
    print(f"  Val MSE: {val_mse:.6f}  Test MSE: {test_mse:.6f}")
    run_dir = args.shared_run_dir / f"Wavelet_{args.noise_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "best_params.json"
    with open(save_path, "w") as f:
        json.dump({"best_params": best_params, "val_mse": val_mse, "test_mse": test_mse}, f, indent=2)
    print(f"  Saved: {save_path}")
    return {
        'model': 'Wavelet', 'noise_type': args.noise_type,
        'val_snr': None, 'test_metrics': {'MSE': test_mse},
        'weights_path': str(save_path), 'per_snr_results': {},
    }


def run_hybrid(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_hybrid import HybridUnetTrainer
    dsge_variant = getattr(args, 'dsge_variant', 'A')
    dsge_basis = getattr(args, 'dsge_basis', 'robust')
    dsge_order = getattr(args, 'dsge_order', 3)
    print("\n" + "=" * 60)
    print(f"=== HybridDSGE_UNet ({dsge_basis} S={dsge_order} v{dsge_variant}, MSELoss) ===")
    print("=" * 60)
    bs = _resolve_batch_size(args, "hybrid")
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["hybrid"]
    return HybridUnetTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        dsge_order=dsge_order,
        dsge_basis=dsge_basis,
        dsge_variant=dsge_variant,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        noverlap=args.nperseg * 3 // 4,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
        run_id=args.run_id,
        device=args.device,
    ).train()


_RUNNERS = {
    "unet":        run_unet,
    "resnet":      run_resnet,
    "vae":         run_vae,
    "transformer": run_transformer,
    "wavelet":     run_wavelet,
    "hybrid":      run_hybrid,
}


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(results: list, dataset_dir: Path, args, weights_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_dir.name,
        "noise_type": args.noise_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "models": [r for r in results if r is not None],
    }

    json_path = weights_dir / f"training_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_path = weights_dir / f"training_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Training Report\n\n")
        f.write(f"**Dataset:** `{dataset_dir.name}`  \n")
        f.write(f"**Noise type:** {args.noise_type}  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Epochs:** {args.epochs} | **Batch:** {args.batch_size} | **LR:** {args.lr}  \n\n")
        f.write("## Results\n\n")
        f.write("| Model | Val SNR | Test SNR | Test MSE | Weights |\n")
        f.write("|-------|--------:|--------:|---------:|---------|\n")
        for r in results:
            if r is None:
                continue
            val_snr  = r.get('val_snr')
            test_snr = r.get('test_metrics', {}).get('SNR')
            test_mse = r.get('test_metrics', {}).get('MSE')
            val_str  = f"{val_snr:.2f} dB"  if val_snr  is not None else "—"
            snr_str  = f"{test_snr:.2f} dB" if test_snr is not None else "—"
            mse_str  = f"{test_mse:.6f}"    if test_mse is not None else "—"
            wpath = Path(r.get('weights_path', ''))
            try:
                wname = str(wpath.relative_to(weights_dir))
            except ValueError:
                wname = wpath.name
            f.write(f"| {r['model']} ({r.get('noise_type', '')}) | {val_str} | {snr_str} | {mse_str} | `{wname}` |\n")

        # per-SNR table (first model that has it)
        for r in results:
            if r and r.get('per_snr_results'):
                f.write("\n## Per-SNR Performance\n\n")
                f.write("| Model | SNR_in | SNR_out | MSE |\n")
                f.write("|-------|-------:|--------:|----:|\n")
                for lbl, m in sorted(r['per_snr_results'].items(),
                                     key=lambda kv: kv[1]['snr_in_db']):
                    f.write(f"| {r['model']} | {m['snr_in_db']:.0f} dB | "
                            f"{m['SNR']:.2f} dB | {m['MSE']:.6f} |\n")

    print(f"\n📋 Report → {md_path}")
    return md_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train all denoising models on a dataset")
    p.add_argument("--dataset",       required=True,
                   help="Path to dataset folder (absolute or relative to project root)")
    p.add_argument("--noise-types",   default="all",
                   help="Comma-separated or 'all'. Options: gaussian, non_gaussian")
    p.add_argument("--models",        default="all",
                   help=f"Comma-separated or 'all'. Options: {', '.join(ALL_MODELS)}")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=None,
                   help="Override batch size for all models (default: per-model from MODEL_BATCH_SIZES)")
    p.add_argument("--lr",            type=float, default=None,
                   help="Learning rate override for all models (default: per-model from MODEL_LEARNING_RATES)")
    p.add_argument("--nperseg",       type=int,   default=128,
                   help="STFT window size for spectral models (default 128 for 1024-sample signals)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "signal-denoising-v2"),
                   help="W&B project name (empty = disable)")
    p.add_argument("--partial-train", type=float, default=1.0, metavar="FRACTION",
                   help="Fraction of dataset to use (0 < f <= 1). Useful for quick debug runs.")
    p.add_argument("--run-id", default=None,
                   help="Optional run ID (e.g. run_20260330_8c4d2660). If not provided, one will be generated.")
    p.add_argument("--device", default="cuda",
                   choices=["cuda", "mps", "cpu", "auto"],
                   help="Force a specific device (default: auto-detect cuda → mps → cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("🚀 Unified Training Script Starting")
    print("=" * 60)
    print("Input Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 60 + "\n")

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir

    if not dataset_dir.exists() or CLOUD_TRAINING:
        # Try to download from S3 if it's just a name or relative path that doesn't exist
        # OR if CLOUD_TRAINING is True (ensure we have the dataset)
        dataset_name = Path(args.dataset).name
        # We assume datasets are stored in data_generation/datasets/
        default_datasets_root = ROOT / "data_generation" / "datasets"
        potential_dir = default_datasets_root / dataset_name

        if not potential_dir.exists() or CLOUD_TRAINING:
            if download_dataset_from_s3(dataset_name, potential_dir):
                dataset_dir = potential_dir
            elif not potential_dir.exists():
                print(f"ERROR: dataset not found and failed to download: {dataset_dir}")
                sys.exit(1)
        else:
            dataset_dir = potential_dir

    with open(dataset_dir / "dataset_config.json") as f:
        cfg = json.load(f)

    # Resolve device and apply batch-size scaling for MPS / large-memory systems.
    device_pref = args.device if args.device != 'auto' else None
    resolved_device = get_device(device_pref)
    bs_mult = get_batch_size_multiplier(resolved_device)
    args.device = str(resolved_device)          # pass as string to trainer constructors
    args._device = resolved_device              # torch.device for empty_cache()
    args._bs_mult = bs_mult                     # for _resolve_batch_size()

    print(f"Device  : {resolved_device}" + (f" (batch multiplier ×{bs_mult:.1f})" if bs_mult != 1.0 else ""))
    print(f"Dataset : {dataset_dir.name}")
    print(f"Config  : block_size={cfg['block_size']}, sample_rate={cfg['sample_rate']}, "
          f"scenario={cfg.get('scenario', '?')}")
    noise_types = (
        ["gaussian", "non_gaussian"] if args.noise_types == "all"
        else [n.strip() for n in args.noise_types.split(",")]
    )
    if not (0.0 < args.partial_train <= 1.0):
        print(f"ERROR: --partial-train must be in (0, 1], got {args.partial_train}")
        sys.exit(1)
    lr_display = f"{args.lr}" if args.lr is not None else "per-model"
    print(f"Training: noise_types={noise_types}, epochs={args.epochs}, "
          f"batch={args.batch_size}, lr={lr_display}"
          + (f", partial={args.partial_train:.0%}" if args.partial_train < 1.0 else ""))

    if args.run_id:
        shared_run_dir = dataset_dir / "runs" / args.run_id
    else:
        run_date = datetime.now().strftime("%Y%m%d")
        run_uid = _uuid_mod.uuid4().hex[:8]
        args.run_id = f"run_{run_date}_{run_uid}"
        shared_run_dir = dataset_dir / "runs" / args.run_id

    shared_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {shared_run_dir.relative_to(dataset_dir)}")
    args.shared_run_dir = shared_run_dir

    if not args.wandb_project:
        reason = "WANDB_API_KEY not set" if not WANDB_OK else "no --wandb-project given"
        print(f"[W&B] Logging disabled ({reason})")
    else:
        print(f"[W&B] Logging enabled → project='{args.wandb_project}' (one run per model)")

    models_to_train = (
        ALL_MODELS if args.models == "all"
        else [m.strip() for m in args.models.split(",")]
    )
    print("Noise types:", noise_types)
    print("Models to train:", models_to_train)
    print("Count:", len(models_to_train))
    print("Unique count:", len(set(models_to_train)))

    results = []
    for noise_type in noise_types:
        args.noise_type = noise_type
        print(f"\n{'#' * 60}")
        print(f"# Noise type: {noise_type}")
        print(f"{'#' * 60}")
        for m in models_to_train:
            runner = _RUNNERS.get(m)
            if runner is None:
                print(f"Unknown model '{m}', skipping")
                continue

            start_t = time.time()
            try:
                result = runner(dataset_dir, cfg, args)
                results.append(result)

                elapsed = time.time() - start_t
                print(f"\n" + "=" * 60)
                print(f"✅ ALL DONE FOR: {m} ({noise_type})")
                print(f"   Total Runner Time: {elapsed // 60:.0f}m {elapsed % 60:.1f}s")
                print("=" * 60 + "\n")

            except Exception as exc:
                elapsed = time.time() - start_t
                print(f"\n" + "!" * 60)
                print(f"❌ TRAINING FAILED: {m} ({noise_type})")
                print(f"   Time elapsed: {elapsed // 60:.0f}m {elapsed % 60:.1f}s")
                print(f"   Error: {exc}")
                print("!" * 60 + "\n")

                results.append({'model': m, 'noise_type': noise_type, 'error': str(exc)})
                exc.__traceback__ = None  # release GPU tensor refs held in traceback frames
            finally:
                gc.collect()
                try:
                    empty_cache(args._device)
                except Exception:
                    pass

    generate_report(results, dataset_dir, args, shared_run_dir)
    print(f"\n✅ Done. Weights and report saved to: {shared_run_dir}")

    if CLOUD_TRAINING:
        print(f"\n📊 Generating comprehensive comparison report...")
        try:
            run_compare_report(shared_run_dir, nperseg=args.nperseg, seed=args.seed)
        except Exception as e:
            print(f"Warning: Comprehensive comparison report failed: {e}")

        print(f"\n🚀 Cloud Training mode: Pushing results to S3...")
        push_runs_to_s3(dataset_dir.name, str(shared_run_dir))

        print(f"Terminating pod as training is finished...")
        # Run termination script via subprocess for robustness
        import subprocess
        script_path = ROOT / ".github/scripts/terminate_pod.py"
        if script_path.exists():
            subprocess.run([sys.executable, str(script_path)], check=False)
        else:
            print(f"Warning: Termination script not found at {script_path}")


if __name__ == "__main__":
    main()

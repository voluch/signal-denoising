import sys
import os
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pywt

from metrics import MeanSquaredError
from models.wavelet import WaveletDenoising


GRID = {
    "wavelet":    ["db4", "sym4", "coif1"],
    "level":      [2, 3, 4],
    "thresh_mode":["soft", "hard"],
    "per_level":  [False],
    "ext_mode":   ["symmetric"],
}


def grid_search_wavelet(noisy: np.ndarray,
                        clean: np.ndarray,
                        random_state: int = 42,
                        param_grid: dict[str, list] | None = None,
                        max_val_samples: int = 1000,
                        max_test_samples: int = 1000):
    """
    WaveletDenoising grid search by MSE.

    Minimal safety changes:
    - Do not copy full validation/test arrays with noisy[val_idx].
    - Evaluate only a capped number of validation/test samples.
    """
    import time
    start_time = time.time()

    print(f"\nTraining Configuration for Wavelet (Grid Search):")
    print(f"  Random Seed:       {random_state}")
    print(f"  Param Grid:        {param_grid or GRID}")
    print(f"  Max Val Samples:   {max_val_samples}")
    print(f"  Max Test Samples:  {max_test_samples}")

    assert noisy.shape == clean.shape, "noisy/clean shapes must match"
    N = len(noisy)

    rng = np.random.default_rng(random_state)
    idx = np.arange(N)
    rng.shuffle(idx)

    train_end = int(0.5 * N)
    val_end = int(0.75 * N)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    # Minimal crash-prevention change: cap evaluation size.
    # This keeps randomization from rng.shuffle(idx), but avoids evaluating
    # tens/hundreds of thousands of wavelet transforms.
    if max_val_samples is not None and max_val_samples > 0:
        val_idx = val_idx[:min(max_val_samples, len(val_idx))]
    if max_test_samples is not None and max_test_samples > 0:
        test_idx = test_idx[:min(max_test_samples, len(test_idx))]

    if len(val_idx) == 0:
        raise ValueError("Validation split is empty after applying max_val_samples")
    if len(test_idx) == 0:
        raise ValueError("Test split is empty after applying max_test_samples")

    if param_grid is None:
        param_grid = GRID

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_params = combos[0]
    best_val_mse = np.inf

    T = noisy.shape[1]
    denoiser = WaveletDenoising()

    print(f"  Dataset samples: N={N:,}")
    print(f"  Split sizes: train={len(train_idx):,}, val_eval={len(val_idx):,}, test_eval={len(test_idx):,}")
    print(f"  Grid search: {len(combos)} combos x {len(val_idx)} val samples "
          f"= {len(combos) * len(val_idx):,} evaluations")

    for combo in combos:
        params = dict(zip(keys, combo))

        max_level = pywt.dwt_max_level(T, pywt.Wavelet(params["wavelet"]).dec_len)
        lvl = params["level"]
        if lvl is None or (isinstance(lvl, int) and lvl > max_level):
            params["level"] = max_level

        denoiser.set_params(**params)

        total_mse = 0.0
        count = 0

        # Important: use indices directly to avoid full array copies like noisy[val_idx].
        for i in val_idx:
            x_den = denoiser.denoise(noisy[i])
            total_mse += MeanSquaredError.calculate(clean[i], x_den)
            count += 1

        val_mse = float(total_mse / count)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params

    denoiser.set_params(**best_params)

    total_test_mse = 0.0
    test_count = 0

    # Important: use indices directly to avoid full array copies like noisy[test_idx].
    for i in test_idx:
        x_den = denoiser.denoise(noisy[i])
        total_test_mse += MeanSquaredError.calculate(clean[i], x_den)
        test_count += 1

    test_mse = float(total_test_mse / test_count)

    print("Best params:", best_params)
    print(f"Best VAL MSE:  {best_val_mse:.6f}")
    print(f"Final TEST MSE:{test_mse:.6f}")

    elapsed = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"TRAINING FINISHED: Wavelet (Grid Search)")
    print(f"   Total Time: {elapsed // 60:.0f}m {elapsed % 60:.1f}s")
    print(f"   Best Val MSE:  {best_val_mse:.6f}")
    print(f"   Final Test MSE: {test_mse:.6f}")
    print("=" * 60 + "\n")

    return best_params, best_val_mse, test_mse


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    p = argparse.ArgumentParser(description="Wavelet grid search for signal denoising")
    p.add_argument("--dataset",    required=True,
                   help="Path to dataset folder (e.g. data_generation/datasets/<name>)")
    p.add_argument("--noise-type", default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--plot",       action="store_true",
                   help="Plot an example denoising result")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", ""))

    # Minimal new controls for CPU-safe execution.
    p.add_argument("--max-val-samples", type=int, default=2000,
                   help="Maximum validation samples to evaluate during wavelet grid search")
    p.add_argument("--max-test-samples", type=int, default=2000,
                   help="Maximum test samples to evaluate after selecting best params")

    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    with open(dataset_path / "dataset_config.json") as f:
        cfg = json.load(f)

    print(f"Dataset: {dataset_path.name}")
    print(f"Config:  block_size={cfg['block_size']}, noise_type={args.noise_type}")

    noisy = np.load(dataset_path / "train" / f"{args.noise_type}_signals.npy")
    clean = np.load(dataset_path / "train" / "clean_signals.npy")
    assert noisy.shape == clean.shape

    best_params, val_mse, test_mse = grid_search_wavelet(
        noisy,
        clean,
        random_state=args.seed,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    run_dir = dataset_path / "weights" / "runs" / f"Wavelet_{args.noise_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "best_params.json"
    with open(save_path, "w") as f:
        json.dump({"best_params": best_params, "val_mse": val_mse, "test_mse": test_mse}, f, indent=2)
    print(f"Best params saved to: {save_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        denoiser = WaveletDenoising().set_params(**best_params)
        x_noisy, x_clean = noisy[0], clean[0]
        x_den = denoiser.denoise(x_noisy)
        t = np.arange(len(x_noisy))
        plt.figure(figsize=(12, 5))
        plt.plot(t, x_clean, label="Clean", linewidth=2, color="black")
        plt.plot(t, x_noisy, label="Noisy", alpha=0.5, color="gray")
        plt.plot(t, x_den, label="Wavelet (tuned)", linestyle="--", linewidth=2)
        plt.title("Wavelet Denoising with tuned hyperparams (MSE)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
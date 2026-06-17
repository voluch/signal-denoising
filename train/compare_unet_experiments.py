#!/usr/bin/env python3
"""
Iteration-2 U-Net experiment evaluator.

Reads experiment_config.json from all sub-directories of a run, performs
cross-noise evaluation using saved model weights, and produces:

  <run_dir>/
    iteration2_summary.csv
    iteration2_summary.json
    iteration2_report.md
    figures/
      iteration2_snr_bar.png
      iteration2_cross_noise_scatter.png
      iteration2_per_snr_curves.png
      iteration2_mask_diagnostics.png

Usage:
    python train/compare_unet_experiments.py --run <suite_dir>
    python train/compare_unet_experiments.py --run <suite_dir> --device cuda
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import SignalToNoiseRatio, MeanSquaredError
from models.stft_projector import STFTProjector
from models.unet_registry import load_model_from_dir
from train.snr_curve import evaluate_per_snr

# ── constants ─────────────────────────────────────────────────────────────────

# Reference baselines from iteration-1 (for Δ columns)
A01_NG_NG = 7.61
B02_NG_NG = 8.18

NOISE_TYPES = ["gaussian", "non_gaussian"]
NOISE_SHORT = {"gaussian": "G", "non_gaussian": "NG"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_exp_dirs(suite_dir: Path) -> list[Path]:
    dirs = []
    for d in sorted(suite_dir.iterdir()):
        if d.is_dir() and (d / "experiment_config.json").exists():
            dirs.append(d)
    return dirs


def _load_exp_config(exp_dir: Path) -> dict | None:
    cfg_path = exp_dir / "experiment_config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _make_denoise_fn(model, stft_proj, architecture, signal_len, device, batch_size=512):
    model.eval()
    model.to(device)

    def denoise(noisy_np: np.ndarray) -> np.ndarray:
        results = []
        for i in range(0, len(noisy_np), batch_size):
            chunk = torch.tensor(noisy_np[i:i + batch_size], dtype=torch.float32, device=device)
            with torch.no_grad():
                if architecture == "waveunet1d":
                    out = model(chunk.unsqueeze(1))
                    out_wave = chunk - out["pred_noise"].squeeze(1)
                else:
                    spec = stft_proj.stft(chunk)
                    mag = spec.abs().unsqueeze(1)
                    if architecture in ("complex_crm_unet", "complex_stft_unet"):
                        x_in = torch.stack([spec.real, spec.imag, mag.squeeze(1)], dim=1)
                    else:
                        x_in = mag
                    out = model(x_in, noisy_mag=mag, noisy_spec=spec)
                    if "out_wave" not in out:
                        if "out_spec" in out:
                            os = out["out_spec"]
                            out_wave = stft_proj.istft(os, signal_len)
                        else:
                            phase = spec / (spec.abs() + 1e-8)
                            out_wave = stft_proj.istft(out["out_mag"].squeeze(1) * phase, signal_len)
                    else:
                        out_wave = out["out_wave"]
            results.append(out_wave.cpu().numpy())
        return np.concatenate(results)

    return denoise


def _cross_eval_snr(denoise_fn, dataset_dir: Path, test_noise_type: str) -> float | None:
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        return None
    noisy_path = test_dir / f"{test_noise_type}_signals.npy"
    clean_path = test_dir / "clean_signals.npy"
    if not (noisy_path.exists() and clean_path.exists()):
        return None
    noisy = np.load(noisy_path)
    clean = np.load(clean_path)
    # Limit to 10k samples for speed
    n = min(len(noisy), 10000)
    noisy, clean = noisy[:n], clean[:n]
    pred = denoise_fn(noisy)
    return float(SignalToNoiseRatio.calculate(clean, pred))


# ── main evaluation ───────────────────────────────────────────────────────────

def run_iteration2_report(suite_dir: Path, device: str = "cpu", batch_size: int = 512):
    suite_dir = Path(suite_dir)
    exp_dirs = _find_exp_dirs(suite_dir)
    if not exp_dirs:
        print(f"No experiment directories found in {suite_dir}")
        return

    print(f"Found {len(exp_dirs)} experiment directories.")

    # Try to find dataset dir from first experiment config
    dataset_dir: Path | None = None
    for d in exp_dirs:
        cfg = _load_exp_config(d)
        if cfg:
            ds_name = cfg.get("dataset")
            if ds_name:
                candidate = ROOT / "data_generation" / "datasets" / ds_name
                if candidate.exists():
                    dataset_dir = candidate
                    break

    rows = []

    for exp_dir in exp_dirs:
        cfg = _load_exp_config(exp_dir)
        if not cfg:
            continue

        exp_id = cfg.get("exp_id", exp_dir.name)
        train_noise = cfg.get("noise_type", "unknown")
        arch = cfg.get("architecture", "spectral_unet")
        print(f"  Evaluating: {exp_id} ({train_noise})")

        # Load model if available
        model = None
        stft_proj = None
        denoise_fn = None
        try:
            model, model_config = load_model_from_dir(exp_dir, device=device)
            nperseg = model_config.get("nperseg", 128)
            hop = model_config.get("hop_length", 32)
            sig_len = model_config.get("signal_len", 1024)
            stft_proj = STFTProjector(nperseg, hop, sig_len)
            denoise_fn = _make_denoise_fn(model, stft_proj, arch, sig_len, device, batch_size)
        except Exception as e:
            print(f"    Could not load model: {e}")

        # Cross-noise SNR evaluation
        g_snr = cfg.get("test_snr") if train_noise == "gaussian" else None
        ng_snr = cfg.get("test_snr") if train_noise == "non_gaussian" else None

        if denoise_fn is not None and dataset_dir is not None:
            for test_nt in NOISE_TYPES:
                snr = _cross_eval_snr(denoise_fn, dataset_dir, test_nt)
                if snr is not None:
                    if test_nt == "gaussian":
                        g_snr = snr
                    else:
                        ng_snr = snr

        row = {
            "exp_id": exp_id,
            "architecture": arch,
            "train_noise": train_noise,
            "loss_profile": cfg.get("loss_profile", ""),
            "pooling_mode": cfg.get("pooling_mode", ""),
            "output_mode": cfg.get("output_mode", ""),
            "snr_G": round(g_snr, 3) if g_snr is not None else None,
            "snr_NG": round(ng_snr, 3) if ng_snr is not None else None,
            "val_snr": round(cfg.get("val_snr", 0), 3),
            "test_snr_same": round(cfg.get("test_snr", 0), 3),
            "epoch_best": cfg.get("epochs_run", 0),
            "training_time_s": cfg.get("training_time_s", 0),
            "param_count": cfg.get("param_count", 0),
            "mask_diag": cfg.get("mask_diagnostics", {}),
        }

        if g_snr is not None and ng_snr is not None:
            row["mean_snr"] = round((g_snr + ng_snr) / 2, 3)
            row["worst_snr"] = round(min(g_snr, ng_snr), 3)
            row["delta_vs_b02_ng"] = round(ng_snr - B02_NG_NG, 3)
            row["delta_vs_a01_ng"] = round(ng_snr - A01_NG_NG, 3)
        else:
            row["mean_snr"] = None
            row["worst_snr"] = None
            row["delta_vs_b02_ng"] = None
            row["delta_vs_a01_ng"] = None

        rows.append(row)

    # Sort by NG test SNR (primary ranking)
    ng_rows = [r for r in rows if r["train_noise"] == "non_gaussian" and r["snr_NG"] is not None]
    ng_rows.sort(key=lambda r: r["snr_NG"], reverse=True)

    _write_csv(suite_dir, rows)
    _write_json(suite_dir, rows)
    _write_markdown(suite_dir, ng_rows, rows)
    _make_figures(suite_dir, ng_rows)

    print(f"\nOutputs written to {suite_dir}")
    if ng_rows:
        best = ng_rows[0]
        print(f"Best NG→NG: {best['exp_id']} = {best['snr_NG']} dB "
              f"(Δ vs B02: {best['delta_vs_b02_ng']:+.2f} dB)")


# ── output writers ────────────────────────────────────────────────────────────

def _write_csv(suite_dir: Path, rows: list):
    path = suite_dir / "iteration2_summary.csv"
    if not rows:
        return
    fields = [
        "exp_id", "architecture", "train_noise", "loss_profile", "pooling_mode",
        "output_mode", "snr_G", "snr_NG", "mean_snr", "worst_snr",
        "delta_vs_b02_ng", "delta_vs_a01_ng",
        "val_snr", "test_snr_same", "epoch_best", "training_time_s", "param_count",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.name}")


def _write_json(suite_dir: Path, rows: list):
    path = suite_dir / "iteration2_summary.json"
    with open(path, "w") as f:
        json.dump({"generated": datetime.now().isoformat(), "experiments": rows},
                  f, indent=2, default=str)
    print(f"  → {path.name}")


def _write_markdown(suite_dir: Path, ng_rows: list, all_rows: list):
    path = suite_dir / "iteration2_report.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def _fmt(v, fmt=".2f"):
        return f"{v:{fmt}}" if v is not None else "—"

    lines = [
        f"# Iteration-2 U-Net Experiment Report",
        f"",
        f"Generated: {ts}",
        f"",
        f"Reference baselines: A01 NG→NG = {A01_NG_NG} dB | B02 NG→NG = {B02_NG_NG} dB",
        f"",
        f"---",
        f"",
        f"## Ranking (Non-Gaussian training, sorted by NG→NG SNR)",
        f"",
        f"| Rank | Exp ID | Architecture | Test G | Test NG | Mean | Worst | Δ vs B02 NG | Δ vs A01 NG | Epoch | Params |",
        f"|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for i, r in enumerate(ng_rows, 1):
        lines.append(
            f"| {i} | `{r['exp_id']}` | {r['architecture']} "
            f"| {_fmt(r['snr_G'])} | {_fmt(r['snr_NG'])} "
            f"| {_fmt(r['mean_snr'])} | {_fmt(r['worst_snr'])} "
            f"| {_fmt(r['delta_vs_b02_ng'], '+.2f')} | {_fmt(r['delta_vs_a01_ng'], '+.2f')} "
            f"| {r['epoch_best']} | {r['param_count']:,} |"
        )

    # Mask diagnostics section
    diag_rows = [r for r in ng_rows if r.get("mask_diag")]
    if diag_rows:
        lines += [
            f"",
            f"---",
            f"",
            f"## Mask Diagnostics (first validation batch)",
            f"",
            f"| Exp ID | mask_mean | mask_p99 | frac_gt1 | frac_gt2 | frac_gt3 | peak_wave |",
            f"|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in diag_rows[:10]:
            d = r["mask_diag"]
            lines.append(
                f"| `{r['exp_id']}` "
                f"| {_fmt(d.get('mask_mean'), '.3f')} "
                f"| {_fmt(d.get('mask_p99'), '.3f')} "
                f"| {_fmt(d.get('frac_mask_gt1'), '.3f')} "
                f"| {_fmt(d.get('frac_mask_gt2'), '.3f')} "
                f"| {_fmt(d.get('frac_mask_gt3'), '.3f')} "
                f"| {_fmt(d.get('waveform_peak_abs'), '.3f')} |"
            )

    # Decision thresholds
    lines += [
        f"",
        f"---",
        f"",
        f"## Decision Thresholds",
        f"",
        f"| Rule | Threshold |",
        f"|---|---|",
        f"| Promising vs A01 | NG→NG ≥ {A01_NG_NG + 0.3:.2f} dB and NG→G ≥ {A01_NG_NG - 0.2:.2f} dB |",
        f"| Promising vs B02 | NG→NG ≥ {B02_NG_NG + 0.2:.2f} dB and NG→G ≥ {B02_NG_NG - 0.1:.2f} dB |",
        f"| Major improvement | NG→NG ≥ {B02_NG_NG + 0.7:.2f} dB |",
        f"| Repeatability required | any candidate above {B02_NG_NG + 0.2:.2f} dB |",
        f"",
        f"---",
        f"*All SNR values in dB. Δ = improvement over reference baseline.*",
    ]

    path.write_text("\n".join(lines))
    print(f"  → {path.name}")


def _make_figures(suite_dir: Path, ng_rows: list):
    if not ng_rows:
        return
    fig_dir = suite_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    _fig_snr_bar(fig_dir, ng_rows)
    _fig_cross_noise_scatter(fig_dir, ng_rows)
    _fig_mask_diagnostics(fig_dir, ng_rows)


def _fig_snr_bar(fig_dir: Path, rows: list):
    ids = [r["exp_id"] for r in rows[:15]]
    g_vals = [r["snr_G"] or 0 for r in rows[:15]]
    ng_vals = [r["snr_NG"] or 0 for r in rows[:15]]

    x = range(len(ids))
    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.7), 5))
    ax.bar([i - 0.2 for i in x], g_vals, 0.35, label="Test Gaussian", color="#D65F5F", alpha=0.8)
    ax.bar([i + 0.2 for i in x], ng_vals, 0.35, label="Test Non-Gaussian", color="#4878CF", alpha=0.8)
    ax.axhline(B02_NG_NG, color="gray", linestyle="--", linewidth=1, label="B02 baseline")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Iteration-2 SNR Bar Chart (NG-trained, ranked by NG→NG)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "iteration2_snr_bar.png", dpi=150)
    plt.close(fig)


def _fig_cross_noise_scatter(fig_dir: Path, rows: list):
    g_vals = [r["snr_G"] for r in rows if r["snr_G"] is not None and r["snr_NG"] is not None]
    ng_vals = [r["snr_NG"] for r in rows if r["snr_G"] is not None and r["snr_NG"] is not None]
    labels = [r["exp_id"] for r in rows if r["snr_G"] is not None and r["snr_NG"] is not None]
    if not g_vals:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(g_vals, ng_vals, zorder=3)
    for g, n, l in zip(g_vals, ng_vals, labels):
        ax.annotate(l, (g, n), fontsize=6, textcoords="offset points", xytext=(3, 3))
    # reference point
    ax.scatter([B02_NG_NG], [B02_NG_NG], marker="*", s=200, color="gold", zorder=4, label="B02 reference")
    ax.set_xlabel("NG-trained → G-test SNR (dB)")
    ax.set_ylabel("NG-trained → NG-test SNR (dB)")
    ax.set_title("Cross-Noise Generalisation Scatter")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "iteration2_cross_noise_scatter.png", dpi=150)
    plt.close(fig)


def _fig_mask_diagnostics(fig_dir: Path, rows: list):
    diag_rows = [r for r in rows if r.get("mask_diag") and r["mask_diag"].get("mask_mean") is not None]
    if not diag_rows:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No mask diagnostics available", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(fig_dir / "iteration2_mask_diagnostics.png", dpi=150)
        plt.close(fig)
        # Also create placeholder per-SNR curves figure
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No per-SNR data available\n(need evaluate_per_snr data)", ha="center",
                va="center", transform=ax.transAxes)
        fig.savefig(fig_dir / "iteration2_per_snr_curves.png", dpi=150)
        plt.close(fig)
        return

    ids = [r["exp_id"] for r in diag_rows[:10]]
    means = [r["mask_diag"]["mask_mean"] for r in diag_rows[:10]]
    p99s = [r["mask_diag"].get("mask_p99", 0) for r in diag_rows[:10]]
    frac_gt1 = [r["mask_diag"].get("frac_mask_gt1", 0) for r in diag_rows[:10]]

    x = range(len(ids))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(x, means); axes[0].set_title("Mask Mean"); axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    axes[1].bar(x, p99s); axes[1].set_title("Mask P99"); axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    axes[2].bar(x, frac_gt1); axes[2].set_title("Fraction Mask > 1"); axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    plt.suptitle("Mask Diagnostics (NG-trained, val batch)")
    plt.tight_layout()
    fig.savefig(fig_dir / "iteration2_mask_diagnostics.png", dpi=150)
    plt.close(fig)

    # placeholder per-SNR curves (requires per-SNR data from run)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "Per-SNR curves require\nrun_unet_experiment_suite per-SNR data",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("Iteration-2 Per-SNR Curves (placeholder)")
    fig.savefig(fig_dir / "iteration2_per_snr_curves.png", dpi=150)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iteration-2 U-Net experiment evaluator")
    parser.add_argument("--run", required=True, help="Path to suite run directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run_iteration2_report(Path(args.run), device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
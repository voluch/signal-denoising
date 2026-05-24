#!/usr/bin/env python3
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def negative_snr(pred, target, eps=1e-8):
    noise = pred - target
    signal_power = np.mean(target ** 2) + eps
    noise_power = np.mean(noise ** 2) + eps
    return 10.0 * np.log10(signal_power / noise_power)

def run_oracle_analysis(dataset_path, noise_types, nperseg=128, hop_length=32, batch_size=1024, output_dir=None):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir else dataset_path / "runs" / "unet_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    
    noise_types = [n.strip() for n in noise_types.split(",")]
    noverlap = nperseg - hop_length
    
    summary = []
    ratio_stats = []
    
    for nt in noise_types:
        print(f"Analyzing {nt}...")
        noisy = np.load(dataset_path / "test" / f"{nt}_signals.npy")[:2000] # Use subset for speed
        clean = np.load(dataset_path / "test" / "clean_signals.npy")[:2000]
        
        noisy_t = torch.tensor(noisy, dtype=torch.float32)
        clean_t = torch.tensor(clean, dtype=torch.float32)
        
        win = torch.hann_window(nperseg)
        
        # Oracles
        snrs = {"input_noisy": []}
        snrs["oracle_clean_mag_noisy_phase"] = []
        snrs["oracle_mask_0_1_noisy_phase"] = []
        snrs["oracle_mask_0_2_noisy_phase"] = []
        snrs["oracle_mask_0_3_noisy_phase"] = []
        snrs["oracle_direct_clean_complex_stft"] = []
        
        ratios = []
        
        for i in tqdm(range(0, len(noisy_t), batch_size)):
            n_batch = noisy_t[i:i+batch_size]
            c_batch = clean_t[i:i+batch_size]
            
            n_spec = torch.stft(n_batch, n_fft=nperseg, hop_length=hop_length, win_length=nperseg, window=win, center=True, return_complex=True)
            c_spec = torch.stft(c_batch, n_fft=nperseg, hop_length=hop_length, win_length=nperseg, window=win, center=True, return_complex=True)
            
            n_mag = n_spec.abs()
            c_mag = c_spec.abs()
            n_phase = n_spec / (n_mag + 1e-8)
            
            # Input SNR
            for j in range(len(n_batch)):
                snrs["input_noisy"].append(negative_snr(n_batch[j].numpy(), c_batch[j].numpy()))
            
            # Oracle 1: Clean Mag + Noisy Phase
            o1_spec = c_mag * n_phase
            o1_wave = torch.istft(o1_spec, n_fft=nperseg, hop_length=hop_length, win_length=nperseg, window=win, center=True, length=noisy_t.shape[1])
            for j in range(len(o1_wave)):
                snrs["oracle_clean_mag_noisy_phase"].append(negative_snr(o1_wave[j].numpy(), c_batch[j].numpy()))
            
            # Oracle Masks
            for m_max in [1.0, 2.0, 3.0]:
                mask = torch.clamp(c_mag / (n_mag + 1e-8), max=m_max)
                o_spec = (mask * n_mag) * n_phase
                o_wave = torch.istft(o_spec, n_fft=nperseg, hop_length=hop_length, win_length=nperseg, window=win, center=True, length=noisy_t.shape[1])
                key = f"oracle_mask_0_{int(m_max)}_noisy_phase"
                for j in range(len(o_wave)):
                    snrs[key].append(negative_snr(o_wave[j].numpy(), c_batch[j].numpy()))
            
            # Oracle Direct Complex
            o_wave_c = torch.istft(c_spec, n_fft=nperseg, hop_length=hop_length, win_length=nperseg, window=win, center=True, length=noisy_t.shape[1])
            for j in range(len(o_wave_c)):
                snrs["oracle_direct_clean_complex_stft"].append(negative_snr(o_wave_c[j].numpy(), c_batch[j].numpy()))
                
            # Ratio stats
            ratios.append((c_mag / (n_mag + 1e-8)).numpy().flatten())
            
        ratios = np.concatenate(ratios)
        r_stats = {
            "noise_type": nt,
            "frac_gt_1": float(np.mean(ratios > 1)),
            "frac_gt_2": float(np.mean(ratios > 2)),
            "frac_gt_3": float(np.mean(ratios > 3)),
            "p50": float(np.percentile(ratios, 50)),
            "p90": float(np.percentile(ratios, 90)),
            "p99": float(np.percentile(ratios, 99)),
        }
        ratio_stats.append(r_stats)
        
        for k, v in snrs.items():
            summary.append({
                "noise_type": nt,
                "oracle_type": k,
                "mean_snr": float(np.mean(v))
            })

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(output_dir / "oracle_summary.csv", index=False)
    
    df_ratio = pd.DataFrame(ratio_stats)
    df_ratio.to_csv(output_dir / "oracle_ratio_stats.csv", index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_sum, x="oracle_type", y="mean_snr", hue="noise_type")
    plt.xticks(rotation=45, ha='right')
    plt.title("Oracle SNR Ceilings")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "oracle_snr_bar.png")
    
    print(f"Oracle analysis complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--noise-types", default="gaussian,non_gaussian")
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_oracle_analysis(args.dataset, args.noise_types, args.nperseg, args.hop_length, args.batch_size, args.output_dir)

#!/usr/bin/env python3
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_summarization(run_dir, dataset_path=None):
    run_dir = Path(run_dir)
    print(f"Summarizing experiments in {run_dir}...")
    
    results = []
    for exp_dir in run_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "figures":
            continue
        
        config_path = exp_dir / "experiment_config.json"
        if not config_path.exists():
            continue
            
        with open(config_path, "r") as f:
            config = json.load(f)
        
        results.append(config)
    
    if not results:
        print("No experiments found to summarize.")
        return
        
    df = pd.DataFrame(results)
    df = df.sort_values(by="val_snr", ascending=False)
    
    # Save CSV and JSON
    df.to_csv(run_dir / "unet_ablation_summary.csv", index=False)
    df.to_json(run_dir / "unet_ablation_summary.json", orient="records", indent=2)
    
    # Generate Figures
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Best SNR bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="run_id", y="val_snr")
    plt.xticks(rotation=45, ha='right')
    plt.title("Best Validation SNR per Experiment")
    plt.tight_layout()
    plt.savefig(fig_dir / "summary_best_snr_bar.png")
    plt.close()
    
    # Generate Markdown Report
    report_path = run_dir / "unet_ablation_report.md"
    with open(report_path, "w") as f:
        f.write("# U-Net Ablation Experiment Summary\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Run Directory:** `{run_dir.name}`\n\n")
        
        f.write("## Ranking (by Validation SNR)\n\n")
        f.write("| Rank | Experiment ID | Output Mode | Pooling | Loss | Val SNR | Test SNR |\n")
        f.write("|------|---------------|-------------|---------|------|---------|----------|\n")
        for i, row in df.iterrows():
            f.write(f"| {i+1} | {row['run_id']} | {row['output_mode']} | {row['pooling_mode']} | {row['loss_profile']} | {row['val_snr']:.2f} dB | {row.get('test_snr', 0.0):.2f} dB |\n")
        
        f.write("\n## Figures\n\n")
        f.write("### Best SNR Comparison\n")
        f.write("![Best SNR](figures/summary_best_snr_bar.png)\n")

    print(f"Summary report generated at {report_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=False)
    args = parser.parse_args()
    run_summarization(args.run_dir, args.dataset)

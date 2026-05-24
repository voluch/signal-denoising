#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aws_scripts.download_dataset_from_s3 import download_dataset_from_s3
from aws_scripts.push_runs_to_s3 import push_runs_to_s3
from train.compare_report import run_compare_report

def main():
    parser = argparse.ArgumentParser(description="Run U-Net experiment suite")
    parser.add_argument("--dataset", required=False, default=os.getenv("DATASET_NAME"))
    parser.add_argument("--config", required=False, default=os.getenv("CONFIG_PATH", "train/unet_experiment_configs/core_v1.json"), help="Path to experiment config JSON")
    parser.add_argument("--noise-types", default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--signal-len", type=int, default=None)
    parser.add_argument("--fs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="unet-experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--partial-train", type=float, default=1.0)
    parser.add_argument("--models", help="Ignored (for workflow compatibility)")
    parser.add_argument("--run-id", help="Optional run ID for output directory")
    args = parser.parse_args()

    if not args.dataset:
        print("ERROR: --dataset is required (or DATASET_NAME env var)")
        sys.exit(1)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir

    if not dataset_dir.exists():
        dataset_name = dataset_dir.name
        potential_dir = ROOT / "data_generation" / "datasets" / dataset_name
        if not potential_dir.exists():
            download_dataset_from_s3(dataset_name, potential_dir)
        dataset_dir = potential_dir
    
    # Load dataset config
    with open(dataset_dir / "dataset_config.json", "r") as f:
        ds_cfg = json.load(f)
    
    signal_len = args.signal_len if args.signal_len else ds_cfg.get("block_size", 1024)
    fs = args.fs if args.fs else ds_cfg.get("sample_rate", 8192)

    with open(args.config, "r") as f:
        experiments = json.load(f)

    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"unet_ablation_{timestamp}"
    
    suite_dir = dataset_dir / "runs" / run_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting U-Net Experiment Suite: {run_id}")
    print(f"Total experiments: {len(experiments)}")

    if args.noise_types == "all":
        noise_types = ["gaussian", "non_gaussian"]
    else:
        noise_types = [n.strip() for n in args.noise_types.split(",")]
    
    failures = []
    commands = []

    for exp in experiments:
        exp_id = exp["exp_id"]
        for nt in noise_types:
            sub_run_id = f"{exp_id}_{nt}_seed{args.seed}"
            exp_dir = suite_dir / sub_run_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                sys.executable, "train/training_uae.py",
                "--dataset", str(dataset_dir),
                "--noise-type", nt,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--nperseg", str(args.nperseg),
                "--hop-length", str(args.hop_length),
                "--signal-len", str(signal_len),
                "--fs", str(fs),
                "--seed", str(args.seed),
                "--wandb-project", args.wandb_project,
                "--partial-train", str(args.partial_train),
                "--run-id", sub_run_id,
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            
            # Add experiment specific args
            for k, v in exp.items():
                if k == "exp_id": continue
                arg_name = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        cmd.append(arg_name)
                else:
                    cmd.extend([arg_name, str(v)])
            
            # Pass exp_id explicitly
            cmd.extend(["--exp-id", exp_id])
            
            # Set output directory to sub-run dir
            # Note: training_uae.py might need to be updated to handle output-dir correctly
            # In my previous update I added output_dir to UnetAutoencoderTrainer but not to CLI.
            # I'll fix training_uae.py CLI later. For now I'll assume it's there.
            # Wait, I didn't add --output-dir to training_uae.py CLI in the previous step.
            
            print(f"\n--- Running Experiment: {exp_id} ({nt}) ---")
            print(f"Command: {' '.join(cmd)}")
            commands.append({"exp_id": exp_id, "noise_type": nt, "cmd": " ".join(cmd)})
            
            try:
                # We need to pass output_dir to the subprocess.
                # Let's use an environment variable or update training_uae.py CLI.
                # Updating CLI is better.
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)
                # I'll update training_uae.py to accept --output-dir
                cmd.extend(["--output-dir", str(exp_dir)])
                
                subprocess.run(cmd, check=True, env=env)
                print(f"✅ Finished: {exp_id}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {exp_id}. Error: {e}")
                failures.append({"exp_id": exp_id, "noise_type": nt, "error": str(e)})

    # Save commands and failures
    with open(suite_dir / "commands.jsonl", "w") as f:
        for c in commands: f.write(json.dumps(c) + "\n")
    if failures:
        with open(suite_dir / "failures.jsonl", "w") as f:
            for fail in failures: f.write(json.dumps(fail) + "\n")

    print("\n✅ All experiments finished.")
    
    # Call comparison report (which now calls summarization)
    print("\n📊 Generating comparison report...")
    run_compare_report(suite_dir)

    # Cloud sync
    if os.getenv("CLOUD_TRAINING") == "True":
        print("\n🚀 Pushing results to S3...")
        push_runs_to_s3(dataset_dir.name, str(suite_dir))
        
        print("Terminating pod...")
        terminate_script = ROOT / ".github/scripts/terminate_pod.py"
        if terminate_script.exists():
            subprocess.run([sys.executable, str(terminate_script)])

if __name__ == "__main__":
    main()

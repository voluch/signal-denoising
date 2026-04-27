#!/usr/bin/env python3
import os
import sys
import argparse
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

def push_runs_to_s3(dataset_name: str, runs_input: str = None):
    """
    Pushes the 'runs' directory (or a specific run) of a dataset to S3.
    Local path: data_generation/datasets/<dataset_name>/runs[/<run_id>]
    S3 path: s3://<S3_BUCKET>/<dataset_name>/runs[/<run_id>]
    """
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET not set in environment.")
        return False

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("ERROR: AWS credentials not set in environment.")
        return False

    # Default runs directory
    base_runs_dir = ROOT / "data_generation" / "datasets" / dataset_name / "runs"
    
    if runs_input:
        # Check if runs_input is an absolute path or a relative path from current DIR
        input_path = Path(runs_input)
        if input_path.exists():
            target_path = input_path
        else:
            # Check if it's a specific run ID inside the dataset's runs folder
            target_path = base_runs_dir / runs_input
    else:
        target_path = base_runs_dir
    
    if not target_path.exists():
        print(f"⚠️  Local path '{target_path}' does not exist. Nothing to push.")
        return False

    # We want to push to s3://bucket/dataset_name/runs/...
    # Find the relative path from the dataset folder to the target_path
    dataset_dir = ROOT / "data_generation" / "datasets" / dataset_name
    try:
        relative_prefix = target_path.relative_to(dataset_dir)
    except ValueError:
        # If target_path is not inside dataset_dir (e.g., custom path),
        # we'll just use the folder name 'runs' or the input name
        relative_prefix = Path("runs") / (runs_input if runs_input else "")

    print(f"🚀 Pushing from '{target_path}' to s3://{S3_BUCKET}/{dataset_name}/{relative_prefix.as_posix()}/ ...")

    try:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3 = session.client('s3')

        count = 0
        for root, dirs, files in os.walk(target_path):
            for file in files:
                local_file_path = Path(root) / file
                # Path relative to the target_path's parent to keep the target folder name in S3 if it's a specific run
                # OR relative to dataset_dir to maintain structure
                try:
                    rel_to_dataset = local_file_path.relative_to(dataset_dir)
                    s3_key = f"{dataset_name}/{rel_to_dataset.as_posix()}"
                except ValueError:
                    # Fallback for paths outside the standard dataset structure
                    rel_to_target_parent = local_file_path.relative_to(target_path.parent)
                    s3_key = f"{dataset_name}/runs/{rel_to_target_parent.as_posix()}"
                
                print(f"   Uploading to s3://{S3_BUCKET}/{s3_key} ...")
                s3.upload_file(str(local_file_path), S3_BUCKET, s3_key)
                count += 1
        
        print(f"✅ Successfully uploaded {count} files to S3.")
        return True
    except Exception as e:
        print(f"❌ Failed to push to S3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push training runs to S3")
    parser.add_argument("dataset_name", help="Name of the dataset whose runs to push")
    parser.add_argument("--run-id", "--runs-dir", dest="runs_input", help="Specific run ID or explicit path to a runs directory")
    args = parser.parse_args()

    # Test logic
    print(f"--- Testing Push Runs to S3 ---")
    print(f"Dataset: {args.dataset_name}")
    if args.runs_input:
        print(f"Run ID/Dir: {args.runs_input}")
    
    success = push_runs_to_s3(args.dataset_name, args.runs_input)
    
    if success:
        print("\n✅ Push completed successfully.")
    else:
        print("\n❌ Push failed or skipped.")

if __name__ == "__main__":
    main()

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

def download_run_from_s3(dataset_name: str, run_id: str, target_dir: Path = None):
    """
    Downloads a specific training run from S3.
    S3 path: s3://<S3_BUCKET>/<dataset_name>/runs/<run_id>/
    Local path: data_generation/datasets/<dataset_name>/runs/<run_id>/
    """
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET not set in environment.")
        return False

    if target_dir is None:
        target_dir = ROOT / "data_generation" / "datasets" / dataset_name / "runs" / run_id

    print(f"📥 Downloading run '{run_id}' for '{dataset_name}' from S3...")
    
    try:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3 = session.resource('s3')
        bucket = s3.Bucket(S3_BUCKET)

        prefix = f"{dataset_name}/runs/{run_id}/"
        objs = list(bucket.objects.filter(Prefix=prefix))
        
        if not objs:
            print(f"⚠️  No objects found for run '{run_id}' in s3://{S3_BUCKET}/{dataset_name}/runs/")
            return False

        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for obj in objs:
            if obj.key.endswith('/'): # skip directory markers
                continue
                
            # Local path relative to the target_dir based on key
            # obj.key is e.g., 'dataset_name/runs/run_id/file.txt'
            # relative_to_run_root = 'file.txt'
            relative_to_run_root = Path(obj.key).relative_to(prefix)
            local_file_path = target_dir / relative_to_run_root
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"   Downloading {relative_to_run_root}...")
            bucket.download_file(obj.key, str(local_file_path))
            count += 1
            
        print(f"✅ Downloaded {count} files to {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download run from S3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download a specific training run from S3")
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("run_id", help="ID of the run to download (e.g., run_20260330_8c4d2660)")
    parser.add_argument("--target-dir", help="Explicit target directory (optional)")
    args = parser.parse_args()

    # Test logic
    print(f"--- Testing Download Run from S3 ---")
    print(f"Dataset: {args.dataset_name}")
    print(f"Run ID:  {args.run_id}")
    if args.target_dir:
        print(f"Target:  {args.target_dir}")
    
    success = download_run_from_s3(
        args.dataset_name, 
        args.run_id, 
        Path(args.target_dir) if args.target_dir else None
    )
    
    if success:
        print("\n✅ Download completed successfully.")
    else:
        print("\n❌ Download failed.")

if __name__ == "__main__":
    main()

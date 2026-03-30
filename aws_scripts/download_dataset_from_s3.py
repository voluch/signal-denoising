import argparse
import os
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


# ── S3 download ────────────────────────────────────────────────────────────────

def download_dataset_from_s3(dataset_name: str, target_dir: Path):
    """Downloads a dataset from S3 if it doesn't exist locally."""
    s3_bucket = os.getenv("S3_BUCKET")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not s3_bucket:
        print("ERROR: S3_BUCKET not set in environment. Cannot download dataset.")
        return False

    print(f"\n📦 Dataset '{dataset_name}' not found locally.")
    print(f"   Attempting to download from s3://{s3_bucket}/{dataset_name} ...")

    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        s3 = session.resource('s3')
        bucket = s3.Bucket(s3_bucket)

        # List objects with prefix
        objs = list(bucket.objects.filter(Prefix=dataset_name))
        if not objs:
            print(f"   ERROR: No objects found in s3://{s3_bucket}/{dataset_name}")
            return False

        target_dir.mkdir(parents=True, exist_ok=True)
        for obj in objs:
            # Create local path
            local_file_path = target_dir.parent / obj.key
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            if obj.key.endswith('/'):  # Skip directory markers
                continue

            print(f"   Downloading {obj.key} ...")
            bucket.download_file(obj.key, str(local_file_path))

        print(f"✅ Dataset downloaded to {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download from S3: {e}")
        return False


def list_files(startpath):
    startpath = Path(startpath)
    if not startpath.exists():
        print(f"Error: {startpath} does not exist.")
        return
    for root, dirs, files in os.walk(startpath):
        level = root.replace(str(startpath), '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root) if os.path.basename(root) else startpath.name}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')


def main():
    parser = argparse.ArgumentParser(description="Test S3 dataset download")
    parser.add_argument("dataset_name", help="Name of the dataset to download from S3")
    parser.add_argument("--target-dir",
                        help="Local directory to download to (default: data_generation/datasets/<dataset_name>)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    dataset_name = args.dataset_name
    if args.target_dir:
        target_dir = Path(args.target_dir)
    else:
        target_dir = ROOT / "data_generation" / "datasets" / dataset_name

    print(f"--- Testing S3 Download ---")
    print(f"Dataset name: {dataset_name}")
    print(f"Target dir:   {target_dir}")
    print(f"S3_BUCKET:    {os.getenv('S3_BUCKET')}")
    print(f"---------------------------")

    success = download_dataset_from_s3(dataset_name, target_dir)

    if success:
        print(f"\n✅ Download successful!")
        print(f"\nStructure of {target_dir}:")
        list_files(target_dir)
    else:
        print(f"\n❌ Download failed.")


if __name__ == "__main__":
    main()

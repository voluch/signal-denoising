import os
import sys
import argparse
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
# Assuming .env is in the project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Configuration
# Default values can be overridden by environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

def upload_directory(path, bucket, s3_prefix=""):
    """Recursively uploads a directory to S3."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    s3 = session.client('s3')

    path = Path(path)
    if not path.exists():
        print(f"ERROR: Local path '{path}' does not exist.")
        sys.exit(1)

    # Use the directory name as the S3 prefix if s3_prefix is not provided
    if not s3_prefix:
        s3_prefix = path.name

    print(f"Uploading directory: {path} -> s3://{bucket}/{s3_prefix}/")

    # Walk through the directory and upload each file
    for root, dirs, files in os.walk(path):
        for file in files:
            local_file_path = Path(root) / file
            # Relative path from the base directory
            relative_path = local_file_path.relative_to(path)
            # S3 key is prefix + relative path
            s3_key = f"{s3_prefix}/{relative_path.as_posix()}"
            
            print(f"  Uploading {relative_path}...")
            try:
                s3.upload_file(str(local_file_path), bucket, s3_key)
            except Exception as e:
                print(f"    ERROR: Could not upload {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload a dataset directory to AWS S3")
    parser.add_argument("dataset", help="Path to the dataset folder to upload")
    args = parser.parse_args()

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in .env file.")
        sys.exit(1)

    if not S3_BUCKET:
        print("ERROR: S3_BUCKET must be set in .env file.")
        sys.exit(1)

    upload_directory(args.dataset, S3_BUCKET, args.prefix)
    print("\n✅ Upload complete!")

if __name__ == "__main__":
    main()

# AWS S3 Utility Scripts

This directory contains utility scripts for managing datasets and training results on AWS S3. These scripts are used to synchronize data between local environments and the S3 bucket configured in your `.env` file.

## Prerequisites

Ensure you have a `.env` file in the project root with the following variables:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your-s3-bucket-name
```

## Scripts Overview

### 1. `download_dataset_from_s3.py`
Downloads a complete dataset from S3 to the local `data_generation/datasets/` directory.

**Usage:**
```bash
python aws_scripts/download_dataset_from_s3.py <dataset_name>
```

**Options:**
- `--target-dir`: Explicitly specify the local directory to download to (default: `data_generation/datasets/<dataset_name>`).

**Example:**
```bash
python aws_scripts/download_dataset_from_s3.py deep_space_polygauss_qpsk_bs1024_n3000_993897f3
```

---

### 2. `push_dataset_to_s3.py`
Uploads a local dataset directory to S3.

**Usage:**
```bash
python aws_scripts/push_dataset_to_s3.py <local_dataset_path>
```

**Example:**
```bash
python aws_scripts/push_dataset_to_s3.py data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n3000_993897f3
```

---

### 3. `download_run_from_s3.py`
Downloads a specific training run (by its run ID) from S3 for a given dataset.

**Usage:**
```bash
python aws_scripts/download_run_from_s3.py <dataset_name> <run_id>
```

**Options:**
- `--target-dir`: Explicitly specify the local directory to download to.

**Example:**
```bash
python aws_scripts/download_run_from_s3.py deep_space_polygauss_qpsk_bs1024_n3000_993897f3 run_20260330_8c4d2660
```

---

### 4. `push_runs_to_s3.py`
Uploads the `runs` directory or a specific run for a dataset to S3. This is used by `train/train_all.py` automatically when `CLOUD_TRAINING=True`.

**Usage:**
```bash
python aws_scripts/push_runs_to_s3.py <dataset_name>
```

**Options:**
- `--run-id` or `--runs-dir`: Specify a specific run ID or an explicit path to a runs directory to upload.

**Example (uploading all runs for a dataset):**
```bash
python aws_scripts/push_runs_to_s3.py deep_space_polygauss_qpsk_bs1024_n3000_993897f3
```

**Example (uploading a specific run):**
```bash
python aws_scripts/push_runs_to_s3.py deep_space_polygauss_qpsk_bs1024_n3000_993897f3 --run-id run_20260330_8c4d2660
```

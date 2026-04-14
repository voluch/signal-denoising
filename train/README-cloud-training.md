# Cloud Training Guide

This guide describes how to train signal denoising models on cloud infrastructure (RunPod) using the provided scripts and GitHub Actions.

## 1. Generate Dataset

First, generate a synthetic dataset locally or on a powerful machine.

- **Location**: `data_generation/`
- **Script**: `data_generation/generation.py`
- **Documentation**: See [data_generation/README.md](../data_generation/README.md) for detailed parameter descriptions.

### Example Commands:

```bash
# Generate a standard FPV telemetry dataset with 400,000 samples
python data_generation/generation.py --num_train 400000 --samples_per_snr 10000

# Generate a deep space scenario dataset
python data_generation/generation.py --deep_space --num_train 100000
```

The generated dataset will be stored in `data_generation/datasets/` with a unique UID (e.g., `fpv_telemetry_polygauss_qpsk_bs1024_n400000_6d07aecc`).

## 2. Upload Dataset to S3

To make the dataset accessible to cloud training pods, upload it to an S3 bucket.

- **Script**: `data_generation/push_dataset_to_s3.py`
- **Configuration**: Ensure your `.env` file contains the following keys:
  ```bash
  AWS_ACCESS_KEY_ID=your_access_key
  AWS_SECRET_ACCESS_KEY=your_secret_key
  S3_BUCKET=signal-denoising-datasets
  ```

### Example Command:

```bash
# Upload the specific dataset folder
python data_generation/push_dataset_to_s3.py data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n400000_6d07aecc
```

## 3. Build Image and Push to ECR

Whenever you make changes to the training scripts or models, you need to update the Docker image.

1.  Navigate to **GitHub Actions** in your repository.
2.  Select the **"CI - Build and Push to ECR"** workflow (or `.github/workflows/ci.yml`).
3.  Click **"Run workflow"**.
4.  This will:
    - Build the Docker image from the `Dockerfile`.
    - Push the image to Amazon ECR.
    - (Optional) Start a training run if you fill in the deployment parameters.

## 4. Manually Start Training (RunPod)

You can trigger a training run using the **"CD - Deploy to RunPod and Train"** GitHub Action. This action automates pod creation, template setup, and training initiation.

At first check available GPU machines on [runpod](https://console.runpod.io/deploy). Don't forget to filter for GPU instances in EU and North America. 
### Action Inputs & Parameters:

- **GPU Type**: Choose the desired GPU (e.g., `NVIDIA RTX A5000` or `NVIDIA GeForce RTX 4090`).
- **Region Scope**: `EU`, `US`, or `EU+US` to find available pods.
- **Dataset Path**: Provide the **name** of the dataset folder you uploaded to S3 (e.g., `fpv_telemetry_polygauss_qpsk_bs1024_n400000_6d07aecc`). The script will automatically download it.
- **Model**: Select `all` to train all models or a specific one (`resnet`, `unet`, `vae`, etc.).
- **Extra Envs**: Used to override per-model parameters like batch sizes or learning rates.
  - **Example (Batch Size Override)**: `MODEL_BATCH_SIZES='{"unet": 512, "vae": 2048}'`
  - **Example (Learning Rate Override)**: `MODEL_LEARNING_RATES='{"transformer": 5e-4}'`
- **Extra Docker Args**: Additional CLI arguments passed to `train/train_all.py`.
  - **Example**: `--epochs 100 --partial-train 0.5`

### Monitoring Pods:
After dispatching, check the **Job Summary** in GitHub Actions. It will display:
- **Run ID**: The unique ID for this training session.
- **Pod URL**: A link to monitor the pod on RunPod (if applicable).
- **Run ID**: If a pod was already running, it shows the new Run ID assigned to it.

## 5. Check Online Results (W&B)

All training metrics, loss curves, and audio samples (if logged) are sent to Weights & Biases.

- **Project**: Specified in the `wandb_project` input (default: `signal-denoising-v2`).
- **Run Names**: Each model in a session gets a run named like `{MODEL_NAME}_{noise_type}_{dataset_uid}_{run_id}`.

You can compare different models and noise types directly on the W&B dashboard.

## 6. Download Run Results

Once training is complete, the results (model weights, training summaries, and the comparison report) are automatically pushed back to S3.

- **Location in S3**: `signal-denoising-datasets/datasets/<dataset_name>/runs/<run_id>/`
- **Script to Download**: `aws_scripts/download_run_from_s3.py`

### Example Command:

```bash
# Download a specific run's results
python aws_scripts/download_run_from_s3.py --dataset fpv_telemetry_... --run-id run_20240414_abcdef12
```

This will download a directory structure like:
- `training_report_<timestamp>.md`: A Markdown summary of the results.
- `training_report_<timestamp>.json`: Raw metrics for all trained models.
- `<MODEL_NAME>_<noise_type>/`:
    - `model_best.pth`: Trained model checkpoint.
    - `figures/`:
        - `snr_curve.png`: Performance across SNR range.
        - `training_curves.png`: Loss and metric history.
    - `dsge_state.npz` / `spec_norm.json`: Model-specific metadata.

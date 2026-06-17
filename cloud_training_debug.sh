git clone https://github.com/voluch/signal-denoising.git
cd signal-denoising
git checkout unet-experiments
pip install --no-cache-dir -r requirements.txt &&     rm -rf /root/.cache/pip
pip install --no-cache-dir -r requirements-cuda.txt &&     rm -rf /root/.cache/pip
export AWS_ACCESS_KEY_ID="key"
export AWS_SECRET_ACCESS_KEY="key"
export AWS_REGION="us-east-1"
export S3_BUCKET="signal-denoising-datasets"
export DATASET_NAME="deep_space_polygauss_qpsk_bs1024_n3000_993897f3"
python3 aws_scripts/download_dataset_from_s3.py $DATASET_NAME
export RUN_ID="run_$(date +%Y%m%d_%H%M)$(openssl rand -hex 4)"
echo "$RUN_ID"

# Debug: run combo_v2 suite on a small subsample of data (2 epochs, 5% of data, non_gaussian only)
# Tests that the full pipeline works before launching a full cloud training run
python3 train/run_unet_experiment_suite.py \
    --dataset $DATASET_NAME \
    --config train/unet_experiment_configs/combo_v2.json \
    --noise-types non_gaussian \
    --epochs 2 \
    --batch-size 512 \
    --partial-train 0.5 \
    --seed 42 \
    --wandb-project unet-experiments-debug \
    --run-id $RUN_ID

python3 train/compare_report.py --run "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n3000_993897f3/runs/${RUN_ID}"

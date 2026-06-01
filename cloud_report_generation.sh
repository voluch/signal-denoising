git clone https://github.com/voluch/signal-denoising.git
cd signal-denoising
git checkout unet-experiments
ls
pip install -r requirements-cuda.txt
pip install --no-cache-dir -r requirements.txt &&     rm -rf /root/.cache/pip
pip install --no-cache-dir -r requirements-cuda.txt &&     rm -rf /root/.cache/pip
export AWS_ACCESS_KEY_ID=key
export AWS_SECRET_ACCESS_KEY=key
export AWS_REGION=us-east-1
export S3_BUCKET=signal-denoising-datasets
export DATASET_NAME=deep_space_polygauss_qpsk_bs1024_n400000_c054e749
python3 aws_scripts/download_dataset_from_s3.py $DATASET_NAME
export RUN_ID=run_20260524_1516bd43_github_59
python3 aws_scripts/download_run_from_s3.py $DATASET_NAME $RUN_ID
python3 train/compare_report.py --run "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749/runs/run_20260524_1516bd43_github_59"
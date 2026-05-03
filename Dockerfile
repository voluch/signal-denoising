# Use PyTorch runtime image with Python 3.12 and CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies and clean up apt cache in one layer to reduce size
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies and clean up pip cache
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip
RUN pip install --no-cache-dir -r requirements-cuda.txt && \
    rm -rf /root/.cache/pip

# Copy the rest of the project (uses .dockerignore to skip unnecessary files)
COPY . .

# Default command to run the training script.
# Expects a dataset directory mounted to /app/data_generation/datasets/<dataset_name>
# or similar, or provided via --dataset CLI argument.
ENTRYPOINT ["python", "train/train_all.py"]

# Default argument (optional)
CMD ["--help"]

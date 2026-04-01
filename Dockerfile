# Use PyTorch runtime image with Python 3.12 and CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (for matplotlib and other potential libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command to run the training script.
# Expects a dataset directory mounted to /app/data_generation/datasets/<dataset_name>
# or similar, or provided via --dataset CLI argument.
ENTRYPOINT ["python", "train/train_all.py"]

# Default argument (optional)
CMD ["--help"]

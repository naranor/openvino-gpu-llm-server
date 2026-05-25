#!/bin/bash

# Script to set up and run an OpenAI-compatible API server using OpenVINO on Intel GPU

set -e

echo "=== OpenVINO OpenAI API Server Setup (FP16) ==="

# 1. Source oneAPI environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "Sourcing Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
fi

# 2. Model Configuration
# We use DeepSeek-R1-Distill-Qwen-7B-int8-ov: The most powerful reasoning model that fits the shared memory limits
# Verified to bypass the 4GB max_alloc limit by chunking layers in OpenVINO
MODEL_REPO="OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int8-ov"
MODEL_DIR="models/DeepSeek-R1-Distill-Qwen-7B-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek-R1 7B reasoning model (INT8 OpenVINO)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
else
    echo "Model already exists in $MODEL_DIR"
fi

# 3. Run the server
PORT=8000
DEVICE="GPU"

echo "Starting OpenVINO server on port $PORT using $DEVICE..."
echo "Model: $MODEL_DIR"

# Set stability flags for Intel Gen 9.5
export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OPENVINO_LOG_LEVEL=0
export OV_GPU_WAIT_TYPE=SLEEP

# Run the python server
python3 ov_server.py --model "$MODEL_DIR" --device "$DEVICE" --port "$PORT" --n_ctx 32768

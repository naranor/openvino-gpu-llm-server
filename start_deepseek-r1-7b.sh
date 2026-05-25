#!/bin/bash

# Script to set up and run OpenVINO GPU API Server
set -e

echo "=== OpenVINO GPU API Server: DeepSeek-R1-Qwen-7B (Reasoning) ==="

# 1. Environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# 2. Model Configuration
MODEL_REPO="OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int8-ov"
MODEL_DIR="models/DeepSeek-R1-Distill-Qwen-7B-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek-R1-Qwen-7B (INT8 OpenVINO)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
fi

# 3. Run Server
PORT=8000
DEVICE="GPU"

echo "Starting on $DEVICE. Model: $MODEL_REPO"
export OV_GPU_FP16_SKIP_OPTIMIZATION=1

python3 ov_server.py --model "$MODEL_DIR" --device "$DEVICE" --port "$PORT" --n_ctx 32768

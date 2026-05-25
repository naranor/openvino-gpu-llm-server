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
# We use DeepSeek-R1-Distill-Qwen-1.5B-int8-ov: The best reasoning/coding model for 1.5B class (May 2026)
# INT8 is stable and fast on Intel Gen 9.5 GPU
MODEL_REPO="OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
MODEL_DIR="models/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DeepSeek-R1 reasoning model (INT8 OpenVINO)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
else
    echo "Model already exists in $MODEL_DIR"
fi

# 3. Run the server
PORT=8000
DEVICE="GPU"

echo "Starting OpenVINO server on port $PORT using $DEVICE..."
echo "Model: $MODEL_DIR"

# Set stability flags
export OV_GPU_FP16_SKIP_OPTIMIZATION=1

# Run the python server
python3 ov_server.py --model "$MODEL_DIR" --device "$DEVICE" --port "$PORT"

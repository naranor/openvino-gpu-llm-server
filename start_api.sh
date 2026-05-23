#!/bin/bash

# Script to set up and run an OpenAI-compatible API server using OpenVINO on Intel GPU

set -e

echo "=== OpenVINO OpenAI API Server Setup ==="

# 1. Source oneAPI environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "Sourcing Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
else
    echo "WARNING: /opt/intel/oneapi/setvars.sh not found. GPU inference might fail if drivers/runtimes are not in path."
fi

# 2. Install/Verify dependencies
echo "Verifying Python dependencies..."
pip install -q fastapi uvicorn openvino-genai huggingface_hub pydantic

# 3. Model Configuration
# We use Qwen2.5-Coder-3B-Instruct-int4-ov
MODEL_REPO="OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov"
MODEL_DIR="models/Qwen2.5-Coder-3B-Instruct-int4-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading model $MODEL_REPO..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
else
    echo "Model already exists in $MODEL_DIR"
fi

# 4. Run the server
PORT=8000
DEVICE="GPU"

echo "Starting OpenVINO server on port $PORT using $DEVICE..."
echo "Model: $MODEL_DIR"

# Set some environment variables for better performance/stability on Intel iGPU
export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OPENVINO_LOG_LEVEL=0

# Run the python server
python3 ov_server.py --model "$MODEL_DIR" --device "$DEVICE" --port "$PORT"

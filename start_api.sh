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
# We use Qwen2.5-Coder-1.5B-Instruct-fp16-ov as it is verified to work on UHD 620
MODEL_DIR="models/Qwen2.5-Coder-1.5B-Instruct-fp16-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Exporting model to FP16 OpenVINO format..."
    pip install -q optimum-intel
    optimum-cli export openvino --model Qwen/Qwen2.5-Coder-1.5B-Instruct --task text-generation-with-past --weight-format fp16 "$MODEL_DIR"
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

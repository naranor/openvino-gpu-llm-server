#!/bin/bash
set -e

echo "=== OpenVINO GPU API Server: GPT-OSS-20B (INT8 Heavy Coder) ==="

# 1. Environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# 2. Model Configuration
# GPT-OSS-20B is a massive model for its class, strong in general open-source knowledge
MODEL_REPO="OpenVINO/gpt-oss-20b-int8-ov"
MODEL_DIR="models/gpt-oss-20b-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading GPT-OSS-20B OpenVINO INT8 (~21GB)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
fi

# 3. Execution
PORT=8000
DEVICE="GPU"

# Stability flags for Intel Gen 9.5
export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OPENVINO_LOG_LEVEL=0
export OV_GPU_WAIT_TYPE=SLEEP

# Run server with 32k context and conservative batch size
# Note: 20B model will take significant time to prefill
python3 ov_server.py \
    --model "$MODEL_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --n_ctx 32768 \
    --batch_size 128 \
    --log_level INFO

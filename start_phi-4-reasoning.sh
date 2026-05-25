#!/bin/bash
set -e

echo "=== OpenVINO GPU API Server: Phi-4-Reasoning (14B INT8) ==="

# 1. Environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# 2. Model Configuration
# Phi-4-Reasoning is the 2026 logic leader in the 14B class
MODEL_REPO="OpenVINO/Phi-4-reasoning-int8-ov"
MODEL_DIR="models/Phi-4-reasoning-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading Microsoft Phi-4-Reasoning OpenVINO INT8 (~15GB)..."
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
# 14B model requires more memory bandwidth, 128 batch is safe
python3 ov_server.py \
    --model "$MODEL_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --n_ctx 32768 \
    --batch_size 128 \
    --log_level INFO

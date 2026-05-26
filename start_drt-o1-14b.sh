#!/bin/bash
set -e

echo "=== OpenVINO GPU API Server: DRT-o1-14B (Reasoning Plus INT8) ==="

# 1. Environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# 2. Model Configuration
# DRT-o1-14B is a deep reasoning model distilled using advanced CoT methods
MODEL_REPO="AIFunOver/DRT-o1-14B-openvino-8bit"
MODEL_DIR="models/DRT-o1-14B-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading DRT-o1-14B reasoning model (~15GB)..."
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
python3 ov_server.py \
    --model "$MODEL_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --n_ctx 32768 \
    --batch_size 128 \
    --log_level INFO

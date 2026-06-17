#!/bin/bash
set -e

echo "=== OpenVINO Diffusion Text Server: Dream-Coder-7B (naranor/HF) ==="

# 1. Environment & Stability Flags (Crucial for Intel UHD 620)
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OV_GPU_WAIT_TYPE=SLEEP

# 2. Model Configuration
MODEL_REPO="naranor/Dream-Coder-7B-ov-int8"
FINAL_OV_DIR="models/Dream-Coder-7B-ov-int8"

# Step 1: Check if model exists and is valid
if [ ! -s "$FINAL_OV_DIR/model.xml" ]; then
    echo "Pre-converted OpenVINO model not found locally."
    echo "Downloading Dream-Coder-7B (INT8 OpenVINO) from $MODEL_REPO..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$FINAL_OV_DIR')"
    echo "Download complete."
else
    echo "OpenVINO model found in $FINAL_OV_DIR. Ready to start."
fi

# 3. Execution
PORT=8002
DEVICE="GPU"
LOG_LEVEL=${1:-"INFO"}

echo "Starting Dream-Coder Server on $DEVICE (Port $PORT)"
python3 diffusion_server.py \
    --model "$FINAL_OV_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --log_level "$LOG_LEVEL"

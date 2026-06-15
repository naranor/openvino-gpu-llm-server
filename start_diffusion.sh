#!/bin/bash
set -e

echo "=== OpenVINO Diffusion Text Server: DiffuCoder-7B (Apple) ==="

# 1. Environment & Stability Flags
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OV_GPU_WAIT_TYPE=SLEEP

# 2. Model Configuration
MODEL_REPO="apple/DiffuCoder-7B-Instruct"
SOURCE_DIR="models/tmp_diffucoder"
FINAL_OV_DIR="models/DiffuCoder-7B-Instruct-ov-int8"

# Step 1: Check if conversion is needed
if [ ! -d "$FINAL_OV_DIR" ]; then
    echo "Final OpenVINO model not found."
    
    # Step 2: Check if we have source weights locally
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "Source weights not found. Downloading from $MODEL_REPO..."
        mkdir -p "$SOURCE_DIR"
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$SOURCE_DIR')"
    else
        echo "Found existing source weights in $SOURCE_DIR. Skipping download."
    fi
    
    # Step 3: Convert to OpenVINO
    echo "Converting $SOURCE_DIR to OpenVINO IR (INT8)..."
    python3 convert_to_ov.py --model "$SOURCE_DIR" --output "$FINAL_OV_DIR" --precision int8
    
    echo "Conversion complete. Source weights preserved in $SOURCE_DIR."
else
    echo "OpenVINO model found in $FINAL_OV_DIR. Ready to start."
fi

# 3. Execution
PORT=8001
DEVICE="GPU"

echo "Starting Diffusion Server on $DEVICE (Port $PORT)"
python3 diffusion_server.py \
    --model "$FINAL_OV_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --log_level INFO

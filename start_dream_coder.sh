#!/bin/bash
set -e

echo "=== OpenVINO Diffusion Text Server: Dream-Coder-7B (SOTA 2025) ==="

# 1. Environment & Stability Flags (Crucial for Intel UHD 620)
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OV_GPU_WAIT_TYPE=SLEEP

# 2. Model Configuration
# Note: Using the official repo for Dream-Coder 7B
MODEL_REPO="Dream-org/Dream-Coder-v0-Instruct-7B"
SOURCE_DIR="models/tmp_dreamcoder"
FINAL_OV_DIR="models/Dream-Coder-7B-ov-int8"

# Step 1: Check if conversion is needed
# Check if model.xml exists and is not empty
if [ ! -s "$FINAL_OV_DIR/model.xml" ]; then
    echo "Final OpenVINO model not found or invalid (0B)."
    rm -rf "$FINAL_OV_DIR" # Clear any broken 0B files
    
    # Step 2: Check if we have source weights locally
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "Source weights not found. Downloading from $MODEL_REPO..."
        mkdir -p "$SOURCE_DIR"
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$SOURCE_DIR')"
    else
        echo "Found existing source weights in $SOURCE_DIR. Skipping download."
    fi
    
    # Step 3: Convert to OpenVINO (Using INT8 for stability on UHD 620)
    echo "Converting $SOURCE_DIR to OpenVINO IR (INT8)..."
    
    # BRUTE FORCE: Overwrite the cached module if it exists
    CACHE_MOD_DIR="/home/naranor/.cache/huggingface/modules/transformers_modules/tmp_dreamcoder"
    if [ -d "$CACHE_MOD_DIR" ]; then
        echo "Forcing patches into Transformers cache..."
        cp "$SOURCE_DIR/modeling_dream.py" "$CACHE_MOD_DIR/"
    fi
    
    python3 convert_to_ov.py --model "$SOURCE_DIR" --output "$FINAL_OV_DIR" --precision int8
    
    # Copy required architecture files for local loading
    echo "Configuring custom architecture files..."
    cp "$SOURCE_DIR"/*.py "$FINAL_OV_DIR/" 2>/dev/null || true
    cp "$SOURCE_DIR"/config.json "$FINAL_OV_DIR/" 2>/dev/null || true
    
    echo "Conversion complete."
else
    echo "OpenVINO model found in $FINAL_OV_DIR. Ready to start."
fi

# 3. Execution
PORT=8002
DEVICE="GPU"
LOG_LEVEL=${1:-"INFO"}

echo "Starting Dream-Coder Server on $DEVICE (Port $PORT)"
# We reuse diffusion_server.py as it has the correct manual denoising loop
python3 diffusion_server.py \
    --model "$FINAL_OV_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --log_level "$LOG_LEVEL"

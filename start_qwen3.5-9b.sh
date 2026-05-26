#!/bin/bash
set -e
echo "=== OpenVINO GPU API Server: Qwen 3.5 9B Instruct (GGUF) ==="

# 1. Environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then source /opt/intel/oneapi/setvars.sh; fi

# 2. Model Configuration
MODEL_REPO="bartowski/Qwen_Qwen3.5-9B-GGUF"
MODEL_FILE="Qwen_Qwen3.5-9B-Q4_K_M.gguf"
MODEL_DIR="models_gguf/qwen3.5-9b"

if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Downloading Qwen 3.5 9B GGUF..."
    mkdir -p "$MODEL_DIR"
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$MODEL_REPO', filename='$MODEL_FILE', local_dir='$MODEL_DIR')"
fi

# 3. Execution
PORT=8000
DEVICE="GPU"

# Stability flags
export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OPENVINO_LOG_LEVEL=0
export OV_GPU_WAIT_TYPE=SLEEP

# Run server with 32k context and conservative batch
python3 ov_server.py \
    --model "$MODEL_DIR/$MODEL_FILE" \
    --device "$DEVICE" \
    --port "$PORT" \
    --n_ctx 32768 \
    --batch_size 128 \
    --log_level INFO

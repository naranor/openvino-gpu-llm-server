#!/bin/bash
set -e
echo "=== OpenVINO GPU API Server: Qwen 3.5 14B Reasoning (GGUF) ==="

# 1. Environment
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then source /opt/intel/oneapi/setvars.sh; fi

# 2. Model Configuration
# This is a high-performance reasoning model distilled with Opus 4.6 logic
MODEL_REPO="Oleg-On/Qwen3.5-14B-A3B-Claude-4.6-Opus-Reasoning-Distilled-reap-Q4_K_M-GGUF"
MODEL_FILE="qwen3.5-14b-a3b-claude-4.6-opus-reasoning-distilled-reap-q4_k_m.gguf"
MODEL_DIR="models_gguf/qwen3.5-14b-reasoning"

if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Downloading Qwen 3.5 14B Reasoning GGUF (~10GB)..."
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

#!/bin/bash
set -e
echo "=== OpenVINO GPU API Server: OpenCoder-8B-Instruct (INT8) ==="
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then source /opt/intel/oneapi/setvars.sh; fi

MODEL_REPO="AIFunOver/OpenCoder-8B-Instruct-openvino-8bit"
MODEL_DIR="models/OpenCoder-8B-Instruct-int8-ov"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading OpenCoder-8B..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
fi

export OV_GPU_FP16_SKIP_OPTIMIZATION=1
export GPU_DISABLE_WINOGRAD_CONVOLUTION=1
export OPENVINO_LOG_LEVEL=0
export OV_GPU_WAIT_TYPE=SLEEP

python3 ov_server.py --model "$MODEL_DIR" --device "GPU" --port 8000 --n_ctx 32768 --batch_size 128

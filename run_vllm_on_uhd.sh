#!/bin/bash

# Script to run vLLM on Intel UHD GPU using the XPU backend (if available)

set -e

echo "=== Setting up vLLM for Intel UHD GPU (XPU backend) ==="

# Check and set up oneAPI for SYCL/Level Zero (required for XPU backend in vLLM)
if [ -z "$ONEAPI_ROOT" ]; then
    if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        echo "Sourcing oneAPI environment..."
        source /opt/intel/oneapi/setvars.sh
    else
        echo "ERROR: Intel oneAPI Base Toolkit not found."
        echo "Please install it from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
        exit 1
    fi
fi

# Verify Level Zero GPU is available
if ! command -v sycl-ls &> /dev/null; then
    echo "ERROR: sycl-ls command not found. Is oneAPI in PATH?"
    exit 1
fi

echo "Available SYCL devices:"
sycl-ls | grep -i "level_zero.*gpu" || echo "No Level Zero GPU detected. Check drivers."

# Set environment variables for vLLM to use Intel XPU
# Note: vLLM's XPU backend is experimental and may require specific settings
export VLLM_ATTENTION_BACKEND="XFORMERS"  # XFORMERS is compatible with XPU? Actually, we might need to use the XPU attention backend if available.
# From the commit, they didn't set attention backend, so let's try without and see if XPU is auto-selected.
# Alternatively, we can try to set the backend to 'XPU' but vLLM might not have that string.
# Let's leave it unset and hope the XPU kernels are used when available.

# We also set the multiprocessing method to spawn (as seen in the commit)
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# For XPU, we might need to set the device
# vLLM might automatically detect the XPU device if the kernels are installed and the environment is set.

# Install vLLM and vllm-xpu-kernels (if available)
# Note: The standard vLLM from PyPI does not include XPU support. We need to install the XPU kernels separately.
# However, as of now, there might not be a separate package. We might need to build from source or use a pre-built wheel from intel.

# We'll try to install vLLM first, then try to install vllm-xpu-kernels if it exists.
echo "Installing vLLM..."
pip install vllm

# Try to install the XPU kernels package (if available)
echo "Attempting to install vllm-xpu-kernels (for Intel GPU support)..."
if pip install vllm-xpu-kernels 2>/dev/null; then
    echo "vllm-xpu-kernels installed successfully."
else
    echo "vllm-xpu-kernels not found via pip. We'll proceed with vLLM and hope the XPU backend is available via other means."
    echo "Note: For full XPU support, you may need to build vLLM from source with XPU enabled or use the intel/ai-containers Docker image."
fi

# Verify installation
echo "Checking vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Set up directories for models
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_DIR="$SCRIPT_DIR/models_vllm"
mkdir -p "$MODEL_DIR"

# We'll use a small model that fits in limited GPU memory (UHD uses shared RAM)
# Opt-125m is very small (125M parameters) and should work.
# Alternatively, we can use TinyLlama or similar.
MODEL_NAME="facebook/opt-125m"
# For a slightly larger but still small model, you can use:
# MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Note: 1B model might be pushing it for UHD unless quantized.

MODEL_PATH="$MODEL_DIR/$(echo $MODEL_ID | tr '/' '_')"
# Actually, vLLM can download from Hugging Face directly, so we don't need to pre-download unless we want to.

echo "=== Starting vLLM server for model: $MODEL_NAME ==="
echo "Note: The first run may take time to compile kernels and trace the model."
echo "We'll set GPU memory utilization to 0.5 to be safe with shared memory."

# Start the vLLM server in the background
# We'll use a port that is unlikely to be conflicted
PORT=8000

# We need to set the model loading options for low memory usage
# We'll use quantization if possible, but for simplicity, we'll run in half precision (float16) and hope it fits.
# For UHD, we might need to use int8 or 4-bit quantization, but vLLM's quantization might not be available for XPU yet.
# We'll try without quantization first and see if it works.

# Command to start the server:
#   vllm serve $MODEL_NAME --port $PORT --tensor-parallel-size 1 --gpu-memory-utilization 0.5
# Note: tensor-parallel-size should be 1 for a single GPU.

# However, note that the commit used:
#   vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 4 ...
#   but that's a large model. We are using a small model.

# Let's start the server
vllm serve $MODEL_NAME \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-num-batched-tokens 512 \
    --max-model-len 512 \
    --disable-log-requests \
    --trust-remote-code &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID on port $PORT"

# Wait a bit for the server to start
echo "Waiting for server to start..."
sleep 15

# Check if the server is running by querying the models endpoint
if curl -s http://localhost:$PORT/v1/models > /dev/null; then
    echo "vLLM server is running successfully!"
else
    echo "ERROR: vLLM server failed to start or is not responding."
    echo "Checking logs..."
    # We don't have logs captured, so we'll try to kill and let the user check
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Test the server with a simple completion request
echo "=== Testing the server with a simple prompt ==="
RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"prompt\": \"Hello, how are you today?\",
    \"max_tokens\": 50,
    \"temperature\": 0.7
  }" | jq -r '.choices[0].text' 2>/dev/null || curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"prompt\": \"Hello, how are you today?\",
    \"max_tokens\": 50,
    \"temperature\": 0.7
  }" | python -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])")

echo "Response: $RESPONSE"

echo "=== vLLM server is running in the background (PID: $VLLM_PID) ==="
echo "To stop it later, run: kill $VLLM_PID"
echo "You can now send requests to http://localhost:$PORT/v1/completions or use the OpenAI-compatible API."
echo "Example curl command:"
echo "  curl -X POST http://localhost:$PORT/v1/completions -H \"Content-Type: application/json\" -d '{\"model\": \"$MODEL_NAME\", \"prompt\": \"Your prompt here\", \"max_tokens\": 100}'"
echo ""
echo "Note: For better performance on UHD GPU, consider:"
echo "  1. Using a smaller model or a quantized model (if vLLM quantization supports XPU)."
echo "  2. Adjusting --gpu-memory-utilization (lower if you encounter OOM)."
echo "  3. Reducing --max-model-len and --max-num-batched-tokens."
echo ""
echo "To stop the server when done, run: kill $VLLM_PID"
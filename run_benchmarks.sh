#!/bin/bash

# Master benchmark script
MODELS=(
    "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
    "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int8-ov"
    "OpenVINO/Qwen3-8B-int8-ov"
    "OpenVINO/Qwen2.5-Coder-7B-Instruct-int8-ov"
    "OpenVINO/gemma-3-12b-it-int8-ov"
)

# Device argument
DEVICE=${1:-"GPU"}
RESULTS_FILE="results_${DEVICE,,}.txt"

# Source oneAPI
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

echo "ID | Model | Speed (t/s) | Load Time (s)" > "$RESULTS_FILE"
echo "---|---|---|---" >> "$RESULTS_FILE"

for i in "${!MODELS[@]}"; do
    REPO="${MODELS[$i]}"
    DIR="models_bench/${REPO////_}"
    
    echo "--- Benchmarking $REPO on $DEVICE ($((i+1))/${#MODELS[@]}) ---"
    
    # Download
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO', local_dir='$DIR')"
    
    # Benchmark
    RESULT=$(python3 benchmark.py "$DIR" "$DEVICE" | grep "Speed:" | awk '{print $2}')
    LOAD=$(python3 benchmark.py "$DIR" "$DEVICE" | grep "loaded in" | awk '{print $4}' | sed 's/s//')
    
    echo "$((i+1)) | $REPO | $RESULT | $LOAD" >> "$RESULTS_FILE"
    
    # Cleanup
    rm -rf "$DIR"
    echo "Space cleared."
done

cat "$RESULTS_FILE"

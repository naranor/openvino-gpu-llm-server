#!/bin/bash

# Master benchmark script
MODELS=(
    "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
    "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int8-ov"
    "OpenVINO/Qwen3-8B-int8-ov"
    "OpenVINO/Qwen2.5-Coder-7B-Instruct-int8-ov"
    "OpenVINO/gemma-3-12b-it-int8-ov"
)

# Source oneAPI
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

echo "ID | Model | Speed (t/s) | Load Time (s)" > results.txt
echo "---|---|---|---" >> results.txt

for i in "${!MODELS[@]}"; do
    REPO="${MODELS[$i]}"
    DIR="models_bench/${REPO////_}"
    
    echo "--- Benchmarking $REPO ($((i+1))/${#MODELS[@]}) ---"
    
    # Download
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO', local_dir='$DIR')"
    
    # Benchmark
    RESULT=$(python3 benchmark.py "$DIR" | grep "Speed:" | awk '{print $2}')
    LOAD=$(python3 benchmark.py "$DIR" | grep "loaded in" | awk '{print $4}' | sed 's/s//')
    
    echo "$((i+1)) | $REPO | $RESULT | $LOAD" >> results.txt
    
    # Cleanup
    rm -rf "$DIR"
    echo "Space cleared."
done

cat results.txt

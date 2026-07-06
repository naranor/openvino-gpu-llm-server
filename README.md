# OpenVINO GPU LLM Server

An experimental OpenAI-compatible API server optimized for LLM inference on **Intel Integrated GPUs (iGPU)**. Specifically engineered for the Intel UHD 620 generation and beyond, utilizing the **OpenVINO GenAI** backend and **oneAPI Level Zero** driver.

## Overview

This project provides a robust, production-ready environment for running large-scale coding models (up to 12B+ parameters) on hardware previously considered "weak". By leveraging shared memory addressing and modern Intel runtimes, we enable local AI assistance on mobile workstations without dedicated VRAM.

## Technical Investigation & Rationale (May 2026)

### Hardware Specification (Benchmark Host)
All benchmarks and tests were conducted on the following hardware:
*   **CPU:** Intel(R) Core(TM) i7-8565U @ 1.80GHz (4 Cores / 8 Threads)
*   **GPU:** Intel(R) UHD Graphics 620 (Integrated, Gen 9.5)
*   **RAM:** 32 GB DDR4 (Shared with GPU via Level Zero)
*   **Disk:** WD Black SN750 NVMe SSD (High-speed model loading)
*   **OS:** Linux (Ubuntu 24.04 LTS)
*   **Runtime:** OpenVINO™ 2026.1.0 with oneAPI Level Zero driver

### 1. The Memory Wall: Bypassing the 4GB Aperture
*   **Challenge:** Standard OpenCL/GPU drivers often impose a 4GB segment limit for iGPUs.
*   **Finding:** By utilizing the **oneAPI Level Zero** driver, we discovered that OpenVINO can address the **entire shared system memory**.
*   **Result:** Verified visibility of **28.29 GB** of GPU memory on a 32GB RAM host. This allowed us to shift from tiny 1.5B models to massive 7B-12B models on the GPU.

### 2. The Quantization Dilemma: INT4 vs INT8
*   **Challenge:** Initial attempts to run **INT4** quantized models (standard for most LLMs) resulted in fatal `clBuildProgram` errors and kernel compilation hangs.
*   **Root Cause Analysis:** Intel Gen 9.5 hardware lacks specific hardware instructions (like DPAS) required by modern highly-optimized INT4 kernels in some runtimes.
*   **Solution:** Extensive testing proved that **INT8 (8-bit integer)** and **FP16 (16-bit float)** are perfectly stable. 
*   **Performance Insight:** DeepSeek-R1-Distill-Qwen-7B in INT8 format provides a superior "intelligence-per-second" ratio on this hardware.

### 3. Backend Selection: OpenVINO GenAI vs vLLM/llama.cpp
*   **vLLM (XPU):** Encountered device inference issues and high overhead for iGPU.
*   **llama.cpp (SYCL):** Provided stable loading but suffered from synchronization hangs during generation on UHD 620.
*   **OpenVINO GenAI (Winner):** The most lightweight and reliable path. It demonstrated the lowest latency (sub-second response for 1.5B models) and native support for advanced techniques like Speculative Decoding.

## Supported Models (Pre-configured)

Verified performance on **Intel UHD 620** (Shared Memory enabled):

| Model Name | Optimization | GPU Speed | CPU Speed | GPU Load | Best For... |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-R1-Distill-Qwen-7B** | INT8 | **3.01 t/s** | 2.44 t/s | 22s | **Reasoning (7B)** |
| **Qwen3-8B-Instruct** | INT8 | **2.71 t/s** | 2.29 t/s | 31s | **SOTA Speed (8B)** |
| **Qwen2.5-Coder-7B-Instruct** | INT8 | **2.46 t/s** | 2.02 t/s | 28s | **Reliable Coder** |
| **DeepSeek-R1-Distill-Qwen-1.5B**| INT8 | **12.12 t/s**| 10.50 t/s| 4s | **Low Latency** |
| **Gemma-2-9B-it** | INT8 | **1.78 t/s** | 1.39 t/s | 40s | **Pure Logic (9B)** |

## Diffusion Text Models (dLLMs)

Starting June 2026, we have introduced support for **Discrete Diffusion Language Models** optimized for block-based code generation. Unlike autoregressive models, these models generate entire blocks of code iteratively, providing superior context awareness and structural integrity.

### Optimized Diffusion Pipeline
Our implementation features the **LocalLeap (Anchor-Propagation)** algorithm, which allows the model to commit confident neighboring tokens alongside anchors, significantly reducing the number of diffusion steps required for high-quality output.

| Model Name | Optimization | Port | Use Case |
| :--- | :--- | :--- | :--- |
| **DiffuCoder-7B** | INT8 | **8001** | Coherent code generation with bidirectional context. |
| **Dream-Coder-7B** | INT8 | **8002** | SOTA 2025 diffusion model for complex architectural tasks. |

### Running Diffusion Models
Diffusion models use a separate server module and dedicated ports:
```bash
# Launch DiffuCoder
./start_diffucoder.sh

# Launch Dream-Coder
./start_dream_coder.sh
```

## Installation

### 1. Requirements
*   **OS:** Linux (Ubuntu 24.04+ recommended)
*   **Drivers:** Intel Compute Runtime (Level Zero)
*   **Toolkit:** Intel oneAPI Base Toolkit (specifically `setvars.sh` must be present)

### 2. Setup Environment
```bash
git clone <your-repo-url>
cd lmm_gpu
pip install -r requirements.txt
```

## Usage

Each model has its own optimized launch script. Before running, ensure you have sourced the oneAPI environment (though the scripts attempt to do this automatically).

### Launching the Reasoning model (DeepSeek-R1):
```bash
./start_deepseek-r1-7b.sh
```

### Launching the high-speed Qwen3 model:
```bash
./start_qwen3-8b.sh
```

### API Endpoint (OpenAI Compatible)
The server binds to `127.0.0.1` by default (localhost only). Use `--host 0.0.0.0` to expose on the LAN.

**Health check:**
```bash
curl http://localhost:8000/health
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "7B",
    "messages": [{"role": "user", "content": "Explain binary search"}],
    "stream": true
  }'
```

## Project Structure

```
lmm_gpu/
├── ov_server.py              # AR LLM server (Continuous Batching, streaming SSE, port 8000)
├── diffusion_server.py       # Diffusion LLM server (LocalLeap, ports 8001/8002)
├── convert_to_ov.py          # HF → OpenVINO IR conversion (requires torch, nncf)
├── benchmark.py              # Tokens/sec benchmark utility
├── push_to_hf.py             # Upload converted models to Hugging Face Hub
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
│
├── start_*.sh                # Model launch scripts (see table below)
├── run_benchmarks.sh         # Batch benchmark runner
│
├── test_ov.py                # OpenVINO device diagnostics
├── test_ov_fp16.py           # FP16 model smoke test
├── test_gpu_capacity.py      # Batch size / prefill stress test
├── test_agent_workload.py    # E2E streaming test with large context
│
└── models/                   # Local model weights (gitignored, downloaded from HF)
```

### Launch Scripts

| Script | Model | Port |
| :--- | :--- | :--- |
| `start_api.sh` | DeepSeek-R1-Distill-Qwen-7B | 8000 |
| `start_deepseek-r1-7b.sh` | DeepSeek-R1-Distill-Qwen-7B | 8000 |
| `start_deepseek-r1-1.5b.sh` | DeepSeek-R1-Distill-Qwen-1.5B | 8000 |
| `start_qwen3-8b.sh` | Qwen3-8B-Instruct | 8000 |
| `start_qwen2.5-coder-7b.sh` | Qwen2.5-Coder-7B-Instruct | 8000 |
| `start_gemma2-9b.sh` | Gemma-2-9B-it | 8000 |
| `start_gemma3-12b.sh` | Gemma-3-12B-it | 8000 |
| `start_mistral-7b.sh` | Mistral-7B-Instruct-v0.3 | 8000 |
| `start_mistral-nemo-12b.sh` | Mistral-Nemo-12B-Instruct | 8000 |
| `start_phi-3-medium.sh` | Phi-3-medium-4k-instruct | 8000 |
| `start_phi-4-reasoning.sh` | Phi-4-reasoning | 8000 |
| `start_glm-4-9b.sh` | GLM-4-9B-Chat | 8000 |
| `start_opencoder-8b.sh` | OpenCoder-8B-Instruct | 8000 |
| `start_diffucoder.sh` | DiffuCoder-7B | 8001 |
| `start_dream_coder.sh` | Dream-Coder-7B | 8002 |

## License
MIT License — see [LICENSE](LICENSE). Check individual model licenses (mostly Apache 2.0).

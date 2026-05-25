# OpenVINO GPU LLM Server

An OpenAI-compatible API server optimized for high-performance LLM inference on **Intel Integrated GPUs (iGPU)**. Specifically engineered for the Intel UHD 620 generation and beyond, utilizing the **OpenVINO GenAI** backend and **oneAPI Level Zero** driver.

## 🚀 Overview

This project provides a robust, production-ready environment for running large-scale coding models (up to 12B+ parameters) on hardware previously considered "weak". By leveraging shared memory addressing and modern Intel runtimes, we enable local AI assistance on mobile workstations without dedicated VRAM.

## 🧪 Technical Investigation & Rationale (May 2026)

The development of this server involved a rigorous research phase to overcome hardware-specific limitations of Intel Gen 9.5 (UHD 620) graphics.

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

## 📦 Supported Models (Pre-configured)

Verified performance on **Intel UHD 620** (Shared Memory enabled):

| Model Name | Optimization | Speed (t/s) | Load Time | Best For... |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-R1-Distill-Qwen-7B** | INT8 | **3.01** | 22s | **Coding & Reasoning (Recommended)** |
| **Qwen3-8B-Instruct** | INT8 | **2.71** | 31s | **Raw Generation Speed** |
| **Qwen2.5-Coder-7B-Instruct** | INT8 | **2.46** | 28s | **Reliable Code Completion** |
| **DeepSeek-R1-Distill-Qwen-1.5B**| INT8 | **12.12**| 4s | **Ultra-low latency** |
| **Gemma-3-12B-it** | INT8 | N/A | 47s | **Vision (Better on CPU)** |

## 🛠 Installation

### 1. Requirements
*   **OS:** Linux (Ubuntu 24.04+ recommended)
*   **Drivers:** Intel Compute Runtime (OpenCL/Level Zero)
*   **Toolkit:** Intel oneAPI Base Toolkit (specifically `setvars.sh` must be present)

### 2. Setup Environment
```bash
git clone <your-repo-url>
cd lmm_gpu
pip install -r requirements.txt
```

## ⚡ Usage

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
The server runs on `http://localhost:8000`. You can point any OpenAI-compatible client (Aider, Continue.dev, VS Code extensions) to this URL.

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

## 📂 Project Structure
*   `ov_server.py`: Core FastAPI wrapper for OpenVINO GenAI.
*   `start_*.sh`: Model-specific orchestration scripts.
*   `test_ov.py`: Diagnostic tool for GPU memory and device properties.
*   `requirements.txt`: Python dependencies.

## 🤝 Contributing
Contributions are welcome! Specifically in the areas of:
*   Speculative decoding support for larger models.
*   NPU acceleration for Core Ultra processors.
*   Support for DeepSeek-V3 MoE architectures.

## 📜 License
MIT License. Check individual model licenses (mostly Apache 2.0).

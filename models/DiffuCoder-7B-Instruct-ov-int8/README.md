---
library_name: openvino
pipeline_tag: text-generation
tags:
- openvino
- int8
- nncf
- code-generation
- diffusion
- diffucoder
base_model: apple/DiffuCoder-7B-Instruct
---

# DiffuCoder-7B-Instruct OpenVINO INT8

This is the OpenVINO IR version of the [apple/DiffuCoder-7B-Instruct](https://huggingface.co/apple/DiffuCoder-7B-Instruct) model, optimized for Intel GPUs and CPUs. 
The model weights have been compressed to **INT8** using [NNCF](https://github.com/openvinotoolkit/nncf) for improved inference performance and reduced memory footprint.

DiffuCoder is a discrete diffusion model designed for code generation.

## Usage

This model requires custom architecture files. When loading, you must use `trust_remote_code=True`.

### Using with OpenVINO GenAI

Currently, standard `openvino_genai` pipelines might not fully support the custom "Dream" architecture natively without a custom denoising loop. 
For a complete implementation of the Discrete Diffusion loop (including optimizations like LocalLeap), refer to the custom server implementation.

### Manual Inference (Python)

```python
import openvino as ov
from transformers import AutoTokenizer, AutoConfig

model_path = "your_hf_username/DiffuCoder-7B-Instruct-ov-int8"

core = ov.Core()
ov_model = core.read_model(f"{model_path}/model.xml")
model = core.compile_model(ov_model, "GPU") # or "CPU"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Note: Execution requires a discrete diffusion sampling loop.
# See the repository's diffusion_server.py for the full loop implementation.
```

## Optimization Details
- **Quantization:** NNCF Weight-Only Quantization (INT8_ASYM)
- **Target Hardware:** Intel integrated GPUs (e.g., UHD 620) and CPUs.

## Repository
For the complete server implementation and inference scripts designed specifically for Intel integrated graphics, please visit the main project repository:
[https://github.com/naranor/openvino-gpu-llm-server](https://github.com/naranor/openvino-gpu-llm-server)

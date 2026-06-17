---
library_name: openvino
pipeline_tag: text-generation
tags:
- openvino
- int8
- nncf
- code-generation
- diffusion
- dreamcoder
base_model: Dream-org/Dream-Coder-v0-Instruct-7B
---

# Dream-Coder-7B OpenVINO INT8

This is the OpenVINO IR version of the [Dream-org/Dream-Coder-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-Coder-v0-Instruct-7B) model, optimized for Intel GPUs and CPUs. 
The model weights have been compressed to **INT8** using [NNCF](https://github.com/openvinotoolkit/nncf) for improved inference performance and reduced memory footprint on integrated graphics.

Dream-Coder is a state-of-the-art discrete diffusion language model that supports any-order code generation and excels at architectural planning.

## Usage

This model requires custom architecture files. When loading, you must use `trust_remote_code=True`.

### Manual Inference (Python)

```python
import openvino as ov
from transformers import AutoTokenizer, AutoConfig

model_path = "your_hf_username/Dream-Coder-7B-ov-int8"

core = ov.Core()
ov_model = core.read_model(f"{model_path}/model.xml")
model = core.compile_model(ov_model, "GPU") # or "CPU"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Note: Execution requires a discrete diffusion sampling loop.
# See the repository's diffusion_server.py for the full loop implementation with LocalLeap support.
```

## Optimization Details
- **Quantization:** NNCF Weight-Only Quantization (INT8_ASYM)
- **Target Hardware:** Intel integrated GPUs (e.g., Gen9.5 UHD 620) and CPUs.
- **Patches Applied:** Includes fixes for standard OpenVINO conversion paths to bypass `aten::cat` axis issues in the custom attention block.

## Repository
For the complete server implementation, including the discrete diffusion sampling loop and optimizations like LocalLeap designed for Intel integrated graphics, please visit the main project repository:
[https://github.com/naranor/openvino-gpu-llm-server](https://github.com/naranor/openvino-gpu-llm-server)

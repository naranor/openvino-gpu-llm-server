import time
from llama_cpp import Llama
import os

# Source setvars.sh usually happens in shell, assuming environment is ready
print("Loading 0.5B model on GPU (FP16)...")
llm = Llama(
    model_path="models_gguf/qwen2.5-coder-0.5b-instruct-fp16.gguf",
    n_gpu_layers=-1,
    n_ctx=512,
    n_batch=128,
    n_threads=4,
    verbose=True
)

print("Starting generation...")
start = time.time()
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "1+1="}],
    max_tokens=10,
    stream=False
)
end = time.time()

print(f"Response: {output['choices'][0]['message']['content']}")
print(f"Time taken: {end - start:.2f}s")

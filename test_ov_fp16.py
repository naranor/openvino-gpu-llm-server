import openvino_genai as ov_genai
import time

model_dir = "models/Qwen2.5-Coder-1.5B-Instruct-fp16-ov"
device = "GPU"

print(f"Loading 1.5B FP16 model to {device}...")
pipe = ov_genai.LLMPipeline(model_dir, device)

print("Starting generation...")
start = time.time()
result = pipe.generate("1+1=", max_new_tokens=10)
end = time.time()

print(f"Response: {result}")
print(f"Time taken: {end - start:.2f}s")

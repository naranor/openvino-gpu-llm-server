import openvino_genai as ov_genai
import time
import sys
import os

def benchmark(model_dir):
    device = "GPU"
    print(f"Benchmarking {model_dir} on {device}...")
    
    try:
        # Set stability flag
        os.environ["OV_GPU_FP16_SKIP_OPTIMIZATION"] = "1"
        
        start_load = time.time()
        pipe = ov_genai.LLMPipeline(model_dir, device)
        load_time = time.time() - start_load
        print(f"  Model loaded in {load_time:.2f}s")

        prompt = "Write a high-performance quicksort in Python."
        max_tokens = 100
        
        # Warm-up run
        pipe.generate("Hello", max_new_tokens=5)
        
        start_gen = time.time()
        # Using simple generate to measure total time for simplicity
        # Real t/s would require streamer or perf metrics if available in this version
        result = pipe.generate(prompt, max_new_tokens=max_tokens)
        end_gen = time.time()
        
        total_time = end_gen - start_gen
        # Estimate tokens (this is a rough estimate since we don't have the exact token count here without tokenizer)
        # But we can assume it hit max_tokens or we can count words as proxy if needed.
        # Actually, let's use a streamer to count exactly.
        
        class CounterStreamer(ov_genai.StreamerBase):
            def __init__(self):
                super().__init__()
                self.count = 0
            def write(self, text):
                self.count += 1 # In GenAI, write is called per subword/token chunk
                return ov_genai.StreamingStatus.RUNNING

        counter = CounterStreamer()
        start_gen = time.time()
        pipe.generate(prompt, max_new_tokens=max_tokens, streamer=counter.write)
        end_gen = time.time()
        
        gen_time = end_gen - start_gen
        tps = counter.count / gen_time
        
        print(f"  Result: {counter.count} tokens in {gen_time:.2f}s")
        print(f"  Speed: {tps:.2f} tokens/sec")
        return tps, load_time
    except Exception as e:
        print(f"  Error benchmarking {model_dir}: {e}")
        return None, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <model_dir>")
        sys.exit(1)
    benchmark(sys.argv[1])

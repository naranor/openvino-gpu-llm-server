import openvino_genai as ov_genai
import time
import os

model_path = "models/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
device = "GPU"
# Large prompt to test batching/prefill limit
prompt = "Explain the history of the universe. " * 300 

def test_batch_size(batch_size):
    print(f"\n[{batch_size}] Testing identical server logic...")
    
    # EXACT same config as in ov_server.py
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.max_num_batched_tokens = batch_size
    scheduler_config.num_kv_blocks = 32768 // 16
    scheduler_config.dynamic_split_fuse = True

    try:
        # Use exact constructor signature verified earlier
        pipe = ov_genai.ContinuousBatchingPipeline(model_path, scheduler_config, device)
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 10 # Generate some tokens to ensure decoding works too
        
        start = time.perf_counter()
        handle = pipe.add_request(0, prompt, config)
        
        tokens_received = 0
        step_latencies = []
        
        while True:
            s_start = time.perf_counter()
            pipe.step()
            s_end = time.perf_counter()
            step_latencies.append((s_end - s_start) * 1000)
            
            # Handle token reading logic identical to server
            while handle.can_read():
                res = handle.read()
                if 0 in res:
                    tokens_received += len(res[0].generated_ids)
            
            if handle.get_status() == ov_genai.GenerationStatus.FINISHED:
                break
            
            # Timeout for prefill (should not take more than 5 mins)
            if time.perf_counter() - start > 300:
                print("FAILED: Global timeout reached.")
                return False

        duration = time.perf_counter() - start
        avg_step = sum(step_latencies) / len(step_latencies)
        print(f"SUCCESS: Total time {duration:.2f}s. Avg step: {avg_step:.2f}ms.")
        return True
        
    except Exception as e:
        print(f"FAILED: error: {e}")
        return False

if __name__ == "__main__":
    # Same stability env as in start_api.sh
    os.environ["OV_GPU_FP16_SKIP_OPTIMIZATION"] = "1"
    os.environ["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "1"
    os.environ["OV_GPU_WAIT_TYPE"] = "SLEEP"
    
    # Power of two test
    for size in [512, 1024, 2048, 4096]:
        if not test_batch_size(size):
            break

    print("\nBenchmark complete.")

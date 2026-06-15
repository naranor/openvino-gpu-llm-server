import os
import sys

# 1. ENVIRONMENT SETUP (MUST BE FIRST)
# Mask GPU from PyTorch to prevent deadlocks
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ZE_AFFINITY_MASK"] = ""
# OpenVINO Stability Flags for Intel Gen 9.5
os.environ["OV_GPU_FP16_SKIP_OPTIMIZATION"] = "1"
os.environ["GPU_DISABLE_WINOGRAD_CONVOLUTION"] = "1"
os.environ["OV_GPU_WAIT_TYPE"] = "SLEEP"

import argparse
import json
import time
import uuid
import logging
import traceback
import threading
from typing import List, Optional, Union, Dict, Any

import openvino as ov
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

# Initialize logger
logger = logging.getLogger("diffusion_server")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] [%(threadName)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(title="OpenVINO Diffusion API (Full Debug)")

# Global State
model = None
tokenizer = None
config = None
model_name = ""
is_ready = False

class ChatMessage(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_new_tokens: Optional[int] = 128
    num_steps: Optional[int] = 15
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def sample_tokens_numpy(logits, temperature=0.0, top_p=None, top_k=None):
    if temperature > 0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < np.partition(logits, -top_k, axis=-1)[..., -top_k, None]
        logits = np.copy(logits)
        logits[indices_to_remove] = -np.inf
    probs = softmax(logits)
    if temperature > 0:
        res_indices = []
        res_confidences = []
        for i in range(probs.shape[0]):
            idx = np.random.choice(probs.shape[1], p=probs[i])
            res_indices.append(idx)
            res_confidences.append(probs[i, idx])
        return np.array(res_confidences), np.array(res_indices)
    else:
        indices = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        return confidences, indices

def model_loader_thread(model_path, device):
    global model, tokenizer, config, is_ready
    try:
        logger.info(">>> LOADING THREAD STARTED <<<")
        
        logger.info("Step 1/5: Creating OpenVINO Core...")
        core = ov.Core()
        
        cache_dir = os.path.expanduser("~/.cache/ov_diffusion_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Stability properties for UHD 620
        props = {
            'CACHE_DIR': cache_dir,
            'PERFORMANCE_HINT': 'LATENCY',
            'NUM_STREAMS': '1',
            'INFERENCE_PRECISION_HINT': 'f16',
            'COMPILATION_NUM_THREADS': '2',
        }
        
        if "GPU" in device:
            logger.info("Step 1.1: Applying GPU-specific tweaks (SDPA=NO, Winograd=OFF)...")
            props['GPU_ENABLE_SDPA_OPTIMIZATION'] = 'NO'
            props['GPU_DISABLE_WINOGRAD_CONVOLUTION'] = 'YES'

        logger.info(f"Step 2/5: Loading model files from {model_path}...")
        ov_model = core.read_model(f"{model_path}/model.xml")
        logger.info("Step 2/5: XML/BIN read complete.")

        logger.info(f"Step 3/5: Compiling model for {device}... (HANG ALERT: Monitor CPU/GPU now)")
        start_jit = time.perf_counter()
        # This call is the bottleneck
        model = core.compile_model(ov_model, device, props)
        logger.info(f"Step 3/5: COMPILATION SUCCESSFUL! Duration: {time.perf_counter() - start_jit:.2f}s")

        logger.info("Step 4/5: Loading Tokenizer & Config (Transformers)...")
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Step 4/5: Transformers components loaded.")

        logger.info("Step 5/5: Verifying network signatures...")
        for i, input_obj in enumerate(model.inputs):
            logger.info(f"  Input[{i}]: {input_obj.get_any_name()}")
        for o, output_obj in enumerate(model.outputs):
            logger.info(f"  Output[{o}]: {output_obj.get_any_name()}")

        is_ready = True
        logger.info(">>> MODEL LOADING FINISHED: SERVER IS READY TO SERVE <<<")
    except Exception as e:
        logger.error(f"!!! FATAL ERROR IN LOADER: {e}")
        logger.error(traceback.format_exc())

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "openvino", "ready": is_ready}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: CompletionRequest):
    global model, tokenizer, config, is_ready
    if not is_ready:
        raise HTTPException(status_code=503, detail="Model is still being compiled for GPU. Check console logs.")

    request_id = f"diff-chatcmpl-{uuid.uuid4()}"
    try:
        logger.info(f"[{request_id}] Incoming request for block generation...")
        
        history = [{"role": m.role, "content": m.content} for m in body.messages]
        prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        
        mask_token_id = getattr(tokenizer, 'mask_token_id', None) or getattr(config, 'mask_token_id', None)
        if mask_token_id is None:
            mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

        steps = body.num_steps
        max_new_tokens = body.max_new_tokens
        max_length = input_len + max_new_tokens

        x = np.pad(input_ids, ((0, 0), (0, max_new_tokens)), constant_values=mask_token_id)
        att_mask = np.ones((1, max_length), dtype=np.int64)
        timesteps = np.linspace(1, 1e-12, steps + 1)

        start_gen = time.perf_counter()
        for i in range(steps):
            logger.debug(f"[{request_id}] Diffusion Step {i+1}/{steps}")
            mask_index = (x == mask_token_id)
            if not np.any(mask_index): 
                logger.info(f"[{request_id}] No more mask tokens, stopping at step {i}")
                break

            res = model({"input_ids": x, "attention_mask": att_mask})
            logits = res[0] 
            
            # Dream architecture requires a logit shift
            logits = np.concatenate([logits[:, :1, :], logits[:, :-1, :]], axis=1)
            
            mask_logits = logits[mask_index]
            t, s = timesteps[i], timesteps[i + 1]
            p_transfer = 1 - s / t if i < steps - 1 else 1
            num_to_sample = int(np.sum(mask_index) * p_transfer)
            
            confidence, x0 = sample_tokens_numpy(mask_logits, temperature=body.temperature, top_p=body.top_p, top_k=body.top_k)
            
            if num_to_sample > 0:
                topk_indices = np.argsort(confidence)[-num_to_sample:]
                mask_flat_indices = np.where(mask_index.flatten())[0]
                selected_flat_indices = mask_flat_indices[topk_indices]
                x.flat[selected_flat_indices] = x0[topk_indices]

        duration = time.perf_counter() - start_gen
        generated_ids = x[0, input_len:]
        generated_ids = generated_ids[generated_ids != mask_token_id]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        logger.info(f"[{request_id}] Generation finished in {duration:.2f}s")
        return {
            "id": request_id, "object": "chat.completion", "created": int(time.time()), "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_len, "completion_tokens": len(generated_ids), "total_tokens": input_len + len(generated_ids)},
            "diffusion_stats": {"steps": steps, "seconds": duration}
        }
    except Exception as e:
        logger.error(f"ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    # Apply log levels
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    # Map Python log levels to OpenVINO log levels
    ov_log_map = {"DEBUG": "4", "INFO": "3", "WARNING": "2", "ERROR": "1"}
    os.environ["OPENVINO_LOG_LEVEL"] = ov_log_map.get(args.log_level.upper(), "3")

    model_name = args.model
    
    loader = threading.Thread(target=model_loader_thread, args=(args.model, args.device), daemon=True, name="ModelLoader")
    loader.start()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

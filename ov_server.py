import asyncio
import argparse
import json
import time
import uuid
import threading
import queue
import datetime
import traceback
import logging
from typing import List, Optional, Union, Dict, Any

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Define TRACE level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)
logging.Logger.trace = trace

# Initialize logger
logger = logging.getLogger("ov_server")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log(msg):
    # Backward compatibility for my internal calls, now uses logger
    logger.info(msg)

app = FastAPI(title="OpenVINO GPU API (Continuous Batching)")

# Global State
pipe = None
tokenizer = None
model_name = ""
global_config = {}

# Data structure to track requests
active_requests = {}
requests_to_add = queue.Queue()
requests_to_cancel = queue.Queue()
request_id_counter = 0

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL ERROR: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc), "type": "server_error"}}
    )

def pipeline_worker():
    global pipe, active_requests, request_id_counter, tokenizer
    logger.info("Pipeline worker started.")
    
    while True:
        # 1. Add new requests to the pipeline
        while not requests_to_add.empty():
            try:
                req_data = requests_to_add.get_nowait()
                r_id = req_data['request_id']
                internal_id = request_id_counter
                request_id_counter += 1
                logger.info(f"[{r_id}] Adding request to pipeline with internal_id {internal_id}...")
                
                handle = pipe.add_request(internal_id, req_data['prompt'], req_data['config'])
                active_requests[r_id] = {
                    'handle': handle,
                    'token_queue': req_data['token_queue'],
                    'request_id': r_id,
                    'processed_tokens': 0
                }
            except Exception as e:
                logger.error(f"!!! Error adding request: {e}\n{traceback.format_exc()}")

        # 2. Process cancellations
        while not requests_to_cancel.empty():
            try:
                r_id = requests_to_cancel.get_nowait()
                if r_id in active_requests:
                    logger.info(f"[{r_id}] Canceling request...")
                    try:
                        active_requests[r_id]['handle'].cancel()
                    except Exception as ce:
                        logger.warning(f"[{r_id}] Warning during cancel: {ce}")
                    del active_requests[r_id]
            except Exception as e:
                logger.error(f"!!! Error processing cancellation: {e}")

        # 3. Advance the pipeline
        if active_requests:
            try:
                # Log metrics occasionally
                if request_id_counter % 100 == 0:
                    try:
                        m = pipe.get_metrics()
                        logger.info(f"METRICS: Cache usage: {m.cache_usage:.2f}, Requests (active/scheduled): {m.requests}/{m.scheduled_requests}")
                    except: pass

                logger.debug(f"Calling pipe.step() for {len(active_requests)} active requests. Pipeline busy: {pipe.has_non_finished_requests()}")
                start_step = time.perf_counter()
                pipe.step()
                step_time = (time.perf_counter() - start_step) * 1000
                logger.debug(f"pipe.step() finished in {step_time:.2f}ms")
                
                # 4. Extract tokens and handle completion
                finished_ids = []
                for r_id in list(active_requests.keys()):
                    try:
                        req = active_requests[r_id]
                        handle = req['handle']
                        
                        status_val = handle.get_status()
                        logger.debug(f"[{r_id}] Current status: {status_val.name}")

                        # Read tokens
                        while handle.can_read():
                            res = handle.read()
                            logger.trace(f"[{r_id}] handle.read() returned keys: {list(res.keys())}")
                            
                            for beam_idx, out in res.items():
                                new_ids = out.generated_ids
                                logger.trace(f"[{r_id}] New raw IDs from this step: {new_ids}")
                                
                                if new_ids:
                                    try:
                                        token_text = tokenizer.decode(new_ids)
                                        logger.trace(f"[{r_id}] New token (beam {beam_idx}): {repr(token_text)}")
                                        req['token_queue'].put(token_text)
                                    except Exception as de:
                                        logger.error(f"[{r_id}] Decode error: {de}")
                        
                        # Status check
                        if status_val == ov_genai.GenerationStatus.FINISHED:
                            logger.info(f"[{r_id}] Worker detected finish (FINISHED).")
                            req['token_queue'].put(None)
                            finished_ids.append(r_id)
                        elif status_val == ov_genai.GenerationStatus.CANCEL:
                            logger.info(f"[{r_id}] Worker detected finish (CANCEL).")
                            req['token_queue'].put(None)
                            finished_ids.append(r_id)
                    except Exception as re:
                        logger.error(f"[{r_id}] Error processing handle: {re}")
                        finished_ids.append(r_id)
                
                for r_id in finished_ids:
                    if r_id in active_requests:
                        del active_requests[r_id]
                
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"!!! CRITICAL WORKER STEP ERROR: {e}\n{traceback.format_exc()}")
        else:
            time.sleep(0.01)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "openvino"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    global pipe, tokenizer, global_config
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    try:
        logger.info(f"[{request_id}] Incoming Request:")
        for m in body.messages:
            logger.info(f"  {m.role.upper()}: {m.content[:100]}..." if len(m.content) > 100 else f"  {m.role.upper()}: {m.content}")

        # Prepare prompt
        history = ov_genai.ChatHistory()
        for msg in body.messages:
            history.append({"role": msg.role, "content": msg.content})
        prompt = tokenizer.apply_chat_template(history, True)

        # Merge body config with global defaults
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = body.max_tokens or global_config['max_tokens']
        config.temperature = body.temperature or global_config['temperature']
        config.top_p = body.top_p or global_config['top_p']
        
        if config.temperature > 0:
            config.do_sample = True
        
        if body.stop:
            config.stop_strings = {body.stop} if isinstance(body.stop, str) else set(body.stop)

        token_queue = queue.Queue()
        requests_to_add.put({
            'request_id': request_id,
            'prompt': prompt,
            'config': config,
            'token_queue': token_queue
        })

        created_time = int(time.time())

        async def stream_generator():
            logger.info(f"[{request_id}] Starting stream generator...")
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': body.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            token_count = 0
            full_text = ""
            try:
                while True:
                    if await request.is_disconnected():
                        logger.info(f"[{request_id}] HTTP client disconnected.")
                        requests_to_cancel.put(request_id)
                        break
                    
                    try:
                        token = token_queue.get_nowait()
                        if token is None: 
                            logger.info(f"[{request_id}] Worker signaled completion.")
                            break
                        
                        token_count += 1
                        full_text += token
                        
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": body.model,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"[{request_id}] Error in streaming loop: {e}")
                        break
            finally:
                logger.info(f"[{request_id}] Stream finished. Sent {token_count} tokens.")
                logger.info(f"[{request_id}] Final Response Summary: {full_text[:200]}..." if len(full_text) > 200 else f"[{request_id}] Final Response: {full_text}")

            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        if body.stream:
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            full_text = ""
            logger.info(f"[{request_id}] Processing blocking request...")
            while True:
                if await request.is_disconnected():
                    logger.info(f"[{request_id}] Request canceled via disconnect.")
                    requests_to_cancel.put(request_id)
                    break
                try:
                    token = token_queue.get(timeout=0.01)
                    if token is None: break
                    full_text += token
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"[{request_id}] Error in blocking loop: {e}")
                    break
            
            logger.info(f"[{request_id}] Blocking request finished. Length: {len(full_text)}")
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created_time,
                "model": body.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
            }
    except Exception as e:
        logger.error(f"[{request_id}] Error in chat_completions: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="OpenVINO OpenAI Compatible API Server")
        parser.add_argument("--model", type=str, required=True, help="Path to OpenVINO model directory")
        parser.add_argument("--device", type=str, default="GPU", help="Inference device (GPU, CPU, etc.)")
        parser.add_argument("--port", type=int, default=8000, help="Server port")
        parser.add_argument("--n_ctx", type=int, default=32768, help="Context window size")
        parser.add_argument("--batch_size", type=int, default=128, help="Max num batched tokens (prefill chunk)")
        parser.add_argument("--temperature", type=float, default=0.7, help="Default temperature")
        parser.add_argument("--top_p", type=float, default=0.9, help="Default top_p")
        parser.add_argument("--max_tokens", type=int, default=4096, help="Default max new tokens")
        parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
        args = parser.parse_args()

        # Update logging level from args
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

        model_name = args.model
        logger.info(f"Server starting. Model: {args.model}, Device: {args.device}, Context: {args.n_ctx}, Max tokens: {args.max_tokens}")
        
        global_config = {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens
        }

        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.max_num_batched_tokens = args.batch_size
        scheduler_config.num_kv_blocks = args.n_ctx // 16
        scheduler_config.dynamic_split_fuse = True
        scheduler_config.enable_prefix_caching = True
        
        logger.info("Initializing ContinuousBatchingPipeline...")
        pipe = ov_genai.ContinuousBatchingPipeline(args.model, scheduler_config, args.device)
        tokenizer = pipe.get_tokenizer()

        logger.info("Starting worker thread...")
        worker_thread = threading.Thread(target=pipeline_worker, daemon=True)
        worker_thread.start()

        import uvicorn
        logger.info(f"Starting Uvicorn on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except Exception as e:
        logger.error(f"FATAL STARTUP ERROR: {e}\n{traceback.format_exc()}")

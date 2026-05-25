import asyncio
import argparse
import json
import time
import uuid
import threading
import queue
import datetime
import traceback
from typing import List, Optional, Union, Dict, Any

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="OpenVINO GPU API (Continuous Batching)")

# Global State
pipe = None
tokenizer = None
model_name = ""

# Data structure to track requests
active_requests = {}
requests_to_add = queue.Queue()
requests_to_cancel = queue.Queue()
request_id_counter = 0

def log(msg):
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{now}] {msg}", flush=True)
    except:
        pass # Fallback if logging itself fails

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log(f"GLOBAL ERROR: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc), "type": "server_error"}}
    )

def pipeline_worker():
    global pipe, active_requests, request_id_counter, tokenizer
    log("Pipeline worker started.")
    
    while True:
        # 1. Add new requests to the pipeline
        while not requests_to_add.empty():
            try:
                req_data = requests_to_add.get_nowait()
                r_id = req_data['request_id']
                internal_id = request_id_counter
                request_id_counter += 1
                log(f"[{r_id}] Adding request to pipeline with internal_id {internal_id}...")
                
                handle = pipe.add_request(internal_id, req_data['prompt'], req_data['config'])
                active_requests[r_id] = {
                    'handle': handle,
                    'token_queue': req_data['token_queue'],
                    'request_id': r_id,
                    'processed_tokens': 0
                }
            except Exception as e:
                log(f"!!! Error adding request: {e}\n{traceback.format_exc()}")

        # 2. Process cancellations
        while not requests_to_cancel.empty():
            try:
                r_id = requests_to_cancel.get_nowait()
                if r_id in active_requests:
                    log(f"[{r_id}] Canceling request...")
                    try:
                        active_requests[r_id]['handle'].cancel()
                    except Exception as ce:
                        log(f"[{r_id}] Warning during cancel: {ce}")
                    del active_requests[r_id]
            except Exception as e:
                log(f"!!! Error processing cancellation: {e}")

        # 3. Advance the pipeline
        if active_requests:
            try:
                # Throttled metrics log
                if request_id_counter % 50 == 0:
                    try:
                        m = pipe.get_metrics()
                        log(f"METRICS: Cache usage: {m.cache_usage:.2f}, Requests (active/scheduled): {m.requests}/{m.scheduled_requests}")
                    except: pass

                log(f"DEBUG: Calling pipe.step() for {len(active_requests)} active requests. Pipeline busy: {pipe.has_non_finished_requests()}")
                start_step = time.perf_counter()
                pipe.step()
                step_time = (time.perf_counter() - start_step) * 1000
                log(f"DEBUG: pipe.step() finished in {step_time:.2f}ms")
                
                # 4. Extract tokens and handle completion
                finished_ids = []
                for r_id in list(active_requests.keys()):
                    try:
                        req = active_requests[r_id]
                        handle = req['handle']
                        
                        status_val = handle.get_status()
                        log(f"[{r_id}] Current status: {status_val.name}")

                        # Read tokens
                        while handle.can_read():
                            res = handle.read()
                            log(f"DEBUG: [{r_id}] handle.read() returned keys: {list(res.keys())}")
                            
                            for beam_idx, out in res.items():
                                new_ids = out.generated_ids
                                log(f"DEBUG: [{r_id}] New raw IDs from this step: {new_ids}")
                                
                                if new_ids:
                                    try:
                                        token_text = tokenizer.decode(new_ids)
                                        log(f"[{r_id}] New token (beam {beam_idx}): {repr(token_text)}")
                                        req['token_queue'].put(token_text)
                                    except Exception as de:
                                        log(f"[{r_id}] Decode error: {de}")
                        
                        # Status check
                        if status_val == ov_genai.GenerationStatus.FINISHED:
                            log(f"[{r_id}] Worker detected finish (FINISHED).")
                            req['token_queue'].put(None)
                            finished_ids.append(r_id)
                        elif status_val == ov_genai.GenerationStatus.CANCEL:
                            log(f"[{r_id}] Worker detected finish (CANCEL).")
                            req['token_queue'].put(None)
                            finished_ids.append(r_id)
                    except Exception as re:
                        log(f"[{r_id}] Error processing handle: {re}")
                        finished_ids.append(r_id)
                
                for r_id in finished_ids:
                    if r_id in active_requests:
                        del active_requests[r_id]
                
                time.sleep(0.001)
            except Exception as e:
                log(f"!!! CRITICAL WORKER STEP ERROR: {e}\n{traceback.format_exc()}")
        else:
            time.sleep(0.01)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 4096
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
    global pipe, tokenizer
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    try:
        log(f"[{request_id}] Incoming Request:")
        for m in body.messages:
            log(f"  {m.role.upper()}: {m.content[:100]}..." if len(m.content) > 100 else f"  {m.role.upper()}: {m.content}")

        # Prepare prompt
        history = ov_genai.ChatHistory()
        for msg in body.messages:
            history.append({"role": msg.role, "content": msg.content})
        prompt = tokenizer.apply_chat_template(history, True)

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = body.max_tokens
        config.temperature = body.temperature
        config.top_p = body.top_p
        if body.temperature > 0:
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
            log(f"[{request_id}] Starting stream generator...")
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': body.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            token_count = 0
            full_text = ""
            try:
                while True:
                    if await request.is_disconnected():
                        log(f"[{request_id}] HTTP client disconnected.")
                        requests_to_cancel.put(request_id)
                        break
                    
                    try:
                        token = token_queue.get_nowait()
                        if token is None: 
                            log(f"[{request_id}] Worker signaled completion.")
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
                        log(f"[{request_id}] Error in streaming loop: {e}")
                        break
            finally:
                log(f"[{request_id}] Stream finished. Sent {token_count} tokens.")
                log(f"[{request_id}] Final Response Summary: {full_text[:200]}..." if len(full_text) > 200 else f"[{request_id}] Final Response: {full_text}")

            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        if body.stream:
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            full_text = ""
            log(f"[{request_id}] Processing blocking request...")
            while True:
                if await request.is_disconnected():
                    log(f"[{request_id}] Request canceled via disconnect.")
                    requests_to_cancel.put(request_id)
                    break
                try:
                    token = token_queue.get_nowait()
                    if token is None: break
                    full_text += token
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    log(f"[{request_id}] Error in blocking loop: {e}")
                    break
            
            log(f"[{request_id}] Blocking request finished. Length: {len(full_text)}")
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created_time,
                "model": body.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
            }
    except Exception as e:
        log(f"[{request_id}] Error in chat_completions: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--device", type=str, default="GPU")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--n_ctx", type=int, default=32768)
        args = parser.parse_args()

        model_name = args.model
        log(f"Server starting. Model: {args.model}, Device: {args.device}, Context: {args.n_ctx}")
        
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.max_num_batched_tokens = 128
        scheduler_config.num_kv_blocks = args.n_ctx // 16
        scheduler_config.dynamic_split_fuse = True
        
        log("Initializing ContinuousBatchingPipeline...")
        pipe = ov_genai.ContinuousBatchingPipeline(args.model, scheduler_config, args.device)
        tokenizer = pipe.get_tokenizer()

        log("Starting worker thread...")
        worker_thread = threading.Thread(target=pipeline_worker, daemon=True)
        worker_thread.start()

        import uvicorn
        log(f"Starting Uvicorn on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except Exception as e:
        log(f"FATAL STARTUP ERROR: {e}\n{traceback.format_exc()}")

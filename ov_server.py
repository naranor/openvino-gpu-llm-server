import argparse
import json
import time
import uuid
from typing import List, Optional, Union, Dict, Any

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="OpenVINO OpenAI Compatible API")

# Global variables for the model and pipeline
pipe = None
model_name = ""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openvino"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Prepare chat history
    # openvino_genai.LLMPipeline handles chat history if we use its ChatHistory class,
    # but for a stateless API, we can just pass the messages.
    # Note: openvino_genai expect a specific format for some models.
    
    # Use ChatHistory for better compatibility with chat models
    history = ov_genai.ChatHistory()
    for msg in request.messages:
        history.append({"role": msg.role, "content": msg.content})

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    # Set performance hint if possible via config or env
    
    if request.stop:
        if isinstance(request.stop, str):
            config.stop_strings = {request.stop}
        else:
            config.stop_strings = set(request.stop)

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if request.stream:
        # Real streaming implementation for OpenVINO GenAI
        async def async_stream_generator():
            # Send initial role
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # We use a custom streamer to yield tokens
            import queue
            q = queue.Queue()

            def streamer_callback(subword):
                q.put(subword)
                return ov_genai.StreamingStatus.RUNNING

            # We need to run generation in a separate thread to not block the event loop
            import threading
            def run_gen():
                try:
                    pipe.generate(history, config, streamer_callback)
                except Exception as e:
                    print(f"Error during generation: {e}")
                q.put(None) # Signal end
            
            thread = threading.Thread(target=run_gen)
            thread.start()
            
            while True:
                try:
                    import asyncio
                    # Use a small timeout to keep the loop responsive
                    token = q.get(timeout=0.01)
                    if token is None:
                        break
                    
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.01)
            
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(async_stream_generator(), media_type="text/event-stream")

    else:
        # Non-streaming
        start_time = time.time()
        # Pass history directly
        result = pipe.generate(history, config)
        end_time = time.time()
        
        output_text = str(result)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to OpenVINO model directory")
    parser.add_argument("--device", type=str, default="GPU", help="Device to run inference on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    model_name = args.model
    print(f"Loading model from {args.model} to {args.device}...")
    
    # Load model
    # LLMPipeline handles tokenizer and model loading
    try:
        print(f"Loading model from {args.model} to {args.device}...")
        pipe = ov_genai.LLMPipeline(args.model, args.device)
    except Exception as e:
        print(f"Failed to load model on {args.device}: {e}")
        if args.device != "CPU":
            print("Falling back to CPU...")
            pipe = ov_genai.LLMPipeline(args.model, "CPU")
        else:
            raise e
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

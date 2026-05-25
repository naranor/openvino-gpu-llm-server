import argparse
import json
import time
import uuid
from typing import List, Optional, Union, Dict, Any

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="OpenVINO GPU API (FP16)")

# Global variables
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

    history = ov_genai.ChatHistory()
    for msg in request.messages:
        history.append({"role": msg.role, "content": msg.content})

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    if request.stop:
        config.stop_strings = {request.stop} if isinstance(request.stop, str) else set(request.stop)

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if request.stream:
        async def async_stream_generator():
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            import queue
            import threading
            q = queue.Queue()

            def streamer_callback(subword):
                q.put(subword)
                return ov_genai.StreamingStatus.RUNNING

            def run_gen():
                try:
                    pipe.generate(history, config, streamer_callback)
                except Exception as e:
                    print(f"Error: {e}")
                q.put(None)
            
            threading.Thread(target=run_gen).start()
            
            while True:
                try:
                    token = q.get(timeout=0.01)
                    if token is None: break
                    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
                except queue.Empty:
                    import asyncio
                    await asyncio.sleep(0.01)
            
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(async_stream_generator(), media_type="text/event-stream")

    else:
        result = pipe.generate(history, config)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": str(result)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_name = args.model
    print(f"Loading model {args.model} to {args.device}...")
    pipe = ov_genai.LLMPipeline(args.model, args.device)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

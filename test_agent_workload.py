import requests
import json
import time

def simulate_agent_request():
    url = "http://localhost:8000/v1/chat/completions"
    
    # 1. Generate a realistic project context
    files = {
        "main.py": "import os\nfrom core import process\n\ndef run():\n    data = [1, 2, 3]\n    print(process(data))",
        "core.py": "def process(data):\n    return [x * 2 for x in data]",
        "utils.py": "def helper():\n    pass\n" * 20
    }
    
    large_system_content = "You are a professional software architect. Here are the files of your project:\n"
    for name, content in files.items():
        large_system_content += f"\n--- {name} ---\n{content}\n"
    
    large_system_content += "\nAlways begin your response with <thought> tags to explain your reasoning process."
    
    # 2. Generate message history (~1500 tokens)
    history = [
        {"role": "system", "content": large_system_content},
        {"role": "user", "content": "I need to implement a new feature. Here is the current state of the app:\n" + "def old_func():\n    pass\n" * 150},
        {"role": "assistant", "content": "I see. I will help you with that. What is the specific logic?"},
        {"role": "user", "content": "Please write a complex quicksort implementation in Python with detailed comments and error handling. Explain your reasoning first."}
    ]

    payload = {
        "model": "DeepSeek-R1-1.5B",
        "messages": history,
        "temperature": 0.6,
        "max_tokens": 512,
        "stream": True
    }

    print(f"Sending request with {len(json.dumps(history)) // 4} estimated tokens...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=300)
        response.raise_for_status()
        
        first_token_time = None
        full_response = ""
        
        print("Waiting for response...")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    content = line_str[6:]
                    if content == "[DONE]":
                        break
                    
                    data = json.loads(content)
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            if first_token_time is None:
                                first_token_time = time.time()
                                print(f"Time to first token: {first_token_time - start_time:.2f}s")
                            
                            text = delta["content"]
                            full_response += text
                            print(text, end="", flush=True)
        
        end_time = time.time()
        print(f"\n\nTotal time: {end_time - start_time:.2f}s")
        print(f"Total tokens generated (approx): {len(full_response) // 4}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    simulate_agent_request()

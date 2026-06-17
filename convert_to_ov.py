import torch
import openvino as ov
from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
import os
from pathlib import Path

def convert_model(model_id, output_dir, precision="int4"):
    print(f"Loading model from {model_id}...")
    
    import sys
    import importlib.util
    model_path = Path(model_id).resolve()
    sys.path.insert(0, str(model_path))
    
    try:
        print("Importing local modeling_dream.py...")
        # We try to import DreamModel specifically if it exists
        if (model_path / "modeling_dream.py").exists():
            spec = importlib.util.spec_from_file_location("modeling_dream", model_path / "modeling_dream.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model_class = getattr(mod, "DreamModel", AutoModel)
        else:
            model_class = AutoModel
            
        print(f"Using model class: {model_class}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        
        model = model_class.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            local_files_only=True,
            use_cache=False
        )
    except Exception as e:
        print(f"Loading failed: {e}. Falling back to standard AutoModel...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            local_files_only=True,
            use_cache=False
        )
    
    model.eval()

    print("Converting to OpenVINO IR... (Using static shape tracing)")
    seq_len = 256
    dummy_input = {
        "input_ids": torch.ones((1, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.float16)
    }
    
    print("Step 1: convert_model starting...")
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=dummy_input)
        
    print("Step 2: Reshaping to dynamic...")
    ov_model.reshape({
        "input_ids": ov.PartialShape([1, -1]),
        "attention_mask": ov.PartialShape([1, -1])
    })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Step 3: Saving base IR to {output_path}...")
    ov.save_model(ov_model, output_path / "model.xml", compress_to_fp16=(precision != "int4"))
    tokenizer.save_pretrained(output_path)

    if precision == "int4":
        print("Applying INT4 quantization using NNCF...")
        import nncf
        quantized_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128)
        ov.save_model(quantized_model, output_path / "model.xml")
    elif precision == "int8":
        print("Applying INT8 quantization using NNCF...")
        import nncf
        quantized_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        print("Saving quantized model...")
        ov.save_model(quantized_model, output_path / "model.xml")

    print(f"Success! Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--precision", type=str, default="int4")
    args = parser.parse_args()
    
    convert_model(args.model, args.output, args.precision)

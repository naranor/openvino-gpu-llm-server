import argparse
import os
from huggingface_hub import HfApi

def upload_model(local_dir, repo_id, token):
    if not os.path.exists(local_dir):
        print(f"Error: Directory '{local_dir}' does not exist.")
        return

    print(f"Preparing to upload contents of '{local_dir}' to 'https://huggingface.co/{repo_id}'")
    
    api = HfApi()
    
    # Check token/login
    if token:
        import huggingface_hub
        huggingface_hub.login(token=token)
        
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print(f"Repository '{repo_id}' created or already exists.")
        
        print("Uploading files... This may take a while for large model.bin files.")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial commit: Upload INT8 OpenVINO converted model"
        )
        print(f"Upload complete! Model available at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload an OpenVINO model directory to Hugging Face")
    parser.add_argument("--dir", type=str, required=True, help="Local directory containing the model files (e.g., models/DiffuCoder-7B-Instruct-ov-int8)")
    parser.add_argument("--repo", type=str, required=True, help="Target Hugging Face repository ID (e.g., your_username/DiffuCoder-7B-Instruct-ov-int8)")
    parser.add_argument("--token", type=str, help="Hugging Face User Access Token (Write permission required). If omitted, relies on huggingface-cli login.")
    
    args = parser.parse_args()
    upload_model(args.dir, args.repo, args.token)

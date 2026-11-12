"""
Automated Hugging Face Dataset Publisher
========================================
Uploads the prepared air-quality-timeseries dataset (train.csv, validation.csv, test.csv, README.md)
to a public Hugging Face Dataset repository.
"""

import os
import sys
from huggingface_hub import HfApi, login

REPO_NAME = "air-quality-timeseries"
DATASET_DIR = "HF_Air_Quality_Dataset"

def publish_dataset():
    api = HfApi()
    
    # 1. Check authentication
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[OK] Authenticated with Hugging Face as user: '{username}'")
    except Exception as e:
        print("[!] Not authenticated with Hugging Face.")
        print("Please authenticate using one of the following methods:")
        print("  1. Run in PowerShell/Terminal:  huggingface-cli login")
        print("     or:  python -m huggingface_hub.cli.hf auth login")
        print("  2. Or set the HF_TOKEN environment variable.")
        print("\nNote: Generate your token at: https://huggingface.co/settings/tokens (Role: Write)")
        return False

    repo_id = f"{username}/{REPO_NAME}"
    print(f"[*] Target Public Dataset Repository: https://huggingface.co/datasets/{repo_id}")
    
    # 2. Create the public repository if it does not already exist
    try:
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"[OK] Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"[!] Error creating repository: {e}")
        return False

    # 3. Upload files
    print(f"[*] Uploading files from '{DATASET_DIR}'...")
    try:
        api.upload_folder(
            folder_path=DATASET_DIR,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initial release of UCI Air Quality Time-Series Dataset (70/10/20 train/val/test splits)",
            ignore_patterns=["*.pyc", "__pycache__"]
        )
        print(f"[OK] All dataset files successfully uploaded!")
        print(f"\n=======================================================")
        print(f"PUBLIC REPOSITORY LINK FOR SUBMISSION:")
        print(f"https://huggingface.co/datasets/{repo_id}")
        print(f"=======================================================")
        return True
    except Exception as e:
        print(f"[!] Upload error: {e}")
        return False

if __name__ == "__main__":
    publish_dataset()

"""
download_model.py — Download model from Hugging Face if not present locally.

Downloads the model specified in .env (MODEL_NAME) into MODELS_FOLDER/base_model/.
If the model already exists locally, the download is skipped.

Directory Structure:
    models/
        base_model/          # Downloaded base models go here
            Qwen2.5-3B-Instruct/
        fine_tuned/          # Fine-tuned models saved here by train_grpo.py
            Qwen2.5-3B-Instruct_v0/
            Qwen2.5-3B-Instruct_v1/

Requirements:
    pip install huggingface_hub python-dotenv

Usage:
    python training/download_model.py

    # or specify another model
    python training/download_model.py Qwen/Qwen2.5-3B-Instruct
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def download_model(hf_model_id: Optional[str] = None):
    """
    Download a Hugging Face model repository to a local directory.
    Saves to models/base_model/ folder.
    """

    # Read configuration
    if hf_model_id is None:
        hf_model_id = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

    models_folder = os.getenv("MODELS_FOLDER", "./models")

    model_name = hf_model_id.split("/")[-1]
    local_path = Path(models_folder) / "base_model" / model_name

    # Skip download if model already exists
    if (local_path / "config.json").exists():
        print(f"✓ Model already exists: {local_path}")
        return str(local_path)

    print("📥 Downloading model")
    print(f"   Hugging Face ID : {hf_model_id}")
    print(f"   Local path      : {local_path}")
    print()

    try:
        # Create models/base_model/ directory structure if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Directory ready: {local_path.parent}")
        print()

        snapshot_download(
            repo_id=hf_model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN"),
            resume_download=True,
        )

        print()
        print("✅ Model downloaded successfully")
        print(f"   Location: {local_path}")

        return str(local_path)

    except Exception as e:
        print()
        print("❌ Error downloading model")
        print(e)
        print()
        print("Common issues:")
        print("- Invalid model ID")
        print("- Network connection problems")
        print("- Private model without authentication")
        print("- Disk space insufficient")
        print()
        print("If authentication is required:")
        print("    huggingface-cli login")

        sys.exit(1)


if __name__ == "__main__":

    hf_model_id = sys.argv[1] if len(sys.argv) > 1 else None

    if hf_model_id:
        print("📋 Using custom model")
        print(f"   {hf_model_id}")
        print()

    else:
        print("📋 Using model from .env")
        print(f"   MODEL_NAME={os.getenv('MODEL_NAME')}")
        print()

    download_model(hf_model_id)
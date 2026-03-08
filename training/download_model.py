"""
download_model.py — Download models from Hugging Face if not present locally.

Downloads both:
  1. Base model (MODEL_NAME) for training
  2. Judge model (JUDGE_MODEL) for scoring

into MODELS_FOLDER/base_model/. If models already exist locally, downloads are skipped.

Directory Structure:
    models/
        base_model/          # Downloaded base models go here
            Qwen2.5-3B-Instruct/
            Qwen2.5-7B-Instruct/
        fine_tuned/          # Fine-tuned models saved here by train_grpo.py
            Qwen2.5-3B-Instruct_v0/
            Qwen2.5-3B-Instruct_v1/

Requirements:
    pip install huggingface_hub python-dotenv

Usage:
    python training/download_model.py
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


def download_model(hf_model_id: str, model_type: str = "model"):
    """
    Download a Hugging Face model repository to a local directory.
    Saves to models/base_model/ folder.

    Args:
        hf_model_id: Hugging Face model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
        model_type: Description of model type for logging (e.g., "base", "judge")

    Returns:
        str: Local path to downloaded model
    """
    models_folder = os.getenv("MODELS_FOLDER", "./models")
    model_name = hf_model_id.split("/")[-1]
    local_path = Path(models_folder) / "base_model" / model_name

    # Skip download if model already exists
    if (local_path / "config.json").exists():
        print(f"✓ {model_type.capitalize()} model already exists: {local_path}")
        return str(local_path)

    print(f"📥 Downloading {model_type} model")
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
        print(f"✅ {model_type.capitalize()} model downloaded successfully")
        print(f"   Location: {local_path}")
        print()

        return str(local_path)

    except Exception as e:
        print()
        print(f"❌ Error downloading {model_type} model")
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
    print("=" * 60)
    print("DOWNLOADING MODELS FOR APEX TRAINING")
    print("=" * 60)
    print()

    # Get model IDs from .env or command line
    base_model_id = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    judge_model_id = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # Allow override via command line
    if len(sys.argv) > 1:
        base_model_id = sys.argv[1]
    if len(sys.argv) > 2:
        judge_model_id = sys.argv[2]

    print("📋 Configuration:")
    print(f"   Base model  (for training) : {base_model_id}")
    print(f"   Judge model (for scoring)  : {judge_model_id}")
    print()

    # Download base model
    print("=" * 60)
    print("1. DOWNLOADING BASE MODEL")
    print("=" * 60)
    download_model(base_model_id, model_type="base")

    # Download judge model (skip if same as base model)
    if judge_model_id != base_model_id:
        print("=" * 60)
        print("2. DOWNLOADING JUDGE MODEL")
        print("=" * 60)
        download_model(judge_model_id, model_type="judge")
    else:
        print("=" * 60)
        print("2. JUDGE MODEL")
        print("=" * 60)
        print("✓ Judge model is same as base model (already downloaded)")
        print()

    print("=" * 60)
    print("✅ ALL MODELS READY")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Start the APEX server:")
    print("     cd apex_env && uvicorn server.app:app --host 0.0.0.0 --port 8001")
    print()
    print("  2. Run training:")
    print("     python training/train_grpo.py")
    print()
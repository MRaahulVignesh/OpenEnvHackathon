"""
download_model.py — Download models from Hugging Face if not present locally
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def download_model(hf_model_id: str, model_type: str = "model") -> str:
    """
    Download a Hugging Face model repository if not already present locally.
    """

    models_folder = os.getenv("MODELS_FOLDER", "./models")
    model_name = hf_model_id.split("/")[-1]
    local_path = Path(models_folder) / "base_model" / model_name

    if (local_path / "config.json").exists():
        print(f"{model_type.capitalize()} model already present → {local_path}")
        return str(local_path)

    print(f"Downloading {model_type} model: {hf_model_id}")

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=hf_model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN"),
            resume_download=True,
        )

        print(f"Saved to: {local_path}\n")
        return str(local_path)

    except Exception as e:
        print(f"Failed to download {model_type} model: {e}")
        sys.exit(1)


def main():
    base_model_id = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    judge_model_id = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    if len(sys.argv) > 1:
        base_model_id = sys.argv[1]
    if len(sys.argv) > 2:
        judge_model_id = sys.argv[2]

    print("Model setup")
    print(f"Base model : {base_model_id}")
    print(f"Judge model: {judge_model_id}\n")

    download_model(base_model_id, "base")

    if judge_model_id != base_model_id:
        download_model(judge_model_id, "judge")


if __name__ == "__main__":
    main()
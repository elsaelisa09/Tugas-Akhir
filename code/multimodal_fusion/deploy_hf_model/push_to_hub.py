import argparse
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_FILES = [
    "model_config.json",
    "modeling_archA.py",
    "inference.py",
]


def push_model(repo_id, checkpoint_path, local_dir, private=False):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    local_dir = Path(local_dir)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    for filename in DEFAULT_FILES:
        path = local_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File wajib tidak ditemukan: {path}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=checkpoint_path.name,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Selesai upload model ke https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model archA ke Hugging Face Model Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="contoh: username/nama-repo-model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../output/bestmodel_mlA-class-imbalance-FL.pth",
        help="path checkpoint .pth",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=".",
        help="folder deploy_hf_model",
    )
    parser.add_argument("--private", action="store_true", help="buat repo model private")
    args = parser.parse_args()

    push_model(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint_path,
        local_dir=args.local_dir,
        private=args.private,
    )

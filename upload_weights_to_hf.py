"""
Upload ESM3 and SaProt weight files to a private Hugging Face model repo.

Usage:
    python upload_weights_to_hf.py --repo yehlincho/ESM3-SaProt-dG-weights
    python upload_weights_to_hf.py --repo yehlincho/ESM3-SaProt-dG-weights --token hf_xxx
"""

import argparse
import os
from huggingface_hub import HfApi

WEIGHT_DIRS = [
    "esm3dg_weights",
    "saprotdg_weights",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. yehlincho/ESM3-SaProt-dG-weights")
    parser.add_argument("--token", default=None, help="HF write token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable")

    api = HfApi()

    # Create repo if it doesn't exist (private by default)
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=token,
    )
    print(f"Repo ready: https://huggingface.co/{args.repo}")

    for weight_dir in WEIGHT_DIRS:
        if not os.path.isdir(weight_dir):
            print(f"Skipping {weight_dir} (not found)")
            continue
        print(f"Uploading {weight_dir}/ ...")
        ckpt_files = [f for f in os.listdir(weight_dir) if f.endswith(".ckpt")]
        for fname in ckpt_files:
            local_path = os.path.join(weight_dir, fname)
            repo_path = f"{weight_dir}/{fname}"
            print(f"  {repo_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=args.repo,
                repo_type="model",
                token=token,
            )
        print(f"Done: {weight_dir}/")

    print("\nAll weights uploaded.")
    print(f"Download example:\n")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  hf_hub_download(repo_id='{args.repo}', filename='esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt', token=HF_TOKEN)")

if __name__ == "__main__":
    main()

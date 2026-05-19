#!/usr/bin/env bash
# Download fine-tuned weights from Yehlin/absolute-stability on Hugging Face.
# Usage:
#   bash download_weights.sh                   # public or already-logged-in
#   bash download_weights.sh --token hf_xxx    # supply token explicitly
set -euo pipefail

REPO_ID="Yehlin/absolute-stability"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN=""

# Parse --token argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        --token|-t) TOKEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Files to download and where to put them
declare -A FILES=(
    ["esm3dg_weights/ESM3dG_weights_1_lora.ckpt"]="esm3dg_weights/ESM3dG_weights_1_lora.ckpt"
    ["esm3dg_weights/ESM3dG_weights_2_lora.ckpt"]="esm3dg_weights/ESM3dG_weights_2_lora.ckpt"
    ["esm3dg_weights/ESM3dG_weights_3_lora.ckpt"]="esm3dg_weights/ESM3dG_weights_3_lora.ckpt"
    ["esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt"]="esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt"
    ["esm3dg_weights/ESM3dG_weights_augmented_2_lora.ckpt"]="esm3dg_weights/ESM3dG_weights_augmented_2_lora.ckpt"
    ["esm3dg_weights/ESM3dG_weights_augmented_3_lora.ckpt"]="esm3dg_weights/ESM3dG_weights_augmented_3_lora.ckpt"
    ["saprotdg_weights/SaProtdG_weights_1_lora.ckpt"]="saprotdg_weights/SaProtdG_weights_1_lora.ckpt"
    ["saprotdg_weights/SaProtdG_weights_2_lora.ckpt"]="saprotdg_weights/SaProtdG_weights_2_lora.ckpt"
    ["saprotdg_weights/SaProtdG_weights_3_lora.ckpt"]="saprotdg_weights/SaProtdG_weights_3_lora.ckpt"
    ["saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt"]="saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt"
    ["saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt"]="saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt"
    ["saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt"]="saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt"
)

echo "Downloading weights from ${REPO_ID} ..."
mkdir -p "${SCRIPT_DIR}/esm3dg_weights" "${SCRIPT_DIR}/saprotdg_weights"

# ── Try hf / huggingface-cli ──────────────────────────────────────────────────
HF_CMD=""
command -v hf              &>/dev/null && HF_CMD="hf"
command -v huggingface-cli &>/dev/null && [[ -z "$HF_CMD" ]] && HF_CMD="huggingface-cli"

if [[ -n "$HF_CMD" ]]; then
    echo "Using $HF_CMD"
    TOKEN_ARGS=()
    [[ -n "$TOKEN" ]] && TOKEN_ARGS=(--token "$TOKEN")

    for repo_path in "${!FILES[@]}"; do
        local_path="${SCRIPT_DIR}/${FILES[$repo_path]}"
        if [[ -f "$local_path" ]]; then
            echo "  [skip] $repo_path already exists"
            continue
        fi
        echo "  Downloading $repo_path ..."
        "$HF_CMD" download "$REPO_ID" "$repo_path" \
            --local-dir "$SCRIPT_DIR" \
            "${TOKEN_ARGS[@]}"
    done

# ── Fall back to Python huggingface_hub ──────────────────────────────────────
else
    echo "huggingface-cli not found — falling back to Python huggingface_hub"
    python3 - <<PYEOF
import os, sys
from huggingface_hub import hf_hub_download

repo_id = "${REPO_ID}"
script_dir = "${SCRIPT_DIR}"
token = "${TOKEN}" or None

files = [
    "esm3dg_weights/ESM3dG_weights_1_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_2_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_3_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_augmented_2_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_augmented_3_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_1_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_2_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_3_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt",
]

for f in files:
    dest = os.path.join(script_dir, f)
    if os.path.exists(dest):
        print(f"  [skip] {f} already exists")
        continue
    print(f"  Downloading {f} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=f,
        local_dir=script_dir,
        token=token,
    )

print("Done.")
PYEOF
fi

echo ""
echo "All weights are in place:"
echo "  esm3dg_weights/   — $(ls "${SCRIPT_DIR}/esm3dg_weights/"*.ckpt 2>/dev/null | wc -l) files"
echo "  saprotdg_weights/ — $(ls "${SCRIPT_DIR}/saprotdg_weights/"*.ckpt 2>/dev/null | wc -l) files"

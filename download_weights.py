import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent
DIFFUSIONPEN_DIR = BASE_DIR / "DiffusionPen"

print(f"Base Directory: {BASE_DIR}")
print(f"DiffusionPen Directory: {DIFFUSIONPEN_DIR}")

# 1. Download weights from konnik/DiffusionPen
print("\n=== 1/4: Downloading konnik/DiffusionPen weights ===")
for subpath in [
    "diffusionpen_iam_model_path/models/ckpt.pt",
    "diffusionpen_iam_model_path/models/ema_ckpt.pt",
    "diffusionpen_iam_model_path/models/optim.pt",
    "style_models/iam_style_diffusionpen.pth",
    "style_models/iam_style_diffusionpen_class.pth",
    "style_models/iam_style_diffusionpen_triplet.pth",
    "style_models/gnhk_style_diffusionpen.pth",
]:
    target_file = DIFFUSIONPEN_DIR / subpath
    if not target_file.exists():
        print(f"Downloading {subpath}...")
        hf_hub_download(
            repo_id="konnik/DiffusionPen",
            filename=subpath,
            local_dir=str(DIFFUSIONPEN_DIR),
            local_dir_use_symlinks=False
        )
    else:
        print(f"Already exists: {subpath}")

# 2. Download stable-diffusion-v1-5 vae and scheduler
print("\n=== 2/4: Downloading stable-diffusion-v1-5 (vae & scheduler) ===")
sd_dir = DIFFUSIONPEN_DIR / "stable-diffusion-v1-5"
sd_dir.mkdir(parents=True, exist_ok=True)

patterns = ["vae/*", "scheduler/*"]
print("Downloading vae and scheduler components...")
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    allow_patterns=patterns,
    local_dir=str(sd_dir),
    local_dir_use_symlinks=False
)

# 3. Pre-cache google/canine-c
print("\n=== 3/4: Pre-caching google/canine-c ===")
from transformers import CanineTokenizer, CanineModel
CanineTokenizer.from_pretrained("google/canine-c")
CanineModel.from_pretrained("google/canine-c")
print("Canine-c cached successfully.")

# 4. Pre-cache microsoft/trocr-large-handwritten
print("\n=== 4/4: Pre-caching microsoft/trocr-large-handwritten ===")
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
print("TrOCR cached successfully.")

print("\n=== All weights and models downloaded successfully! ===")

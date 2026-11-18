#!/usr/bin/env python3
"""
2_finetune_model.py

Fine-Tuning Orchestrator for DiffusionPen:
1. Stage 1: Fine-tunes the Style Encoder (MobileNetV2 feature backbone) on writer_0
   so the style extractor locks onto your handwriting stroke geometry and letter shapes.
2. Stage 2: Resumes fine-tuning the DiffusionPen UNet from the pretrained
   diffusionpen_iam_model_path checkpoint for ~150 epochs using the fine-tuned
   style encoder weights (batch size 16/32 suitable for RTX 3090 24GB VRAM).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSIONPEN_DIR = SCRIPT_DIR / "DiffusionPen"
DATASET_DIR = SCRIPT_DIR / "data" / "processed_dataset"
CACHE_DIR = SCRIPT_DIR / ".cache"

# Set up environment variables
user_hf = Path("C:/Users/Sami-PC/.cache/huggingface")
if (user_hf / "hub" / "models--google--canine-c").exists():
    os.environ["HF_HOME"] = str(user_hf)
else:
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))

os.environ.setdefault("USERPROFILE", str(CACHE_DIR))
os.environ.setdefault("HOME", str(CACHE_DIR))
os.environ.setdefault("PADDLE_HOME", str(CACHE_DIR / "paddle"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def check_dataset_exists(dataset_dir: Path) -> bool:
    """Verify that processed dataset with words and labels.txt exists."""
    labels_file = dataset_dir / "labels.txt"
    words_dir = dataset_dir / "words"
    if not labels_file.exists():
        print(f"[Error] Labels file not found: {labels_file}")
        print("Please run `python 1_prepare_dataset.py` first to create the dataset.")
        return False
    if not words_dir.exists() or len(list(words_dir.glob("*.png"))) == 0:
        print(f"[Error] No word crops found in: {words_dir}")
        print("Please run `python 1_prepare_dataset.py` first.")
        return False
    return True


def run_stage_1_style_encoder(args, python_exe: str):
    """
    Stage 1: Fine-tune the Style Encoder on writer_0.
    """
    print("\n" + "=" * 65)
    print(" STAGE 1: Fine-Tuning Style Encoder (MobileNetV2)")
    print("=" * 65)
    
    style_save_dir = DIFFUSIONPEN_DIR / "style_models"
    style_save_dir.mkdir(parents=True, exist_ok=True)
    
    pretrained_style_path = DIFFUSIONPEN_DIR / "style_models" / "iam_style_diffusionpen.pth"
    if not pretrained_style_path.exists():
        pretrained_style_path = DIFFUSIONPEN_DIR / "style_models" / "iam_style_diffusionpen_triplet.pth"
        
    cmd = [
        python_exe,
        "style_encoder_train.py",
        "--dataset", "custom",
        "--custom_data_dir", str((SCRIPT_DIR / args.dataset_dir).resolve()),
        "--epochs", str(args.style_epochs),
        "--batch_size", str(args.style_batch_size),
        "--model", "mobilenetv2_100",
        "--mode", "mixed",
        "--device", args.device,
        "--save_path", "./style_models",
        "--pretrained", "True",
        "--pretrained_path", str(pretrained_style_path.resolve()) if pretrained_style_path.exists() else "./style_models/iam_style_diffusionpen.pth",
        "--num_workers", "0"
    ]
    
    print(f"Executing in {DIFFUSIONPEN_DIR}:")
    print(" ".join(cmd))
    print("-" * 65)
    
    ret = subprocess.run(cmd, cwd=str(DIFFUSIONPEN_DIR))
    if ret.returncode != 0:
        print(f"\n[Warning] Stage 1 exited with code {ret.returncode}.")
        print("Checking if existing style weights are available...")
    else:
        print("\n[Stage 1 Complete] Style encoder fine-tuned successfully.")


def run_stage_2_diffusionpen(args, python_exe: str):
    """
    Stage 2: Resume fine-tuning the Diffusion UNet from pretrained checkpoint.
    """
    print("\n" + "=" * 65)
    print(" STAGE 2: Fine-Tuning DiffusionPen UNet (150 Epochs)")
    print("=" * 65)
    
    # Target directory for fine-tuned checkpoints
    finetuned_dir = DIFFUSIONPEN_DIR / args.save_path
    finetuned_models_dir = finetuned_dir / "models"
    finetuned_models_dir.mkdir(parents=True, exist_ok=True)
    
    pretrained_base = DIFFUSIONPEN_DIR / "diffusionpen_iam_model_path"
    
    # Check for newly fine-tuned style model or fallback
    custom_style_model = DIFFUSIONPEN_DIR / "style_models" / "mixed_custom_mobilenetv2_100.pth"
    if not custom_style_model.exists():
        custom_style_model = DIFFUSIONPEN_DIR / "style_models" / "iam_style_diffusionpen.pth"
    print(f"Using Style Model: {custom_style_model}")
    
    # If finetuned_models_dir is empty, copy base checkpoint so train.py resumes seamlessly
    target_ckpt = finetuned_models_dir / "ckpt.pt"
    src_ckpt = pretrained_base / "models" / "ckpt.pt"
    if not target_ckpt.exists() and src_ckpt.exists():
        print(f"Seeding initial weights from {src_ckpt} -> {target_ckpt}...")
        shutil.copy2(src_ckpt, target_ckpt)
        src_ema = pretrained_base / "models" / "ema_ckpt.pt"
        if src_ema.exists():
            shutil.copy2(src_ema, finetuned_models_dir / "ema_ckpt.pt")
        src_optim = pretrained_base / "models" / "optim.pt"
        if src_optim.exists():
            shutil.copy2(src_optim, finetuned_models_dir / "optim.pt")
            
    cmd = [
        python_exe,
        "train.py",
        "--dataset", "custom",
        "--custom_data_dir", str((SCRIPT_DIR / args.dataset_dir).resolve()),
        "--epochs", str(args.diffusion_epochs),
        "--batch_size", str(args.batch_size),
        "--model_name", "diffusionpen",
        "--save_path", f"./{args.save_path}",
        "--resume_path", "./diffusionpen_iam_model_path",
        "--style_path", str(custom_style_model.resolve()),
        "--stable_dif_path", "./stable-diffusion-v1-5",
        "--load_check", "True",
        "--device", args.device,
        "--num_workers", "0"
    ]
    
    print(f"Executing in {DIFFUSIONPEN_DIR}:")
    print(" ".join(cmd))
    print("-" * 65)
    
    ret = subprocess.run(cmd, cwd=str(DIFFUSIONPEN_DIR))
    if ret.returncode != 0:
        print(f"\n[Error] Stage 2 exited with code {ret.returncode}.")
        sys.exit(ret.returncode)
        
    print("\n[Stage 2 Complete] DiffusionPen fine-tuned successfully!")
    print(f"Saved Checkpoints in: {finetuned_models_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Style Encoder and DiffusionPen UNet.")
    parser.add_argument("--dataset_dir", type=str, default="data/processed_dataset",
                        help="Path to processed dataset directory (words/ and labels.txt)")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "style_only", "diffusion_only"],
                        help="Which fine-tuning stages to execute")
    parser.add_argument("--style_epochs", type=int, default=25,
                        help="Number of epochs for Stage 1 style encoder fine-tuning")
    parser.add_argument("--style_batch_size", type=int, default=32,
                        help="Batch size for Stage 1 style encoder")
    parser.add_argument("--diffusion_epochs", type=int, default=150,
                        help="Number of epochs for Stage 2 DiffusionPen fine-tuning (default: 150)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for Stage 2 DiffusionPen (16 or 32 for 24GB VRAM)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for training (e.g. cuda:0)")
    parser.add_argument("--save_path", type=str, default="finetuned_diffusionpen",
                        help="Folder inside DiffusionPen to save fine-tuned checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = (SCRIPT_DIR / args.dataset_dir).resolve()
    
    if not check_dataset_exists(dataset_path):
        sys.exit(1)
        
    python_exe = sys.executable
    print("\n" + "=" * 65)
    print(" DiffusionPen Fine-Tuning Pipeline")
    print("=" * 65)
    print(f"Dataset Location : {dataset_path}")
    print(f"Target Device    : {args.device}")
    print(f"Diffusion Epochs : {args.diffusion_epochs}")
    print(f"Batch Size       : {args.batch_size}")
    print(f"Save Path        : {DIFFUSIONPEN_DIR / args.save_path}")
    
    if args.stage in ["all", "style_only"]:
        run_stage_1_style_encoder(args, python_exe)
        
    if args.stage in ["all", "diffusion_only"]:
        run_stage_2_diffusionpen(args, python_exe)

    print("\n" + "=" * 65)
    print(" Fine-Tuning Pipeline Complete!")
    print(" You can now run 3_render_on_template.py to write out any PDF.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

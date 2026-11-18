#!/usr/bin/env python3
"""
1_prepare_dataset.py

Dataset Preparation Pipeline:
1. Reads handwritten assignment photos from data/raw_assignments/
2. Uses PaddleOCR (det=True, rec=False) to detect word bounding boxes (with 4px padding).
3. Applies OpenCV adaptive thresholding to strip ruled lines, notebook grids, and camera shadows.
4. Auto-transcribes each clean black-on-white crop using microsoft/trocr-large-handwritten on CUDA.
5. Filters out noisy/empty crops (length 2-15 alphanumeric) and writes:
   - data/processed_dataset/words/word_XXXX.png
   - data/processed_dataset/labels.txt (format: word_XXXX.png,writer_0,transcription)
"""

import os
import sys
import re
import random
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configure cache directories
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# If Hugging Face cache exists in user directory, use it so offline loading works instantly
user_hf = Path("C:/Users/Sami-PC/.cache/huggingface")
if (user_hf / "hub" / "models--microsoft--trocr-large-handwritten").exists():
    os.environ["HF_HOME"] = str(user_hf)
else:
    os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")

os.environ["USERPROFILE"] = str(CACHE_DIR)
os.environ["HOME"] = str(CACHE_DIR)
os.environ["HOMEDRIVE"], os.environ["HOMEPATH"] = os.path.splitdrive(str(CACHE_DIR))
os.environ["PADDLE_HOME"] = str(CACHE_DIR / "paddle")

import torch
import fitz  # PyMuPDF for extracting pages from handwriting PDFs
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR


def extract_pdf_pages(input_dir: Path, dpi: int = 300) -> list:
    """
    Search input_dir for PDF files containing scans/photos of handwriting,
    and extract all pages as high-resolution images into input_dir/extracted_pdf_pages/.
    Returns list of extracted image Path objects.
    """
    pdf_files = [p for p in input_dir.rglob("*.pdf") if "extracted_pdf_pages" not in p.parts]
    extracted_paths = []
    
    if not pdf_files:
        return extracted_paths
        
    extract_dir = input_dir / "extracted_pdf_pages"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_path in pdf_files:
        print(f"\n[PDF Ingestion] Found PDF with handwriting samples: {pdf_path.name}")
        try:
            with fitz.open(str(pdf_path)) as doc:
                num_pages = len(doc)
                print(f"  Extracting {num_pages} page(s) at {dpi} DPI...")
                matrix = fitz.Matrix(dpi / 72, dpi / 72)
                for page_idx in range(num_pages):
                    page = doc[page_idx]
                    pix = page.get_pixmap(matrix=matrix)
                    out_img_path = extract_dir / f"{pdf_path.stem}_page_{page_idx + 1:03d}.png"
                    pix.save(str(out_img_path))
                    extracted_paths.append(out_img_path)
            print(f"  Successfully extracted {len(extracted_paths)} page(s) to {extract_dir.name}/")
        except Exception as e:
            print(f"  [Error] Failed to extract pages from {pdf_path.name}: {e}")
            
    return extracted_paths


def get_image_paths(input_dir: Path) -> list:
    """Find all image files in input directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]
    return sorted(image_paths)


def preprocess_crop(crop_bgr: np.ndarray, block_size: int = 25, c_val: int = 12) -> np.ndarray:
    """
    Convert crop to grayscale and apply adaptive thresholding to remove
    ruled lines, notebook grids, and non-uniform lighting/shadows.
    Returns binary image: 0 for ink (black), 255 for paper (white).
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Mild blur to suppress scanner/sensor high-frequency noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Ensure block size is odd and > 1
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    
    # Adaptive thresholding: dark ink -> 0, bright paper -> 255
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_val
    )
    
    return thresh


def is_valid_ink_crop(thresh: np.ndarray) -> bool:
    """Check if thresholded crop contains a reasonable amount of ink pixels."""
    total_pixels = thresh.size
    ink_pixels = np.sum(thresh == 0)
    ink_ratio = ink_pixels / float(total_pixels)
    
    # Reject crops that are completely blank or overwhelmingly dark noise
    return 0.015 <= ink_ratio <= 0.80


def segment_line_into_words(binary_crop: np.ndarray, min_gap: int = 6, min_word_w: int = 12) -> list:
    """
    Segment a text-line binary crop into individual word crops
    using vertical projection profiling (finding whitespace column gaps).
    Returns list of (start_x, end_x) coordinate tuples.
    """
    col_ink = np.sum(binary_crop == 0, axis=0)
    is_ink = col_ink > 0
    
    # Smooth small 1-2 pixel intra-character letter breaks
    smoothed = np.copy(is_ink)
    for i in range(1, len(is_ink) - 1):
        if not is_ink[i] and is_ink[i - 1] and is_ink[i + 1]:
            smoothed[i] = True
            
    word_spans = []
    in_word = False
    start_x = 0
    
    for x, ink_present in enumerate(smoothed):
        if ink_present and not in_word:
            in_word = True
            start_x = x
        elif not ink_present and in_word:
            # Check gap length to next ink pixel
            remaining = smoothed[x:]
            next_ink_indices = np.where(remaining)[0]
            if len(next_ink_indices) > 0 and next_ink_indices[0] < min_gap:
                continue
            else:
                end_x = x
                if end_x - start_x >= min_word_w:
                    word_spans.append((start_x, end_x))
                in_word = False
                
    if in_word and len(smoothed) - start_x >= min_word_w:
        word_spans.append((start_x, len(smoothed)))
        
    if not word_spans:
        word_spans = [(0, binary_crop.shape[1])]
        
    return word_spans


def clean_transcription(text: str) -> str:
    """Sanitize transcription string for labels.txt format."""
    # Remove commas to prevent CSV delimiter clash
    text = text.replace(",", "").strip()
    return text


def filter_word(transcription: str, min_len: int = 2, max_len: int = 15) -> bool:
    """Validate word against single word, length, and alphanumeric requirements."""
    # Must be a single word (no internal spaces)
    if " " in transcription:
        return False
    cleaned = re.sub(r'[^A-Za-z0-9]', '', transcription)
    if not (min_len <= len(cleaned) <= max_len):
        return False
    # Must contain at least one letter or digit
    return bool(re.search(r'[A-Za-z0-9]', transcription))


def create_demo_assignment(filepath: Path):
    """Generate a realistic synthetic assignment page with handwritten notes."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 2200
    img = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)
    
    # Draw faint ruled notebook lines
    for y in range(160, height - 120, 64):
        draw.line([(90, y), (width - 90, y)], fill=(215, 225, 238), width=1)
    # Draw left red margin line
    draw.line([(180, 80), (180, height - 80)], fill=(240, 180, 180), width=2)
    
    # Load realistic font
    font = None
    for font_candidate in ["C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(font_candidate):
            try:
                font = ImageFont.truetype(font_candidate, 34)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
        
    sample_lines = [
        ["Natural", "language", "processing", "and", "deep", "learning"],
        ["DiffusionPen", "learns", "personalized", "stroke", "patterns"],
        ["Few", "shot", "diffusion", "models", "conditioned", "on", "style"],
        ["High", "quality", "text", "rendering", "on", "lined", "paper"],
        ["Machine", "learning", "assignment", "notes", "and", "formulas"],
        ["Gradient", "descent", "optimization", "with", "adaptive", "rates"],
        ["Neural", "networks", "approximate", "complex", "trajectories"]
    ]
    
    y = 150
    for line_words in sample_lines:
        x = 210
        for w in line_words:
            # Subtle vertical jitter to simulate human writing
            yj = y + random.randint(-2, 2)
            draw.text((x, yj), w, fill=(20, 30, 75), font=font)
            # Advance with word width + space
            bbox = draw.textbbox((x, yj), w, font=font)
            w_width = bbox[2] - bbox[0]
            x += w_width + random.randint(22, 35)
        y += 64
        
    img.save(filepath)
    print(f"[Demo] Created realistic assignment photo at: {filepath}")


def parse_args():
    parser = argparse.ArgumentParser(description="Crop, clean, and auto-label handwritten word dataset.")
    parser.add_argument("--input_dir", type=str, default="data/raw_assignments",
                        help="Path to folder containing raw assignment photos")
    parser.add_argument("--output_dir", type=str, default="data/processed_dataset",
                        help="Path to output processed dataset directory")
    parser.add_argument("--writer_id", type=str, default="writer_0",
                        help="Writer identifier for labels.txt")
    parser.add_argument("--block_size", type=int, default=25,
                        help="OpenCV adaptive threshold block size (odd integer)")
    parser.add_argument("--c_val", type=int, default=12,
                        help="OpenCV adaptive threshold constant C")
    parser.add_argument("--padding", type=int, default=4,
                        help="Padding in pixels added around each detected word bounding box")
    parser.add_argument("--min_len", type=int, default=2,
                        help="Minimum character length for accepted words")
    parser.add_argument("--max_len", type=int, default=15,
                        help="Maximum character length for accepted words")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device for TrOCR (e.g. cuda or cpu)")
    parser.add_argument("--clean", action="store_true",
                        help="Clean out existing processed words and labels before processing")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo assignment if raw_assignments is empty")
    return parser.parse_args()


def main():
    args = parse_args()
    
    base_dir = SCRIPT_DIR
    input_dir = base_dir / args.input_dir
    output_dir = base_dir / args.output_dir
    words_dir = output_dir / "words"
    labels_file = output_dir / "labels.txt"
    
    words_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    
    if args.clean and output_dir.exists():
        import shutil
        print(f"[Dataset Preparation] Cleaning previous processed dataset at {output_dir}...")
        if words_dir.exists():
            shutil.rmtree(words_dir)
        if labels_file.exists():
            labels_file.unlink()
        words_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest any multi-image/multi-page handwriting PDFs present in input_dir
    extract_pdf_pages(input_dir)
    
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print(f"[Warning] No assignment photos or handwriting PDFs found in {input_dir}.")
        if args.demo:
            demo_path = input_dir / "demo_assignment_1.png"
            create_demo_assignment(demo_path)
            image_paths = [demo_path]
        else:
            print("Please drop your handwriting PDF or 5-10 assignment photos into:")
            print(f"  {input_dir}")
            print("Or pass --demo to run with a generated sample.")
            return

    print(f"\n=======================================================")
    print(f" 1. Handwritten Word Extraction & Auto-Labeling Pipeline")
    print(f"=======================================================")
    print(f"Found {len(image_paths)} assignment image(s) to process.")
    print(f"Device: {args.device}")
    print(f"Output Words: {words_dir}")
    print(f"Output Labels: {labels_file}")
    
    # 1. Initialize PaddleOCR detector
    print("\n[1/3] Initializing PaddleOCR text detector...")
    ocr = PaddleOCR(use_angle_cls=False, lang="en")
    
    # 2. Initialize TrOCR model and processor
    print("\n[2/3] Loading microsoft/trocr-large-handwritten...")
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    trocr_model.to(args.device)
    trocr_model.eval()
    
    print("\n[3/3] Processing images, cropping words, and transcribing...")
    word_counter = 1
    labels_records = []
    
    # If labels.txt already exists, check existing count
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    word_counter += 1
    
    total_detected_boxes = 0
    total_saved_words = 0
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [Skipped] Could not read image: {img_path}")
            continue
            
        img_h, img_w = img_bgr.shape[:2]
        
        # Run PaddleOCR detection (bounding boxes only)
        try:
            results = ocr.ocr(img_bgr, det=True, rec=False)
        except Exception as e:
            print(f"  [Error in PaddleOCR]: {e}")
            continue
            
        boxes = results[0] if results and len(results) > 0 and results[0] is not None else []
        if not boxes:
            print(f"  No text boxes detected.")
            continue
            
        print(f"  Detected {len(boxes)} text regions.")
        total_detected_boxes += len(boxes)
        
        # Sort bounding boxes top-to-bottom, left-to-right
        def box_sort_key(b):
            pts = np.array(b, dtype=np.float32)
            y_min = np.min(pts[:, 1])
            x_min = np.min(pts[:, 0])
            # Group by ~30px line bands
            return (round(y_min / 30.0), x_min)
            
        boxes = sorted(boxes, key=box_sort_key)
        
        for box in tqdm(boxes, desc=f"Transcribing {img_path.name}", unit="word"):
            pts = np.array(box, dtype=np.float32)
            x_min = int(np.floor(np.min(pts[:, 0]))) - args.padding
            x_max = int(np.ceil(np.max(pts[:, 0]))) + args.padding
            y_min = int(np.floor(np.min(pts[:, 1]))) - args.padding
            y_max = int(np.ceil(np.max(pts[:, 1]))) + args.padding
            
            # Clamp to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            # Discard too tiny crops
            if box_w < 12 or box_h < 12:
                continue
                
            crop_bgr = img_bgr[y_min:y_max, x_min:x_max]
            
            # Adaptive thresholding to remove ruled lines and shadow gradients
            thresh_crop = preprocess_crop(crop_bgr, block_size=args.block_size, c_val=args.c_val)
            
            # Segment multi-word lines into individual word crops
            if box_w / float(max(1, box_h)) > 2.2:
                word_spans = segment_line_into_words(thresh_crop, min_gap=8, min_word_w=12)
            else:
                word_spans = [(0, thresh_crop.shape[1])]
                
            for w_start, w_end in word_spans:
                word_crop = thresh_crop[:, max(0, w_start - args.padding) : min(thresh_crop.shape[1], w_end + args.padding)]
                
                if not is_valid_ink_crop(word_crop):
                    continue
                    
                # Convert binary crop (0=ink, 255=paper) to RGB PIL Image for TrOCR
                crop_rgb = cv2.cvtColor(word_crop, cv2.COLOR_GRAY2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                
                # Transcribe with TrOCR
                try:
                    pixel_values = trocr_processor(images=crop_pil, return_tensors="pt").pixel_values.to(args.device)
                    with torch.no_grad():
                        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=20)
                    transcription = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    transcription = clean_transcription(transcription)
                except Exception as e:
                    continue
                    
                # Filter clean alphanumeric words of length 2-15
                if not filter_word(transcription, min_len=args.min_len, max_len=args.max_len):
                    continue
                    
                # Save word crop PNG
                word_filename = f"word_{word_counter:04d}.png"
                word_out_path = words_dir / word_filename
                crop_pil.save(word_out_path)
                
                # Record label entry: filename.png,writer_0,transcription
                record_line = f"{word_filename},{args.writer_id},{transcription}\n"
                labels_records.append(record_line)
                
                with open(labels_file, "a", encoding="utf-8") as f:
                    f.write(record_line)
                    
                word_counter += 1
                total_saved_words += 1

    print(f"\n=======================================================")
    print(f" Dataset Preparation Complete!")
    print(f"=======================================================")
    print(f"Total Text Regions Detected : {total_detected_boxes}")
    print(f"Total Clean Words Saved     : {total_saved_words}")
    print(f"Words Directory             : {words_dir}")
    print(f"Labels Manifest File        : {labels_file}")
    
    if total_saved_words > 0:
        print("\nSample Transcriptions:")
        for sample in labels_records[:5]:
            print(f"  {sample.strip()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

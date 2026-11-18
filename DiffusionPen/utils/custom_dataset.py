import os
import random
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.auxilary_functions import image_resize_PIL, centered_PIL, affine_transformation


class CustomWordDataset(Dataset):
    """
    Dataset loader for fine-tuning DiffusionPen on custom handwritten word crops.
    Reads from data/processed_dataset/words/ and data/processed_dataset/labels.txt
    without requiring the full IAM dataset.
    """
    def __init__(self, basefolder: str = '../data/processed_dataset', transforms=None, args=None):
        self.basefolder = Path(basefolder)
        self.transforms = transforms
        self.args = args
        self.fixed_size = (64, 256)
        
        # Locate labels.txt and words directory
        if (self.basefolder / "labels.txt").exists():
            self.labels_path = self.basefolder / "labels.txt"
            self.words_dir = self.basefolder / "words"
        elif Path(basefolder).is_file():
            self.labels_path = Path(basefolder)
            self.words_dir = self.labels_path.parent / "words"
        else:
            # Fallback relative to project structure
            alt_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed_dataset"
            self.labels_path = alt_path / "labels.txt"
            self.words_dir = alt_path / "words"

        self.samples = []
        self.writer_to_id = {}
        
        if not self.labels_path.exists():
            print(f"[Warning] Custom dataset labels file not found at: {self.labels_path}")
            return

        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 2)
                if len(parts) == 3:
                    fname, writer_str, transcr = parts
                elif len(parts) == 2:
                    fname, transcr = parts
                    writer_str = "writer_0"
                else:
                    continue
                
                # Check if image exists
                img_path = self.words_dir / fname
                if not img_path.exists():
                    # Try direct relative path
                    if Path(fname).exists():
                        img_path = Path(fname)
                    else:
                        continue
                        
                if writer_str not in self.writer_to_id:
                    self.writer_to_id[writer_str] = len(self.writer_to_id)
                wid_int = self.writer_to_id[writer_str]
                
                self.samples.append((str(img_path), transcr.strip(), wid_int))

        print(f"[CustomWordDataset] Loaded {len(self.samples)} word samples across {len(self.writer_to_id)} writer(s).")
        self.data = self._load_and_preprocess_all()

    def _load_and_preprocess_all(self):
        preprocessed = []
        target_h, target_w = self.fixed_size
        
        for img_path, transcr, wid in self.samples:
            try:
                img = Image.open(img_path).convert("RGB")
                
                # Resize height to 64 preserving aspect ratio
                img_w, img_h = img.size
                if img_h <= 0 or img_w <= 0:
                    continue
                scale = target_h / float(img_h)
                new_w = max(4, int(img_w * scale))
                img = img.resize((new_w, target_h), Image.Resampling.BILINEAR)
                
                # Pad or fit into 256x64
                if new_w < target_w:
                    out_img = ImageOps.pad(img, size=(target_w, target_h), color="white")
                else:
                    while img.size[0] > target_w:
                        img = image_resize_PIL(img, width=img.size[0] - 20)
                    out_img = centered_PIL(img, (target_h, target_w), border_value=255.0)
                    
                preprocessed.append((out_img, transcr, wid, img_path))
            except Exception as e:
                continue
                
        return preprocessed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_pil, transcr, wid, img_path = self.data[index]
        
        # Style reference crops: 5 crops from the same writer
        same_writer = [d for d in self.data if d[2] == wid]
        if len(same_writer) >= 5:
            sampled_refs = random.sample(same_writer, 5)
        else:
            # Repeat with replacement if fewer than 5
            sampled_refs = [random.choice(same_writer) for _ in range(5)]
            
        cor_sample = random.choice(same_writer)
        
        if self.transforms is not None:
            img_tensor = self.transforms(img_pil)
            cor_tensor = self.transforms(cor_sample[0])
            s_imgs = torch.stack([self.transforms(ref[0]) for ref in sampled_refs])
        else:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img_pil)
            cor_tensor = to_tensor(cor_sample[0])
            s_imgs = torch.stack([to_tensor(ref[0]) for ref in sampled_refs])

        return img_tensor, transcr, wid, s_imgs, img_path, cor_tensor

    def collate_fn(self, batch):
        img, transcr, wid, s_imgs, img_path, cor_im = zip(*batch)
        images_batch = torch.stack(img)
        wid_batch = torch.tensor(wid, dtype=torch.long)
        s_imgs_batch = torch.stack(s_imgs)
        cor_images_batch = torch.stack(cor_im)
        return images_batch, transcr, wid_batch, s_imgs_batch, img_path, cor_images_batch


class CustomStyleDataset(Dataset):
    """
    Dataset loader for fine-tuning the DiffusionPen Style Encoder (MobileNetV2)
    on user's handwritten notes. Generates anchor, positive, and negative pairs.
    """
    def __init__(self, basefolder: str = '../data/processed_dataset', transforms=None, args=None):
        self.basefolder = Path(basefolder)
        self.transforms = transforms
        self.args = args
        self.fixed_size = (64, 256)
        
        if (self.basefolder / "labels.txt").exists():
            self.labels_path = self.basefolder / "labels.txt"
            self.words_dir = self.basefolder / "words"
        elif Path(basefolder).is_file():
            self.labels_path = Path(basefolder)
            self.words_dir = self.labels_path.parent / "words"
        else:
            alt_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed_dataset"
            self.labels_path = alt_path / "labels.txt"
            self.words_dir = alt_path / "words"

        self.samples = []
        self.writer_to_id = {}
        
        if not self.labels_path.exists():
            print(f"[Warning] Custom style dataset labels file not found at: {self.labels_path}")
            return

        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 2)
                if len(parts) == 3:
                    fname, writer_str, transcr = parts
                elif len(parts) == 2:
                    fname, transcr = parts
                    writer_str = "writer_0"
                else:
                    continue
                
                img_path = self.words_dir / fname
                if not img_path.exists() and Path(fname).exists():
                    img_path = Path(fname)
                if not img_path.exists():
                    continue
                    
                if writer_str not in self.writer_to_id:
                    self.writer_to_id[writer_str] = len(self.writer_to_id)
                wid_int = self.writer_to_id[writer_str]
                self.samples.append((str(img_path), wid_int, transcr.strip()))

        print(f"[CustomStyleDataset] Loaded {len(self.samples)} style samples.")

    def __len__(self):
        return len(self.samples)

    def _format_pil(self, img_pil):
        target_h, target_w = self.fixed_size
        img_w, img_h = img_pil.size
        if img_h > 0:
            scale = target_h / float(img_h)
            new_w = max(4, int(img_w * scale))
            img_pil = img_pil.resize((new_w, target_h), Image.Resampling.BILINEAR)
        if img_pil.size[0] < target_w:
            return ImageOps.pad(img_pil, size=(target_w, target_h), color="white")
        else:
            while img_pil.size[0] > target_w:
                img_pil = image_resize_PIL(img_pil, width=img_pil.size[0] - 20)
            return centered_PIL(img_pil, (target_h, target_w), border_value=255.0)

    def __getitem__(self, index):
        img_path, wid, transcr = self.samples[index]
        img_pil = Image.open(img_path).convert("RGB")
        
        # Positive sample: another word from same writer or affine augmented version
        same_writer = [s for s in self.samples if s[1] == wid]
        if len(same_writer) > 1:
            pos_info = random.choice([s for s in same_writer if s[0] != img_path] or same_writer)
            pos_pil = Image.open(pos_info[0]).convert("RGB")
        else:
            # Apply subtle affine stroke transformation for positive pair
            np_img = np.array(img_pil)
            warped = affine_transformation(np_img, s=0.15, border_value=255.0)
            pos_pil = Image.fromarray(np.uint8(warped))
            
        # Negative sample: different writer if available, else synthetic perturbed style
        diff_writer = [s for s in self.samples if s[1] != wid]
        if diff_writer:
            neg_info = random.choice(diff_writer)
            neg_pil = Image.open(neg_info[0]).convert("RGB")
        else:
            # Generate negative with significant geometric shear/slant to contrast style
            np_img = np.array(pos_pil)
            warped_neg = affine_transformation(np_img, m=1.2, s=0.35, border_value=255.0)
            neg_pil = Image.fromarray(np.uint8(warped_neg))

        img_norm = self._format_pil(img_pil)
        pos_norm = self._format_pil(pos_pil)
        neg_norm = self._format_pil(neg_pil)

        if self.transforms is not None:
            img_t = self.transforms(img_norm)
            pos_t = self.transforms(pos_norm)
            neg_t = self.transforms(neg_norm)
        else:
            to_t = transforms.ToTensor()
            img_t = to_t(img_norm)
            pos_t = to_t(pos_norm)
            neg_t = to_t(neg_norm)

        # Returns matching WordStyleDataset format: (img, transcr, wid, pos, neg, img_path)
        return img_t, transcr, wid, pos_t, neg_t, img_path

    def collate_fn(self, batch):
        img, transcr, wid, pos, neg, img_path = zip(*batch)
        images_batch = torch.stack(img)
        wid_batch = torch.tensor(wid, dtype=torch.long)
        pos_batch = torch.stack(pos)
        neg_batch = torch.stack(neg)
        return images_batch, transcr, wid_batch, pos_batch, neg_batch, img_path

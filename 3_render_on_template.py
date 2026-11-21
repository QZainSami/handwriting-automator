#!/usr/bin/env python3
"""
3_render_on_template.py

Authentic Personal Handwriting & Dual-Page Notebook Compositing Pipeline:
1. Automatically discovers input document (PDF, TXT, MD) in input_pdfs/ (e.g. assig.txt).
2. Uses the user's authentic personal handwriting glyph library (royal blue ballpoint ink,
   custom letterforms, all 3 variations of lowercase 'r': print, hook, and cursive).
3. Multi-variant glyph engine preventing clone repetition of repeated letters (ee, tt, oo, ll, ss, aa).
4. Physical ink-paper Multiply blending (ruled lines and paper fibers show naturally through ink).
5. Organic left margin drift and smart end-of-line margin squeezing.
6. Auto-filled handwritten Date headers on printed notebook template lines.
7. Freehand organic ballpoint underlines under major section headings.
8. Ballpoint micro-physics & velocity pooling at stroke vertices.
9. Context-aware organic kerning for natural ligature pairs (th, he, in, on, st, er, an, etc.).
10. Natural hand fatigue and progressive relaxation over pages 1 through 5.
11. Photorealistic mobile camera optics: soft lens vignette & spiral contact ambient shadow.
12. Handwritten corner page numbering ('Page 1', 'Page 2', etc.) in outer header margins.
13. Authentic human touch: neat single-stroke pen correction / strike-through.
14. Compiles final output into output_pages/final_handwritten.pdf and individual page images.
"""

import os
import sys
import re
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
GLYPH_DIR = SCRIPT_DIR / "data" / "user_glyphs"
TEMPLATES_DIR = SCRIPT_DIR / "data" / "blank_templates"
INPUT_DIR = SCRIPT_DIR / "input_pdfs"
OUTPUT_DIR = SCRIPT_DIR / "output_pages"

# Configure cache directories
CACHE_DIR = SCRIPT_DIR / ".cache"
os.environ.setdefault("USERPROFILE", str(CACHE_DIR))
os.environ.setdefault("HOME", str(CACHE_DIR))

# Common cursive ligature pairs for natural stroke flow
COMMON_LIGATURE_PAIRS = {
    ("t", "h"), ("h", "e"), ("i", "n"), ("o", "n"), ("s", "t"),
    ("e", "r"), ("a", "n"), ("c", "h"), ("o", "r"), ("e", "n"),
    ("r", "e"), ("t", "i"), ("a", "l"), ("l", "e"), ("n", "d"),
    ("n", "t"), ("d", "e"), ("v", "e"), ("a", "t"), ("e", "d"),
    ("i", "s"), ("o", "u"), ("a", "r"), ("p", "a"), ("m", "a"),
    ("q", "u"), ("e", "x"), ("o", "f")
}


def find_input_document(requested_path: str = None) -> Path:
    """
    Locates the input document. If requested_path exists, use it.
    Otherwise, automatically searches input_pdfs/ for assig.txt or any .pdf / .txt file.
    """
    if requested_path:
        p = Path(requested_path)
        if not p.is_absolute():
            p = (SCRIPT_DIR / p).resolve()
        if p.exists() and p.is_file():
            return p
        print(f"[Notice] Specified input '{requested_path}' not found on disk.")

    # Search in input_pdfs/
    if INPUT_DIR.exists():
        # 1. Prefer assig.txt if present
        assig_txt = INPUT_DIR / "assig.txt"
        if assig_txt.exists():
            print(f"[Auto-Detector] Found competition document: {assig_txt.name}")
            return assig_txt

        # 2. Prefer any PDF
        pdfs = list(INPUT_DIR.glob("*.pdf"))
        if pdfs:
            print(f"[Auto-Detector] Found PDF document: {pdfs[0].name}")
            return pdfs[0]

        # 3. Any text or markdown file
        txts = list(INPUT_DIR.glob("*.txt")) + list(INPUT_DIR.glob("*.md"))
        if txts:
            print(f"[Auto-Detector] Found text document: {txts[0].name}")
            return txts[0]

    raise FileNotFoundError(
        f"No input document found in {INPUT_DIR}. Please place 'assig.txt' or your PDF into {INPUT_DIR}."
    )


def extract_content_lines(doc_path: Path) -> list:
    """
    Extracts text lines from PDF or TXT, cleaning bullet points and unicode formatting.
    Returns a list of structured lines/paragraphs.
    """
    lines = []
    if doc_path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(str(doc_path))
        for page in doc:
            txt = page.get_text("text")
            for raw_line in txt.split("\n"):
                clean = raw_line.strip()
                if clean:
                    lines.append(clean)
    else:
        with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Clean common UTF-8 encoding artifacts
        content = content.replace("\ufeff", "").replace("Ã¢â‚¬Â¢", "â€¢").replace("\r\n", "\n")
        raw_lines = content.split("\n")
        for raw_line in raw_lines:
            clean = raw_line.strip()
            # Normalize tab separated bullet items
            clean = clean.replace("\t", " ")
            clean = " ".join(clean.split())
            if clean:
                lines.append(clean)

    return lines


def blend_ink_onto_page(page_arr: np.ndarray, ink_img: Image.Image, x: int, y: int):
    """
    Physically blends translucent ballpoint ink onto notebook paper.
    Paper fibers, texture, and ruled notebook lines naturally show through
    the translucent royal blue dye, eliminating artificial opaque cutout edges.
    Boundary safe with sub-pixel integer clipping.
    """
    if ink_img is None:
        return

    gw, gh = ink_img.size
    page_h, page_w = page_arr.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(page_w, x + gw)
    y2 = min(page_h, y + gh)

    if x1 >= x2 or y1 >= y2:
        return

    gx1 = x1 - x
    gy1 = y1 - y
    gx2 = gx1 + (x2 - x1)
    gy2 = gy1 + (y2 - y1)

    ink_arr = np.array(ink_img)[gy1:gy2, gx1:gx2]
    alpha = ink_arr[:, :, 3:4].astype(float) / 255.0
    if not np.any(alpha > 0.02):
        return

    ink_rgb = ink_arr[:, :, :3].astype(float)
    bg_rgb = page_arr[y1:y2, x1:x2].astype(float)

    # Physical ballpoint Multiply blending formula:
    # (bg * ink) / 255.0 lets paper fibers and ruled lines show naturally through the dye
    mult_rgb = (bg_rgb * ink_rgb) / 255.0 * 1.12
    blended = np.clip(bg_rgb * (1.0 - alpha) + mult_rgb * alpha, 0, 255).astype(np.uint8)
    page_arr[y1:y2, x1:x2] = blended


def draw_handwritten_underline(page_arr: np.ndarray, x1: int, x2: int, y: int, slope: float = 0.0):
    """
    Draws a natural, freehand royal blue ballpoint underline under major headings.
    Features organic pressure taper, subtle curvature/sag, micro-jitter, and physical multiply blending.
    """
    length = x2 - x1
    if length <= 15:
        return

    num_pts = max(15, length // 12)
    xs = np.linspace(x1, x2, num_pts)
    sag_depth = random.uniform(1.2, 3.2)
    ys = []
    for i, x in enumerate(xs):
        t = (x - x1) / float(length)
        curve_y = 4.0 * sag_depth * t * (1.0 - t)
        jitter_y = random.uniform(-0.5, 0.5)
        slope_y = slope * (x - x1)
        ys.append(y + curve_y + slope_y + jitter_y)

    pad = 15
    min_x, max_x = int(min(xs)) - pad, int(max(xs)) + pad
    min_y, max_y = int(min(ys)) - pad, int(max(ys)) + pad
    cw = max_x - min_x
    ch = max_y - min_y

    stroke_canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    blue_ink = (32, 75, 215)  # RGB royal blue ballpoint

    pts = list(zip(xs - min_x, np.array(ys) - min_y))
    for i in range(len(pts) - 1):
        pt1 = (int(round(pts[i][0])), int(round(pts[i][1])))
        pt2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        t = i / float(len(pts))
        pressure = math.sin(t * math.pi) ** 0.6
        thickness = max(2, int(round(2.0 + 1.8 * pressure)))
        alpha_val = int(round(160 + 85 * pressure))

        cv2.line(stroke_canvas, pt1, pt2, (*blue_ink, alpha_val), thickness, cv2.LINE_AA)

    stroke_img = Image.fromarray(stroke_canvas, mode="RGBA")
    blend_ink_onto_page(page_arr, stroke_img, min_x, min_y)


def draw_handwritten_strikethrough(page_arr: np.ndarray, x1: int, x2: int, y: int, slope: float = 0.0):
    """
    Draws a neat, realistic ballpoint pen strike-through across an accidental slip-up.
    Simulates a human student correcting an error with a single clean horizontal stroke.
    """
    length = x2 - x1
    if length <= 5:
        return

    num_pts = max(10, length // 8)
    xs = np.linspace(x1 - 3, x2 + 3, num_pts)
    sag = random.uniform(-0.6, 0.6)
    ys = []
    for x in xs:
        t = (x - (x1 - 3)) / float(length + 6)
        slope_y = slope * (x - x1)
        curve_y = 4.0 * sag * t * (1.0 - t)
        jitter_y = random.uniform(-0.35, 0.35)
        ys.append(y + slope_y + curve_y + jitter_y)

    pad = 10
    min_x, max_x = int(min(xs)) - pad, int(max(xs)) + pad
    min_y, max_y = int(min(ys)) - pad, int(max(ys)) + pad
    cw = max_x - min_x
    ch = max_y - min_y

    stroke_canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
    blue_ink = (32, 75, 215)
    pts = list(zip(xs - min_x, np.array(ys) - min_y))
    for i in range(len(pts) - 1):
        pt1 = (int(round(pts[i][0])), int(round(pts[i][1])))
        pt2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        cv2.line(stroke_canvas, pt1, pt2, (*blue_ink, 220), 2, cv2.LINE_AA)

    stroke_img = Image.fromarray(stroke_canvas, mode="RGBA")
    blend_ink_onto_page(page_arr, stroke_img, min_x, min_y)


def apply_photorealistic_optics(page_arr: np.ndarray, page_side: str = "left") -> np.ndarray:
    """
    Applies physical mobile camera optics to the composited page:
    1. Subtle radial lens vignette (soft 3.5% light falloff towards outer corners).
    2. Soft spiral contact shadow (ambient occlusion where paper curls near spiral wire coils, ~2.5%).
    Makes the page look 100% like a real smartphone photo taken on a desk.
    """
    h, w = page_arr.shape[:2]
    page_float = page_arr.astype(np.float32)

    # 1. Subtle radial lens vignette
    yc, xc = h * 0.48, w * 0.50
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = ((x_grid - xc) / (w * 0.55)) ** 2 + ((y_grid - yc) / (h * 0.55)) ** 2
    vignette = 1.0 - 0.038 * np.clip(dist_sq, 0.0, 1.2)

    # 2. Spiral contact shadow (ambient occlusion)
    # Left page: spiral is on the right side
    # Right page: spiral is on the left side
    spiral_shadow = np.ones((h, w), dtype=np.float32)
    if page_side == "left":
        spiral_edge = int(w * 0.88)
        shadow_width = int(w * 0.11)
        x_dist = np.clip((x_grid - (spiral_edge - shadow_width)), 0, shadow_width)
        shadow_factor = 1.0 - 0.030 * (x_dist / float(shadow_width)) ** 1.5
        spiral_shadow = np.minimum(spiral_shadow, shadow_factor)
    else:
        spiral_edge = int(w * 0.14)
        shadow_width = int(w * 0.11)
        x_dist = np.clip((spiral_edge + shadow_width - x_grid), 0, shadow_width)
        shadow_factor = 1.0 - 0.030 * (x_dist / float(shadow_width)) ** 1.5
        spiral_shadow = np.minimum(spiral_shadow, shadow_factor)

    lighting = (vignette * spiral_shadow).astype(np.float32)
    for c in range(3):
        page_float[:, :, c] *= lighting

    return np.clip(page_float, 0, 255).astype(np.uint8)


def draw_page_number(page_arr: np.ndarray, engine, page_num: int, page_side: str, t_name: str):
    """
    Draws an authentic handwritten page number ('Page 1', 'Page 2', etc.) in the outer header corner.
    """
    page_text = f"Page {page_num}"
    header_coords = {
        "left_page.jpg": (280, 520, 40),
        "left_page2.jpg": (240, 500, 42),
        "right_page.jpg": (4700, 580, 44),
        "right_page2.jpg": (4400, 475, 42),
    }

    if t_name in header_coords:
        hx, hy, h_cap_h = header_coords[t_name]
    else:
        h, w = page_arr.shape[:2]
        hx = int(w * 0.08) if page_side == "left" else int(w * 0.88)
        hy = int(h * 0.08)
        h_cap_h = 38

    jx = hx + random.randint(-4, 4)
    jy = hy + random.randint(-2, 2)
    res = engine.render_word(page_text, target_cap_height=h_cap_h)
    if res:
        p_img, p_base = res
        paste_y = int(jy - p_base)
        blend_ink_onto_page(page_arr, p_img, jx, paste_y)


def apply_ballpoint_micro_physics(word_img: Image.Image) -> Image.Image:
    """
    Simulates real ballpoint pen ink mechanics:
    1. Subtle rolling-ball micro-texture across paper fibers.
    2. Micro-pooling at stroke turnarounds / direction vertices.
    """
    arr = np.array(word_img)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    if not np.any(alpha > 40):
        return word_img

    # 1. Micro-texture
    tex = np.random.uniform(0.96, 1.04, (h, w)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (3, 3), 0.6)
    rgb = arr[:, :, :3].astype(np.float32)
    for c in range(3):
        rgb[:, :, c] = np.clip(rgb[:, :, c] * tex, 0, 255)
    arr[:, :, :3] = rgb.astype(np.uint8)

    # 2. Micro-pooling at stroke vertices
    bin_mask = (alpha > 80).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(bin_mask, maxCorners=4, qualityLevel=0.35, minDistance=8)
    if corners is not None:
        pool_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for pt in corners:
            cx, cy = int(pt[0][0]), int(pt[0][1])
            if 0 <= cy < h and 0 <= cx < w and alpha[cy, cx] > 60:
                cv2.circle(pool_overlay, (cx, cy), 1, (16, 42, 135, 55), -1)

        ov_a = pool_overlay[:, :, 3:4].astype(float) / 255.0
        ov_rgb = pool_overlay[:, :, :3].astype(float)
        cur_rgb = arr[:, :, :3].astype(float)
        blended_rgb = cur_rgb * (1.0 - ov_a) + ov_rgb * ov_a
        arr[:, :, :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)
        arr[:, :, 3] = np.clip(arr[:, :, 3].astype(float) + pool_overlay[:, :, 3].astype(float) * 0.25, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


class AuthenticHandwritingEngine:
    """
    Personal handwriting generation engine using authentic royal blue ballpoint pen glyphs
    extracted directly from the user's handwriting specimen.
    Supports multi-variant selection to prevent identical cloned letters (ee, tt, oo, ll, ss, aa).
    """
    def __init__(self, glyph_base_dir: Path = GLYPH_DIR, r_mode: str = "mix"):
        self.glyph_base_dir = Path(glyph_base_dir)
        self.r_mode = r_mode
        self.upper_glyphs = defaultdict(list)
        self.lower_glyphs = defaultdict(list)
        self.digit_glyphs = defaultdict(list)
        self.symbol_glyphs = defaultdict(list)
        self.last_used_indices = defaultdict(int)
        self.load_glyphs()

        # Descender fraction below baseline
        self.descenders = {
            "g": 0.42,
            "j": 0.48,
            "p": 0.36,
            "q": 0.52,
            "y": 0.42,
            ",": 0.35
        }

        # Elevated punctuation fraction above baseline
        self.high_puncts = {
            "\"": 0.72,
            "'": 0.72,
            "-": 0.35,
            "–": 0.35,
            "—": 0.35,
            "=": 0.35
        }

        self.punct_map = {
            ".": "dot", ",": "comma", "!": "exclam", "?": "question",
            ":": "colon", ";": "semicolon", "-": "minus", "_": "minus",
            "+": "plus", "=": "equal", "/": "slash", "(": "lparen",
            ")": "rparen", "%": "percent", "$": "dollar", "\"": "quote",
            "'": "squote", "•": "bullet", "*": "bullet", "—": "dash",
            "[": "bracket_l", "]": "bracket_r", "#": "hashtag", "@": "at"
        }

        self.r_variants = ["r_print", "r_hook", "r_cursive"]

    def load_glyphs(self):
        upper_dir = self.glyph_base_dir / "upper"
        lower_dir = self.glyph_base_dir / "lower"
        digit_dir = self.glyph_base_dir / "digits"
        sym_dir = self.glyph_base_dir / "symbols"

        if upper_dir.exists():
            for p in sorted(upper_dir.glob("*.png")):
                img = Image.open(p).convert("RGBA")
                stem = p.stem
                base = stem.split("_")[0]
                self.upper_glyphs[base].append(img)
                if stem != base:
                    self.upper_glyphs[stem].append(img)

        if lower_dir.exists():
            for p in sorted(lower_dir.glob("*.png")):
                img = Image.open(p).convert("RGBA")
                stem = p.stem
                if stem.startswith("r_"):
                    self.lower_glyphs[stem].append(img)
                    self.lower_glyphs["r"].append(img)
                else:
                    base = stem.split("_")[0]
                    self.lower_glyphs[base].append(img)
                    if stem != base:
                        self.lower_glyphs[stem].append(img)

        if digit_dir.exists():
            for p in sorted(digit_dir.glob("*.png")):
                img = Image.open(p).convert("RGBA")
                stem = p.stem
                base = stem.split("_")[0]
                self.digit_glyphs[base].append(img)
                if stem != base:
                    self.digit_glyphs[stem].append(img)

        if sym_dir.exists():
            for p in sorted(sym_dir.glob("*.png")):
                img = Image.open(p).convert("RGBA")
                stem = p.stem
                base = stem.split("_")[0]
                self.symbol_glyphs[base].append(img)
                if stem != base:
                    self.symbol_glyphs[stem].append(img)

        total_u = sum(len(v) for v in self.upper_glyphs.values())
        total_l = sum(len(v) for v in self.lower_glyphs.values())
        total_d = sum(len(v) for v in self.digit_glyphs.values())
        total_s = sum(len(v) for v in self.symbol_glyphs.values())

        print(f"[Handwriting Engine] Loaded {total_u + total_l + total_d + total_s} authentic user glyph variants (Royal Blue Ballpoint Ink).")
        print(f"  - Uppercase letters : {len(self.upper_glyphs)} classes | {total_u} total variants")
        print(f"  - Lowercase letters : {len(self.lower_glyphs)} classes | {total_l} total variants (including 3 'r' styles)")
        print(f"  - Digits            : {len(self.digit_glyphs)} classes | {total_d} total variants")
        print(f"  - Symbols           : {len(self.symbol_glyphs)} classes | {total_s} total variants")

    def _pick_variant(self, glyph_list: list, key: str, is_repeated: bool = False) -> Image.Image:
        """Picks a variant from glyph_list, cycling variants and avoiding identical repeats."""
        if not glyph_list:
            return None

        if len(glyph_list) == 1:
            img = glyph_list[0]
            if is_repeated:
                # Organic micro-variation for single-variant repeated characters (e.g. 'mm')
                w, h = img.size
                scale_w = random.uniform(0.96, 1.04)
                scale_h = random.uniform(0.97, 1.03)
                nw, nh = max(2, int(round(w * scale_w))), max(2, int(round(h * scale_h)))
                img = img.resize((nw, nh), Image.BICUBIC)
                rot = random.uniform(-0.5, 0.5)
                img = img.rotate(rot, resample=Image.BICUBIC, expand=True)
            return img

        # Multiple physical variants available on disk
        last_idx = self.last_used_indices[key]
        if is_repeated:
            # Strictly pick a DIFFERENT variant for immediate double letters (ee, tt, oo, ll, ss)
            choices = [i for i in range(len(glyph_list)) if i != last_idx]
            next_idx = random.choice(choices) if choices else 0
        else:
            # Varied selection with natural rotation
            choices = list(range(len(glyph_list)))
            weights = [0.20 if i == last_idx else 0.80 / (len(glyph_list) - 1) for i in range(len(glyph_list))]
            next_idx = random.choices(choices, weights=weights, k=1)[0]

        self.last_used_indices[key] = next_idx
        return glyph_list[next_idx]

    def get_char_glyph(self, ch: str, r_choice: str = None, is_repeated: bool = False) -> tuple:
        """Returns (PIL.Image, descender_ratio, high_ratio)."""
        if ch == "r":
            if self.r_mode == "print":
                choice = "r_print"
            elif self.r_mode == "hook":
                choice = "r_hook"
            elif self.r_mode == "cursive":
                choice = "r_cursive"
            else:
                choice = r_choice if r_choice in self.r_variants else random.choice(self.r_variants)
            
            cand = self.lower_glyphs.get(choice) or self.lower_glyphs.get("r")
            img = self._pick_variant(cand, "r", is_repeated=is_repeated)
            return img, 0.0, 0.0

        if ch in self.punct_map:
            key = self.punct_map[ch]
            cand = self.symbol_glyphs.get(key)
            if cand:
                img = self._pick_variant(cand, key, is_repeated=is_repeated)
                return img, self.descenders.get(ch, 0.0), self.high_puncts.get(ch, 0.0)

        if ch.isupper() and ch in self.upper_glyphs:
            img = self._pick_variant(self.upper_glyphs[ch], ch, is_repeated=is_repeated)
            return img, 0.0, 0.0

        if ch.islower() and ch in self.lower_glyphs:
            img = self._pick_variant(self.lower_glyphs[ch], ch, is_repeated=is_repeated)
            return img, self.descenders.get(ch, 0.0), 0.0

        if ch.isdigit() and ch in self.digit_glyphs:
            img = self._pick_variant(self.digit_glyphs[ch], ch, is_repeated=is_repeated)
            return img, 0.0, 0.0

        # Case fallbacks
        if ch.upper() in self.upper_glyphs:
            img = self._pick_variant(self.upper_glyphs[ch.upper()], ch.upper(), is_repeated=is_repeated)
            return img, 0.0, 0.0
        if ch.lower() in self.lower_glyphs:
            img = self._pick_variant(self.lower_glyphs[ch.lower()], ch.lower(), is_repeated=is_repeated)
            return img, self.descenders.get(ch.lower(), 0.0), 0.0

        return None, 0.0, 0.0

    def render_word(self, word: str, target_cap_height: int = 110, line_slope: float = 0.0,
                    jitter: bool = True, fatigue_slant: float = 0.0) -> tuple:
        """
        Renders a single handwritten word with authentic glyphs and natural handwriting flow.
        Follows local notebook line slope so every character sits flush on the ruled line.
        Includes ballpoint micro-physics and context-aware organic ligatures.
        Returns (PIL.Image RGBA, baseline_y_at_start).
        """
        if not word:
            return None

        # Base glyph cap height in scanned dataset is ~80px
        scale_factor = target_cap_height / 80.0

        # Choose a dominant 'r' variant for this word or sample naturally
        word_r_choice = random.choice(self.r_variants)

        char_items = []
        max_above = 0
        max_below = 0

        prev_ch = None
        for ch in word:
            is_repeated = (ch == prev_ch)
            g_img, desc_ratio, high_ratio = self.get_char_glyph(ch, r_choice=word_r_choice, is_repeated=is_repeated)
            prev_ch = ch
            if g_img is None:
                continue

            gw = max(2, int(round(g_img.width * scale_factor)))
            gh = max(2, int(round(g_img.height * scale_factor)))
            scaled_g = g_img.resize((gw, gh), Image.LANCZOS)

            # Human micro-jitter + progressive hand fatigue slant
            total_rot = fatigue_slant
            if jitter:
                total_rot += random.uniform(-0.35, 0.35)

            if abs(total_rot) > 0.08:
                scaled_g = scaled_g.rotate(total_rot, resample=Image.BICUBIC, expand=True)
                gw, gh = scaled_g.size

            descender_px = int(gh * desc_ratio)
            high_px = int(target_cap_height * high_ratio)

            above_baseline = gh - descender_px + high_px
            below_baseline = descender_px

            char_items.append({
                "img": scaled_g,
                "w": gw,
                "h": gh,
                "above": above_baseline,
                "below": below_baseline,
                "ch": ch
            })

            max_above = max(max_above, above_baseline)
            max_below = max(max_below, below_baseline)

        if not char_items:
            return None

        base_ref_y = max_above + 4

        # Assemble characters with realistic handwriting kerning and line slope alignment
        curr_x = 0
        placements = []
        for i, item in enumerate(char_items):
            kern = int(random.uniform(-1.5, 1.5))
            if i > 0:
                prev_c = char_items[i - 1]["ch"]
                if prev_c in [".", ",", ":", ";", "!"]:
                    kern += int(12 * scale_factor)
                elif prev_c in ["f", "t", "r"]:
                    kern -= int(2 * scale_factor)
                elif (prev_c.lower(), item["ch"].lower()) in COMMON_LIGATURE_PAIRS:
                    # Organic cursive ligature connection: kiss adjacent letter strokes
                    kern -= int(3.0 * scale_factor)

            char_x = max(0, curr_x + kern)
            bj = random.uniform(-0.65, 0.65) if jitter else 0.0
            slope_dy = line_slope * char_x
            char_y = int(base_ref_y - item["above"] + slope_dy + bj)

            placements.append((item["img"], char_x, char_y))
            # Natural stroke overlap
            curr_x = char_x + item["w"] - int(3 * scale_factor)

        all_min_y = min(cy for _, _, cy in placements)
        all_max_y = max(cy + img.height for img, _, cy in placements)

        y_pad = max(4, 4 - all_min_y)
        total_w = curr_x + 8
        total_h = (all_max_y - all_min_y) + y_pad + 8

        word_img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        for g_img, cx, cy in placements:
            word_img.paste(g_img, (cx, cy + y_pad), g_img)

        # Apply ballpoint ink micro-physics (micro-texture and turnaround pooling)
        word_img = apply_ballpoint_micro_physics(word_img)

        # Baseline position at the word's initial X coordinate (char_x = 0)
        baseline_y = base_ref_y + y_pad
        return word_img, baseline_y


class TemplateProfile:
    """Represents a notebook page template with detected 2D ruled line curves and margins."""
    def __init__(self, path: Path, side: str, line_curves: list, left_margin: int, right_margin: int, line_spacing: int):
        self.path = path
        self.side = side
        self.line_curves = line_curves  # list of (full_xs, full_ys) numpy arrays for each ruled line
        self.num_lines = len(line_curves)
        self.ruled_lines = [int(ys[len(ys) // 2]) for _, ys in line_curves]  # center baseline for backward-compat
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.line_spacing = line_spacing
        # Proportion: capital letter height is ~55% of line spacing
        self.target_cap_h = int(line_spacing * 0.55)

    def get_line_y(self, line_idx: int, x: int) -> int:
        idx = min(max(0, line_idx), self.num_lines - 1)
        xs, ys = self.line_curves[idx]
        return int(np.interp(x, xs, ys))

    def get_line_slope(self, line_idx: int, x: int) -> float:
        idx = min(max(0, line_idx), self.num_lines - 1)
        xs, ys = self.line_curves[idx]
        y1 = np.interp(x - 50, xs, ys)
        y2 = np.interp(x + 50, xs, ys)
        return float((y2 - y1) / 100.0)


def analyze_template_profile(template_path: Path, page_side: str = "left") -> TemplateProfile:
    """
    Analyzes template photo to accurately track 2D horizontal lines and margins across columns,
    intelligently ignoring the printed top header ('Date: ...') and completely clearing spiral wire bindings.
    """
    img_bgr = cv2.imread(str(template_path))
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Margin offsets:
    # On Left pages, outer margin is on the left; spiral is on the right side.
    # On Right pages, spiral is on the left side; outer margin is on the right.
    if page_side == "left":
        left_m = int(w * 0.065)  # Matches user handwriting start position (~0.06*w)
        right_m = int(w * 0.78)  # Natural margin before right-side spiral coils
    else:
        left_m = int(w * 0.17)   # Kept comfortably close to spiral (~0.17*w) while clearing coils & holes
        right_m = int(w * 0.92)

    xc = int(w * 0.50)
    strip_c = np.mean(gray[:, xc - 25:xc + 25], axis=1)

    expected_spacing = max(40, int(h / 32))
    min_dist = max(30, int(expected_spacing * 0.75))

    peaks, _ = find_peaks(-strip_c, distance=min_dist, prominence=1.2)
    # Strictly exclude top header (top 12.5%) and bottom 5%
    valid_c = [int(y) for y in peaks if int(h * 0.125) <= y <= int(h * 0.94)]

    if len(valid_c) >= 5:
        spacing = int(np.median(np.diff(valid_c)))
        # Drop header line if abnormally spaced
        if len(valid_c) > 2 and (valid_c[1] - valid_c[0]) > 1.35 * spacing:
            valid_c = valid_c[1:]
    else:
        spacing = int(h / 30)
        valid_c = list(range(int(h * 0.14), int(h * 0.92), spacing))

    # Track 2D line curves across the writable width from center outwards
    sample_xs = list(range(left_m, right_m + 1, 250))
    line_curves = []

    for line_idx in range(len(valid_c)):
        y_curr = valid_c[line_idx]

        # Walk left from center
        left_xs = [x for x in sample_xs if x <= xc][::-1]
        left_ys = []
        y_walk = y_curr
        for x in left_xs:
            window = gray[max(0, y_walk - 25):min(h, y_walk + 25), max(0, x - 15):min(w, x + 15)]
            y_local = np.argmin(np.mean(window, axis=1)) - 25
            y_walk += y_local
            left_ys.append(y_walk)

        # Walk right from center
        right_xs = [x for x in sample_xs if x > xc]
        right_ys = []
        y_walk = y_curr
        for x in right_xs:
            window = gray[max(0, y_walk - 25):min(h, y_walk + 25), max(0, x - 15):min(w, x + 15)]
            y_local = np.argmin(np.mean(window, axis=1)) - 25
            y_walk += y_local
            right_ys.append(y_walk)

        full_xs = np.array(left_xs[::-1] + right_xs)
        full_ys = np.array(left_ys[::-1] + right_ys)
        line_curves.append((full_xs, full_ys))

    return TemplateProfile(
        path=template_path,
        side=page_side,
        line_curves=line_curves,
        left_margin=left_m,
        right_margin=right_m,
        line_spacing=spacing
    )


def discover_templates(templates_dir: Path) -> tuple:
    """Discovers left and right template photos in data/blank_templates/."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = [p for p in templates_dir.glob("*") if p.suffix.lower() in valid_exts]

    left_pages = sorted([p for p in all_files if "left" in p.name.lower()])
    right_pages = sorted([p for p in all_files if "right" in p.name.lower()])

    if not left_pages and not right_pages:
        if all_files:
            left_pages = [all_files[0]]
            right_pages = [all_files[1 % len(all_files)]]
        else:
            raise FileNotFoundError(f"No notebook templates found in {templates_dir}")

    if not left_pages:
        left_pages = right_pages
    if not right_pages:
        right_pages = left_pages

    return left_pages, right_pages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Authentic Personal Handwriting Page Renderer (Dual-Page Notebook)"
    )
    parser.add_argument("--pdf", "--input", "--file", dest="input_file", type=str, default=None,
                        help="Input PDF or text document (auto-detects assig.txt if omitted)")
    parser.add_argument("--templates_dir", type=str, default="data/blank_templates",
                        help="Directory with blank notebook page photos")
    parser.add_argument("--output", type=str, default="output_pages/final_handwritten.pdf",
                        help="Path for compiled multi-page handwritten PDF")
    parser.add_argument("--r_style", type=str, default="mix", choices=["mix", "print", "hook", "cursive"],
                        help="Lowercase 'r' variation preference (mix, print, hook, cursive)")
    parser.add_argument("--date", type=str, default="18 / 11 / 2026",
                        help="Handwritten date string to auto-fill onto notebook Date header")
    parser.add_argument("--no_page_numbers", action="store_true",
                        help="Disable handwritten corner page numbering")
    parser.add_argument("--no_optics", action="store_true",
                        help="Disable mobile camera lens vignette and spiral contact ambient shadow")
    parser.add_argument("--no_fatigue", action="store_true",
                        help="Disable natural handwriting fatigue progression across pages")
    parser.add_argument("--no_correction", action="store_true",
                        help="Disable authentic human neat strike-through correction")
    parser.add_argument("--use_diffusion", action="store_true",
                        help="Use DiffusionPen model instead of authentic personal glyph engine")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Locate and validate input document
    input_path = find_input_document(args.input_file)
    templates_dir = (SCRIPT_DIR / args.templates_dir).resolve()
    output_path = (SCRIPT_DIR / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_templates, right_templates = discover_templates(templates_dir)

    print("\n" + "=" * 70)
    print(" Authentic Handwriting Page Renderer (Dual-Page Spiral Notebook)")
    print("=" * 70)
    print(f"Input Document   : {input_path.name}")
    print(f"Left Templates   : {[p.name for p in left_templates]}")
    print(f"Right Templates  : {[p.name for p in right_templates]}")
    print(f"Lowercase 'r'    : {args.r_style.upper()} (User Variations: print, hook, cursive)")
    print(f"Ink Type         : Authentic Royal Blue Ballpoint Pen (Physical Multiply Blending)")
    print(f"Date Header      : '{args.date}' (Auto-filled on printed Date line)")
    print(f"Corner Numbers   : {'Disabled' if args.no_page_numbers else 'Enabled (Page 1, Page 2, ...)'}")
    print(f"Mobile Optics    : {'Disabled' if args.no_optics else 'Enabled (Lens Vignette + Spiral Ambient Shadow)'}")
    print(f"Hand Fatigue     : {'Disabled' if args.no_fatigue else 'Enabled (Progressive Slant & Space Relaxation)'}")
    print(f"Human Correction : {'Disabled' if args.no_correction else 'Enabled (Authentic Ballpoint Strike-Through)'}")
    print(f"Output PDF       : {output_path}")
    print("=" * 70 + "\n")

    # 2. Extract content lines
    content_lines = extract_content_lines(input_path)
    if not content_lines:
        print("[Error] No readable text found in document.")
        return
    print(f"[1/4] Extracted {len(content_lines)} content lines from document.")

    # 3. Analyze templates
    print("[2/4] Detecting horizontal ruled lines and margins on notebook templates...")
    template_profiles = {}
    for p in left_templates:
        prof = analyze_template_profile(p, page_side="left")
        template_profiles[p] = prof
        print(f"  - (LEFT)  {p.name}: {len(prof.ruled_lines)} lines | Spacing: {prof.line_spacing}px | First line: y={prof.ruled_lines[0]}")
    for p in right_templates:
        prof = analyze_template_profile(p, page_side="right")
        template_profiles[p] = prof
        print(f"  - (RIGHT) {p.name}: {len(prof.ruled_lines)} lines | Spacing: {prof.line_spacing}px | First line: y={prof.ruled_lines[0]}")

    # 4. Initialize Handwriting Engine
    engine = AuthenticHandwritingEngine(glyph_base_dir=GLYPH_DIR, r_mode=args.r_style)

    # 5. Dual-Page Writing Loop (Left <-> Right)
    print("\n[3/4] Writing document onto notebook pages (alternating Left <-> Right)...")
    rendered_pages = []
    left_idx = 0
    right_idx = 0

    # Calibrated coordinates for printed Date header across template photos
    date_coords = {
        "left_page.jpg": (920, 558, 38),
        "left_page2.jpg": (780, 525, 48),
        "right_page.jpg": (1270, 615, 55),
        "right_page2.jpg": (1420, 485, 44),
    }

    def switch_page(side: str):
        nonlocal left_idx, right_idx
        if side == "left":
            t_path = left_templates[left_idx % len(left_templates)]
            left_idx += 1
        else:
            t_path = right_templates[right_idx % len(right_templates)]
            right_idx += 1

        prof = template_profiles[t_path]
        page_arr = np.array(Image.open(t_path).convert("RGB"))
        page_num = len(rendered_pages) + 1
        print(f"\n  >>> [Page {page_num}] Switched to {side.upper()} Notebook Page ({t_path.name})")

        # Auto-fill handwritten Date header on printed Date line
        if args.date and t_path.name in date_coords:
            dx, dy, d_cap_h = date_coords[t_path.name]
            jx = dx + random.randint(-4, 4)
            jy = dy + random.randint(-2, 2)
            date_res = engine.render_word(args.date, target_cap_height=d_cap_h)
            if date_res:
                d_img, d_base = date_res
                d_paste_y = int(jy - d_base)
                blend_ink_onto_page(page_arr, d_img, jx, d_paste_y)

        # Handwritten corner page numbering
        if not args.no_page_numbers:
            draw_page_number(page_arr, engine, page_num=page_num, page_side=side, t_name=t_path.name)

        return page_arr, prof, side

    def finalize_page(page_arr: np.ndarray, page_side: str) -> Image.Image:
        if not args.no_optics:
            page_arr = apply_photorealistic_optics(page_arr, page_side=page_side)
        return Image.fromarray(page_arr)

    current_page_arr, current_prof, current_side = switch_page("left")
    current_line_idx = 0
    total_words = 0
    correction_done = False
    total_est_pages = 5

    for line_idx, line in enumerate(tqdm(content_lines, desc="Writing Document", unit="line")):
        # Calculate natural page fatigue progression
        page_progress = min(1.0, len(rendered_pages) / float(max(1, total_est_pages - 1)))
        page_slant = 0.0 if args.no_fatigue else (page_progress * 0.70)  # 0.0 to 0.70 deg forward drift
        fatigue_spacing_factor = 1.0 if args.no_fatigue else (1.0 + page_progress * 0.045)
        fatigue_wander_scale = 1.0 if args.no_fatigue else (1.0 + page_progress * 0.30)

        # Check if line is a bullet item or section title
        is_bullet = line.startswith("•") or line.startswith("-") or line.startswith("*")
        is_major_heading = bool(re.match(r'^[1-9]\.\s+[A-Za-z]', line)) and len(line) < 60
        is_named_section = line.strip() in ["Allowed Items", "Disallowed Items", "Tie-Breaker Policy"]
        is_main_title = (line_idx == 0 and "RULEBOOK" in line.upper())
        is_section_title = is_major_heading or is_named_section

        # 1. Orphan Section Header Prevention (Keep with Next):
        remaining_lines = current_prof.num_lines - current_line_idx
        if is_section_title and remaining_lines <= 4 and current_line_idx > 0:
            rendered_pages.append(finalize_page(current_page_arr, current_side))
            next_side = "right" if current_side == "left" else "left"
            current_page_arr, current_prof, current_side = switch_page(next_side)
            current_line_idx = 0

        # Extra spacing before major section headings
        if is_section_title and current_line_idx > 0:
            current_line_idx += 1

        # Check page overflow before starting new item
        if current_line_idx >= current_prof.num_lines:
            rendered_pages.append(finalize_page(current_page_arr, current_side))
            next_side = "right" if current_side == "left" else "left"
            current_page_arr, current_prof, current_side = switch_page(next_side)
            current_line_idx = 0

        # Starting X for this line with Organic Left Margin Drift
        indent = int(current_prof.line_spacing * 0.45) if is_bullet else 0
        if is_section_title:
            indent = int(current_prof.line_spacing * 0.20)

        # Human wrist margin drift down the page (±4-8px wander)
        margin_wander = int((math.sin(current_line_idx * 0.38 + len(rendered_pages) * 1.5) * 7.5 + random.uniform(-2.5, 2.5)) * fatigue_wander_scale)
        current_x = current_prof.left_margin + indent + margin_wander

        # 2. Text Hierarchy: Make text after title/colon slightly smaller (~82%)
        has_colon_label = (":" in line) and (not line.isupper()) and (not is_section_title)
        is_after_colon = False

        title_start_x = current_x
        title_end_x = current_x
        title_line_idx = current_line_idx

        words = line.split(" ")
        for w_idx, word in enumerate(words):
            if not word.strip():
                continue
            clean_w = word.strip()

            # Target cap height: normal for title/label, slightly smaller (~82%) for text after title
            if has_colon_label and is_after_colon:
                word_cap_h = int(current_prof.target_cap_h * 0.82)
            else:
                word_cap_h = current_prof.target_cap_h

            # Calculate local line slope at current_x
            local_slope = current_prof.get_line_slope(current_line_idx, int(current_x))

            # Authentic human slip-up / strike-through correction (on Page 3 under Zero Tolerance)
            if not args.no_correction and not correction_done and len(rendered_pages) >= 2 and clean_w.lower() == "cheat,":
                res = engine.render_word(clean_w, target_cap_height=word_cap_h, line_slope=local_slope, fatigue_slant=page_slant)
                if res is not None:
                    w_img, w_base = res
                    line_y = current_prof.get_line_y(current_line_idx, int(current_x))
                    paste_y = int(line_y - w_base)
                    blend_ink_onto_page(current_page_arr, w_img, int(current_x), paste_y)
                    strike_y = int(line_y - w_base + w_img.height * 0.58)
                    draw_handwritten_strikethrough(current_page_arr, int(current_x), int(current_x + w_img.width), strike_y, slope=local_slope)
                    current_x += w_img.width + int(12 * (word_cap_h / 80.0))
                    correction_done = True
                    continue

            # Render handwritten word with authentic glyphs following local line slope
            res = engine.render_word(clean_w, target_cap_height=word_cap_h, line_slope=local_slope, fatigue_slant=page_slant)
            if res is None:
                continue
            w_img, w_base = res
            word_w = w_img.width

            # Smart End-of-Line Margin Squeeze or Line Wrap
            if current_x + word_w > current_prof.right_margin:
                overflow = (current_x + word_w) - current_prof.right_margin
                squeeze_factor = (current_prof.right_margin - current_x) / float(word_w)

                # If word barely overflows (by <= 20%) and is short/medium length, squeeze slightly to fit
                if 0.84 <= squeeze_factor < 1.0 and len(clean_w) <= 8 and (current_x + word_w) <= current_prof.right_margin + 60:
                    sq_w = max(2, int(round(word_w * squeeze_factor)))
                    w_img = w_img.resize((sq_w, w_img.height), Image.LANCZOS)
                    word_w = sq_w
                else:
                    # Wrap cleanly to next line
                    current_line_idx += 1
                    if current_line_idx >= current_prof.num_lines:
                        rendered_pages.append(finalize_page(current_page_arr, current_side))
                        next_side = "right" if current_side == "left" else "left"
                        current_page_arr, current_prof, current_side = switch_page(next_side)
                        current_line_idx = 0

                    # Wrapped line indent (align under bullet content)
                    wrap_indent = int(current_prof.line_spacing * 0.70) if is_bullet else 0
                    wrap_wander = int((math.sin(current_line_idx * 0.38 + len(rendered_pages) * 1.5) * 7.5 + random.uniform(-2.5, 2.5)) * fatigue_wander_scale)
                    current_x = current_prof.left_margin + wrap_indent + wrap_wander + random.randint(-2, 2)

                    # Re-calculate slope and render for wrapped line position
                    local_slope = current_prof.get_line_slope(current_line_idx, int(current_x))
                    res = engine.render_word(clean_w, target_cap_height=word_cap_h, line_slope=local_slope, fatigue_slant=page_slant)
                    if res is not None:
                        w_img, w_base = res
                        word_w = w_img.width

            # Align word directly onto the notebook ruled line at its exact X position
            line_y = current_prof.get_line_y(current_line_idx, int(current_x))
            v_jitter = random.uniform(-0.5, 0.5)
            paste_y = int(line_y - w_base + v_jitter)

            # Physically blend translucent royal blue ink onto notebook paper
            blend_ink_onto_page(current_page_arr, w_img, int(current_x), paste_y)

            # Track end of title words for freehand underline
            title_end_x = int(current_x + word_w)

            # Advance X with natural, comfortable word spacing (with slight fatigue loosening)
            space_jitter = random.uniform(0.90, 1.25)
            char_w = max(10, word_w // max(1, len(clean_w)))
            space_w = int(char_w * 1.35 * space_jitter * fatigue_spacing_factor)
            current_x += word_w + space_w
            total_words += 1

            # Switch to smaller size for all subsequent words once colon is encountered
            if ":" in clean_w:
                is_after_colon = True

        # Freehand ballpoint underlines under major section headers & main document title
        if is_section_title or is_main_title:
            u_line_y = current_prof.get_line_y(title_line_idx, int(title_start_x))
            u_y = u_line_y + int(current_prof.line_spacing * 0.18)
            u_slope = current_prof.get_line_slope(title_line_idx, int(title_start_x))
            draw_handwritten_underline(current_page_arr, title_start_x, title_end_x, u_y, slope=u_slope)

        # Move to next line for next paragraph/entry
        current_line_idx += 1

    # Append final page
    rendered_pages.append(finalize_page(current_page_arr, current_side))

    # 6. Save individual page images and compile multi-page PDF
    print(f"\n[4/4] Compiling {len(rendered_pages)} notebook page(s) into {output_path.name}...")
    for i, page_img in enumerate(rendered_pages):
        page_png = OUTPUT_DIR / f"page_{i + 1:03d}.png"
        page_img.save(str(page_png), quality=95)
        print(f"  - Saved Page Image: {page_png.name}")

    first_page = rendered_pages[0]
    rest_pages = rendered_pages[1:] if len(rendered_pages) > 1 else []
    first_page.save(str(output_path), save_all=True, append_images=rest_pages, resolution=300.0)

    print("\n" + "=" * 70)
    print(" Pipeline Completed Successfully!")
    print(f"Total Words Written   : {total_words}")
    print(f"Total Pages Generated : {len(rendered_pages)}")
    print(f"Final Compiled PDF    : {output_path}")
    print(f"Page PNG Outputs      : {OUTPUT_DIR}/page_XXX.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

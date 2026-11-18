#!/usr/bin/env python3
"""
3_render_on_template.py

Authentic Personal Handwriting & Notebook Page Compositing Pipeline:
1. Automatically discovers input document (PDF, TXT, MD) in input_pdfs/ (e.g. assig.txt).
2. Uses the user's authentic personal handwriting glyph library (royal blue ballpoint ink,
   custom letterforms, all 3 variations of lowercase 'r': print, hooked, and cursive flat-top).
3. Automatically detects horizontal ruled lines and margins on blank notebook page photos
   (skipping the printed top 'Date: ...' header).
4. Alternates seamlessly between Left and Right notebook pages (left_page <-> right_page).
5. Adds subtle human micro-variations (baseline jitter, rotation jitter, natural kerning).
6. Compiles final output into output_pages/final_handwritten.pdf and individual page images.
"""

import os
import sys
import re
import math
import random
import argparse
from pathlib import Path
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
        content = content.replace("\ufeff", "").replace("â€¢", "•").replace("\r\n", "\n")
        raw_lines = content.split("\n")
        for raw_line in raw_lines:
            clean = raw_line.strip()
            # Normalize tab separated bullet items
            clean = clean.replace("\t", " ")
            clean = " ".join(clean.split())
            if clean:
                lines.append(clean)

    return lines


class AuthenticHandwritingEngine:
    """
    Personal handwriting generation engine using authentic royal blue ballpoint pen glyphs
    extracted directly from the user's handwriting specimen.
    """
    def __init__(self, glyph_base_dir: Path = GLYPH_DIR, r_mode: str = "mix"):
        self.glyph_base_dir = Path(glyph_base_dir)
        self.r_mode = r_mode
        self.upper_glyphs = {}
        self.lower_glyphs = {}
        self.digit_glyphs = {}
        self.symbol_glyphs = {}
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
            "'": "squote", "•": "bullet", "*": "bullet", "—": "dash"
        }

        self.r_variants = ["r_print", "r_hook", "r_cursive"]

    def load_glyphs(self):
        upper_dir = self.glyph_base_dir / "upper"
        lower_dir = self.glyph_base_dir / "lower"
        digit_dir = self.glyph_base_dir / "digits"
        sym_dir = self.glyph_base_dir / "symbols"

        if upper_dir.exists():
            for p in upper_dir.glob("*.png"):
                self.upper_glyphs[p.stem] = Image.open(p).convert("RGBA")
        if lower_dir.exists():
            for p in lower_dir.glob("*.png"):
                self.lower_glyphs[p.stem] = Image.open(p).convert("RGBA")
        if digit_dir.exists():
            for p in digit_dir.glob("*.png"):
                self.digit_glyphs[p.stem] = Image.open(p).convert("RGBA")
        if sym_dir.exists():
            for p in sym_dir.glob("*.png"):
                self.symbol_glyphs[p.stem] = Image.open(p).convert("RGBA")

        total = len(self.upper_glyphs) + len(self.lower_glyphs) + len(self.digit_glyphs) + len(self.symbol_glyphs)
        print(f"[Handwriting Engine] Loaded {total} authentic user glyphs (Royal Blue Ballpoint Ink).")
        print(f"  - Uppercase letters : {len(self.upper_glyphs)} (A-Z)")
        print(f"  - Lowercase letters : {len(self.lower_glyphs)} (a-z + 3 'r' styles)")
        print(f"  - Digits            : {len(self.digit_glyphs)} (0-9)")
        print(f"  - Symbols           : {len(self.symbol_glyphs)}")

    def get_char_glyph(self, ch: str, r_choice: str = None) -> tuple:
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
            img = self.lower_glyphs.get(choice, self.lower_glyphs.get("r"))
            return img, 0.0, 0.0

        if ch in self.punct_map:
            key = self.punct_map[ch]
            img = self.symbol_glyphs.get(key)
            if img:
                return img, self.descenders.get(ch, 0.0), self.high_puncts.get(ch, 0.0)

        if ch.isupper() and ch in self.upper_glyphs:
            return self.upper_glyphs[ch], 0.0, 0.0

        if ch.islower() and ch in self.lower_glyphs:
            return self.lower_glyphs[ch], self.descenders.get(ch, 0.0), 0.0

        if ch.isdigit() and ch in self.digit_glyphs:
            return self.digit_glyphs[ch], 0.0, 0.0

        # Case fallbacks
        if ch.upper() in self.upper_glyphs:
            return self.upper_glyphs[ch.upper()], 0.0, 0.0
        if ch.lower() in self.lower_glyphs:
            return self.lower_glyphs[ch.lower()], self.descenders.get(ch.lower(), 0.0), 0.0

        return None, 0.0, 0.0

    def render_word(self, word: str, target_cap_height: int = 110, line_slope: float = 0.0, jitter: bool = True) -> tuple:
        """
        Renders a single handwritten word with authentic glyphs and natural handwriting flow.
        Follows the local notebook line slope so every character sits flush on the ruled line.
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

        for ch in word:
            g_img, desc_ratio, high_ratio = self.get_char_glyph(ch, r_choice=word_r_choice)
            if g_img is None:
                continue

            gw = max(2, int(g_img.width * scale_factor))
            gh = max(2, int(g_img.height * scale_factor))
            scaled_g = g_img.resize((gw, gh), Image.LANCZOS)

            # Human micro-jitter
            if jitter:
                rot = random.uniform(-0.35, 0.35)
                scaled_g = scaled_g.rotate(rot, resample=Image.BICUBIC, expand=True)
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
                prev_ch = char_items[i - 1]["ch"]
                if prev_ch in [".", ",", ":", ";", "!"]:
                    kern += int(12 * scale_factor)
                elif prev_ch in ["f", "t", "r"]:
                    kern -= int(2 * scale_factor)

            char_x = max(0, curr_x + kern)
            bj = random.uniform(-0.75, 0.75) if jitter else 0.0
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
    strip_c = np.mean(gray[:, xc-25:xc+25], axis=1)

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
            window = gray[max(0, y_walk-25):min(h, y_walk+25), max(0, x-15):min(w, x+15)]
            y_local = np.argmin(np.mean(window, axis=1)) - 25
            y_walk += y_local
            left_ys.append(y_walk)

        # Walk right from center
        right_xs = [x for x in sample_xs if x > xc]
        right_ys = []
        y_walk = y_curr
        for x in right_xs:
            window = gray[max(0, y_walk-25):min(h, y_walk+25), max(0, x-15):min(w, x+15)]
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
        # Fallback to any images
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
    print(f"Ink Type         : Authentic Royal Blue Ballpoint Pen")
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

    def switch_page(side: str):
        nonlocal left_idx, right_idx
        if side == "left":
            t_path = left_templates[left_idx % len(left_templates)]
            left_idx += 1
        else:
            t_path = right_templates[right_idx % len(right_templates)]
            right_idx += 1

        prof = template_profiles[t_path]
        img = Image.open(t_path).convert("RGB")
        page_num = len(rendered_pages) + 1
        print(f"\n  >>> [Page {page_num}] Switched to {side.upper()} Notebook Page ({t_path.name})")
        return img, prof, side

    current_page, current_prof, current_side = switch_page("left")
    current_line_idx = 0
    current_x = current_prof.left_margin
    total_words = 0

    for line_idx, line in enumerate(tqdm(content_lines, desc="Writing Document", unit="line")):
        # Check if line is a bullet item or section title
        is_bullet = line.startswith("•") or line.startswith("-") or line.startswith("*")
        is_major_heading = bool(re.match(r'^[1-9]\.\s+[A-Za-z]', line)) and len(line) < 60
        is_named_section = line.strip() in ["Allowed Items", "Disallowed Items", "Tie-Breaker Policy"]
        is_section_title = is_major_heading or is_named_section

        # 1. Orphan Section Header Prevention (Keep with Next):
        # A human writer would never write just a major section title (e.g. '3. Exam Format Structure')
        # on the very last 1-3 lines of a page. If <= 4 lines remain, start it cleanly at the top of the next page!
        # Normal body/bullet text belonging to the current section continues on the current page.
        remaining_lines = current_prof.num_lines - current_line_idx
        if is_section_title and remaining_lines <= 4 and current_line_idx > 0:
            rendered_pages.append(current_page)
            next_side = "right" if current_side == "left" else "left"
            current_page, current_prof, current_side = switch_page(next_side)
            current_line_idx = 0

        # Extra spacing before major section headings
        if is_section_title and current_line_idx > 0:
            current_line_idx += 1

        # Check page overflow before starting new item
        if current_line_idx >= current_prof.num_lines:
            rendered_pages.append(current_page)
            next_side = "right" if current_side == "left" else "left"
            current_page, current_prof, current_side = switch_page(next_side)
            current_line_idx = 0

        # Starting X for this line
        indent = int(current_prof.line_spacing * 0.45) if is_bullet else 0
        if is_section_title:
            indent = int(current_prof.line_spacing * 0.2)
        current_x = current_prof.left_margin + indent

        # 2. Text Hierarchy: Make text after title/colon slightly smaller (~82%)
        # (e.g. 'Competition Name:' is normal size, 'Pseudo War' is a little smaller)
        has_colon_label = (":" in line) and (not line.isupper()) and (not is_section_title)
        is_after_colon = False

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

            # Render handwritten word with authentic glyphs following local line slope
            res = engine.render_word(clean_w, target_cap_height=word_cap_h, line_slope=local_slope)
            if res is None:
                continue
            w_img, w_base = res
            word_w = w_img.width

            # Line wrap check
            if current_x + word_w > current_prof.right_margin:
                current_line_idx += 1
                # Check page overflow
                if current_line_idx >= current_prof.num_lines:
                    rendered_pages.append(current_page)
                    next_side = "right" if current_side == "left" else "left"
                    current_page, current_prof, current_side = switch_page(next_side)
                    current_line_idx = 0

                # Wrapped line indent (align under bullet content)
                wrap_indent = int(current_prof.line_spacing * 0.7) if is_bullet else 0
                current_x = current_prof.left_margin + wrap_indent + random.randint(-3, 3)

                # Re-calculate slope and render for wrapped line position
                local_slope = current_prof.get_line_slope(current_line_idx, int(current_x))
                res = engine.render_word(clean_w, target_cap_height=word_cap_h, line_slope=local_slope)
                if res is not None:
                    w_img, w_base = res
                    word_w = w_img.width

            # Align word directly onto the notebook ruled line at its exact X position
            line_y = current_prof.get_line_y(current_line_idx, int(current_x))
            v_jitter = random.uniform(-0.5, 0.5)
            paste_y = int(line_y - w_base + v_jitter)

            # Paste transparent word onto notebook page
            current_page.paste(w_img, (int(current_x), paste_y), mask=w_img)

            # Advance X with natural, comfortable word spacing
            space_jitter = random.uniform(0.90, 1.25)
            char_w = max(10, word_w // max(1, len(clean_w)))
            space_w = int(char_w * 1.35 * space_jitter)
            current_x += word_w + space_w
            total_words += 1

            # Switch to smaller size for all subsequent words in this item once colon is encountered
            if ":" in clean_w:
                is_after_colon = True

        # Move to next line for next paragraph/entry
        current_line_idx += 1

    # Append final page
    rendered_pages.append(current_page)

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

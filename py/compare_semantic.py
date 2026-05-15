#!/usr/bin/env python3
"""
4-panel semantic comparison for KIST raw_data_curve
  Panel 1: GT RGB
  Panel 2: 3DGS RGB render
  Panel 3: GT Semantic (InverseForm _comp.png)
  Panel 4: 3DGS Semantic render

Output: /media/ms/WD_BLACK_4TB/KIST/raw_data_curve/compare_semantic/CAM_FRONT_XXXXXX.png
"""

import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BASE = Path("/media/ms/WD_BLACK_4TB/KIST/raw_data_curve")

PATHS = {
    "gt_rgb":      BASE / "images",
    "render_rgb":  BASE / "export/test/ours_30000/render",
    "render_sem":  BASE / "export/test/ours_30000/semantic",
    "gt_sem":      BASE / "recon_HUGSIM/semantics",
}

LABELS = {
    "gt_rgb": "GT RGB",
    "render_rgb": "3DGS RGB",
    "gt_sem": "GT Semantic",
    "render_sem": "3DGS Semantic",
}


def add_label(img: Image.Image, text: str, font_size: int = 20) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, font_size + 6], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return img


def make_panel(frame_str: str, cam: str) -> Image.Image:
    gt_rgb_path  = PATHS["gt_rgb"]  / cam / f"{frame_str}.jpg"
    render_path  = PATHS["render_rgb"] / f"{cam}_{frame_str}.png"
    sem_path     = PATHS["render_sem"] / f"{cam}_{frame_str}.png"
    gt_sem_path  = PATHS["gt_sem"] / cam / f"{frame_str}_comp.png"

    imgs = []
    sources = [
        (gt_rgb_path,  LABELS["gt_rgb"]),
        (render_path,  LABELS["render_rgb"]),
        (gt_sem_path,  LABELS["gt_sem"]),
        (sem_path,     LABELS["render_sem"]),
    ]

    ref_size = None
    for path, label in sources:
        if path.exists():
            img = Image.open(path).convert("RGB")
            if ref_size is None:
                ref_size = img.size
            else:
                img = img.resize(ref_size, Image.LANCZOS)
        else:
            print(f"  MISSING: {path}")
            w, h = ref_size if ref_size else (800, 450)
            img = Image.new("RGB", (w, h), (40, 40, 40))
            draw = ImageDraw.Draw(img)
            draw.text((10, h // 2), f"MISSING\n{path.name}", fill=(200, 50, 50))
        imgs.append(add_label(img.copy(), label))

    if ref_size is None:
        ref_size = (800, 450)

    W, H = ref_size
    canvas = Image.new("RGB", (W * 2, H * 2 + 4), (20, 20, 20))
    canvas.paste(imgs[0], (0, 0))
    canvas.paste(imgs[1], (W, 0))
    canvas.paste(imgs[2], (0, H + 4))
    canvas.paste(imgs[3], (W, H + 4))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", default="CAM_FRONT", help="Camera name, or ALL")
    parser.add_argument("--n", type=int, default=20, help="Number of frames to sample per camera. Use 0 for all frames.")
    parser.add_argument("--out", default=str(BASE / "compare_semantic"), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT",
        "CAM_BACK",
    ] if args.cam.upper() == "ALL" else [args.cam]

    total_saved = 0
    for cam in cams:
        prefix = cam + "_"
        render_files = sorted([
            f for f in PATHS["render_rgb"].glob(f"{prefix}*.png")
            if f.stem[len(prefix):].isdigit()
        ])
        if not render_files:
            print(f"No render files found for {cam}")
            continue

        if args.n <= 0 or args.n >= len(render_files):
            selected = render_files
        else:
            indices = np.linspace(0, len(render_files) - 1, args.n, dtype=int)
            selected = [render_files[i] for i in indices]

        print(f"Camera : {cam}")
        print(f"Frames : {len(selected)} / {len(render_files)} total")
        print(f"Output : {out_dir}")

        for f in selected:
            frame_str = f.stem[len(prefix):]  # 000001
            out_path = out_dir / f"{f.stem}.png"
            panel = make_panel(frame_str, cam)
            panel.save(out_path)
            total_saved += 1
            print(f"  saved: {out_path.name}")

    print(f"\nDone. {total_saved} panels saved to {out_dir}")


if __name__ == "__main__":
    main()

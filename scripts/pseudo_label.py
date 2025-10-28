import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List

from PIL import Image, ImageOps
import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact inference preprocessing and model
from backend.infer import predict_mask  # noqa: E402


def list_images(src_dir: Path) -> List[Path]:
  exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
  files = []
  for p in sorted(src_dir.rglob("*")):
    if p.suffix.lower() in exts:
      files.append(p)
  return files


def sanitize_name(name: str) -> str:
  safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name)
  while "--" in safe:
    safe = safe.replace("--", "-")
  return safe.strip("-")


def main():
  ap = argparse.ArgumentParser(description="Generate pseudo-label masks for unlabeled EM images and add to training set.")
  ap.add_argument("--src_dir", type=str, required=True, help="Folder containing EM images to pseudo-label")
  ap.add_argument("--out_root", type=str, default=str(REPO_ROOT/"data"/"train"), help="Root split to write into (contains images/ and masks/)")
  ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for mask binarization")
  ap.add_argument("--tta", action="store_true", help="Use test-time augmentation for more robust masks")
  ap.add_argument("--max_side", type=int, default=1024, help="Resize larger images for faster inference")
  args = ap.parse_args()

  src_dir = Path(args.src_dir)
  out_root = Path(args.out_root)
  out_images = out_root/"images"
  out_masks = out_root/"masks"
  out_images.mkdir(parents=True, exist_ok=True)
  out_masks.mkdir(parents=True, exist_ok=True)

  imgs = list_images(src_dir)
  if not imgs:
    print("No images found in", src_dir)
    return

  # Reserve prefix to avoid colliding with existing dataset
  prefix = "pl_"
  existing = {p.name for p in out_images.glob("*.png")}

  added = 0
  for p in imgs:
    try:
      pil = Image.open(p)
      pil = ImageOps.exif_transpose(pil).convert("L")
      arr, mask = predict_mask(pil, thr=args.threshold, tta=args.tta, max_side=args.max_side)
      # Filename
      base = sanitize_name(p.stem)
      name = f"{prefix}{base}.png"
      # Ensure uniqueness
      idx = 1
      while name in existing:
        name = f"{prefix}{base}_{idx}.png"
        idx += 1
      existing.add(name)
      # Save
      Image.fromarray(arr).save(out_images/name)
      Image.fromarray(mask).save(out_masks/name)
      added += 1
      print(f"Added {name}")
    except Exception as e:
      print(f"Skip {p}: {e}")

  print(f"Done. Added {added} pseudo-labeled images to {out_root}.")


if __name__ == "__main__":
  main()



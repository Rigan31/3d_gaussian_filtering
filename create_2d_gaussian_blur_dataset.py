import os
import shutil
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ALL_SPLITS = ["train", "test", "val"]
NERF_CLASSES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]


def gaussian_blur_image(img_pil, sigma, kernel_size):
    """
    Apply Gaussian blur to a PIL image.
    Preserves alpha channel (RGBA) if present.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    has_alpha = img_pil.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img_pil.split()
        img_rgb = Image.merge("RGB", (r, g, b))
    else:
        img_rgb = img_pil.convert("RGB")

    img_np   = np.array(img_rgb)
    blurred  = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
    out_pil  = Image.fromarray(blurred, mode="RGB")

    if has_alpha:
        out_pil = out_pil.convert("RGBA")
        out_pil.putalpha(a)

    return out_pil


def copy_static_files(src_obj_dir, dst_obj_dir):
    """Copy non-image files (JSON, PLY) and create split folders."""
    os.makedirs(dst_obj_dir, exist_ok=True)
    for fname in os.listdir(src_obj_dir):
        src = os.path.join(src_obj_dir, fname)
        dst = os.path.join(dst_obj_dir, fname)
        if os.path.isfile(src) and Path(fname).suffix.lower() not in IMAGE_EXTS:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    for split in ALL_SPLITS:
        if os.path.isdir(os.path.join(src_obj_dir, split)):
            os.makedirs(os.path.join(dst_obj_dir, split), exist_ok=True)


def process_object(obj_name, src_obj_dir, dst_obj_dir, sigma, kernel_size):
    copy_static_files(src_obj_dir, dst_obj_dir)
    total = 0

    for split in ALL_SPLITS:
        split_src = os.path.join(src_obj_dir, split)
        split_dst = os.path.join(dst_obj_dir, split)
        if not os.path.isdir(split_src):
            continue
        os.makedirs(split_dst, exist_ok=True)

        files = sorted([
            f for f in os.listdir(split_src)
            if Path(f).suffix.lower() in IMAGE_EXTS
        ])
        if not files:
            continue

        print(f"  [{obj_name}/{split}]  {len(files)} images")
        for fname in tqdm(files, desc=f"    {split}", leave=False):
            img     = Image.open(os.path.join(split_src, fname))
            blurred = gaussian_blur_image(img, sigma, kernel_size)
            blurred.save(os.path.join(split_dst, fname))
            total += 1

    return total


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply 2D Gaussian blur to a NeRF Synthetic dataset"
    )
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sigma",      type=float, default=1.0)
    p.add_argument("--kernel",     type=int,   default=5)
    p.add_argument("--classes",    nargs="+",  default=NERF_CLASSES)
    return p.parse_args()


def main():
    args = parse_args()
    kernel = args.kernel if args.kernel % 2 == 1 else args.kernel + 1

    print("=" * 60)
    print("  2D Gaussian Blur Dataset")
    print("=" * 60)
    print(f"  Input   : {args.input_dir}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Sigma   : {args.sigma}")
    print(f"  Kernel  : {kernel}x{kernel}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "blur_params.json"), "w") as f:
        json.dump({"sigma": args.sigma, "kernel": kernel,
                   "input_dir": args.input_dir}, f, indent=2)

    total = 0
    for obj in args.classes:
        src = os.path.join(args.input_dir,  obj)
        dst = os.path.join(args.output_dir, obj)
        if not os.path.isdir(src):
            print(f"  [SKIP] {obj} not found")
            continue
        print(f"\n  {obj}")
        n = process_object(obj, src, dst, args.sigma, kernel)
        total += n
        print(f"  Done: {obj}  ({n} images)")

    print(f"\n  Total: {total} images blurred")
    print(f"  Saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
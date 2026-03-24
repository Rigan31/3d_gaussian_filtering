import os
import csv
import re
import argparse
import numpy as np
from pathlib import Path

import torch
import clip
from PIL import Image
from torchvision import transforms


NERF_CLASSES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]

CONDITIONS = {
    "adv_eps4_2d_gaussian_blur": ("D", "4.0", "2d_gaussian_blur"),
    "adv_eps8_2d_gaussian_blur": ("D", "8.0", "2d_gaussian_blur"),
}

CLIP_MEAN  = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD   = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = (224, 224)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

_clip_model    = None
_text_features = None
_clip_norm     = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
_to_tensor     = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def init_clip(device):
    global _clip_model, _text_features
    print("  Loading CLIP ViT-B/16 ...")
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    _clip_model = model
    prompts = [f"a photo of a {c}" for c in NERF_CLASSES]
    tokens  = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    _text_features = feats / feats.norm(dim=-1, keepdim=True)
    print(f"  Text features: {_text_features.shape}")


def clip_scores(img_pil, true_label, device):
    img_t = _to_tensor(img_pil.convert("RGB")).unsqueeze(0).to(device)
    img_n = _clip_norm(img_t)
    with torch.no_grad():
        feats  = _clip_model.encode_image(img_n)
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        logits = _clip_model.logit_scale.exp() * (feats @ _text_features.T)
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    true_idx  = NERF_CLASSES.index(true_label)
    true_conf = float(probs[true_idx])
    top3_idx  = torch.topk(probs, 3).indices.tolist()
    return true_conf, int(top3_idx[0] == true_idx), int(true_idx in top3_idx)


def compute_clip_metrics(render_dir, true_label, device):
    if not os.path.isdir(render_dir):
        return None, None, None
    files = sorted([
        f for f in os.listdir(render_dir)
        if Path(f).suffix.lower() in IMAGE_EXTS
    ])
    if not files:
        return None, None, None
    confs, top1s, top3s = [], [], []
    for fname in files:
        img = Image.open(os.path.join(render_dir, fname))
        c, t1, t3 = clip_scores(img, true_label, device)
        confs.append(c); top1s.append(t1); top3s.append(t3)
    return (round(float(np.mean(confs)), 4),
            round(float(np.mean(top1s)), 4),
            round(float(np.mean(top3s)), 4))


def read_benchmark_log(exp_run_dir):
    log_path = os.path.join(exp_run_dir, "benchmark_result.log")
    result   = {"psnr": None, "ssim": None,
                "n_gaussians": None, "training_time_min": None}
    if not os.path.exists(log_path):
        print(f"    [WARN] benchmark_result.log not found: {log_path}")
        return result
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            m = re.search(r"PSNR\s*:\s*([\d.]+)", line, re.IGNORECASE)
            if m: result["psnr"] = float(m.group(1))
            m = re.search(r"SSIM\s*:\s*([\d.]+)", line, re.IGNORECASE)
            if m: result["ssim"] = float(m.group(1))
            m = re.search(r"Max Gaussian.*?([\d.]+)\s*M", line, re.IGNORECASE)
            if m: result["n_gaussians"] = int(float(m.group(1)) * 1_000_000)
            m = re.search(r"Training time\s*:\s*([\d.]+)", line, re.IGNORECASE)
            if m: result["training_time_min"] = float(m.group(1))
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_root",   required=True)
    p.add_argument("--clean_data", required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    init_clip(args.device)

    fieldnames = [
        "condition", "condition_label", "epsilon", "defense_method",
        "object", "psnr", "ssim",
        "clip_conf", "clip_top1", "clip_top3",
        "n_gaussians", "training_time_min",
    ]
    rows = []

    for cond, (cond_label, eps, method) in CONDITIONS.items():
        cond_dir = os.path.join(args.log_root, cond)
        if not os.path.isdir(cond_dir):
            print(f"  [SKIP] Not found: {cond_dir}")
            continue

        print(f"\n  Condition : {cond}")
        print(f"  {'─' * 52}")

        for obj in NERF_CLASSES:
            exp_dir    = os.path.join(cond_dir, obj, "exp_run_1")
            render_dir = os.path.join(exp_dir, "render_comparison", "renders")

            if not os.path.isdir(exp_dir):
                print(f"  [SKIP] {obj}: exp_run_1 not found")
                continue

            print(f"  {obj} ...", end=" ", flush=True)

            bench = read_benchmark_log(exp_dir)
            clip_conf, clip_top1, clip_top3 = compute_clip_metrics(
                render_dir, obj, args.device
            )
            if clip_conf is None:
                print(f"[WARN: no renders] ", end="")

            rows.append({
                "condition":         cond,
                "condition_label":   cond_label,
                "epsilon":           eps,
                "defense_method":    method,
                "object":            obj,
                "psnr":              bench["psnr"]              or "",
                "ssim":              bench["ssim"]              or "",
                "clip_conf":         clip_conf                  if clip_conf is not None else "",
                "clip_top1":         clip_top1                  if clip_top1 is not None else "",
                "clip_top3":         clip_top3                  if clip_top3 is not None else "",
                "n_gaussians":       bench["n_gaussians"]       or "",
                "training_time_min": bench["training_time_min"] or "",
            })

            print(f"PSNR={bench['psnr']}  SSIM={bench['ssim']}  "
                  f"Top1={clip_top1}  N={bench['n_gaussians']}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Saved {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from torchvision import transforms



NERF_CLASSES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]

CLIP_MEAN  = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD   = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = (224, 224)

def load_clip_model(device: str = "cpu"):
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    return model


def build_text_features(class_names: list, clip_model, device: str) -> torch.Tensor:
    """Return normalised CLIP text embeddings [C, D]."""
    prompts = [f"a photo of a {c}" for c in class_names]
    tokens  = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)



_clip_norm = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
_to_tensor = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def classify_image(
    img_pil:       Image.Image,
    clip_model,
    text_features: torch.Tensor,   # [C, D] normalised
    device:        str = "cpu",
) -> dict:
    
    img_tensor = _to_tensor(img_pil.convert("RGB")).unsqueeze(0).to(device)
    img_norm   = _clip_norm(img_tensor)

    with torch.no_grad():
        image_features = clip_model.encode_image(img_norm)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp()
        logits      = logit_scale * (image_features @ text_features.T)  # [1, C]
        probs       = torch.softmax(logits, dim=-1).squeeze(0).cpu()    # [C]

    top5_probs, top5_idx = torch.topk(probs, 5)
    return {
        "probs":            probs.numpy(),
        "top1_idx":         top5_idx[0].item(),
        "top5_indices":     top5_idx.tolist(),
        "top5_probs":       top5_probs.tolist(),
        "top1_confidence":  top5_probs[0].item(),
    }



def load_nerf_image_paths(object_dir: str, split: str) -> list:
    """Return list of (abs_path, frame_idx) from transforms_{split}.json."""
    json_path = os.path.join(object_dir, f"transforms_{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path) as f:
        meta = json.load(f)
    paths = []
    for i, frame in enumerate(meta.get("frames", [])):
        rel = frame["file_path"].lstrip("./")
        for ext in [".png", ".jpg", ".jpeg", ""]:
            p = os.path.join(object_dir, rel + ext)
            if os.path.exists(p):
                paths.append((p, i))
                break
    return paths


def evaluate_object(
    object_dir:    str,
    object_name:   str,
    adv_split_dir: str,           # folder containing adv_*.png for this object/split
    clip_model,
    text_features: torch.Tensor,
    class_names:   list,
    split:         str = "train",
    device:        str = "cpu",
    verbose:       bool = True,
) -> dict:
    true_label_idx = class_names.index(object_name)
    image_paths    = load_nerf_image_paths(object_dir, split)

    orig_conf,  adv_conf  = [], []
    orig_top1,  adv_top1  = [], []
    orig_top5,  adv_top5  = [], []
    per_image_rows         = []

    for img_path, frame_idx in image_paths:
        stem = Path(img_path).stem

        # ── Original image ──
        orig_pil = Image.open(img_path)
        orig_res = classify_image(orig_pil, clip_model, text_features, device)

        orig_true_conf = float(orig_res["probs"][true_label_idx])
        o_top1_hit     = int(orig_res["top1_idx"] == true_label_idx)
        o_top5_hit     = int(true_label_idx in orig_res["top5_indices"])

        orig_conf.append(orig_true_conf)
        orig_top1.append(o_top1_hit)
        orig_top5.append(o_top5_hit)

        # ── Adversarial image ──
        adv_path = os.path.join(adv_split_dir, f"adv_{stem}.png")
        adv_res  = None

        if os.path.exists(adv_path):
            adv_pil = Image.open(adv_path)
            adv_res = classify_image(adv_pil, clip_model, text_features, device)

            a_top1_hit = int(adv_res["top1_idx"] == true_label_idx)
            a_top5_hit = int(true_label_idx in adv_res["top5_indices"])

            
            if a_top1_hit:
                a_conf = float(adv_res["probs"][true_label_idx])
            else:
                a_conf = adv_res["top1_confidence"]   # confidence in wrong class

            adv_conf.append(a_conf)
            adv_top1.append(a_top1_hit)
            adv_top5.append(a_top5_hit)
        else:
            if verbose:
                print(f"  ⚠ adversarial image missing: {adv_path}")

        per_image_rows.append({
            "frame":          frame_idx,
            "stem":           stem,
            "orig_true_conf": orig_true_conf,
            "orig_top1":      o_top1_hit,
            "orig_top5":      o_top5_hit,
            "adv_path_found": adv_res is not None,
            "adv_conf":       adv_conf[-1] if adv_res is not None else None,
            "adv_top1":       adv_top1[-1] if adv_res is not None else None,
            "adv_top5":       adv_top5[-1] if adv_res is not None else None,
        })

    summary = {
        "object":          object_name,
        "split":           split,
        "n_images":        len(orig_top1),
        "n_adv_found":     len(adv_top1),
        # --- Original ---
        "orig_confidence": float(np.mean(orig_conf))  if orig_conf  else 0.0,
        "orig_top1_acc":   float(np.mean(orig_top1))  if orig_top1  else 0.0,
        "orig_top5_acc":   float(np.mean(orig_top5))  if orig_top5  else 0.0,
        # --- Adversarial ---
        "adv_confidence":  float(np.mean(adv_conf))   if adv_conf   else 0.0,
        "adv_top1_acc":    float(np.mean(adv_top1))   if adv_top1   else 0.0,
        "adv_top5_acc":    float(np.mean(adv_top5))   if adv_top5   else 0.0,
        # Per-image breakdown for debugging
        "per_image":       per_image_rows,
    }
    return summary



def print_results_table(results: list):
    """Print a Table-2-style summary to stdout."""
    COL = 22
    header = (
        f"{'Object + Split':<{COL}} | "
        f"{'Conf1':>7} {'Top1':>6} {'Top5':>6} | "
        f"{'Conf2':>7} {'Top1':>6} {'Top5':>6}"
    )
    sep = "─" * len(header)

    print()
    print("  CLIP ViT-B/16 classification: Original vs Adversarial")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        label = f"{r['object']} ({r['split']})"
        print(
            f"{label:<{COL}} | "
            f"{r['orig_confidence']:>7.3f} {r['orig_top1_acc']:>6.3f} {r['orig_top5_acc']:>6.3f} | "
            f"{r['adv_confidence']:>7.3f} {r['adv_top1_acc']:>6.3f} {r['adv_top5_acc']:>6.3f}"
        )

    # Average rows per split
    for split in ["train", "test", "val"]:
        sub = [r for r in results if r["split"] == split]
        if not sub:
            continue
        keys = [
            "orig_confidence", "orig_top1_acc", "orig_top5_acc",
            "adv_confidence",  "adv_top1_acc",  "adv_top5_acc",
        ]
        avgs = {k: float(np.mean([r[k] for r in sub])) for k in keys}
        print(sep)
        label = f"Average ({split})"
        print(
            f"{label:<{COL}} | "
            f"{avgs['orig_confidence']:>7.3f} {avgs['orig_top1_acc']:>6.3f} "
            f"{avgs['orig_top5_acc']:>6.3f} | "
            f"{avgs['adv_confidence']:>7.3f} {avgs['adv_top1_acc']:>6.3f} "
            f"{avgs['adv_top5_acc']:>6.3f}"
        )
    print(sep)
    print()



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP accuracy on original vs adversarial nerf_synthetic images"
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Root of nerf_synthetic (contains chair/, drums/, …)")
    parser.add_argument("--adv_root", required=True,
                        help="Root of adversarial output (produced by run_attack.py)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "test", "val"])
    parser.add_argument("--classes", nargs="+", default=NERF_CLASSES)
    parser.add_argument("--output_json", default="results.json")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device : {args.device}")
    clip_model    = load_clip_model(args.device)
    text_features = build_text_features(NERF_CLASSES, clip_model, args.device)

    all_results = []
    for obj in args.classes:
        obj_dir = os.path.join(args.dataset_root, obj)
        if not os.path.isdir(obj_dir):
            print(f"  Skipping {obj}: not found at {obj_dir}")
            continue
        for split in args.splits:
            adv_split_dir = os.path.join(args.adv_root, obj, split)
            print(f"  Evaluating {obj} / {split} …")
            result = evaluate_object(
                object_dir    = obj_dir,
                object_name   = obj,
                adv_split_dir = adv_split_dir,
                clip_model    = clip_model,
                text_features = text_features,
                class_names   = NERF_CLASSES,
                split         = split,
                device        = args.device,
            )
            all_results.append(result)

    # Strip per_image to keep the JSON compact (save separately if you want detail)
    json_out = [{k: v for k, v in r.items() if k != "per_image"} for r in all_results]
    with open(args.output_json, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Results saved → {args.output_json}")

    print_results_table(all_results)


if __name__ == "__main__":
    main()
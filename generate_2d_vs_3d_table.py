import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NERF_CLASSES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def sf(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def get(rows, condition):
    return [r for r in rows if r["condition"] == condition]


def avg(rows, key):
    vals = [sf(r[key]) for r in rows if sf(r[key]) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def obj_val(rows, obj, key):
    for r in rows:
        if r["object"] == obj:
            return sf(r[key])
    return None


def fmt(v, dec=4):
    return f"{v:.{dec}f}" if v is not None else "—"


def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {os.path.basename(path)}")


def make_comparison_table(data_3d, data_2d, eps_tag, out_dir):
    """
    Columns:
        Object | Adv no-def | 2D Gaussian blur | 3D Gaussian smooth
        Each showing PSNR / SSIM / Top-1
    """
    cB  = get(data_3d, f"adv_eps{eps_tag}_no_defense")
    c2d = get(data_2d, f"adv_eps{eps_tag}_2d_gaussian_blur")
    c3d = get(data_3d, f"adv_eps{eps_tag}_defense_gaussian")

    headers = [
        "object",
        "adv_psnr",   "adv_ssim",   "adv_top1",
        "blur2d_psnr","blur2d_ssim","blur2d_top1",
        "gauss3d_psnr","gauss3d_ssim","gauss3d_top1",
    ]

    rows = []
    for obj in NERF_CLASSES:
        rows.append({
            "object":        obj,
            "adv_psnr":      fmt(obj_val(cB,  obj, "psnr"), 2),
            "adv_ssim":      fmt(obj_val(cB,  obj, "ssim")),
            "adv_top1":      fmt(obj_val(cB,  obj, "clip_top1")),
            "blur2d_psnr":   fmt(obj_val(c2d, obj, "psnr"), 2),
            "blur2d_ssim":   fmt(obj_val(c2d, obj, "ssim")),
            "blur2d_top1":   fmt(obj_val(c2d, obj, "clip_top1")),
            "gauss3d_psnr":  fmt(obj_val(c3d, obj, "psnr"), 2),
            "gauss3d_ssim":  fmt(obj_val(c3d, obj, "ssim")),
            "gauss3d_top1":  fmt(obj_val(c3d, obj, "clip_top1")),
        })

    # Average row
    rows.append({
        "object":        "Average",
        "adv_psnr":      fmt(avg(cB,  "psnr"), 2),
        "adv_ssim":      fmt(avg(cB,  "ssim")),
        "adv_top1":      fmt(avg(cB,  "clip_top1")),
        "blur2d_psnr":   fmt(avg(c2d, "psnr"), 2),
        "blur2d_ssim":   fmt(avg(c2d, "ssim")),
        "blur2d_top1":   fmt(avg(c2d, "clip_top1")),
        "gauss3d_psnr":  fmt(avg(c3d, "psnr"), 2),
        "gauss3d_ssim":  fmt(avg(c3d, "ssim")),
        "gauss3d_top1":  fmt(avg(c3d, "clip_top1")),
    })

    write_csv(
        os.path.join(out_dir, f"table_2d_vs_3d_eps{eps_tag}.csv"),
        headers, rows
    )


def make_fig_psnr(data_3d, data_2d, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PSNR: 2D Gaussian blur vs 3D Gaussian smooth", fontsize=12)

    for ax, eps_tag in zip(axes, ["4", "8"]):
        cB  = get(data_3d, f"adv_eps{eps_tag}_no_defense")
        c2d = get(data_2d, f"adv_eps{eps_tag}_2d_gaussian_blur")
        c3d = get(data_3d, f"adv_eps{eps_tag}_defense_gaussian")

        x     = np.arange(len(NERF_CLASSES))
        width = 0.25

        b_vals  = [obj_val(cB,  o, "psnr") or 0 for o in NERF_CLASSES]
        d2_vals = [obj_val(c2d, o, "psnr") or 0 for o in NERF_CLASSES]
        d3_vals = [obj_val(c3d, o, "psnr") or 0 for o in NERF_CLASSES]

        ax.bar(x - width, b_vals,  width, label="Adv (no def.)",       color="#D85A30", alpha=0.85)
        ax.bar(x,         d2_vals, width, label="2D Gaussian blur",     color="#BA7517", alpha=0.85)
        ax.bar(x + width, d3_vals, width, label="3D Gaussian smooth",   color="#1D9E75", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(NERF_CLASSES, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"epsilon = {eps_tag}.0")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_2d_vs_3d_psnr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def make_fig_clip(data_3d, data_2d, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CLIP Top-1: 2D Gaussian blur vs 3D Gaussian smooth", fontsize=12)

    for ax, eps_tag in zip(axes, ["4", "8"]):
        cB  = get(data_3d, f"adv_eps{eps_tag}_no_defense")
        c2d = get(data_2d, f"adv_eps{eps_tag}_2d_gaussian_blur")
        c3d = get(data_3d, f"adv_eps{eps_tag}_defense_gaussian")

        x     = np.arange(len(NERF_CLASSES))
        width = 0.25

        b_vals  = [obj_val(cB,  o, "clip_top1") or 0 for o in NERF_CLASSES]
        d2_vals = [obj_val(c2d, o, "clip_top1") or 0 for o in NERF_CLASSES]
        d3_vals = [obj_val(c3d, o, "clip_top1") or 0 for o in NERF_CLASSES]

        ax.bar(x - width, b_vals,  width, label="Adv (no def.)",       color="#D85A30", alpha=0.85)
        ax.bar(x,         d2_vals, width, label="2D Gaussian blur",     color="#BA7517", alpha=0.85)
        ax.bar(x + width, d3_vals, width, label="3D Gaussian smooth",   color="#1D9E75", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(NERF_CLASSES, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("CLIP Top-1 Accuracy")
        ax.set_title(f"epsilon = {eps_tag}.0")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_2d_vs_3d_clip.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def make_summary(data_3d, data_2d, out_dir):
    lines = ["=" * 62,
             "  2D Gaussian Blur vs 3D Gaussian Smooth -- Summary",
             "=" * 62]

    for eps_tag in ["4", "8"]:
        cB  = get(data_3d, f"adv_eps{eps_tag}_no_defense")
        c2d = get(data_2d, f"adv_eps{eps_tag}_2d_gaussian_blur")
        c3d = get(data_3d, f"adv_eps{eps_tag}_defense_gaussian")

        lines.append(f"\n  Epsilon = {eps_tag}.0")
        lines.append("  " + "─" * 42)

        for label, rows in [
            ("Adv (no defense)",     cB),
            ("2D Gaussian blur",     c2d),
            ("3D Gaussian smooth",   c3d),
        ]:
            p  = avg(rows, "psnr")
            s  = avg(rows, "ssim")
            t1 = avg(rows, "clip_top1")
            t3 = avg(rows, "clip_top3")
            lines.append(
                f"  {label:<24}  "
                f"PSNR={fmt(p,2)}  SSIM={fmt(s)}  "
                f"Top1={fmt(t1)}  Top3={fmt(t3)}"
            )

        # Key comparison
        p_2d = avg(c2d, "psnr"); p_3d = avg(c3d, "psnr")
        t_2d = avg(c2d, "clip_top1"); t_3d = avg(c3d, "clip_top1")
        if p_2d and p_3d:
            diff = round(p_3d - p_2d, 2)
            lines.append(f"\n  3D smooth PSNR advantage over 2D blur : {diff:+.2f} dB")
        if t_2d and t_3d:
            diff = round((t_3d - t_2d) * 100, 1)
            lines.append(f"  3D smooth Top-1 vs 2D blur             : {diff:+.1f}%")

    lines.append("\n" + "=" * 62)
    txt = "\n".join(lines)

    path = os.path.join(out_dir, "summary_2d_vs_3d.txt")
    with open(path, "w") as f:
        f.write(txt)
    print(f"  Saved: summary_2d_vs_3d.txt")
    print(txt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_3d", required=True,
                   help="all_results.csv from compute_all_metrics.py")
    p.add_argument("--results_2d", required=True,
                   help="2d_blur_results.csv from compute_2d_blur_metrics.py")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_3d = load_csv(args.results_3d)
    data_2d = load_csv(args.results_2d)
    print(f"\n  3D results : {len(data_3d)} rows")
    print(f"  2D results : {len(data_2d)} rows\n")

    for eps_tag in ["4", "8"]:
        make_comparison_table(data_3d, data_2d, eps_tag, args.output_dir)

    make_fig_psnr(data_3d, data_2d, args.output_dir)
    make_fig_clip(data_3d, data_2d, args.output_dir)
    make_summary(data_3d, data_2d, args.output_dir)

    print(f"\n  All outputs -> {args.output_dir}/")


if __name__ == "__main__":
    main()
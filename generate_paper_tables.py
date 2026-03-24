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

COND_DISPLAY = {
    "clean_no_defense":           "Clean",
    "adv_eps4_no_defense":        "Adv (no def.)",
    "adv_eps8_no_defense":        "Adv (no def.)",
    "adv_eps4_defense_gaussian":  "Adv + Gaussian",
    "adv_eps4_defense_median":    "Adv + Median",
    "adv_eps4_defense_bilateral": "Adv + Bilateral",
    "adv_eps8_defense_gaussian":  "Adv + Gaussian",
    "adv_eps8_defense_median":    "Adv + Median",
    "adv_eps8_defense_bilateral": "Adv + Bilateral",
}

COLORS = {
    "clean_no_defense":           "#378ADD",
    "adv_no_defense":             "#D85A30",
    "defense_gaussian":           "#1D9E75",
    "defense_median":             "#7F77DD",
    "defense_bilateral":          "#BA7517",
}


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def get(data, condition):
    return [r for r in data if r["condition"] == condition]


def avg(rows, key):
    vals = [safe_float(r[key]) for r in rows if safe_float(r[key]) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def obj_val(rows, obj, key):
    for r in rows:
        if r["object"] == obj:
            return safe_float(r[key])
    return None


def fmt(v, dec=4):
    return f"{v:.{dec}f}" if v is not None else "—"


def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {os.path.basename(path)}")


def make_table1(data, eps_tag, out_dir):
    cA  = get(data, "clean_no_defense")
    cB  = get(data, f"adv_eps{eps_tag}_no_defense")
    cG  = get(data, f"adv_eps{eps_tag}_defense_gaussian")
    cM  = get(data, f"adv_eps{eps_tag}_defense_median")
    cBi = get(data, f"adv_eps{eps_tag}_defense_bilateral")

    headers = [
        "object",
        "clean_conf",   "clean_top1",   "clean_top3",
        "adv_conf",     "adv_top1",     "adv_top3",
        "gauss_conf",   "gauss_top1",   "gauss_top3",
        "median_conf",  "median_top1",  "median_top3",
        "bilat_conf",   "bilat_top1",   "bilat_top3",
    ]

    rows = []
    for obj in NERF_CLASSES:
        rows.append({
            "object":       obj,
            "clean_conf":   fmt(obj_val(cA,  obj, "clip_conf")),
            "clean_top1":   fmt(obj_val(cA,  obj, "clip_top1")),
            "clean_top3":   fmt(obj_val(cA,  obj, "clip_top3")),
            "adv_conf":     fmt(obj_val(cB,  obj, "clip_conf")),
            "adv_top1":     fmt(obj_val(cB,  obj, "clip_top1")),
            "adv_top3":     fmt(obj_val(cB,  obj, "clip_top3")),
            "gauss_conf":   fmt(obj_val(cG,  obj, "clip_conf")),
            "gauss_top1":   fmt(obj_val(cG,  obj, "clip_top1")),
            "gauss_top3":   fmt(obj_val(cG,  obj, "clip_top3")),
            "median_conf":  fmt(obj_val(cM,  obj, "clip_conf")),
            "median_top1":  fmt(obj_val(cM,  obj, "clip_top1")),
            "median_top3":  fmt(obj_val(cM,  obj, "clip_top3")),
            "bilat_conf":   fmt(obj_val(cBi, obj, "clip_conf")),
            "bilat_top1":   fmt(obj_val(cBi, obj, "clip_top1")),
            "bilat_top3":   fmt(obj_val(cBi, obj, "clip_top3")),
        })

    # Average row
    rows.append({
        "object":       "Average",
        "clean_conf":   fmt(avg(cA,  "clip_conf")),
        "clean_top1":   fmt(avg(cA,  "clip_top1")),
        "clean_top3":   fmt(avg(cA,  "clip_top3")),
        "adv_conf":     fmt(avg(cB,  "clip_conf")),
        "adv_top1":     fmt(avg(cB,  "clip_top1")),
        "adv_top3":     fmt(avg(cB,  "clip_top3")),
        "gauss_conf":   fmt(avg(cG,  "clip_conf")),
        "gauss_top1":   fmt(avg(cG,  "clip_top1")),
        "gauss_top3":   fmt(avg(cG,  "clip_top3")),
        "median_conf":  fmt(avg(cM,  "clip_conf")),
        "median_top1":  fmt(avg(cM,  "clip_top1")),
        "median_top3":  fmt(avg(cM,  "clip_top3")),
        "bilat_conf":   fmt(avg(cBi, "clip_conf")),
        "bilat_top1":   fmt(avg(cBi, "clip_top1")),
        "bilat_top3":   fmt(avg(cBi, "clip_top3")),
    })

    write_csv(
        os.path.join(out_dir, f"table1_clip_eps{eps_tag}.csv"),
        headers, rows
    )



def make_table2(data, eps_tag, out_dir):
    cA  = get(data, "clean_no_defense")
    cB  = get(data, f"adv_eps{eps_tag}_no_defense")
    cG  = get(data, f"adv_eps{eps_tag}_defense_gaussian")
    cM  = get(data, f"adv_eps{eps_tag}_defense_median")
    cBi = get(data, f"adv_eps{eps_tag}_defense_bilateral")

    headers = [
        "object",
        "clean_psnr",  "clean_ssim",
        "adv_psnr",    "adv_ssim",
        "gauss_psnr",  "gauss_ssim",
        "median_psnr", "median_ssim",
        "bilat_psnr",  "bilat_ssim",
        "psnr_recovery_gauss",
    ]

    rows = []
    for obj in NERF_CLASSES:
        a_p = obj_val(cA, obj, "psnr")
        b_p = obj_val(cB, obj, "psnr")
        g_p = obj_val(cG, obj, "psnr")

        if a_p and b_p and g_p and (a_p - b_p) > 0:
            rec = round((g_p - b_p) / (a_p - b_p) * 100, 1)
        else:
            rec = None

        rows.append({
            "object":               obj,
            "clean_psnr":           fmt(a_p, 2),
            "clean_ssim":           fmt(obj_val(cA, obj, "ssim")),
            "adv_psnr":             fmt(b_p, 2),
            "adv_ssim":             fmt(obj_val(cB, obj, "ssim")),
            "gauss_psnr":           fmt(g_p, 2),
            "gauss_ssim":           fmt(obj_val(cG, obj, "ssim")),
            "median_psnr":          fmt(obj_val(cM, obj, "psnr"), 2),
            "median_ssim":          fmt(obj_val(cM, obj, "ssim")),
            "bilat_psnr":           fmt(obj_val(cBi, obj, "psnr"), 2),
            "bilat_ssim":           fmt(obj_val(cBi, obj, "ssim")),
            "psnr_recovery_gauss":  f"{rec}%" if rec is not None else "—",
        })

    # Average row
    a_avg = avg(cA, "psnr"); b_avg = avg(cB, "psnr"); g_avg = avg(cG, "psnr")
    avg_rec = None
    if a_avg and b_avg and g_avg and (a_avg - b_avg) > 0:
        avg_rec = round((g_avg - b_avg) / (a_avg - b_avg) * 100, 1)

    rows.append({
        "object":               "Average",
        "clean_psnr":           fmt(a_avg, 2),
        "clean_ssim":           fmt(avg(cA, "ssim")),
        "adv_psnr":             fmt(b_avg, 2),
        "adv_ssim":             fmt(avg(cB, "ssim")),
        "gauss_psnr":           fmt(g_avg, 2),
        "gauss_ssim":           fmt(avg(cG, "ssim")),
        "median_psnr":          fmt(avg(cM, "psnr"), 2),
        "median_ssim":          fmt(avg(cM, "ssim")),
        "bilat_psnr":           fmt(avg(cBi, "psnr"), 2),
        "bilat_ssim":           fmt(avg(cBi, "ssim")),
        "psnr_recovery_gauss":  f"{avg_rec}%" if avg_rec is not None else "—",
    })

    write_csv(
        os.path.join(out_dir, f"table2_psnr_ssim_eps{eps_tag}.csv"),
        headers, rows
    )


def _bar_chart(ax, objects, groups, ylabel, title, ylim=None):
    x     = np.arange(len(objects))
    n     = len(groups)
    width = 0.65 / n
    off   = -(n - 1) / 2 * width

    for i, (label, color, vals) in enumerate(groups):
        safe = [v if v is not None else 0.0 for v in vals]
        ax.bar(x + off + i * width, safe, width,
               label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(objects, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


def make_fig_psnr(data, eps_tag, out_dir):
    cA  = get(data, "clean_no_defense")
    cB  = get(data, f"adv_eps{eps_tag}_no_defense")
    cG  = get(data, f"adv_eps{eps_tag}_defense_gaussian")
    cM  = get(data, f"adv_eps{eps_tag}_defense_median")
    cBi = get(data, f"adv_eps{eps_tag}_defense_bilateral")

    groups = [
        ("Clean",          "#378ADD", [obj_val(cA,  o, "psnr") for o in NERF_CLASSES]),
        ("Adv (no def.)",  "#D85A30", [obj_val(cB,  o, "psnr") for o in NERF_CLASSES]),
        ("Adv + Gaussian", "#1D9E75", [obj_val(cG,  o, "psnr") for o in NERF_CLASSES]),
        ("Adv + Median",   "#7F77DD", [obj_val(cM,  o, "psnr") for o in NERF_CLASSES]),
        ("Adv + Bilateral","#BA7517", [obj_val(cBi, o, "psnr") for o in NERF_CLASSES]),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    _bar_chart(ax, NERF_CLASSES, groups,
               "PSNR (dB)", f"PSNR per object — epsilon={eps_tag}.0")
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_psnr_eps{eps_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def make_fig_clip(data, eps_tag, out_dir):
    cA  = get(data, "clean_no_defense")
    cB  = get(data, f"adv_eps{eps_tag}_no_defense")
    cG  = get(data, f"adv_eps{eps_tag}_defense_gaussian")
    cM  = get(data, f"adv_eps{eps_tag}_defense_median")
    cBi = get(data, f"adv_eps{eps_tag}_defense_bilateral")

    groups = [
        ("Clean",          "#378ADD", [obj_val(cA,  o, "clip_top1") for o in NERF_CLASSES]),
        ("Adv (no def.)",  "#D85A30", [obj_val(cB,  o, "clip_top1") for o in NERF_CLASSES]),
        ("Adv + Gaussian", "#1D9E75", [obj_val(cG,  o, "clip_top1") for o in NERF_CLASSES]),
        ("Adv + Median",   "#7F77DD", [obj_val(cM,  o, "clip_top1") for o in NERF_CLASSES]),
        ("Adv + Bilateral","#BA7517", [obj_val(cBi, o, "clip_top1") for o in NERF_CLASSES]),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    _bar_chart(ax, NERF_CLASSES, groups,
               "CLIP Top-1 Accuracy",
               f"CLIP Top-1 per object — epsilon={eps_tag}.0",
               ylim=(0, 1.1))
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_clip_top1_eps{eps_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


def make_fig_gaussian_count(data, out_dir):
    conds = [
        ("clean_no_defense",           "Clean",           "#378ADD"),
        ("adv_eps4_no_defense",        "Adv eps4",        "#D85A30"),
        ("adv_eps4_defense_gaussian",  "eps4 + Gaussian", "#1D9E75"),
        ("adv_eps4_defense_bilateral", "eps4 + Bilateral","#BA7517"),
        ("adv_eps8_no_defense",        "Adv eps8",        "#993C1D"),
        ("adv_eps8_defense_gaussian",  "eps8 + Gaussian", "#085041"),
        ("adv_eps8_defense_bilateral", "eps8 + Bilateral","#633806"),
    ]

    groups = []
    for cond_key, label, color in conds:
        rows = get(data, cond_key)
        vals = [(obj_val(rows, o, "n_gaussians") or 0) / 1000
                for o in NERF_CLASSES]
        groups.append((label, color, vals))

    fig, ax = plt.subplots(figsize=(16, 5))
    _bar_chart(ax, NERF_CLASSES, groups,
               "Gaussians (K)",
               "Final Gaussian count per object and condition")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_gaussian_count.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")



def make_summary(data, out_dir):
    lines = ["=" * 62, "  Paper Results Summary", "=" * 62]

    for eps_tag in ["4", "8"]:
        cA  = get(data, "clean_no_defense")
        cB  = get(data, f"adv_eps{eps_tag}_no_defense")
        cG  = get(data, f"adv_eps{eps_tag}_defense_gaussian")
        cM  = get(data, f"adv_eps{eps_tag}_defense_median")
        cBi = get(data, f"adv_eps{eps_tag}_defense_bilateral")

        lines.append(f"\n  Epsilon = {eps_tag}.0")
        lines.append("  " + "─" * 42)

        for label, rows in [
            ("Clean (no def.)",    cA),
            (f"Adv eps{eps_tag} (no def.)", cB),
            (f"Adv eps{eps_tag} + Gaussian",  cG),
            (f"Adv eps{eps_tag} + Median",    cM),
            (f"Adv eps{eps_tag} + Bilateral", cBi),
        ]:
            p  = avg(rows, "psnr")
            s  = avg(rows, "ssim")
            t1 = avg(rows, "clip_top1")
            t3 = avg(rows, "clip_top3")
            lines.append(
                f"  {label:<32}  "
                f"PSNR={fmt(p,2)}  SSIM={fmt(s)}  "
                f"Top1={fmt(t1)}  Top3={fmt(t3)}"
            )

        # Recovery stats (vs Gaussian defense)
        a_t1 = avg(cA, "clip_top1"); b_t1 = avg(cB, "clip_top1")
        g_t1 = avg(cG, "clip_top1")
        a_p  = avg(cA, "psnr");      b_p  = avg(cB, "psnr")
        g_p  = avg(cG, "psnr")

        if a_t1 and b_t1 and g_t1 and (a_t1 - b_t1) > 0:
            t1_rec = (g_t1 - b_t1) / (a_t1 - b_t1) * 100
            lines.append(f"\n  CLIP Top-1 recovery (Gaussian) : {t1_rec:.1f}%")
        if a_p and b_p and g_p and (a_p - b_p) > 0:
            p_rec = (g_p - b_p) / (a_p - b_p) * 100
            lines.append(f"  PSNR recovery (Gaussian)       : {p_rec:.1f}%")

    lines.append("\n" + "=" * 62)
    txt = "\n".join(lines)

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(txt)
    print(f"  Saved: summary.txt")
    print(txt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", required=True)
    p.add_argument("--output_dir",  required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_csv(args.results_csv)
    print(f"\n  Loaded {len(data)} rows from {args.results_csv}\n")

    for eps_tag in ["4", "8"]:
        print(f"  Generating tables for eps={eps_tag}.0 ...")
        make_table1(data, eps_tag, args.output_dir)
        make_table2(data, eps_tag, args.output_dir)
        make_fig_psnr(data, eps_tag, args.output_dir)
        make_fig_clip(data, eps_tag, args.output_dir)

    make_fig_gaussian_count(data, args.output_dir)
    make_summary(data, args.output_dir)

    print(f"\n  All outputs -> {args.output_dir}/")


if __name__ == "__main__":
    main()
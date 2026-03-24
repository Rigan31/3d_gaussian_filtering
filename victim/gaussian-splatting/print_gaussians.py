import torch
import time


def print_gaussian_info(gaussians, k_neighbors=16, num_points_to_print=10):

    C0 = 0.28209479177387814  # SH normalization constant

    xyz = gaussians.get_xyz.detach()                    # [N, 3]
    sh_dc = gaussians._features_dc.detach()             # [N, 1, 3]
    opacity = gaussians.get_opacity.detach()             # [N, 1]
    N = xyz.shape[0]

    # Convert SH DC to RGB
    rgb = sh_dc.squeeze(1) * C0 + 0.5                   # [N, 3]
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # ---- Print point info ----
    print("=" * 80)
    print(f"Total Gaussians: {N}")
    print("=" * 80)
    print(f"{'Index':<8} {'X':>10} {'Y':>10} {'Z':>10} "
          f"{'R':>8} {'G':>8} {'B':>8} {'Opacity':>10}")
    print("-" * 80)

    for i in range(min(num_points_to_print, N)):
        x, y, z = xyz[i].cpu().tolist()
        r, g, b = rgb[i].cpu().tolist()
        o = opacity[i].item()
        print(f"{i:<8} {x:>10.4f} {y:>10.4f} {z:>10.4f} "
              f"{r:>8.4f} {g:>8.4f} {b:>8.4f} {o:>10.4f}")

    if N > num_points_to_print:
        print(f"... ({N - num_points_to_print} more points)")

    # ---- Time KNN ----
    print("\n" + "=" * 80)
    print(f"Timing KNN (K={k_neighbors}) on {N} Gaussians...")
    print("=" * 80)

    k = min(k_neighbors, N - 1)
    batch_size = 4096

    torch.cuda.synchronize()
    start = time.time()

    total_batches = 0
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        dists = torch.cdist(xyz[s:e], xyz)           # [B, N]
        _, nn_idx = dists.topk(k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:, 1:]                        # drop self
        total_batches += 1

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"KNN completed in {elapsed:.4f} seconds")
    print(f"  Num Gaussians : {N}")
    print(f"  K neighbors   : {k}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Num batches   : {total_batches}")
    print(f"  Time per batch: {elapsed / total_batches:.4f} seconds")
    print("=" * 80)
"""
approach2_intraining_smooth.py

Approach 2 -- In-training 3D smoothing defense.

Plugs into the 3DGS training loop (benchmark.py) and applies spatial
smoothing to Gaussian attributes after every densification step.

Smoothable attributes (select via constructor argument 'targets'):
    rgb      -- DC colour (f_dc), the primary colour corrupted by adversarial noise
    opacity  -- raw opacity values, attenuates transparent noise Gaussians
    xyz      -- positions (optional)

Filters (select via constructor argument 'method'):
    gaussian   -- Gaussian-weighted mean of k nearest neighbours
    median     -- Geometric median of k nearest neighbours
    bilateral  -- Edge-preserving: spatial distance + colour similarity

How to integrate into benchmark.py
------------------------------------
Step 1 -- import at the top of benchmark.py:

    from approach2_intraining_smooth import GaussianSmoothingDefense

Step 2 -- create the defense object before the training loop:

    smooth_defense = GaussianSmoothingDefense(
        method  = "gaussian",         # "gaussian" | "median" | "bilateral"
        targets = ["rgb", "opacity"], # what to smooth
        k       = 10,
        sigma   = 0.05,
        sigma_c = 0.1,                # bilateral only
    )

Step 3 -- call it right after densify_and_prune inside the training loop:

    if iteration in densification_iterations:
        gaussians.densify_and_prune(...)
        smooth_defense.step(gaussians, iteration)   # <-- add this one line

Design notes
------------
- Smoothing is done inside torch.no_grad() with in-place .copy_() so the
  optimizer's momentum and variance buffers for each parameter are preserved.
  Replacing a tensor with a new one would lose those buffers and destabilize
  training -- copy_() avoids that entirely.

- The KD-tree is built from XYZ positions each time (to find spatial
  neighbours), even if xyz is not in targets. Positions define the
  neighbourhood; what we update is controlled by targets.

- Bilateral weighting always uses current RGB colour similarity to compute
  edge-preserving weights, regardless of whether rgb is in targets.

Requirements
------------
    pip install numpy scipy tqdm
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometric median (Weiszfeld algorithm)
# ---------------------------------------------------------------------------

def _geometric_median(points, n_iter=10):
    m = points.mean(axis=0)
    for _ in range(n_iter):
        d = np.maximum(np.linalg.norm(points - m, axis=1, keepdims=True), 1e-8)
        w = 1.0 / d
        m = (w * points).sum(0) / w.sum()
    return m


# ---------------------------------------------------------------------------
# Main defense class
# ---------------------------------------------------------------------------

class GaussianSmoothingDefense:
    """
    In-training smoothing defense for 3DGS.

    Parameters
    ----------
    method   : "gaussian" | "median" | "bilateral"
    targets  : list of attributes to smooth.
               Options: "rgb", "opacity", "xyz"
               Default: ["rgb", "opacity"]
               Note: xyz positions are always used to build the KD-tree
               (spatial neighbourhood), but only updated if "xyz" is in targets.
    k        : number of nearest neighbours
    sigma    : spatial bandwidth for gaussian / bilateral kernels
    sigma_c  : colour bandwidth for bilateral kernel
    verbose  : print per-call statistics
    """

    def __init__(
        self,
        method:  str   = "gaussian",
        targets: list  = None,
        k:       int   = 10,
        sigma:   float = 0.05,
        sigma_c: float = 0.1,
        verbose: bool  = False,
    ):
        assert method in ("gaussian", "median", "bilateral"), \
            f"Unknown method '{method}'. Choose: gaussian | median | bilateral"

        self.method  = method
        self.targets = targets if targets is not None else ["rgb", "opacity"]
        self.k       = k
        self.sigma   = sigma
        self.sigma_c = sigma_c
        self.verbose = verbose
        self._n_calls = 0

        # Validate targets
        valid = {"rgb", "opacity", "xyz"}
        bad   = set(self.targets) - valid
        assert not bad, f"Invalid targets: {bad}. Valid: {valid}"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def step(self, gaussians, iteration: int):
        """
        Apply smoothing to the Gaussian model right now.
        Call this once after each densify_and_prune call.

        Parameters
        ----------
        gaussians : GaussianModel instance from the 3DGS codebase
        iteration : current training iteration (for logging only)
        """
        self._n_calls += 1
        n = gaussians.get_xyz.shape[0]

        if self.verbose:
            print(f"\n  [SmoothDefense | iter={iteration} | call=#{self._n_calls}]"
                  f"  method={self.method}  targets={self.targets}"
                  f"  N={n:,}")

        # Extract current values from Gaussian model
        xyz     = self._get_xyz(gaussians)      # (N, 3) numpy float32
        rgb     = self._get_rgb(gaussians)      # (N, 3) numpy float32 in [0,1]
        opacity = self._get_opacity(gaussians)  # (N,)   numpy float32 in [0,1]

        # Build KD-tree from positions (used by all methods)
        tree       = cKDTree(xyz)
        dists, idx = tree.query(xyz, k=self.k + 1, workers=-1)
        dists      = dists[:, 1:].astype(np.float32)  # (N, k), self excluded
        idx        = idx[:, 1:]                        # (N, k)

        # Run selected filter
        if self.method == "gaussian":
            results = self._gaussian(xyz, rgb, opacity, dists, idx)
        elif self.method == "median":
            results = self._median(xyz, rgb, opacity, idx)
        elif self.method == "bilateral":
            results = self._bilateral(xyz, rgb, opacity, dists, idx)

        # Write results back into Gaussian model tensors
        device = gaussians._xyz.device

        if "xyz" in results:
            self._report("XYZ",     xyz,     results["xyz"])
            new_t = torch.from_numpy(results["xyz"]).to(device)
            with torch.no_grad():
                gaussians._xyz.copy_(new_t)

        if "rgb" in results:
            self._report("RGB",     rgb,     results["rgb"])
            # Convert smoothed RGB back to raw f_dc
            C0      = 0.28209479177387814
            new_fdc = ((results["rgb"] - 0.5) / C0).astype(np.float32)
            new_t   = torch.from_numpy(new_fdc).unsqueeze(1).to(device)
            with torch.no_grad():
                gaussians._features_dc.copy_(new_t)

        if "opacity" in results:
            self._report("Opacity", opacity, results["opacity"])
            # Convert back: raw = logit(opacity)
            p     = np.clip(results["opacity"], 1e-6, 1 - 1e-6)
            raw   = np.log(p / (1 - p)).astype(np.float32)
            new_t = torch.from_numpy(raw).unsqueeze(1).to(device)
            with torch.no_grad():
                gaussians._opacity.copy_(new_t)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _gaussian(self, xyz, rgb, opacity, dists, idx):
        """Gaussian-weighted mean: weight = exp(-d^2 / 2*sigma^2)"""
        weights = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))  # (N, k)
        w_sum   = weights.sum(axis=1, keepdims=True) + 1e-8
        results = {}

        if "xyz" in self.targets:
            new_xyz = np.empty_like(xyz)
            for d in range(3):
                new_xyz[:, d] = (weights * xyz[idx, d]).sum(1) / w_sum[:, 0]
            results["xyz"] = new_xyz

        if "rgb" in self.targets:
            new_rgb = np.empty_like(rgb)
            for c in range(3):
                new_rgb[:, c] = (weights * rgb[idx, c]).sum(1) / w_sum[:, 0]
            results["rgb"] = new_rgb

        if "opacity" in self.targets:
            results["opacity"] = (weights * opacity[idx]).sum(1) / w_sum[:, 0]

        return results

    def _median(self, xyz, rgb, opacity, idx):
        """Geometric median of k neighbours (robust to outliers)."""
        results = {}

        if "xyz" in self.targets:
            new_xyz = np.empty_like(xyz)
            for i in tqdm(range(len(xyz)), desc="    median xyz",
                          leave=False, mininterval=2):
                new_xyz[i] = _geometric_median(xyz[idx[i]])
            results["xyz"] = new_xyz

        if "rgb" in self.targets:
            new_rgb = np.empty_like(rgb)
            for i in tqdm(range(len(rgb)), desc="    median rgb",
                          leave=False, mininterval=2):
                new_rgb[i] = _geometric_median(rgb[idx[i]])
            results["rgb"] = new_rgb

        if "opacity" in self.targets:
            results["opacity"] = np.array(
                [np.median(opacity[idx[i]]) for i in range(len(opacity))]
            )

        return results

    def _bilateral(self, xyz, rgb, opacity, dists, idx):
        """
        Edge-preserving smooth.
        weight = exp(-d_space^2/2*ss^2) * exp(-d_colour^2/2*sc^2)
        Preserves colour/opacity boundaries, smooths within flat regions.
        """
        w_space = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))  # (N, k)

        new_xyz     = xyz.copy()
        new_rgb     = rgb.copy()
        new_opacity = opacity.copy()

        for i in range(len(xyz)):
            # Colour-space distance always used for bilateral weights
            colour_diff = rgb[idx[i]] - rgb[i]                     # (k, 3)
            colour_d2   = (colour_diff ** 2).sum(axis=1)           # (k,)
            w_colour    = np.exp(-colour_d2 / (2 * self.sigma_c ** 2))

            w     = w_space[i] * w_colour                          # (k,)
            w_sum = w.sum() + 1e-8

            if "xyz"     in self.targets:
                new_xyz[i]     = (w[:, None] * xyz[idx[i]]).sum(0) / w_sum
            if "rgb"     in self.targets:
                new_rgb[i]     = (w[:, None] * rgb[idx[i]]).sum(0) / w_sum
            if "opacity" in self.targets:
                new_opacity[i] = (w * opacity[idx[i]]).sum() / w_sum

        results = {}
        if "xyz"     in self.targets: results["xyz"]     = new_xyz
        if "rgb"     in self.targets: results["rgb"]     = new_rgb
        if "opacity" in self.targets: results["opacity"] = new_opacity
        return results

    # ------------------------------------------------------------------
    # Tensor extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_xyz(gaussians):
        return gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _get_rgb(gaussians):
        C0  = 0.28209479177387814
        fdc = gaussians._features_dc.detach().cpu().numpy()  # (N, 1, 3)
        rgb = np.clip(C0 * fdc[:, 0, :] + 0.5, 0.0, 1.0)
        return rgb.astype(np.float32)

    @staticmethod
    def _get_opacity(gaussians):
        # _opacity is raw (pre-sigmoid), shape (N, 1)
        raw = gaussians._opacity.detach().cpu().numpy()[:, 0]
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _report(self, label, old, new):
        if self.verbose:
            diff = np.abs(new - old)
            print(f"    {label:<8} change  mean={diff.mean():.5f}"
                  f"  max={diff.max():.5f}")
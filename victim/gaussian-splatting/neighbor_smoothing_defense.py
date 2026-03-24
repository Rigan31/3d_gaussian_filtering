import torch
from torch import Tensor
from typing import Optional
import numpy as np


class NeighborMedianSmoothing:

    def __init__(
        self,
        k_neighbors: int = 16,        # how many nearest neighbors
        blend_factor: float = 0.3,     # 0.0 = full median, 1.0 = keep original
        smooth_sh_dc: bool = True,     # smooth base color (DC component of SH)
        smooth_sh_rest: bool = False,  # smooth higher-order SH
        smooth_scale: bool = False,    # smooth the scale (log-space internally)
        smooth_opacity: bool = False,  # smooth opacity
        batch_size: int = 4096,        # process this many Gaussians at a time to save VRAM

        # --- Sorted neighbor options ---
        use_sorted: bool = False,      # if True, use sorted-window instead of KNN search
        window_mode: str = 'centered', # 'centered' = +-k/2 neighbors, 'forward' = next k neighbors

        # --- 2D sliding window options ---
        use_2d_window: bool = False,   # if True, use 2D XY sliding window approach
        window_size_x: float = 0.5,    # half-width of the window in world-space X units
        window_size_y: float = 0.5,    # half-height of the window in world-space Y units
    ):
        self.k_neighbors = k_neighbors
        self.blend_factor = blend_factor
        self.smooth_sh_dc = smooth_sh_dc
        self.smooth_sh_rest = smooth_sh_rest
        self.smooth_scale = smooth_scale
        self.smooth_opacity = smooth_opacity
        self.batch_size = batch_size

        self.use_sorted = use_sorted
        self.window_mode = window_mode

        self.use_2d_window = use_2d_window
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y

    @torch.no_grad()
    def apply(self, gaussians) -> dict:
        """
            Expected attributes:
                gaussians.get_xyz          -> [N, 3]  positions
                gaussians._features_dc     -> [N, 1, 3]  DC SH coefficients
                gaussians._features_rest   -> [N, SH-1, 3]  rest of SH
                gaussians._scaling         -> [N, 3]  log-space scales
                gaussians._opacity         -> [N, 1]  pre-sigmoid opacity
        """

        xyz = gaussians.get_xyz.detach()  # [N, 3]
        N = xyz.shape[0]
        k = min(self.k_neighbors, N - 1)

        if N < k + 1:
            print(f"[NeighborSmoothing] Only {N} Gaussians, skipping (need > {k+1})")
            return {"num_gaussians": N, "smoothed_attrs": []}

        smoothed_attrs = []

        # --- Pre-compute sort indices once (used by sorted and 2d window modes) ---
        if self.use_sorted or self.use_2d_window:
            sort_indices   = self._get_sort_indices(xyz)       # [N]
            unsort_indices = torch.argsort(sort_indices)       # [N]
            xyz_sorted     = xyz[sort_indices]                 # [N, 3] sorted xyz
        else:
            sort_indices = unsort_indices = xyz_sorted = None

        # ------ SH DC (base color) ------
        if self.smooth_sh_dc:
            self._smooth_tensor(xyz, gaussians._features_dc, k, N, sort_indices, unsort_indices, xyz_sorted)
            smoothed_attrs.append("sh_dc")

        # ------ SH rest (higher harmonics) ------
        if self.smooth_sh_rest:
            self._smooth_tensor(xyz, gaussians._features_rest, k, N, sort_indices, unsort_indices, xyz_sorted)
            smoothed_attrs.append("sh_rest")

        # ------ Scale (in log space) ------
        if self.smooth_scale:
            self._smooth_tensor(xyz, gaussians._scaling, k, N, sort_indices, unsort_indices, xyz_sorted)
            smoothed_attrs.append("scaling")

        # ------ Opacity (pre-sigmoid) ------
        if self.smooth_opacity:
            self._smooth_tensor(xyz, gaussians._opacity, k, N, sort_indices, unsort_indices, xyz_sorted)
            smoothed_attrs.append("opacity")

        if self.use_2d_window:
            mode_str = f"2d_window(wx={self.window_size_x},wy={self.window_size_y})"
        elif self.use_sorted:
            mode_str = f"sorted(x->y->z,{self.window_mode})"
        else:
            mode_str = "knn"
        print(f"[NeighborSmoothing] Smoothed {N} Gaussians | "
              f"K={k} | blend={self.blend_factor} | mode={mode_str} | attrs={smoothed_attrs}")

        return {"num_gaussians": N, "smoothed_attrs": smoothed_attrs}

    def _get_sort_indices(self, xyz: Tensor) -> Tensor:
        """
        Sort Gaussians by x first, then y, then z (multi-key sort on float values).
        Uses torch.lexsort which sorts by the LAST key first, so we pass (z, y, x).
        """
        x = xyz[:, 0].cpu()
        y = xyz[:, 1].cpu()
        z = xyz[:, 2].cpu()

        # lexsort sorts by last key first: x is primary, y secondary, z tertiary
        sort_indices = torch.from_numpy(
            __import__('numpy').lexsort((z.numpy(), y.numpy(), x.numpy()))
        ).to(xyz.device)

        return sort_indices

    @torch.no_grad()
    def _smooth_tensor_sorted(
        self,
        param: Tensor,
        sort_indices: Tensor,
        unsort_indices: Tensor,
        k: int,
        N: int,
    ):
        original      = param.data.clone()
        original_flat = original.reshape(N, -1)   # [N, D]

        # Reorder data along sorted axis
        sorted_flat    = original_flat[sort_indices]   # [N, D]
        smoothed_sorted = torch.empty_like(sorted_flat)

        half_k = k // 2

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_idx = torch.arange(start, end, device=param.device)  # [B]

            if self.window_mode == 'centered':
                win_start = (batch_idx - half_k).clamp(0, N - 1)
            else:  # 'forward'
                win_start = batch_idx.clamp(0, N - 1)

            # Build [B, k] index window
            offsets = torch.arange(k, device=param.device)             # [k]
            indices = (win_start.unsqueeze(1) + offsets.unsqueeze(0))  # [B, k]
            indices = indices.clamp(0, N - 1)

            nn_vals     = sorted_flat[indices]               # [B, k, D]
            median_vals = nn_vals.median(dim=1).values       # [B, D]

            blended = (self.blend_factor * sorted_flat[start:end]
                       + (1.0 - self.blend_factor) * median_vals)

            smoothed_sorted[start:end] = blended

        # Restore original Gaussian order
        smoothed_original_order = smoothed_sorted[unsort_indices]
        param.data[:] = smoothed_original_order.reshape(param.data.shape)



    @torch.no_grad()
    def _smooth_tensor_2d_window(
        self,
        param: Tensor,
        sort_indices: Tensor,
        unsort_indices: Tensor,
        xyz_sorted: Tensor,      
        N: int,
    ):
        

        original      = param.data.clone()
        original_flat = original.reshape(N, -1)


        sorted_flat     = original_flat[sort_indices]
        smoothed_sorted = torch.empty_like(sorted_flat)


        x_sorted = xyz_sorted[:, 0].cpu().numpy()
        y_sorted = xyz_sorted[:, 1].cpu().numpy()      

        hw_x = self.window_size_x   # half-width  in X
        hw_y = self.window_size_y   # half-height in Y

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)

            # For each Gaussian in the batch, find all points inside its XY window
            # using binary search on the x-sorted array, then filter by y.
            batch_medians = []

            for i in range(start, end):
                px = x_sorted[i]
                py = y_sorted[i]

                # binary search on X to get a candidate range
                # All points with x in [px - hw_x, px + hw_x]
                lo = int(np.searchsorted(x_sorted, px - hw_x, side='left'))
                hi = int(np.searchsorted(x_sorted, px + hw_x, side='right'))
                # candidate indices in sorted space: [lo, hi)

                # filter candidates by Y range 
                candidate_y = y_sorted[lo:hi]                        
                y_mask = (candidate_y >= py - hw_y) & (candidate_y <= py + hw_y)
                # convert mask to global sorted indices
                neighbor_indices = np.arange(lo, hi)[y_mask]        # numpy array

                if len(neighbor_indices) == 0:
                    # no neighbors found — keep original value unchanged
                    batch_medians.append(sorted_flat[i])
                    continue

                # gather attribute values of neighbors
                neighbor_idx_t = torch.from_numpy(neighbor_indices).long().to(param.device)
                neighbor_vals  = sorted_flat[neighbor_idx_t]


                median_val = neighbor_vals.median(dim=0).values    
                batch_medians.append(median_val)


            batch_medians_t = torch.stack(batch_medians, dim=0)


            blended = (self.blend_factor * sorted_flat[start:end]
                       + (1.0 - self.blend_factor) * batch_medians_t)

            smoothed_sorted[start:end] = blended

        smoothed_original_order = smoothed_sorted[unsort_indices]
        param.data[:] = smoothed_original_order.reshape(param.data.shape)




_smoother = None

def apply_neighbor_smoothing(
    gaussians,
    iteration: int,

    k_neighbors: int = 16,
    blend_factor: float = 0.3,
    smooth_sh_dc: bool = True,
    smooth_sh_rest: bool = False,
    smooth_scale: bool = True,
    smooth_opacity: bool = False,
    every_n_steps: int = 1,
    batch_size: int = 4096,


    use_sorted: bool = False,      
    window_mode: str = 'centered', 


    use_2d_window: bool = False,
    window_size_x: float = 0.5,  
    window_size_y: float = 0.5,
):
    global _smoother

    if _smoother is None:
        _smoother = NeighborMedianSmoothing(
            k_neighbors=k_neighbors,
            blend_factor=blend_factor,
            smooth_sh_dc=smooth_sh_dc,
            smooth_sh_rest=smooth_sh_rest,
            smooth_scale=smooth_scale,
            smooth_opacity=smooth_opacity,
            batch_size=batch_size,
            use_sorted=use_sorted,
            window_mode=window_mode,
            use_2d_window=use_2d_window,
            window_size_x=window_size_x,
            window_size_y=window_size_y,
        )

    if every_n_steps > 1:
        densification_interval = 100
        step_number = iteration // densification_interval
        if step_number % every_n_steps != 0:
            return

    _smoother.apply(gaussians)
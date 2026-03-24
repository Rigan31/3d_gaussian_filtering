
import torch
from torch import Tensor
from typing import Optional


class NeighborMedianSmoothing:


    def __init__(
        self,
        k_neighbors: int = 16,       # how many nearest neighbors
        blend_factor: float = 0.3,    # 0.0 = full median, 1.0 = keep original
        smooth_sh_dc: bool = True,    # smooth base color (DC component of SH)
        smooth_sh_rest: bool = False, # smooth higher-order SH
        smooth_scale: bool = False,    # smooth the scale (log-space internally)
        smooth_opacity: bool = False, # smooth opacity 
        batch_size: int = 4096,       # process this many Gaussians at a time to save VRAM
    ):
        self.k_neighbors = k_neighbors
        self.blend_factor = blend_factor
        self.smooth_sh_dc = smooth_sh_dc
        self.smooth_sh_rest = smooth_sh_rest
        self.smooth_scale = smooth_scale
        self.smooth_opacity = smooth_opacity
        self.batch_size = batch_size

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
        
        xyz = gaussians.get_xyz.detach()         # [N, 3]
        N = xyz.shape[0]
        k = min(self.k_neighbors, N - 1)

        if N < k + 1:
            print(f"[NeighborSmoothing] Only {N} Gaussians, skipping (need > {k+1})")
            return {"num_gaussians": N, "smoothed_attrs": []}

        smoothed_attrs = []

        # ------ SH DC (base color) ------
        if self.smooth_sh_dc:
            self._smooth_tensor(xyz, gaussians._features_dc, k, N)
            smoothed_attrs.append("sh_dc")

        # ------ SH rest (higher harmonics) ------
        if self.smooth_sh_rest:
            self._smooth_tensor(xyz, gaussians._features_rest, k, N)
            smoothed_attrs.append("sh_rest")

        # ------ Scale (in log space) ------
        if self.smooth_scale:
            self._smooth_tensor(xyz, gaussians._scaling, k, N)
            smoothed_attrs.append("scaling")

        # ------ Opacity (pre-sigmoid) ------
        if self.smooth_opacity:
            self._smooth_tensor(xyz, gaussians._opacity, k, N)
            smoothed_attrs.append("opacity")

        print(f"[NeighborSmoothing] Smoothed {N} Gaussians | "
              f"K={k} | blend={self.blend_factor} | attrs={smoothed_attrs}")

        return {"num_gaussians": N, "smoothed_attrs": smoothed_attrs}
    
    @torch.no_grad()
    def _smooth_tensor_bakcup(self, xyz: Tensor, param: Tensor, k: int, N: int):

        original = param.data.clone()
        original_shape = original.shape

        original_flat = original.reshape(N, -1)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_xyz = xyz[start:end]  # [B, 3]
            print("x, y, z", xyz[0])
            print("---- batch xyz ---- ")
            print(batch_xyz)


            dists = torch.cdist(batch_xyz, xyz)  # [B, N]
            


            print("Distances sample:", dists[0, :10])  # print distances to

            kernel = torch.zeros((3, 3, 3))
            current_point = batch_xyz[0]

            median_vals = param.data[start]

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        neighbor_point = current_point + xyz[i-1, j-1, k-1]
                        print("neighbor point", neighbor_point)
                    break             
            


            _, nn_indices = dists.topk(k + 1, dim=-1, largest=False)  # [B, k+1]
            nn_indices = nn_indices[:, 1:]  

            nn_vals = original_flat[nn_indices]  # [B, k, D]

            print("---- nn_vals ---- ")
            print(nn_vals)

            median_vals = nn_vals.median(dim=1).values  # [B, D]

            print("---- median_vals ---- ")
            print(median_vals)
            blended = (self.blend_factor * original_flat[start:end]
                       + (1.0 - self.blend_factor) * median_vals)

            print("start, end", start, end)
            print("batch size", self.batch_size)
            print("N", N)
            print("Blended", blended)
            print("param : ", param.data[start:end])
            param.data[start:end] = blended.reshape(param.data[start:end].shape)

            print("after change param : ", param.data[start:end])
            
            break

        
    @torch.no_grad()
    def _smooth_tensor(self, xyz: Tensor, param: Tensor, k: int, N: int):

        original = param.data.clone()
        original_shape = original.shape

        original_flat = original.reshape(N, -1)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_xyz = xyz[start:end]  # [B, 3]



            dists = torch.cdist(batch_xyz, xyz)  # [B, N]

 
            
            _, nn_indices = dists.topk(k + 1, dim=-1, largest=False)  # [B, k+1]
            nn_indices = nn_indices[:, 1:]  

            nn_vals = original_flat[nn_indices]  # [B, k, D]

            median_vals = nn_vals.median(dim=1).values  # [B, D]

            blended = (self.blend_factor * original_flat[start:end]
                       + (1.0 - self.blend_factor) * median_vals)

            param.data[start:end] = blended.reshape(param.data[start:end].shape)


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
        )


    if every_n_steps > 1:
        densification_interval = 100 
        step_number = iteration // densification_interval
        if step_number % every_n_steps != 0:
            return

    _smoother.apply(gaussians)
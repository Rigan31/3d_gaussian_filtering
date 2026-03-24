"""
Pruning Defense Against Poison-Splat Computation Cost Attacks.

This module provides a budget-aware pruning mechanism that can be integrated
into the vanilla 3D Gaussian Splatting training loop. It defends against
Poison-Splat attacks by:

1. Enforcing a hard Gaussian count budget (prevents OOM)
2. Periodically pruning low-importance Gaussians using accumulated gradient scores
3. Detecting abnormal growth rates that signal an attack

Usage:
    Place this file at: utils/pruning_defense.py
    Then import and use in train.py (see guide for integration details).

Reference:
    - PUP 3D-GS: Hanson et al., "Principled Uncertainty Pruning for 3D Gaussian Splatting" (CVPR 2025)
    - Poison-Splat: Lu et al., "Computation Cost Attack on 3D Gaussian Splatting" (ICLR 2025)
"""

import torch
import numpy as np
from typing import Optional


class PruningDefense:
    """
    Budget-aware pruning defense for 3D Gaussian Splatting training.

    Integrates into the training loop to prevent Poison-Splat attacks from
    causing unbounded Gaussian growth and GPU memory exhaustion.

    The importance score is inspired by PUP 3D-GS's Fisher-based sensitivity
    score, but uses a lightweight gradient accumulation proxy that doesn't
    require CUDA kernels or a converged model.

    Args:
        max_gaussians: Hard upper bound on Gaussian count. Exceeding this
                       triggers immediate pruning.
        soft_budget: Target count after budget-triggered pruning. Should be
                     less than max_gaussians to provide headroom.
        pruning_interval: Check for periodic pruning every N iterations.
        pruning_start_iter: Don't prune before this iteration (warmup).
        growth_rate_threshold: If Gaussian count grows by more than this
                               factor between checks, flag as attack.
        aggressive_prune_ratio: Fraction to prune when attack is detected.
        normal_prune_ratio: Fraction to prune during normal periodic checks.
        score_type: Importance scoring method. Options:
                    'gradient' - accumulated gradient magnitudes (recommended)
                    'opacity_volume' - opacity * volume (simple fallback)
                    'hybrid' - combination of both
    """

    def __init__(
        self,
        max_gaussians: int = 500_000,
        soft_budget: int = 300_000,
        pruning_interval: int = 500,
        pruning_start_iter: int = 1000,
        growth_rate_threshold: float = 2.0,
        aggressive_prune_ratio: float = 0.5,
        normal_prune_ratio: float = 0.1,
        score_type: str = "gradient",
        verbose: bool = True,
    ):
        self.max_gaussians = max_gaussians
        self.soft_budget = min(soft_budget, max_gaussians)
        self.pruning_interval = pruning_interval
        self.pruning_start_iter = pruning_start_iter
        self.growth_rate_threshold = growth_rate_threshold
        self.aggressive_prune_ratio = aggressive_prune_ratio
        self.normal_prune_ratio = normal_prune_ratio
        self.score_type = score_type
        self.verbose = verbose

        # Internal tracking state
        self._prev_count: Optional[int] = None
        self._accumulated_grad_xyz: Optional[torch.Tensor] = None
        self._accumulated_grad_scale: Optional[torch.Tensor] = None
        self._accumulation_steps: int = 0
        self._count_history: list = []
        self._prune_history: list = []
        self._attack_detected_count: int = 0

    def log(self, msg: str):
        """Print defense log messages."""
        if self.verbose:
            print(f"[DEFENSE] {msg}")

    # ------------------------------------------------------------------
    # Gradient Accumulation
    # ------------------------------------------------------------------

    def accumulate_gradients(self, gaussians) -> None:
        """
        Accumulate gradient magnitudes for importance scoring.

        Call this every iteration AFTER loss.backward() and BEFORE
        optimizer.step() (so gradients are still available).

        The accumulated |grad(xyz)| and |grad(scaling)| approximate the
        diagonal of the Fisher Information matrix used by PUP 3D-GS.
        This is a cheap proxy: PUP 3D-GS computes the full 6x6 block
        Fisher and takes its log-determinant, but during training we
        just need a relative ranking.

        Args:
            gaussians: The GaussianModel instance.
        """
        xyz_grad = gaussians._xyz.grad
        if xyz_grad is None:
            return

        grad_xyz = xyz_grad.detach().abs()

        scale_grad = gaussians._scaling.grad
        grad_scale = scale_grad.detach().abs() if scale_grad is not None \
                     else torch.zeros_like(grad_xyz)

        n_current = grad_xyz.shape[0]

        # Reset accumulator if Gaussian count changed (densification happened)
        if (self._accumulated_grad_xyz is None or
                self._accumulated_grad_xyz.shape[0] != n_current):
            self._accumulated_grad_xyz = grad_xyz.clone()
            self._accumulated_grad_scale = grad_scale.clone()
            self._accumulation_steps = 1
        else:
            self._accumulated_grad_xyz += grad_xyz
            self._accumulated_grad_scale += grad_scale
            self._accumulation_steps += 1

    # ------------------------------------------------------------------
    # Importance Scoring
    # ------------------------------------------------------------------

    def _score_gradient(self, gaussians) -> torch.Tensor:
        """
        Gradient-based importance score (approximates PUP 3D-GS sensitivity).

        Score_i = opacity_i * (||avg_grad_xyz_i|| + ||avg_grad_scale_i||)

        Higher score = more important = keep.
        """
        opacity = gaussians.get_opacity.detach().squeeze(-1)  # [N]

        if self._accumulated_grad_xyz is not None and self._accumulation_steps > 0:
            avg_grad_xyz = self._accumulated_grad_xyz / self._accumulation_steps
            avg_grad_scale = self._accumulated_grad_scale / self._accumulation_steps
            sensitivity = torch.norm(avg_grad_xyz, dim=-1) + \
                          torch.norm(avg_grad_scale, dim=-1)
            return opacity * sensitivity
        else:
            # Fallback if no gradients accumulated yet
            return self._score_opacity_volume(gaussians)

    def _score_opacity_volume(self, gaussians) -> torch.Tensor:
        """
        Simple opacity * volume score (similar to LightGaussian heuristic).

        This is the fallback when gradient info is unavailable.
        """
        opacity = gaussians.get_opacity.detach().squeeze(-1)
        scales = gaussians.get_scaling.detach()  # [N, 3]
        volume = torch.prod(scales, dim=-1)
        return opacity * volume

    def _score_hybrid(self, gaussians) -> torch.Tensor:
        """
        Hybrid score combining gradient sensitivity and opacity/volume.

        Useful as a robust scoring method that works even with sparse
        gradient accumulation.
        """
        grad_score = self._score_gradient(gaussians)
        vol_score = self._score_opacity_volume(gaussians)

        # Normalize both to [0, 1] range, then combine
        grad_norm = grad_score / (grad_score.max() + 1e-8)
        vol_norm = vol_score / (vol_score.max() + 1e-8)

        return 0.7 * grad_norm + 0.3 * vol_norm

    def compute_importance_score(self, gaussians) -> torch.Tensor:
        """
        Compute per-Gaussian importance score using the configured method.

        Args:
            gaussians: The GaussianModel instance.

        Returns:
            Tensor of shape [N] with importance scores (higher = more important).
        """
        if self.score_type == "gradient":
            return self._score_gradient(gaussians)
        elif self.score_type == "opacity_volume":
            return self._score_opacity_volume(gaussians)
        elif self.score_type == "hybrid":
            return self._score_hybrid(gaussians)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    # ------------------------------------------------------------------
    # Attack Detection
    # ------------------------------------------------------------------

    def detect_attack(self, current_count: int) -> bool:
        """
        Detect abnormal Gaussian growth indicating a Poison-Splat attack.

        Monitors the ratio of current count to previous count. Clean scenes
        have predictable, gradual growth during densification. Poison-Splat
        causes explosive growth.

        Args:
            current_count: Current number of Gaussians.

        Returns:
            True if attack is suspected.
        """
        self._count_history.append(current_count)

        if self._prev_count is None or self._prev_count == 0:
            self._prev_count = current_count
            return False

        growth_rate = current_count / self._prev_count
        self._prev_count = current_count

        if growth_rate > self.growth_rate_threshold:
            self._attack_detected_count += 1
            self.log(
                f"ATTACK WARNING #{self._attack_detected_count}: "
                f"Gaussian growth rate = {growth_rate:.2f}x "
                f"(threshold: {self.growth_rate_threshold}x). "
                f"Count: {current_count}"
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Pruning Logic
    # ------------------------------------------------------------------

    def should_prune(self, iteration: int, current_count: int) -> bool:
        """Check whether pruning should be triggered at this iteration."""
        if iteration < self.pruning_start_iter:
            return False

        # Always prune if hard budget is exceeded
        if current_count > self.max_gaussians:
            return True

        # Periodic pruning check
        if iteration % self.pruning_interval == 0:
            return True

        return False

    def prune(self, gaussians, iteration: int) -> int:
        """
        Perform importance-based pruning on the Gaussian model.

        This is the main entry point. Call it every iteration after
        densification. It will only actually prune when needed (based on
        budget, schedule, or attack detection).

        Args:
            gaussians: The GaussianModel instance.
            iteration: Current training iteration.

        Returns:
            Number of Gaussians removed (0 if no pruning occurred).
        """
        current_count = gaussians.get_xyz.shape[0]

        if not self.should_prune(iteration, current_count):
            return 0

        # Detect attack based on growth rate
        attack_detected = self.detect_attack(current_count)

        # Determine pruning ratio
        if current_count > self.max_gaussians:
            # Hard budget exceeded: prune to soft budget
            n_to_remove = current_count - self.soft_budget
            prune_ratio = min(n_to_remove / current_count, 0.8)  # Cap at 80%
            reason = "BUDGET_EXCEEDED"
        elif attack_detected:
            prune_ratio = self.aggressive_prune_ratio
            n_to_remove = int(current_count * prune_ratio)
            reason = "ATTACK_DETECTED"
        else:
            prune_ratio = self.normal_prune_ratio
            n_to_remove = int(current_count * prune_ratio)
            reason = "PERIODIC"

        if n_to_remove <= 0 or prune_ratio <= 0:
            return 0

        # Compute importance scores
        scores = self.compute_importance_score(gaussians)

        # Determine threshold for pruning (remove lowest-scoring Gaussians)
        prune_ratio_clamped = min(max(prune_ratio, 0.0), 0.9)  # Safety clamp
        threshold = torch.quantile(scores, prune_ratio_clamped)
        prune_mask = scores <= threshold

        # Apply pruning using 3D-GS's built-in mechanism
        # (this handles optimizer state cleanup, etc.)
        gaussians.prune_points(prune_mask)

        # Reset gradient accumulators since indices changed
        self._reset_accumulators()

        # Record history
        removed = prune_mask.sum().item()
        remaining = gaussians.get_xyz.shape[0]
        self._prune_history.append({
            "iteration": iteration,
            "reason": reason,
            "removed": removed,
            "remaining": remaining,
            "prune_ratio": prune_ratio_clamped,
        })

        self.log(
            f"Iter {iteration} [{reason}]: Pruned {removed} Gaussians "
            f"({prune_ratio_clamped:.1%}). Remaining: {remaining}"
        )

        return removed

    def _reset_accumulators(self):
        """Reset gradient accumulators after pruning changes indices."""
        self._accumulated_grad_xyz = None
        self._accumulated_grad_scale = None
        self._accumulation_steps = 0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a summary of defense activity."""
        return {
            "total_prune_events": len(self._prune_history),
            "total_attack_warnings": self._attack_detected_count,
            "current_count": self._count_history[-1] if self._count_history else 0,
            "max_observed_count": max(self._count_history) if self._count_history else 0,
            "history": self._prune_history,
        }

    def print_summary(self):
        """Print a summary of defense activity."""
        summary = self.get_summary()
        self.log("=" * 60)
        self.log("DEFENSE SUMMARY")
        self.log(f"  Total prune events:    {summary['total_prune_events']}")
        self.log(f"  Attack warnings:       {summary['total_attack_warnings']}")
        self.log(f"  Max observed count:    {summary['max_observed_count']}")
        self.log(f"  Final count:           {summary['current_count']}")
        self.log("=" * 60)


# ======================================================================
# Integration helper: drop-in function for train.py
# ======================================================================

def create_defense(args) -> Optional[PruningDefense]:
    """
    Factory function to create a PruningDefense from command-line args.

    Usage in train.py:
        from utils.pruning_defense import create_defense
        defense = create_defense(opt)
    """
    if not getattr(args, "defense_enabled", True):
        return None

    max_g = getattr(args, "max_gaussians", 500_000)

    return PruningDefense(
        max_gaussians=max_g,
        soft_budget=int(max_g * 0.6),
        pruning_interval=500,
        pruning_start_iter=1000,
        growth_rate_threshold=2.0,
        aggressive_prune_ratio=0.5,
        normal_prune_ratio=0.1,
        score_type="gradient",
        verbose=True,
    )
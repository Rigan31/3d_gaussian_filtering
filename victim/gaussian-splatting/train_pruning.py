#!/usr/bin/env python3
"""
Modified 3D-GS victim training script with Smart Pruning Defense.

This file replaces:
    victim/gaussian-splatting/defense/smart_pruning/train.py

It is based on the standard 3D-GS train.py (as used in victim/gaussian-splatting/train.py)
with the SmartPruningDefense integrated into the training loop.

Changes from the original train.py are marked with:
    # === DEFENSE: ... ===

Usage:
    python victim/gaussian-splatting/defense/smart_pruning/train.py \
        -s <SCENE_DATA_PATH> -m <OUTPUT_MODEL_PATH> \
        --max_gaussians 500000 \
        --defense_prune_interval 500 \
        --defense_score_type gradient
"""

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import os
import sys
import torch
import uuid
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr

from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from defense.pup_pruning.pruning_defense import SmartPruningDefense
# === DEFENSE: Import the smart pruning defense ===
# Adjust the import path based on where you place the module.
# Option A: If pruning_defense.py is in the same directory as this train.py:
# try:
#     from pruning_defense import SmartPruningDefense
# except ImportError:
#     # Option B: If running from the repo root:
#     try:
#         from victim.gaussian_splatting.defense.smart_pruning.pruning_defense import SmartPruningDefense
#     except ImportError:
#         # Option C: Direct path append
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         if script_dir not in sys.path:
#             sys.path.insert(0, script_dir)
#         from pruning_defense import SmartPruningDefense
# # === END DEFENSE IMPORT ===


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples every N iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                           for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0, 1.0)
                    gt = viewpoint.original_image[0:3, :, :]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt[None], global_step=iteration)
                    l1_test += l1_loss(image, gt).mean().double()
                    psnr_test += psnr(image, gt).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss',
                                         l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr',
                                         psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram",
                                     scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points',
                                  scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from,
             # === DEFENSE: Additional arguments ===
             defense_args=None):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # ===================================================================
    # === DEFENSE: Initialize the Smart Pruning Defense ===
    # ===================================================================
    defense = None
    if defense_args is not None and defense_args.get("enabled", True):
        defense = SmartPruningDefense(
            max_gaussians=defense_args.get("max_gaussians", 500_000),
            soft_budget=defense_args.get("soft_budget",
                                         int(defense_args.get("max_gaussians", 500_000) * 0.6)),
            pruning_interval=defense_args.get("prune_interval", 500),
            pruning_start_iter=defense_args.get("prune_start_iter", 1000),
            growth_rate_threshold=defense_args.get("growth_rate_threshold", 2.0),
            aggressive_prune_ratio=defense_args.get("aggressive_prune_ratio", 0.5),
            normal_prune_ratio=defense_args.get("normal_prune_ratio", 0.1),
            score_type=defense_args.get("score_type", "gradient"),
            verbose=True,
        )
        defense.log(f"Smart Pruning Defense ENABLED")
        defense.log(f"  max_gaussians={defense.max_gaussians}")
        defense.log(f"  soft_budget={defense.soft_budget}")
        defense.log(f"  score_type={defense.score_type}")
        defense.log(f"  prune_interval={defense.pruning_interval}")
        defense.log(f"  normal_prune_ratio={defense.normal_prune_ratio}")
        defense.log(f"  aggressive_prune_ratio={defense.aggressive_prune_ratio}")
    # ===================================================================

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
               opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "#pts": gaussians.get_xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ======================================================
            # === DEFENSE: Accumulate gradients BEFORE densification
            # === (gradients from loss.backward() match current count)
            # ======================================================
            if defense is not None:
                defense.accumulate_gradients(gaussians)
            # ======================================================

            # --------------------------------------------------------
            # Densification (standard 3D-GS logic, UNCHANGED)
            # --------------------------------------------------------
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor,
                                                   visibility_filter)

                if (iteration > opt.densify_from_iter and
                        iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0 or
                        (dataset.white_background and
                         iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # ======================================================
            # === DEFENSE: Smart pruning (runs AFTER densification)
            # ======================================================
            if defense is not None:
                # Prune if needed (scoring handles size mismatches
                # gracefully by falling back to opacity*volume)
                defense.prune(gaussians, iteration)

                # Log Gaussian count to tensorboard
                if tb_writer and iteration % 100 == 0:
                    tb_writer.add_scalar(
                        'defense/gaussian_count',
                        gaussians.get_xyz.shape[0], iteration)
            # ======================================================

            # Checkpoint
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    # === DEFENSE: Print summary at end of training ===
    if defense is not None:
        defense.print_summary()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script with Smart Pruning Defense")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # === DEFENSE: Command-line arguments ===
    parser.add_argument("--defense_enabled", action="store_true", default=True,
                        help="Enable smart pruning defense (default: True)")
    parser.add_argument("--no_defense", action="store_true", default=False,
                        help="Disable defense entirely")
    parser.add_argument("--max_gaussians", type=int, default=500_000,
                        help="Hard upper bound on Gaussian count")
    parser.add_argument("--defense_prune_interval", type=int, default=500,
                        help="Check for pruning every N iterations")
    parser.add_argument("--defense_prune_start", type=int, default=1000,
                        help="Start defense pruning after this iteration")
    parser.add_argument("--defense_normal_ratio", type=float, default=0.1,
                        help="Fraction to prune in periodic maintenance")
    parser.add_argument("--defense_aggressive_ratio", type=float, default=0.5,
                        help="Fraction to prune when attack detected")
    parser.add_argument("--defense_growth_threshold", type=float, default=2.0,
                        help="Growth rate threshold to flag attack")
    parser.add_argument("--defense_score_type", type=str, default="gradient",
                        choices=["gradient", "opacity_volume", "hybrid"],
                        help="Importance scoring method")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # === DEFENSE: Build defense config dict ===
    defense_args = None
    if not args.no_defense:
        defense_args = {
            "enabled": True,
            "max_gaussians": args.max_gaussians,
            "soft_budget": int(args.max_gaussians * 0.6),
            "prune_interval": args.defense_prune_interval,
            "prune_start_iter": args.defense_prune_start,
            "growth_rate_threshold": args.defense_growth_threshold,
            "aggressive_prune_ratio": args.defense_aggressive_ratio,
            "normal_prune_ratio": args.defense_normal_ratio,
            "score_type": args.defense_score_type,
        }

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from,
        defense_args=defense_args,
    )

    print("\nTraining complete.")



#
# benchmark_filter_defense.py
#
# Same as benchmark.py but with GaussianSmoothingDefense applied
# after every densification step. Smoothing method is passed via
# --smooth_method argument so the shell script can control it.
#
# Usage:
#   python benchmark_filter_defense.py \
#       -s [data] -m [output] --gpu 0 \
#       --smooth_method gaussian \
#       --smooth_k 10 --smooth_sigma 0.05

import torch
import numpy as np
import os
import sys
import random
from random import randint
import time
import re
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import multiprocessing
from gpuinfo import GPUInfo
from datetime import datetime
import matplotlib.pyplot as plt

from utils.approach2_intraining_smooth import GaussianSmoothingDefense


def gpu_monitor_worker(stop_event, log_file_handle, gpuid=0):
    while not stop_event.is_set():
        now = datetime.now()
        formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
        percent, memory = GPUInfo.gpu_usage()
        if isinstance(percent, list):
            percent = [percent[gpuid]]
            memory  = [memory[gpuid]]
        log_file_handle.write(
            f'[{formatted_date}] GPU:{gpuid} uses {percent}% and {memory} MB\n'
        )
        log_file_handle.flush()
        time.sleep(0.2)
    print(f'GPU {gpuid} monitor stops')


def plot_record(file_name, record_name, xlabel='Iteration'):
    if not os.path.exists(file_name):
        return
    record = np.load(file_name)
    plt.figure()
    plt.plot(record, label=record_name)
    plt.xlabel(xlabel)
    plt.ylabel(record_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name.replace('.npy', '.png'))
    plt.close()


def fix_all_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def victim_training(dataset, opt, pipe, testing_iterations, saving_iterations,
                    checkpoint_iterations, checkpoint, debug_from, exp_run):

    os.makedirs(f'{args.model_path}/exp_run_{exp_run}/', exist_ok=True)

    # ── Monitoring ────────────────────────────────────────────────────────
    record_gaussian_num = []
    record_iter_elapse  = []
    record_l1   = []
    record_ssim = []
    record_psnr = []

    gpu_monitor_stop_event = multiprocessing.Event()
    gpu_log_file_handle = open(
        f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'w'
    )
    gpu_monitor_process = multiprocessing.Process(
        target=gpu_monitor_worker,
        args=(gpu_monitor_stop_event, gpu_log_file_handle, args.gpu)
    )
    fix_all_random_seed()

    first_iter = 0
    gaussians  = GaussianModel(dataset.sh_degree)
    scene      = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack  = None
    ema_loss_for_log = 0.0
    first_iter      += 1
    gpu_monitor_process.start()

    # ── Defense: constructed from CLI args ────────────────────────────────
    smooth_defense = GaussianSmoothingDefense(
        method  = args.smooth_method,
        targets = args.smooth_targets,
        k       = args.smooth_k,
        sigma   = args.smooth_sigma,
        sigma_c = args.smooth_sigma_c,
        verbose = True,
    )
    print(f"\n  [Defense] method={args.smooth_method}  "
          f"targets={args.smooth_targets}  "
          f"k={args.smooth_k}  sigma={args.smooth_sigma}\n")
    # ──────────────────────────────────────────────────────────────────────

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image                  = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter      = render_pkg["visibility_filter"]
        radii                  = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1      = l1_loss(image, gt_image)
        Lssim    = ssim(image, gt_image)
        loss     = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 1000 == 0:
                print(f'[GPU:{args.gpu}] Run {exp_run} iter {iteration}  '
                      f'loss={ema_loss_for_log:.3f}  '
                      f'N={gaussians._xyz.shape[0]:,}')

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold
                    )
                    # ── Apply smoothing defense after densification ────
                    smooth_defense.step(gaussians, iteration)
                    # ──────────────────────────────────────────────────

                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background
                            and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            try:
                record_iter_elapse.append(iter_start.elapsed_time(iter_end))
            except Exception:
                pass
            record_gaussian_num.append(gaussians._xyz.shape[0])
            record_psnr.append(psnr(image, gt_image).mean().item())
            record_l1.append(Ll1.item())
            record_ssim.append(Lssim.item())

    # ── Write results ─────────────────────────────────────────────────────
    gpu_monitor_stop_event.set()
    gpu_monitor_process.join()
    gpu_log_file_handle.flush()
    gpu_log_file_handle.close()

    SSIM_views, PSNR_views = [], []
    for cam in scene.getTrainCameras().copy():
        gt_img   = cam.original_image.cuda()
        rendered = render(cam, gaussians, pipe, bg)['render']
        SSIM_views.append(ssim(gt_img, rendered).item())
        PSNR_views.append(psnr(gt_img, rendered).mean().item())
    mean_SSIM = round(sum(SSIM_views) / len(SSIM_views), 4)
    mean_PSNR = round(sum(PSNR_views) / len(PSNR_views), 4)

    gaussians.save_ply(f'{args.model_path}/exp_run_{exp_run}/victim_model.ply')

    base = f'{args.model_path}/exp_run_{exp_run}'
    np.save(f'{base}/gaussian_num_record.npy', np.array(record_gaussian_num))
    plot_record(f'{base}/gaussian_num_record.npy', 'Number of Gaussians')
    np.save(f'{base}/iter_elapse_record.npy', np.array(record_iter_elapse))
    plot_record(f'{base}/iter_elapse_record.npy', 'Iteration Elapse Time [ms]', 'Time')
    np.save(f'{base}/psnr_record.npy', np.array(record_psnr))
    plot_record(f'{base}/psnr_record.npy', 'PSNR')
    np.save(f'{base}/l1_record.npy', np.array(record_l1))
    plot_record(f'{base}/l1_record.npy', 'L1 Loss')
    np.save(f'{base}/ssim_record.npy', np.array(record_ssim))
    plot_record(f'{base}/ssim_record.npy', 'SSIM')

    gpu_log   = open(f'{base}/gpu.log', 'r')
    timestamps, gpu_mem_cost = [], []
    for line in gpu_log:
        matches = re.findall(r'\[(.*?)\]', line)
        timestamps.append(matches[0])
        gpu_mem_cost.append(int(matches[2]))
    plt.figure()
    plt.plot(gpu_mem_cost, label='GPU memory cost [MB]')
    plt.xlabel('Training time'); plt.ylabel('GPU memory cost [MB]')
    plt.legend(); plt.tight_layout()
    plt.savefig(f'{base}/gpu_mem_cost.png'); plt.close()

    t_start       = datetime.strptime(timestamps[0],  "%Y-%m-%d %H:%M:%S")
    t_end         = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
    training_time = (t_end - t_start).seconds / 60

    result_str  = f"Defense method  : {args.smooth_method}\n"
    result_str += f"Smooth targets  : {args.smooth_targets}\n"
    result_str += f"k / sigma       : {args.smooth_k} / {args.smooth_sigma}\n"
    result_str += f"Max Gaussians   : {max(record_gaussian_num)/1e6:.3f} M\n"
    result_str += f"Max GPU mem     : {max(gpu_mem_cost)} MB\n"
    result_str += f"Training time   : {training_time:.3f} min\n"
    result_str += f"SSIM            : {mean_SSIM:.4f}\n"
    result_str += f"PSNR            : {mean_PSNR:.4f}\n"
    print(result_str)
    with open(f'{base}/benchmark_result.log', 'w') as f:
        f.write(result_str)


def conclude_victim_multiple_runs(args):
    max_g, max_m, times = [], [], []
    for exp_run in range(1, args.exp_runs + 1):
        gauss = np.load(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy')
        gpu_log = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log')
        ts, mem = [], []
        for line in gpu_log:
            m = re.findall(r'\[(.*?)\]', line)
            ts.append(m[0]); mem.append(int(m[2]))
        t0 = datetime.strptime(ts[0],  "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        times.append((t1 - t0).seconds / 60)
        max_g.append(max(gauss) / 1e6)
        max_m.append(max(mem))

    result_str  = (f"Max Gaussians : {np.mean(max_g):.3f} M "
                   f"+- {np.std(max_g):.3f} M\n")
    result_str += (f"Max GPU mem   : {int(np.mean(max_m))} MB "
                   f"+- {int(np.std(max_m))} MB\n")
    result_str += (f"Training time : {np.mean(times):.2f} min "
                   f"+- {np.std(times):.2f} min\n")
    print(result_str)
    with open(f'{args.model_path}/benchmark_result.log', 'w') as f:
        f.write(result_str)


if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS -- Filter Defense")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip',            type=str,  default="127.0.0.1")
    parser.add_argument('--port',          type=int,  default=6009)
    parser.add_argument('--debug_from',    type=int,  default=-1)
    parser.add_argument('--detect_anomaly',action='store_true', default=False)
    parser.add_argument("--test_iterations",  nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--save_iterations",  nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--quiet",         action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu",           type=int,  default=0)
    parser.add_argument("--exp_runs",      type=int,  default=1)

    # ── Defense arguments ─────────────────────────────────────────────────
    parser.add_argument("--smooth_method",  type=str,  default="gaussian",
                        choices=["gaussian", "median", "bilateral"],
                        help="Smoothing method to apply after densification")
    parser.add_argument("--smooth_targets", nargs="+", default=["rgb", "opacity"],
                        choices=["rgb", "opacity", "xyz"],
                        help="Which Gaussian attributes to smooth")
    parser.add_argument("--smooth_k",       type=int,   default=10,
                        help="Number of nearest neighbours")
    parser.add_argument("--smooth_sigma",   type=float, default=0.05,
                        help="Spatial bandwidth (gaussian / bilateral)")
    parser.add_argument("--smooth_sigma_c", type=float, default=0.1,
                        help="Colour bandwidth (bilateral only)")
    # ──────────────────────────────────────────────────────────────────────

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    for exp_run in range(1, args.exp_runs + 1):
        victim_training(
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint,
            args.debug_from, exp_run
        )

    print("\nTraining complete.")
    conclude_victim_multiple_runs(args)
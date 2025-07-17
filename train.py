#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, HairParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def compute_hair_losses(gaussians, hair_params):
    """
    Compute all hair-specific regularization losses.
    
    Args:
        gaussians: GaussianModel instance
        hair_params: HairParams instance with loss weights
        
    Returns:
        dict: Dictionary containing individual loss components
    """
    losses = {
        'opacity_loss': 0.0,
        'geometry_loss': 0.0, 
        'direction_loss': 0.0,
        'total_hair_loss': 0.0
    }
    
    # Check if we have hair data
    if not (hasattr(gaussians, '_group_id') and hasattr(gaussians, '_strand_id')):
        return losses
        
    hair_mask = gaussians._group_id != -1
    if not torch.any(hair_mask):
        return losses
    
    # Opacity continuity loss
    if hair_params.lambda_opacity > 0:
        losses['opacity_loss'] = compute_opacity_continuity_loss(gaussians, hair_mask)
    
    # Geometry continuity loss  
    if hair_params.lambda_geometry > 0:
        losses['geometry_loss'] = compute_geometry_continuity_loss(gaussians, hair_mask)
        
    # Direction alignment loss
    if hair_params.lambda_direction > 0:
        losses['direction_loss'] = compute_direction_alignment_loss(gaussians, hair_mask)
    
    # Compute total weighted loss
    losses['total_hair_loss'] = (
        hair_params.lambda_opacity * losses['opacity_loss'] +
        hair_params.lambda_geometry * losses['geometry_loss'] + 
        hair_params.lambda_direction * losses['direction_loss']
    )
    
    return losses


def compute_opacity_continuity_loss(gaussians, hair_mask):
    """Compute opacity continuity loss for hair strands"""
    group_ids = gaussians._group_id[hair_mask].contiguous()
    opacities = gaussians.get_opacity[hair_mask].contiguous()
    strand_ids = gaussians._strand_id[hair_mask].contiguous()

    if len(opacities) < 3:
        return torch.tensor(0.0, device=opacities.device)

    # Stable sorting
    sort_key = (group_ids << 20) + strand_ids
    sorted_indices = torch.argsort(sort_key)
    
    sorted_opacities = opacities[sorted_indices]
    sorted_group_ids = group_ids[sorted_indices]
    sorted_strand_ids = strand_ids[sorted_indices]

    # Compute differences
    prev_diff = torch.abs(sorted_opacities[1:] - sorted_opacities[:-1])
    avg_diff = (prev_diff[:-1] + prev_diff[1:]) / 2

    # Continuity check
    same_group = (sorted_group_ids[:-1] == sorted_group_ids[1:])
    consecutive = (sorted_strand_ids[1:] - sorted_strand_ids[:-1]) == 1
    valid_mask = same_group[:-1] & same_group[1:] & consecutive[:-1] & consecutive[1:]

    return torch.mean(avg_diff[valid_mask]) if torch.any(valid_mask) else torch.tensor(0.0, device=opacities.device)


def compute_geometry_continuity_loss(gaussians, hair_mask):
    """Compute geometry continuity loss for hair strands"""
    if not torch.any(hair_mask):
        return torch.tensor(0.0, device="cuda")
    
    # Get sorted data
    group_ids = gaussians._group_id[hair_mask].contiguous()
    scales = gaussians.get_scaling[hair_mask].contiguous()
    strand_ids = gaussians._strand_id[hair_mask].contiguous()
    
    sort_key = (group_ids << 20) + strand_ids
    sorted_indices = torch.argsort(sort_key)
    
    sorted_scales = scales[sorted_indices].contiguous()
    sorted_group_ids = group_ids[sorted_indices].contiguous()
    sorted_strand_ids = strand_ids[sorted_indices].contiguous()
    
    # Scale continuity loss
    continuity_loss = compute_scale_continuity_loss(sorted_scales, sorted_group_ids, sorted_strand_ids)
    
    # Scale size penalty
    size_penalty = compute_scale_size_penalty(sorted_scales)
    
    total_loss = continuity_loss + 0.5 * size_penalty
    return torch.clamp(total_loss, min=0.0, max=1.0)


def compute_scale_continuity_loss(sorted_scales, sorted_group_ids, sorted_strand_ids):
    """Compute scale continuity loss within strands"""
    if len(sorted_scales) < 3:
        return torch.tensor(0.0, device="cuda")
    
    scale_diff = torch.norm(sorted_scales[1:] - sorted_scales[:-1], dim=1)
    scale_diff = scale_diff / (torch.mean(torch.norm(sorted_scales, dim=1)) + 1e-6)
    scale_avg_diff = (scale_diff[:-1] + scale_diff[1:]) / 2
    
    same_group = (sorted_group_ids[:-1] == sorted_group_ids[1:])
    consecutive = (sorted_strand_ids[1:] - sorted_strand_ids[:-1]) == 1
    valid_pairs = same_group & consecutive
    valid_mask = valid_pairs[:-1] & valid_pairs[1:]

    if torch.any(valid_mask):
        return torch.mean(scale_avg_diff[valid_mask])
    return torch.tensor(0.0, device="cuda")


def compute_scale_size_penalty(sorted_scales):
    """Compute penalty for oversized hair Gaussians"""
    hair_volumes = torch.prod(sorted_scales, dim=1)
    volume_median = torch.median(hair_volumes)
    volume_threshold = volume_median * 4.0
    
    oversized_mask = hair_volumes > volume_threshold
    if torch.any(oversized_mask):
        excess_volumes = hair_volumes[oversized_mask] - volume_threshold
        return torch.mean(excess_volumes / volume_threshold)
    return torch.tensor(0.0, device="cuda")


def compute_direction_alignment_loss(gaussians, hair_mask):
    """Compute direction alignment loss between Gaussians and hair tangents"""
    if not torch.any(hair_mask):
        return torch.tensor(0.0, device="cuda")
    
    # Safe data extraction
    group_ids = gaussians._group_id[hair_mask].contiguous()
    xyz = gaussians._xyz[hair_mask].contiguous()
    rotations = gaussians.get_rotation[hair_mask].contiguous()
    strand_ids = gaussians._strand_id[hair_mask].contiguous()
    
    if len(xyz) < 2:
        return torch.tensor(0.0, device="cuda")
    
    # Stable sorting
    sort_key = (group_ids << 20) + strand_ids
    sorted_indices = torch.argsort(sort_key)
    
    sorted_xyz = xyz[sorted_indices].contiguous()
    sorted_rotations = rotations[sorted_indices].contiguous()
    sorted_group_ids = group_ids[sorted_indices].contiguous()
    sorted_strand_ids = strand_ids[sorted_indices].contiguous()
    
    # Vectorized tangent computation
    same_strand = (sorted_group_ids[:-1] == sorted_group_ids[1:]) & \
                ((sorted_strand_ids[1:] - sorted_strand_ids[:-1]) == 1)
    
    all_forward_diff = sorted_xyz[1:] - sorted_xyz[:-1]
    tangent_dirs = torch.zeros_like(sorted_xyz)
    
    # Forward difference
    forward_valid = torch.cat([same_strand, torch.tensor([False], device="cuda")])
    tangent_dirs[forward_valid] = all_forward_diff[same_strand]
    
    # Backward difference
    backward_valid = torch.cat([torch.tensor([False], device="cuda"), same_strand])
    backward_only = backward_valid & ~forward_valid
    if torch.any(backward_only):
        backward_indices = torch.where(backward_only)[0]
        tangent_dirs[backward_only] = all_forward_diff[backward_indices - 1].clone()
    
    # Center difference
    both_valid = forward_valid & backward_valid
    if torch.any(both_valid[1:-1]):
        center_indices = torch.where(both_valid[1:-1])[0] + 1
        center_diff = (sorted_xyz[center_indices + 1] - sorted_xyz[center_indices - 1]) / 2.0
        tangent_dirs[center_indices] = center_diff
    
    # Normalize tangents
    tangent_norms = torch.norm(tangent_dirs, dim=1, keepdim=True)
    valid_tangents = tangent_norms.squeeze() > 1e-6
    
    if not torch.any(valid_tangents):
        return torch.tensor(0.0, device="cuda")
    
    tangent_dirs = tangent_dirs[valid_tangents]
    tangent_norms = tangent_norms[valid_tangents]
    normalized_tangents = tangent_dirs / tangent_norms
    
    # Extract Gaussian Z-axis
    from utils.general_utils import build_rotation
    rotation_matrices = build_rotation(sorted_rotations[valid_tangents])
    gaussian_z_axes = rotation_matrices[:, :, 2].contiguous()
    
    # Compute alignment scores
    alignment_scores = torch.sum(gaussian_z_axes * normalized_tangents, dim=1)
    alignment_scores = torch.clamp(alignment_scores, -1.0, 1.0)
    
    return torch.mean(1.0 - torch.abs(alignment_scores))


def print_training_stats(iteration, loss, Ll1, ssim_value, Ll1depth, gaussians, hair_losses):
    """Print comprehensive training statistics"""
    if iteration % 100 != 0:
        return
        
    print(f"[ITER {iteration}] Total Loss: {loss.item():.4f} | " 
          f"L1: {Ll1.item():.4f} | "
          f"SSIM: {(1 - ssim_value).item():.4f} | "
          f"Depth: {Ll1depth:.4f}")
    
    if not hasattr(gaussians, '_group_id'):
        print("No hair gaussians found")
        return
        
    hair_mask = gaussians._group_id != -1
    hair_count = hair_mask.sum().item() if hair_mask.any() else 0
    
    if hair_count > 0:
        hair_scales = gaussians.get_scaling[hair_mask].detach()
        hair_volumes = torch.prod(hair_scales, dim=1)
        max_scales = torch.max(hair_scales, dim=1)[0]
        
        print(f"Hair Losses - Opacity: {hair_losses['opacity_loss']:.4f} | "
              f"Geometry: {hair_losses['geometry_loss']:.4f} | "
              f"Direction: {hair_losses['direction_loss']:.4f} | "
              f"Hair Count: {hair_count}")
        
        print(f"Hair Stats - Volume: med={torch.median(hair_volumes).item():.2e}, "
              f"max={torch.max(hair_volumes).item():.2e} | "
              f"Scale: med={torch.median(max_scales).item():.4f}, "
              f"max={torch.max(max_scales).item():.4f}")


def apply_hair_scale_clamping(gaussians, iteration, hair_params):
    """Apply real-time scale clamping for oversized hair Gaussians"""
    if not (hasattr(gaussians, '_group_id') and iteration > 1000):
        return
    
    with torch.no_grad():
        hair_mask = gaussians._group_id != -1
        if not torch.any(hair_mask):
            return
        
        hair_scales = gaussians.get_scaling[hair_mask]
        hair_volumes = torch.prod(hair_scales, dim=1)
        
        volume_median = torch.median(hair_volumes)
        volume_threshold = volume_median * 5.0
        
        oversized_mask = hair_volumes > volume_threshold
        if not torch.any(oversized_mask):
            return
        
        # Create global mask and apply clamping
        global_oversized_mask = torch.zeros_like(gaussians._group_id, dtype=torch.bool).squeeze()
        global_oversized_mask[hair_mask] = oversized_mask
        
        oversized_real_scales = gaussians.get_scaling[global_oversized_mask]
        oversized_volumes = torch.prod(oversized_real_scales, dim=1, keepdim=True)
        scale_factor = torch.pow(volume_threshold / oversized_volumes, 1/3).clamp(max=1.0)
        
        new_real_scales = oversized_real_scales * scale_factor
        gaussians._scaling[global_oversized_mask] = gaussians.scaling_inverse_activation(new_real_scales)
        
        if iteration % 500 == 0:
            print(f"[ITER {iteration}] Clamped {oversized_mask.sum().item()} oversized hair gaussians")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, hair_params):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, hair_params=hair_params)
    scene.save(0)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Set hair position fixing if requested
    if hair_params.fix_hair_positions:
        gaussians.set_fix_hair_positions(True)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer or 1.0, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss computation
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Hair-specific losses
        hair_losses = compute_hair_losses(gaussians, hair_params)
        loss += hair_losses['total_hair_loss']

        # Print training statistics
        print_training_stats(iteration, loss, Ll1, ssim_value, Ll1depth, gaussians, hair_losses)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "GS_nums": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification with hair-aware strategies
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    
                    # Hair-aware densification
                    if hasattr(gaussians, '_group_id') and hair_params.enable_strand_clone:
                        if iteration < opt.densify_until_iter:
                            strand_stats = gaussians.smart_strand_clone(grads, opt.densify_grad_threshold * 0.25, scene.cameras_extent, iteration, hair_params.strand_clone_threshold)
                            if iteration % 100 == 0 and strand_stats['strands_cloned'] > 0:
                                print(f"[ITER {iteration}] Strand densification: {strand_stats['strands_cloned']} strands cloned ({strand_stats['points_cloned']} new points)")
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if hair_params.fix_hair_positions:
                    gaussians.fix_hair_xyz_gradients()

                # Apply scale clamping
                apply_hair_scale_clamping(gaussians, iteration, hair_params)

                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = HairParams(parser)  # Add hair parameters
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, hp.extract(args))

    # All done
    print("\nTraining complete.")

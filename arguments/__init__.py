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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False

        # Hair-related parameters
        self.hair_init = False 
        self.hair_data = ""  # Path to hair strand data
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if hasattr(g, 'hair_data') and g.hair_data:
            g.hair_data = os.path.abspath(g.hair_data)
        return g


class HairParams(ParamGroup):
    """Hair-specific parameters for Gaussian4Hair"""
    def __init__(self, parser):
        # Hair initialization parameters
        self.hair_radius = 3e-4  # Cylinder radius for hair Gaussians
        self.hair_height = 3e-3  # Cylinder height for hair Gaussians
        self.hair_color = [0.6, 0.4, 0.2]  # Default hair color (RGB)
        self.hair_init_style = "cylinder"  # "cylinder" or "ellipsoid"
        
        # Hair filtering parameters  
        self.min_strand_points = 20  # Minimum points per strand to keep (原始版本使用20)
        self.hair_body_threshold = 0.1  # Distance threshold for removing body points near hair
        
        # Hair regularization weights (恢复原始版本的权重)
        self.lambda_opacity = 1.0  # Weight for opacity continuity loss
        self.lambda_geometry = 1.0  # Weight for geometry continuity loss
        self.lambda_direction = 0.1  # Weight for direction alignment loss
        
        # Hair training parameters
        self.fix_hair_positions = True  # Whether to fix hair positions during training
        self.hair_opacity_init = 0.8  # Initial opacity for hair Gaussians
        self.strand_clone_threshold = 0.3  # Threshold for smart strand cloning
        
        # Hair densification parameters
        self.enable_strand_clone = True  # Enable intelligent strand-level cloning
        self.hair_densify_grad_boost = 1.0  # Gradient boost for hair points
        
        super().__init__(parser, "Hair Parameters")


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 7_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 7_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 6_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

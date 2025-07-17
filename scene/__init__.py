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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import struct
from plyfile import PlyData

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], hair_params=None):
        """
        Initialize Scene with optional hair parameters
        
        Args:
            args: ModelParams containing dataset configuration
            gaussians: GaussianModel instance
            load_iteration: Optional iteration to load from checkpoint
            shuffle: Whether to shuffle cameras
            resolution_scales: List of resolution scales
            hair_params: HairParams instance with hair-specific configuration
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.hair_params = hair_params

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            if not args.hair_init:
                # Standard Gaussian Splatting initialization
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
            else:
                # Hair-aware initialization
                print("Initializing with hair data...")
                self._initialize_with_hair_data(args, scene_info)

    def _initialize_with_hair_data(self, args, scene_info):
        """Initialize Gaussians with hair data using parameters from hair_params"""
        
        # Check if hair data path is provided
        if not hasattr(args, 'hair_data') or not args.hair_data:
            print("Warning: hair_init is True but no hair_data path provided. Using standard initialization.")
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
            return
            
        if not os.path.exists(args.hair_data):
            print(f"Warning: Hair data file not found at {args.hair_data}. Using standard initialization.")
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
            return

        # Load hair data
        strands, tangents = self._load_hair_data(args.hair_data)
        
        if not strands:
            print("Warning: No valid hair strands loaded. Using standard initialization.")
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
            return

        # Get hair parameters (use defaults if hair_params is None)
        hair_radius = self.hair_params.hair_radius if self.hair_params else 1e-4
        hair_height = self.hair_params.hair_height if self.hair_params else 1e-3
        hair_color = self.hair_params.hair_color if self.hair_params else [0.6, 0.4, 0.2]
        hair_init_style = self.hair_params.hair_init_style if self.hair_params else "cylinder"
        
        print(f"Hair initialization parameters:")
        print(f"  - Radius: {hair_radius}")
        print(f"  - Height: {hair_height}")
        print(f"  - Color: {hair_color}")
        print(f"  - Style: {hair_init_style}")
        print(f"  - Strands loaded: {len(strands)}")

        # Initialize combined Gaussians
        self.gaussians.create_combined_gaussians(
            pcd=scene_info.point_cloud,
            cam_infos=scene_info.train_cameras,
            spatial_lr_scale=self.cameras_extent,
            strands=strands,
            tangents=tangents,
            radius=hair_radius,
            height=hair_height,
            hair_color=hair_color,
            include_body=True,
            hair_init_style=hair_init_style
        )

    def _load_hair_data(self, hair_file_path):
        """
        Load hair data from various formats
        
        Args:
            hair_file_path: Path to hair data file
            
        Returns:
            tuple: (strands, tangents) where strands is list of arrays
        """
        min_strand_length = self.hair_params.min_strand_points if self.hair_params else 5
        
        file_ext = os.path.splitext(hair_file_path)[1].lower()
        
        try:
            if file_ext == '.hair':
                return self._load_hair_format(hair_file_path, min_strand_length)
            elif file_ext == '.ply':
                return self._load_ply_format(hair_file_path, min_strand_length)
            else:
                print(f"Unsupported hair data format: {file_ext}")
                return [], []
                
        except Exception as e:
            print(f"Error loading hair data from {hair_file_path}: {e}")
            return [], []

    def _load_hair_format(self, hair_file_path, min_strand_length):
        """Load MonoHair .hair format"""
        with open(hair_file_path, mode='rb') as f:
            num_strand = f.read(4)
            (num_strand,) = struct.unpack('I', num_strand)
            point_count = f.read(4)
            (point_count,) = struct.unpack('I', point_count)

            segments = f.read(2 * num_strand)
            segments = struct.unpack('H' * num_strand, segments)
            segments = list(segments)

            num_points = sum(segments)

            points = f.read(4 * num_points * 3)
            points = struct.unpack('f' * num_points * 3, points)

        points = np.array(points).reshape(-1, 3)

        beg = 0
        strands = []
        tangents = []
        
        for seg in segments:
            end = beg + seg
            strand = points[beg:end]
            
            # Filter by minimum length
            if len(strand) >= min_strand_length:
                strands.append(strand)
                
                # Compute tangents
                if len(strand) > 1:
                    diffs = np.zeros_like(strand)
                    diffs[:-1] = strand[1:] - strand[:-1]  # Forward difference
                    diffs[-1] = strand[-1] - strand[-2]    # Last point uses previous diff
                    # Normalize
                    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
                    diffs = diffs / (norms + 1e-8)
                else:
                    diffs = np.array([[0.0, 0.0, 1.0]])
                tangents.append(diffs)
            
            beg += seg

        print(f"Loaded {len(strands)} strands from .hair file (filtered from {num_strand})")
        if strands:
            strand_lengths = [len(s) for s in strands]
            print(f"Strand lengths: min={min(strand_lengths)}, max={max(strand_lengths)}, avg={np.mean(strand_lengths):.1f}")
        
        return strands, tangents

    def _load_ply_format(self, hair_file_path, min_strand_length):
        """Load PLY format hair data (assumes fixed-length strands)"""
        ply_data = PlyData.read(hair_file_path)
        vertices = ply_data['vertex']
        
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        total_points = len(points)
        
        # Try common configurations
        possible_configs = [
            (1900, 100),  # Common MonoHair config
            (total_points // 100, 100),  # Auto-detect
        ]
        
        for n_strands, n_points_per_strand in possible_configs:
            if total_points == n_strands * n_points_per_strand:
                strands = points.reshape(n_strands, n_points_per_strand, 3)
                
                # Compute tangents efficiently
                diffs = np.diff(strands, axis=1)
                last_diffs = diffs[:, -1:, :]
                tangents = np.concatenate([diffs, last_diffs], axis=1)
                norms = np.linalg.norm(tangents, axis=2, keepdims=True)
                tangents = np.divide(tangents, norms, where=norms>0)
                
                # Convert to list format for consistency
                strands_list = [strands[i] for i in range(n_strands)]
                tangents_list = [tangents[i] for i in range(n_strands)]
                
                print(f"Loaded {n_strands} strands x {n_points_per_strand} points from PLY file")
                return strands_list, tangents_list
        
        raise ValueError(f"Cannot determine strand configuration for {total_points} points")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        os.makedirs(point_cloud_path, exist_ok=True)
        
        # Save main point cloud
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # Save additional visualization files if hair data exists
        if hasattr(self.gaussians, '_group_id'):
            self.gaussians.save_colored_ply(os.path.join(point_cloud_path, "pcd_colors.ply"))
            self.gaussians.save_hair_ply(os.path.join(point_cloud_path, "point_cloud_hair.ply"))
        
        # Save exposure data
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

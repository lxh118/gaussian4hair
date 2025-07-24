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

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from typing import Optional, List, Tuple
from scipy.spatial import KDTree

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        # TODO
        self._group_id = torch.empty(0, dtype=torch.long, device="cuda")  # 新增组标记属性
        self._strand_id = torch.empty(0, dtype=torch.long, device="cuda")  # 新增strand ID属性
        self.fix_hair_positions = True  # 新增：是否固定头发位置的标志

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # TODO
    @property
    def get_group_id(self):
        return self._group_id

    @property
    def get_strand_id(self):
        return self._strand_id
        
    def _get_tangents(self, group_ids, strand_ids):
        """计算每个点的切线方向（安全版本）"""
        # 创建复合键
        query_keys = (group_ids << 20) + strand_ids
        
        # 获取所有头发点
        hair_mask = self._group_id >= 0
        all_group_ids = self._group_id[hair_mask].clone()
        all_strand_ids = self._strand_id[hair_mask].clone()
        all_positions = self._xyz[hair_mask].clone()
        
        # 创建所有点的复合键
        all_keys = (all_group_ids << 20) + all_strand_ids
        
        # 排序所有键和位置
        sorted_indices = torch.argsort(all_keys)
        all_keys_sorted = all_keys[sorted_indices]
        all_positions_sorted = all_positions[sorted_indices]
        
        # 为每个查询点找到对应位置
        positions = torch.searchsorted(all_keys_sorted, query_keys)
        
        # 初始化切线
        tangents = torch.zeros_like(self._xyz[0:len(group_ids)])
        
        # 标记有效位置
        valid_positions = (positions > 0) & (positions < len(all_keys_sorted))
        
        # 验证匹配
        valid_positions_clone = valid_positions.clone()
        valid_positions_clone[valid_positions] &= (all_keys_sorted[positions[valid_positions]] == query_keys[valid_positions])
        
        # 计算切线（使用中心差分）
        if torch.any(valid_positions_clone):
            valid_idx = torch.where(valid_positions_clone)[0]
            pos_idx = positions[valid_idx]
            
            # 计算前向和后向差分
            forward_diff = all_positions_sorted[pos_idx + 1] - all_positions_sorted[pos_idx]
            backward_diff = all_positions_sorted[pos_idx] - all_positions_sorted[pos_idx - 1]
            
            # 使用中心差分并直接归一化
            tangent_vectors = (forward_diff + backward_diff) / 2.0
            norms = torch.norm(tangent_vectors, dim=1, keepdim=True).clamp(min=1e-6)
            tangent_vectors = tangent_vectors / norms
            tangents[valid_idx] = tangent_vectors
        
        return tangents

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def create_combined_gaussians(
        self, 
        pcd: BasicPointCloud, 
        cam_infos: int, 
        spatial_lr_scale: float,
        strands: Optional[np.ndarray] = None,
        tangents: Optional[np.ndarray] = None,
        radius: float = 3e-4 , # 1e-4
        height: float = 3e-3 , # 1e-3
        hair_color: List[float] = [0.6, 0.4, 0.2],
        include_body: bool = True,
        hair_init_style: str = "cylinder" #  "ellipsoid" "cylinder"
        ):
        """
        创建高斯模型（灵活控制头发和身体的初始化方式）
        
        参数:
            pcd: 身体点云（可为None）
            strands: 发丝数据（可为None）[N_strands, N_points_per_strand, 3]
            tangents: 发丝切线方向（cylinder模式需要）[N_strands, N_points_per_strand, 3]
            include_body: 是否包含身体部分
            hair_init_style: 头发初始化方式 ("cylinder"|"ellipsoid")
        """
        self.spatial_lr_scale = spatial_lr_scale
        
        # ===================== 初始化工具函数 =====================
        def init_ellipsoid(points: np.ndarray, colors: Optional[np.ndarray] = None) -> Tuple:
            """通用椭圆高斯初始化"""
            points_t = torch.tensor(points, dtype=torch.float32, device="cuda")
            
            # 自适应尺度计算
            dist2 = torch.clamp_min(distCUDA2(points_t), 1e-7)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            
            # 单位四元数旋转
            rots = torch.zeros((points_t.shape[0], 4), device="cuda")
            rots[:, 0] = 1.0
            
            # 颜色处理
            if colors is None:
                colors_t = torch.tensor(hair_color, device="cuda").float().repeat(points_t.shape[0], 1)
            else:
                colors_t = torch.tensor(colors, dtype=torch.float32, device="cuda")
            sh_color = RGB2SH(colors_t)
            
            return points_t, scales, rots, sh_color

        def init_cylinder(strands: np.ndarray, tangents: np.ndarray) -> Tuple:
            """头发圆柱高斯初始化"""
            # 输入验证
            if tangents is None:
                raise ValueError("Tangents must be provided for cylinder initialization")
            
            # 将所有strands的点展平成一个大数组
            if isinstance(strands, list):
                # 计算每个strand的长度
                strand_lengths = [len(strand) for strand in strands]
                # 展平所有点
                all_points = np.vstack(strands)
                all_tangents = np.vstack(tangents)
            else:
                # 兼容原有的固定长度格式
                all_points = strands.reshape(-1, 3)
                all_tangents = tangents.reshape(-1, 3)
            
            # 转换为tensor
            strand_t = torch.tensor(all_points, dtype=torch.float32, device="cuda")
            tangent_t = torch.tensor(all_tangents, dtype=torch.float32, device="cuda")
            
            # 处理零向量切线（向量化）
            zero_mask = torch.all(tangent_t == 0, dim=-1)
            tangent_t[zero_mask] = torch.tensor([0., 0., 1.], device="cuda")
            
            # 归一化切线方向（向量化）
            tangent_norms = torch.norm(tangent_t, dim=-1, keepdim=True)
            tangent_t = tangent_t / (tangent_norms + 1e-8)
            
            # 改进的旋转矩阵计算（完全向量化）
            z_axis = tangent_t  # z轴直接使用切线方向
            
            # 构造正交的x轴和y轴
            z_close_to_up = torch.abs(z_axis[..., 2]) > 0.9
            
            reference = torch.zeros_like(z_axis)
            reference[z_close_to_up] = torch.tensor([1., 0., 0.], device="cuda")
            reference[~z_close_to_up] = torch.tensor([0., 0., 1.], device="cuda")
            
            # 计算x轴（垂直于z轴）
            x_axis = torch.cross(z_axis, reference, dim=-1)
            x_axis_norm = torch.norm(x_axis, dim=-1, keepdim=True)
            x_axis = x_axis / (x_axis_norm + 1e-8)
            
            # 计算y轴（垂直于x轴和z轴）
            y_axis = torch.cross(z_axis, x_axis, dim=-1)
            y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + 1e-8)
            
            # 构造旋转矩阵 [x, y, z]
            rot_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)
            rotations = self.matrix_to_quaternion(rot_matrix)
            
            # 创建各向异性的圆柱参数
            radius_xy = radius  # 横截面半径
            length_z = height   # 沿发丝方向的长度
            scales = torch.log(torch.tensor([radius_xy, radius_xy, length_z], device="cuda")
                .repeat(strand_t.shape[0], 1))
            
            # 颜色处理
            color_t = torch.tensor(hair_color, dtype=torch.float32, device="cuda")
            hair_sh_color = RGB2SH(color_t.repeat(strand_t.shape[0], 1))
            
            print(f"圆柱高斯初始化完成:")
            print(f"  横截面半径: {radius_xy}")
            print(f"  轴向长度: {length_z}")
            print(f"  切线对齐: z轴与发丝切线对齐")
            print(f"  总点数: {strand_t.shape[0]}")
            
            # 如果输入是列表，返回strand_lengths供后续使用
            if isinstance(strands, list):
                return strand_t, scales, rotations, hair_sh_color, strand_lengths
            else:
                return strand_t, scales, rotations, hair_sh_color

        def prepare_features(points: torch.Tensor, colors: torch.Tensor, max_sh_degree: int) -> Tuple:
            """统一特征初始化"""
            # 确保颜色数据与点数匹配
            if colors.shape[0] != points.shape[0]:
                raise ValueError(f"Color count {colors.shape[0]} doesn't match point count {points.shape[0]}")
            
            features = torch.zeros((points.shape[0], 3, (max_sh_degree + 1) ** 2), device="cuda")
            features[:, :3, 0] = colors
            return (
                features[:, :, 0:1].transpose(1, 2).contiguous(),
                features[:, :, 1:].transpose(1, 2).contiguous()
            )

        # ===================== 数据准备 =====================
        # 1. 初始化头发数据
        hair_xyz = hair_scales = hair_rots = hair_sh_color = None
        strand_lengths = None  # 新增变量存储strand长度信息
        if strands is not None:
            if hair_init_style == "cylinder":
                cylinder_result = init_cylinder(strands, tangents)
                if len(cylinder_result) == 5:  # 不等长strands
                    hair_xyz, hair_scales, hair_rots, hair_sh_color, strand_lengths = cylinder_result
                else:  # 固定长度strands
                    hair_xyz, hair_scales, hair_rots, hair_sh_color = cylinder_result
                hair_xyz = hair_xyz.view(-1, 3)
                hair_scales = hair_scales.view(-1, 3)
                hair_rots = hair_rots.view(-1, 4)
            else:  # ellipsoid
                if isinstance(strands, list):
                    # 不等长strands，需要展平
                    strand_lengths = [len(strand) for strand in strands]
                    all_points = np.vstack(strands)
                else:
                    # 固定长度
                    all_points = strands.reshape(-1, 3)
                hair_xyz, hair_scales, hair_rots, hair_sh_color = init_ellipsoid(all_points)
        
        # 2. 初始化身体数据
        body_xyz = body_scales = body_rots = body_sh_color = None
        if include_body and pcd is not None:
            body_xyz, body_scales, body_rots, body_sh_color = init_ellipsoid(
                np.asarray(pcd.points), np.asarray(pcd.colors))
        
        # 3. 数据合并与验证
        if hair_xyz is None and body_xyz is None:
            raise ValueError("At least one of hair or body data must be provided")
        
        def remove_body_points_near_hair(hair_xyz, body_xyz, body_scales, body_rots, body_colors, threshold_distance=0.1):
            """
            移除靠近头发点的身体点，并同步更新其属性。
            
            参数:
                hair_xyz (torch.Tensor or np.ndarray): 头发点的位置数组，形状为 (N, 3)。
                body_xyz (torch.Tensor or np.ndarray): 身体点的位置数组，形状为 (M, 3)。
                body_scales (torch.Tensor or np.ndarray): 身体点的尺度数组，形状为 (M, *)。
                body_rots (torch.Tensor or np.ndarray): 身体点的旋转数组，形状为 (M, *)。
                body_colors (torch.Tensor or np.ndarray): 身体点的颜色数组，形状为 (M, *)。
                threshold_distance (float): 阈值距离，小于该距离的身体点将被移除。
                
            返回:
                tuple: 过滤后的身体点位置、尺度、旋转和颜色数组。
            """
            # 如果输入是PyTorch张量，则转换为NumPy数组
            if isinstance(hair_xyz, torch.Tensor):
                hair_xyz = hair_xyz.cpu().numpy()
            if isinstance(body_xyz, torch.Tensor):
                body_xyz = body_xyz.cpu().numpy()
            if isinstance(body_scales, torch.Tensor):
                body_scales = body_scales.cpu().numpy()
            if isinstance(body_rots, torch.Tensor):
                body_rots = body_rots.cpu().numpy()
            if isinstance(body_colors, torch.Tensor):
                body_colors = body_colors.cpu().numpy()

            if hair_xyz is None or len(hair_xyz) == 0:
                return body_xyz, body_scales, body_rots, body_colors  # 如果没有头发点，则无需过滤
            
            # 构建头发点的KDTree
            hair_tree = KDTree(hair_xyz)
            
            # 查询每个身体点到最近的头发点的距离
            distances, _ = hair_tree.query(body_xyz)
            
            # 找出保留的身体点索引
            keep_indices = distances > threshold_distance
            
            # 使用相同的索引更新所有属性
            filtered_body_xyz = body_xyz[keep_indices]
            filtered_body_scales = body_scales[keep_indices]
            filtered_body_rots = body_rots[keep_indices]
            filtered_body_colors = body_colors[keep_indices]
            
            return filtered_body_xyz, filtered_body_scales, filtered_body_rots, filtered_body_colors

   
        if hair_xyz is not None and body_xyz is not None:
            body_xyz, body_scales, body_rots, body_sh_color = remove_body_points_near_hair(
                hair_xyz, body_xyz, body_scales, body_rots, body_sh_color, threshold_distance=0.1)
            
            # 将 NumPy 数组重新转换为 PyTorch 张量
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            body_xyz = torch.tensor(body_xyz, dtype=torch.float32).to(device)
            body_scales = torch.tensor(body_scales, dtype=torch.float32).to(device)
            body_rots = torch.tensor(body_rots, dtype=torch.float32).to(device)
            body_sh_color = torch.tensor(body_sh_color, dtype=torch.float32).to(device)

        # 合并点数据
        all_xyz = []
        all_scales = []
        all_rots = []
        all_colors = []
        
        if hair_xyz is not None:
            all_xyz.append(hair_xyz)
            all_scales.append(hair_scales)
            all_rots.append(hair_rots)
            all_colors.append(hair_sh_color)
        
        if body_xyz is not None:
            all_xyz.append(body_xyz)
            all_scales.append(body_scales)
            all_rots.append(body_rots)
            all_colors.append(body_sh_color)
        
        xyz = torch.cat(all_xyz, dim=0)
        scales = torch.cat(all_scales, dim=0)
        rots = torch.cat(all_rots, dim=0)
        merged_colors = torch.cat(all_colors, dim=0)
        
        # 特征初始化
        features_dc, features_rest = prepare_features(xyz, merged_colors, self.max_sh_degree)
        
        # ===================== 分组信息 =====================
        # 初始化所有点为-1
        group_id = torch.full((xyz.shape[0],), -1, dtype=torch.long, device="cuda")
        self._strand_id = torch.full((xyz.shape[0],), -1, dtype=torch.long, device="cuda")

        if hair_xyz is not None and hair_init_style == "cylinder":
            # 圆柱高斯初始化设置详细分组
            if strand_lengths is not None:  # 不等长strands
                # 使用向量化操作分配group_id
                strand_lengths_tensor = torch.tensor(strand_lengths, dtype=torch.long, device="cuda")
                n_strands = len(strand_lengths)
                group_id[:hair_xyz.shape[0]] = torch.repeat_interleave(
                    torch.arange(n_strands, device="cuda"), 
                    strand_lengths_tensor
                )
                
                # 使用向量化操作分配strand_id
                # 创建一个累积索引数组
                cumsum = torch.cumsum(torch.cat([torch.zeros(1, device="cuda", dtype=torch.long), strand_lengths_tensor]), dim=0)
                # 使用searchsorted找到每个位置属于哪个strand
                positions = torch.arange(hair_xyz.shape[0], device="cuda")
                strand_assignment = torch.searchsorted(cumsum[1:], positions, right=True)
                # 计算每个点在其strand内的局部索引
                self._strand_id[:hair_xyz.shape[0]] = positions - cumsum[strand_assignment]
            else:  # 固定长度strands（向后兼容）
                n_strands = strands.shape[0]
                n_points_per_strand = strands.shape[1]
                group_id[:hair_xyz.shape[0]] = torch.arange(n_strands, device="cuda").repeat_interleave(n_points_per_strand)
                self._strand_id[:hair_xyz.shape[0]] = torch.arange(n_points_per_strand, device="cuda").repeat(n_strands)
        
        # ===================== 参数设置 =====================
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))

        # self._opacity = nn.Parameter(0.1 * torch.ones((xyz.shape[0], 1), device="cuda"))

        # 分别设置头发和身体的不透明度
        opacities = torch.ones((xyz.shape[0], 1), device="cuda") * 0.1  # 默认身体不透明度
        if hair_xyz is not None:
            # 头发高斯使用更高的初始不透明度
            hair_opacity_init = 0.8  # 头发初始不透明度设为0.8
            opacities[:hair_xyz.shape[0]] = hair_opacity_init
            print(f"头发高斯初始不透明度设置为: {hair_opacity_init}")
        
        self._opacity = nn.Parameter(opacities)
        self._group_id = group_id
        
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam.image_name: i for i, cam in enumerate(cam_infos)}
        self._exposure = nn.Parameter(
            torch.eye(3,4,device="cuda")[None].repeat(len(cam_infos),1,1).requires_grad_(True))
        self.pretrained_exposures = None

        print(f"初始化完成 - 总点数: {self._xyz.shape[0]}")

    def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        鲁棒的旋转矩阵转四元数实现（支持批量处理）
        参考: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        
        参数:
            matrix: [..., 3, 3] 的旋转矩阵
            
        返回:
            quaternion: [..., 4] 的四元数 (w, x, y, z) (COLMAP顺序)
        """
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"输入矩阵形状应为[...,3,3]，实际得到{matrix.shape}")
        
        m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
        m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
        m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
        
        trace = m00 + m11 + m22
        q = torch.empty((*matrix.shape[:-2], 4), device=matrix.device, dtype=matrix.dtype)

        # Reshape all tensors to 2D for easier indexing
        orig_shape = q.shape
        q_flat = q.view(-1, 4)
        trace_flat = trace.view(-1)
        m00_flat = m00.view(-1)
        m11_flat = m11.view(-1)
        m22_flat = m22.view(-1)
        m01_flat = m01.view(-1)
        m10_flat = m10.view(-1)
        m02_flat = m02.view(-1)
        m20_flat = m20.view(-1)
        m12_flat = m12.view(-1)
        m21_flat = m21.view(-1)

        # 分支1: trace > 0
        cond1 = trace_flat > 0
        S1 = torch.sqrt(trace_flat[cond1] + 1.0) * 2  # S=4*w
        q_flat[cond1, 0] = 0.25 * S1  # w
        q_flat[cond1, 1] = (m21_flat[cond1] - m12_flat[cond1]) / S1  # x
        q_flat[cond1, 2] = (m02_flat[cond1] - m20_flat[cond1]) / S1  # y
        q_flat[cond1, 3] = (m10_flat[cond1] - m01_flat[cond1]) / S1  # z

        # 分支2: m00最大
        cond2 = (~cond1) & (m00_flat > m11_flat) & (m00_flat > m22_flat)
        S2 = torch.sqrt(1.0 + m00_flat[cond2] - m11_flat[cond2] - m22_flat[cond2]) * 2  # S=4*x
        q_flat[cond2, 0] = (m21_flat[cond2] - m12_flat[cond2]) / S2  # w
        q_flat[cond2, 1] = 0.25 * S2  # x
        q_flat[cond2, 2] = (m01_flat[cond2] + m10_flat[cond2]) / S2  # y
        q_flat[cond2, 3] = (m02_flat[cond2] + m20_flat[cond2]) / S2  # z

        # 分支3: m11最大
        cond3 = (~cond1) & (~cond2) & (m11_flat > m22_flat)
        S3 = torch.sqrt(1.0 + m11_flat[cond3] - m00_flat[cond3] - m22_flat[cond3]) * 2  # S=4*y
        q_flat[cond3, 0] = (m02_flat[cond3] - m20_flat[cond3]) / S3  # w
        q_flat[cond3, 1] = (m01_flat[cond3] + m10_flat[cond3]) / S3  # x
        q_flat[cond3, 2] = 0.25 * S3  # y
        q_flat[cond3, 3] = (m12_flat[cond3] + m21_flat[cond3]) / S3  # z

        # 分支4: m22最大
        cond4 = (~cond1) & (~cond2) & (~cond3)
        S4 = torch.sqrt(1.0 + m22_flat[cond4] - m00_flat[cond4] - m11_flat[cond4]) * 2  # S=4*z
        q_flat[cond4, 0] = (m10_flat[cond4] - m01_flat[cond4]) / S4  # w
        q_flat[cond4, 1] = (m02_flat[cond4] + m20_flat[cond4]) / S4  # x
        q_flat[cond4, 2] = (m12_flat[cond4] + m21_flat[cond4]) / S4  # y
        q_flat[cond4, 3] = 0.25 * S4  # z

        return q_flat.view(orig_shape)  # (w, x, y, z) COLMAP顺序

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                         lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                         lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                         max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        # 添加group_id和strand_id属性
        l.append('group_id')  # TODO
        l.append('strand_id')

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 检查并处理group_id和strand_id数据
        if self._group_id.numel() > 0 and self._group_id.shape[0] == xyz.shape[0]:
            group_id = self._group_id.detach().cpu().numpy().astype(np.int32)
            strand_id = self._strand_id.detach().cpu().numpy().astype(np.int32)
        else:
            # 如果没有头发数据，使用默认值-1
            group_id = np.full((xyz.shape[0],), -1, dtype=np.int32)
            strand_id = np.full((xyz.shape[0],), -1, dtype=np.int32)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 修改group_id和strand_id的数据类型为整数
        dtype_full[-2] = ('group_id', 'i4')  # int32
        dtype_full[-1] = ('strand_id', 'i4')  # int32

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # 始终包含group_id和strand_id，但在没有头发数据时使用-1
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, 
                            group_id.reshape(-1, 1), strand_id.reshape(-1, 1)), axis=1)
      
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print(f"Saved {xyz.shape[0]} gaussians with group_id and strand_id to {path}")

    def save_colored_ply(self, path):
        import os
        import numpy as np
        from plyfile import PlyData, PlyElement

        # 创建输出目录
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 准备基础数据
        xyz = self._xyz.detach().cpu().numpy().astype(np.float32)
        
        # 检查是否有头发数据
        if self._group_id.numel() > 0 and self._group_id.shape[0] == xyz.shape[0]:
            group_ids = self._group_id.cpu().numpy()
            valid_mask = group_ids >= 0
            
            # 生成颜色映射
            unique_groups = np.unique(group_ids[valid_mask])
            color_map = {g: np.random.randint(0, 256, 3, dtype=np.uint8) for g in unique_groups}
            
            # 生成颜色数组
            colors = np.full((len(group_ids), 3), 128, dtype=np.uint8)  # 背景点默认灰色
            for idx, gid in enumerate(group_ids):
                if gid in color_map:
                    colors[idx] = color_map[gid]
        else:
            # 没有头发数据时，所有点都使用默认灰色
            colors = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)
            valid_mask = np.zeros(xyz.shape[0], dtype=bool)  # 没有有效的头发点

        # 构建PLY元素的通用函数
        def create_ply_element(vertices, vertex_colors):
            vertex_data = np.zeros(vertices.shape[0], dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            vertex_data['x'] = vertices[:, 0]
            vertex_data['y'] = vertices[:, 1]
            vertex_data['z'] = vertices[:, 2]
            vertex_data['red'] = vertex_colors[:, 0]
            vertex_data['green'] = vertex_colors[:, 1]
            vertex_data['blue'] = vertex_colors[:, 2]
            return PlyElement.describe(vertex_data, 'vertex')

        # 保存带背景的完整点云
        full_ply = PlyData([create_ply_element(xyz, colors)])
        full_ply.write(path)  # 原始路径保存完整点云

        # 保存过滤后的点云（仅group_id >= 0）
        if np.any(valid_mask):
            filtered_path = os.path.join(
                os.path.dirname(path),
                f"{os.path.splitext(os.path.basename(path))[0]}_filtered.ply"
            )
            filtered_ply = PlyData([create_ply_element(
                xyz[valid_mask], 
                colors[valid_mask]
            )])
            filtered_ply.write(filtered_path)
        else:
            print("没有符合条件 (group_id >= 0) 的点，未生成过滤PLY文件。")
   
    def save_hair_ply(self, path):
        """保存只有头发部分的高斯（group_id >= 0）"""
        mkdir_p(os.path.dirname(path))
        
        # 检查是否有group_id属性
        if not hasattr(self, '_group_id'):
            print("Warning: No group_id found, saving all gaussians as hair")
            self.save_ply(path)
            return
        
        # 创建头发mask（group_id >= 0）
        hair_mask = self._group_id >= 0
        
        if not torch.any(hair_mask):
            print("Warning: No hair gaussians found (group_id >= 0)")
            # 创建空的PLY文件
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            dtype_full[-2] = ('group_id', 'i4')  # int32
            dtype_full[-1] = ('strand_id', 'i4')  # int32

            elements = np.empty(0, dtype=dtype_full)
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
            return
        
        # 提取头发部分的数据
        xyz = self._xyz[hair_mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[hair_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[hair_mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[hair_mask].detach().cpu().numpy()
        scale = self._scaling[hair_mask].detach().cpu().numpy()
        rotation = self._rotation[hair_mask].detach().cpu().numpy()

        # 添加头发的group_id和strand_id数据
        group_id = self._group_id[hair_mask].detach().cpu().numpy().astype(np.int32)
        strand_id = self._strand_id[hair_mask].detach().cpu().numpy().astype(np.int32)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        dtype_full[-2] = ('group_id', 'i4')  # int32
        dtype_full[-1] = ('strand_id', 'i4')  # int32

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,
                            group_id.reshape(-1, 1), strand_id.reshape(-1, 1)), axis=1)
        

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        print(f"Saved {xyz.shape[0]} hair gaussians with group_id and strand_id to {path}")

    def reset_opacity(self):
        # 为头发和身体设置不同的不透明度下限
        if hasattr(self, '_group_id'):
            hair_mask = self._group_id >= 0
            body_mask = self._group_id == -1
            
            # 头发高斯保持更高的不透明度下限
            hair_min_opacity = 0.3  # 头发最小不透明度
            body_min_opacity = 0.01  # 身体最小不透明度
            
            current_opacity = self.get_opacity
            new_opacity = current_opacity.clone()
            
            if hair_mask.any():
                hair_opacity_limit = torch.ones_like(current_opacity[hair_mask]) * hair_min_opacity
                new_opacity[hair_mask] = torch.max(current_opacity[hair_mask], hair_opacity_limit)
                
            if body_mask.any():
                body_opacity_limit = torch.ones_like(current_opacity[body_mask]) * body_min_opacity
                new_opacity[body_mask] = torch.min(current_opacity[body_mask], body_opacity_limit)
            
            opacities_new = self.inverse_opacity_activation(new_opacity)
            print(f"不透明度重置 - 头发最小值: {hair_min_opacity}, 身体最小值: {body_min_opacity}")
        else:
            # 原始逻辑（没有group_id时）
            opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        

    def load_ply(self, path, use_train_test_exp=False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for
                                             image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        # 加载group_id和strand_id（如果存在）
        if 'group_id' in [p.name for p in plydata.elements[0].properties]:
            group_id = np.asarray(plydata.elements[0]["group_id"]).astype(np.int64)
            self._group_id = torch.tensor(group_id, dtype=torch.long, device="cuda")
            print(f"Loaded group_id: {len(np.unique(group_id))} unique groups")
        else:
            # 如果没有group_id，设置默认值（所有点为身体）
            self._group_id = torch.full((xyz.shape[0],), -1, dtype=torch.long, device="cuda")
            print("No group_id found in PLY file, setting all gaussians as body (group_id=-1)")
        
        if 'strand_id' in [p.name for p in plydata.elements[0].properties]:
            strand_id = np.asarray(plydata.elements[0]["strand_id"]).astype(np.int64)
            self._strand_id = torch.tensor(strand_id, dtype=torch.long, device="cuda")
            print(f"Loaded strand_id: range {strand_id.min()}-{strand_id.max()}")
        else:
            # 如果没有strand_id，设置默认值
            self._strand_id = torch.full((xyz.shape[0],), -1, dtype=torch.long, device="cuda")
            print("No strand_id found in PLY file, setting all strand_id=-1")

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask

        # 修剪属性
        self._group_id = self._group_id[valid_points_mask]
        self._strand_id = self._strand_id[valid_points_mask] 

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        try:
            if hasattr(self, 'tmp_radii'):
                self.tmp_radii = self.tmp_radii[valid_points_mask]
        except Exception as e:
            print("############ prune big gaussian after densification ######################")
            pass



    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        # 如果tmp_radii不存在，说明这不是在densify_and_prune期间调用的，跳过即可
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        hair_mask = self._group_id >= 0
        # print(sum(hair_mask))
        # print(padded_grad[hair_mask][:15]) # 1e-7
        boosted_grads = padded_grad.clone()
        # boosted_grads[hair_mask] *= 100

        # grad_threshold: 2e-4
        selected_pts_mask = torch.where(boosted_grads >= grad_threshold, True, False)

        # print(self.percent_dense * scene_extent) # 0.01*5.18
        # print(self.get_scaling[hair_mask].max(dim=1).values[:15]) # 0.001
        # 分组设置阈值
        body_threshold = self.percent_dense * scene_extent
        # hair_threshold = 0.00005 * scene_extent  # 头发更敏感  
        hair_threshold = body_threshold  # 没有启用头发敏感分裂
        # 分裂条件
        split_mask = torch.where(
            hair_mask,
            self.get_scaling.max(dim=1).values > hair_threshold,
            self.get_scaling.max(dim=1).values > body_threshold
        )
        selected_pts_mask = torch.logical_and(selected_pts_mask, split_mask)
 
        # ================== 新增部分：处理头发strand的特殊分裂 ==================
        # 获取选择点的属性
        selected_group_ids = self._group_id[selected_pts_mask].clone()
        selected_strand_ids = self._strand_id[selected_pts_mask].clone()
        is_hair = selected_group_ids >= 0

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)

        print("分裂总数和头发部分分裂数：", selected_pts_mask.sum(), sum(is_hair))
        # 头发strand沿切线方向分裂
        if torch.any(is_hair):
            # 获取头发点的切线方向
            hair_mask = selected_group_ids >= 0
            hair_indices = torch.where(hair_mask)[0]
            
            # 使用修改后的安全版本获取切线
            tangents = self._get_tangents(selected_group_ids[hair_mask], selected_strand_ids[hair_mask])
            
            # 沿切线方向偏移
            hair_samples = samples[hair_indices].clone()
            hair_rots = build_rotation(self._rotation[selected_pts_mask][hair_mask])
            hair_samples = torch.bmm(hair_rots, hair_samples.unsqueeze(-1)).squeeze(-1)
            
            # 调整位移方向（50%沿正方向，50%沿负方向）
            dir_sign = torch.where(torch.rand(len(hair_indices), device="cuda")>0.5, 1.0, -1.0)
            tangent_offset = dir_sign[:, None] * tangents * self.get_scaling[selected_pts_mask][hair_mask, 2:3]*0.3
            # 在现有切线偏移基础上添加噪声
            random_offset = torch.randn_like(tangent_offset) * 0.1 * self.get_scaling[selected_pts_mask][hair_mask, 2:3]
            hair_samples += (tangent_offset + random_offset)

            # print(hair_indices[:20], hair_samples[:20])
            samples[hair_indices] = hair_samples

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
        
        # ================== 新增部分：处理头发strand的特殊分裂 ==================
        # 获取选择点的属性
        selected_group_ids = self._group_id[selected_pts_mask].clone()
        selected_strand_ids = self._strand_id[selected_pts_mask].clone()
        is_hair = selected_group_ids >= 0

        print("分裂总数和头发部分分裂数：", selected_pts_mask.sum(), sum(is_hair))
        
        # 为分裂的点生成新的group_id和strand_id
        total_new_points = selected_pts_mask.sum().item() * N
        
        # 生成新的group_id（为头发高斯分配新的group_id，身体高斯保持-1）
        if torch.any(is_hair):
            # 获取当前最大的group_id
            max_existing_group_id = torch.max(self._group_id).item()
            
            # 计算有多少个头发点需要分裂
            hair_points_count = is_hair.sum().item()
            total_hair_new_points = hair_points_count * N
            
            # 为每个分裂出来的头发点分配新的group_id
            # 策略：每个原始点分裂出N个新点，每个新点都有独立的group_id
            new_hair_group_ids = torch.arange(
                max_existing_group_id + 1, 
                max_existing_group_id + 1 + total_hair_new_points,
                device="cuda", 
                dtype=torch.long
            )
            
            # 创建完整的新group_id数组
            new_group_id = torch.full((total_new_points,), -1, dtype=torch.long, device="cuda")
            
            # 为头发点分配新的group_id
            hair_mask_expanded = is_hair.repeat(N)
            new_group_id[hair_mask_expanded] = new_hair_group_ids
            
            # 身体点保持原来的group_id（-1）
            body_mask_expanded = ~hair_mask_expanded
            new_group_id[body_mask_expanded] = selected_group_ids.repeat(N)[body_mask_expanded]
            
            print(f"为分裂创建了 {total_hair_new_points} 个新头发strand (group_id: {max_existing_group_id + 1} - {max_existing_group_id + total_hair_new_points})")
        else:
            # 如果没有头发点，直接复制原有的group_id
            new_group_id = selected_group_ids.repeat(N)
        
        # strand_id设置：新分裂的点在每个新strand内从0开始
        new_strand_id = torch.zeros_like(new_group_id, dtype=torch.long)
        
        # 更新属性
        self._group_id = torch.cat([self._group_id, new_group_id])
        self._strand_id = torch.cat([self._strand_id, new_strand_id])
        
        # 修剪原始点（分裂后移除原始点）
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, iteration):
        # Extract points that satisfy the gradient condition

        hair_mask = self._group_id >= 0
        # print(grads[:15])

        # 动态梯度放大（随训练衰减）
        # hair_grad_boost = max(100 / (1 + iteration//100), 1.0)
        boosted_grads = grads.clone()  # 未启用动态放大
        # boosted_grads[hair_mask] *= hair_grad_boost

        # grad_threshold: 2e-4
        selected_pts_mask = torch.where(torch.norm(boosted_grads, dim=-1) >= grad_threshold, True, False)
    
        body_threshold = self.percent_dense * scene_extent
        # hair_threshold = 0.00005 * scene_extent  # 头发更敏感  
        hair_threshold = body_threshold  # 应用原始方法

        # 分裂条件
        split_mask = torch.where(
            hair_mask,
            self.get_scaling.max(dim=1).values <= hair_threshold,
            self.get_scaling.max(dim=1).values <= body_threshold
        )

        selected_pts_mask = torch.logical_and(selected_pts_mask, split_mask)

        # 获取选择点的属性
        selected_group_ids = self._group_id[selected_pts_mask]
        is_hair = selected_group_ids >= 0
        print("克隆总数和头发部分克隆数：",selected_pts_mask.sum(), sum(is_hair))
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # 为克隆的头发高斯生成新的group_id
        selected_group_ids = self._group_id[selected_pts_mask]
        selected_strand_ids = self._strand_id[selected_pts_mask]
        is_hair = selected_group_ids >= 0
        
        print("克隆总数和头发部分克隆数：", selected_pts_mask.sum(), sum(is_hair))
        
        if torch.any(is_hair):
            # 获取当前最大的group_id
            max_existing_group_id = torch.max(self._group_id).item()
            
            # 计算有多少个头发点需要克隆
            hair_points_count = is_hair.sum().item()
            
            # 为每个克隆出来的头发点分配新的group_id
            new_hair_group_ids = torch.arange(
                max_existing_group_id + 1, 
                max_existing_group_id + 1 + hair_points_count,
                device="cuda", 
                dtype=torch.long
            )
            
            # 创建新的group_id数组
            new_group_id = selected_group_ids.clone()
            new_group_id[is_hair] = new_hair_group_ids
            
            print(f"为克隆创建了 {hair_points_count} 个新头发strand (group_id: {max_existing_group_id + 1} - {max_existing_group_id + hair_points_count})")
        else:
            # 如果没有头发点，直接复制原有的group_id
            new_group_id = selected_group_ids
        
        # strand_id设置：新克隆的头发点strand_id设为0，身体点保持原值
        new_strand_id = selected_strand_ids.clone()
        new_strand_id[is_hair] = 0  # 新克隆的头发strand只有一个点，strand_id为0

        # 更新属性
        self._group_id = torch.cat([self._group_id, new_group_id])
        self._strand_id = torch.cat([self._strand_id, new_strand_id])

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, iteration):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 确保tmp_radii的大小与当前高斯点数量匹配
        # 如果radii的大小与当前点数不匹配，需要调整
        current_point_count = self.get_xyz.shape[0]
        if radii.shape[0] != current_point_count:
            # 如果radii比较小，说明在之前的操作中新增了点，需要扩展radii
            if radii.shape[0] < current_point_count:
                # 为新增的点添加默认的radii值（使用0）
                additional_radii = torch.zeros(current_point_count - radii.shape[0], device=radii.device)
                radii = torch.cat([radii, additional_radii])
            else:
                # 如果radii比较大，截取到当前大小
                radii = radii[:current_point_count]
        
        self.tmp_radii = radii
        
        # TODO
        # self.densify_and_clone(grads, max_grad, extent, iteration)
        # self.densify_and_split(grads, max_grad, extent)

        # 只对头发部分（group_id >= 0）执行致密化
        hair_mask = self._group_id >= 0
        if torch.any(hair_mask):
            # 只对头发部分的梯度执行致密化
            hair_grads = torch.zeros_like(grads)
            hair_grads[hair_mask] = grads[hair_mask]
            
            # 执行头发部分的致密化
            self.densify_and_clone(hair_grads, max_grad, extent, iteration)
            self.densify_and_split(hair_grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def fix_hair_xyz_gradients(self):
        """
        固定头发高斯的xyz位置，清零其梯度
        """
        if self.fix_hair_positions and self._xyz.grad is not None:
            # 获取头发高斯的mask (group_id >= 0)
            hair_mask = self._group_id >= 0
            if hair_mask.any():
                # 使用乘法操作而不是直接赋值
                self._xyz.grad[hair_mask] = 0.0

    def set_fix_hair_positions(self, fix: bool):
        """
        设置是否固定头发位置
        
        参数:
            fix: True表示固定头发位置，False表示允许头发位置更新
        """
        self.fix_hair_positions = fix
        print(f"Hair position fixing set to: {fix}")

    def clone_entire_strands_vectorized(self, strand_group_ids, iteration, move_distance_scale=20):
        """
        高效的整条strand批量克隆 - 基于几何的方法
        
        Args:
            strand_group_ids: 需要clone的strand的group_id列表或tensor
            iteration: current iteration
            move_distance_scale: 移动距离缩放因子
            
        Returns:
            dict: 克隆统计信息 {"strands_cloned": int, "points_cloned": int, "new_group_ids": list}
        """
        if len(strand_group_ids) == 0:
            return {"strands_cloned": 0, "points_cloned": 0, "new_group_ids": []}
            
        # 转换为tensor（如果输入是列表）
        if isinstance(strand_group_ids, list):
            strand_group_ids = torch.tensor(strand_group_ids, device="cuda", dtype=torch.long)
        
        # 确保strand_group_ids是唯一的
        strand_group_ids = torch.unique(strand_group_ids)
        
        # 创建所有需要克隆的strand的mask（张量化）
        all_group_ids = self._group_id.unsqueeze(1)  # [N, 1]
        strand_groups_expanded = strand_group_ids.unsqueeze(0)  # [1, n_strands]
        strand_masks = (all_group_ids == strand_groups_expanded)  # [N, n_strands]
        all_strands_mask = torch.any(strand_masks, dim=1)  # [N]
        
        if not torch.any(all_strands_mask):
            return {"strands_cloned": 0, "points_cloned": 0, "new_group_ids": []}
            
        # 获取所有相关数据
        strand_indices = torch.where(all_strands_mask)[0]
        strand_group_ids_all = self._group_id[strand_indices]
        
        # 验证所有选中的点都属于要克隆的strands
        valid_mask = torch.isin(strand_group_ids_all, strand_group_ids)
        if not torch.all(valid_mask):
            strand_indices = strand_indices[valid_mask]
            strand_group_ids_all = strand_group_ids_all[valid_mask]
        
        if len(strand_indices) == 0:
            return {"strands_cloned": 0, "points_cloned": 0, "new_group_ids": []}
        
        # 记录克隆前的点数
        points_before_clone = len(self._xyz)
        
        # 对于整条strand克隆，我们不需要单个点的梯度
        # 而是基于strand的平均属性来移动
        # 计算每个strand的平均尺度和移动方向
        unique_groups = strand_group_ids
        n_groups = len(unique_groups)
        
        # 创建group到index的映射
        strand_group_ids_expanded = strand_group_ids_all.unsqueeze(1)  # [M, 1]
        unique_groups_expanded = unique_groups.unsqueeze(0)  # [1, n_groups]
        matches = (strand_group_ids_expanded == unique_groups_expanded)  # [M, n_groups]
        
        # 验证每个点都能找到对应的group
        has_match = torch.any(matches, dim=1)  # [M]
        if not torch.all(has_match):
            valid_mask = has_match
            strand_indices = strand_indices[valid_mask]
            strand_group_ids_all = strand_group_ids_all[valid_mask]
            
            # 重新计算匹配
            strand_group_ids_expanded = strand_group_ids_all.unsqueeze(1)
            matches = (strand_group_ids_expanded == unique_groups_expanded)
        
        # 使用argmax找到每个点的group索引
        group_inverse = torch.argmax(matches.float(), dim=1)  # [M]
        
        # 计算每个group的点数
        group_counts = torch.zeros(n_groups, device="cuda")
        group_counts.scatter_add_(0, group_inverse, torch.ones_like(group_inverse, dtype=torch.float32))
        
        # 计算每个strand的平均尺度
        strand_scales = self.get_scaling[strand_indices]  # [M, 3]
        scale_sum = torch.zeros(n_groups, 3, device="cuda")
        scale_sum.scatter_add_(0, group_inverse.unsqueeze(1).expand(-1, 3), strand_scales)
        avg_scales = scale_sum / group_counts.unsqueeze(1).clamp(min=1.0)  # [n_groups, 3]
        
        # 计算每个strand的中心位置
        strand_positions = self._xyz[strand_indices]  # [M, 3]
        position_sum = torch.zeros(n_groups, 3, device="cuda")
        position_sum.scatter_add_(0, group_inverse.unsqueeze(1).expand(-1, 3), strand_positions)
        strand_centers = position_sum / group_counts.unsqueeze(1).clamp(min=1.0)  # [n_groups, 3]
        
        # 为每个strand生成一个移动方向
        # 使用径向外扩的方向（从strand中心向外）
        # 完全向量化的方法
        
        # 计算每个点相对于其strand中心的方向
        point_centers = strand_centers[group_inverse]  # [M, 3]
        radial_dirs = strand_positions - point_centers  # [M, 3]
        
        # 对每个strand计算平均径向方向
        radial_sum = torch.zeros(n_groups, 3, device="cuda")
        radial_sum.scatter_add_(0, group_inverse.unsqueeze(1).expand(-1, 3), radial_dirs)
        avg_radials = radial_sum / group_counts.unsqueeze(1).clamp(min=1.0)  # [n_groups, 3]
        
        # 归一化方向
        radial_norms = torch.norm(avg_radials, dim=1, keepdim=True)
        move_directions = avg_radials / (radial_norms + 1e-8)
        
        # 计算梯度方向（使用_xyz.grad）
        gradient_directions = torch.zeros_like(radial_dirs)
        if self._xyz.grad is not None:
            # 计算每个strand的平均梯度方向
            grad_sum = torch.zeros(n_groups, 3, device="cuda")
            point_grads = self._xyz.grad[strand_indices]  # [M, 3] - 直接使用xyz梯度
            grad_sum.scatter_add_(0, group_inverse.unsqueeze(1).expand(-1, 3), point_grads)
            avg_grads = grad_sum / group_counts.unsqueeze(1).clamp(min=1e-8)
            
            # 归一化梯度方向
            grad_norms = torch.norm(avg_grads, dim=1, keepdim=True)
            gradient_directions = avg_grads / (grad_norms + 1e-8)
        
        # 使用自适应方向
        move_directions = self.adaptive_clone_direction(iteration, gradient_directions, move_directions)
        
        # 对于太小的strand（中心和点重合），使用随机方向
        small_strand_mask = radial_norms.squeeze() <= 1e-8
        if torch.any(small_strand_mask):
            random_dirs = torch.randn(small_strand_mask.sum(), 3, device="cuda")
            random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)
            move_directions[small_strand_mask] = random_dirs
        
        # 计算移动距离
        move_distances = torch.mean(avg_scales, dim=1) * move_distance_scale  # [n_groups]
        
        # 计算每个点的移动向量
        point_move_vectors = move_directions[group_inverse] * move_distances[group_inverse].unsqueeze(1)  # [M, 3]
        
        # 批量克隆所有数据
        new_xyz = self._xyz[strand_indices] + point_move_vectors
        new_features_dc = self._features_dc[strand_indices].clone()
        new_features_rest = self._features_rest[strand_indices].clone()
        new_opacity = self._opacity[strand_indices].clone()
        new_scaling = self._scaling[strand_indices].clone()
        new_rotation = self._rotation[strand_indices].clone()
        
        # 为克隆的strand分配新的唯一group_id
        # 找到当前最大的group_id，从下一个开始分配
        max_existing_group_id = torch.max(self._group_id).item()
        
        # 创建新的group_id映射（向量化版本）
        old_group_ids = self._group_id[strand_indices]
        
        # 为每个unique group创建新的group_id序列
        new_group_ids_for_unique = torch.arange(
            max_existing_group_id + 1, 
            max_existing_group_id + 1 + len(unique_groups), 
            device=unique_groups.device, 
            dtype=unique_groups.dtype
        )
        
        # 使用向量化操作进行映射
        # 找到每个old_group_id在unique_groups中的位置
        old_group_ids_expanded = old_group_ids.unsqueeze(1)  # [M, 1]
        unique_groups_expanded = unique_groups.unsqueeze(0)  # [1, n_groups]
        matches = (old_group_ids_expanded == unique_groups_expanded)  # [M, n_groups]
        indices = torch.argmax(matches.float(), dim=1)  # [M] - 每个点对应的unique_group索引
        
        # 使用indices来获取新的group_id
        new_group_id = new_group_ids_for_unique[indices]
        
        # strand_id保持相对结构不变（在每个strand内部的相对位置）
        new_strand_id = self._strand_id[strand_indices].clone()
        
        # 检查tmp_radii是否存在，如果不存在则使用max_radii2D
        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            new_tmp_radii = self.tmp_radii[strand_indices].clone()
        else:
            # 使用max_radii2D作为替代
            new_tmp_radii = self.max_radii2D[strand_indices].clone()
        
        # 添加到模型
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation,
            new_tmp_radii
        )
        
        # 更新strand属性
        self._group_id = torch.cat([self._group_id, new_group_id])
        self._strand_id = torch.cat([self._strand_id, new_strand_id])
        
        # 计算统计信息
        points_after_clone = len(self._xyz)
        points_cloned = points_after_clone - points_before_clone
        strands_cloned = len(unique_groups)
        new_group_ids_list = new_group_ids_for_unique.tolist()
        
        # 打印详细的克隆信息
        print(f"[ITER {iteration}] STRAND CLONE COMPLETED:")
        print(f"  - Original strand IDs: {unique_groups.tolist()}")
        print(f"  - New strand IDs: {new_group_ids_list}")
        print(f"  - Strands cloned: {strands_cloned}")
        print(f"  - Points cloned: {points_cloned}")
        print(f"  - Points per strand: {[int(group_counts[i].item()) for i in range(len(group_counts))]}")
        
        # 返回统计信息
        return {
            "strands_cloned": strands_cloned,
            "points_cloned": points_cloned,
            "new_group_ids": new_group_ids_list,
            "original_group_ids": unique_groups.tolist(),
            "points_per_strand": [int(group_counts[i].item()) for i in range(len(group_counts))]
        }

    def smart_strand_clone(self, grads, grad_threshold, scene_extent, iteration, strand_clone_threshold=0.3):
        """
        智能strand克隆 - 只负责整条strand的克隆
        单点克隆由原有的 densify_and_clone 负责
        
        Args:
            grads: viewspace gradients
            grad_threshold: gradient threshold
            scene_extent: scene extent  
            iteration: current iteration
            strand_clone_threshold: 当strand上需要clone的点超过此比例时，clone整条strand
            
        Returns:
            clone_stats: 克隆统计信息
        """
        
        hair_mask = self._group_id >= 0
        if not torch.any(hair_mask):
            return {"strands_cloned": 0, "points_cloned": 0, "new_group_ids": [], "analysis": "no_hair"}
        
        # 计算需要clone的点
        boosted_grads = grads.clone()
        selected_pts_mask = torch.where(torch.norm(boosted_grads, dim=-1) >= grad_threshold, True, False)
        
        body_threshold = self.percent_dense * scene_extent
        hair_threshold = body_threshold
        
        # 分裂条件（小高斯需要clone）
        split_mask = torch.where(
            hair_mask,
            self.get_scaling.max(dim=1).values <= hair_threshold,
            self.get_scaling.max(dim=1).values <= body_threshold
        )
        
        selected_pts_mask = torch.logical_and(selected_pts_mask, split_mask)
        
        # 快速分析strand-level需求（完全张量化）
        hair_group_ids = self._group_id[hair_mask]
        unique_groups = torch.unique(hair_group_ids[hair_group_ids >= 0])
        
        if len(unique_groups) == 0:
            return {"strands_cloned": 0, "points_cloned": 0, "new_group_ids": [], "analysis": "no_valid_strands"}
        
        # 张量化计算每个strand的clone需求（内存高效版本）
        n_groups = len(unique_groups)
        
        # 避免broadcasting，使用更高效的方法
        # 1. 创建从group_id到索引的映射
        group_to_idx = torch.full((torch.max(self._group_id).item() + 1,), -1, dtype=torch.long, device="cuda")
        group_to_idx[unique_groups] = torch.arange(len(unique_groups), device="cuda")
        
        # 2. 计算每个group的总点数
        group_total_counts = torch.zeros(n_groups, dtype=torch.long, device="cuda")
        valid_hair_mask = torch.isin(self._group_id, unique_groups)
        valid_group_indices = group_to_idx[self._group_id[valid_hair_mask]]
        group_total_counts.scatter_add_(0, valid_group_indices, torch.ones_like(valid_group_indices))
        
        # 3. 计算每个group中需要clone的点数
        group_clone_counts = torch.zeros(n_groups, dtype=torch.long, device="cuda")
        need_clone_and_valid = selected_pts_mask & valid_hair_mask
        if need_clone_and_valid.any():
            clone_group_indices = group_to_idx[self._group_id[need_clone_and_valid]]
            group_clone_counts.scatter_add_(0, clone_group_indices, torch.ones_like(clone_group_indices))
        
        # 计算clone ratio
        clone_ratios = group_clone_counts.float() / group_total_counts.float().clamp(min=1.0)
        
        # 只选择超过阈值的strand进行整体clone
        strand_clone_mask = clone_ratios >= strand_clone_threshold
        strands_to_clone = unique_groups[strand_clone_mask]
        
        # 执行克隆操作
        stats = {"strands_cloned": 0, "points_cloned": 0, "new_group_ids": [], "analysis": "completed"}
        
        # 使用高效的vectorized方法clone整条strand
        if len(strands_to_clone) > 0:
            clone_result = self.clone_entire_strands_vectorized(strands_to_clone, iteration)
            stats.update(clone_result)
            stats["analysis"] = "cloned"
        
        # 详细的分析报告（每100次iteration报告一次，避免过于频繁）
        # if iteration % 100 == 0:
        #     # 计算统计信息
        #     needs_enhancement = (clone_ratios > 0).sum().item()
        #     fully_cloned = strand_clone_mask.sum().item()
        #     partially_need = needs_enhancement - fully_cloned
            
        #     print(f"\n[ITER {iteration}] ===== SMART STRAND CLONE ANALYSIS =====")
        #     print(f"Total hair strands: {len(unique_groups)}")
        #     print(f"Strands needing enhancement: {needs_enhancement}")
        #     print(f"Strands fully cloned: {fully_cloned} (points: {stats['points_cloned']})")
        #     print(f"Strands partially enhanced: {partially_need} (handled by standard densify)")
        #     print(f"Clone threshold: {strand_clone_threshold:.1%}")
            
        #     if len(strands_to_clone) > 0:
        #         print(f"Cloned strand details:")
        #         for i, strand_id in enumerate(strands_to_clone):
        #             total_points = group_total_counts[unique_groups == strand_id].item()
        #             clone_points = group_clone_counts[unique_groups == strand_id].item()
        #             ratio = clone_ratios[unique_groups == strand_id].item()
        #             print(f"  - Strand {strand_id}: {clone_points}/{total_points} points need clone ({ratio:.1%})")
            
        #     # 新strand数量统计
        #     if stats["new_group_ids"]:
        #         print(f"Generated {len(stats['new_group_ids'])} new strand IDs: {stats['new_group_ids']}")
            
        #     print(f"=========================================\n")
        
        return stats

    def adaptive_clone_direction(self, iteration, gradient_direction, radial_direction):
        if iteration < 2500:
            # 早期：纯径向外扩，确保结构稳定
            return radial_direction
        elif iteration < 7000:
            # 中期：以径向为主，逐渐引入梯度信息
            weight = min((iteration - 2500) / 5000, 0)
            return (1 - weight) * radial_direction + weight * gradient_direction


    def smart_clone_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """
        专门为smart_strand_clone设计的后处理函数
        与densification_postfix的主要区别是不重置梯度累积
        """
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 只为新添加的点设置临时半径
        if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
            
        # 扩展现有的统计数组以匹配新的大小，但保持现有值不变
        n_new = len(new_xyz)
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros((n_new, 1), device="cuda")
        ])
        self.denom = torch.cat([
            self.denom,
            torch.zeros((n_new, 1), device="cuda")
        ])
        self.max_radii2D = torch.cat([
            self.max_radii2D,
            torch.zeros(n_new, device="cuda")
        ])

    def get_strand_statistics(self):
        """
        获取详细的strand统计信息
        
        Returns:
            dict: 包含各种strand统计信息的字典
        """
        # 获取头发mask
        hair_mask = self._group_id >= 0
        body_mask = self._group_id == -1
        
        # 基本统计
        total_points = len(self._xyz)
        hair_points = hair_mask.sum().item()
        body_points = body_mask.sum().item()
        
        # Strand统计
        hair_group_ids = self._group_id[hair_mask]
        if len(hair_group_ids) > 0:
            unique_hair_groups = torch.unique(hair_group_ids)
            n_hair_strands = len(unique_hair_groups)
            
            # 计算每个strand的点数
            strand_point_counts = []
            for group_id in unique_hair_groups:
                count = (hair_group_ids == group_id).sum().item()
                strand_point_counts.append(count)
            
            avg_points_per_strand = sum(strand_point_counts) / len(strand_point_counts)
            min_points_per_strand = min(strand_point_counts)
            max_points_per_strand = max(strand_point_counts)
            
            # Group ID范围
            min_group_id = unique_hair_groups.min().item()
            max_group_id = unique_hair_groups.max().item()
        else:
            n_hair_strands = 0
            avg_points_per_strand = 0
            min_points_per_strand = 0
            max_points_per_strand = 0
            min_group_id = None
            max_group_id = None
        
        stats = {
            "total_points": total_points,
            "hair_points": hair_points,
            "body_points": body_points,
            "n_hair_strands": n_hair_strands,
            "avg_points_per_strand": avg_points_per_strand,
            "min_points_per_strand": min_points_per_strand,
            "max_points_per_strand": max_points_per_strand,
            "hair_group_id_range": (min_group_id, max_group_id),
            "hair_density": hair_points / total_points if total_points > 0 else 0
        }
        
        return stats
     
    def print_strand_statistics(self, iteration=None):
        """
        打印详细的strand统计信息
        
        Args:
            iteration: 当前迭代次数（可选）
        """
        stats = self.get_strand_statistics()
        
        if iteration is not None:
            print(f"\n[ITER {iteration}] ===== STRAND STATISTICS =====")
        else:
            print(f"\n===== STRAND STATISTICS =====")
        
        print(f"Total gaussians: {stats['total_points']}")
        print(f"Hair gaussians: {stats['hair_points']} ({stats['hair_density']:.1%})")
        print(f"Body gaussians: {stats['body_points']}")
        print(f"Hair strands: {stats['n_hair_strands']}")
        
        if stats['n_hair_strands'] > 0:
            print(f"Points per strand: avg={stats['avg_points_per_strand']:.1f}, "
                  f"min={stats['min_points_per_strand']}, max={stats['max_points_per_strand']}")
            print(f"Hair group_id range: {stats['hair_group_id_range'][0]} - {stats['hair_group_id_range'][1]}")
        
        print(f"==============================\n")
        
        return stats
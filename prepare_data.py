#!/usr/bin/env python3
"""
Gaussian4Hair Data Preparation Script

Enhanced version with integrated point cloud alignment and transformation tools.
Supports both automatic alignment and manual fine-tuning.

Usage:
    python prepare_data.py --colmap_path <path> --hair_data <path> --output_dir <path>
    python prepare_data.py --config <config.json>  # Use configuration file
"""

import os
import sys
import argparse
import numpy as np
import struct
import json
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import shutil
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile is required. Install with: pip install plyfile")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Warning: OpenCV not found. Some visualization features will be disabled.")
    cv2 = None


@dataclass
class TransformParams:
    """Transformation parameters for hair alignment"""
    scale_factor: float = 1.0
    translation: List[float] = None
    rotation_matrix: List[List[float]] = None
    
    # Manual rotation adjustments (in degrees)
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    
    # Additional fine-tuning offset
    fine_translation: List[float] = None
    
    def __post_init__(self):
        if self.translation is None:
            self.translation = [0.0, 0.0, 0.0]
        if self.rotation_matrix is None:
            self.rotation_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if self.fine_translation is None:
            self.fine_translation = [0.0, 0.0, 0.0]

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class HairAlignmentTool:
    """Advanced hair alignment tool with Procrustes analysis and manual adjustments"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('HairAlignment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def get_rotation_matrix(self, axis: str, angle_degrees: float) -> np.ndarray:
        """Generate rotation matrix around specified axis"""
        angle_rad = np.radians(angle_degrees)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])
        elif axis == 'y':
            R = np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
        else:  # z
            R = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
        
        return R

    def procrustes_analysis(self, source_points: np.ndarray, target_points: np.ndarray,
                          sample_size: int = 10000) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Perform Procrustes analysis for initial alignment
        
        Returns:
            scale_factor, translation, rotation_matrix
        """
        self.logger.info("Performing Procrustes analysis for initial alignment...")
        
        # Sample points for efficiency
        np.random.seed(42)  # Reproducible results
        
        if len(source_points) > sample_size:
            step = len(source_points) // sample_size
            source_sample = source_points[::step][:sample_size]
        else:
            source_sample = source_points.copy()
        
        if len(target_points) > sample_size:
            step = len(target_points) // sample_size
            target_sample = target_points[::step][:sample_size]
        else:
            target_sample = target_points.copy()
        
        # Center the point clouds
        A_centroid = np.mean(source_sample, axis=0)
        B_centroid = np.mean(target_sample, axis=0)
        
        A_centered = source_sample - A_centroid
        B_centered = target_sample - B_centroid
        
        # Calculate scale factors
        A_scale = np.sqrt(np.sum(A_centered**2) / len(A_centered))
        B_scale = np.sqrt(np.sum(B_centered**2) / len(B_centered))
        
        # Normalize
        A_normalized = A_centered / A_scale
        B_normalized = B_centered / B_scale
        
        # Calculate cross-covariance matrix
        H = np.dot(A_normalized.T, B_normalized)
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Calculate scale factor and translation
        scale_factor = B_scale / A_scale
        translation = B_centroid - np.dot(A_centroid * scale_factor, R.T)
        
        # Verify alignment quality
        transformed_source = np.dot(source_sample * scale_factor, R.T) + translation
        error = np.mean(np.linalg.norm(transformed_source - target_sample, axis=1))
        
        self.logger.info(f"Procrustes analysis complete. Average error: {error:.6f}")
        self.logger.info(f"Scale factor: {scale_factor:.6f}")
        self.logger.info(f"Translation: {translation}")
        
        return scale_factor, translation, R

    def apply_transform(self, points: np.ndarray, params: TransformParams) -> np.ndarray:
        """Apply transformation parameters to point cloud"""
        # Convert rotation matrix from list to numpy array
        R_base = np.array(params.rotation_matrix)
        
        # Apply base transformation
        transformed = np.dot(points * params.scale_factor, R_base.T) + np.array(params.translation)
        
        # Apply manual rotation adjustments around center
        if any([params.rotation_x, params.rotation_y, params.rotation_z]):
            center = np.mean(transformed, axis=0)
            centered = transformed - center
            
            # Apply rotations in order: X, Y, Z
            if params.rotation_x != 0:
                R_x = self.get_rotation_matrix('x', params.rotation_x)
                centered = np.dot(centered, R_x.T)
            
            if params.rotation_y != 0:
                R_y = self.get_rotation_matrix('y', params.rotation_y)
                centered = np.dot(centered, R_y.T)
            
            if params.rotation_z != 0:
                R_z = self.get_rotation_matrix('z', params.rotation_z)
                centered = np.dot(centered, R_z.T)
            
            transformed = centered + center
        
        # Apply fine translation adjustment
        if any(params.fine_translation):
            transformed += np.array(params.fine_translation)
        
        return transformed

    def auto_align(self, source_points: np.ndarray, target_points: np.ndarray,
                   base_params: Optional[TransformParams] = None) -> TransformParams:
        """Automatically calculate alignment parameters"""
        if base_params is None:
            # Perform Procrustes analysis
            scale, translation, rotation = self.procrustes_analysis(source_points, target_points)
            
            params = TransformParams(
                scale_factor=scale,
                translation=translation.tolist(),
                rotation_matrix=rotation.tolist()
            )
        else:
            # Use provided base parameters
            params = base_params
        
        return params

    def save_transform_config(self, params: TransformParams, config_path: Path):
        """Save transformation parameters to config file"""
        config = {
            'transform_params': params.to_dict(),
            'created_at': datetime.now().isoformat(),
            'description': 'Hair alignment transformation parameters'
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Transform config saved to: {config_path}")

    def load_transform_config(self, config_path: Path) -> TransformParams:
        """Load transformation parameters from config file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        params = TransformParams.from_dict(config['transform_params'])
        self.logger.info(f"Transform config loaded from: {config_path}")
        
        return params


class DataProcessor:
    """Enhanced data processor with integrated alignment tools"""
    
    def __init__(self, colmap_path: str, hair_data_path: str, output_dir: str):
        self.colmap_path = Path(colmap_path)
        self.hair_data_path = Path(hair_data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize alignment tool
        self.alignment_tool = HairAlignmentTool(self.output_dir)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Validate inputs
        self._validate_inputs()
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('DataProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
        
    def _validate_inputs(self):
        """Validate input paths and data"""
        if not self.colmap_path.exists():
            raise FileNotFoundError(f"COLMAP path not found: {self.colmap_path}")
            
        if not self.hair_data_path.exists():
            raise FileNotFoundError(f"Hair data path not found: {self.hair_data_path}")
        
        self.logger.info(f"✓ COLMAP path validated: {self.colmap_path}")
        self.logger.info(f"✓ Hair data path validated: {self.hair_data_path}")
        self.logger.info(f"✓ Output directory: {self.output_dir}")

    def load_hair_data(self, min_strand_length: int = 15) -> Tuple[List[int], np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Load hair strand data with segment information
        
        Returns:
            segments: List of strand lengths
            all_points: All points as numpy array
            strands: List of individual strand arrays
            tangents: List of tangent arrays for each strand
        """
        file_ext = self.hair_data_path.suffix.lower()
        
        if file_ext == '.hair':
            return self._load_hair_format(min_strand_length)
        elif file_ext == '.ply':
            return self._load_ply_hair_format(min_strand_length)
        else:
            raise ValueError(f"Unsupported hair data format: {file_ext}")

    def _load_hair_format(self, min_strand_length: int) -> Tuple[List[int], np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Load MonoHair .hair format with full structure preservation"""
        self.logger.info(f"Loading .hair format from {self.hair_data_path}")
        
        with open(self.hair_data_path, 'rb') as f:
            # Read header
            num_strand = struct.unpack('I', f.read(4))[0]
            point_count = struct.unpack('I', f.read(4))[0]
            
            # Read segments
            segments = struct.unpack('H' * num_strand, f.read(2 * num_strand))
            segments = list(segments)
            
            # Read points
            num_points = sum(segments)
            points_data = struct.unpack('f' * num_points * 3, f.read(4 * num_points * 3))
            points = np.array(points_data).reshape(-1, 3)

        # Process strands
        valid_segments = []
        valid_strands = []
        valid_tangents = []
        
        beg = 0
        for seg in segments:
            end = beg + seg
            strand = points[beg:end]
            
            if len(strand) >= min_strand_length:
                valid_segments.append(seg)
                valid_strands.append(strand.copy())
                
                # Compute tangents
                if len(strand) > 1:
                    diffs = np.zeros_like(strand)
                    diffs[:-1] = strand[1:] - strand[:-1]
                    diffs[-1] = strand[-1] - strand[-2]
                    
                    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
                    diffs = diffs / (norms + 1e-8)
                else:
                    diffs = np.array([[0.0, 0.0, 1.0]])
                
                valid_tangents.append(diffs)
            
            beg += seg

        # Reconstruct point array from valid strands
        all_valid_points = np.vstack(valid_strands) if valid_strands else np.empty((0, 3))

        self.logger.info(f"Loaded {len(valid_strands)} valid strands (filtered from {num_strand})")
        if valid_strands:
            strand_lengths = [len(s) for s in valid_strands]
            self.logger.info(f"Strand lengths: min={min(strand_lengths)}, max={max(strand_lengths)}, avg={np.mean(strand_lengths):.1f}")
        
        return valid_segments, all_valid_points, valid_strands, valid_tangents

    def _load_ply_hair_format(self, min_strand_length: int) -> Tuple[List[int], np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Load PLY format hair data"""
        self.logger.info(f"Loading PLY hair format from {self.hair_data_path}")
        
        ply_data = PlyData.read(str(self.hair_data_path))
        vertices = ply_data['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        
        total_points = len(points)
        
        # Try to determine strand configuration
        possible_configs = [
            (1900, 100),  # Common MonoHair config
            (total_points // 100, 100),
            (total_points // 50, 50),
            (total_points // 200, 200),
        ]
        
        for n_strands, n_points_per_strand in possible_configs:
            if total_points == n_strands * n_points_per_strand:
                strands_array = points.reshape(n_strands, n_points_per_strand, 3)
                
                # Filter by length
                valid_indices = [i for i in range(n_strands) if n_points_per_strand >= min_strand_length]
                
                if not valid_indices:
                    continue
                
                segments = [n_points_per_strand] * len(valid_indices)
                strands = [strands_array[i] for i in valid_indices]
                
                # Compute tangents
                tangents = []
                for strand in strands:
                    diffs = np.diff(strand, axis=0)
                    last_diff = diffs[-1:] if len(diffs) > 0 else np.array([[0.0, 0.0, 1.0]])
                    tangents_strand = np.vstack([diffs, last_diff])
                    norms = np.linalg.norm(tangents_strand, axis=1, keepdims=True)
                    tangents_strand = np.divide(tangents_strand, norms, where=norms>0)
                    tangents.append(tangents_strand)
                
                all_points = np.vstack(strands)
                
                self.logger.info(f"Loaded {len(strands)} strands x {n_points_per_strand} points")
                return segments, all_points, strands, tangents
        
        raise ValueError(f"Cannot determine strand configuration for {total_points} points")

    def load_target_points(self, target_path: Optional[Path] = None, max_points: int = 50000) -> np.ndarray:
        """Load target point cloud for alignment"""
        if target_path is None:
            # Try to find COLMAP point cloud
            sparse_dir = self.colmap_path / "sparse" / "0"
            if not sparse_dir.exists():
                sparse_dir = self.colmap_path / "sparse"
            
            points3d_path = sparse_dir / "points3D.txt"
            if points3d_path.exists():
                return self._load_colmap_points(points3d_path, max_points)
            else:
                # Look for PLY files
                ply_files = list(self.colmap_path.rglob("*.ply"))
                if ply_files:
                    return self._load_ply_points(ply_files[0], max_points)
        else:
            if target_path.suffix.lower() == '.ply':
                return self._load_ply_points(target_path, max_points)
            elif target_path.suffix.lower() == '.txt':
                return self._load_colmap_points(target_path, max_points)
        
        raise FileNotFoundError("No suitable target point cloud found")

    def _load_colmap_points(self, points3d_path: Path, max_points: int) -> np.ndarray:
        """Load COLMAP points3D.txt file"""
        points = []
        with open(points3d_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    # points3D.txt format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] 
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
                    if len(points) >= max_points:
                        break
        
        self.logger.info(f"Loaded {len(points)} COLMAP 3D points")
        return np.array(points)

    def _load_ply_points(self, ply_path: Path, max_points: int) -> np.ndarray:
        """Load PLY point cloud"""
        points = []
        with open(ply_path, 'r') as f:
            # Skip header
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == 'end_header':
                        in_header = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(points) >= max_points:
                        break
        
        self.logger.info(f"Loaded {len(points)} points from PLY file")
        return np.array(points)

    def save_hair_data(self, segments: List[int], strands: List[np.ndarray], 
                      output_path: Path, downsample_factor: int = 1):
        """Save hair data in .hair format with optional downsampling"""
        if downsample_factor > 1:
            # Downsample strands
            selected_strands = strands[::downsample_factor]
            selected_segments = segments[::downsample_factor]
            total_points = sum(selected_segments)
            self.logger.info(f"Downsampling: {len(strands)} -> {len(selected_strands)} strands")
        else:
            selected_strands = strands
            selected_segments = segments
            total_points = sum(selected_segments)
        
        with open(output_path, 'wb') as f:
            # Write header
            num_strands = len(selected_segments)
            f.write(struct.pack('I', num_strands))
            f.write(struct.pack('I', total_points))
            
            # Write segments
            f.write(struct.pack('H' * num_strands, *selected_segments))
            
            # Write points
            all_points = np.vstack(selected_strands)
            points_flat = all_points.flatten()
            f.write(struct.pack('f' * len(points_flat), *points_flat))
        
        self.logger.info(f"Saved {num_strands} strands to {output_path}")

    def save_hair_as_ply(self, strands: List[np.ndarray], output_path: Path, 
                        sample_rate: int = 10, color_per_strand: bool = True):
        """Save hair data as colored PLY for visualization"""
        all_points = []
        all_colors = []
        
        if color_per_strand:
            # Generate distinct colors for each strand
            np.random.seed(42)
            for i, strand in enumerate(strands):
                # Sample points
                sampled_points = strand[::sample_rate]
                all_points.append(sampled_points)
                
                # Generate color for this strand
                color = np.random.randint(50, 255, 3)
                strand_colors = np.tile(color, (len(sampled_points), 1))
                all_colors.append(strand_colors)
        else:
            # Use single color for all hair
            for strand in strands:
                sampled_points = strand[::sample_rate]
                all_points.append(sampled_points)
                strand_colors = np.tile([139, 69, 19], (len(sampled_points), 1))  # Brown
                all_colors.append(strand_colors)
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        # Create PLY data
        vertex = np.zeros(len(points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]
        
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(str(output_path))
        
        self.logger.info(f"Saved {len(points)} hair points to PLY: {output_path}")

    def analyze_hair_data(self, points: np.ndarray, strands: List[np.ndarray]) -> Dict[str, Any]:
        """Comprehensive hair data analysis"""
        if len(points) == 0:
            return {}
        
        strand_lengths = [len(s) for s in strands]
        
        analysis = {
            'num_strands': len(strands),
            'total_points': len(points),
            'strand_lengths': {
                'min': min(strand_lengths) if strand_lengths else 0,
                'max': max(strand_lengths) if strand_lengths else 0,
                'mean': np.mean(strand_lengths) if strand_lengths else 0,
                'std': np.std(strand_lengths) if strand_lengths else 0
            },
            'spatial_extent': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist(),
                'center': points.mean(axis=0).tolist(),
                'std': points.std(axis=0).tolist(),
                'bbox_size': (points.max(axis=0) - points.min(axis=0)).tolist()
            }
        }
        
        return analysis

    def process_alignment(self, segments: List[int], hair_points: np.ndarray, strands: List[np.ndarray],
                         target_points: Optional[np.ndarray] = None,
                         transform_config: Optional[Path] = None,
                         auto_align: bool = True) -> Tuple[List[np.ndarray], TransformParams]:
        """
        Process hair alignment with various options
        
        Args:
            segments: Strand segment lengths
            hair_points: All hair points
            strands: Individual strand arrays
            target_points: Target point cloud for alignment
            transform_config: Path to existing transform config
            auto_align: Whether to perform automatic alignment
            
        Returns:
            aligned_strands, transform_params
        """
        if transform_config and transform_config.exists():
            # Load existing transform parameters
            params = self.alignment_tool.load_transform_config(transform_config)
            self.logger.info("Using existing transform configuration")
        elif auto_align and target_points is not None:
            # Perform automatic alignment
            params = self.alignment_tool.auto_align(hair_points, target_points)
            self.logger.info("Computed automatic alignment parameters")
        else:
            # Use identity transform
            params = TransformParams()
            self.logger.info("Using identity transform (no alignment)")
        
        # Apply transformation to each strand
        aligned_strands = []
        for strand in strands:
            aligned_strand = self.alignment_tool.apply_transform(strand, params)
            aligned_strands.append(aligned_strand)
        
        # Save transform parameters
        config_path = self.output_dir / "transform_params.json"
        self.alignment_tool.save_transform_config(params, config_path)
        
        return aligned_strands, params

    def process_all(self, min_strand_length: int = 15, 
                   target_cloud_path: Optional[str] = None,
                   transform_config_path: Optional[str] = None,
                   auto_align: bool = True,
                   save_visualization: bool = True,
                   downsample_factor: int = 1) -> Dict[str, str]:
        """Complete enhanced data processing pipeline"""
        self.logger.info("Starting enhanced Gaussian4Hair data processing...")
        
        # Step 1: Load hair data
        self.logger.info("\n1. Loading hair data...")
        segments, hair_points, strands, tangents = self.load_hair_data(min_strand_length)
        
        if not strands:
            raise ValueError("No valid hair strands found after filtering")
        
        # Step 2: Analyze original hair data
        self.logger.info("\n2. Analyzing hair data...")
        original_analysis = self.analyze_hair_data(hair_points, strands)
        
        # Step 3: Load target points if needed
        target_points = None
        if auto_align or target_cloud_path:
            self.logger.info("\n3. Loading target point cloud...")
            target_path = Path(target_cloud_path) if target_cloud_path else None
            try:
                target_points = self.load_target_points(target_path)
                target_analysis = self.analyze_hair_data(target_points, [target_points])
            except Exception as e:
                self.logger.warning(f"Could not load target points: {e}")
                target_points = None
                auto_align = False
        
        # Step 4: Process alignment
        self.logger.info("\n4. Processing hair alignment...")
        transform_config = Path(transform_config_path) if transform_config_path else None
        aligned_strands, transform_params = self.process_alignment(
            segments, hair_points, strands, target_points, transform_config, auto_align)
        
        # Step 5: Analyze aligned data
        self.logger.info("\n5. Analyzing aligned data...")
        aligned_points = np.vstack(aligned_strands)
        aligned_analysis = self.analyze_hair_data(aligned_points, aligned_strands)
        
        # Step 6: Save results
        self.logger.info("\n6. Saving processed data...")
        
        # Save aligned hair data
        aligned_hair_path = self.output_dir / "aligned_hair.hair"
        self.save_hair_data(segments, aligned_strands, aligned_hair_path)
        
        # Save downsampled version if requested
        downsampled_path = None
        if downsample_factor > 1:
            downsampled_path = self.output_dir / f"aligned_hair_downsampled_{downsample_factor}x.hair"
            self.save_hair_data(segments, aligned_strands, downsampled_path, downsample_factor)
        
        # Save visualization files
        visualization_paths = {}
        if save_visualization:
            # Original hair PLY
            original_ply = self.output_dir / "original_hair.ply"
            self.save_hair_as_ply(strands, original_ply)
            visualization_paths['original_ply'] = str(original_ply)
            
            # Aligned hair PLY
            aligned_ply = self.output_dir / "aligned_hair.ply"
            self.save_hair_as_ply(aligned_strands, aligned_ply)
            visualization_paths['aligned_ply'] = str(aligned_ply)
        
        # Step 7: Save analysis and setup output structure
        self.logger.info("\n7. Finalizing output structure...")
        
        # Save comprehensive analysis
        analysis = {
            'original': original_analysis,
            'aligned': aligned_analysis,
            'transform_params': transform_params.to_dict(),
            'processing_info': {
                'min_strand_length': min_strand_length,
                'auto_align': auto_align,
                'downsample_factor': downsample_factor,
                'processed_at': datetime.now().isoformat()
            }
        }
        
        if target_points is not None:
            analysis['target'] = target_analysis
        
        analysis_path = self.output_dir / "processing_report.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Copy COLMAP data
        colmap_output = self.output_dir / "colmap"
        if colmap_output.exists():
            shutil.rmtree(colmap_output)
        shutil.copytree(self.colmap_path, colmap_output)
        
        # Prepare results summary
        results = {
            'aligned_hair': str(aligned_hair_path),
            'colmap_data': str(colmap_output),
            'analysis': str(analysis_path),
            'transform_config': str(self.output_dir / "transform_params.json")
        }
        
        if downsampled_path:
            results['downsampled_hair'] = str(downsampled_path)
        
        results.update(visualization_paths)
        
        # Log completion
        self.logger.info("\n✓ Enhanced data processing completed!")
        self.logger.info("\nOutput files:")
        for key, path in results.items():
            self.logger.info(f"  {key}: {path}")
        
        self.logger.info("\nNext steps:")
        self.logger.info("1. Review alignment results in visualization files")
        self.logger.info("2. Adjust transform_params.json if needed and rerun")
        self.logger.info("3. Use for Gaussian4Hair training:")
        self.logger.info(f"   python train.py -s {colmap_output} --hair_data {aligned_hair_path} --hair_init")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Gaussian4Hair Data Preparation Tool")
    parser.add_argument("--colmap_path", type=str, required=True,
                        help="Path to COLMAP reconstruction directory")
    parser.add_argument("--hair_data", type=str, required=True,
                        help="Path to hair strand data (.hair or .ply format)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration JSON file")
    parser.add_argument("--min_strand_length", type=int, default=15,
                        help="Minimum points per strand to keep (default: 15)")
    parser.add_argument("--target_cloud", type=str, default=None,
                        help="Path to target point cloud for alignment (.ply or .txt)")
    parser.add_argument("--transform_config", type=str, default=None,
                        help="Path to existing transform configuration (.json)")
    parser.add_argument("--no_auto_align", action="store_true",
                        help="Disable automatic alignment (use identity transform)")
    parser.add_argument("--no_visualization", action="store_true",
                        help="Skip saving visualization files")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor for creating lighter version (default: 1)")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Override with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    try:
        processor = DataProcessor(args.colmap_path, args.hair_data, args.output_dir)
        results = processor.process_all(
            min_strand_length=args.min_strand_length,
            target_cloud_path=args.target_cloud,
            transform_config_path=args.transform_config,
            auto_align=not args.no_auto_align,
            save_visualization=not args.no_visualization,
            downsample_factor=args.downsample
        )
        
        print(f"\n🎉 Success! Enhanced processed data ready in: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
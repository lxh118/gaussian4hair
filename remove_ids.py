#!/usr/bin/env python3
"""
Remove Group and Strand IDs from PLY files

This tool removes group_id and strand_id attributes from Gaussian PLY files
to improve compatibility with standard Gaussian Splatting viewers.

Usage:
    python remove_ids.py <input.ply> <output.ply>
    python remove_ids.py --input <input.ply> --output <output.ply>
"""

import argparse
import sys
import numpy as np
from pathlib import Path

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile is required. Install with: pip install plyfile")
    sys.exit(1)


def remove_group_and_strand_ids(input_path, output_path):
    """
    从PLY文件中移除group_id和strand_id属性
    
    参数:
        input_path: 输入的PLY文件路径
        output_path: 输出的PLY文件路径（不包含group_id和strand_id）
    """
    try:
        # 检查输入文件是否存在
        if not Path(input_path).exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 读取原始PLY文件
        print(f"读取PLY文件: {input_path}")
        plydata = PlyData.read(input_path)
        
        # 获取顶点数据
        vertices = plydata['vertex']
        
        # 获取所有属性名称
        original_properties = vertices.data.dtype.names
        print(f"原始属性: {original_properties}")
        
        # 确定要保留的属性（排除group_id和strand_id）
        properties_to_keep = [prop for prop in original_properties 
                             if prop not in ('group_id', 'strand_id')]
        
        # 检查是否有需要移除的属性
        removed_props = [prop for prop in original_properties 
                        if prop in ('group_id', 'strand_id')]
        
        if not removed_props:
            print("警告: 文件中未找到group_id或strand_id属性")
        else:
            print(f"将移除属性: {removed_props}")
        
        print(f"保留属性: {properties_to_keep}")
        
        # 创建新的顶点数据
        new_vertices = np.zeros(vertices.count, dtype=[(prop, vertices.data.dtype[prop]) 
                                                     for prop in properties_to_keep])
        
        # 复制保留的属性数据
        for prop in properties_to_keep:
            new_vertices[prop] = vertices[prop]
        
        # 创建新的PlyElement
        new_vertex_element = PlyElement.describe(new_vertices, 'vertex')
        
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存新的PLY文件
        PlyData([new_vertex_element]).write(output_path)
        
        print(f"✅ 成功保存 {vertices.count} 个高斯点到: {output_path}")
        if removed_props:
            print(f"已移除属性: {', '.join(removed_props)}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="从PLY文件中移除group_id和strand_id属性")
    parser.add_argument('input', nargs='?', help='输入PLY文件路径')
    parser.add_argument('output', nargs='?', help='输出PLY文件路径')
    parser.add_argument('--input', '-i', dest='input_file', help='输入PLY文件路径')
    parser.add_argument('--output', '-o', dest='output_file', help='输出PLY文件路径')
    
    args = parser.parse_args()
    
    # 确定输入输出文件路径
    input_path = args.input or args.input_file
    output_path = args.output or args.output_file
    
    if not input_path or not output_path:
        parser.print_help()
        print("\n示例用法:")
        print("  python remove_ids.py input.ply output.ply")
        print("  python remove_ids.py --input input.ply --output output.ply")
        sys.exit(1)
    
    remove_group_and_strand_ids(input_path, output_path)


if __name__ == "__main__":
    main()
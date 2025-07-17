# Gaussian4Hair 项目结构

简洁项目结构，专注于核心功能。

## 📁 核心文件结构

```
gaussian4hair/
├── README.md                    # 项目主文档
├── environment.yml              # 环境配置
├── LICENSE.md                  # 许可证
│
├── train.py                    # 训练脚本 (主要)
├── prepare_data.py             # 数据准备工具 (重要)
├── run.sh                      # 快速训练脚本
├── remove_ids.py               # PLY文件ID属性清理工具
│
├── arguments/                  # 参数配置
│   └── __init__.py            # 头发参数、模型参数等
│
├── scene/                     # 场景管理
│   ├── __init__.py           # 场景类
│   ├── gaussian_model.py     # 高斯模型 (头发增强)
│   ├── cameras.py            # 相机工具
│   ├── colmap_loader.py      # COLMAP加载器
│   └── dataset_readers.py    # 数据读取器
│
├── utils/                     # 工具函数
│   ├── camera_utils.py       # 相机工具
│   ├── general_utils.py      # 通用工具
│   ├── graphics_utils.py     # 图形工具
│   ├── image_utils.py        # 图像工具
│   └── loss_utils.py         # 损失函数
│
├── gaussian_renderer/         # 渲染器
│   ├── __init__.py
│   └── network_gui.py
│
├── submodules/               # 子模块依赖
│   ├── diff-gaussian-rasterization/
│   ├── fused-ssim/
│   └── simple-knn/
│
├── configs/                  # 配置文件
│   └── alignment_config.json # 对齐配置模板
│
├── docs/                     # 文档
  └── hair_alignment_guide.md # 头发对齐指南

```

## 🚀 主要工具

### 1. **数据处理** - `prepare_data.py`
完整的头发数据预处理工具，包含：
- 自动Procrustes对齐
- 手动精调支持
- 多格式支持 (.hair/.ply)
- 可视化输出

### 2. **训练** - `train.py`
头发感知的3D高斯训练：
- 头发专用损失函数
- 智能致密化
- 位置固定选项

### 3. **快速启动** - `run.sh`
简化的训练脚本：
- 固定7000次迭代训练
- 自动参数配置

### 4. **环境配置** - `setup.sh`
一键环境搭建：
- Conda环境创建
- 依赖安装
- 子模块编译

### 5. **兼容性工具** - `remove_ids.py`
PLY文件兼容性处理：
- 移除group_id和strand_id属性
- 提高与标准高斯查看器的兼容性
- 支持批量处理

## 📋 使用流程

1. **环境搭建**:
   ```bash
   ./setup.sh
   ```

2. **数据准备**:
   ```bash
   python prepare_data.py --colmap_path /data/colmap --hair_data /data/hair.hair --output_dir /output
   ```

3. **训练**:
   ```bash
   ./run.sh /output/colmap /output/aligned_hair.hair /models
   ```

4. **渲染**:
   ```bash
   python render.py -m /models/standard
   ```

5. **清理PLY文件** (可选):
   ```bash
   python remove_ids.py input.ply output.ply
   ```


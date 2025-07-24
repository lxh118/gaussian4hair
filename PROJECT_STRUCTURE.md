# Gaussian4Hair 项目结构

现代化项目结构，专注于头发渲染的核心功能。

## 📁 核心文件结构

```
gaussian4hair/
├── README.md                    # 项目主文档
├── environment.yml              # 环境配置 (gaussian_splatting)
├── setup.sh                     # 自动环境配置脚本
├── LICENSE.md                  # 许可证
│
├── train.py                    # 训练脚本 (主要)
├── prepare_data.py             # 数据准备工具 (重要)
├── run.sh                      # 快速训练脚本 (7000次迭代)
├── render.py                   # 渲染脚本
├── metrics.py                  # 质量评估脚本
├── remove_ids.py               # PLY文件ID属性清理工具
│
├── arguments/                  # 参数配置
│   └── __init__.py            # 头发参数、模型参数、优化参数
│
├── scene/                     # 场景管理
│   ├── __init__.py           # 场景类 (头发数据加载)
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
│   ├── loss_utils.py         # 损失函数 (头发专用)
│   └── system_utils.py       # 系统工具
│
├── gaussian_renderer/         # 渲染器
│   ├── __init__.py           # 主渲染函数
│   └── network_gui.py        # 网络GUI (可选)
│
├── submodules/               # 子模块依赖
│   ├── diff-gaussian-rasterization/  # CUDA光栅化
│   ├── fused-ssim/                  # SSIM损失
│   └── simple-knn/                  # KNN加速
│
├── configs/                  # 配置文件
│   └── alignment_config.json # 对齐配置模板
│
├── output/                   # 训练输出 (gitignore)
├── data/                     # 数据目录 (gitignore)
└── docs/                     # 文档 (可选)
    └── hair_alignment_guide.md
```

## 🚀 主要工具详解

### 1. **环境配置** - `setup.sh`
自动化环境搭建脚本：
- 检测现有`gaussian_splatting`环境
- 创建/更新conda环境
- 自动安装子模块依赖
- CUDA环境验证

```bash
./setup.sh
```

### 2. **数据处理** - `prepare_data.py`
完整的头发数据预处理工具：
- 自动Procrustes对齐
- 支持`.hair`和`.ply`格式
- 手动精调支持
- 可视化输出

```bash
python prepare_data.py --colmap_path /data/colmap --hair_data /data/hair.hair --output_dir /output
```

### 3. **快速训练** - `run.sh`
简化的训练脚本：
- 固定7000次迭代训练
- 自动头发数据路径解析
- 完整的训练-渲染-评估流程
- 支持多场景切换

```bash
./run.sh [场景名称]    # 默认: jenya2
```

### 4. **核心训练** - `train.py`
头发感知的3D高斯训练：
- 头发专用初始化 (`--hair_init`)
- 头发数据加载 (`--hair_data`)
- 智能致密化策略
- 位置固定选项

```bash
python train.py -s /data/scene -m /output/model --hair_init --hair_data /data/hair.hair
```

### 5. **渲染评估** - `render.py` & `metrics.py`
模型输出和质量评估：
- 多视角渲染
- PSNR/SSIM/LPIPS评估
- 头发质量专项指标

```bash
python render.py -m /model
python metrics.py -m /model
```

### 6. **兼容性工具** - `remove_ids.py`
PLY文件兼容性处理：
- 移除group_id和strand_id属性
- 提高与标准高斯查看器的兼容性
- 支持批量处理

```bash
python remove_ids.py input.ply output.ply
```

## 📋 完整使用流程

### 初次设置
```bash
# 1. 克隆项目
git clone https://github.com/lxh118/gaussian4hair.git
cd gaussian4hair

# 2. 环境搭建
./setup.sh
```

### 数据准备
```bash
# 3. 激活环境
conda activate gaussian_splatting

# 4. 准备数据 (可选，如果数据未对齐)
python prepare_data.py \
    --colmap_path /home/ubuntu/data/jenya2/sparse \
    --hair_data /home/ubuntu/data/jenya2/connected_strands.hair \
    --output_dir /home/ubuntu/data/jenya2_aligned
```

### 训练和评估
```bash
# 5a. 快速训练 (推荐)
./run.sh jenya2

# 5b. 或手动训练
python train.py \
    -s /home/ubuntu/data/jenya2 \
    --hair_data /home/ubuntu/data/jenya2/connected_strands_aligned2_downsampled.hair \
    -m ./output/jenya2_model \
    --hair_init

# 6. 渲染 (如果使用手动训练)
python render.py -m ./output/jenya2_model

# 7. 评估 (如果使用手动训练)
python metrics.py -m ./output/jenya2_model
```

## 🔧 配置文件结构

### 环境配置 - `environment.yml`
```yaml
name: gaussian_splatting
dependencies:
  - python=3.10
  - pytorch>=2.0
  - pytorch-cuda=12.1
  # ... 其他依赖
```

### 参数配置 - `arguments/__init__.py`
- `ModelParams`: 模型路径、数据路径、头发数据路径
- `OptimizationParams`: 学习率、迭代次数、致密化参数
- `HairParams`: 头发专用参数（半径、高度、颜色等）

### 数据目录结构
```
/home/ubuntu/data/
├── jenya2/                  # 场景数据
│   ├── images/             # 输入图像
│   ├── sparse/             # COLMAP稀疏重建
│   ├── connected_strands_aligned2_downsampled.hair  # 头发数据
│   └── ...
├── counter/
├── room/
└── treehill/
```

## 💻 系统要求与兼容性

### 硬件要求
- **GPU**: NVIDIA GPU (8GB+ VRAM)
- **内存**: 16GB+ 系统内存
- **存储**: 10GB+ 可用空间

### 软件环境
- **操作系统**: Linux (Ubuntu 18.04+)
- **Python**: 3.10+ 
- **CUDA**: 12.1+ (兼容 11.8+)
- **PyTorch**: 2.0+

### 已验证环境
- Ubuntu 20.04 + CUDA 12.1 + PyTorch 2.5.1
- Python 3.10.12
- NVIDIA GPU (RTX 系列)

## 🔄 开发和调试

### 快速验证
```bash
# 检查环境
conda activate gaussian_splatting
python -c "import torch; print(torch.cuda.is_available())"

# 小规模测试
python train.py -s /data/scene -m /test --iterations 100 --disable_viewer
```

### 日志和输出
- 训练日志: 控制台输出
- 模型保存: `./output/[场景名]/`
- 渲染结果: `./output/[场景名]/test/renders/`
- 评估报告: `./output/[场景名]/results.json`

这个项目结构专注于实用性和易用性，通过自动化脚本最大化简化了复杂的3D头发渲染流程。


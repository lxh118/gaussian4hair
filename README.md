# Gaussian4Hair: 3D头发渲染工具

基于3D Gaussian Splatting的高质量头发渲染系统，专门针对头发几何和外观进行优化。

## ✨ 核心特性

- **头发感知初始化**: 圆柱形高斯初始化，与发丝切线对齐
- **头发专用正则化**: 透明度连续性、几何连续性、方向对齐损失
- **智能致密化**: 发丝级别的智能克隆和分裂策略
- **位置固定**: 可选的头发位置锁定，保持原始几何
- **自动对齐**: 集成的头发数据与COLMAP点云对齐工具

## 🚀 快速开始

### 1. 环境配置
```bash
git clone https://github.com/lxh118/gaussian4hair.git
cd gaussian4hair
conda env create -f environment.yml
conda activate gaussian4hair

# 安装子模块
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim  
pip install submodules/simple-knn
```

### 2. 数据准备
```bash
# 自动对齐头发数据与COLMAP重建
python prepare_data.py \
    --colmap_path /path/to/colmap \
    --hair_data /path/to/hair.hair \
    --output_dir /path/to/output
```

### 3. 训练
```bash
python train.py \
    -s /path/to/output/colmap \
    --hair_data /path/to/output/aligned_hair.hair \
    --model_path /path/to/model \
    --hair_init
```

### 4. 渲染
```bash
python render.py -m /path/to/model
```

## ⚙️ 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hair_radius` | 3e-4 | 头发高斯圆柱半径 |
| `--hair_height` | 3e-3 | 头发高斯圆柱高度 |
| `--fix_hair_positions` | True | 是否固定头发位置 |
| `--lambda_opacity` | 0.1 | 透明度连续性损失权重 |
| `--lambda_geometry` | 0.1 | 几何连续性损失权重 |
| `--iterations` | 7000 | 训练迭代次数 |
| `--densify_until_iter` | 6000 | 致密化停止迭代 |

## 📁 输出文件

训练完成后，模型目录包含：
- `point_cloud.ply` - 最终的高斯点云
- `cameras.json` - 相机参数
- `cfg_args` - 训练配置
- `checkpoints/` - 训练检查点

## 🛠️ 工具脚本

- `prepare_data.py` - 数据预处理和头发对齐
- `run.sh` - 一键训练脚本
- `render.py` - 渲染脚本
- `metrics.py` - 质量评估
- `remove_ids.py` - 移除PLY文件中的group_id和strand_id属性

## 🛠️ 高级用法

### 数据对齐精调
```bash
# 生成初始对齐
python prepare_data.py --colmap_path /data --hair_data /hair.hair --output_dir /out

# 编辑对齐参数
vim /out/transform_params.json

# 重新应用调整
python prepare_data.py --transform_config /out/transform_params.json \
    --colmap_path /data --hair_data /hair.hair --output_dir /out
```

### PLY文件兼容性处理
```bash
# 移除ID属性以提高兼容性
python remove_ids.py input.ply output.ply
```

### 批量处理
```bash
# 创建配置文件
cp configs/alignment_config.json my_config.json
# 编辑配置
python prepare_data.py --config my_config.json
```

详细的对齐工具使用方法请参考 [`docs/hair_alignment_guide.md`](docs/hair_alignment_guide.md)

## 📋 数据格式

**支持的头发数据格式：**
- `.hair` - MonoHair标准格式
- `.ply` - 点云格式（需要指定发丝结构）

**COLMAP数据要求：**
- `sparse/` - COLMAP稀疏重建结果
- `images/` - 输入图像

## 💻 系统要求

- Python 3.8+
- CUDA 11.3+
- GPU内存 > 8GB (推荐)
- PyTorch 1.12+

## 🔧 故障排除

**常见问题：**

1. **头发对齐不准确**
   - 检查 `aligned_hair.ply` 可视化结果
   - 手动调整 `transform_params.json` 中的参数

2. **训练内存不足**
   - 使用 `--downsample` 参数减少头发数据
   - 降低图像分辨率

3. **训练不收敛**
   - 确保 `--fix_hair_positions` 开启
   - 检查头发数据质量和对齐

## 🙏 致谢

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - 基础实现
- [MonoHair](https://github.com/MonoHair/MonoHair) - 头发重建
- INRIA GraphDeco团队 - 原始研究

## 📄 许可证

本项目遵循与原始3D Gaussian Splatting相同的许可证。详见 [LICENSE.md](LICENSE.md)。

## 📚 引用

```bibtex
@article{gaussian4hair2025,
  title={Gaussian4Hair: Hair-Aware 3D Gaussian Splatting for High-Fidelity Hair Rendering},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

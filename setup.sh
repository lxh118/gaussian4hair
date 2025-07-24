#!/bin/bash

# Gaussian4Hair 项目设置脚本

echo "🚀 Gaussian4Hair 项目设置"
echo "========================="

# 检查conda是否存在
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 检查现有环境
echo "📋 检查现有环境..."
if conda env list | grep -q "gaussian_splatting"; then
    echo "✅ 检测到现有 'gaussian_splatting' 环境"
    echo "⚠️  是否重新创建环境? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  移除现有环境..."
        conda env remove -n gaussian_splatting -y
        echo "📦 创建新环境..."
        conda env create -f environment.yml
    else
        echo "📦 使用现有环境..."
    fi
else
    echo "📦 创建conda环境..."
    conda env create -f environment.yml
fi

# 激活环境
echo "🔄 激活环境..."
eval "$(conda shell.bash hook)"
conda activate gaussian_splatting

# 检查CUDA可用性
echo "🔍 检查CUDA环境..."
python -c "import torch; print(f'✅ PyTorch版本: {torch.__version__}'); print(f'✅ CUDA可用: {torch.cuda.is_available()}'); print(f'✅ CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 安装子模块
echo "📋 安装子模块..."
if [ -d "submodules/diff-gaussian-rasterization" ]; then
    echo "  安装 diff-gaussian-rasterization..."
    pip install submodules/diff-gaussian-rasterization/
else
    echo "⚠️  子模块 diff-gaussian-rasterization 不存在"
fi

if [ -d "submodules/fused-ssim" ]; then
    echo "  安装 fused-ssim..."
    pip install submodules/fused-ssim/
else
    echo "⚠️  子模块 fused-ssim 不存在"
fi

if [ -d "submodules/simple-knn" ]; then
    echo "  安装 simple-knn..."
    pip install submodules/simple-knn/
else
    echo "⚠️  子模块 simple-knn 不存在"
fi

echo ""
echo "✅ 设置完成！"
echo ""
echo "📝 下一步："
echo "1. 激活环境: conda activate gaussian_splatting"
echo "2. 准备数据: python prepare_data.py --help"
echo "3. 开始训练: ./run.sh [场景名称]"
echo ""
echo "💡 环境信息："
echo "   - Python: $(python --version)"
echo "   - 工作目录: $(pwd)"
echo "   - 数据目录: /home/ubuntu/data" 
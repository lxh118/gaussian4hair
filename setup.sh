#!/bin/bash

# Gaussian4Hair 项目设置脚本

echo "🚀 Gaussian4Hair 项目设置"
echo "========================="

# 检查conda是否存在
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建conda环境
echo "📦 创建conda环境..."
if conda env list | grep -q "gaussian4hair"; then
    echo "⚠️  环境 'gaussian4hair' 已存在，是否重新创建? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n gaussian4hair -y
        conda env create -f environment.yml
    fi
else
    conda env create -f environment.yml
fi

# 激活环境
echo "🔄 激活环境..."
eval "$(conda shell.bash hook)"
conda activate gaussian4hair

# 安装子模块
echo "📋 安装子模块..."
if [ -d "submodules/diff-gaussian-rasterization" ]; then
    pip install submodules/diff-gaussian-rasterization/
else
    echo "⚠️  子模块目录不存在，请确保已正确克隆项目"
fi

if [ -d "submodules/fused-ssim" ]; then
    pip install submodules/fused-ssim/
fi

if [ -d "submodules/simple-knn" ]; then
    pip install submodules/simple-knn/
fi

echo ""
echo "✅ 设置完成！"
echo ""
echo "下一步："
echo "1. 激活环境: conda activate gaussian4hair"
echo "2. 准备数据: python prepare_data.py --help"
echo "3. 开始训练: ./run.sh" 
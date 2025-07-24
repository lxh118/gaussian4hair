#!/bin/bash

# Gaussian4Hair 训练脚本

# 高斯头发根部分布不均匀，对比下和原始点点区别，看看为什么

echo "=== Gaussian4Hair 训练流程 ==="

# 数据根目录设置
DATA_ROOT_DIR=/home/ubuntu/data
OUTPUT_ROOT_DIR=./output/gaussian4hair

# 默认场景设置
DEFAULT_SCENE="jenya2"
SCENE=${1:-$DEFAULT_SCENE}

echo "数据根目录: $DATA_ROOT_DIR"
echo "场景名称: $SCENE"
echo "输出路径: $OUTPUT_ROOT_DIR"

# 检查场景数据路径是否存在
SCENE_DATA_PATH="${DATA_ROOT_DIR}/${SCENE}"
if [ ! -d "$SCENE_DATA_PATH" ]; then
    echo "❌ 错误: 场景数据路径不存在: $SCENE_DATA_PATH"
    echo "用法: $0 [场景名称]"
    echo "可用场景请检查: $DATA_ROOT_DIR/"
    exit 1
fi

# 固定训练参数
ITERATIONS=7000
RENDER_ITERATIONS=7000

# 创建输出目录
mkdir -p $OUTPUT_ROOT_DIR

echo ""
echo "=== 1. 开始训练 (7000次迭代) ==="
CMD_TRAIN="python train.py -s ${SCENE_DATA_PATH} -m ${OUTPUT_ROOT_DIR}/${SCENE} --eval --port 7001 \
    --iterations ${ITERATIONS} --save_iterations 7000 --densify_until_iter 6000 \
    --test_iterations 1000 3000 5000 7000 --hair_init --hair_data ${SCENE_DATA_PATH}/connected_strands_aligned2_downsampled.hair"

eval $CMD_TRAIN

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "=== 2. 开始渲染 ==="
CMD_RENDER="python -W ignore::UserWarning render.py -s ${SCENE_DATA_PATH} -m ${OUTPUT_ROOT_DIR}/${SCENE} \
    --skip_train --eval --iteration ${RENDER_ITERATIONS}"

eval $CMD_RENDER

if [ $? -ne 0 ]; then
    echo "❌ 渲染失败"
    exit 1
fi

echo ""
echo "=== 3. 开始评估 ==="
CMD_METRICS="python -W ignore::UserWarning metrics.py -m ${OUTPUT_ROOT_DIR}/${SCENE}"

eval $CMD_METRICS

if [ $? -ne 0 ]; then
    echo "❌ 评估失败"
    exit 1
fi

echo ""
echo "✅ 完整流程完成！"
echo "训练结果: ${OUTPUT_ROOT_DIR}/${SCENE}"
echo "渲染图片: ${OUTPUT_ROOT_DIR}/${SCENE}/test/ours_${RENDER_ITERATIONS}/renders/"
echo "评估报告: ${OUTPUT_ROOT_DIR}/${SCENE}/results.json"
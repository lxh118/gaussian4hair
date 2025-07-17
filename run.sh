#!/bin/bash

# Gaussian4Hair 训练脚本

echo "=== Gaussian4Hair 训练流程 ==="

# 默认路径设置
DEFAULT_DATA_PATH="./data"
DEFAULT_OUTPUT_PATH="./output"

# 解析命令行参数
DATA_PATH=${1:-$DEFAULT_DATA_PATH}
OUTPUT_PATH=${2:-$DEFAULT_OUTPUT_PATH}

echo "数据路径: $DATA_PATH"
echo "输出路径: $OUTPUT_PATH"

# 检查路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 错误: 数据路径不存在: $DATA_PATH"
    echo "用法: $0 <数据路径> [输出路径]"
    exit 1
fi

# 固定训练参数
ITERATIONS=7000
RENDER_ITERATIONS=7000

echo ""
echo "=== 1. 开始训练 (7000次迭代) ==="
CMD_TRAIN="python train.py -s ${DATA_PATH} -m ${OUTPUT_PATH} --eval --port 7001 \
    --iterations ${ITERATIONS} --save_iterations 7000 --densify_until_iter 6000 \
    --test_iterations 1000 3000 5000 7000 --hair_init"

eval $CMD_TRAIN

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "=== 2. 开始渲染 ==="
CMD_RENDER="python -W ignore::UserWarning render.py -s ${DATA_PATH} -m ${OUTPUT_PATH} \
    --skip_train --eval --iteration ${RENDER_ITERATIONS}"

eval $CMD_RENDER

if [ $? -ne 0 ]; then
    echo "❌ 渲染失败"
    exit 1
fi

echo ""
echo "=== 3. 开始评估 ==="
CMD_METRICS="python -W ignore::UserWarning metrics.py -m ${OUTPUT_PATH}"

eval $CMD_METRICS

if [ $? -ne 0 ]; then
    echo "❌ 评估失败"
    exit 1
fi

echo ""
echo "✅ 完整流程完成！"
echo "训练结果: $OUTPUT_PATH"
echo "渲染图片: $OUTPUT_PATH/test/ours_${RENDER_ITERATIONS}/renders/"
echo "评估报告: $OUTPUT_PATH/results.json"
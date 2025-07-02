#!/bin/bash

# Training script for ASR model with TAID

# 帮助函数
print_usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help                显示帮助信息"
    echo ""
    echo "执行流程:"
    echo "  1. 运行超参数优化 (50次试验)"
    echo "  2. 将最佳参数自动更新到配置文件"
    echo "  3. 使用最佳参数进行完整 TAID 训练"
    echo ""
    echo "注意: 整个流程会自动完成，无需手动干预"
}

# 解析命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            print_usage
            exit 1
            ;;
    esac
done

# 设置Python路径
export PYTHONPATH="$PYTHONPATH:$(cd .. && pwd)"

# 运行完整训练流程
echo "启动完整 TAID 训练流程..."
echo "执行命令: python train_taid.py"
echo ""

python train_taid.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TAID 训练流程成功完成！"
else
    echo ""
    echo "❌ TAID 训练流程执行失败"
    exit 1
fi

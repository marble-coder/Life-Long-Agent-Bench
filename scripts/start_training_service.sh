#!/bin/bash
# 启动 RLLM 训练服务
#
# 使用方式:
#   ./scripts/start_training_service.sh --model_path /path/to/model --device cuda:1
#
# 环境变量:
#   TRAINING_SERVICE_MODEL_PATH: 模型路径
#   TRAINING_SERVICE_DEVICE: 训练设备 (默认 cuda:1)
#   TRAINING_SERVICE_PORT: 服务端口 (默认 5555)
#   TRAINING_SERVICE_LORA_DIR: LoRA 输出目录

set -e

# 默认值
MODEL_PATH="${TRAINING_SERVICE_MODEL_PATH:-}"
DEVICE="${TRAINING_SERVICE_DEVICE:-cuda:1}"
PORT="${TRAINING_SERVICE_PORT:-5555}"
LORA_DIR="${TRAINING_SERVICE_LORA_DIR:-./outputs/training_service_lora}"
LEARNING_RATE="${TRAINING_SERVICE_LR:-2e-5}"
LORA_R="${TRAINING_SERVICE_LORA_R:-16}"
LORA_ALPHA="${TRAINING_SERVICE_LORA_ALPHA:-32}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --lora_output_dir)
            LORA_DIR="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model_path PATH      Path to the base model (required)"
            echo "  --device DEVICE        Training device (default: cuda:1)"
            echo "  --port PORT            Service port (default: 5555)"
            echo "  --lora_output_dir DIR  LoRA output directory"
            echo "  --learning_rate LR     Learning rate (default: 2e-5)"
            echo "  --lora_r R             LoRA rank (default: 16)"
            echo "  --lora_alpha ALPHA     LoRA alpha (default: 32)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 --model_path /path/to/model [options]"
    exit 1
fi

# 创建输出目录
mkdir -p "$LORA_DIR"
mkdir -p "$(dirname "$LORA_DIR")/logs"

LOG_FILE="$(dirname "$LORA_DIR")/logs/training_service_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "RLLM Training Service"
echo "========================================"
echo "Model Path:     $MODEL_PATH"
echo "Device:         $DEVICE"
echo "Port:           $PORT"
echo "LoRA Output:    $LORA_DIR"
echo "Learning Rate:  $LEARNING_RATE"
echo "LoRA Rank:      $LORA_R"
echo "LoRA Alpha:     $LORA_ALPHA"
echo "Log File:       $LOG_FILE"
echo "========================================"

# 设置 CUDA 可见设备（从 device 参数提取 GPU 编号）
if [[ "$DEVICE" == cuda:* ]]; then
    GPU_ID="${DEVICE#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    DEVICE="cuda:0"  # 重新映射到 cuda:0
    echo "Setting CUDA_VISIBLE_DEVICES=$GPU_ID, using $DEVICE"
fi

# 启动服务
python -m src.rllm_training_service.training_service \
    --model_path "$MODEL_PATH" \
    --lora_output_dir "$LORA_DIR" \
    --device "$DEVICE" \
    --port "$PORT" \
    --learning_rate "$LEARNING_RATE" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    2>&1 | tee "$LOG_FILE"

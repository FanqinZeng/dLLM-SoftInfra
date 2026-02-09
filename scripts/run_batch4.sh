#!/bin/bash

# ================= 配置区域 =================

# 修正为绝对路径
PRE_COMMANDS="export LLADA_INST_PATH=/remote-home/fanqinzeng/LLaDA-8B-Instruct && \
export HF_ALLOW_CODE_EVAL=\"1\" && \
export HF_ENDPOINT=https://hf-mirror.com && \
cd /remote-home/fanqinzeng/d2Cache && \
source /remote-home/fanqinzeng/d2Cache/d2Cache/bin/activate"

# 批次 4 的 8 条命令 (DualCache 专项测试)
# 核心配置: cache=prefix + cache.use_dual=true
declare -a cmds=(
    # [GPU 0] DualCache + Vanilla (Full Sequence / MaskGIT)
    # block_length=256 (全长不分块), steps=256
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=vanilla generation.block_length=256 generation.steps=256 model=llada-inst eval_args.limit=10"

    # [GPU 1] DualCache + Semi-AR (标准半自回归)
    # block_length=32 (分块生成)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=vanilla generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 2] DualCache + Parallel (并行解码)
    # 启用 threshold=0.9
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=vanilla generation.threshold=0.9 generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 3] DualCache + PC-Sampler
    # 启用 debias=true
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=pc_sampler generation.debias=true generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 4] DualCache + Certainty Prior Decoding
    # 启用 sigma=10
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=vanilla generation.sigma=10 generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 5] DualCache + DAEDAL
    # 动态长度扩展
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=daedal generation.initial_gen_length=128 model=llada-inst eval_args.limit=10"

    # [GPU 6] DualCache + KLASS
    # KL 散度引导
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=klass generation.kl_threshold=0.01 generation.kl_history_length=2 generation.block_length=64 model=llada-inst eval_args.limit=10"

    # [GPU 7] DualCache + EB-Sampler
    # 熵界采样
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix cache.use_dual=true generation=eb_sampler generation.gamma=0.001 generation.block_length=64 model=llada-inst eval_args.limit=10"
)

# ================= tmux 启动逻辑 =================
SESSION_NAME="batch_4"

tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

echo "正在启动 [批次 4] (DualCache 全系兼容性测试)..."

for i in {0..7}; do
    WINDOW_NAME="GPU$i"
    if [ $i -eq 0 ]; then
        tmux rename-window -t $SESSION_NAME:0 "$WINDOW_NAME"
    else
        tmux new-window -t $SESSION_NAME:$i -n "$WINDOW_NAME"
    fi
    FULL_CMD="export CUDA_VISIBLE_DEVICES=$i && $PRE_COMMANDS && ${cmds[$i]}"
    tmux send-keys -t $SESSION_NAME:$i "$FULL_CMD" C-m
    echo " -> [GPU $i] 任务已分发 (Mode: DualCache + ${cmds[$i]:160:15}...)"
done

echo "批次 4 启动完毕！正在进入 tmux..."
sleep 1
tmux attach -t $SESSION_NAME
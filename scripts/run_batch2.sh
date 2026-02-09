#!/bin/bash

# ================= 配置区域 =================

PRE_COMMANDS="export LLADA_INST_PATH=/remote-home/fanqinzeng/LLaDA-8B-Instruct && \
export HF_ALLOW_CODE_EVAL=\"1\" && \
export HF_ENDPOINT=https://hf-mirror.com && \
cd /remote-home/fanqinzeng/d2Cache && \
source /remote-home/fanqinzeng/d2Cache/d2Cache/bin/activate"

# 批次 2 的 8 条命令 (严格对照您的列表)
declare -a cmds=(
    # [GPU 0] dLLM + KLASS
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=klass generation.kl_threshold=0.01 generation.kl_history_length=2 generation.block_length=64 model=llada-inst eval_args.limit=10"

    # [GPU 1] dLLM + EB-Sampler
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=eb_sampler generation.gamma=0.001 generation.block_length=64 model=llada-inst eval_args.limit=10"

    # [GPU 2] dLLM + DAEDAL
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm generation=daedal generation.initial_gen_length=128 model=llada-inst eval_args.limit=10"

    # [GPU 3] d2Cache + Vanilla
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=vanilla generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 4] d2Cache + PC-Sampler
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=pc_sampler generation.debias=true generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 5] d2Cache + KLASS
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=klass generation.kl_threshold=0.01 generation.kl_history_length=2 generation.block_length=64 model=llada-inst eval_args.limit=10"

    # [GPU 6] d2Cache + EB-Sampler
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=eb_sampler generation.gamma=0.001 generation.block_length=64 model=llada-inst eval_args.limit=10"

    # [GPU 7] d2Cache + DAEDAL (按照列表，此处无 cache.sigma 参数)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 generation=daedal generation.initial_gen_length=128 model=llada-inst eval_args.limit=10"
)

# ================= tmux 启动逻辑 =================
SESSION_NAME="batch_2"

tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

echo "正在启动 [批次 2] (dLLM剩余 + d2Cache全系)..."

for i in {0..7}; do
    WINDOW_NAME="GPU$i"
    if [ $i -eq 0 ]; then
        tmux rename-window -t $SESSION_NAME:0 "$WINDOW_NAME"
    else
        tmux new-window -t $SESSION_NAME:$i -n "$WINDOW_NAME"
    fi
    FULL_CMD="export CUDA_VISIBLE_DEVICES=$i && $PRE_COMMANDS && ${cmds[$i]}"
    tmux send-keys -t $SESSION_NAME:$i "$FULL_CMD" C-m
    echo " -> [GPU $i] 任务已分发"
done

echo "批次 2 启动完毕！正在进入 tmux..."
sleep 1
tmux attach -t $SESSION_NAME
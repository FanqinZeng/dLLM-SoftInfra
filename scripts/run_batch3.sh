#!/bin/bash

# ================= 配置区域 =================

# 修正为绝对路径
PRE_COMMANDS="export LLADA_INST_PATH=/remote-home/fanqinzeng/LLaDA-8B-Instruct && \
export HF_ALLOW_CODE_EVAL=\"1\" && \
export HF_ENDPOINT=https://hf-mirror.com && \
cd /remote-home/fanqinzeng/d2Cache && \
source /remote-home/fanqinzeng/d2Cache/d2Cache/bin/activate"

# 批次 3 的 8 条命令
# 覆盖: Vanilla(Full), Certainty Prior, Parallel
# 组合: Prefix, dLLM, d2Cache
declare -a cmds=(
    # [GPU 0] Prefix + Vanilla (Full Sequence / MaskGIT)
    # block_length=256 意味着不分块，一次性生成
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix generation=vanilla generation.block_length=256 generation.steps=256 model=llada-inst eval_args.limit=10"

    # [GPU 1] Prefix + Certainty Prior
    # 启用 sigma=10 进行确定性先验引导
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix generation=vanilla generation.sigma=10 generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 2] Prefix + Parallel
    # 启用 threshold=0.9 进行并行解码
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix generation=vanilla generation.threshold=0.9 generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 3] dLLM + Vanilla (Full Sequence / MaskGIT)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=vanilla generation.block_length=256 generation.steps=256 model=llada-inst eval_args.limit=10"

    # [GPU 4] dLLM + Certainty Prior
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=vanilla generation.sigma=10 generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 5] dLLM + Parallel
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=vanilla generation.threshold=0.9 generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 6] d2Cache + Vanilla (Full Sequence / MaskGIT)
    # 必须开启 eager attention
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=vanilla generation.block_length=256 generation.steps=256 model=llada-inst eval_args.limit=10"

    # [GPU 7] d2Cache + Parallel
    # 特殊配置：parallel 需显式禁用 generation.sigma 并设置 inflate_w (参考文档)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 cache.inflate_w=4 generation=vanilla generation.threshold=0.9 generation.sigma=0 generation.block_length=32 model=llada-inst eval_args.limit=10"
)

# ================= tmux 启动逻辑 =================
SESSION_NAME="batch_3"

tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

echo "正在启动 [批次 3] (VanillaFull + CP + Parallel 补充实验)..."

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

echo "批次 3 启动完毕！正在进入 tmux..."
sleep 1
tmux attach -t $SESSION_NAME
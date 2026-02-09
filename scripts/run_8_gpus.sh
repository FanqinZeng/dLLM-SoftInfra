#!/bin/bash

# ================= 配置区域 =================

# 1. 定义公共的前置环境设置命令 (将应用到每个窗口)
# 注意：这里使用了 && 连接符，确保上一步成功才执行下一步
PRE_COMMANDS="export LLADA_INST_PATH=/remote-home/fanqinzeng/LLaDA-8B-Instruct && \
export HF_ALLOW_CODE_EVAL=\"1\" && \
export HF_ENDPOINT=https://hf-mirror.com && \
cd /remote-home/fanqinzeng/d2Cache && \
source /remote-home/fanqinzeng/d2Cache/d2Cache/bin/activate"

# 2. 定义 8 条具体的测试命令 (对应 GPU 0 - GPU 7)
# 这里为您精选了 8 个具有代表性的组合 (涵盖 Prefix, dLLM, d2Cache 和不同的 Decoding 策略)
declare -a cmds=(
    # [GPU 0] Prefix Cache + Vanilla (基准测试)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix generation=vanilla generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 1] Prefix Cache + PC-Sampler
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix generation=pc_sampler generation.debias=true generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 2] Prefix Cache + KLASS
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=prefix generation=klass generation.kl_threshold=0.01 generation.kl_history_length=2 generation.block_length=64 model=llada-inst eval_args.limit=10"

    # [GPU 3] dLLM-Cache + Vanilla
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=vanilla generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 4] dLLM-Cache + PC-Sampler
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 cache=dllm cache.rou=0.25 cache.kp=50 cache.kr=1 generation=pc_sampler generation.debias=true generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 5] d2Cache + Vanilla (官方推荐配置)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=vanilla generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 6] d2Cache + PC-Sampler
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=pc_sampler generation.debias=true generation.block_length=32 model=llada-inst eval_args.limit=10"

    # [GPU 7] d2Cache + EB-Sampler (探索性测试)
    "accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 attn_implementation=eager cache=d2cache cache.rollout_p=0.1 cache.current_k=32 cache.sigma=10 generation=eb_sampler generation.gamma=0.001 generation.block_length=64 model=llada-inst eval_args.limit=10"
)

# ================= 脚本逻辑 =================

SESSION_NAME="benchmark_suite"

# 1. 创建 tmux 会话 (如果已存在则先杀掉，避免冲突，可选)
tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

echo "正在启动 8 个并行任务..."

# 2. 循环分发任务
for i in {0..7}; do
    # 命名窗口
    WINDOW_NAME="GPU$i"
    if [ $i -eq 0 ]; then
        tmux rename-window -t $SESSION_NAME:0 "$WINDOW_NAME"
    else
        tmux new-window -t $SESSION_NAME:$i -n "$WINDOW_NAME"
    fi
    
    # 构造完整命令：
    # 1. 导出当前 GPU ID
    # 2. 执行前置环境配置 (PRE_COMMANDS)
    # 3. 执行具体的测试命令
    FULL_CMD="export CUDA_VISIBLE_DEVICES=$i && $PRE_COMMANDS && ${cmds[$i]}"
    
    # 发送命令到 tmux 窗口
    tmux send-keys -t $SESSION_NAME:$i "$FULL_CMD" C-m
    
    echo " -> [GPU $i] 任务已分发"
done

# 3. 进入 tmux 查看
echo "所有任务已启动！正在进入 tmux 监控面板..."
sleep 1
tmux attach -t $SESSION_NAME
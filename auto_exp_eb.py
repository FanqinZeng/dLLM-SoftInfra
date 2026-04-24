import subprocess
import time
import os

# ================= 配置区域 =================

# 1. 允许使用的 GPU ID 列表（指定为 2 和 3）
ALLOWED_GPUS = ["0", "1", "2", "3","4", "5", "6", "7"]

# 2. 在这里填入你所有要运行的实验指令
COMMANDS = [
    ### Dream
    ## gsm8k
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=dream-inst hydra.run.dir=./outputs_dream/gsm8k/eb_sampler_0001",
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.0001 model=dream-inst hydra.run.dir=./outputs_dream/gsm8k/eb_sampler_00001",

    ## humaneval_instruct
    # eb_sampler
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=humaneval_instruct batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=dream-inst hydra.run.dir=./outputs_dream/humaneval/eb_sampler_0001",
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=humaneval_instruct batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.0001 model=dream-inst hydra.run.dir=./outputs_dream/humaneval/eb_sampler_00001",

    ## ifeval
    # eb_sampler
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=ifeval batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=dream-inst hydra.run.dir=./outputs_dream/ifeval/eb_sampler_0001",
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=ifeval batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.0001 model=dream-inst hydra.run.dir=./outputs_dream/ifeval/eb_sampler_00001",
 
    ## gpqa_main_generative_n_shot
    # eb_sampler
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gpqa_main_generative_n_shot batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=dream-inst hydra.run.dir=./outputs_dream/gpqa/eb_sampler_0001",
    "DREAM_INST_PATH=/remote-home/fanqinzeng/DLLM/model/dParallel-Dream-7B-instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gpqa_main_generative_n_shot batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.0001 model=dream-inst hydra.run.dir=./outputs_dream/gpqa/eb_sampler_00001",

    ### LLaDA
    ## gsm8k
    # eb_sampler
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=llada-inst hydra.run.dir=./outputs_llada/gsm8k/eb_sampler_0001",
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gsm8k batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.01 model=llada-inst hydra.run.dir=./outputs_llada/gsm8k/eb_sampler_001",
    ## humaneval_instruct
    # eb_sampler
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=humaneval_instruct batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=llada-inst hydra.run.dir=./outputs_llada/humaneval/eb_sampler_0001",
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=humaneval_instruct batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.01 model=llada-inst hydra.run.dir=./outputs_llada/humaneval/eb_sampler_001",
  
    ## gpqa_main_generative_n_shot
    # eb_sampler
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gpqa_main_generative_n_shot batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=llada-inst hydra.run.dir=./outputs_llada/gpqa/eb_sampler_0001",
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=gpqa_main_generative_n_shot batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.01 model=llada-inst hydra.run.dir=./outputs_llada/gpqa/eb_sampler_001",

    ## ifeval
    # eb_sampler
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=ifeval batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.001 model=llada-inst hydra.run.dir=./outputs_llada/ifeval/eb_sampler_0001",
    "LLADA_INST_PATH=/remote-home/fanqinzeng/DLLM/model/LLaDA-8B-Instruct accelerate launch --num_machines 1 --num_processes 1 eval.py dataset.name=ifeval batch_size=1 seed=1234 generation=eb_sampler generation.gen_length=512 generation.block_length=32 generation.gamma=0.01 model=llada-inst hydra.run.dir=./outputs_llada/ifeval/eb_sampler_001",
]

# 3. 判定 GPU 空闲的显存阈值 (单位: MB)
FREE_MEMORY_THRESHOLD = 1024 

# 4. 检查 GPU 状态的时间间隔 (单位: 秒)
CHECK_INTERVAL = 15

# ==========================================

def get_idle_gpus(threshold, allowed_gpus):
    """通过 nvidia-smi 获取指定范围内空闲的 GPU ID 列表"""
    try:
        # 将列表拼接成 "2,3" 的格式
        gpu_ids_str = ",".join(allowed_gpus)
        
        # 使用 -i 参数让 nvidia-smi 只查询我们关心的 GPU
        smi_output = subprocess.check_output(
            ['nvidia-smi', f'-i={gpu_ids_str}', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        idle_gpus = []
        for line in smi_output.strip().split('\n'):
            if not line:
                continue
            gpu_id, memory_used = line.split(',')
            gpu_id = gpu_id.strip()
            
            # 检查显存是否低于阈值
            if int(memory_used.strip()) < threshold:
                idle_gpus.append(gpu_id)
        return idle_gpus
    except FileNotFoundError:
        print("错误: 找不到 nvidia-smi 命令。")
        return []
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi 执行失败，请检查指定的 GPU ID {allowed_gpus} 是否存在。")
        return []
    except Exception as e:
        print(f"检查 GPU 状态时发生错误: {e}")
        return []

def main():
    pending_commands = COMMANDS.copy()
    running_processes = {} # 格式 {gpu_id: subprocess.Popen对象}
    
    print(f"任务队列已初始化，共需运行 {len(pending_commands)} 个实验。")
    print(f"受限调度模式开启，仅监控和使用 GPU: {ALLOWED_GPUS}")

    while pending_commands or running_processes:
        # 1. 检查正在运行的进程是否已结束，释放 GPU
        finished_gpus = []
        for gpu_id, process in running_processes.items():
            if process.poll() is not None: 
                print(f"[释放] GPU {gpu_id} 上的任务已结束 (返回码: {process.returncode})。")
                finished_gpus.append(gpu_id)
        
        for gpu_id in finished_gpus:
            del running_processes[gpu_id]

        # 2. 寻找空闲的、且在我们允许列表里的 GPU 并派发任务
        if pending_commands:
            idle_gpus = get_idle_gpus(FREE_MEMORY_THRESHOLD, ALLOWED_GPUS)
            
            # 过滤掉我们自己脚本已经分配，但 `nvidia-smi` 还没来得及反映出显存增长的 GPU
            available_gpus = [g for g in idle_gpus if g not in running_processes]

            while pending_commands and available_gpus:
                gpu_to_use = available_gpus.pop(0)
                cmd = pending_commands.pop(0)
                
                print(f"[启动] 分配 GPU {gpu_to_use} 运行: {cmd}")
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_to_use
                
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    env=env
                )
                
                running_processes[gpu_to_use] = process

        # 3. 等待下一轮检查
        if pending_commands or running_processes:
            time.sleep(CHECK_INTERVAL)

    print("所有实验均已执行完毕！")

if __name__ == "__main__":
    main()
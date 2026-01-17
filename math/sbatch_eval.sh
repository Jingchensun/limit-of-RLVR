#!/bin/bash

# SLURM job directives
#SBATCH --job-name=verl                     # 作业名称
#SBATCH --output=verl.out                   # 输出日志文件
#SBATCH --error=verl.err                    # 错误日志文件
#SBATCH --ntasks=1                             # 启动 1 个任务
#SBATCH --cpus-per-task=16                      # 每个任务 8 个 CPU 核心
#SBATCH --mem=64GB                            # 每个 CPU 核心分配 32GB 内存
#SBATCH --time=23:00:00                        # 运行时间（1小时）
#SBATCH --partition=prod                       # 分区设置（prod 分区）
#SBATCH --gres=gpu:2

export WANDB_API_KEY="595cc8071abc681aa346ae6017f73fc16a9b2033"  # 替换为你的API Key
export WANDB_MODE=online  # 确保 wandb 处于在线模式


# 加载 Conda 环境
export PATH=/usr/local/cuda-11.8/bin:$PATH   #
export CUDA_HOME=/usr/local/cuda-11.8   # 
export CUDA=1
export PATH=/home/onsi/jsun/miniconda3/envs/verl/bin:$PATH  # Conda 环境路径
source /home/onsi/jsun/miniconda3/bin/activate verl        # 激活 Conda 环境

bash eval_math_nodes.sh \
    --run_name Qwen2.5-7B_minerva_math_temp0.6_n32_seed1_hf \
    --init_model ./models/Qwen2.5-7B-hf \
    --template qwen-boxed  \
    --tp_size 2 \
    --add_step_0 true  \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks minerva_math \
    --n_sampling 32 \
    --just_wandb false \
    --seed 1
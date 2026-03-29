#!/bin/bash
#SBATCH --job-name=nanochat-finephrase
#SBATCH --partition=batch,backfill,interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --account=nvr_lpr_llm
#SBATCH --output=runs/speedrun_finephrase_%j.log
#SBATCH --error=runs/speedrun_finephrase_%j.log

# Same as speedrun but uses the finephrase dataset.
# Data must be prepared first via: sbatch runs/prepare_finephrase.sh

set -euo pipefail

CONTAINER=/lustre/fsw/IOR-test/huay/image/nvcr.io_nvidia__pytorch__25.02-py3.sqsh
LUSTRE_HOME=/lustre/fsw/portfolios/llmservice/users/sdiao
PROJECT=$LUSTRE_HOME/Projects/nanochat

srun --container-image=$CONTAINER \
     --container-mounts=$LUSTRE_HOME:$LUSTRE_HOME \
     --container-writable \
     bash -c "
set -euo pipefail
cd $PROJECT

pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
pip install kernels rustbpe tiktoken tokenizers datasets psutil wandb fastapi uvicorn filelock 2>&1 | tail -2

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=$LUSTRE_HOME/.cache/nanochat_finephrase
export NANOCHAT_DATA_DIR=$LUSTRE_HOME/.cache/nanochat/base_data_finephrase
export HF_HOME=$LUSTRE_HOME/.cache/huggingface
export TRITON_CACHE_DIR=$LUSTRE_HOME/.cache/triton
export XDG_CACHE_HOME=$LUSTRE_HOME/.cache
mkdir -p \$NANOCHAT_BASE_DIR

WANDB_RUN=\${WANDB_RUN:-dummy}

python -m nanochat.report reset

# Tokenizer: train on finephrase data
python -m scripts.tok_train
python -m scripts.tok_eval

# Base model: same config as original speedrun
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=\$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# SFT
curl -L -o \$NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=\$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Report
python -m nanochat.report generate
"

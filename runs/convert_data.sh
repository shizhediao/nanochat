#!/bin/bash
#SBATCH --job-name=convert-arrow
#SBATCH --partition=batch,backfill,interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --account=nvr_lpr_llm
#SBATCH --output=runs/convert_%j.log
#SBATCH --error=runs/convert_%j.log

set -euo pipefail

LUSTRE_HOME=/lustre/fsw/portfolios/llmservice/users/sdiao
INPUT_DIR=$LUSTRE_HOME/data/climb_nm5.5_phase3_400b_shuffled_text_only_global_shuffle
OUTPUT_DIR=$LUSTRE_HOME/.cache/nanochat/base_data_climb_nm55

srun --container-image=/lustre/fsw/IOR-test/huay/image/nvcr.io_nvidia__pytorch__25.02-py3.sqsh \
     --container-mounts=$LUSTRE_HOME:$LUSTRE_HOME \
     --container-writable \
     python3 $LUSTRE_HOME/Projects/nanochat/scripts/convert_arrow_to_parquet.py \
         --input-dir $INPUT_DIR \
         --output-dir $OUTPUT_DIR \
         --num-workers 32

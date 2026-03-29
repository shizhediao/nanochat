#!/bin/bash
#SBATCH --job-name=prep-finephrase
#SBATCH --partition=batch,batch_short,backfill,interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --account=nvr_lpr_llm
#SBATCH --output=runs/prep_finephrase_%j.log
#SBATCH --error=runs/prep_finephrase_%j.log

set -euo pipefail

LUSTRE_HOME=/lustre/fsw/portfolios/llmservice/users/sdiao
OUTPUT_DIR=$LUSTRE_HOME/.cache/nanochat/base_data_finephrase

srun --container-image=/lustre/fsw/IOR-test/huay/image/nvcr.io_nvidia__pytorch__25.02-py3.sqsh \
     --container-mounts=$LUSTRE_HOME:$LUSTRE_HOME \
     --container-writable \
     bash -c "
set -euo pipefail

export HF_HOME=$LUSTRE_HOME/.cache/huggingface
pip install datasets pyarrow 2>&1 | tail -2

python3 << 'PYEOF'
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

output_dir = \"$OUTPUT_DIR\"
os.makedirs(output_dir, exist_ok=True)

ROW_GROUP_SIZE = 1024
TARGET_ROWS = 86016  # match climbmix shard size

# Use streaming to avoid downloading all 27k files at once
print('Loading finephrase (streaming)...')
ds = load_dataset('HuggingFaceFW/finephrase', 'all', split='train', streaming=True)

shard_idx = 0
buffer = []
buffer_rows = 0

# Check if we already have some shards (resume support)
existing = set(f for f in os.listdir(output_dir) if f.endswith('.parquet'))
if existing:
    shard_idx = max(int(f.split('_')[1].split('.')[0]) for f in existing) + 1
    skip_rows = shard_idx * TARGET_ROWS
    print(f'Resuming: {len(existing)} shards exist, skipping {skip_rows} rows, starting at shard {shard_idx}')
else:
    skip_rows = 0

row_count = 0
for example in ds:
    row_count += 1
    if row_count <= skip_rows:
        if row_count % 500000 == 0:
            print(f'  Skipping... {row_count}/{skip_rows}')
        continue

    buffer.append({'text': example['text']})
    buffer_rows += 1

    if buffer_rows >= TARGET_ROWS:
        table = pa.table({'text': [r['text'] for r in buffer]})
        out_path = os.path.join(output_dir, f'shard_{shard_idx:05d}.parquet')
        pq.write_table(table, out_path, row_group_size=ROW_GROUP_SIZE)
        if shard_idx % 10 == 0:
            print(f'  Wrote shard_{shard_idx:05d}.parquet ({table.num_rows} rows, total {row_count} processed)')
        shard_idx += 1
        buffer = []
        buffer_rows = 0

# Write remaining
if buffer:
    table = pa.table({'text': [r['text'] for r in buffer]})
    out_path = os.path.join(output_dir, f'shard_{shard_idx:05d}.parquet')
    pq.write_table(table, out_path, row_group_size=ROW_GROUP_SIZE)
    print(f'  Wrote final shard_{shard_idx:05d}.parquet ({table.num_rows} rows)')
    shard_idx += 1

print(f'Done! {shard_idx} shards, {row_count} total rows in {output_dir}')

# Verify
files = sorted(f for f in os.listdir(output_dir) if f.endswith('.parquet'))
pf = pq.ParquetFile(os.path.join(output_dir, files[0]))
print(f'Verification: {len(files)} files, first: {pf.num_row_groups} rg, {pf.metadata.num_rows} rows')
PYEOF
"

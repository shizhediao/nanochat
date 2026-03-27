"""
Convert arrow IPC stream files to parquet format for nanochat.

Usage:
    python -m scripts.convert_arrow_to_parquet \
        --input-dir /path/to/arrow/files \
        --output-dir /path/to/output/parquets \
        --num-workers 8
"""

import os
import argparse
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def convert_single(args):
    input_path, output_path = args
    if os.path.exists(output_path):
        return f"Skipping {os.path.basename(output_path)} (exists)"

    # Read all batches from arrow IPC stream
    batches = []
    with open(input_path, "rb") as f:
        reader = ipc.open_stream(f)
        schema = reader.schema
        while True:
            try:
                batches.append(reader.read_next_batch())
            except StopIteration:
                break

    table = pa.Table.from_batches(batches, schema=schema)
    pq.write_table(table, output_path, row_group_size=1024)
    return f"Converted {os.path.basename(output_path)} ({table.num_rows} rows)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build file list: arrow -> parquet, rename to shard_NNNNN.parquet
    arrow_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".arrow"))
    tasks = []
    for i, fname in enumerate(arrow_files):
        input_path = os.path.join(args.input_dir, fname)
        output_name = f"shard_{i:05d}.parquet"
        output_path = os.path.join(args.output_dir, output_name)
        tasks.append((input_path, output_path))

    print(f"Converting {len(tasks)} files using {args.num_workers} workers...")
    print(f"Output: {args.output_dir}")

    with Pool(processes=args.num_workers) as pool:
        for result in pool.imap_unordered(convert_single, tasks):
            print(result)

    print(f"Done! {len(tasks)} files converted.")

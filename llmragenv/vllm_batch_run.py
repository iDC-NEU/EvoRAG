import subprocess
import argparse


parser = argparse.ArgumentParser(description="LLMRag Workload")
parser.add_argument("--dataset", type=str, help="dataset", default='rgb_zyz')
args = parser.parse_args()

# for batch_size in [1, 2, 3, 4, 5, 8, 10, 16, 20, 24, 30, 32, 40, 50]:
for batch_size in [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]:
# for batch_size in [1, 2, 3, 4, 5, 8, 10]:
# for batch_size in [1]:
    print(f"Running batch size {batch_size}...")
    cmd = [
        "python", "./llmragenv/vllm_batch.py",
        "--batch_size", str(batch_size),
        "--dataset", args.dataset
    ]
    subprocess.run(cmd)

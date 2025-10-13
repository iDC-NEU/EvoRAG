#!/usr/bin/env bash
set -euo pipefail

# Paper-standard feedback-noise experiment. Override values, for example:
# RATE=0.2 START=3 END=6 bash run_noise.sh
DB="${DB:-rgb_zyz}"
DATA="${DATA:-rgb}"
PROC="${PROC:-1}"
START="${START:-0}"
END="${END:-6}"
RATE="${RATE:-0.1}"
ALGORITHM="${ALGORITHM:-standard_batch}"
LLM="${LLM:-Qwen2.5-7B-Instruct}"
LLMBACKEND="${LLMBACKEND:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$ALGORITHM" != "standard_batch" ]]; then
    echo "run_noise.sh reproduces the paper-standard PyTorch update; set ALGORITHM=standard_batch." >&2
    exit 2
fi

mkdir -p "logs/${ALGORITHM}/output"

for ((i = START; i < END; i++)); do
    echo "Running paper-standard noise iteration ${i} (rate=${RATE})..."
    "$PYTHON_BIN" -m database.insert_triples \
        --db "$DB" --data "$DATA" --algorithm "$ALGORITHM" \
        --proc "$PROC" --iteration "$i"

    "$PYTHON_BIN" kg_modify.py \
        --dataset_name "$DATA" --llm "$LLM" \
        --graphdb nebulagraph --space_name "$DB" \
        --option evolve_batch --algorithm "$ALGORITHM" \
        --llmbackend "$LLMBACKEND" --iteration "$i" --rate "$RATE" \
        > "logs/${ALGORITHM}/output/${LLMBACKEND}_${DATA}_evolve_batch_noise_${RATE}_${LLM}_${ALGORITHM}_${i}.log" 2>&1
done

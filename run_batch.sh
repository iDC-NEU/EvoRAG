#!/usr/bin/env bash
set -euo pipefail

# leaderboardStrategy full feedback loop. Override any value at invocation time,
# for example: LLM=Qwen2.5-32B-Instruct END=3 bash run_batch.sh
DB="${DB:-rgb_zyz}"
DATA="${DATA:-rgb}"
PROC="${PROC:-1}"
START="${START:-0}"
END="${END:-7}"
ALGORITHM="${ALGORITHM:-leaderboardStrategy}"
LLM="${LLM:-Qwen2.5-7B-Instruct}"
LLMBACKEND="${LLMBACKEND:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$ALGORITHM" != "leaderboardStrategy" ]]; then
    echo "run_batch.sh requires ALGORITHM=leaderboardStrategy." >&2
    exit 2
fi

mkdir -p "logs/${ALGORITHM}/output"

for ((i = START; i < END; i++)); do
    echo "Running leaderboardStrategy iteration ${i}..."
    "$PYTHON_BIN" -m database.insert_triples \
        --db "$DB" --data "$DATA" --algorithm "$ALGORITHM" \
        --proc "$PROC" --iteration "$i"

    "$PYTHON_BIN" kg_modify.py \
        --dataset_name "$DATA" --llm "$LLM" \
        --graphdb nebulagraph --space_name "$DB" \
        --option evolve_batch --algorithm "$ALGORITHM" \
        --llmbackend "$LLMBACKEND" --iteration "$i" \
        > "logs/${ALGORITHM}/output/${LLMBACKEND}_${DATA}_evolve_batch_${LLM}_${ALGORITHM}_${i}.log" 2>&1
done

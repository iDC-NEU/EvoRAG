#!/bin/bash
# ps -u zhangyz -o pid,lstart,%cpu,%mem,etime,time,cmd


PRUNINGS=(4 6 8 10 12 14 16)
# PRUNINGS=(6 10 12 14 16)
MAX_JOBS=2

DB="rgb_zyz" # nebulagraph数据库图空间名称
DATA="rgb" # 数据集名称
PROC=1
ITERATION=0 # 起始 iteration
LLM="Qwen2.5-32B-Instruct"
ALGORITHM="basic_batch"

python -m database.insert_triples --db "$DB" --data "$DATA" --proc "$PROC" --iteration "$ITERATION"

for p in "${PRUNINGS[@]}"
do
    echo "Starting task with --pruning=$p"

    python kg_modify.py --dataset_name "$DATA" --llm "$LLM" --graphdb nebulagraph --space_name "$DB" --option evolve_batch --llmbackend vllm --iteration "$i" > ./logs/${ALGORITHM}/output/${DATA}_evolve_batch_${LLM}_${ALGORITHM}_${i}.log 2>&1

    # 启动后台任务
    nohup python kg_modify.py \
        --dataset_name "$DATA" \
        --llm "$LLM" \
        --graphdb nebulagraph \
        --space_name "$DB" \
        --option evolve_batch \
        --llmbackend vllm \
        --iteration "$ITERATION" \
        --entity "$p" \
        --pruning 10 > ./logs/${ALGORITHM}/output/${DATA}_evolve_batch_${LLM}_${ALGORITHM}_${i}_${p}*10.log 2>&1 &

    # 控制最大并发数
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 10
    done
done

wait
echo "rgb所有任务已完成！"

# ps -u zhangyz -o pid,lstart,%cpu,%mem,etime,time,cmd
# nohup bash run_case_pruning.sh > bash_run_case_pruning.log 2>&1
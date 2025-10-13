#!/bin/bash

# 设置参数
# ps -u zhangyz -o pid,lstart,%cpu,%mem,etime,time,cmd

DB="rgb_zyz" # nebulagraph数据库图空间名称
DATA="rgb" # 数据集名称
PROC=1
START=7 # 起始 iteration
END=13 # 最后一个 iteration（不包含这个数）
# ALGORITHM="standard_batch"
ALGORITHM="basic_batch"
# LLM="Llama-3.1-8B-Instruct" # 问答与反馈大模型名称
LLM="Qwen2.5-32B-Instruct"
# LLM="Llama-3-8B-Instruct"
# LLM="llama3.1:8b-instruct-fp16"
LLMBACKEND="vllm"
# LLMBACKEND="llama_index"

# 遍历 iteration
for ((i=START; i<END; i++)); do
    echo "Running iteration $i..."
    python -m database.insert_triples --db "$DB" --data "$DATA" --algorithm "$ALGORITHM" --proc "$PROC" --iteration "$i"

    python kg_modify.py --dataset_name "$DATA" --llm "$LLM" --graphdb nebulagraph --space_name "$DB" --option evolve_batch --llmbackend "$LLMBACKEND" --iteration "$i" > ./logs/${ALGORITHM}/output/${LLMBACKEND}_${DATA}_evolve_batch_${LLM}_${ALGORITHM}_${i}.log 2>&1

done

# nohup bash run_batch.sh > ./logs/bash_run_btach.log 2>&1

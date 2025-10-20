#!/bin/bash


DB="rgb"
DATA="rgb"
PROC=1
START=0
END=8



for ((i=START; i<END; i++)); do
    echo "Running iteration $i..."
    python -m database.insert_triples --db "$DB" --data "$DATA" --proc "$PROC" --iteration "$i"
    python kg_modify.py --dataset_name "$DATA" --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name "$DB" --option kg_modify_llama_reproduce_forword --llmbackend huggingface --iteration "$i" > ./logs/${DATA}_kg_modify_llama_reproduce_forward_Meta-Llama-3-8B-Instruct_output_${i}_reproduce.log 2>&1
    python kg_modify.py --dataset_name "$DATA" --llm llama3.3 --graphdb nebulagraph --space_name "$DB" --option kg_modify_llama_reproduce_feedback --llmbackend llama_index --iteration "$i" > ./logs/${DATA}_kg_modify_llama_reproduce_feedback_llama3.3_llama_index_output_${i}_reproduce.log 2>&1
    sleep 20
done

# nohup bash run.sh > ./logs/bash_run.log 2>&1


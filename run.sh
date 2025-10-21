
###
 # @Author: fzb0316 fzb0316@163.com
 # @Date: 2024-10-18 15:20:13
 # @LastEditors: fzb0316 fzb0316@163.com
 # @LastEditTime: 2024-11-16 17:19:04
 # @FilePath: /BigModel/RAGWebUi_demo/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# test
# python kg_modify.py --dataset_name rgb --llm llama2:70b --graphdb nebulagraph --space_name rgb --option test --llmbackend llama_index
# python kg_modify.py --dataset_name multihop --llm llama2:70b --graphdb nebulagraph --space_name multihop --option test --llmbackend llama_index

# rgb
# triplets: 26876, entities: 30953

# withouot rag
# python kg_modify.py --dataset_name rgb --llm llama2:70b --graphdb nebulagraph --space_name rgb --option without_rag --llmbackend llama_index

# graph_rag
# python kg_modify.py --dataset_name rgb --llm llama2:70b --graphdb nebulagraph --space_name rgb --option graph_rag --llmbackend llama_index

# modify
# python kg_modify.py --dataset_name rgb --llm 8b --graphdb nebulagraph --space_name rgb_zyz --option graph_rag --llmbackend llama_index
# python kg_modify.py --dataset_name rgb --llm llqwenama2:70b --graphdb nebulagraph --space_name rgb_zyz --option KG_modify --llmbackend llama_index

# score
# python kg_modify.py --dataset_name rgb --llm llama2:70b --graphdb nebulagraph --space_name rgb --option graph_rag_score --llmbackend llama_index

# qwen3 测试
cp /home/hdd/zhangyz/rag-data/rgb_zyz-entity-embedding-standard_bge_bak.npz /home/hdd/zhangyz/rag-data/rgb_zyz-entity-embedding-standard.npz

python -m database.insert_triples --db rgb_zyz --data rgb --proc 1 --iteration 0

nohup python kg_modify.py --dataset_name rgb --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_forward --llmbackend huggingface --iteration 0 --type unchanged > rgb_feedback_forward_Meta-Llama-3-8B-Instruct_output_0_test.log 2>&1


# nohup python kg_modify.py --dataset_name rgb --llm Qwen3-32B --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_feedback --llmbackend huggingface --iteration 0 --type unchanged > rgb_feedback_back_Qwen3-32B_output_0.log 2>&1

# python -m database.insert_triples --db rgb_zyz --data rgb --proc 1 --iteration 1

# nohup python kg_modify.py --dataset_name rgb --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_forward --llmbackend huggingface --iteration 1 --type unchanged > rgb_feedback_forward_Meta-Llama-3-8B-Instruct_output_1.log 2>&1

# nohup python kg_modify.py --dataset_name rgb --llm Qwen3-32B --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_feedback --llmbackend huggingface --iteration 1 --type unchanged > rgb_feedback_back_Qwen3-32B_output_1.log 2>&1

# python -m database.insert_triples --db rgb_zyz --data rgb --proc 1 --iteration 2

# nohup python kg_modify.py --dataset_name rgb --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_forward --llmbackend huggingface --iteration 2 --type unchanged > rgb_feedback_forward_Meta-Llama-3-8B-Instruct_output_2.log 2>&1

# nohup python kg_modify.py --dataset_name rgb --llm Qwen3-32B --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_feedback --llmbackend huggingface --iteration 2 --type unchanged > rgb_feedback_back_Qwen3-32B_output_2.log 2>&1


# # 前向
# nohup python kg_modify.py --dataset_name rgb --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_forward --llmbackend huggingface --iteration 0 --type unchanged > rgb_feedback_forward_Meta-Llama-3-8B-Instruct_output_0.log 2>&1
# 反向
# nohup python kg_modify.py --dataset_name rgb --llm Qwen2.5-32B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_feedback --llmbackend huggingface --iteration 0 --type unchanged > rgb_feedback_back_Qwen2.5-32B-Instruct_output_0.log 2>&1

# # 前向
# nohup python kg_modify.py --dataset_name rgb --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_forward --llmbackend huggingface --iteration 1 --type unchanged > rgb_feedback_forward_Meta-Llama-3-8B-Instruct_output_1.log 2>&1
# # 反向
# nohup python kg_modify.py --dataset_name rgb --llm Qwen2.5-32B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_feedback --llmbackend huggingface --iteration 1 --type unchanged > rgb_feedback_back_Qwen2.5-32B-Instruct_output_1.log 2>&1

# # 前向
# nohup python kg_modify.py --dataset_name rgb --llm Meta-Llama-3-8B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_forward --llmbackend huggingface --iteration 2 --type unchanged > rgb_feedback_forward_Meta-Llama-3-8B-Instruct_output_2.log 2>&1
# # 反向
# nohup python kg_modify.py --dataset_name rgb --llm Qwen2.5-32B-Instruct --graphdb nebulagraph --space_name rgb_zyz --option kg_modify_feedback --llmbackend huggingface --iteration 2 --type unchanged > rgb_feedback_back_Qwen2.5-32B-Instruct_output_2.log 2>&1


# multihop

# without rag
# python kg_modify.py --dataset_name multihop --llm llama2:70b --graphdb nebulagraph --space_name multihop --option without_rag --llmbackend llama_index

# graph rag
# python kg_modify.py --dataset_name multihop --llm llama2:70b --graphdb nebulagraph --space_name multihop --option graph_rag --llmbackend llama_index

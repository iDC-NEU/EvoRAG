import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # 必须在导入 torch 或 vllm 之前设置！
from vllm import LLM, SamplingParams
import json
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chat.chat_prompt import * # 假设您的prompt模板都在这里
import argparse


def parse_vllm_metrics(metrics):
    # ... (您的 metrics 解析函数保持不变)
    # print(f"metrics: {metrics}")
    arrival_time = metrics.arrival_time
    first_scheduled_time = metrics.first_scheduled_time
    first_token_time = metrics.first_token_time
    finished_time = metrics.finished_time
    last_token_time = metrics.last_token_time
    time_in_queue = metrics.time_in_queue

    ttft_time = first_token_time - first_scheduled_time
    decode_time = finished_time - first_token_time
    generate_time = finished_time - first_scheduled_time
    wait_scheduled_time = first_scheduled_time - arrival_time

    end_to_end = finished_time - arrival_time

    return end_to_end, generate_time, ttft_time, decode_time, wait_scheduled_time


def main_original_sequential():
    """ 您原始的逐条处理函数，保留用于对比 """
    # ... (将您原来的 main 函数代码粘贴到这里)
    pass


def main_batched(args):
    """ 优化后的批量处理版本 """
    print("🚀 Running Batched Version for Maximum Throughput 🚀")
    
    # model_path = "/home/hdd/model/Llama-3.1-8B-Instruct"
    model_path = "/home/hdd/model/Qwen2.5-32B-Instruct"

    # 1. 启动 vLLM 模型 (与原代码相同)
    llm = LLM(
        # model="/home/hdd/model/Qwen2.5-32B-Instruct",
        # model="/home/hdd/model/Llama-3.1-8B-Instruct",
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        swap_space=128,
        max_model_len=1024*5, # 8192
        # use_v2_block_manager=True,      # 启用 PagedAttention
        # enable_chunked_prefill=True,    # 启用流式预填充
        # max_num_seqs=128,       # 控制并发序列数
        # block_size=16,          # 默认 16，可尝试 32 减少碎片
    )

    # 2. 定义采样参数 (与原代码相同)
    params = SamplingParams(
        temperature=0.0, 
        max_tokens=256,
        stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
    )

    params_2 = SamplingParams(
        temperature=0.0, 
        max_tokens=80,
        stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
    )

    # 3. 加载数据 (与原代码相同)
    try:
        # with open(f"./logs/stage/Qwen2.5-32B-Instruct_evolve_basic_forward_evolve_shared_prefix_{args.dataset}_0.json", 'r', encoding='utf-8') as file:
        #     data_list = json.load(file)
            
        if args.dataset == 'rgb_zyz':
            with open(f"/home/zhangyz/RAG_2025/logs/rgb/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_rgb_0.json", 'r', encoding='utf-8') as file:
                data_list = json.load(file)
        elif args.dataset == "multihop_zyz":
            with open(f"/home/zhangyz/RAG_2025/logs/multihop/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_multihop_0.json", 'r', encoding='utf-8') as file:
                data_list = json.load(file)
        elif args.dataset == "hotpotqa_1":
            with open(f"/home/zhangyz/RAG_2025/logs/hotpotqa600/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_hotpotqa600_0.json", 'r', encoding='utf-8') as file:
                data_list = json.load(file)
        # else:
        #     with open(f"./logs/stage/Qwen2.5-32B-Instruct_evolve_basic_forward_evolve_shared_prefix_{args.dataset}_0.json", 'r', encoding='utf-8') as file:
        #         data_list = json.load(file)
            # data_list = data_list[0:1]
    except FileNotFoundError:
        print("警告：找不到指定的JSON文件。将使用一个虚拟数据点进行演示。")
        data_list = [
            {'query': 'What is the capital of France?', 'answer': 'Paris', 'filtered_retrieve_result': ['Paris is the capital and most populous city of France.'], 'response': 'The capital of France is Paris.', 'label': True},
            {'query': 'Who wrote "Hamlet"?', 'answer': 'William Shakespeare', 'filtered_retrieve_result': ['Hamlet is a tragedy written by William Shakespeare.'], 'response': 'William Shakespeare wrote "Hamlet".', 'label': True}
        ]


    # for batch_size in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]:
    batch_size = args.batch_size
    print(f"\n\n===== Processing Batch Size: {batch_size} =====")
    print(f"================================================================================================================")
    print(f"================================================================================================================")
    print(f"================================================================================================================")
    data_list = data_list[0:batch_size]
    # --- 阶段一：准备并执行所有 Forward 请求 ---

    print(f"--- Preparing {len(data_list)} forward prompts for batching... ---")
    all_forward_prompts = []
    for data_item in data_list:
        context = "".join([f"Path {idx}:\t{sentence}\n" for idx, sentence in enumerate(data_item['filtered_retrieve_result'])])
        prefix = shared_prefix.format(knowledge_sequences=context)
        query_prompt_user = chat_with_graphrag_for_response_user_shared_prefix.format(question=data_item['query'])
        
        # 构建完整的prompt
        if "Qwen2.5" in model_path:
            full_prompt = qwen2_instruct_system.format(system_instruction=prefix) + qwen2_instruct_user.format(user_instruction=query_prompt_user)
        elif "Llama" in model_path:
            full_prompt = llama_instruct_system.format(system_instruction=prefix) + llama_instruct_user.format(user_instruction=query_prompt_user)
        all_forward_prompts.append(full_prompt)

    # 吞吐量
    print("--- Starting Stage 1: Generating all forward responses (batched)... ---")
    batch_forward_start_time = time.perf_counter()
    outputs1 = llm.generate(all_forward_prompts, params)
    batch_forward_end_time = time.perf_counter()
    print(f"--- Stage 1 finished in {batch_forward_end_time - batch_forward_start_time:.4f} seconds. ---")
    print(f"--- Average time per request: {(batch_forward_end_time - batch_forward_start_time) / len(data_list):.4f} seconds ---")
    print(f"answer : {outputs1[0].outputs[0].text}")
    print(f"prompt len: {len(outputs1[0].prompt_token_ids)}")
    print(f"token len: {len(outputs1[0].outputs[0].token_ids)}")
    print(f"num_cached_tokens: {outputs1[0].num_cached_tokens}")
    print(f"avg token len per request: {sum([len(output.prompt_token_ids) for output in outputs1]) / len(outputs1):.2f}")
    print(f"avg cached tokens per request: {sum([output.num_cached_tokens for output in outputs1]) / len(outputs1):.2f}")

    # --- 阶段一点五：准备并执行所有 align_human 请求 ---
    print(f"--- Preparing {len(data_list)} align human prompts for batching... ---")
    all_align_human_prompts = []
    # 注意：我们必须保持原始 data_list 和 outputs1 的顺序一致性
    for data_item, forward_output in zip(data_list, outputs1):
        # 从第一次请求的结果中获取 response
        generated_response = forward_output.outputs[0].text
        cur_align_human_prompt_user = align_human_user_prompt.format(question = data_item['query'], model_answer = generated_response, gold_answer = str(data_item['answer']))

        
        # 构建完整的prompt
        if "Qwen2.5" in model_path:
            full_prompt = qwen2_instruct_system.format(system_instruction=align_human_system_prompt) + qwen2_instruct_user.format(user_instruction=cur_align_human_prompt_user)
        elif "Llama" in model_path:
            full_prompt = llama_instruct_system.format(system_instruction=align_human_system_prompt) + llama_instruct_user.format(user_instruction=cur_align_human_prompt_user)
        all_align_human_prompts.append(full_prompt)

    print("--- Starting Stage 1.5: Generating all align human acc (batched)... ---")
    batch_align_human_start_time = time.perf_counter()
    outputs1_5 = llm.generate(all_align_human_prompts, params_2)
    batch_align_human_end_time = time.perf_counter()
    print(f"--- Stage 1.5 finished in {batch_align_human_end_time - batch_align_human_start_time:.4f} seconds. ---")
    print(f"--- Average time per request: {(batch_align_human_end_time - batch_align_human_start_time) / len(data_list):.4f} seconds ---")
    print(f"answer : {outputs1_5[0].outputs[0].text}")
    print(f"prompt len: {len(outputs1_5[0].prompt_token_ids)}")
    print(f"token len: {len(outputs1_5[0].outputs[0].token_ids)}")
    print(f"num_cached_tokens: {outputs1_5[0].num_cached_tokens}")
    print(f"avg token len per request: {sum([len(output.prompt_token_ids) for output in outputs1_5]) / len(outputs1_5):.2f}")
    print(f"avg cached tokens per request: {sum([output.num_cached_tokens for output in outputs1_5]) / len(outputs1_5):.2f}")

    
    # --- 阶段二：准备并执行所有 Feedback 请求 ---

    print(f"--- Preparing {len(data_list)} feedback prompts for batching... ---")
    all_feedback_prompts = []
    # 注意：我们必须保持原始 data_list 和 outputs1 的顺序一致性
    for data_item, forward_output in zip(data_list, outputs1):
        # 从第一次请求的结果中获取 response
        generated_response = forward_output.outputs[0].text

        context = "".join([f"Path {idx}:\t{sentence}\n" for idx, sentence in enumerate(data_item['filtered_retrieve_result'])])
        prefix = shared_prefix.format(knowledge_sequences=context)
        # feedback_prompt_user = score_feedback_prompt_standard_user_test_optimized.format(question=data_item['query'], last_response=generated_response)
        feedback_prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question=data_item['query'], last_response=generated_response)

        # 构建完整的prompt
        if "Qwen2.5" in model_path:
            full_prompt = qwen2_instruct_system.format(system_instruction=prefix) + qwen2_instruct_user.format(user_instruction=feedback_prompt_user)
        elif "Llama" in model_path:
            full_prompt = llama_instruct_system.format(system_instruction=prefix) + llama_instruct_user.format(user_instruction=feedback_prompt_user)
        all_feedback_prompts.append(full_prompt)

    print("--- Starting Stage 2: Generating all feedback scores (batched)... ---")
    batch_feedback_start_time = time.perf_counter()
    outputs2 = llm.generate(all_feedback_prompts, params_2)
    batch_feedback_end_time = time.perf_counter()
    print(f"--- Stage 2 finished in {batch_feedback_end_time - batch_feedback_start_time:.4f} seconds. ---")
    print(f"--- Average time per request: {(batch_feedback_end_time - batch_feedback_start_time) / len(data_list):.4f} seconds ---")
    print(f"answer : {outputs2[0].outputs[0].text}")
    print(f"prompt len: {len(outputs2[0].prompt_token_ids)}")
    print(f"token len: {len(outputs2[0].outputs[0].token_ids)}")
    print(f"num_cached_tokens: {outputs2[0].num_cached_tokens}")

    print(f"avg token len per request: {sum([len(output.prompt_token_ids) for output in outputs2]) / len(outputs2):.2f}")
    print(f"avg cached tokens per request: {sum([output.num_cached_tokens for output in outputs2]) / len(outputs2):.2f}")

    # --- 4. 结果处理与指标计算 ---
    
    # 初始化指标列表
    forward_metrics_totals = [[] for _ in range(5)]
    align_human_metrics_totals = [[] for _ in range(5)]
    feedback_metrics_totals = [[] for _ in range(5)]


    metric_names = ["end2end", "generate", "prefill", "decode", "wait_scheduled"]
    print("\n--- Processing Batch Results ---")
    for i in range(len(data_list)):
        forward_output = outputs1[i]
        align_human_output = outputs1_5[i]
        feedback_output = outputs2[i]
        
        # 解析并累加 Forward 指标
        metrics = parse_vllm_metrics(forward_output.metrics)
        for j in range(5):
            forward_metrics_totals[j].append(metrics[j])
            # print(f"{metric_names[j]}_time_forward_item_{i}: {metrics[j]:.4f} seconds")
        
        # 解析并累加 align_human 指标
        metrics = parse_vllm_metrics(align_human_output.metrics)
        for j in range(5):
            align_human_metrics_totals[j].append(metrics[j])
            
        # 解析并累加 Feedback 指标
        metrics = parse_vllm_metrics(feedback_output.metrics)
        for j in range(5):
            feedback_metrics_totals[j].append(metrics[j])

        # (可选) 打印单项结果
        # if i < min(5, len(data_list)): # 只打印前5个示例结果
        #     print(f"\n--- Item {i+1} ---")
        #     print(f"Query: {data_list[i]['query']}")
        #     print("Answer 1:", forward_output.outputs[0].text)
        #     # print("feedback prompt:", all_feedback_prompts[i])
        #     print("Answer 2:", feedback_output.outputs[0].text)
        #     print(outputs1[i])
        #     print("-" * 20)

    # 计算并打印平均指标
    # if data_list:
    #     metric_names = ["end2end", "generate", "prefill", "decode", "wait_scheduled"]
        
    #     print("\n--- Forward Average Times (Batched) ---")
    #     for name, totals in zip(metric_names, forward_metrics_totals):
    #         avg_time = sum(totals) / len(totals)
    #         print(f"  Average {name}_time: {avg_time:.4f} seconds")

    #     print("\n--- Align Human Average Times (Batched) ---")
    #     for name, totals in zip(metric_names, align_human_metrics_totals):
    #         avg_time = sum(totals) / len(totals)
    #         print(f"  Average {name}_time: {avg_time:.4f} seconds")
            
    #     print("\n--- Feedback Average Times (Batched) ---")
    #     for name, totals in zip(metric_names, feedback_metrics_totals):
    #         avg_time = sum(totals) / len(totals)
    #         print(f"  Average {name}_time: {avg_time:.4f} seconds")


if __name__ == "__main__":
    # 运行优化后的批量处理版本

    parser = argparse.ArgumentParser(description="LLMRag Workload")
    parser.add_argument("--dataset", type=str, help="dataset", default='rgb_zyz')
    parser.add_argument("--batch_size", type=int, help="batch_size", default=5)
    
    args = parser.parse_args()

    main_batched(args)
    
    # 如果需要，可以取消注释以运行原始版本进行对比
    # main_original_sequential()

    # python -m llmragenv.vllm_batch > ./logs/test/vllm_batch.log 2>&1


# 10

# --- Forward Average Times (Batched) ---
#   Average end2end_time: 9.5421 seconds
#   Average generate_time: 9.4892 seconds
#   Average prefill_time: 8.7937 seconds
#   Average decode_time: 0.6955 seconds
#   Average wait_scheduled_time: 0.0529 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 4.9838 seconds
#   Average generate_time: 4.9415 seconds
#   Average prefill_time: 2.3816 seconds
#   Average decode_time: 2.5599 seconds
#   Average wait_scheduled_time: 0.0423 seconds

# 30

# --- Forward Average Times (Batched) ---
#   Average end2end_time: 28.9216 seconds
#   Average generate_time: 20.1355 seconds
#   Average prefill_time: 10.7428 seconds
#   Average decode_time: 9.3927 seconds
#   Average wait_scheduled_time: 8.7861 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 14.5462 seconds
#   Average generate_time: 11.4905 seconds
#   Average prefill_time: 2.7282 seconds
#   Average decode_time: 8.7623 seconds
#   Average wait_scheduled_time: 3.0557 seconds

# 分明是一个线性的时间

# 批处理提升的是吞吐量


# --- Stage 1 finished in 307.4585 seconds. ---

# --- Stage 2 finished in 424.2424 seconds. ---

# --- Forward Average Times (Batched) ---
#   Average end2end_time: 184.0141 seconds
#   Average generate_time: 18.4649 seconds
#   Average prefill_time: 3.5531 seconds
#   Average decode_time: 14.9118 seconds
#   Average wait_scheduled_time: 165.5493 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 232.8921 seconds
#   Average generate_time: 20.7339 seconds
#   Average prefill_time: 2.0145 seconds
#   Average decode_time: 18.7194 seconds
#   Average wait_scheduled_time: 212.1582 seconds


# 1
# --- Stage 1 finished in 1.6973 seconds. ---
# --- Stage 2 finished in 3.7658 seconds. ---
# --- Forward Average Times (Batched) ---
#   Average end2end_time: 1.6515 seconds
#   Average generate_time: 1.6317 seconds
#   Average prefill_time: 1.1104 seconds
#   Average decode_time: 0.5214 seconds
#   Average wait_scheduled_time: 0.0198 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 3.7138 seconds
#   Average generate_time: 3.7035 seconds
#   Average prefill_time: 0.2694 seconds
#   Average decode_time: 3.4341 seconds
#   Average wait_scheduled_time: 0.0104 seconds

# 2
# --- Forward Average Times (Batched) ---
#   Average end2end_time: 2.0607 seconds
#   Average generate_time: 2.0438 seconds
#   Average prefill_time: 1.6749 seconds
#   Average decode_time: 0.3690 seconds
#   Average wait_scheduled_time: 0.0169 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 4.3086 seconds
#   Average generate_time: 4.2940 seconds
#   Average prefill_time: 0.4938 seconds
#   Average decode_time: 3.8002 seconds
#   Average wait_scheduled_time: 0.0145 seconds

# 3
# --- Stage 1 finished in 3.2989 seconds. ---
# --- Stage 2 finished in 4.9588 seconds. ---
# --- Forward Average Times (Batched) ---
#   Average end2end_time: 3.0123 seconds
#   Average generate_time: 2.9898 seconds
#   Average prefill_time: 2.6862 seconds
#   Average decode_time: 0.3036 seconds
#   Average wait_scheduled_time: 0.0224 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 4.3431 seconds
#   Average generate_time: 4.3270 seconds
#   Average prefill_time: 0.7342 seconds
#   Average decode_time: 3.5928 seconds
#   Average wait_scheduled_time: 0.0161 seconds

# 5
# --- Forward Average Times (Batched) ---
#   Average end2end_time: 5.3266 seconds
#   Average generate_time: 5.2878 seconds
#   Average prefill_time: 4.6834 seconds
#   Average decode_time: 0.6043 seconds
#   Average wait_scheduled_time: 0.0389 seconds

# --- Feedback Average Times (Batched) ---
#   Average end2end_time: 3.8419 seconds
#   Average generate_time: 3.8184 seconds
#   Average prefill_time: 1.1977 seconds
#   Average decode_time: 2.6208 seconds
#   Average wait_scheduled_time: 0.0234 seconds
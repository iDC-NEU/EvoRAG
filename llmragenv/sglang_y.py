from vllm import LLM, SamplingParams

import json
import time
from chat.chat_prompt import *

# import third_party.sglang.python.sglang as sgl
# from third_party.sglang.python.sglang.srt.server_args import ServerArgs
import sglang as sgl
from sglang.srt.server_args import ServerArgs

# 1. 启动 vLLM 模型
# llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
# llm = LLM(model="/home/zhangyz/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct")
def main():
    # llm = LLM(model="/home/zhangyz/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a")
    # llm = LLM(model="/home/hdd/model/Qwen2.5-32B-Instruct")


    server_args = ServerArgs(
        model_path="/home/hdd/model/Meta-Llama-3-8B-Instruct",
        port=31000,
        host="127.0.0.1",
        device="cuda",
        tp_size=3,
        base_gpu_id=0,
        chunked_prefill_size=100000,
        mem_fraction_static=0.9,
        # context_length=20000,  # 输入长度会超过默认值，需要设置
        log_level="info",
        disable_overlap_schedule=True,
    )

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 2048,
        "stop": ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
    }

    llm = sgl.Engine(server_args=server_args)



    # 3. 定义采样参数
    params = SamplingParams(
        temperature=0.0, 
        max_tokens=2048,
        stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
    )



    # 2. 加载数据
    try:
        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_evolve_basic_forward_evolve_shared_prefix_rgb_zyz_0.json", 'r', encoding='utf-8') as file:
            data_list = json.load(file)
    except FileNotFoundError:
        print("警告：找不到指定的JSON文件。将使用一个虚拟数据点进行演示。")
        data_list = [
            {
                'query': 'What is the capital of France?',
                'answer': 'Paris',
                'filtered_retrieve_result': ['Paris is the capital and most populous city of France.'],
                'response': 'The capital of France is Paris.',
                'label': True
            }
        ]


    prefix_total_time = []
    forward_total_time = []
    feedback_total_time = []

    # 3. 循环处理数据，保持您原有的计时逻辑
    for index, data_item in enumerate(data_list):
        if index >= 1:
            break
        print(f"--- Processing item {index+1}/{len(data_list)} ---")
        query = data_item['query']
        answer = data_item['answer']
        filtered_retrieve_result = data_item['filtered_retrieve_result']
        response = data_item['response']
        
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context += f"Path {idx}:\t{sentence}\n"
        
        # 准备 prompts
        prefix = shared_prefix.format(knowledge_sequences=context)
        query_prompt_user = chat_with_graphrag_for_response_user_shared_prefix.format(question=query)
        feedback_prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question=query, last_response=response)

        
        # 4. 第一次请求：构建 prefix cache
        forward_start_time = time.perf_counter()
        # outputs1 = llm.generate(
        #     [prefix+query_prompt_user],
        #     sampling_params=params,
        #     use_prefix_cache=True  # 开启前缀缓存
        # )
        # outputs1 = llm.generate(
        #     # [prefix+query_prompt_user],
        #     # [llama_instruct_system.format(system_instruction = prefix) + llama_instruct_user.format(user_instruction = query_prompt_user)],
        #     [qwen2_instruct_system.format(system_instruction = prefix) + qwen2_instruct_user.format(user_instruction = query_prompt_user)],
        #     params,
        #     # use_prefix_cache=True  # 开启前缀缓存
        # )
        outputs1 = llm.generate(qwen2_instruct_system.format(system_instruction = prefix) + qwen2_instruct_user.format(user_instruction = query_prompt_user), sampling_params)
        forward_end_time = time.perf_counter()
        

        
        feedback_start_time = time.perf_counter()
        # 5. 第二次请求：复用 prefix cache（vLLM 自动识别相同前缀）
        # outputs2 = llm.generate(
        #     [prefix+feedback_prompt_user],
        #     sampling_params=params,
        #     use_prefix_cache=True
        # )
        # outputs2 = llm.generate(
        #     # [llama_instruct_system.format(system_instruction = prefix) + llama_instruct_user.format(user_instruction = feedback_prompt_user)],
        #     [qwen2_instruct_system.format(system_instruction = prefix) + qwen2_instruct_user.format(user_instruction = feedback_prompt_user)],
        #     params,
        #     # use_prefix_cache=True
        # )
        outputs2 = "ss"
        feedback_end_time = time.perf_counter()
        
        
        forward_time = forward_end_time - forward_start_time
        feedback_time = feedback_end_time - feedback_start_time
        
        forward_total_time.append(forward_time)
        feedback_total_time.append(feedback_time)

        print(f"Query: {query}")
        print("Answer 1:", outputs1[0].outputs[0].text)
        print("Answer 2:", outputs2[0].outputs[0].text)
        print(f"Time - Forward: {forward_time:.4f}s, Feedback: {feedback_time:.4f}s")
        print("-" * 20)


    # 4. 计算平均时间
    if data_list:
        avg_forward = sum(forward_total_time) / len(forward_total_time)
        avg_feedback = sum(feedback_total_time) / len(feedback_total_time)
        print("\n--- Average Times ---")
        print(f"Average Forward Time: {avg_forward:.4f} seconds")
        print(f"Average Feedback Time: {avg_feedback:.4f} seconds")
if __name__ == "__main__":
    main()


# python -m llmragenv.vllm

# llama
# Time - Forward: 0.5247s, Feedback: 11.5081s
# Time - Forward: 0.5261s, Feedback: 12.0195s
# qwen

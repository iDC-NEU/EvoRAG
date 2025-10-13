
# 确保 sglang 已正确安装
# pip install "sglang[srt]"
from sglang import Engine
import json
import time
from chat.chat_prompt import *

# --- 为使代码可独立运行，提供示例的 prompt 模板 ---
# 在您的实际代码中，请使用 from chat.chat_prompt import *


# 1. 启动模型
# 【关键修改】: 将 tensor_parallel_size 重命名为 tp_size
# engine = Engine(model_path="Qwen/Qwen2.5-32B-Instruct", tp_size=1)
engine = Engine(model_path="meta-llama/Meta-Llama-3-8B-Instruct", tp_size=1)


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
    print(f"--- Processing item {index+1}/{len(data_list)} ---")
    query = data_item['query']
    answer = data_item['answer']
    filtered_retrieve_result = data_item['filtered_retrieve_result']
    response = data_item['response']
    
    context = ""
    for idx, sentence in enumerate(filtered_retrieve_result, start=0):
        context += f"Path {idx}:\t{sentence}\n"
    
    # 准备 prompts
    prefix = shared_prefix.format(knowledge_paths=context)
    query_prompt_user = chat_with_graphrag_for_response_user_shared_prefix.format(question=query)
    feedback_prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question=query, last_response=response)

    # 在每个数据点的循环内创建 session
    with engine.session() as sess:
        # 计时1: 预计算 prefix cache
        prefix_start_time = time.perf_counter()
        prefix_state = sess.prefill(prefix)
        prefix_end_time = time.perf_counter()
        
        # 计时2: 使用缓存生成第一个问题的答案
        forward_start_time = time.perf_counter()
        out1 = sess.generate(
            prompt=[{"role": "user", "content": query_prompt_user}],
            prefix_state=prefix_state,
            max_new_tokens=2048,
            temperature=0.0
        )
        forward_end_time = time.perf_counter()
        
        # 计时3: 使用同一缓存生成第二个问题的反馈
        feedback_start_time = time.perf_counter()
        out2 = sess.generate(
            prompt=[{"role": "user", "content": feedback_prompt_user}],
            prefix_state=prefix_state,
            max_new_tokens=2048,
            temperature=0.0
        )
        feedback_end_time = time.perf_counter()

        # 收集和打印结果
        prefix_time = prefix_end_time - prefix_start_time
        forward_time = forward_end_time - forward_start_time
        feedback_time = feedback_end_time - feedback_start_time
        
        prefix_total_time.append(prefix_time)
        forward_total_time.append(forward_time)
        feedback_total_time.append(feedback_time)

        print(f"Query: {query}")
        print(f"Answer 1 (Response): {out1.text}")
        print(f"Answer 2 (Feedback): {out2.text}")
        print(f"Time - Prefix: {prefix_time:.4f}s, Forward: {forward_time:.4f}s, Feedback: {feedback_time:.4f}s")
        print("-" * 20)


# 4. 计算平均时间
if data_list:
    avg_prefix = sum(prefix_total_time) / len(prefix_total_time)
    avg_forward = sum(forward_total_time) / len(forward_total_time)
    avg_feedback = sum(feedback_total_time) / len(feedback_total_time)
    print("\n--- Average Times ---")
    print(f"Average Prefix Time: {avg_prefix:.4f} seconds")
    print(f"Average Forward Time: {avg_forward:.4f} seconds")
    print(f"Average Feedback Time: {avg_feedback:.4f} seconds")
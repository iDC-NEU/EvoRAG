import sglang as sgl
import json
import time
from chat.chat_prompt import *

# 1. 定义 SGLang 生成函数
# 这个函数将作为我们的 prompt 模板。SGLang 会将函数内的逻辑编译成高效的请求。
@sgl.function
def generate_with_prefix(s, system_template, user_template, common_prefix, user_prompt):
    """
    SGLang 函数，用于构建并执行一次生成请求。
    - s: SGLang 内部状态对象，必须作为第一个参数。
    - system_template/user_template: 模型的对话模板。
    - common_prefix: 共享的前缀内容。
    - user_prompt: 每次请求不同的用户指令部分。
    """
    s += system_template.format(system_instruction=common_prefix)
    s += user_template.format(user_instruction=user_prompt)
    s += sgl.gen("result") # "result" 是我们为生成内容指定的变量名

def main():
    # 2. 启动 SGLang Runtime 后端
    # 参数与 vLLM 类似，但名称可能不同 (e.g., model -> model_path, tensor_parallel_size -> tp_size)
    runtime = sgl.Runtime(
        model_path="/home/hdd/model/Qwen2.5-32B-Instruct",
        tp_size=2,
        # SGLang 会自动管理内存，通常不需要 gpu_memory_utilization
        # log_level="DEBUG" # 可在调试时开启
    )
    sgl.set_default_backend(runtime)
    print("SGLang Runtime a-OK! 🚀")

    # 3. 加载数据 (与原代码相同)
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

    forward_total_time = []
    feedback_total_time = []

    # 4. 循环处理数据
    for index, data_item in enumerate(data_list):
        if index >= 1:
            break
        print(f"--- Processing item {index+1}/{len(data_list)} ---")
        query = data_item['query']
        response = data_item['response']
        filtered_retrieve_result = data_item['filtered_retrieve_result']
        
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context += f"Path {idx}:\t{sentence}\n"
        
        # 准备 prompts (与原代码相同)
        prefix = shared_prefix.format(knowledge_sequences=context)
        query_prompt_user = chat_with_graphrag_for_response_user_shared_prefix.format(question=query)
        feedback_prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question=query, last_response=response)
        
        # ----- SGLang 调用方式 -----
        
        # 5. 第一次请求 (Forward)
        forward_start_time = time.perf_counter()
        state1 = generate_with_prefix.run(
            system_template=qwen2_instruct_system,
            user_template=qwen2_instruct_user,
            common_prefix=prefix,
            user_prompt=query_prompt_user,
            temperature=0.0,
            max_new_tokens=2048,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]
        )
        forward_end_time = time.perf_counter()

        # 6. 第二次请求 (Feedback)
        # SGLang 的 Runtime 会自动检测到 common_prefix 部分与上一次请求相同，
        # 并直接复用其 KV Cache，极大地提升了第二次请求的速度。
        feedback_start_time = time.perf_counter()
        state2 = generate_with_prefix.run(
            system_template=qwen2_instruct_system,
            user_template=qwen2_instruct_user,
            common_prefix=prefix,
            user_prompt=feedback_prompt_user,
            temperature=0.0,
            max_new_tokens=2048,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]
        )
        feedback_end_time = time.perf_counter()
        
        forward_time = forward_end_time - forward_start_time
        feedback_time = feedback_end_time - feedback_start_time
        
        forward_total_time.append(forward_time)
        feedback_total_time.append(feedback_time)

        # 7. 获取结果
        # 结果存储在返回的 state 对象中，通过 sgl.gen() 中定义的变量名 ("result") 访问。
        print(f"Query: {query}")
        print("Answer 1:", state1["result"])
        print("Answer 2:", state2["result"])
        print(f"Time - Forward: {forward_time:.4f}s, Feedback: {feedback_time:.4f}s")
        print("-" * 20)

    # 计算平均时间 (与原代码相同)
    if data_list:
        avg_forward = sum(forward_total_time) / len(forward_total_time)
        avg_feedback = sum(feedback_total_time) / len(feedback_total_time)
        print("\n--- Average Times ---")
        print(f"Average Forward Time: {avg_forward:.4f} seconds")
        print(f"Average Feedback Time: {avg_feedback:.4f} seconds")
        
    # 关闭 runtime (可选，但在脚本结束时是好习惯)
    runtime.shutdown()

if __name__ == "__main__":
    main()

# python -m llmragenv.sglang_test
from vllm import LLM, SamplingParams

import json
import time
from chat.chat_prompt import *

from logger import Logger

def checkanswer(prediction, ground_truth, verbose=False): # 特别针对RGB这种有多个答案的
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))

    return labels

def parse_vllm_metrics(metrics):
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
    # 端到端时间 推理总时长 TTFT 解码时长 排队延时


# 1. 启动 vLLM 模型
# llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
def main():
    # llm = LLM(model="/home/zhangyz/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a")
    # llm = LLM(model="/home/hdd/model/Qwen2.5-32B-Instruct")

    llm = LLM(
        model="/home/hdd/model/Qwen2.5-32B-Instruct",
        # model="/home/hdd/model/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        # disable_custom_all_reduce=True,
        # enforce_eager=True,
        # swap_space=128,
        # tokenizer_mode="auto",
    )


    # 3. 定义采样参数
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



    # 2. 加载数据
    try:
        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_evolve_basic_forward_evolve_shared_prefix_rgb_zyz_0.json", 'r', encoding='utf-8') as file:
            data_list = json.load(file)
            # data_list = data_list[0:10]
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

    forward_generate_time_total = []
    forward_prefill_time_total = []
    forward_decode_time_total = []
    forward_wait_scheduled_time_total = []
    forward_end2end_time_total = []

    feedback_generate_time_total = []
    feedback_prefill_time_total = []
    feedback_decode_time_total = []
    feedback_wait_scheduled_time_total = []
    feedback_end2end_time_total = []
    logger = Logger("./test/vllm_test")
    data = []
    TF_lable = []

    # 3. 循环处理数据，保持您原有的计时逻辑
    for index, data_item in enumerate(data_list):
        # if index >= 1:
        #     break
        logger.log(f"--- index : {index} ---")
        print(f"--- Processing item {index+1}/{len(data_list)} ---")
        query = data_item['query']
        answer = data_item['answer']
        filtered_retrieve_result = data_item['filtered_retrieve_result']

        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context += f"Path {idx}:\t{sentence}\n"
        
        # 准备 prompts
        prefix = shared_prefix.format(knowledge_sequences=context)
        query_prompt_user = chat_with_graphrag_for_response_user_shared_prefix.format(question=query)
        # feedback_prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question=query, last_response=response)
        

        
        # 4. 第一次请求：构建 prefix cache
        forward_start_time = time.perf_counter()
        outputs1 = llm.generate(
            # [prefix+query_prompt_user],
            # [llama_instruct_system.format(system_instruction = prefix) + llama_instruct_user.format(user_instruction = query_prompt_user)],
            [qwen2_instruct_system.format(system_instruction = prefix) + qwen2_instruct_user.format(user_instruction = query_prompt_user)],
            # "who are you?",
            params,
            # use_prefix_cache=True  # 开启前缀缓存
        )
        forward_end_time = time.perf_counter()
        response = outputs1[0].outputs[0].text
        data_item['response'] = response
        logger.log(f"Query: {query}")
        logger.log(f"Ground Truth: {answer}")
        logger.log(f"Answer: {response}")
        flag_label = checkanswer(response, answer)
        flag_TF = sum(flag_label) == len(flag_label)
        TF_lable.append(flag_TF)
        data_item['label'] = flag_TF
        logger.log(f"Label: {flag_TF}")
        

        prompt_len = len(outputs1[0].prompt_token_ids)
        generate_len = len(outputs1[0].outputs[0].token_ids)
        end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time= (
            parse_vllm_metrics(outputs1[0].metrics)
        )
        logger.log(
            f"prompt_len {prompt_len} generate_len {generate_len}\nend2end_time {end2end_time:.3f} generate_time {generate_time:.3f} prefill_time {prefill_time:.3f} decode_time {decode_time:.3f} wait_scheduled_time {wait_scheduled_time:.3f}\n"
        )

        forward_end2end_time_total.append(end2end_time)
        forward_generate_time_total.append(generate_time)
        forward_prefill_time_total.append(prefill_time)
        forward_decode_time_total.append(decode_time)
        forward_wait_scheduled_time_total.append(wait_scheduled_time)


        # feedback_prompt_user = score_feedback_prompt_standard_user_test_optimized.format(question=query, last_response=response)
        feedback_prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question=query, last_response=response)

        feedback_start_time = time.perf_counter()
        # 5. 第二次请求：复用 prefix cache（vLLM 自动识别相同前缀）
        outputs2 = llm.generate(
            # [llama_instruct_system.format(system_instruction = prefix) + llama_instruct_user.format(user_instruction = feedback_prompt_user)],
            [qwen2_instruct_system.format(system_instruction = prefix) + qwen2_instruct_user.format(user_instruction = feedback_prompt_user)],
            params_2,
            # use_prefix_cache=True
        )
        feedback_end_time = time.perf_counter()

        feedback_response = outputs2[0].outputs[0].text
        data_item['feedback_response'] = feedback_response
        logger.log(f"retrieve path:\n{context}")
        logger.log(f"Feedback Answer: {feedback_response}")
        prompt_len = len(outputs2[0].prompt_token_ids)
        generate_len = len(outputs2[0].outputs[0].token_ids)
        end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = (
            parse_vllm_metrics(outputs2[0].metrics)
        )
        logger.log(
            f"prompt_len {prompt_len} generate_len {generate_len}\nend2end_time {end2end_time:.3f} generate_time {generate_time:.3f} prefill_time {prefill_time:.3f} decode_time {decode_time:.3f} wait_scheduled_time {wait_scheduled_time:.3f}\n"
        )
        
        # forward_time = forward_end_time - forward_start_time
        # feedback_time = feedback_end_time - feedback_start_time
        # forward_total_time.append(forward_time)
        # feedback_total_time.append(feedback_time)

        feedback_end2end_time_total.append(end2end_time)
        feedback_generate_time_total.append(generate_time)
        feedback_prefill_time_total.append(prefill_time)
        feedback_decode_time_total.append(decode_time)
        feedback_wait_scheduled_time_total.append(wait_scheduled_time)


        # print(f"Query: {query}")
        # print("Answer 1:", outputs1[0].outputs[0].text)
        # print("Answer 2:", outputs2[0].outputs[0].text)
        # print("-" * 20)
        data.append(data_item)


    # 4. 计算平均时间
    if data_list:
        # avg_forward = sum(forward_total_time) / len(forward_total_time)
        # avg_feedback = sum(feedback_total_time) / len(feedback_total_time)
        # print("\n--- Average Times ---")
        # print(f"Average Forward Time: {avg_forward:.4f} seconds")
        # print(f"Average Feedback Time: {avg_feedback:.4f} seconds")

        logger.log(f"\n--- Forward Average Times ---\n forward_end2end_time_total:{sum(forward_end2end_time_total) / len(forward_end2end_time_total):.4f} seconds\n forward_generate_time_total:{sum(forward_generate_time_total) / len(forward_generate_time_total):.4f} seconds\n forward_prefill_time_total:{sum(forward_prefill_time_total) / len(forward_prefill_time_total):.4f} seconds\n forward_decode_time_total:{sum(forward_decode_time_total) / len(forward_decode_time_total):.4f} seconds\n forward_wait_scheduled_time_total:{sum(forward_wait_scheduled_time_total) / len(forward_wait_scheduled_time_total):.4f} seconds\n")
        logger.log(f"--- Feedback Average Times ---\n feedback_end2end_time_total:{sum(feedback_end2end_time_total) / len(feedback_end2end_time_total):.4f} seconds\n feedback_generate_time_total:{sum(feedback_generate_time_total) / len(feedback_generate_time_total):.4f} seconds\n feedback_prefill_time_total:{sum(feedback_prefill_time_total) / len(feedback_prefill_time_total):.4f} seconds\n feedback_decode_time_total:{sum(feedback_decode_time_total) / len(feedback_decode_time_total):.4f} seconds\n feedback_wait_scheduled_time_total:{sum(feedback_wait_scheduled_time_total) / len(feedback_wait_scheduled_time_total):.4f} seconds\n")
        logger.log(f"--- Total Accuracy ---\n {sum(TF_lable)}/{len(TF_lable)} {sum(TF_lable) / len(TF_lable):.4f}\n")

    with open(f"./logs/stage/vllm_test.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()


# python -m llmragenv.vllm

# llama
# Time - Forward: 0.5247s, Feedback: 11.5081s
# Time - Forward: 0.5261s, Feedback: 12.0195s
# qwen


# 原提示词加速
# --- Average Times ---
# Average Forward Time: 2.3293 seconds
# Average Feedback Time: 47.3767 seconds


# 新提示词不加速
# --- Average Times ---
# Average Forward Time: 2.2041 seconds
# Average Feedback Time: 4.1030 seconds

# --- Average Times ---
# Average Forward Time: 1.9103 seconds
# Average Feedback Time: 4.0683 seconds

# Time - Forward: 1.9042s, Feedback: 4.0330s

# prompt_len 3471 generate_len 11
# generate_time 1.833 prefill_time 1.305 decode_time 0.528 wait_scheduled_time 0.018

# prompt_len 4031 generate_len 48
# generate_time 3.970 prefill_time 1.520 decode_time 2.450 wait_scheduled_time 0.011

# 新提示词加速
# --- Average Times ---
# Average Forward Time: 2.2004 seconds
# Average Feedback Time: 2.8494 seconds

# --- Average Times ---
# Average Forward Time: 2.4041 seconds
# Average Feedback Time: 2.9153 seconds

# --- Average Times ---
# Average Forward Time: 1.9062 seconds
# Average Feedback Time: 2.8274 seconds

# 批量去加速？？？


# 简单测试10个
# 未加速
# --- Forward Average Times ---
#  forward_end2end_time_total:1.8445 seconds
#  forward_generate_time_total:1.8284 seconds
#  forward_prefill_time_total:1.3090 seconds
#  forward_decode_time_total:0.5194 seconds
#  forward_wait_scheduled_time_total:0.0162 seconds

# --- Feedback Average Times ---
#  feedback_end2end_time_total:3.9721 seconds
#  feedback_generate_time_total:3.9628 seconds
#  feedback_prefill_time_total:1.5171 seconds
#  feedback_decode_time_total:2.4458 seconds
#  feedback_wait_scheduled_time_total:0.0093 seconds

# 加速
# --- Forward Average Times ---
#  forward_end2end_time_total:1.5866 seconds
#  forward_generate_time_total:1.5775 seconds
#  forward_prefill_time_total:1.0283 seconds
#  forward_decode_time_total:0.5492 seconds
#  forward_wait_scheduled_time_total:0.0091 seconds

# --- Feedback Average Times ---
#  feedback_end2end_time_total:3.1309 seconds
#  feedback_generate_time_total:3.1230 seconds
#  feedback_prefill_time_total:0.2704 seconds
#  feedback_decode_time_total:2.8526 seconds
#  feedback_wait_scheduled_time_total:0.0079 seconds

# --- Forward Average Times ---
#  forward_end2end_time_total:1.6571 seconds
#  forward_generate_time_total:1.6475 seconds
#  forward_prefill_time_total:1.0429 seconds
#  forward_decode_time_total:0.6046 seconds
#  forward_wait_scheduled_time_total:0.0096 seconds

# --- Feedback Average Times ---
#  feedback_end2end_time_total:3.2814 seconds
#  feedback_generate_time_total:3.2735 seconds
#  feedback_prefill_time_total:0.2702 seconds
#  feedback_decode_time_total:3.0033 seconds
#  feedback_wait_scheduled_time_total:0.0079 seconds

# --- Forward Average Times ---
#  forward_end2end_time_total:1.6039 seconds
#  forward_generate_time_total:1.5917 seconds
#  forward_prefill_time_total:1.0371 seconds
#  forward_decode_time_total:0.5546 seconds
#  forward_wait_scheduled_time_total:0.0122 seconds

# --- Feedback Average Times ---
#  feedback_end2end_time_total:3.1716 seconds
#  feedback_generate_time_total:3.1611 seconds
#  feedback_prefill_time_total:0.2738 seconds
#  feedback_decode_time_total:2.8873 seconds
#  feedback_wait_scheduled_time_total:0.0105 seconds
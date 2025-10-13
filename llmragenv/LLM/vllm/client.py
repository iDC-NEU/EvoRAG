
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import re
from typing import List, overload
from overrides import override
from llama_index.core.utils import print_text

from llmragenv.LLM.llm_base import LLMBase
from logger import Logger

from vllm import LLM, SamplingParams
from utils.config import path_config

import json
import time

import time
import json
from chat.chat_prompt import llama_instruct_system, llama_instruct_user, qwen2_instruct_system, qwen2_instruct_user, qwen3_instruct_system, qwen3_instruct_user_noThink, qwen3_instruct_user_think

class Vllm(LLMBase):

    def __init__(self, model_name, url, key):
        super().__init__()



        llm_config = path_config["LLM"]['vllm']
        if model_name in llm_config:
            self.type = llm_config[model_name]['template_format']
            if llm_config[model_name]['modelpath']:
                self.path = llm_config[model_name]['modelpath']
            else:
                self.path = model_name
        else:
            raise ValueError(f"模型 {model_name} 未在配置文件中找到，请检查 path-local.yaml 配置项。")

        self.model_name  = model_name
        self.llmbackend = "vllm"
        # if model_name == 'Meta-Llama-3-8B-Instruct':
        #     self.type = 'Llama'
        #     # self.model_name_url = 'meta-llama/Meta-Llama-3-8B-Instruct'
        #     self.path = "/home/zhangyz/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
        # elif model_name == 'Llama-3.1-8B-Instruct':
        #     self.type = 'Llama'
        #     self.path = "/home/hdd/model/Llama-3.1-8B-Instruct"
        # elif model_name == 'Qwen2.5-32B-Instruct':
        #     self.type = 'Qwen2.5'
        #     # self.model_name_url = 'Qwen/Qwen2.5-32B-Instruct'
        #     self.path = "/home/zhangyz/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd" # 修改了
        #     # self.path = "/home/hdd/model/Qwen2.5-32B-Instruct"
        # elif model_name == 'Qwen2.5-72B-Instruct-GPTQ-Int8': # 没有下载完
        #     # self.type = 'Qwen2.5'
        #     # self.model_name_url = 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'
        #     self.path = "/home/hdd2/zhangyz/huggingface_local/models--Qwen--Qwen2.5-72B-Instruct-GPTQ-Int8/snapshots/4f953f7634fef56affc59a40656ebe7461f7e545"
        # elif model_name == 'Qwen3-32B':
        #     self.type = 'Qwen3'
        #     # self.model_name_url = 'Qwen/Qwen3-32B'
        #     self.path = "/home/hdd/zhangyz/huggingface_local/models--Qwen--Qwen3-32B/models--Qwen--Qwen3-32B"  # 修改了
        base_url = re.sub(r'/v\d+', '', url)


        self.llm = LLM(
            model=self.path,
            # model="/home/hdd/model/Meta-Llama-3-8B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.95,
            enable_prefix_caching=True,
            # disable_custom_all_reduce=True,
            # enforce_eager=True,
            swap_space=64,
            # tokenizer_mode="auto",
        )
        self.params = SamplingParams(
            temperature=0.0, 
            max_tokens=256,
            stop=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        )

        print("Use vllm backend to generate")
        if self.type == 'Qwen2.5' or self.type == 'Qwen3':
            response = self.chat_with_ai("<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant")
            # response = self.chat_with_ai("who are you?")
        elif self.type == 'Llama':
            response = self.chat_with_ai("<|start_header_id|>user<|end_header_id|>\nwho are you?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>")
            # response = self.chat_with_ai("who are you?")
        print_text(f"\n test huggingface model {self.model_name} : {response}\n", color='yellow')

    def parse_vllm_metrics(self, metrics):
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


        
    @override
    def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        outputs = self.llm.generate(
            prompt,
            self.params,
        )
        prompt_len = len(outputs[0].prompt_token_ids)
        generate_len = len(outputs[0].outputs[0].token_ids)   
        response = outputs[0].outputs[0].text
        end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = (
            self.parse_vllm_metrics(outputs[0].metrics)
        )

        return response, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time
       
    @override
    def chat_with_ai_with_system(self, prompt_system: str | List[str], prompt_user: str | List[str], history: List[List[str]] | None = None, enable_thinking = False) -> str | None:
        # assert False, "HuggingfaceClient does not support chat_with_ai_with_system method. Please use chat_with_ai instead."
        if isinstance(prompt_system, list):
            prompt_system_user_instruct = []
            for prompt_system_item, prompt_user_item in zip(prompt_system, prompt_user):
                if self.type == 'Llama':
                    prompt_system_user_instruct.append(llama_instruct_system.format(system_instruction = prompt_system_item)+llama_instruct_user.format(user_instruction = prompt_user_item))
                elif self.type == 'Qwen2.5':
                    prompt_system_user_instruct.append(qwen2_instruct_system.format(system_instruction = prompt_system_item)+qwen2_instruct_user.format(user_instruction = prompt_user_item))
                elif self.type == 'Qwen3':
                    if enable_thinking:
                        prompt_system_user_instruct.append(qwen3_instruct_system.format(system_instruction = prompt_system_item)+qwen3_instruct_user_think.format(user_instruction = prompt_user_item))
                    else:
                        prompt_system_user_instruct.append(qwen3_instruct_system.format(system_instruction = prompt_system_item)+qwen3_instruct_user_noThink.format(user_instruction = prompt_user_item))
            return self.chat_with_ai_batch(prompt_system_user_instruct)
        else:
            if self.type == 'Llama':
                prompt_system_instruct = llama_instruct_system.format(system_instruction = prompt_system)
                prompt_user_instruct = llama_instruct_user.format(user_instruction = prompt_user)
            elif self.type == 'Qwen2.5':
                prompt_system_instruct = qwen2_instruct_system.format(system_instruction = prompt_system)
                prompt_user_instruct = qwen2_instruct_user.format(user_instruction = prompt_user)
            elif self.type == 'Qwen3':
                if enable_thinking:
                    prompt_system_instruct = qwen3_instruct_system.format(system_instruction = prompt_system)
                    prompt_user_instruct = qwen3_instruct_user_think.format(user_instruction = prompt_user)
                else:
                    prompt_system_instruct = qwen3_instruct_system.format(system_instruction = prompt_system)
                    prompt_user_instruct = qwen3_instruct_user_noThink.format(user_instruction = prompt_user)
            return self.chat_with_ai(prompt_system_instruct + prompt_user_instruct, history)  

    def chat_with_ai_batch(self, prompt: List[str], history: List[List[str]] | None = None) -> str | None:
        start_time = time.time()
        outputs = self.llm.generate(
            prompt,
            self.params,
        )
        response = []
        prompt_len = []
        end_time = time.time()
        for index in range(len(outputs)):
            response.append(outputs[index].outputs[0].text)
            prompt_len.append(len(outputs[index].prompt_token_ids))
                  
        return response, prompt_len, end_time - start_time


    @override
    def chat_with_ai_stream(self, prompt: str, history: List[List[str]] | None = None):
        
        return self.client.stream_complete(prompt)
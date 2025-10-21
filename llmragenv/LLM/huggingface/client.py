import re
from typing import List, overload
from overrides import override
from llama_index.core.utils import print_text

from llmragenv.LLM.llm_base import LLMBase
from logger import Logger
from utils.config import path_config

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import (  # PreTrainedModel,; set_seed,; AutoConfig
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
)
from transformers.models.llama.modeling_llama import (  # LlamaRotaryEmbedding, LlamaConfig,
    LlamaForCausalLM,
)

import time
import json
from chat.chat_prompt import llama_instruct_system, llama_instruct_user, qwen2_instruct_system, qwen2_instruct_user, qwen3_instruct_system, qwen3_instruct_user_noThink, qwen3_instruct_user_think

class HuggingfaceClient(LLMBase):

    def __init__(self, model_name, url, key):
        super().__init__()


        llm_config = path_config["LLM"]['huggingface']
        if model_name in llm_config:
            self.type = llm_config[model_name]['template_format']
            self.path = llm_config[model_name]['modelpath']
            self.access_token = llm_config[model_name]['access_token']
        else:
            raise ValueError(f"模型 {model_name} 未在配置文件中找到，请检查 path-local.yaml 配置项。")
        
        self.model_name  = model_name
        self.llmbackend = "huggingface"
        # if model_name == 'Meta-Llama-3-8B-Instruct':
        #     self.type = 'Llama'
        #     self.model_name_url = 'meta-llama/Meta-Llama-3-8B-Instruct'
        #     self.path = "/home/zhangyz/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
        # elif model_name == 'Meta-Llama-3.1-8B-Instruct':
        #     self.type = 'Llama'
        #     self.model_name_url = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        #     self.path = "/home/hdd/model/Llama-3.1-8B-Instruct"
        # elif model_name == 'Qwen2.5-32B-Instruct':
        #     self.type = 'Qwen2.5'
        #     self.model_name_url = 'Qwen/Qwen2.5-32B-Instruct'
        #     self.path = "/home/zhangyz/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd" # 修改了
        #     # self.path = "/home/hdd/model/Qwen2.5-32B-Instruct"
        # elif model_name == 'Qwen2.5-72B-Instruct-GPTQ-Int8': # 没有下载完
        #     self.type = 'Qwen2.5'
        #     self.model_name_url = 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'
        #     self.path = "/home/hdd2/zhangyz/huggingface_local/models--Qwen--Qwen2.5-72B-Instruct-GPTQ-Int8/snapshots/4f953f7634fef56affc59a40656ebe7461f7e545"
        # elif model_name == 'Qwen3-32B':
        #     self.type = 'Qwen3'
        #     self.model_name_url = 'Qwen/Qwen3-32B'
        #     self.path = "/home/hdd/zhangyz/huggingface_local/models--Qwen--Qwen3-32B/models--Qwen--Qwen3-32B"  # 修改了
        base_url = re.sub(r'/v\d+', '', url)

        self.device = "cuda:1"

        torch.cuda.set_device(self.device)
        # if self.model_name == 'Meta-Llama-3-8B-Instruct' or self.model_name == 'Meta-Llama-3.1-8B-Instruct':
        if self.model_name == 'Llama-3-8B-Instruct' or self.model_name == 'Llama-3.1-8B-Instruct':
            self.client: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
                # self.model_name,
                pretrained_model_name_or_path=self.path,
                # pretrained_model_name_or_path=model_name,
                dtype=torch.bfloat16,
                # torch_dtype=torch.bfloat16,
                # device_map=f"cuda:{device}",
                device_map=self.device,
                # device_map="auto", # 自动划分在多个GPU
                attn_implementation="sdpa",
                token=self.access_token,
            )   
        else:
            self.client: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
                # self.model_name,
                pretrained_model_name_or_path=self.path,
                # pretrained_model_name_or_path=model_name,
                # torch_dtype='auto',
                dtype='auto',
                # device_map=f"cuda:{device}",
                # device_map=self.device
                device_map="auto", # 自动划分在多个GPU
                attn_implementation="sdpa",
                token=self.access_token,
            )    
        # ).cuda()
        # attn_implementation="flash_attention_2")
        self.client.eval()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            # pretrained_model_name_or_path=self.model_name, 
            pretrained_model_name_or_path=self.path,
            # use_fast=False,
            token=self.access_token,
        )

        print("Use huggingface backend to generate")
        if self.type == 'Qwen2.5' or self.type == 'Qwen3':
            response = self.chat_with_ai("<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant")
            # response = self.chat_with_ai("who are you?")
        elif self.type == 'Llama':
            response = self.chat_with_ai("<|start_header_id|>user<|end_header_id|>\nwho are you?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>")
            # response = self.chat_with_ai("who are you?")
        print_text(f"\n test huggingface model {self.model_name} : {response}\n", color='yellow')

        
    @override
    def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        start_time_total = time.perf_counter()
        max_tokens = 32768
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        num_input_tokens = len(input_ids)

        input_ids = torch.tensor(data=[input_ids], dtype=torch.int64).cuda()
        
        # inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # input_ids = inputs["input_ids"].to(self.client.device)

        input_length = input_ids.size(-1)

        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )

        generation_config = GenerationConfig(
            do_sample=False,
            # temperature=0,
            # do_sample = True,
            # temperature = 0.6,
            # min_p = 0.0,
            # top_p = 0.95,
            # top_k = 20,
            repetition_penalty=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            # stop_strings=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        )

        # 记录模型生成起始时间
        start_time_generation = time.perf_counter()

        outputs = self.client.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            use_cache=True,
            # eos_token_id=[self.tokenizer.eos_token_id],
            # eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            tokenizer=self.tokenizer,
        )

        # 记录模型生成结束时间
        end_time_generation = time.perf_counter()
        generation_time_ms = (end_time_generation - start_time_generation) * 1000
        num_generated_tokens = len(outputs[0][input_length:].tolist())
        num_generated_thinking_tokens = None
        if self.model_name == "Qwen/Qwen3-32B":
            new_tokens = outputs[0][input_length:].tolist() 
            try:
                # rindex finding 151668 (</think>)
                index = len(new_tokens) - new_tokens[::-1].index(151668)
            except ValueError:
                index = 0
            num_generated_thinking_tokens = len(new_tokens[:index])
            thinking_content = self.tokenizer.decode(new_tokens[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(new_tokens[index:], skip_special_tokens=True).strip("\n")

            # thinking_content = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip("\n")
            # content = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip("\n")

            print(f"----------chat_with_ai----------- thinking_content\n{thinking_content}")
            print(f"----------chat_with_ai----------- content\n{content}")
            # return content
        else:
            content = self.tokenizer.decode(token_ids=outputs[0][input_length:].tolist()).replace("<|eot_id|>", "").replace("<|im_end|>", "").strip()
        
        end_time_total = time.perf_counter()
        response_time = (end_time_total - start_time_total)

        # result_info = {
        #     "generated_text": content,
        #     "thinking_content": thinking_content if self.model_name == "Qwen/Qwen3-32B" else None, # 仅Qwen3时有意义
        #     "num_input_tokens": num_input_tokens, # 只看输入就够用了
        #     "num_generated_tokens": num_generated_tokens,
        #     "num_generated_thinking_tokens": num_generated_thinking_tokens,
        #     "response_time_ms": round(response_time_ms, 2),
        #     "generation_time_ms": round(generation_time_ms, 2),
        #     "tokens_per_second": round(num_generated_tokens / (generation_time_ms / 1000), 2) if generation_time_ms > 0 else 0,
        # }
        # print(f"chat_with_ai: {json.dumps(result_info, indent=2)}")
        # query_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
        # if prompt.startswith(query_start):
        #     return content, num_input_tokens, response_time
                
        return content # .replace("<|im_end|>", "")
    
    @override
    def chat_with_ai_with_system(self, prompt_system: str, prompt_user: str, history: List[List[str]] | None = None, enable_thinking = False) -> str | None:
        # assert False, "HuggingfaceClient does not support chat_with_ai_with_system method. Please use chat_with_ai instead."
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


    
    def chat_with_ai_multi_round(self, prompt: str, history: List[List[str]] | None = None) -> str | None:

        # conversation = []
        # for user_msg, assistant_msg in history:
        #     conversation.append({"role": "user", "content": user_msg})
        #     conversation.append({"role": "assistant", "content": assistant_msg})
        # # 添加当前这轮的用户提示
        # conversation.append({"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True, # 非常重要，用于指示模型生成下一轮回复
                tokenize=True,              # 需要返回 token ID 而不是格式化后的字符串
                return_tensors="pt"         # 返回 PyTorch 张量
                # enable_thinking=False
            )
            # ).to(self.device)

        max_tokens = 32768
        # input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # input_ids = torch.tensor(data=[input_ids], dtype=torch.int64).cuda()
        input_length = input_ids.size(-1)

        generation_config = GenerationConfig(
            do_sample=False,
            # do_sample = True,
            temperature = 0.6,
            min_p = 0.0,
            top_p = 0.95,
            top_k = 20,
            repetition_penalty=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            max_new_tokens=max_tokens,
            stop_strings=["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        )

        with torch.no_grad(): # 在推理(inference)阶段不需要计算梯度，可以节省显存和计算资源
            outputs = self.client.generate(
                input_ids=input_ids,            # 输入的 token ID 张量
                generation_config=generation_config, # 生成参数配置
                use_cache=True,                 # 启用 KV 缓存以加快长对话的生成速度
                # 通常不需要在这里传递 tokenizer 参数了，因为输入已经 token 化
                tokenizer=self.tokenizer,
                
            )
        new_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if self.model_name == "Qwen/Qwen3-32B":

            new_tokens = outputs[0][input_length:].tolist() 
            try:
                # rindex finding 151668 (</think>)
                index = len(new_tokens) - new_tokens[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(new_tokens[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(new_tokens[index:], skip_special_tokens=True).strip("\n")

            print(f"---------chat_with_ai_multi_round--------- thinking_content\n{thinking_content}")
            print(f"---------chat_with_ai_multi_round--------- content\n{content}")
            return content

        return response_text.replace("<|im_end|>", "").strip()
    
    # @override
    # def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        
    #     response = self.client.complete(prompt)
        
    #     return response.text

    @override
    def chat_with_ai_stream(self, prompt: str, history: List[List[str]] | None = None):
        
        return self.client.stream_complete(prompt)
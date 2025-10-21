'''
Author: lpz 1565561624@qq.com
Date: 2024-09-17 08:50:26
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-04 10:29:25
FilePath: /lipz/fzb_rag_demo/RAGWebUi_demo/llmragenv/LLM/ollama/client.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import re
from typing import List, overload
from llama_index.llms.ollama import Ollama
# from llama_index.core.llms.types import ChatMessage
from llama_index.core.llms import ChatMessage
# from llama_index.llms.types import ChatMessage

from overrides import override
from llama_index.core.utils import print_text

from llmragenv.LLM.llm_base import LLMBase
from logger import Logger
from chat.chat_prompt import llama_instruct_system, llama_instruct_user, qwen2_instruct_system, qwen2_instruct_user, qwen3_instruct_system, qwen3_instruct_user_noThink, qwen3_instruct_user_think


class OllamaClient(LLMBase):

    def __init__(self, model_name, url, key):
        super().__init__()
        self.model_name  = model_name
        self.llmbackend = "ollama"
        base_url = re.sub(r'/v\d+', '', url)

        self.client = Ollama(model=model_name, request_timeout=2000000, base_url=base_url)

        # self.logger = Logger("AIClient")
        # self.logger.info("Use llama_index backend to generate")
        print("Use llama_index backend to generate")
        
        response = self.chat_with_ai("who are you?")
        # response = self.complete("who are you?")
        # print(response)
        # assert False
        print_text(f"\n test llm model {self.model_name} : {response}\n", color='yellow')

        
    @override
    def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        
        response = self.client.complete(prompt)
        
        return response.text
    
    @override
    def chat_with_ai_with_system(self, prompt_system: str, prompt_user: str, history: List[List[str]] | None = None, enable_thinking = False) -> str | None:
        # assert False, "HuggingfaceClient does not support chat_with_ai_with_system method. Please use chat_with_ai instead."
        return self.chat_with_ai(prompt_system + prompt_user, history)

    @override
    def chat_with_ai_mulRounds(self, history: list[dict]) -> str:

        if history and isinstance(history[0], dict):
            # 转换成 ChatMessage
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in history]
        else:
            messages = history
    
        response = self.client.chat(messages)
        assistant_reply = response.message.content
        return assistant_reply
    
    @override
    def chat_with_ai_stream(self, prompt: str, history: List[List[str]] | None = None):
        
        return self.client.stream_complete(prompt)
    
    def chat_with_messages(self, messages) -> str:

        response = self.client.chat(messages)
        # print(response.raw)
        return response.message.content, response.raw.get("prompt_eval_count", 0)
    


    


# class OllamaClient(LLMBase):

#     def __init__(self, model_name, url, key):
#         super().__init__()
#         self.model_name  = model_name
#         base_url = re.sub(r'/v\d+', '', url)

#         self.client = Ollama(model=model_name, request_timeout=2000000, base_url=base_url)

#         # self.logger = Logger("AIClient")
#         # self.logger.info("Use llama_index backend to generate")
#         print("Use llama_index backend to generate")
        
#         response = self.chat_with_ai("who are you?")
#         print_text(f"\n test llm model {self.model_name} : {response}\n", color='yellow')

        
#     @override
#     def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        
#         response = self.client.complete(prompt)
        
#         return response.text

#     @override
#     def chat_with_ai_stream(self, prompt: str, history: List[List[str]] | None = None):
        
#         return self.client.stream_complete(prompt)


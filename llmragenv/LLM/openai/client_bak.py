'''
Author: fzb fzb0316@163.com
Date: 2024-09-20 13:37:09
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-04 14:57:49
FilePath: /RAGWebUi_demo/llmragenv/LLM/client_generic.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import json
from typing import List, Dict, Tuple

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from llmragenv.LLM.llm_base import LLMBase
from overrides import override
from logger import Logger

class OpenAIClient(LLMBase):
    
    # _logger: Logger = Logger("AIClient")

    def __init__(self, model_name, url, key):
        super().__init__()
        self.client = OpenAI(
            api_key = key,
            base_url = url,
        )
        self.model_name  = model_name
        # models = self.client.models.list()

        # for model in models.data:
        #     print(model.id)
        
        # Logger.info(f"Use openai backend to generate")
        print(f"Use openai backend to generate")
        response = self.chat_with_ai("who are you?")
        print(f"\n test openai model {self.model_name} : {response}\n")


        assert(False)
        # for i in self.client.models.list().data:
        #     print(i)
    

    def construct_messages(self, prompt: str, history: List[List[str]]) -> List[Dict[str, str]]:
        messages = []
            # {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的回答。"}]

        for user_input, ai_response in history:
            messages.append({"role": "user", "content": user_input})
            messages.append(
                {"role": "assistant", "content": ai_response.__repr__()})

        messages.append({"role": "user", "content": prompt})
        return messages
    
    def chat_with_ai_multi_round(self, prompt: str, history: List[Dict[str, str]]):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            # top_p=0.7,
            # temperature=0.95,
            temperature=0,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content

    @override
    def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.construct_messages(prompt, history if history else []),
            # top_p=0.7,
            # temperature=0.95,
            temperature=0,
            max_tokens=1024,
        )

        return response.choices[0].message.content

    @override
    def chat_with_ai_stream(self, prompt: str, history: List[List[str]] | None = None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.construct_messages(prompt, history if history else []),
            top_p=0.7,
            temperature=0.95,
            max_tokens=1024,
            stream=True,
        )

        
        result = ""
        for chunk in response:
            result =  result + chunk.choices[0].delta.content or ""

            yield result



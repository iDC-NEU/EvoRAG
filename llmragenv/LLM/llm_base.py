'''
Author: fzb fzb0316@163.com
Date: 2024-09-20 13:37:09
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-25 18:04:25
FilePath: /RAGWebUi_demo/llmragenv/LLM/client_generic.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import json
from typing import List, Dict, Tuple

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from utils.singleton import Singleton
from abc import abstractmethod

# metaclass=Singleton
class LLMBase():
    """
    LLMBase is a generic LLM client that can be used to interact with any language model. But if you want to
    specify the model name, you can extend ClientBase.
    """

    def __init__(self):
        pass

    @abstractmethod
    def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
        raise NotImplementedError()

    @abstractmethod
    def chat_with_ai_stream(self, prompt: str, history: List[List[str]] | None = None):
        raise NotImplementedError()
        

   
   




'''
Author: fzb fzb0316@163.com
Date: 2024-09-16 18:35:02
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-20 15:39:46
FilePath: /RAGWebUi_demo/chat/chat_base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



from abc import abstractmethod
from typing import List, Optional
from llmragenv.LLM.llm_base import LLMBase



class ChatBase((object)):
    def __init__(self, llm : LLMBase):
        self._llm = llm

    
    @abstractmethod
    def web_chat(self, message: str, history: List[Optional[List]] | None):
        raise NotImplementedError()
    
    @abstractmethod
    def retrieval_result(self):
        raise NotImplementedError()
    
    @abstractmethod
    def chat_without_stream(self, message: str):
        raise NotImplementedError()
    

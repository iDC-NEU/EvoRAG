'''
Author: fzb fzb0316@163.com
Date: 2024-09-15 21:00:58
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-25 13:32:01
FilePath: /RAGWebUi_demo/chat/chat_withoutrag.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''




from icecream import ic
from typing import List, Optional
from overrides import override


from chat.chat_base import ChatBase
from llmragenv.LLM.llm_base import LLMBase

class ChatWithoutRAG(ChatBase):

    def __init__(self, llm: LLMBase):
        super().__init__(llm)


    @override
    def retrieval_result(self):
        return None


    @override
    def web_chat(self, message: str, history: List[Optional[List]] | None):
        
        ic(message)
        ic(history)

        answers = self._llm.chat_with_ai_stream(message, history)
        result = ""
        for chunk in answers:
            result =  result + chunk.choices[0].delta.content or ""
            yield result

        ic(result)
        # return result

    
    @override
    def chat_without_stream(self, message: str):
        return self._llm.chat_with_ai(message)
    

    def chat_without_stream_with_llamaindex(self, message: str):
        return self._llm.chat_with_ai(message)



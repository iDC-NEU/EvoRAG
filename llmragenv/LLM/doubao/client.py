'''
Author: fzb0316 fzb0316@163.com
Date: 2024-09-21 19:23:18
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-25 10:46:11
FilePath: /BigModel/RAGWebUi_demo/llmragenv/LLM/doubao/client.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from llmragenv.LLM.openai.client import OpenAIClient


class DoubaoClient(OpenAIClient):
    """
    豆包 AI Client
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

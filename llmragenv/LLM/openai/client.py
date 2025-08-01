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
import requests
import json

def get_access_token():
    """
    使用应用API Key，应用Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=OlvUN2cqFVeQDGoB6N36Gh2f&client_secret=T71K1J8xzDaRrpMQK8ckFDY5lfeFXjAy"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

class OpenAIClient(LLMBase):
    
    # _logger: Logger = Logger("AIClient")

    def __init__(self, model_name, url, key):
        super().__init__()
        if model_name == 'llama_3_8b':
            self.model_name = 'llama_3_8b'
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_8b?access_token=" + get_access_token()
            print(f"Use openai backend to generate")
            response = self.chat_with_ai("who are you?")
            print(f"\n test openai model {self.model_name} : {response}\n")
            
            # assert False
        elif model_name == 'qwen3_32b':
            self.model_name = 'qwen3_32b'

            self.client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                # api_key=os.getenv("DASHSCOPE_API_KEY"),
                api_key="sk-9d5825d8cc254d0ea03ceab8b8241a18",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            print(f"Use openai backend to generate")
            response = self.chat_with_ai("who are you?")
            print(f"\n test openai model {self.model_name} : {response}\n")
            
        elif model_name == 'meta-llama-3-8b':
        
            self.client = OpenAI(
                api_key = "bce-v3/ALTAK-lGkipQ7AFP2oP2uq3Z5II/e816c8829f4f6ad0b165fe71bb5283831db2536b",
                base_url = "https://qianfan.baidubce.com/v2",
            )
            print("No longer supported meta-llama-3-8b(openai:client.py)")
            assert False
        else:
            print("A model that does not exist in OpenAI(openai:client.py)")
            assert False
        # self.model_name  = model_name
        # # models = self.client.models.list()

        # # for model in models.data:
        # #     print(model.id)
        
        # # Logger.info(f"Use openai backend to generate")
        # print(f"Use openai backend to generate")
            # response = self.chat_with_ai("who are you?")
        # print(f"\n test openai model {self.model_name} : {response}\n")


        # assert False

    

    def construct_messages(self, prompt: str, history: List[List[str]]) -> List[Dict[str, str]]:
        messages = []
            # {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的回答。"}]

        for user_input, ai_response in history:
            messages.append({"role": "user", "content": user_input})
            messages.append(
                {"role": "assistant", "content": ai_response.__repr__()})

        messages.append({"role": "user", "content": prompt})
        return messages
    
    
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
        if self.model_name == 'llama_3_8b': 
            payload = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # "temperature": 0.7,
                # "top_k": 5,
                # "top_p": 0.5,
            })
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.request("POST", self.url, headers=headers, data=payload)
    
            # print(f"response.text {response.text}")
            # print(f"json.loads(response.text)['result']\n {json.loads(response.text)['result']}")
            # print(f"json.loads(response.text)['usage']['prompt_tokens']\n {json.loads(response.text)['usage']['prompt_tokens']}")
            query_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
            if prompt.startswith(query_start):
                return json.loads(response.text)['result'], json.loads(response.text)['usage']['prompt_tokens']

            return json.loads(response.text)['result']

            # assert False
        elif self.model_name == 'qwen3_32b':
            # completion = self.client.chat.completions.create(
            #     # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            #     model="qwen3-32b",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": "你是谁？"},
            #     ],
            #     # temperature = 0.6, # temperature和top_p只建议设置一个
            #     # top_p = 0.95,
            #     # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            #     # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            #     extra_body={
            #         "enable_thinking": False,
            #         # "top_k": 20,
            #         },
            # )
            completion = self.client.chat.completions.create(
                model="qwen3-32b",  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': prompt}],
                # temperature = 0.6, # temperature和top_p只建议设置一个
                # top_p = 0.95,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={
                    "enable_thinking": True,
                    # "top_k": 20,
                    },
                )
            # print(completion.model_dump_json())
            # print(completion.usage.prompt_tokens)
            # print(completion.choices[0].message.content)
            # print(json.loads((completion.model_dump_json()))['usage']['prompt_tokens'])
            # print(json.loads((completion.model_dump_json()))['choices'][0]['message']['content'])
            full_content = ""
            think_content = ""
            input_tokens = 0
            output_tokens = 0

            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content = delta.content
                        full_content += content
                        # print(content, end='')  # 实时打印返回内容，不换行
                    if delta.reasoning_content:
                        think_content += delta.reasoning_content

                    

                # 提取 token 使用情况
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
            print(f"----------think content---------\n{think_content}")
            query_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
            if prompt.startswith(query_start):
                return full_content, input_tokens

            return full_content
        else:
            assert False

    def chat_with_ai_multi_round(self, prompt: str, history: List[Dict[str, str]]): # 只有qwen可以
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            # top_p=0.95,
            # temperature=0.6,
            # temperature=0,
            max_tokens=1024,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={
                "enable_thinking": False,
                # "top_k": 20,
                "thinking_budget": 38912,
                },
        )

        full_content = ""
        think_content = ""
        input_tokens = 0
        output_tokens = 0

        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content = delta.content
                    full_content += content
                    # print(content, end='')  # 实时打印返回内容，不换行
                if delta.reasoning_content:
                    think_content += delta.reasoning_content

                

            # 提取 token 使用情况
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
        print(f"----------think content---------\n{think_content}")
        
        return full_content

    # def chat_with_ai_multi_round(self, prompt: str, history: List[Dict[str, str]]):
    #     response = self.client.chat.completions.create(
    #         model=self.model_name,
    #         messages=history,
    #         # top_p=0.7,
    #         # temperature=0.95,
    #         temperature=0,
    #         max_tokens=1024,
    #     )
        
    #     return response.choices[0].message.content

    # @override
    # def chat_with_ai(self, prompt: str, history: List[List[str]] | None = None) -> str | None:
    #     response = self.client.chat.completions.create(
    #         model=self.model_name,
    #         messages=self.construct_messages(prompt, history if history else []),
    #         # top_p=0.7,
    #         # temperature=0.95,
    #         temperature=0,
    #         max_tokens=1024,
    #     )

    #     return response.choices[0].message.content

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



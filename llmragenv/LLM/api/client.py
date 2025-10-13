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
from utils.config import path_config

# def get_access_token():
def get_access_token(key):
    """
    使用应用API Key，应用Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    url = key
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

class APIClient(LLMBase):
    
    # _logger: Logger = Logger("AIClient")

    def __init__(self, model_name, url, key):
        super().__init__()

        self.model_name  = model_name
        self.llmbackend = "api"

        llm_config = path_config["LLM"]['api']
        if model_name in llm_config:
            self.url = llm_config[model_name]['url']
            self.key = llm_config[model_name]['key']
        else:
            raise ValueError(f"模型 {model_name} 未在配置文件中找到，请检查 path-local.yaml 配置项。")
        
        if model_name == 'llama_3_8b':
            self.url_access_token = self.url + get_access_token(self.key)
            print(f"Use openai backend to generate")
            messages=[{'role': 'user', 'content': "who are you?"}]
            # response = self.chat_with_ai("who are you?")
            response, token_num = self.chat_with_ai("", messages)
            print(f"\n test openai model {self.model_name} (tokens: {token_num}): {response}\n")
            
            # assert False
        elif model_name == 'qwen3_32b':
            self.client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                # api_key=os.getenv("DASHSCOPE_API_KEY"),
                api_key = self.key,
                base_url = self.url,
            )
            print(f"Use openai backend to generate")
            # response = self.chat_with_ai("", "who are you?")
            messages=[{'role': 'user', 'content': "who are you?"}]
            response = self.chat_with_ai("", messages)
            print(f"\n test openai model {self.model_name} : {response}\n")
            
        elif model_name == 'meta-llama-3-8b':
            self.client = OpenAI(
                api_key = self.key,
                base_url = self.url,
            )
            print("No longer supported meta-llama-3-8b(openai:client.py)")
            assert False
        else:
            print("A model that does not exist in OpenAI(openai:client.py)")
            assert False
    

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
    
        if self.model_name == 'llama_3_8b': 
            payload = json.dumps({
                # "messages": [
                #     {
                #         "role": "user",
                #         "content": prompt
                #     }
                # ],
                "messages": history,
                # "temperature": 0.7,
                # "top_k": 5,
                # "top_p": 0.5,
            })
            headers = {
                'Content-Type': 'application/json'
            }
            try:
                response = requests.request("POST", self.url, headers=headers, data=payload)
                if 'error_code' in json.loads(response.text):
                    return 'API call throws exception', 0
    
                print(f"response.text {response.text}")
                print(f"json.loads(response.text)['result']\n {json.loads(response.text)['result']}")
                print(f"json.loads(response.text)['usage']['prompt_tokens']\n {json.loads(response.text)['usage']['prompt_tokens']}")
                return json.loads(response.text)['result'], json.loads(response.text)['usage']['prompt_tokens']
            except Exception as e:
                return 'API call throws exception', 0
            # return json.loads(response.text)['result']

            # assert False
        # elif self.model_name == 'qwen2.5-32b-instruct':
        #     try:
        #         completion = self.client.chat.completions.create(
        #             model="qwen2.5-32b-instruct",
        #             messages=history,
        #             # top_p=0.95,
        #             # temperature=0.6,
        #             # temperature=0,
        #             max_tokens=1024,
        #         )
        #         completion.choices[0].message.content

            
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

            # print(f"----------type history---------\n{type(history)}")
            # print(f"----------history---------\n{history}")


            enable_thinking = False
            if prompt == 'think' or prompt == 'think_return':
                enable_thinking = True

            if enable_thinking:
                temperature = 0.6 # 思考，temperature和top_p只建议设置一个
                top_p = 0.95
            else:
                temperature = 0.7 # 非思考
                top_p = 0.8

            try:
                # print("4444")
                completion = self.client.chat.completions.create(
                    model="qwen3-32b",  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                    # messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                    #             {'role': 'user', 'content': prompt}],
                    messages=history,
                    # temperature = 0.7, # 非思考
                    # top_p = 0.8,
                    # temperature = 0.6, # temperature和top_p只建议设置一个
                    # top_p = 0.95,
                    # Python 示例
                    temperature = temperature,
                    top_p = top_p,
                    
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={
                        # "enable_thinking": True,
                        "enable_thinking": enable_thinking,
                        "top_k": 20, # 40
                        },
                    )
                # print(completion.model_dump_json())
                # print(completion.usage.prompt_tokens)
                # print(completion.choices[0].message.content)
                # print(json.loads((completion.model_dump_json()))['usage']['prompt_tokens'])
                # print(json.loads((completion.model_dump_json()))['choices'][0]['message']['content'])
                # print("555")
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
                        if enable_thinking and delta.reasoning_content:
                            think_content += delta.reasoning_content
                    # 提取 token 使用情况
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                print(f"----------think content---------\n{think_content}")
                print(f"----------full content---------\n{full_content}")

                if prompt == 'think_return':
                    return full_content, think_content
                else:
                    return full_content
            except Exception as e:
                print(f"chat_with_api Exception: {e}\nllm name: {self.model_name}")
                if prompt == 'think_return':
                    return 'API call throws exception', ""
                else:
                    return 'API call throws exception'

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



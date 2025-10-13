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
import http.client
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from llmragenv.LLM.llm_base import LLMBase
from overrides import override
from logger import Logger
import requests
import json
from utils.config import path_config
from openai import OpenAI

import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from chat.chat_prompt import llama_instruct_system, llama_instruct_user, qwen2_instruct_system, qwen2_instruct_user, qwen3_instruct_system, qwen3_instruct_user_noThink, qwen3_instruct_user_think
import httpx

# 1. 创建支持 IPv4 的 Transport
# 通过设置 local_address 为 "0.0.0.0" 强制底层套接字使用 IPv4
transport = httpx.HTTPTransport(
    local_address="0.0.0.0",
    retries=3
)

# 2. 创建 httpx 客户端，并配置代理 (注意 0.28.1 的 proxy 参数写法)
# 将下面的 127.0.0.1:7890 替换为你实际的代理地址
proxy_url = "http://127.0.0.1:7890" 
# proxy_url = "http://127.0.0.1:14400" 

http_client = httpx.Client(
    # transport=transport,
    proxy=proxy_url,      # 0.28.1 推荐使用单数 proxy
    trust_env=False,      # 避免系统环境变量冲突，完全由代码控制
    timeout=2.0
)
# http_client = httpx.Client(
#     transport=httpx.HTTPTransport(local_address="0.0.0.0"),
#     trust_env=False, 
#     timeout=60.0
# )

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
        self.type = "API" # 不添加聊天模版

        llm_config = path_config["LLM"]['api']
        if model_name in llm_config:
            self.url = llm_config[model_name]['url']
            self.key = llm_config[model_name]['key']
        else:
            raise ValueError(f"模型 {model_name} 未在配置文件中找到，请检查 path-local.yaml 配置项。")
        # print(f"model_name: {model_name} self.model_name: {self.model_name} self.url: {self.url} self.key: {self.key}")
        # assert False
        
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
        elif model_name == 'gpt-4o-mini_http':
            self.client = http.client.HTTPSConnection("127.0.0.1", 7890)
            self.client.set_tunnel(self.url)
            # self.client = http.client.HTTPSConnection(self.url)
            headers = {
                'Accept': 'application/json',
                'Authorization': self.key,
                'Content-Type': 'application/json'
            }
            payload = json.dumps({
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Who are you?"
                    }
                ]
            })
            self.client.request("POST", "/v1/chat/completions", payload, headers)
            res = self.client.getresponse()
            raw_data = res.read()
            data = json.loads(raw_data.decode("utf-8"))
            response = data["choices"][0]["message"]["content"]
            prompt_tokens = data["usage"]["prompt_tokens"]
            print(f"\n test openai model {self.model_name} : {response}\n")
        elif model_name == 'gpt-4o-mini' or model_name == "deepseek-chat" or model_name == "deepseek-v3.2" or model_name == "qwen2.5-32b-instruct" or model_name == "gpt-4o-mini-2024-07-18" or model_name == "qwen-flash"  or model_name == "qwen3.5-flash":
            self.client = OpenAI(base_url=self.url, api_key=self.key) # , http_client=http_client
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant"
                        },
                        {
                            "role": "user",
                            "content": "Who are you?",
                        }
                    ],
                    # temperature=0,
                    # stream=False,
                    
                )
                print(response)
                print(response.choices[0].message.content.strip())
            except Exception as e:
                # print(f"Error in LLM API call: {e}")
                print(f"Detailed Error: {type(e).__name__}: {e}") # 打印具体错误类型和内容
                import traceback
                traceback.print_exc() # 打印完整堆栈
                assert False
                return None
            
        elif model_name == 'gpt-4o-mini_request': 
            pass
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
    
    # 包含系统提示词
    def chat_with_ai_two_prompt(self, prompt_system: str, prompt_user: str) -> str | None:
        # print(f"prompt_system: {prompt_system}")
        # print(f"prompt_user: {prompt_user}")
        if self.model_name == 'gpt-4o-mini' or self.model_name == 'deepseek-chat' or self.model_name == 'deepseek-v3.2' or self.model_name == "qwen2.5-32b-instruct" or self.model_name == "gpt-4o-mini-2024-07-18" or self.model_name == "qwen-flash" or self.model_name == "qwen3.5-flash":
            try:
                start_time = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt_system
                        },
                        {
                            "role": "user",
                            "content": prompt_user,
                        }
                    ],
                    # temperature=1.8, # gpt-4o-mini-2024-07-18
                    # top_p = 1, #0.1
                    # max_tokens = 16,
                    extra_body={"enable_thinking":False},
                    stream=False,
                )
                end_time = time.perf_counter()
                # print(response)
                # print(response.choices[0].message.content.strip())
                # print(response.usage.prompt_tokens)
                # print(f"prompt_system: {prompt_system}\n prompt_user: {prompt_user}\n response: {response.choices[0].message.content.strip()}\n")
                # assert False
                return response.choices[0].message.content.strip(), response.usage.prompt_tokens, end_time - start_time
            except Exception as e:
                print(f"Error in LLM API call: {e}")
                time.sleep(5)
                return 'API call throws exception', 0, 0
        else:
            assert False
                
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

        elif self.model_name == 'gpt-4o-mini_http': 
            headers = {
                'Accept': 'application/json',
                'Authorization': self.key,
                'Content-Type': 'application/json'
            }
            payload = json.dumps({
                "model": self.model_name,
                # "messages": history,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            try:
                start_time = time.perf_counter()
                self.client.request("POST", "/v1/chat/completions", payload, headers)
                res = self.client.getresponse()
                end_time = time.perf_counter()
                raw_data = res.read()
                data = json.loads(raw_data.decode("utf-8"))
                response = data["choices"][0]["message"]["content"]
                prompt_tokens = data["usage"]["prompt_tokens"]
                # if 'error_code' in json.loads(response.text):
                #     return 'API call throws exception', 0
                # print(f"prompt: {prompt}")
                # print(f"response.text {data}")
                # print(f"data['choices'][0]['message']['content']\n {response}")
                # print(f"data['usage']['prompt_tokens']\n {prompt_tokens}")
                return response, prompt_tokens, end_time - start_time
            except Exception as e:
                print(f"Error in gpt-4o-mini API call: {e}")
                return 'API call throws exception', 0, 0
        elif self.model_name == 'gpt-4o-mini' or self.model_name == 'deepseek-chat' or self.model_name == 'deepseek-v3.2' or self.model_name == "qwen2.5-32b-instruct" or self.model_name == "qwen-flash":
            try:
                start_time = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    # temperature=0,
                    stream=False,
                )
                end_time = time.perf_counter()
                # print(response)
                # print(response.choices[0].message.content.strip())
                # print(response.usage.prompt_tokens)
                return response.choices[0].message.content.strip(), response.usage.prompt_tokens, end_time - start_time
            except Exception as e:
                print(f"Error in LLM API call: {e}")
                time.sleep(5)
                return 'API call throws exception', 0, 0
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

    @override
    def chat_with_ai_with_system(self, prompt_system: str | List[str], prompt_user: str | List[str], history: List[List[str]] | None = None, enable_thinking = False) -> str | None:
        # assert False, "HuggingfaceClient does not support chat_with_ai_with_system method. Please use chat_with_ai instead."
        if isinstance(prompt_system, list):
            prompt_system_instruct = []
            prompt_user_instruct = []
            for prompt_system_item, prompt_user_item in zip(prompt_system, prompt_user):
                # if self.type == 'Llama':
                #     prompt_system_user_instruct.append(llama_instruct_system.format(system_instruction = prompt_system_item)+llama_instruct_user.format(user_instruction = prompt_user_item))
                # elif self.type == 'Qwen2.5':
                #     prompt_system_user_instruct.append(qwen2_instruct_system.format(system_instruction = prompt_system_item)+qwen2_instruct_user.format(user_instruction = prompt_user_item))
                # elif self.type == 'Qwen3':
                #     if enable_thinking:
                #         prompt_system_user_instruct.append(qwen3_instruct_system.format(system_instruction = prompt_system_item)+qwen3_instruct_user_think.format(user_instruction = prompt_user_item))
                #     else:
                #         prompt_system_user_instruct.append(qwen3_instruct_system.format(system_instruction = prompt_system_item)+qwen3_instruct_user_noThink.format(user_instruction = prompt_user_item))
                # elif self.type == 'API':
                #     prompt_system_user_instruct.append(prompt_system_item + prompt_user_item)
                prompt_system_instruct.append(prompt_system_item)
                prompt_user_instruct.append(prompt_user_item)
            # return self.chat_with_ai_batch(prompt_system_instruct, prompt_user_instruct)
            return self.chat_with_ai_batch_Thread(prompt_system_instruct, prompt_user_instruct)
        else:
            # if self.type == 'Llama':
            #     prompt_system_instruct = llama_instruct_system.format(system_instruction = prompt_system)
            #     prompt_user_instruct = llama_instruct_user.format(user_instruction = prompt_user)
            # elif self.type == 'Qwen2.5':
            #     prompt_system_instruct = qwen2_instruct_system.format(system_instruction = prompt_system)
            #     prompt_user_instruct = qwen2_instruct_user.format(user_instruction = prompt_user)
            # elif self.type == 'Qwen3':
            #     if enable_thinking:
            #         prompt_system_instruct = qwen3_instruct_system.format(system_instruction = prompt_system)
            #         prompt_user_instruct = qwen3_instruct_user_think.format(user_instruction = prompt_user)
            #     else:
            #         prompt_system_instruct = qwen3_instruct_system.format(system_instruction = prompt_system)
            #         prompt_user_instruct = qwen3_instruct_user_noThink.format(user_instruction = prompt_user)
            # elif self.type == 'API':
            #         prompt_system_instruct = prompt_system
            #         prompt_user_instruct = prompt_user
            prompt_system_instruct = prompt_system
            prompt_user_instruct = prompt_user
            # return self.chat_with_ai(prompt_system_instruct + prompt_user_instruct)
            return self.chat_with_ai_two_prompt(prompt_system_instruct, prompt_user_instruct)
    
    def chat_with_ai_batch(self, prompt_system: List[str], prompt_user: List[str]) -> str | None:
        start_time = time.time()
        response = []
        prompt_len = []
        for prompt_system_item, prompt_user_item in zip(prompt_system, prompt_user):
            # response_item, prompt_len_item, generate_time_item = self.chat_with_ai(prompt_system_item + prompt_user_item)
            response_item, prompt_len_item, generate_time_item = self.chat_with_ai_two_prompt(prompt_system_item, prompt_user_item)
            response.append(response_item)
            prompt_len.append(prompt_len_item)
        end_time = time.time()

        return response, prompt_len, end_time - start_time
    
    def chat_with_ai_batch_Thread(self, prompt_system: List[str], prompt_user: List[str]) -> tuple:
        start_time = time.time()
        
        # 1. 确定线程数：通常根据你的 API 限制（RPM）或列表长度来定
        max_workers = len(prompt_system)
        
        # 2. 使用 ThreadPoolExecutor 进行并发
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 map 会保持结果顺序与输入顺序一致
            # map 接收函数名和对应的参数序列
            results = list(executor.map(self.chat_with_ai_two_prompt, prompt_system, prompt_user))

        # 3. 解构结果
        # results 的格式为 [(resp1, len1, time1), (resp2, len2, time2), ...]
        response = [r[0] for r in results]
        prompt_len = [r[1] for r in results]
        
        end_time = time.time()
        return response, prompt_len, end_time - start_time

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



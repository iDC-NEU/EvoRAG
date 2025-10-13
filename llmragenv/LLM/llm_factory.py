'''
Author: fzb fzb0316@163.com
Date: 2024-09-20 13:37:09
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-04 15:33:49
FilePath: /RAGWebUi_demo/llmragenv/LLM/llm_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from llmragenv.LLM.baichuan.client import BaichuanClient
from llmragenv.LLM.client_error import ClientUrlFormatError, ClientAPIUnsupportedError, ClientError
from llmragenv.LLM.deepseek.client import DeepseekClient
from llmragenv.LLM.doubao.client import DoubaoClient
from llmragenv.LLM.lingyiwanwu.client import LingyiwanwuClient
from llmragenv.LLM.llm_base import LLMBase
from llmragenv.LLM.moonshot.client import MoonshotClient
from llmragenv.LLM.ollama.client import OllamaClient
from llmragenv.LLM.qwen.client import QwenClient
from llmragenv.LLM.zhipu.client import ZhipuClient
from llmragenv.LLM.openai.client import OpenAIClient
from llmragenv.LLM.api.client import APIClient
from llmragenv.LLM.vllm.client import Vllm
from llmragenv.LLM.huggingface.client import HuggingfaceClient
from utils.singleton import Singleton
from utils.url_paser import is_valid_url
from config.config import Config


LLMProvider = {
    "zhipu" : [],
    "baichuan" : [],
    "qwen" : [],
    "moonshot" : [],
    "lingyiwanwu" : [],
    "deepseek" : [],
    "doubao" : [],
    "gpt" : ["gpt-4o-mini"],
    "llama" : ["qwen:0.5b", "llama2:7b", "llama2:13b", "llama2:70b","llama3:70b","llama3.1:70b","qwen:7b","qwen:14b","qwen:72b", "llama3.3", "deepseek-r1:70b", "llama3:8b-instruct-fp16", "meta-llama-3-8b", "llama_3_8b", "qwen2.5:72b", "qwen2.5:32b-instruct-fp16", 'qwen3:32b'],
    "huggingface" : ["Meta-Llama-3-8B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct-GPTQ-Int8", "Qwen3-32B", "Meta-Llama-3.1-8B-Instruct"],
    "vllm" : ["Meta-Llama-3-8B-Instruct", "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct-GPTQ-Int8", "Qwen3-32B", "Llama-3.1-8B-Instruct"],
    "api" : ['qwen3_32b', 'llama_3_8b'],
}



# metaclass=Singleton
class ClientFactory():
    
    def __init__(self, model_name, llmbackend = "openai"):
        self.model_name = model_name
        self.llmbackend = llmbackend
        # self.pid = pid


    def get_client(self) -> LLMBase:
        if self.llmbackend == "openai":
            if self.model_name in LLMProvider["zhipu"]:
                url = Config.get_instance().get_with_nested_params("llm", "zhipu", "url")
                key = Config.get_instance().get_with_nested_params("llm", "zhipu", "key")
                return ZhipuClient()

            elif self.model_name in LLMProvider["moonshot"]:
                url = Config.get_instance().get_with_nested_params("llm", "moonshot", "url")
                key = Config.get_instance().get_with_nested_params("llm", "moonshot", "key")
                return MoonshotClient()

            elif self.model_name in LLMProvider["baichuan"]:
                url = Config.get_instance().get_with_nested_params("llm", "baichuan", "url")
                key = Config.get_instance().get_with_nested_params("llm", "baichuan", "key")
                return BaichuanClient()

            elif self.model_name in LLMProvider["qwen"]:
                url = Config.get_instance().get_with_nested_params("llm", "qwen", "url")
                key = Config.get_instance().get_with_nested_params("llm", "qwen", "key")
                return QwenClient()

            elif self.model_name in LLMProvider["lingyiwanwu"]:
                url = Config.get_instance().get_with_nested_params("llm", "lingyiwanwu", "url")
                key = Config.get_instance().get_with_nested_params("llm", "lingyiwanwu", "key")
                return LingyiwanwuClient()

            elif self.model_name in LLMProvider["deepseek"]:
                url = Config.get_instance().get_with_nested_params("llm", "deepseek", "url")
                key = Config.get_instance().get_with_nested_params("llm", "deepseek", "key")
                return DeepseekClient()

            elif self.model_name in LLMProvider["doubao"]:
                url = Config.get_instance().get_with_nested_params("llm", "doubao", "url")
                key = Config.get_instance().get_with_nested_params("llm", "doubao", "key")
                return DoubaoClient()

            elif self.model_name in LLMProvider["gpt"]:
                url = Config.get_instance().get_with_nested_params("llm", "gpt", "url")
                key = Config.get_instance().get_with_nested_params("llm", "gpt", "key")
                return OpenAIClient(self.model_name, url, key)

            elif self.model_name in LLMProvider["llama"]:
                url = Config.get_instance().get_with_nested_params("llm", "llama", "url")
                key = Config.get_instance().get_with_nested_params("llm", "llama", "key")
                return OpenAIClient(self.model_name, url, key)

            else:
                raise ClientAPIUnsupportedError("No client API adapted")
        if self.llmbackend == "api":
            # if self.model_name in LLMProvider["api"]:
            #     return APIClient(self.model_name, "", "")
            # else:
            #     raise ClientAPIUnsupportedError("No client API adapted")
            return APIClient(self.model_name, "", "")
        elif self.llmbackend == "llama_index":
            # if self.model_name in LLMProvider["llama"]:
            #     url = Config.get_instance().get_with_nested_params("llm", "llama", "url")
            #     key = Config.get_instance().get_with_nested_params("llm", "llama", "key")
            #     return OllamaClient(self.model_name, url, key)
            return OllamaClient(self.model_name, "http://localhost:11434/v1", "")
            
        elif self.llmbackend == "huggingface":
            # if self.model_name in LLMProvider["huggingface"]:
            #     print(f"model name {self.model_name}")
            #     url = Config.get_instance().get_with_nested_params("llm", "llama", "url")
            #     key = Config.get_instance().get_with_nested_params("llm", "llama", "key")
            #     return HuggingfaceClient(self.model_name, url, key)
            return HuggingfaceClient(self.model_name, "", "")

        elif self.llmbackend == "vllm":
            # if self.model_name in LLMProvider["vllm"]:
            #     print(f"model name {self.model_name}")
            #     return Vllm(self.model_name, "", "")
            return Vllm(self.model_name, "", "")
            
        else:
            raise ClientError(f"No llm_backend {self.llmbackend}")

if __name__ == "__main__":
    factory1 = ClientFactory()
    factory2 = ClientFactory()

    print(factory1 is factory2)

'''
Author: fzb0316 fzb0316@163.com
Date: 2024-09-21 19:23:18
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-21 19:22:02
FilePath: /BigModel/RAGWebUi_demo/llmragenv/Cons_Retri/Embedding_Model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from transformers import AutoTokenizer, AutoModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-ada-002": 1536,
}
class EmbeddingEnv:
    
    def get_free_device(self):
        import torch
        num_gpus = torch.cuda.device_count()
        free_memory = []
        
        for i in range(num_gpus):
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            
            free_memory.append(total_memory - allocated_memory)
        
        best_gpu = free_memory.index(max(free_memory))
        return f"cuda:{best_gpu}"

    def __init__(self,
                 embed_name="BAAI/bge-large-en-v1.5",
                 embed_batch_size=20,
                 device="cuda:0"):
        self.embed_name = embed_name
        self.embed_batch_size = embed_batch_size

        assert embed_name in EMBEDD_DIMS
        self.dim = EMBEDD_DIMS[embed_name]

        #device = self.get_free_device()
        device="cuda:0"


        if 'BAAI' in embed_name:
            print(f"use huggingface embedding {embed_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_name,
                embed_batch_size=embed_batch_size,
                device=device)
        # else:
        #     print(f"use openai embedding {embed_name}")
        #     self.embed_model = OpenAIEmbedding(
        #         model=embed_name, embed_batch_size=embed_batch_size)

        # Settings.embed_model = self.embed_model
        # print_text(
        #     f"EmbeddingEnv: embed_name {embed_name}, embed_batch_size {self.embed_batch_size}, dim {self.dim}\n",
        #     color='red')

    def __str__(self):
        return f"{self.embed_name} {self.embed_batch_size}"

    def get_embedding(self, query_str):
        embedding = self.embed_model._get_text_embedding(query_str)
        return embedding

    def get_embeddings(self, query_str_list):
        import torch
        with torch.no_grad():  # 禁用梯度计算
            embeddings = self.embed_model._get_text_embeddings(query_str_list)
            # embeddings = torch.nn.DataParallel(self.embed_model._get_text_embeddings)(query_str_list)
        torch.cuda.empty_cache()
        return embeddings

    def calculate_similarity(self, query1, query2):
        # Cosine similarity
        embedding1 = self.embed_model.get_embedding(query1)
        embedding2 = self.embed_model.get_embedding(query2)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = round(dot_product / (norm1 * norm2), 6)
        return similarity
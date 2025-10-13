# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 必须在导入 torch 或 vllm 之前设置！

    
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer

# EMBEDD_DIMS = {
#     "BAAI/bge-large-en-v1.5": 1024,
#     "BAAI/bge-base-en-v1.5": 768,
#     "BAAI/bge-small-en-v1.5": 384,
#     "text-embedding-ada-002": 1536,
# }

# class EmbeddingEnv:
#     def __init__(self,
#                  embed_name="BAAI/bge-large-en-v1.5",
#                  embed_batch_size=256,  # 既然显存大，建议调高默认值
#                  device="cuda:0",
#                  num_workers=4,         # 新增：控制并行进程数，防止吃满
#                  args=None):
        
#         if args:
#             embed_name = getattr(args, 'model', embed_name)
#             embed_batch_size = getattr(args, 'embed_batch_size', embed_batch_size)
#             num_workers = getattr(args, 'num_workers', num_workers)
#             device = getattr(args, 'device', device)

#         self.embed_name = embed_name
#         self.embed_batch_size = embed_batch_size
#         self.device = device
#         self.num_workers = num_workers

#         assert embed_name in EMBEDD_DIMS
#         self.dim = EMBEDD_DIMS[embed_name]

#         print(f"EmbeddingEnv: {embed_name}, Batch: {embed_batch_size}, Device: {device}, Workers: {num_workers}")

#         if 'BAAI' in embed_name:
#             # 使用 SentenceTransformer 加载，并开启 FP16 加速
#             self.model = SentenceTransformer(
#                 embed_name, 
#                 device=device,
#                 model_kwargs={"torch_dtype": torch.float16} 
#             )
#             # 初始化多进程池
#             # 如果只想限制在某一张卡，重复 device ID 即可
#             self.pool = self.model.start_multi_process_pool(target_devices=[device] * num_workers)
#         else:
#             raise NotImplementedError("OpenAI or other models not supported in this multi-process wrapper.")

#     def __str__(self):
#         return f"{self.embed_name} {self.embed_batch_size}"

#     def get_embedding(self, query_str):
#         # 单条查询不建议用多进程，直接编码更快
#         return self.model.encode(query_str, batch_size=self.embed_batch_size, show_progress_bar=False)

#     def get_embeddings(self, query_str_list):
#         """
#         使用多进程加速生成嵌入
#         """
#         # 注意：encode_multi_process 内部已包含 torch.no_grad()
#         # embeddings = self.model.encode_multi_process(
#         # # embeddings = self.model.encode(
#         #     query_str_list, 
#         #     self.pool, 
#         #     batch_size=self.embed_batch_size
#         # )
#         # return embeddings
#         return self.model.encode(query_str_list, batch_size=self.embed_batch_size)

#     def calculate_batch_similarity(self, query_embedding, context_embeddings):
#         """
#         query_embedding: 单个向量 (1, D)
#         context_embeddings: 候选向量矩阵 (N, D)
#         """
        
#         # 转换为 Tensor 并移至 GPU
#         q = torch.tensor(query_embedding).to(self.device)
#         c = torch.tensor(context_embeddings).to(self.device)
        
#         # 1. 归一化 (如果是单位向量，这步可以省略)
#         q = torch.nn.functional.normalize(q, p=2, dim=-1)
#         c = torch.nn.functional.normalize(c, p=2, dim=-1)
        
#         # 2. 矩阵点积即为余弦相似度
#         # (1, D) * (D, N) -> (1, N)
#         similarities = torch.mm(q.unsqueeze(0), c.transpose(0, 1))
        
#         return similarities.cpu().numpy()
    
#     def calculate_similarity(self, query1, query2):
#         # 直接使用 model.encode 得到的是 numpy 数组
#         embedding1 = self.get_embedding(query1)
#         embedding2 = self.get_embedding(query2)
        
#         # 优化：可以直接用 sentence_transformers 提供的 util 计算，或者保持原样
#         dot_product = np.dot(embedding1, embedding2)
#         norm1 = np.linalg.norm(embedding1)
#         norm2 = np.linalg.norm(embedding2)
#         return round(dot_product / (norm1 * norm2), 6)

#     def close(self):
#         """记得在程序结束前关闭进程池"""
#         if hasattr(self, 'pool'):
#             self.model.stop_multi_process_pool(self.pool)
            
#     def get_top_k_similar(self, question_embed, all_rel_embeddings):
#         """
#         高性能相似度计算引擎
#         question_embed: np.array (1, D) 或 (D,)
#         all_rel_embeddings: np.array (N, D)
#         """
#         # 转换为 Tensor 并移动到 GPU (使用 float16 进一步加速计算)
#         q_tensor = torch.from_numpy(question_embed).to(self.device).to(torch.float16)
#         rels_tensor = torch.from_numpy(all_rel_embeddings).to(self.device).to(torch.float16)

#         # 如果是 1D 向量则增加维度
#         if q_tensor.dim() == 1:
#             q_tensor = q_tensor.unsqueeze(0)

#         # 1. 预归一化：使模长为 1，这样点积即为余弦相似度
#         q_norm = torch.nn.functional.normalize(q_tensor, p=2, dim=-1)
#         rels_norm = torch.nn.functional.normalize(rels_tensor, p=2, dim=-1)

#         # 2. 矩阵乘法 (1, D) @ (D, N) -> (1, N)
#         # 计算完成后立即转回 float32 保证统计精度
#         with torch.no_grad():
#             similarity_matrix = torch.matmul(q_norm, rels_norm.transpose(0, 1))
        
#         # 3. 转回 CPU numpy
#         return similarity_matrix.cpu().to(torch.float32).numpy()[0]






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

    def __init__(self,
                 embed_name="BAAI/bge-large-en-v1.5",
                 embed_batch_size=20,
                 device="cuda:0",
                 args=None):
        if args:
            embed_name = args.model if hasattr(args, 'model') else embed_name
            embed_batch_size = args.embed_batch_size if hasattr(args, 'embed_batch_size') else embed_batch_size
            device = args.device if hasattr(args, 'device') else device
            
        
        self.embed_name = embed_name
        self.embed_batch_size = embed_batch_size

        assert embed_name in EMBEDD_DIMS
        self.dim = EMBEDD_DIMS[embed_name]

        #device = self.get_free_device()
        # device="cuda:0"
        
        print(f"EmbeddingEnv: embed_name {embed_name}, embed_batch_size {embed_batch_size}, dim {self.dim}, device {device}")


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
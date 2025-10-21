'''
Author: fzb0316 fzb0316@163.com
Date: 2024-10-18 16:47:15
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-30 11:10:27
FilePath: /BigModel/RAGWebUi_demo/kg_modify.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 必须在导入 torch 或 vllm 之前设置！
import argparse


from config.config import Config
from llmragenv.llmrag_env import LLMRAGEnv
from database.graph.graph_dbfactory import GraphDBFactory
from dataset.dataset import Dataset
from utils.config import algorithm_config
from box import Box
from pathlib import Path

import torch
import torch.distributed as dist
from multiprocessing import resource_tracker
import gc



def run_with_dataset(args):


    LLMRAGEnv().chat_to_KG_modify(args)


def cleanup():
    # 清理分布式进程组
    if dist.is_initialized():
        dist.destroy_process_group()

    # 手动触发垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLMRag Workload")

    parser.add_argument("--dataset_name", type=str, help="dataset name", default='rgb')
    parser.add_argument("--llm", type=str, help="llm env (e.g., qwen0.5b, llama2:7b, llama2:13b, llama2:70b)", default='llama2:7b')
    parser.add_argument("--forward_llm", type=str, help="llm env (e.g., qwen0.5b, llama2:7b, llama2:13b, llama2:70b)", default='Meta-Llama-3-8B-Instruct')
    parser.add_argument("--llm_fb", type=str, help="llm env (e.g., qwen0.5b, llama2:7b, llama2:13b, llama2:70b)", default='None')
    parser.add_argument("--graphdb", type=str, help="graph database baskend (e.g., neo4j, ) ", default='nebulagraph')
    parser.add_argument("--space_name", type=str, help="graph database space name (e.g., rgb, ) ", default='rgb')
    parser.add_argument("--option", type=str, help="execution way (e.g., without_rag, graph_rag ) ", default='graph_rag')
    parser.add_argument("--llmbackend", type=str, help="openai or llama_index", default="openai")
    parser.add_argument("--iteration", type=int, help="number of iteration (e.g., 3, 5 ) ", default='12')
    parser.add_argument("--type", type=str, help="data of iteration changed or unchanged (e.g., changed, unchanged ) ", default='unchanged')
    parser.add_argument("--pruning", type=int, help="The number of search paths for each entity", default=10)
    parser.add_argument("--entity", type=int, help="Number of entities retrieved per question", default=10)
    parser.add_argument("--hop", type=int, help="Number of entities retrieved per question", default=2)
    parser.add_argument("--similar", type=bool, help="Number of entities retrieved per question", default=False)
    parser.add_argument("--rate", type=float, help="feedback noise rate", default=0.0)
    # parser.add_argument("--case_rate", type=float, help="feedback noise rate", default=0.05)

    
    args = parser.parse_args()
    print(args)
    print(algorithm_config)


    # 1. 获取 algorithm 配置并提升到顶层
    algorithm_name = algorithm_config["algorithm"]
    nested_config = algorithm_config.get(algorithm_name, {})
    algorithm_config.update(nested_config)  # 把算法配置提升

    # 2. 把命令行参数更新到 config，实现覆盖
    algorithm_config.update(vars(args))

    # 3. 转成 Box 对象（支持点访问）
    data = Box(algorithm_config)

    print(data)
    # assert False

    # # 定义路径
    # logs_dir = Path("logs")
    # data_dir = logs_dir / data.algorithm
    # new_dir = [data_dir / "data", data_dir / "output", data_dir / "stage", data_dir / "triplets",]


    # # 检查并创建目录
    # if not data_dir.exists():
    #     data_dir.mkdir(parents=True, exist_ok=True)
    #     print(f"Created directory: {data_dir}")

    # for triple_dir in new_dir:
    #     if not triple_dir.exists():
    #         triple_dir.mkdir(parents=True, exist_ok=True)
    #         print(f"Created directory: {triple_dir}")

    print("All required directories are ready.")

    run_with_dataset(data)
    # run_with_dataset(args)

    cleanup()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # 必须在导入 torch 或 vllm 之前设置！
import argparse


from config.config import Config
from llmragenv.llmrag_env import LLMRAGEnv
from database.graph.graph_dbfactory import GraphDBFactory
from dataset.dataset import Dataset
from utils.config import algorithm_config
from box import Box
from pathlib import Path

# import torch
# import torch.distributed as dist
# from multiprocessing import resource_tracker
# import gc



def run_with_dataset(args):


    LLMRAGEnv().chat_to_KG_modify(args)


# def cleanup():
#     # 清理分布式进程组
#     if dist.is_initialized():
#         dist.destroy_process_group()

#     # 手动触发垃圾回收
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()



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
    parser.add_argument("--iteration", type=int, help="number of iteration (e.g., 3, 5 ) ", default=0)
    parser.add_argument("--type", type=str, help="data of iteration changed or unchanged (e.g., changed, unchanged ) ", default='unchanged')
    parser.add_argument("--pruning", type=int, help="The number of search paths for each entity", default=10)
    parser.add_argument("--entity", type=int, help="Number of entities retrieved per question", default=10)
    parser.add_argument("--hop", type=int, help="Number of entities retrieved per question", default=2)
    parser.add_argument("--similar", type=bool, help="Number of entities retrieved per question", default=False)
    parser.add_argument("--rate", type=float, help="feedback noise rate", default=0.0)
    parser.add_argument("--cost", type=bool, help="token cost", default=False)
    parser.add_argument("--align_human", type=bool, help="token cost", default=False)
    parser.add_argument("--statistics_path", type=bool, help="statistics_path", default=False)
    parser.add_argument("--use_concurrent", type=bool, help="use_concurrent", default=False) # 一定是配合 evo_data or evo_no_retrieve 使用
    parser.add_argument("--evo_data", type=bool, help="获取evo的data，包含问答", default=False)
    parser.add_argument("--evo_no_retrieve", type=bool, help="使用已有的evo的data train，不再检索", default=False)
    parser.add_argument("--skip_empty_entities", type=bool, help="跳过空检索路径的实体", default=False)
    parser.add_argument("--evo_case", type=bool, help="使用已有的evo的data 进行case", default=False)
    # parser.add_argument("--evo_case_final_iteration_for_only_error", type=bool, help="使用已有的evo的data 进行case，对于only_error train拼接的数据，需要确定截止轮次来定位数据名称", default=False)
    parser.add_argument("--only_error", type=bool, help="仅反馈错误的", default=False)
    parser.add_argument("--depcache_case", type=bool, help="复现depcache，针对hotpotqa1M_v2，对hotpotqa1M并不适用", default=False)
    
    
    # 一些为了并行实验不受配置文件影响的参数
    parser.add_argument("--feedback_off", type=bool, help="关闭反馈，用这种方法因为feedback参数配置文件中也有", default=False)  
    parser.add_argument("--abaltion_on", type=bool, help="使用消融反馈，用这种方法因为feedback参数配置文件中也有", default=False) 
    # parser.add_argument("--simThreshold_off", type=bool, help="相似度阈值为0，对于hotpotqa1M_v2，完成depcache/ragcache的实验", default=False) 
    # parser.add_argument("--depcache_case", type=bool, help="使用已有的depcache的data", default=False)
    # parser.add_argument("--evo_cache", type=bool, help="获取evo的data", default=False)

    # parser.add_argument("--case_rate", type=float, help="feedback noise rate", default=0.05)
    parser.add_argument("--f1", type=bool, help="feedback by f1 score", default=False)
    parser.add_argument("--human", type=bool, help="feedback by human score", default=False)
    parser.add_argument("--human_advanced", type=bool, help="feedback by human score", default=False)

    
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

    # 特殊参数
    if data.feedback_off:
        data.feedback = False
        
    if data.abaltion_on:
        data.feedback = True
        data.feedback_process = "leaderboardStrategy_abaltion"
        data.algorithm = "leaderboardStrategy_abaltion"

        
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

    # cleanup()
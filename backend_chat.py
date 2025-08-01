'''
Author: fzb fzb0316@163.com
Date: 2024-09-15 17:14:37
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-15 15:18:27
FilePath: /RAGWebUi_demo/webui_chat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import argparse


from config.config import Config
from llmragenv.llmrag_env import LLMRAGEnv
from database.graph.graph_dbfactory import GraphDBFactory
from dataset.dataset import Dataset




def run_backent_chat(args):


    # LLMRAGEnv().backend_chat(["What is the name of Apple's headset?"], args.llm, args.func, args.graphdb, args.vectordb, args.llmbackend)


    dataset = Dataset(args.dataset_name)
    LLMRAGEnv().chat_with_dataset(dataset, args)

    # dataset = Dataset(args.dataset_name)
    # dataset.get_corpus("positive")
    # LLMRAGEnv().chat_to_KG_construction(dataset.corpus, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLMRag Workload")

    parser.add_argument("--dataset_name", type=str, help="dataset name", default='rgb')
    parser.add_argument("--llm", type=str, help="llm env (e.g., qwen0.5b, llama2:7b, llama2:13b, llama2:70b)", default='llama2:70b')
    parser.add_argument("--func", type=str, help="function (e.g., Without RAG, Graph RAG, Vector RAG, Union RAG)", default='Graph RAG')
    parser.add_argument("--graphdb", type=str, help="graph database baskend (e.g., neo4j, ) ", default='nebulagraph')
    parser.add_argument("--vectordb", type=str, help="vector database baskend (e.g., MilvusDB, ) ", default='MilvusDB')
    parser.add_argument("--space_name", type=str, help="space_name baskend (e.g., rgb, multihop) ", default='rgb')
    parser.add_argument("--llmbackend", type=str, help="openai or llama_index", default="llama_index")

    args = parser.parse_args()
    print(args)

    run_backent_chat(args)




'''
Author: fzb fzb0316@163.com
Date: 2024-09-19 08:48:47
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-18 14:02:04
FilePath: /RAGWebUi_demo/llmragenv/Retriever/retriever_graph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''

    这个文件用来根据用户输入来抽取实体、转换向量数据等操作数据库操作前、后的处理工作
    也包括prompt设置、让大模型生成知识图谱等操作
    也可以直接操作数据库

'''
from icecream import ic
from llmragenv.LLM.llm_base import LLMBase
from database.graph.graph_database import GraphDatabase
import numpy as np
from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv

import cupy as cp
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import json
import statistics
import re
import time
import math


keyword_extract_prompt = (
    # "A question is provided below. Given the question, extract up to {max_keywords} "
    # "keywords from the text. Focus on extracting the keywords that we can use "
    # "to best lookup answers to the question. Avoid stopwords.\n"
    # "Note, result should be in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
    # # "Only response the results, do not say any word or explain.\n"
    # "---------------------\n"
    # "question: {question}\n"
    # "---------------------\n"
    # # "KEYWORDS: "
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "Note, result should be in the following comma-separated format, and start with KEYWORDS:'\n"
    "Only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "question: {question}\n"
    "---------------------\n"
)

# keyword_supplement_prompt = (
#     "A question and a set of existing keywords are provided below. Based on the "
#     "question and the existing keywords, supplement up to {max_keywords} new keywords "
#     "that are not already in the existing list. Only provide additional relevant "
#     "keywords that improve retrieval accuracy. If the existing keywords are already sufficient, return an empty result.\n"
#     "Avoid stopwords and redundant terms.\n"
#     "Provide the result in a comma-separated format, starting with 'KEYWORDS:'.\n"
#     "Only output the result, without any explanations.\n"
#     "---------------------\n"
#     "question: {question}\n"
#     "existing_keywords: {existing_keywords}\n"
#     "---------------------\n"
# )

keyword_supplement_prompt = (
    "A question and a set of existing keywords are provided below. Based on the question, supplement up to {max_keywords} new keywords that are not already in the existing list and that are directly relevant and useful for retrieving information in a knowledge graph. "
    "Only provide additional relevant keywords that improve retrieval accuracy. If the existing keywords are already sufficient, return 'No keywords to add.'\n"
    "Avoid stopwords and redundant terms.\n"
    "Provide the result in a comma-separated format, starting with 'KEYWORDS:'. Only output the result, without any explanations.\n"
    "question: {question}\n"
    "existing_keywords: {existing_keywords}\n"
)

llama_synonym_expand_prompt = (
    # "Generate synonyms or possible form of keywords up to {max_keywords} in total, "
    # "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    # "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # # "Only response the results, do not say any word or explain.\n"
    # "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    # "---------------------\n"
    # "KEYWORDS: {question}\n"
    # "---------------------\n"
    # # "SYNONYMS: "
    "Generate synonyms or possible form of keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # "Only response the results, do not say any word or explain.\n"
    "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "KEYWORDS: {question}\n"
    "---------------------\n")


embed_model = None
embed_model_small = None
embed_model_MiniLM = None

def get_text_embeddings(texts, embed_model_name="BAAI/bge-large-en-v1.5", step=400, args = None):
    global embed_model
    # embed_model = None
    if not embed_model:
        embed_model = EmbeddingEnv(embed_name=embed_model_name,
                                   embed_batch_size=4,
                                   args = args)

    all_embeddings = []
    n_text = len(texts)
    for start in range(0, n_text, step):
        input_texts = texts[start:min(start + step, n_text)]
        embeddings = embed_model.get_embeddings(input_texts)

        all_embeddings += embeddings
        
    return all_embeddings

def get_text_embedding(text, embed_model_name="BAAI/bge-large-en-v1.5", args = None):
    global embed_model
    # embed_model = None
    if not embed_model:
        embed_model = EmbeddingEnv(embed_name=embed_model_name,
                                   embed_batch_size=2,
                                   args = args)
    embedding = embed_model.get_embedding(text)
    return embedding

def get_text_embedding_MiniLM(text, embed_model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
    # global embed_model_MiniLM
    # if not embed_model_MiniLM:
    #     embed_model_MiniLM = SentenceTransformer(embed_model_name, device = "cuda:2")
    # import torch
    # with torch.no_grad(): 
    #     embedding = embed_model_MiniLM.encode(text)
    # # torch.cuda.empty_cache()
    # return embedding
    return None

def get_text_embeddings_MiniLM(texts, embed_model_name = 'sentence-transformers/all-MiniLM-L6-v2', step=40):
    # global embed_model_MiniLM
    # if not embed_model_MiniLM:
    #     embed_model_MiniLM = SentenceTransformer(embed_model_name, device = "cuda:2")
    # import torch
    # all_embeddings = []
    # n_text = len(texts)
    # for start in range(0, n_text, step):
    #     input_texts = texts[start:min(start + step, n_text)]
    #     with torch.no_grad(): 
    #         embeddings = embed_model_MiniLM.encode(input_texts)
    #     # torch.cuda.empty_cache()
    #     all_embeddings.extend(embeddings)
        
    # return all_embeddings
    return None

def get_text_embeddings_small(texts, embed_model_name="BAAI/bge-large-en-v1.5", step=400):
    global embed_model_small
    # embed_model = embed_model_small
    # embed_model = None
    if not embed_model_small:
        embed_model_small = EmbeddingEnv(embed_name=embed_model_name,
                                   embed_batch_size=2)

    all_embeddings = []
    n_text = len(texts)
    for start in range(0, n_text, step):
        input_texts = texts[start:min(start + step, n_text)]
        embeddings = embed_model_small.get_embeddings(input_texts)

        all_embeddings += embeddings
        
    return all_embeddings

def get_text_embedding_small(text, embed_model_name="BAAI/bge-large-en-v1.5"):
    global embed_model_small
    # embed_model = embed_model_small
    # embed_model = None
    if not embed_model_small:
        embed_model_small = EmbeddingEnv(embed_name=embed_model_name,
                                   embed_batch_size=2)
    embedding = embed_model_small.get_embedding(text)
    return embedding

def cosine_similarity_cp(
    embeddings1,
    embeddings2,
) -> float:
    embeddings1_gpu = cp.asarray(embeddings1)
    embeddings2_gpu = cp.asarray(embeddings2)

    product = cp.dot(embeddings1_gpu, embeddings2_gpu.T)

    norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True)
    norm2 = cp.linalg.norm(embeddings2_gpu, axis=1, keepdims=True)

    norm_product = cp.dot(norm1, norm2.T)

    cosine_similarities = product / norm_product

    return cp.asnumpy(cosine_similarities)

import cupy as cp

def cosine_similarity_cp_batch(
    embeddings1,
    embeddings2,
    batch_size=512
) -> float:
    embeddings1_gpu = cp.asarray(embeddings1)
    embeddings2_gpu = cp.asarray(embeddings2)

    norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True) #范数
    cosine_similarities = []

    num_batches = (embeddings2_gpu.shape[0] + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, embeddings2_gpu.shape[0])
        embeddings2_batch = embeddings2_gpu[start_idx:end_idx]

        norm2_batch = cp.linalg.norm(embeddings2_batch, axis=1, keepdims=True)

        product = cp.dot(embeddings1_gpu, embeddings2_batch.T)
        #norm_product = cp.dot(norm1, norm2[start_idx:end_idx].T)
        norm_product = cp.dot(norm1, norm2_batch.T)
        batch_cosine_similarities = product / norm_product

        cosine_similarities.append(batch_cosine_similarities)

    final_cosine_similarities = cp.concatenate(cosine_similarities, axis=1)

    return cp.asnumpy(final_cosine_similarities)


class RetrieverGraph(object):
    def __init__(self,llm:LLMBase, graphdb : GraphDatabase, args = None):
        self.graph_database = graphdb
        self._llm = llm
        # print(args)
        self.embedding = args.embedding
        
        # self.triplet2id = self.graph_database.triplet2id
        # self.triplet_embeddings = self.graph_database.triplet_embeddings
        # self.entity = self.graph_database.entity2id
        self.entity = self.graph_database.entities
        self.entity_embeddings = self.graph_database.entity_embeddings
        self.triplets_score = {}
        self.logger = None

    def extract_keyword(self, question, existing_keywords, max_keywords = 2):
        # prompt = keyword_extract_prompt.format(question=question, max_keywords=max_keywords)
        prompt = keyword_supplement_prompt.format(question=question, existing_keywords = existing_keywords, max_keywords=max_keywords)
        
        # print(f"extract keyword(supplement) response: {prompt}")
        
        # 获取 LLM 的 response
        # if self._llm.__class__.__name__ == "OllamaClient":
        #     response = self._llm.chat_with_ai(prompt, info = "keyword")
        # else:
        #     response = self._llm.chat_with_ai(prompt)
        response = self._llm.chat_with_ai(prompt)
        
        # 处理 response，去掉 "KEYWORDS:" 前缀
        # if response.startswith("No keywords to add."): # 有问题，可能是KEYWORDS: No keywords to add.
        if "No keywords to add." in response:
            return []
        if response.startswith("KEYWORDS:"):
            response = response[len("KEYWORDS:"):].strip()  # 去掉前缀并去除多余的空格
        # ic(response)
        
        # 按逗号分割，并去除空格，转为小写
        keywords = [keyword.strip().lower() for keyword in response.split(",")]
        # keywords = ["'东北大学'", "'Ral'"]
        capitalized_keywords= [keyword.replace("'", '') for keyword in keywords]
        # ic(capitalized_keywords)

        # 只将每个关键词的第一个字母大写
        capitalized_keywords = [keyword.capitalize() for keyword in capitalized_keywords]
        
        # ic(capitalized_keywords)
        # print(f"capitalized_keywords: {capitalized_keywords}")

        return capitalized_keywords

    def retrieve_2hop(self, question, pruning = None, build_node = False):
        self.pruning = pruning

        keywords = self.extract_keyword(question)
        query_results = {}

        if pruning:
            rel_map = self.graph_database.get_rel_map(entities=keywords, limit=1000000)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        clean_rel_map = self.graph_database.clean_rel_map(rel_map)

        query_results.update(clean_rel_map)

        knowledge_sequence = self.graph_database.get_knowledge_sequence(query_results)

        if knowledge_sequence == []:
            return knowledge_sequence

        if self.pruning:
            pruning_knowledge_sequence, pruning_knowledge_dict = self.postprocess(question, knowledge_sequence)
            
            if build_node:
                self.nodes = self.graph_database.build_nodes(pruning_knowledge_sequence,
                                pruning_knowledge_dict)
        else:
            pruning_knowledge_sequence = knowledge_sequence
            if build_node:       
                self.nodes = self.graph_database.build_nodes(knowledge_sequence, rel_map)

        return pruning_knowledge_sequence
    

    def retrieve_2hop_with_keywords(self, question, keywords = [], pruning = None, build_node = False):
        self.pruning = pruning

        query_results = {}

        if pruning:
            rel_map = self.graph_database.get_rel_map(entities=keywords, limit=1000000)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        # print(f"key_words : {keywords}")
        # print(f"rel_map : {rel_map}\n")

        clean_rel_map = self.graph_database.clean_rel_map(rel_map)

        # print(f"clean_rel_map : {clean_rel_map}\n")

        query_results.update(clean_rel_map)

        # print(f"query results : {query_results}\n")

        knowledge_sequence = self.graph_database.get_knowledge_sequence(query_results)

        if knowledge_sequence == []:
            return knowledge_sequence
        
        # print(f"knowledge_sequence : {knowledge_sequence}\n")

        if self.pruning:
            pruning_knowledge_sequence, pruning_knowledge_dict = self.postprocess(question, knowledge_sequence)
            
            if build_node:
                self.nodes = self.graph_database.build_nodes(pruning_knowledge_sequence,
                                pruning_knowledge_dict)
        else:
            pruning_knowledge_sequence = knowledge_sequence
            if build_node:       
                self.nodes = self.graph_database.build_nodes(knowledge_sequence, rel_map)

        print(f"pruning_knowledge_sequence : {pruning_knowledge_sequence}")
                
        return pruning_knowledge_sequence
    
    def retrieve_path_with_keywords(self, question, keywords = [], path_depth = 2, pruning = None, build_node = False):
        self.pruning = pruning

        query_results = {}

        all_rel_map = {}
        if pruning:
            # rel_map = self.graph_database.get_rel_map(entities=keywords, limit=1000000)
            for entity in keywords:
                rel_map = self.graph_database.get_rel_map(entities=[entity], depth=path_depth, limit=1000000)
                all_rel_map.update(rel_map)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        # print(f"key_words : {keywords}")
        # print(f"rel_map : {rel_map}\n")

        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(all_rel_map)
        # clean_rel_map = self.graph_database.clean_rel_map(rel_map)

        # print(f"clean_rel_map : {clean_rel_map}\n")

        query_results.update(clean_rel_map)

        # print(f"query results : {query_results}\n")

        knowledge_sequences = []
        sentence_to_triple_3d = []
        sentences = []

        for k, v in clean_rel_map.items():
            # print(k, type(v))
            kg_seqs = self.graph_database.get_knowledge_sequence({k: v})
            knowledge_sequences.append(kg_seqs)
        for i, item in  enumerate(sentence_to_triple_4d):
            sentence_to_triple_3d.extend(item)
        # for item in sentences_2d:
        #     sentences.append(item)
        sentences = [ item for sentence_list in sentences_2d for item in sentence_list]
            
        # print(f"len(sentences_2d): {len(sentences_2d)}")
        # print(f"len(sentences): {len(sentences)}")
        
        # print([x for x in knowledge_sequences])

        # print([len(x) for x in knowledge_sequences])

        knowledge_sequences_list = [item for sublist in knowledge_sequences for item in sublist]
        # print(f"knowledge_sequences_list{np.array(knowledge_sequences_list).shape}")
        # print(f"knowledge_sequences_list: {[x for x in knowledge_sequences_list]}")
        
        if pruning > 0:
            knowledge_sequences_pruning = []
            sentence_to_triple_pruning = []
            # sorted_all_rel_scores = self.semantic_pruning_triplets_batch(question, knowledge_sequences_list, rel_embeddings=None, topk=pruning, batch_size=1)
            top_all_rel_scores = self.semantic_pruning_sentences_batch(question, sentences, rel_embeddings=None, topk=pruning, batch_size=8)
            # knowledge_sequences_pruning = [rel for rel, _ in sorted_all_rel_scores]
            # knowledge_sequences = knowledge_sequences_pruning
            for idx, score in top_all_rel_scores:
                knowledge_sequences_pruning.append(sentences[idx])
                sentence_to_triple_pruning.append(sentence_to_triple_3d[idx])
        
        # print(len(knowledge_sequences))
        # print(knowledge_sequences)
        return knowledge_sequences_pruning, sentence_to_triple_pruning

        # if pruning > 0:
        #     knowledge_sequences_result = []
        #     for listp in knowledge_sequences:
            
        #         sorted_all_rel_scores = self.semantic_pruning_triplets_batch(question, listp, rel_embeddings=None, topk=pruning, batch_size=8)
        #         knowledge_sequences_pruning = [rel for rel, _ in sorted_all_rel_scores]
        #         knowledge_sequences_result.append(knowledge_sequences_pruning)
        # return knowledge_sequences_result

    # 通过带symbol的path过滤，返回带symbol的path
    def retrieve_path_with_keywords_v1(self, question, keywords = [], path_depth = 2, pruning = None, build_node = False):
        self.pruning = pruning

        rel_map = self.graph_database.get_rel_map(entities=keywords, depth=path_depth, limit=1000000)
        # clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map)
        # clean_rel_map_list = [sublist for sublist in clean_rel_map.values()]

        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map, keywords = keywords) # 加入keywords确保按照keywords排序，可以一一对应
        clean_rel_map_list = []
        for keyword in keywords:
            if keyword in clean_rel_map:
                clean_rel_map_list.append(clean_rel_map[keyword])
        # print(f"clean_rel_map_list\n{json.dumps(clean_rel_map_list, indent=2)}")
        # print(f"sentences_2d\n{json.dumps(sentences_2d, indent=2)}")
        
        result_filter = []
        result_triple = []
        result_sentence = []
        if pruning > 0:
            # for sentences_list  in sentences_2d: # 不要通过句子过滤了，还是改为path
            for path_list, sentence_to_triple_3d, sentences in zip(clean_rel_map_list, sentence_to_triple_4d, sentences_2d):
                rel_scores = self.semantic_pruning_sentences_batch(question, path_list, rel_embeddings = None, topk = pruning, batch_size = 8)

                for idx, score in rel_scores:
                    result_filter.append(path_list[idx])
                    result_triple.append(sentence_to_triple_3d[idx])
                    result_sentence.append(sentences[idx])
                            
        return result_filter, result_triple, result_sentence
    
    # 通过带symbol的path过滤，返回带symbol的path
    def retrieve_path_with_keywords_v1_data(self, question, keywords = [], path_depth = 2, pruning = None, build_node = False):
        self.pruning = pruning

        rel_map = self.graph_database.get_rel_map(entities=keywords, depth=path_depth, limit=1000000)
        # clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map)
        # clean_rel_map_list = [sublist for sublist in clean_rel_map.values()]

        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map, keywords = keywords) # 加入keywords确保按照keywords排序，可以一一对应
        clean_rel_map_list = []
        for keyword in keywords:
            if keyword in clean_rel_map:
                clean_rel_map_list.append(clean_rel_map[keyword])
        # print(f"clean_rel_map_list\n{json.dumps(clean_rel_map_list, indent=2)}")
        # print(f"sentences_2d\n{json.dumps(sentences_2d, indent=2)}")
        
        result_filter = []
        result_triple = []
        result_sentence = []
        if pruning > 0:
            # for sentences_list  in sentences_2d: # 不要通过句子过滤了，还是改为path
            for path_list, sentence_to_triple_3d, sentences in zip(clean_rel_map_list, sentence_to_triple_4d, sentences_2d):
                rel_scores = self.semantic_pruning_sentences_batch(question, path_list, rel_embeddings = None, topk = pruning, batch_size = 8)
                result_filter_tmp_for_4d = []
                result_triple_tmp_for_4d = []
                result_sentence_tmp_for_4d = []
                for idx, score in rel_scores:
                    # result_filter.append(path_list[idx])
                    # result_triple.append(sentence_to_triple_3d[idx])
                    # result_sentence.append(sentences[idx])
                    result_filter_tmp_for_4d.append(path_list[idx])
                    result_triple_tmp_for_4d.append(sentence_to_triple_3d[idx])
                    result_sentence_tmp_for_4d.append(sentences[idx])
                result_filter.append(result_filter_tmp_for_4d)
                result_triple.append(result_triple_tmp_for_4d)
                result_sentence.append(result_sentence_tmp_for_4d)
                            
        return result_filter, result_triple, result_sentence
    
    def sort_strings_by_frequency_unique(self, string_list):
        from collections import Counter
        if not string_list:
            return []

        frequency_counter = Counter(string_list)

        sorted_items = frequency_counter.most_common()

        unique_sorted_strings = [item[0] for item in sorted_items]

        return unique_sorted_strings

    # 用于相似实体检索二跳邻居，返回连通度最高的一跳邻居
    def retrieve_path_for_redundant_entity(self, redundant_entity = [], path_depth = 2, pruning = 5):
        self.pruning = pruning
        keywords = [item for group in redundant_entity for item in group]

        rel_map = self.graph_database.get_rel_map(entities=keywords, depth=path_depth, limit=1000000)
        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map, keywords = keywords) # 已经有了一次过滤，加入关键词可以实现输出路径按照关键词排序
        # clean_rel_map_list = [sublist for sublist in clean_rel_map.values()] # 发现了一个问题，clean_rel_map如果按照关键词排序，字典按照插入顺序排序则是对应的，如果无序则不对应，保险起见不这么写
        clean_rel_map_list = []
        for keyword in keywords:
            if keyword in clean_rel_map:
                clean_rel_map_list.append(clean_rel_map[keyword])

        res_sentences_2d = []
        for sentence_to_triple_3d  in sentence_to_triple_4d:
            sentence_list = []
            # seen = set()
            for sentence_to_triple_2d in sentence_to_triple_3d:
                if sentence_to_triple_2d[0][3] == '->':
                    sentence_tmp = f"{sentence_to_triple_2d[0][0]} {sentence_to_triple_2d[0][1].replace('_',' ')} {sentence_to_triple_2d[0][2]}"
                elif sentence_to_triple_2d[0][3] == '<-':
                    sentence_tmp = f"{sentence_to_triple_2d[0][2]} {sentence_to_triple_2d[0][1].replace('_',' ')} {sentence_to_triple_2d[0][0]}"
                else:
                    print(f"Error in KG_Retriever.py retrieve_path_with_keywords_v3 function")
                # if sentence_tmp not in seen:
                #     seen.add(sentence_tmp)
                sentence_list.append(sentence_tmp)
            sentence_frequency = self.sort_strings_by_frequency_unique(sentence_list)
            res_sentences_2d.append(sentence_frequency[:pruning])
        res_sentences_3d = []
        
        idx = 0
        # print(f"redundant_entity {json.dumps(redundant_entity, indent=2)}")
        for group in redundant_entity:
            list_tmp = []
            for idy in range(len(group)):
                list_tmp.append(res_sentences_2d[idy+idx])
            res_sentences_3d.append(list_tmp)
            idx += len(group)

        return res_sentences_3d

    # +++
    # 把二维三元组转化为path_with_symbol
    def build_path_simple(self, four_tuples):
        if not four_tuples:
            return ""

        path = []

        for i, (e1, rel, e2, direction) in enumerate(four_tuples):
            if i == 0:
                if direction == "->":
                    path.append(e1)
                    path.append(f" - {rel} -> ")
                    path.append(e2)
                else:
                    path.append(e2)
                    path.append(f" <- {rel} - ")
                    path.append(e1)
            else:
                # 检查是否能连接到末尾
                last_node = path[-1]
                if direction == "->":
                    if e1 == last_node:
                        path.append(f" - {rel} -> ")
                        path.append(e2)
                    else:
                        path.append(f", {e1} - {rel} -> ")
                        path.append(e2)
                    # else:
                    #     assert False, "Path cannot be connected to the last node"
                else:
                    if e1 == last_node:
                        path.append(f" <- {rel} - ")
                        path.append(e2)
                    else:
                        path.append(f", {e1} <- {rel} - ")
                        path.append(e2)
                    # else:
                    #     assert False, "Path cannot be connected to the last node"
        return " ".join(path)
    
    # +++
    # 使用相似度和分数联合过滤，对于每个实体保留pruning个(改用带symbol的path进行向量相似度过滤而不是sentence)
    def retrieve_path_with_keywords_basic(self, question, keywords = [], path_depth = 2, pruning = 10, threshold=0.55, score_threshold = 71, build_node = False):
        nebula_retrieve_start_time = time.perf_counter()
        self.pruning = pruning
        triplets_score = self.triplets_score
        
        
        rel_map = self.graph_database.get_rel_map(entities=keywords, depth=path_depth, limit=1000000)
        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map, keywords = keywords) # 已经有了一次过滤，加入关键词可以实现输出路径按照关键词排序
        # clean_rel_map_list = [sublist for sublist in clean_rel_map.values()] # 发现了一个问题，clean_rel_map如果按照关键词排序，字典按照插入顺序排序则是对应的，如果无序则不对应，保险起见不这么写
        clean_rel_map_list = []
        for keyword in keywords:
            if keyword in clean_rel_map:
                clean_rel_map_list.append(clean_rel_map[keyword])
        nebula_retrieve_end_time = time.perf_counter()
        filter_retrieve_start_time = time.perf_counter()


        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []

        result_filter = []  # 经过向量相似度阈值过滤的所有结果 3d
        result_final = [] # 经过向量相似度和分数联合过滤的结果--sentence
        result_final_symbol = [] # 经过向量相似度和分数联合过滤的结果--symbol
        result_final_to_triple_3d = []
        result_final_to_triple_3d_more = []
        if pruning > 0:
            # for sentences_list, sentence_to_triple_3d  in zip(sentences_2d, sentence_to_triple_4d):
            # for index_temp, (sentences_list, sentence_to_triple_3d)  in enumerate(zip(sentences_2d, sentence_to_triple_4d)): # 用句子生成嵌入
            for index_temp, (sentences_list, sentence_to_triple_3d)  in enumerate(zip(clean_rel_map_list, sentence_to_triple_4d)): # 用符号形式的path生成嵌入

                # 路径嵌入复用 太慢了
                # path2id, path_embeddings = self.graph_database.generate_path_embeddings_with_cache(sentences_list)
                # print(f"retrieve {index_temp} len(path2id):{len(path2id)} len(path_embeddings):{len(path_embeddings)}")
                
                # path_id_list = [path2id[path] for path in sentences_list]
                # selected_embeddings = path_embeddings[path_id_list]
                selected_embeddings = None

                # rel_scores, max_score, min_score, median_score, mean_score = self.semantic_pruning_sentences_batch_v2(question, sentences_list, rel_embeddings=None, batch_size=8, threshold = threshold)
                rel_scores, max_score, min_score, median_score, mean_score = self.semantic_pruning_sentences_batch_v2(question, sentences_list, rel_embeddings=selected_embeddings, batch_size=128, threshold = threshold)
                if max_score:
                    max_score_avg.append(max_score)
                    min_score_avg.append(min_score)
                    median_score_avg.append(median_score)
                    mean_score_avg.append(mean_score)

                index = 0
                index_tmp = []
                result_final_tmp = []
                result_final_symbol_tmp = []
                result_final_to_triple_tmp = []
                for idx, score in rel_scores:
                    result_filter.append(sentence_to_triple_3d[idx])
                    mapped_sim = (score + 1) * 50  # 映射到0-100 score变量使用完毕
                    
                    score = 1
                    count = 0
                    flag = 0
                    triples_tmp = []
                    symbol_tmp = []
                    sentence_str = ''
                    for triple in sentence_to_triple_3d[idx]:
                        if not (len(triple) == 4):
                            print(f"The length of the triple is not 4: {triple}")
                            flag = 1
                            break
                        if triple[3] == "->":
                            key = f"{triple[0]} {triple[1]} {triple[2]}"
                            sentence_tmp = f"{triple[0]} {triple[1].replace('_',' ')} {triple[2]}"
                        elif triple[3] == "<-":
                            key = f"{triple[2]} {triple[1]} {triple[0]}"
                            sentence_tmp = f"{triple[2]} {triple[1].replace('_',' ')} {triple[0]}"
                        else:
                            print(f"The appearance of position and direction symbols in triple: {triple}")
                            flag = 1
                            break
                        if key not in triplets_score:
                            print(f"triple not exist: {triple}")
                            flag = 1
                            break
                        elif key in triplets_score and triplets_score[key]["score"] < score_threshold: #还是得保留一部分
                            # flag = 1
                            continue # 真的不能break
                        elif key in triplets_score and triplets_score[key]["score"] >= score_threshold:
                            score *= triplets_score[key]["score"]
                            count += 1
                            triples_tmp.append(triple)
                            if sentence_str:
                                sentence_str = sentence_str + ", " + sentence_tmp
                            else:
                                sentence_str = sentence_tmp
                    sentence_str += '.'
                    symbol_tmp = self.build_path_simple(triples_tmp)
                    if flag:
                        continue
                    elif count >= 1:
                        should_skip = False # 做一个子串过滤，但是可能代价非常大啊？
                        # 就这一块还是没想明白
                        # 隐去子串过滤
                        # for id_tmp, existing_sentence in enumerate(result_final_tmp):
                        #     if sentence_str == existing_sentence or \
                        #        sentence_str in existing_sentence:
                        #         # print(f"    Skipping append for '{sentence_str}' because it matches/is substring/is superstring of '{existing_sentence}'")
                        #         should_skip = True
                        #         break # 找到匹配/包含关系，无需再检查
                        #     elif existing_sentence in sentence_str:
                        #         index_tmp[id_tmp] = (id_tmp, mapped_sim + (score/(100**(count-1))) )
                        #         result_final_tmp[id_tmp] = sentence_str
                        #         result_final_symbol_tmp[id_tmp] = symbol_tmp
                        #         result_final_to_triple_tmp[id_tmp] = triples_tmp
                        #         should_skip = True
                        #         break

                        if not should_skip:
                            index_tmp.append((index, mapped_sim + (score/(100**(count-1))) ))
                            index += 1
                            result_final_tmp.append(sentence_str)
                            result_final_symbol_tmp.append(symbol_tmp)
                            result_final_to_triple_tmp.append(triples_tmp)

                index_tmp.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in index_tmp[:pruning]:
                    result_final.append(result_final_tmp[idx]) 
                    result_final_symbol.append(result_final_symbol_tmp[idx])  
                    result_final_to_triple_3d.append(result_final_to_triple_tmp[idx])
                # for idx, _ in index_tmp[:pruning2]:
                #     # result_final.extend(result_final_tmp)   
                #     # result_final_to_triple_3d.extend(result_final_to_triple_tmp)
                #     result_final_to_triple_3d_more.append(result_final_to_triple_tmp[idx])
        
        if max_score_avg:
            max_score = max(max_score_avg)
            min_score = min(min_score_avg)
            median_score = statistics.median(median_score_avg)
            mean_score = sum(mean_score_avg) / len(mean_score_avg)
        else:
            max_score = min_score = median_score = mean_score = None

        # return result_filter, result_final, result_final_to_triple_3d, result_final_to_triple_3d_more, max_score, min_score, median_score, mean_score
        filter_retrieve_end_time = time.perf_counter()
        return result_final_symbol, result_final_to_triple_3d, nebula_retrieve_end_time - nebula_retrieve_start_time, filter_retrieve_end_time - filter_retrieve_start_time # 返回symbol形式的结果
        # return result_final, result_final_to_triple_3d, nebula_retrieve_end_time - nebula_retrieve_start_time, filter_retrieve_end_time - filter_retrieve_start_time # 返回句子形式的结果
        # return result_final, result_final_to_triple_3d

    # +++
    def retrieve_path_with_keywords_standard(self, question, keywords = [], path_depth = 2, pruning = 10, score_weight = 0.5, top_k_per_entity = True, build_node = False):
        nebula_retrieve_start_time = time.perf_counter()
        self.pruning = pruning
        triplets_score = self.triplets_score

        # 检索不变
        rel_map = self.graph_database.get_rel_map(entities=keywords, depth=path_depth, limit=1000000)
        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map, keywords = keywords) # 已经有了一次过滤，加入关键词可以实现输出路径按照关键词排序
        # clean_rel_map_list = [sublist for sublist in clean_rel_map.values()] # 发现了一个问题，clean_rel_map如果按照关键词排序，字典按照插入顺序排序则是对应的，如果无序则不对应，保险起见不这么写
        clean_rel_map_list = []
        for keyword in keywords:
            if keyword in clean_rel_map:
                clean_rel_map_list.append(clean_rel_map[keyword]) # 一个symbol sentence的嵌套，2d
        nebula_retrieve_end_time = time.perf_counter()
        filter_retrieve_start_time = time.perf_counter()
        # 字典应该不是最优解，换成3个一一对应的列表
        result_dict = {} # tiple字典，不重复
        triple_list_standard = [] # 用于存储所有的triple
        sim_list_standard = [] # 用于存储所有的相似度
        score_list_standard = [] # 用于存储所有的分数
        pt_list_standard = [] # 用于存储所有的三元组选择概率
        pl_list_standard = [] # 用于存储所有的路径选择概率
        sentence_list = [] # 句子列表
        seen = set()
        for sentence_to_triple_3d in sentence_to_triple_4d:
            for path in sentence_to_triple_3d:
                for triple in path:
                    if not (len(triple) == 4):
                        print(f"function retrieve_path_with_keywords_standard: The length of the triple is not 4: {triple}")
                    else:
                        if triple[3] == "->":
                            key = f"{triple[0]} {triple[1]} {triple[2]}"
                            sentence_tmp = f"{triple[0]} {triple[1].replace('_',' ')} {triple[2]}"
                        elif triple[3] == "<-":
                            key = f"{triple[2]} {triple[1]} {triple[0]}"
                            sentence_tmp = f"{triple[2]} {triple[1].replace('_',' ')} {triple[0]}"
                        else:
                            continue

                        # if sentence_tmp not in seen:
                        if key not in result_dict:
                            # seen.add(sentence_tmp)
                            result_dict[key] = len(result_dict)
                            triple_list_standard.append(triple)
                            sentence_list.append(sentence_tmp)
                            score_list_standard.append(triplets_score[key]["score"])

        # self.logger.log(f"{json.dumps(result_dict, indent=2)}")
        # === 批量生成嵌入 ===
        # print(f"sentence_list: {sentence_list}")
        # assert False
        if sentence_list:
            # 调用批量嵌入函数
            # 假设返回的是一个形状为 [N, dim] 的 tensor 或 numpy 数组
            # v3版本没有阈值过滤这一说法
            
            # 嵌入复用
            if hasattr(self.graph_database, 'triplet2id'):
                triplet_id_list = [self.graph_database.triplet2id[triplet] for triplet in sentence_list]
                # print(type(self.graph_database.triplet_embeddings))
                # assert False
                selected_embeddings = self.graph_database.triplet_embeddings[triplet_id_list]
            else:
                selected_embeddings = None

            # print(f"selected_embeddings{selected_embeddings}")
            # assert False
            # 计算余弦相似度
            embedding_similarity = self.semantic_pruning_sentences_batch_v3(
                question=question,
                sentences=sentence_list,
                # rel_embeddings=selected_embeddings if selected_embeddings else None,
                rel_embeddings=selected_embeddings,
                batch_size=8,
            )
            
            # for idx, sim_score in embedding_similarity:
            #     sim_list_standard.append((sim_score+1)/2)
            # sim_list_standard = [(score + 1) / 2 for _, score in embedding_similarity]
            # print(f"type(embedding_similarity):{type(embedding_similarity)}")
            # print(embedding_similarity)
            # assert False
            sim_list_standard = [(score + 1) / 2 for score in embedding_similarity]
            # print(f"type(sim_list_standard):{type(sim_list_standard)}")
            # print(f"sim_list_standard: {sim_list_standard}")

        scores_array = np.array(score_list_standard)  # → numpy.ndarray
        # print(f"scores_array: {scores_array}")
        # self.logger.log(f"sentence_list and sim_list_standard:")
        # for idx, (sentence, sim) in enumerate(sorted(list(zip(sentence_list, sim_list_standard)), key=lambda x: x[1], reverse=True)):
        #     # print(f"sentence: {sentence}, sim: {sim}")
        #     self.logger.log(f"question: {question[:30]} sentence{idx}: {sentence}, sim: {sim}")  

        # 归一化
        if scores_array.max() == scores_array.min():
            # score_normalized = np.ones_like(scores_array) # 全1
            score_normalized = np.full_like(scores_array, 0.5, dtype=float) # 全0.5
        else:
            # 如果直接这样启动，那么将得到0，那样归一化之后是nan
            score_normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())

        sim_map = {}
        score_map = {}
        for sim, socre, sentence in zip(sim_list_standard, score_normalized, sentence_list):
            sim_map[sentence] = sim
            score_map[sentence] = socre

        # print(f"score_normalized: {score_normalized}")
        # assert False

        # 计算P(t)
        if sim_list_standard:
            for sim, score in zip(sim_list_standard, score_normalized):
                # 计算三元组选择概率
                # pt = (1 - score_weight) * sim + score_weight * score * sim
                pt = (1 - score_weight) * sim + score_weight * score
                pt_list_standard.append(pt)
        
        # self.logger.log(f"sentence_list and pt_list_standard:")
        # for idx, (sentence, pt) in enumerate(sorted(list(zip(sentence_list, pt_list_standard)), key=lambda x: x[1], reverse=True)):
        #     # print(f"sentence: {sentence}, sim: {sim}")
        #     self.logger.log(f"question: {question[:30]} sentence{idx}: {sentence}, pt: {pt}")  

        # self.logger.log(f"path_list and pl_path:")
        path_list= []
        for sentence_to_triple_3d  in sentence_to_triple_4d:
            for path in sentence_to_triple_3d:
                pl_tmp = 1
                path_length = 0
                pl_str = ""
                path_str = ""
                for triple in path:
                    if not (len(triple) == 4):
                        print(f"function retrieve_path_with_keywords_standard: The length of the triple is not 4: {triple}")
                    else:
                        if triple[3] == "->":
                            key = f"{triple[0]} {triple[1]} {triple[2]}"
                        elif triple[3] == "<-":
                            key = f"{triple[2]} {triple[1]} {triple[0]}"
                        pl_tmp *= pt_list_standard[result_dict[key]]
                        pl_str += f"pt({pt_list_standard[result_dict[key]]}) * "
                        path_str += f"{key} -> "
                        path_length += 1
                pl_tmp = math.log(pl_tmp) / path_length
                pl_list_standard.append(pl_tmp)
                path_list.append(path)

                # self.logger.log(f"question: {question[:30]} path: {path_str} pl: {pl_tmp} ({pl_str} | pl_end: {pl_tmp})")

        # softmax
        pl_array = np.array(pl_list_standard, dtype=float)

        # 数值稳定的 softmax（减去最大值，避免溢出）
        exp_pl = np.exp(pl_array - np.max(pl_array))
        softmax_probs = exp_pl / np.sum(exp_pl)
        softmax_probs = softmax_probs.tolist() # 是和sentence_to_triple_4d一一对应的，以及res_symbol_list、res_triple_3d一一对应

        # self.logger.log(f"path_list and softmax_probs:")  
        # for idx, (path, prob) in enumerate(sorted(list(zip(path_list, softmax_probs)), key=lambda x: x[1], reverse=True)):
        #     # print(f"sentence: {sentence}, sim: {sim}")
        #     self.logger.log(f"question: {question[:30]} path{idx}: {path}, prob: {prob}")  

        # softmax_probs = pl_list_standard

        # 按照每个实体进行选取
        if top_k_per_entity:
            softmax_probs_2d = []
            idx = 0  # 全局指针
            for path_sublist in clean_rel_map_list:
                path_sublist_probs = []
                for path in path_sublist:
                    path_sublist_probs.append(softmax_probs[idx])
                    idx += 1
                softmax_probs_2d.append(path_sublist_probs)

            final_symbols = []
            final_triples = []
            final_probs = []

            for res_softmax_probs, res_symbol_list, res_triple_3d in zip(softmax_probs_2d, clean_rel_map_list, sentence_to_triple_4d):
                zipped = sorted(zip(res_softmax_probs, res_symbol_list, res_triple_3d), 
                    key=lambda x: x[0], reverse=True)

                top_k_zipped = zipped[:pruning]
                topk_probs, topk_symbols, topk_triples = zip(*top_k_zipped) if top_k_zipped else ([], [], [])

                # 转为 list（zip 返回的是 tuple）
                final_probs.extend(list(topk_probs))
                final_symbols.extend(list(topk_symbols))
                final_triples.extend(list(topk_triples))  # 每个元素仍是 list（即二维列表）
            # 结果句子 结果三元组 结果概率 字典：单跳对应序号（理论上就是有序的）三元组概率
            # result_dict, pt_list_standard  同时存在是多此一举啊
            filter_retrieve_end_time = time.perf_counter()
            return final_symbols, final_triples, final_probs, result_dict, pt_list_standard, sim_map, score_map, nebula_retrieve_end_time - nebula_retrieve_start_time, filter_retrieve_end_time - filter_retrieve_start_time

        else:
            res_symbol_list = [ sentence_symbol for sublist in clean_rel_map_list for sentence_symbol in sublist ] # 一维列表
            res_triple_3d = [ sentence_to_triple_2d for sublist in sentence_to_triple_4d for sentence_to_triple_2d in sublist ] # 二维列表

            zipped = sorted(zip(softmax_probs, res_symbol_list, res_triple_3d), 
                key=lambda x: x[0], reverse=True)

            top_k_zipped = zipped[:pruning]
            topk_probs, topk_symbols, topk_triples = zip(*top_k_zipped) if top_k_zipped else ([], [], [])

            # 转为 list（zip 返回的是 tuple）
            topk_probs = list(topk_probs)
            topk_symbols = list(topk_symbols)
            topk_triples = list(topk_triples)  # 每个元素仍是 list（即二维列表）
            filter_retrieve_end_time = time.perf_counter()
            return topk_symbols, topk_triples, topk_probs, result_dict, pt_list_standard, sim_map, score_map, nebula_retrieve_end_time - nebula_retrieve_start_time, filter_retrieve_end_time - filter_retrieve_start_time


    
    # 使用相似度和分数联合过滤，对于每个实体保留pruning个(改用带symbol的path而不是sentence)
    def retrieve_path_with_keywords_v2_data(self, question, triplets_score, keywords = [], path_depth = 2, pruning = None, pruning2 = 30, threshold=0.65, score_threshold = 95, build_node = False):
        self.pruning = pruning

        rel_map = self.graph_database.get_rel_map(entities=keywords, depth=path_depth, limit=1000000)
        clean_rel_map, sentence_to_triple_4d, sentences_2d = self.graph_database.clean_rel_map(rel_map, keywords = keywords) # 已经有了一次过滤，加入关键词可以实现输出路径按照关键词排序
        # clean_rel_map_list = [sublist for sublist in clean_rel_map.values()] # 发现了一个问题，clean_rel_map如果按照关键词排序，字典按照插入顺序排序则是对应的，如果无序则不对应，保险起见不这么写
        clean_rel_map_list = []
        for keyword in keywords:
            if keyword in clean_rel_map:
                clean_rel_map_list.append(clean_rel_map[keyword])

        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []

        result_filter = []
        result_final = []
        result_final_to_triple_3d = []
        result_final_to_triple_3d_more = []
        if pruning > 0:
            # for sentences_list, sentence_to_triple_3d  in zip(sentences_2d, sentence_to_triple_4d):
            for sentences_list, sentence_to_triple_3d  in zip(clean_rel_map_list, sentence_to_triple_4d):
                rel_scores, max_score, min_score, median_score, mean_score = self.semantic_pruning_sentences_batch_v2(question, sentences_list, rel_embeddings=None, batch_size=8, threshold = threshold)
                if max_score:
                    max_score_avg.append(max_score)
                    min_score_avg.append(min_score)
                    median_score_avg.append(median_score)
                    mean_score_avg.append(mean_score)

                index = 0
                index_tmp = []
                result_final_tmp = []
                result_final_to_triple_tmp = []
                for idx, score in rel_scores:
                    result_filter.append(sentence_to_triple_3d[idx])
                    mapped_sim = (score + 1) * 50  # 映射到0-100 score变量使用完毕
                    
                    score = 1
                    count = 0
                    flag = 0
                    triples_tmp = []
                    sentence_str = ''
                    for triple in sentence_to_triple_3d[idx]:
                        if not (len(triple) == 4):
                            print(f"The length of the triple is not 4: {triple}")
                            flag = 1
                            break
                        if triple[3] == "->":
                            key = f"{triple[0]} {triple[1]} {triple[2]}"
                            sentence_tmp = f"{triple[0]} {triple[1].replace('_',' ')} {triple[2]}"
                        elif triple[3] == "<-":
                            key = f"{triple[2]} {triple[1]} {triple[0]}"
                            sentence_tmp = f"{triple[2]} {triple[1].replace('_',' ')} {triple[0]}"
                        else:
                            print(f"The appearance of position and direction symbols in triple: {triple}")
                            flag = 1
                            break
                        if key not in triplets_score:
                            print(f"triple not exist: {triple}")
                            flag = 1
                            break
                        elif key in triplets_score and triplets_score[key]["score"] < score_threshold: #还是得保留一部分
                            # flag = 1
                            continue # 真的不能break
                        elif key in triplets_score and triplets_score[key]["score"] >= score_threshold:
                            score *= triplets_score[key]["score"]
                            count += 1
                            triples_tmp.append(triple)
                            if sentence_str:
                                sentence_str = sentence_str + ", " + sentence_tmp
                            else:
                                sentence_str = sentence_tmp
                    sentence_str += '.'
                    if flag:
                        continue
                    elif count >= 1:
                        should_skip = False # 做一个子串过滤，但是可能代价非常大啊？
                        # 就这一块还是没想明白
                        for id_tmp, existing_sentence in enumerate(result_final_tmp):
                            if sentence_str == existing_sentence or \
                               sentence_str in existing_sentence:
                                # print(f"    Skipping append for '{sentence_str}' because it matches/is substring/is superstring of '{existing_sentence}'")
                                should_skip = True
                                break # 找到匹配/包含关系，无需再检查
                            elif existing_sentence in sentence_str:
                                index_tmp[id_tmp] = (id_tmp, mapped_sim + (score/(100**(count-1))) )
                                result_final_tmp[id_tmp] = sentence_str
                                result_final_to_triple_tmp[id_tmp] = triples_tmp
                                should_skip = True
                                break

                        if not should_skip:
                            index_tmp.append((index, mapped_sim + (score/(100**(count-1))) ))
                            index += 1
                            # result_final_tmp.append(sentences_list[idx])
                            # result_final_to_triple_tmp.append(sentence_to_triple_3d[idx])
                            result_final_tmp.append(sentence_str)
                            result_final_to_triple_tmp.append(triples_tmp)

                index_tmp.sort(key=lambda x: x[1], reverse=True)
                result_final_tmp_for_4d = []
                result_final_to_triple_3d_tmp_for_4d = []
                for idx, _ in index_tmp[:pruning]:
                    # result_final.extend(result_final_tmp)   
                    # result_final_to_triple_3d.extend(result_final_to_triple_tmp)
                    # result_final.append(result_final_tmp[idx])   
                    # result_final_to_triple_3d.append(result_final_to_triple_tmp[idx])

                    result_final_tmp_for_4d.append(result_final_tmp[idx])
                    result_final_to_triple_3d_tmp_for_4d.append(result_final_to_triple_tmp[idx])
                result_final.append(result_final_tmp_for_4d)
                result_final_to_triple_3d.append(result_final_to_triple_3d_tmp_for_4d)
                result_final_to_triple_3d_more_tmp_for_4d = []
                for idx, _ in index_tmp[:pruning2]:
                    # result_final.extend(result_final_tmp)   
                    # result_final_to_triple_3d.extend(result_final_to_triple_tmp)
                    # result_final_to_triple_3d_more.append(result_final_to_triple_tmp[idx])
                    result_final_to_triple_3d_more_tmp_for_4d.append(result_final_to_triple_tmp[idx])
                result_final_to_triple_3d_more.append(result_final_to_triple_3d_more_tmp_for_4d)
        
        if max_score_avg:
            max_score = max(max_score_avg)
            min_score = min(min_score_avg)
            median_score = statistics.median(median_score_avg)
            mean_score = sum(mean_score_avg) / len(mean_score_avg)
        else:
            max_score = min_score = median_score = mean_score = None

        return result_filter, result_final, result_final_to_triple_3d, result_final_to_triple_3d_more, max_score, min_score, median_score, mean_score
    
    def get_nodes(self):
        return self.nodes

    def path_to_triples(self, paths):
        return self.graph_database.path_to_triples(paths)

    def triple_into_sentence(self, paths):
        return self.graph_database.triple_into_sentence(paths)
    
    def calculate_similarity(self, embedding1, embedding2):

        dot_product = np.dot(embedding1, embedding2)

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        similarity = dot_product / (norm1 * norm2)

        similarity_rounded = round(similarity, 6)

        return similarity_rounded # [-1,1]

    def filter_paths_by_score(self, query, paths, score_list):
        filter_list, filter_list_score = self.graph_database.filter_paths_by_score(paths, score_list)
        
        seen = set()
        unique_list = []
        unique_scores = []
        for s, score in zip(filter_list, filter_list_score):
            if s not in seen:
                seen.add(s)
                unique_list.append(s)
                unique_scores.append(score)

        scored = []
        embedding1 = get_text_embedding(query)
        for s, score in zip(unique_list, unique_scores):
            embedding2 = get_text_embedding(s)
            similarity = self.calculate_similarity(embedding1, embedding2)
            mapped_sim = (similarity + 1) * 50  # 映射到0-100
            total = mapped_sim + score
            # print(f"filter_paths_by_scor: {s} {mapped_sim} {score}")
            scored.append((s, total))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = [item[0] for item in scored]
        return result
    
    def filter_paths_by_score_v2(self, query, triples, score_list):
        filter_list, filter_list_score = self.graph_database.filter_paths_by_score_v2(triples, score_list)  # 3d 1d
        # print(f"KG_Retriever filter_list :\n{filter_list}")
        # print(f"KG_Retriever filter_list_score:\n{filter_list_score}")
        
        seen = set() # set应该并没有必要
        unique_list = []
        unique_scores = []
        unique_triple_3d = []
        for triples_list, score in zip(filter_list, filter_list_score):
            sentence = ""
            for triple_tmp in triples_list:
                if triple_tmp[3] == "->":
                    if not sentence:
                        sentence = sentence + triple_tmp[0] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[2]
                    else:
                        sentence = sentence + ", " + triple_tmp[0] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[2]
                elif triple_tmp[3] == "<-":
                    if not sentence:
                        sentence = sentence + triple_tmp[2] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[0]
                    else:
                        sentence = sentence + ", " + triple_tmp[2] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[0]
            sentence += "."
            if sentence not in seen:
                seen.add(sentence)
                unique_list.append(sentence)
                unique_scores.append(score)
                unique_triple_3d.append(triples_list)
        # print(f"KG_Retriever unique_list: \n{unique_list}")
        # print(f"KG_Retriever unique_scores: \n{unique_scores}")
        # print(f"KG_Retriever unique_triple_3d: \n{unique_triple_3d}")       

        scored = []
        embedding1 = get_text_embedding_MiniLM(query)
        for sentence, score, triples_list in zip(unique_list, unique_scores, unique_triple_3d):
            embedding2 = get_text_embedding_MiniLM(sentence)
            similarity = self.calculate_similarity(embedding1, embedding2)
            mapped_sim = (similarity + 1) * 50  # 映射到0-100
            total = mapped_sim + score
            # print(f"filter_paths_by_scor: {s} {mapped_sim} {score}")
            scored.append((sentence, total, triples_list))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        # print(f"KG_Retriever scored.sort: \n{json.dumps(scored, indent=2)}")
        
        result = [item[0] for item in scored] # 暂时就只返回句子，因为冗余处理是针对过滤前的检索内容
        result_triples = [item[2] for item in scored]
        return result, result_triples   # 1d 3d 
    

    def filter_paths_forward(self, query, triples, score_list):
        filter_list, filter_list_score = self.graph_database.filter_paths_by_score_v2(triples, score_list)  # 3d 1d
        # print(f"KG_Retriever filter_list :\n{filter_list}")
        # print(f"KG_Retriever filter_list_score:\n{filter_list_score}")
        
        seen = set() # set应该并没有必要
        unique_list = []
        unique_scores = []
        unique_triple_3d = []
        for triples_list, score in zip(filter_list, filter_list_score):
            sentence = ""
            for triple_tmp in triples_list:
                if triple_tmp[3] == "->":
                    if not sentence:
                        sentence = sentence + triple_tmp[0] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[2]
                    else:
                        sentence = sentence + ", " + triple_tmp[0] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[2]
                elif triple_tmp[3] == "<-":
                    if not sentence:
                        sentence = sentence + triple_tmp[2] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[0]
                    else:
                        sentence = sentence + ", " + triple_tmp[2] +' '+ triple_tmp[1].replace("_"," ") +' '+ triple_tmp[0]
            sentence += "."
            if sentence not in seen:
                seen.add(sentence)
                unique_list.append(sentence)
                unique_scores.append(score)
                unique_triple_3d.append(triples_list)
        # print(f"KG_Retriever unique_list: \n{unique_list}")
        # print(f"KG_Retriever unique_scores: \n{unique_scores}")
        # print(f"KG_Retriever unique_triple_3d: \n{unique_triple_3d}")       

        scored = []
        embedding1 = get_text_embedding_MiniLM(query)
        for sentence, score, triples_list in zip(unique_list, unique_scores, unique_triple_3d):
            embedding2 = get_text_embedding_MiniLM(sentence)
            similarity = self.calculate_similarity(embedding1, embedding2)
            mapped_sim = (similarity + 1) * 50  # 映射到0-100
            total = mapped_sim + score
            # print(f"filter_paths_by_scor: {s} {mapped_sim} {score}")
            scored.append((sentence, total, triples_list))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        # print(f"KG_Retriever scored.sort: \n{json.dumps(scored, indent=2)}")
        
        result = [item[0] for item in scored] # 暂时就只返回句子，因为冗余处理是针对过滤前的检索内容
        result_triples = [item[2] for item in scored]
        return result, result_triples   # 1d 3d 
    
    def find_redundant_relationship(self, retrieve_result):
        result = self.graph_database.path_to_triples_grouped(retrieve_result)
        return result
    
    # +++
    def find_redundant_relationship_by_entity(self, filtered_triple_3d_more):
        triplet_group_dict = {}
        sentence_group_dict = {}

        for triple_2d in filtered_triple_3d_more:
            for triple in triple_2d:
                entity1, relation, entity2, direction = triple

                # 根据方向决定实际三元组顺序
                if direction == '->':
                    triplet = (entity1, relation, entity2)
                    sentence = f"{entity1} {relation} {entity2}"
                elif direction == '<-':
                    triplet = (entity2, relation, entity1)
                    sentence = f"{entity2} {relation} {entity1}"
                else:
                    raise ValueError(f"Unknown direction: {direction}")

                # 实体对排序，使 (A, B) 和 (B, A) 被视为相同键
                key_entities = tuple(sorted((entity1, entity2)))

                # 初始化字典项
                if key_entities not in triplet_group_dict:
                    triplet_group_dict[key_entities] = []
                if key_entities not in sentence_group_dict:
                    sentence_group_dict[key_entities] = []

                # 避免重复添加相同的 triplet 和 sentence
                if triplet not in triplet_group_dict[key_entities]:
                    triplet_group_dict[key_entities].append(triplet)
                    sentence_group_dict[key_entities].append(sentence)

        # 将字典转换为列表
        # triplet_result = [{key: value} for key, value in triplet_group_dict.items()]
        # sentence_result = [{key: value} for key, value in sentence_group_dict.items()]
        triple_group_list = list(triplet_group_dict.values())
        sentence_group_list = list(sentence_group_dict.values())

        triple_group_list_result = [sublist for sublist in triple_group_list if len(sublist) > 1]
        sentence_group_list_result = [sublist for sublist in sentence_group_list if len(sublist) > 1]

        # return triplet_result, sentence_result
        return triple_group_list_result, sentence_group_list_result
    
    def group_triples(
        self,
        triples,
        method="agglo",          # "agglo" 或 "dbscan"
        embed_method="joint",    # "joint" 或 "average"
        distance_threshold=0.05,  # 控制分组严格度
        eps=0.05,                # DBSCAN参数
        min_samples=2           # DBSCAN参数
    ):
        import numpy as np

        if embed_method == "joint":
            # texts = [f"{h} and {t}" for h, _, t in triples]
            texts = [f"{h} {r.replace('_',' ')} {t}" for h, r, t in triples]
            embeddings = np.array(get_text_embeddings(texts))
        else:  # average
            heads = [h for h, _, _ in triples]
            tails = [t for _, _, t in triples]
            head_embeds = np.array(get_text_embeddings(heads))
            tail_embeds = np.array(get_text_embeddings(tails))
            embeddings = (head_embeds + tail_embeds) / 2

        if method == "agglo":
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=distance_threshold
            )
        else:  # dbscan
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine'
            )
        labels = clustering.fit_predict(embeddings)

        groups = {}
        for idx, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(triples[idx])
        
        return list(groups.values())    
    
    def extract_numbers(self, text):
        return re.findall(r'\d+\.?\d*', text)
    
    # 处理entity之后，在不重新检索的条件下，按照融合的实体重新组织关系
    def find_redundant_relationship_v3(self, retrieve_triple_3d, keep_and_delete_entity_list_2d, distance_threshold = 0.15):
        entity_dict = {}
        for group in keep_and_delete_entity_list_2d:
            if len(group) > 1:
                for entity in group[1:]:
                    entity_dict[str(entity)] = group[0]

        seen_origin = set()
        seen = set()
        triple_list = []
        all_numbers_set = set()
        for triple_2d in retrieve_triple_3d:
            for triple in triple_2d:
                triple_copy = []
                for item in triple:# 元组转换为列表，并且进行实体替换
                    if item in entity_dict:
                        triple_copy.append(entity_dict[item])
                    else:
                        triple_copy.append(item)
                if triple_copy[3] == '->':
                    sentence = triple_copy[0] +' '+ triple_copy[1] +' '+ triple_copy[2] # 替换后的三元组
                    if sentence not in seen:
                        seen.add(sentence)
                        triple_list.append([triple_copy[0], triple_copy[1], triple_copy[2]])
                        numbers = self.extract_numbers(sentence)
                        all_numbers_set.update(numbers)

                    sentence_origin = triple[0] +' '+ triple[1] +' '+ triple[2] # 同时保留原三元组
                    if sentence_origin not in seen_origin:
                        seen_origin.add(sentence_origin)

                elif triple_copy[3] == "<-":
                    sentence = triple_copy[2] +' '+ triple_copy[1] +' '+ triple_copy[0]
                    if sentence not in seen:
                        seen.add(sentence)
                        triple_list.append([triple_copy[2], triple_copy[1], triple_copy[0]])
                        numbers = self.extract_numbers(sentence)
                        all_numbers_set.update(numbers)

                    sentence_origin = triple[2] +' '+ triple[1] +' '+ triple[0] # 同时保留原三元组
                    if sentence_origin not in seen_origin:
                        seen_origin.add(sentence_origin)
            
        unique_number_count = len(all_numbers_set)
        if unique_number_count > 25:
            distance_threshold = 0.05
        else:
            distance_threshold = 0.15

        redundant_relationship_3d = self.group_triples(triple_list, method="agglo", embed_method="joint", distance_threshold = distance_threshold)
        # group_triples(triples, method="agglo", embed_method="average")
        # print(f"redundant_relationship_3d: {redundant_relationship_3d}")
        all_relationship_group_str = ""
        relationship_count = 0
        for i, group in enumerate(redundant_relationship_3d):
            all_relationship_group_str += f"Group {i}:\n"
            for triple in group:
                all_relationship_group_str += f"{relationship_count}: {triple[0]} {triple[1].replace('_',' ')} {triple[2]}\n"
                relationship_count += 1
                
        fliter_relationship_3d = [
            mid_list for mid_list in redundant_relationship_3d  
            if len(mid_list) > 1 
        ]
                
        return fliter_relationship_3d, all_relationship_group_str, triple_list, unique_number_count, seen_origin

    # 处理entity之后，在不重新检索的条件下，按照融合的实体重新组织关系，通过一个映射map
    def find_redundant_relationship_v4(self, retrieve_triple_3d, replacement_map, distance_threshold = 0.15):
        seen_origin = set()
        seen = set()
        triple_list = []
        all_numbers_set = set()
        for triple_2d in retrieve_triple_3d:
            for triple in triple_2d:
                # triple是元组
                triple_copy = [replacement_map.get(triple[0], triple[0]), triple[1], replacement_map.get(triple[2], triple[2]), triple[3]]
                if triple_copy[3] == '->':
                    sentence = triple_copy[0] +' '+ triple_copy[1] +' '+ triple_copy[2] # 替换后的三元组
                    if sentence not in seen:
                        seen.add(sentence)
                        triple_list.append([triple_copy[0], triple_copy[1], triple_copy[2]])
                        numbers = self.extract_numbers(sentence)
                        all_numbers_set.update(numbers)

                    sentence_origin = triple[0] +' '+ triple[1] +' '+ triple[2] # 同时保留原三元组
                    if sentence_origin not in seen_origin:
                        seen_origin.add(sentence_origin)

                elif triple_copy[3] == "<-":
                    sentence = triple_copy[2] +' '+ triple_copy[1] +' '+ triple_copy[0]
                    if sentence not in seen:
                        seen.add(sentence)
                        triple_list.append([triple_copy[2], triple_copy[1], triple_copy[0]])
                        numbers = self.extract_numbers(sentence)
                        all_numbers_set.update(numbers)

                    sentence_origin = triple[2] +' '+ triple[1] +' '+ triple[0] # 同时保留原三元组
                    if sentence_origin not in seen_origin:
                        seen_origin.add(sentence_origin)
            
        unique_number_count = len(all_numbers_set)
        if unique_number_count > 25:
            distance_threshold = 0.05
        else:
            distance_threshold = 0.15

        redundant_relationship_3d = self.group_triples(triple_list, method="agglo", embed_method="joint", distance_threshold = distance_threshold)
        # group_triples(triples, method="agglo", embed_method="average")
        # print(f"redundant_relationship_3d: {redundant_relationship_3d}")
        all_relationship_group_str = ""
        relationship_count = 0
        for i, group in enumerate(redundant_relationship_3d):
            all_relationship_group_str += f"Group {i}:\n"
            for triple in group:
                all_relationship_group_str += f"{relationship_count}: {triple[0]} {triple[1].replace('_',' ')} {triple[2]}\n"
                relationship_count += 1
                
        fliter_relationship_3d = [
            mid_list for mid_list in redundant_relationship_3d  
            if len(mid_list) > 1 
        ]
                
        return fliter_relationship_3d, all_relationship_group_str, triple_list, unique_number_count, seen_origin

    def find_redundant_relationship_v2(self, retrieve_triple_3d, distance_threshold = 0.15):
        seen = set()
        triple_list = []
        all_numbers_set = set()
        for triple_2d in retrieve_triple_3d:
            for triple in triple_2d:
                if triple[3] == '->':
                    sentence = triple[0] +' '+ triple[1] +' '+ triple[2]
                    if sentence not in seen:
                        seen.add(sentence)
                        triple_list.append([triple[0], triple[1], triple[2]])
                        numbers = self.extract_numbers(sentence)
                        all_numbers_set.update(numbers)
                elif triple[3] == "<-":
                    sentence = triple[2] +' '+ triple[1] +' '+ triple[0]
                    if sentence not in seen:
                        seen.add(sentence)
                        triple_list.append([triple[2], triple[1], triple[0]])
                        numbers = self.extract_numbers(sentence)
                        all_numbers_set.update(numbers)
            
        unique_number_count = len(all_numbers_set)
        # if unique_number_count > 25:
        if unique_number_count > 30:
            # distance_threshold = 0.05
            pass
        else:
            # distance_threshold = 0.15
            # distance_threshold = 0.25
            pass

        redundant_relationship_3d = self.group_triples(triple_list, method="agglo", embed_method="joint", distance_threshold = distance_threshold)
        # group_triples(triples, method="agglo", embed_method="average")
        # print(f"redundant_relationship_3d: {redundant_relationship_3d}")
        all_relationship_group_str = ""
        relationship_count = 0
        for i, group in enumerate(redundant_relationship_3d):
            all_relationship_group_str += f"Group {i}:\n"
            for triple in group:
                all_relationship_group_str += f"{relationship_count}: {triple[0]} {triple[1].replace('_',' ')} {triple[2]}\n"
                relationship_count += 1
                
        fliter_relationship_3d = [
            mid_list for mid_list in redundant_relationship_3d  
            if len(mid_list) > 1 
        ]
                
        return fliter_relationship_3d, all_relationship_group_str, triple_list, unique_number_count
    
    def find_redundant_relationship_v2_two_stage(self, triple_2d):
        seen = set()
        triple_list = []
        for triple in triple_2d:
            sentence = triple[0] +' '+ triple[1] +' '+ triple[2]
            if sentence not in seen:
                seen.add(sentence)
                triple_list.append([triple[0], triple[1], triple[2]])
            
        redundant_relationship_3d = self.group_triples(triple_list, method="agglo", embed_method="joint", distance_threshold = 0.1)
        # group_triples(triples, method="agglo", embed_method="average")
        # print(f"redundant_relationship_3d: {redundant_relationship_3d}")
        all_relationship_group_str = ""
        relationship_count = 0
        for i, group in enumerate(redundant_relationship_3d):
            all_relationship_group_str += f"Group {i}:\n"
            for triple in group:
                all_relationship_group_str += f"{relationship_count}: {triple[0]} {triple[1].replace('_',' ')} {triple[2]}\n"
                relationship_count += 1
                
        fliter_relationship_3d = [
            mid_list for mid_list in redundant_relationship_3d  
            if len(mid_list) > 1 
        ]
                
        return fliter_relationship_3d, all_relationship_group_str, triple_list
        
    def group_entities(
        self,
        entities,
        method="agglo",          # "agglo" 或 "dbscan"
        distance_threshold=0.25,  # 控制分组严格度，（起作用）
        eps=0.05,                # DBSCAN参数
        min_samples=2           # DBSCAN参数
    ):
        import numpy as np
        # embeddings = np.array(get_text_embeddings_MiniLM(entities))
        embeddings = np.array(get_text_embeddings(entities))

        if method == "agglo":
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=distance_threshold
            )
        elif method == 'dbscan':  # dbscan
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine'
            )
        labels = clustering.fit_predict(embeddings)

        groups = {}
        for idx, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(entities[idx])
        
        return list(groups.values()) 
    
    def find_redundant_entity(self, retrieve_triple_3d):
        # print(f"retrieve_triple_3d: {json.dumps(retrieve_triple_3d, indent=2)}")
        seen = set()
        entity_list = []
        for triple_2d in retrieve_triple_3d:
            for triple in triple_2d:
                if triple[0] not in seen:
                    seen.add(triple[0])
                    entity_list.append(triple[0])
                if triple[2] not in seen:
                    seen.add(triple[2])
                    entity_list.append(triple[2])
            
        all_entity_group_2d = self.group_entities(entity_list, method="agglo")
        all_entity_group_str = ""
        entity_count = 0
        for i, group in enumerate(all_entity_group_2d):
            all_entity_group_str += f"Group {i}:\n"
            for entity in group:
                all_entity_group_str = all_entity_group_str + str(entity_count) + ': ' + entity + '\n'
                entity_count += 1
        fliter_entity_2d = [
            mid_list for mid_list in all_entity_group_2d  
            if len(mid_list) > 1 
        ]
        # group_triples(triples, method="agglo", embed_method="average")
        return fliter_entity_2d, all_entity_group_str
    
    # def process_redundant_relationship(self, response_for_redundant_relationship, redundant_relationship):
    #     log, delete_list, *extra_values = self.graph_database.process_redundant_relationship(response_for_redundant_relationship, redundant_relationship)
    #     # print("Keep List Log:", log)
    #     # print("Delete List:", delete_list)
    #     # print("Extra Values:", extra_values)
    #     return log, delete_list
    
    def process_redundant_relationship_v2(self, parsed_response_for_relationship, redundant_relationship_3d, TF):
        if not parsed_response_for_relationship:
            return "", []
        keep_list, delete_list = self.graph_database.process_redundant_relationship(parsed_response_for_relationship, redundant_relationship_3d, TF)
        return keep_list, delete_list



    def process_redundant_entity(self, parsed_response_for_entity, redundant_entity_2d, TF):
        if not parsed_response_for_entity:
            return None
        keep_and_delete_str, delete_entity, insert_relationship_list_2d, delete_ralationship_list_2d, insert_relationship_all, delete_relationship_all, keep_and_delete_entity_list_2d = self.graph_database.process_redundant_entity(parsed_response_for_entity, redundant_entity_2d, TF)
        
        return keep_and_delete_str, delete_entity, insert_relationship_list_2d, delete_ralationship_list_2d, insert_relationship_all, delete_relationship_all, keep_and_delete_entity_list_2d
    
    def entity_embedding_bak(self, iteration):
        self.graph_database.entity_embedding_bak(iteration)

    def delete_redundant_entity(self, delete_entity, flag):
        # self.graph_database.remove_entities(delete_entity)
        self.graph_database.remove_entity_embedding(delete_entity, flag)
    
    def generate_new_entity_embedding_standard(self, entities_new):
        self.graph_database.generate_new_entity_embedding_standard(entities_new)
        
    def postprocess(self, question, knowledge_sequence):
        if len(knowledge_sequence) == 0:
            return []
        
        kg_triplets = self.graph_database.kg_seqs_to_triplets(knowledge_sequence)
        kg_triplets = [' '.join(triplet) for triplet in kg_triplets]

        # print(f"kg_triplets :{len(kg_triplets)}\n\n\n")
        # print(f"kg_triplets :{kg_triplets}\n\n\n")

        embedding_idxs = [
            self.triplet2id[triplet] for triplet in kg_triplets
            if triplet in self.triplet2id
        ]
        # print(f"embedding_idxs :{len(embedding_idxs)}\n\n\n")

        # print(f"self.triplet2id :{len(self.triplet2id)}\n\n\n")
        # print(f"self.triplet_embeddings :{len(self.triplet_embeddings)}\n\n\n")
        
        embeddings = self.triplet_embeddings[embedding_idxs]
        # print(f"embeddings :{embeddings}\n\n\n")

        sorted_all_rel_scores = self.semantic_pruning_triplets(
            question,
            kg_triplets,
            rel_embeddings=embeddings,
            topk=self.pruning)

        pruning_knowledge_sequence = [rel for rel, _ in sorted_all_rel_scores]
        pruning_knowledge_dict = {"pruning": pruning_knowledge_sequence}

        return pruning_knowledge_sequence, pruning_knowledge_dict
    
    
    def extract_keywords_with_embedding_find_entity(self, question, max_keywords=5 ): 
        keyword_start_time = time.perf_counter()   
        # question_embed = np.array(get_text_embedding(question)).reshape(1, -1)

        entities = self.graph_database.entities   # 一个集合 set
        entities_list = [str(entity) for entity in sorted(entities)]
        entity_embeddings = self.graph_database.entity_embeddings
        entity_embeddings = np.array(entity_embeddings)

        if isinstance(question, list):
            keyword_start_time = time.perf_counter()
            entity_res =  []
            for q_item in question:
                sorted_all_entity_scores = self.semantic_pruning_entities_batch(q_item, entities_list, rel_embeddings=entity_embeddings, topk=max_keywords, batch_size=128)
                entity_res.append([rel for rel, _ in sorted_all_entity_scores])
            keyword_end_time = time.perf_counter()
            return entity_res, keyword_end_time - keyword_start_time
        else:
            # similarity_cp = cosine_similarity_cp_batch(question_embed, entity_embeddings ,100)[0]

            # similarity = similarity_cp

            # all_entities_scores = [(entity, score)
            #                 for entity, score in zip(entities_list, similarity.tolist())]
            # sorted_all_entities_scores = sorted(all_entities_scores,
            #                             key=lambda x: x[1],
            #                             reverse=True)
            # return [entity for entity, _ in sorted_all_entities_scores[:max_keywords]]
            
            #### two method
            # sorted_all_entity_scores = self.semantic_pruning_triplets_batch(question, entities_list, rel_embeddings=entity_embeddings, topk=max_keywords, batch_size=8)
            sorted_all_entity_scores = self.semantic_pruning_entities_batch(question, entities_list, rel_embeddings=entity_embeddings, topk=max_keywords, batch_size=128)
            keyword_end_time = time.perf_counter()
            return [rel for rel, _ in sorted_all_entity_scores], keyword_end_time - keyword_start_time

    def semantic_pruning_triplets(self, question,
                              triplets,
                              rel_embeddings=None,
                              topk=30):
        question_embed = np.array(get_text_embedding(question)).reshape(1, -1)

        if rel_embeddings is None:
            rel_embeddings = get_text_embeddings(triplets)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        similarity_cp = cosine_similarity_cp(question_embed, rel_embeddings)[0]

        similarity = similarity_cp

        all_rel_scores = [(rel, score)
                        for rel, score in zip(triplets, similarity.tolist())]
        sorted_all_rel_scores = sorted(all_rel_scores,
                                    key=lambda x: x[1],
                                    reverse=True)

        return sorted_all_rel_scores[:topk]

    def semantic_pruning_triplets_batch(self, question, triplets, rel_embeddings=None, topk=30, batch_size=8):
        # get_text_embeddings 已经批处理 这里只是余弦相似度计算批处理，作用不大
        question_embed = np.array(get_text_embedding_small(question)).reshape(1, -1)

        if rel_embeddings is None:
            rel_embeddings = get_text_embeddings_small(triplets)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        num_triplets = len(rel_embeddings)
        all_rel_scores = []

        for i in range(0, num_triplets, batch_size):
            batch_rel_embeddings = rel_embeddings[i:i+batch_size]
            similarity_cp = cosine_similarity_cp(question_embed, batch_rel_embeddings)
            similarity = similarity_cp[0]

            all_rel_scores.extend([(rel, score) for rel, score in zip(triplets[i:i+batch_size], similarity.tolist())])

        # 排序并返回前topk个
        sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
        return sorted_all_rel_scores[:topk]
    
    def semantic_pruning_entities_batch(self, question, triplets, rel_embeddings=None, topk=30, batch_size=8):
        # get_text_embeddings 已经批处理 这里只是余弦相似度计算批处理，作用不大
        # question_embed = np.array(get_text_embedding_MiniLM(question)).reshape(1, -1)
        question_embed = np.array(get_text_embedding(question, args=self.embedding)).reshape(1, -1)

        if rel_embeddings is None:
            # rel_embeddings = get_text_embeddings_MiniLM(triplets)
            rel_embeddings = get_text_embeddings(triplets, args=self.embedding)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        num_triplets = len(rel_embeddings)
        all_rel_scores = []

        for i in range(0, num_triplets, batch_size):
            batch_rel_embeddings = rel_embeddings[i:i+batch_size]
            similarity_cp = cosine_similarity_cp(question_embed, batch_rel_embeddings)
            similarity = similarity_cp[0]

            all_rel_scores.extend([(rel, score) for rel, score in zip(triplets[i:i+batch_size], similarity.tolist())])

        # 排序并返回前topk个
        sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
        return sorted_all_rel_scores[:topk]
    
    # +++ 返回排序后的索引和分数，无过滤阈值
    def semantic_pruning_sentences_batch(self, question, sentences, rel_embeddings=None, topk=None, batch_size=8):
        # get_text_embeddings 已经批处理 这里只是余弦相似度计算批处理，作用不大
        question_embed = np.array(get_text_embedding(question, args=self.embedding)).reshape(1, -1)

        if rel_embeddings is None:
            rel_embeddings = get_text_embeddings(sentences, args=self.embedding)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        num_sentences = len(rel_embeddings)
        all_rel_scores = []

        for i in range(0, num_sentences, batch_size):
            batch_rel_embeddings = rel_embeddings[i:i+batch_size]
            similarity_cp = cosine_similarity_cp(question_embed, batch_rel_embeddings)
            similarity = similarity_cp[0]

            # all_rel_scores.extend([(rel, score) for rel, score in zip(sentences[i:i+batch_size], similarity.tolist())])
            all_rel_scores.extend( #返回索引+分数而不是直接返回句子+分数
                [(i + idx, score) for idx, score in enumerate(similarity.tolist())]
            )

        sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
        return sorted_all_rel_scores[:topk]
    
    # +++ 直接返回嵌入，无排序
    def semantic_pruning_sentences_batch_v3(self, question, sentences, rel_embeddings=None, batch_size=8):
        # get_text_embeddings 已经批处理 这里只是余弦相似度计算批处理，作用不大
        question_embed = np.array(get_text_embedding(question, args=self.embedding)).reshape(1, -1)

        if rel_embeddings is None:
            rel_embeddings = get_text_embeddings(sentences, args=self.embedding)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        num_sentences = len(rel_embeddings)
        all_rel_scores = []

        for i in range(0, num_sentences, batch_size):
            batch_rel_embeddings = rel_embeddings[i:i+batch_size]
            similarity_cp = cosine_similarity_cp(question_embed, batch_rel_embeddings)
            similarity = similarity_cp[0]

            # all_rel_scores.extend([(rel, score) for rel, score in zip(sentences[i:i+batch_size], similarity.tolist())])
            all_rel_scores.extend( #分数
                similarity.tolist()
            )

        # sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
        # return sorted_all_rel_scores[:topk]
        return all_rel_scores  # 返回分数列表而不是索引+分数
    
    # +++ 返回排序后的索引和分数，加入了一个过滤阈值
    def semantic_pruning_sentences_batch_v2(self, question, sentences, rel_embeddings=None, batch_size=8, threshold=0.65):
        # get_text_embeddings 已经批处理 这里只是余弦相似度计算批处理，作用不大
        # question_embed = np.array(get_text_embedding_MiniLM(question)).reshape(1, -1)
        question_embed = np.array(get_text_embedding(question, args=self.embedding)).reshape(1, -1)

        if rel_embeddings is None:
            # rel_embeddings = get_text_embeddings_MiniLM(sentences)
            rel_embeddings = get_text_embeddings(sentences, args=self.embedding)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        num_sentences = len(rel_embeddings)

        filtered_results = []
        all_scores = []

        for i in range(0, num_sentences, batch_size):
            batch_rel_embeddings = rel_embeddings[i:i+batch_size]
            similarity_cp = cosine_similarity_cp(question_embed, batch_rel_embeddings)
            similarity = similarity_cp[0]
            
            for batch_idx, score in enumerate(similarity.tolist()):
                all_scores.append(score)
                if score > threshold:
                    global_idx = i + batch_idx  # 计算在原始sentences中的索引
                    filtered_results.append((global_idx, float(score)))

        # 按分数从高到低排序（可选）
        # filtered_results.sort(key=lambda x: x[1], reverse=True)

        if all_scores:
            max_score = max(all_scores)
            min_score = min(all_scores)
            median_score = statistics.median(all_scores)
            mean_score = sum(all_scores) / len(all_scores)
        else:
            max_score = min_score = median_score = mean_score = None
        
        return filtered_results, max_score, min_score, median_score, mean_score
    

    def kg_seqs_to_triplets_for_ragcache(self, kg_seqs):
        return self.graph_database.kg_seqs_to_triplets_for_ragcache(kg_seqs)
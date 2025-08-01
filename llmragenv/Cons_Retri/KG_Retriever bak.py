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

def get_text_embeddings(texts, embed_model_name="BAAI/bge-large-en-v1.5", step=400):
    global embed_model
    # embed_model = None
    if not embed_model:
        embed_model = EmbeddingEnv(embed_name=embed_model_name,
                                   embed_batch_size=2)

    all_embeddings = []
    n_text = len(texts)
    for start in range(0, n_text, step):
        input_texts = texts[start:min(start + step, n_text)]
        embeddings = embed_model.get_embeddings(input_texts)

        all_embeddings += embeddings
        
    return all_embeddings

def get_text_embedding(text, embed_model_name="BAAI/bge-large-en-v1.5"):
    global embed_model
    # embed_model = None
    if not embed_model:
        embed_model = EmbeddingEnv(embed_name=embed_model_name,
                                   embed_batch_size=2)
    embedding = embed_model.get_embedding(text)
    return embedding

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
    def __init__(self,llm:LLMBase, graphdb : GraphDatabase):
        self.graph_database = graphdb
        self._llm = llm
        
        # self.triplet2id = self.graph_database.triplet2id
        # self.triplet_embeddings = self.graph_database.triplet_embeddings
        self.entity = self.graph_database.entity2id
        self.entity_embeddings = self.graph_database.entity_embeddings

    def extract_keyword(self, question, max_keywords=5):
        prompt = keyword_extract_prompt.format(question=question, max_keywords=max_keywords)
        
        # 获取 LLM 的 response
        # if self._llm.__class__.__name__ == "OllamaClient":
        #     response = self._llm.chat_with_ai(prompt, info = "keyword")
        # else:
        #     response = self._llm.chat_with_ai(prompt)
        response = self._llm.chat_with_ai(prompt)
        
        # 处理 response，去掉 "KEYWORDS:" 前缀
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

        # print(f"pruning_knowledge_sequence : {pruning_knowledge_sequence}")
                
        return pruning_knowledge_sequence
    
    def retrieve_path_with_keywords(self, question, keywords = [], pruning = None, build_node = False):
        self.pruning = pruning

        query_results = {}

        all_rel_map = {}
        if pruning:
            # rel_map = self.graph_database.get_rel_map(entities=keywords, limit=1000000)
            for entity in keywords:
                rel_map = self.graph_database.get_rel_map(entities=[entity], depth=2, limit=1000000)
                all_rel_map.update(rel_map)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        # print(f"key_words : {keywords}")
        # print(f"rel_map : {rel_map}\n")

        clean_rel_map = self.graph_database.clean_rel_map(all_rel_map)
        # clean_rel_map = self.graph_database.clean_rel_map(rel_map)

        # print(f"clean_rel_map : {clean_rel_map}\n")

        query_results.update(clean_rel_map)

        # print(f"query results : {query_results}\n")

        knowledge_sequences = []

        for k, v in clean_rel_map.items():
            # print(k, type(v))
            kg_seqs = self.graph_database.get_knowledge_sequence({k: v})
            knowledge_sequences.append(kg_seqs)
        # print([x for x in knowledge_sequences])

        print([len(x) for x in knowledge_sequences])

        knowledge_sequences_list = [item for sublist in knowledge_sequences for item in sublist]
        #print(f"knowledge_sequences_list{np.array(knowledge_sequences_list).shape}")
        #print([x for x in knowledge_sequences_list])
        if pruning > 0:
            knowledge_sequences_pruning = []
            sorted_all_rel_scores = self.semantic_pruning_triplets_batch(question, knowledge_sequences_list, rel_embeddings=None, topk=pruning, batch_size=1)
            knowledge_sequences_pruning = [rel for rel, _ in sorted_all_rel_scores]
            knowledge_sequences = knowledge_sequences_pruning
        # print(len(knowledge_sequences))
        # print(knowledge_sequences)
        return knowledge_sequences

        # if pruning > 0:
        #     knowledge_sequences_result = []
        #     for listp in knowledge_sequences:
            
        #         sorted_all_rel_scores = self.semantic_pruning_triplets_batch(question, listp, rel_embeddings=None, topk=pruning, batch_size=8)
        #         knowledge_sequences_pruning = [rel for rel, _ in sorted_all_rel_scores]
        #         knowledge_sequences_result.append(knowledge_sequences_pruning)
        # return knowledge_sequences_result

        
    def get_nodes(self):
        return self.nodes

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
        question_embed = np.array(get_text_embedding(question)).reshape(1, -1)

        entities = self.graph_database.entities   # 一个集合 set
        entities_list = [str(entity) for entity in sorted(entities)]
        entity_embeddings = self.graph_database.entity_embeddings
        entity_embeddings = np.array(entity_embeddings)

        # similarity_cp = cosine_similarity_cp_batch(question_embed, entity_embeddings ,100)[0]

        # similarity = similarity_cp

        # all_entities_scores = [(entity, score)
        #                 for entity, score in zip(entities_list, similarity.tolist())]
        # sorted_all_entities_scores = sorted(all_entities_scores,
        #                             key=lambda x: x[1],
        #                             reverse=True)
        # return [entity for entity, _ in sorted_all_entities_scores[:max_keywords]]
        
        #### two method
        sorted_all_entity_scores = self.semantic_pruning_triplets_batch(question, entities_list, rel_embeddings=entity_embeddings, topk=max_keywords, batch_size=8)
        return [rel for rel, _ in sorted_all_entity_scores]

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


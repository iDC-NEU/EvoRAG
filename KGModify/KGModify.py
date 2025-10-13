'''
Author: fzb0316 fzb0316@163.com
Date: 2024-10-18 16:40:43
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-18 17:18:25
FilePath: /BigModel/RAGWebUi_demo/KGModify/KGModify.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


from chat.chat_graphrag import ChatGraphRAG
from dataset.dataset import Dataset
from dataset.generate_similar_questions import Generate_dataset
from logger import Logger
from llmragenv.Cons_Retri.KG_Construction import KGConstruction
from database.graph.nebulagraph.nebulagraph import NebulaDB
from llmragenv.LLM.llm_base import LLMBase

from evaluate import load
import statistics
import json
import datetime
import numpy as np
from ignite.metrics import RougeL
import time





class KGModify:
    # def __init__(self, graphrag : ChatGraphRAG, graphrag2 : ChatGraphRAG, args):
    def __init__(self, graphrag : ChatGraphRAG, graphrag_fb : ChatGraphRAG, args):
        self.graphrag = graphrag
        self.args = args
        self.retriver = graphrag.retriver_graph
        self.llm = graphrag._llm
        self.graph_database = graphrag.graph_database
        
        self.dataset = Dataset(self.args.dataset_name)
        # self.Generate_dataset = Generate_dataset(args)
        # self.dataset_gene = self.Generate_dataset.dataset
        self.dataset_gene = self.dataset

        self.keywords = []
        self.response = []
        self.retrieve_result = []
        self.correction_relation = []
        self.correction_entity = []
        self.correction_error = []
        self.correction_generation = []
        # self.graphrag2 = graphrag2
        self.graphrag_fb = graphrag_fb
        self.logger = None
        self.triplets_score = {}

        
    def run(self):
        if self.args.option == "without_rag":
            self.without_rag()
        elif self.args.option == "graph_rag":
            self.graph_rag()
        elif self.args.option == "graph_rag_keyword_with_similarity":
            self.graph_rag_keyword_with_similarity()
        elif self.args.option == "graph_rag_correction":
            self.graph_rag_correction()
        elif self.args.option == "graph_rag_baseline":
            self.graph_rag_baseline()
        elif self.args.option == "graph_rag_baseline_data":
            self.graph_rag_baseline_data()
        elif self.args.option == "graph_rag_baseline_analysis":
            self.graph_rag_baseline_analysis()
        elif self.args.option == "graph_rag_baseline_redundancy":
            self.graph_rag_baseline_redundancy()
        elif self.args.option == "graph_rag_baseline_redundancy_stage1":
            self.graph_rag_baseline_redundancy_stage1()
        elif self.args.option == "graph_rag_baseline_redundancy_stage2":
            self.graph_rag_baseline_redundancy_stage2()
        elif self.args.option == "baseline_redundancy_stage1_for_ragcache":
            self.baseline_redundancy_stage1_for_ragcache()
        elif self.args.option == "baseline_redundancy_stage2_for_ragcache":
            self.baseline_redundancy_stage2_for_ragcache()
        elif self.args.option == 'baseline_redundancy_for_myoriginal_api':
            self.baseline_redundancy_for_myoriginal_api()
        elif self.args.option == 'graph_rag_sys_redundancy':
            self.graph_rag_sys_redundancy()
        elif self.args.option == "baseline_redundancy_stage2_for_ragcache_api":
            self.baseline_redundancy_stage2_for_ragcache_api()
        elif self.args.option == "baseline_redundancy_stage2_for_ragcache_local":
            self.baseline_redundancy_stage2_for_ragcache_local()
        elif self.args.option == "baseline_redundancy_stage2_for_lightRAG":
            self.baseline_redundancy_stage2_for_lightRAG()
        elif self.args.option == "graph_rag_baseline_reproduction":
            self.graph_rag_baseline_reproduction()
        elif self.args.option == "graph_rag_score":
            self.graph_rag_score(0)
        elif self.args.option == "kg_construction":
            self.dataset.get_corpus("positive")
            KGConstruction(self.llm, self.graph_database, self.args.space_name).run([])
        elif self.args.option == "kg_modify":
            self.kg_modify()
        elif self.args.option == "kg_modify_forward":
            self.kg_modify_forward(int(self.args.iteration))
        elif self.args.option == "kg_modify_feedback":
            self.kg_modify_feedback(int(self.args.iteration))
        elif self.args.option == "kg_modify_api": # 可能还有问题
            self.kg_modify_api(int(self.args.iteration))
        elif self.args.option == "kg_modify_api_qwen":
            self.kg_modify_api_qwen(int(self.args.iteration))
        elif self.args.option == "kg_modify_api_qwen_reproduce":
            self.kg_modify_api_qwen_reproduce(int(self.args.iteration))
        elif self.args.option == "kg_modify_llama_reproduce_forword_data":
            self.kg_modify_llama_reproduce_forword_data(int(self.args.iteration))
        elif self.args.option == "kg_modify_llama_reproduce_forword_case":
            self.kg_modify_llama_reproduce_forword_case(int(self.args.iteration))
        elif self.args.option == "kg_modify_llama_reproduce_forword_verify":
            self.kg_modify_llama_reproduce_forword_verify(int(self.args.iteration))
        elif self.args.option == "evolve_basic_forward":
            self.evolve_basic_forward()
        elif self.args.option == "evolve_basic_feedback":
            self.evolve_basic_feedback()
        elif self.args.option == "evolve_batch":
            self.evolve_batch()
        elif self.args.option == "kg_modify_llama_reproduce_feedback":
            self.kg_modify_llama_reproduce_feedback(int(self.args.iteration))
        elif self.args.option == "kg_modify_llama_reproduce":
            self.kg_modify_llama_reproduce(int(self.args.iteration))
        elif self.args.option == "feedback_iteration":
            self.feedback_iteration(int(self.args.iteration))
        elif self.args.option == "kg_modify_ignore":
            self.kg_modify_ignore()
        elif self.args.option == "test":
            self.test()
        elif self.args.option == "test1":
            self.test1()
        elif self.args.option == "mindmap_test":
            self.mindmap_test()
        elif self.args.option == "transE":
            self.transE()
        elif self.args.option == "random_numbers":
            self.random_numbers()
        elif self.args.option == "demo":
            self.demo()
        elif self.args.option == 'iteration_redundancy_statistics':
            self.iteration_redundancy_statistics()
        elif self.args.option == 'iteration_score_statistics':
            self.iteration_score_statistics()
        elif self.args.option == 'iteration_error_statistics':
            self.iteration_error_statistics()
        elif self.args.option == 'short_cut_test':
            self.short_cut_test()
        else:
            print("The option is not supports!")
            assert(False)

    def without_rag(self):
        for i in range(len(self.dataset.query)):
            print(f"without rag id : {i+1}")
            self.response.append(self.graphrag.chat_without_rag(self.dataset.query[i]))
        
        self.get_accuracy(log="full")
        # self.get_accuracy(log="only false")

    def graph_rag(self):

        for i in range(len(self.dataset.query)):
        # for i in range(1):
            if i>=3:
                break
            # assert(False)
            print(f"graph rag id : {i+1}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)

            keywords = self.retriver.extract_keyword(self.dataset.query[i])
            retrieve_result = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[i], keywords, pruning=30)
            # retrieve_result = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[i], keywords)
            res = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[i], retrieve_result)
            
            self.keywords.append(keywords)
            self.response.append(res)
            self.retrieve_result.append(self.graphrag.get_triplets())
            # assert(False)

        self.get_accuracy(log="full_tmp")
        # self.get_accuracy(log="only false")
    
    def graph_rag_keyword_with_similarity(self):

        for i in range(len(self.dataset.query)):
        # for i in range(1):
            print(f"graph rag id : {i+1}")
            # if i<4:
            #     continue
            # if i != 94:
            #     self.keywords.append([])
            #     self.response.append("")
            #     self.retrieve_result.append([])
            #     continue
            if i>= 5:
                break
                # assert(False)
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)

            # keywords = self.retriver.extract_keyword(self.dataset.query[i]) # 大模型提取
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i], 12) # 问题相似度选

            #retrieve_result = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[i], keywords, pruning=30)
            retrieve_result = self.retriver.retrieve_path_with_keywords(self.dataset.query[i], keywords, path_depth = 2, pruning = 50 ) 
            # print(len(retrieve_result))
            # print(retrieve_result)
            # retrieve_result = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[i], keywords)
            res = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[i], retrieve_result)

            # print(f"self.dataset.query[i]: {self.dataset.query[i]}")
            # print(f"keywords: {keywords}")
            # print(f"retrieve_result: {retrieve_result}")
            # print(f"res: {res}")
            
            self.keywords.append(keywords)
            self.response.append(res)
            # self.retrieve_result.append(retrieve_result)
            self.retrieve_result.append(self.graphrag.get_triplets())
            # assert(False)

        self.get_accuracy(log="full_similarity")
        # self.get_accuracy(log="only false")

    def graph_rag_correction(self):

        for i in range(len(self.dataset.query)):
        # for i in range(1):
            print(f"graph rag id : {i+1}")
            # if i<4:
            #     continue
            # if i>4:
            #     break
            if i >= 5:
                break
                # assert(False)
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)

            # keywords = self.retriver.extract_keyword(self.dataset.query[i]) # 大模型提取
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i],10) # 问题相似度选

            #retrieve_result = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[i], keywords, pruning=30)
            retrieve_result = self.retriver.retrieve_path_with_keywords(self.dataset.query[i], keywords, pruning=60)
            # print(len(retrieve_result))
            # print(retrieve_result)
            # retrieve_result = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[i], keywords)
            res = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[i], retrieve_result)

            retrieve_one_hop = self.retriver.path_to_triples(retrieve_result)
            res_correction_relation = self.graphrag.chat_without_stream_with_all_context(self.dataset.query[i], self.dataset.answer[i], retrieve_one_hop, 1)
            res_correction_entity = self.graphrag.chat_without_stream_with_all_context(self.dataset.query[i], self.dataset.answer[i], res_correction_relation, 2)
            res_correction_error = self.graphrag.chat_without_stream_with_all_context(self.dataset.query[i], self.dataset.answer[i], retrieve_one_hop, 3)
            res_correction_generation = self.graphrag.chat_without_stream_with_all_context(self.dataset.query[i], self.dataset.answer[i], retrieve_one_hop, 4)

            self.keywords.append(keywords)
            self.response.append(res)
            # self.retrieve_result.append(retrieve_result)
            self.retrieve_result.append(self.graphrag.get_triplets())
            # assert(False)
            self.correction_relation.append(res_correction_relation)
            self.correction_entity.append(res_correction_entity)
            self.correction_error.append(res_correction_error)
            self.correction_generation.append(res_correction_generation)

        self.get_accuracy(log="full_correction")
        # self.get_accuracy(log="only false")

    def graph_rag_score_bak(self, iteration):
        print(f"graph_rag_score iteration: {iteration}")
        import json
        #***
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration}.json", "r", encoding="utf-8") as file:
            triplets_score_modify = json.load(file)

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_evaluate_{self.args.type}_latest_{iteration}_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        query_number = 0
        
        for i in range(len(self.dataset.query)):
            # if i >= 1:
            #     break
        # for i in range(1):
            print(f"kg modify id : {i+1}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {self.dataset.query[i]}")
            logger.log(f"answer: {self.dataset.answer[i]}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)

            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i],4) # 8 100 60
            retrieve_result = self.retriver.retrieve_path_with_keywords(self.dataset.query[i], keywords, pruning=60)
            
            # self.retriver.triple_into_sentence(retrieve_one_hop[i])
            # 删除有毒部以及重排
            filtered_retrieve_result = self.retriver.filter_paths_by_score(self.dataset.query[i], retrieve_result, triplets_score_modify)
            
            res = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[i], filtered_retrieve_result[:30])
            
            flag = False
            if isinstance(self.dataset.answer[i], list):
                instance = [j.lower() for j in self.dataset.answer[i]]
                for j in instance:
                    if j in res.lower():
                        flag = True
                        break
            else:
                instance = self.dataset.answer[i].lower()
                if instance in res.lower():
                    flag = True
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")

            logger.log(f"keywords : {keywords}")
            logger.log(f"retrieve_result : {retrieve_result}")
            logger.log(f"filtered_result : {filtered_retrieve_result}")
            logger.log(f"graph rag response: {res}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")

    def graph_rag_baseline_reproduction(self): # 使用llama3:8b指令模型
        # return True

        import json
        from transformers import AutoTokenizer, AutoModelForCausalLM
        #***
        with open(f"./logs/triplets/{self.args.space_name}.json", "r", encoding="utf-8") as file:
            triplets_score_modify = json.load(file)

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        
        file_path = './logs/yuanh/rgb_ent10_pruning30_llama3.json'

        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象
            
        # for idx, item in enumerate(data_list):
        #     if idx <3:
        #         print(json.dumps(item["context"], indent=2))
        #         print(json.dumps(item["context_str"], indent=2))
        # return 1

        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        
        for i in range(len(data_list)):
            if i >= 1:
                break
        # for i in range(1):
            print(f"kg modify id : {i+1}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['question']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            
            retrieve_result_list = [ item for sublist in data_list[i]['context'] for item in sublist]
            retrieve_result = data_list[i]['context_str']

            logger.log(f"retrieve result : {json.dumps(retrieve_result_list, indent=2)}")
            
            res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(data_list[i]['question'], retrieve_result)

            self.response.append(res)
            if self.graphrag2:        
                answer_check = self.graphrag2.chat_without_stream_answer_check(data_list[i]['question'], res, data_list[i]['answer'])
                logger.log(f"-------------graph rag response check: {answer_check}")
                if "Insufficient information error" in answer_check:
                    acc_refuse += 1
                    logger.log(f"-------------graph rag response Insufficient information error----------------------------------")
                elif "True" in answer_check:
                    acc_llm += 1
                    logger.log(f"-------------graph rag response true (llM)----------------------------------")
                elif "Error" in answer_check:
                    logger.log(f"-------------graph rag response false (llM)----------------------------------")
                else:
                    logger.log(f"-------------graph rag response unknown (llM)----------------------------------")

            flag = False
            if isinstance(data_list[i]['answer'], list):
                instance = [j.lower() for j in data_list[i]['answer']]
                for j in instance:
                    if j in res.lower():
                        flag = True
                        break
            else:
                instance = data_list[i]['answer'].lower()
                if instance in res.lower():
                    flag = True
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")

            # logger.log(f"keywords : {keywords}")
            # logger.log(f"retrieve_result : {retrieve_result}")
            # logger.log(f"filtered_result : {filtered_retrieve_result}")
            logger.log(f"graph rag response: {res}")


        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = data_list[i]['answer'] # 对于rgb而言一定是list
            reference_answer_item['answer_start'] = []
            for _ in range(len(reference_answer_item['text'])):
                reference_answer_item['answer_start'].append(0)
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
            # assert(False)

    def graph_rag_baseline_redundancy_stage1(self,): # 使用llama3:8b指令模型
        import json
        
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        redundant = []
        data = []
        
        for i in range(len(self.dataset.query)):
            # if i >= 5:
            #     break
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {self.dataset.query[i]}")
            logger.log(f"answer: {self.dataset.answer[i]}")
            
            tmp_dict = {}
            tmp_dict['question'] = self.dataset.query[i]
            tmp_dict['answer'] = self.dataset.answer[i]
            # if stage == 0:
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i], 10) # 问题相似度选
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[i], keywords, 2)
            logger.log(f"keywords : {keywords}")
            tmp_dict['keyword'] = keywords
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")  
            
            retrieve_result, retrieve_triple_3d, sentences = self.retriver.retrieve_path_with_keywords_v1(self.dataset.query[i], keywords, path_depth = 2, pruning=30)
            logger.log(f"retrieve result : {json.dumps(retrieve_result, indent=2)}")
            tmp_dict['retrieve_result'] = retrieve_result
            tmp_dict['retrieve_triple_3d'] = retrieve_triple_3d
            res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[i], retrieve_result)
            self.response.append(res)
            tmp_dict['response'] = res

            flag = False
            if isinstance(self.dataset.answer[i], list):
                instance = [j.lower() for j in self.dataset.answer[i]]
                # print(f"instance {instance}")
                for j in instance:
                    if j in res.lower():
                        flag = True
                        # print(f"True")
                        break
            else:
                # print("222")
                instance = self.dataset.answer[i].lower()
                if instance in res.lower():
                    flag = True
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")
            tmp_dict['label'] = flag
            data.append(tmp_dict)


        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.dataset.answer[i] # 对于rgb而言一定是list
            reference_answer_item['answer_start'] = []
            for _ in range(len(reference_answer_item['text'])):
                reference_answer_item['answer_start'].append(0)
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        # if self.llm.model_name == '':
        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_{self.args.option}_{self.args.dataset_name}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))   
   
    def graph_rag_baseline_redundancy_stage2(self,): # 使用qwen2.5:32b指令模型
        # return True

        import json
        #***
        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_graph_rag_baseline_redundancy_stage1_{self.args.dataset_name}.json", 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        redundant = []
        
        for i in range(len(data_list)):
            if i >= 5:
                break
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['question']}")
            logger.log(f"answer: {data_list[i]['answer']}")

            keywords = data_list[i]['keyword'] # 问题相似度选
            logger.log(f"keywords : {keywords}")
            
            retrieve_result = data_list[i]['retrieve_result']
            retrieve_triple_3d = data_list[i]['retrieve_triple_3d']
            logger.log(f"retrieve result : {json.dumps(retrieve_result, indent=2)}")
            res = data_list[i]['response']

            redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d)
            logger.log(f"All relationship groups:\n{all_relationship_group_str}") #输出关系所有分组
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            self.response.append(res)
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"Fliter relationship groups:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  # delete_list 2d    
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                else:
                    logger.log(f"{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"{i} response for redundant relationships parse error")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                logger.log(f"{i} No redundant relationships")    
                print(f"{i} No redundant relationships") 

            # 反馈处理，这里其实可以利用前面的处理结果
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retrieve_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1
                    
            logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            num_list = self.graphrag.chat_without_stream_for_error_statistics(data_list[i]['question'], data_list[i]['answer'], one_hop_sentence_list)
            logger.log(f"--------------error relationship:{len(num_list)} {len(num_list)/len(one_hop_sentence_list)}")
            error_joint = []
            error_disjoint = []
            error_joint_sentence = []
            for index_tmp in num_list:
                row = one_hop_sentence_2d[index_tmp]
                error_joint_sentence.append(one_hop_sentence_list[index_tmp])
                if row not in delete_list:
                    error_disjoint.append(row)
                else:
                    error_joint.append(row)
            logger.log(f"--------------error relationship disjoint:{len(error_joint)+len(delete_list)/len(one_hop_sentence_list)}")
            logger.log(f"error joint sentence:\n{json.dumps(error_joint_sentence, indent=2)}")
            logger.log(f"--------------all_redundancy_triple_list:{len(all_redundancy_triple_list)}")
            logger.log(f"--------------one_hop_sentence_list:{len(one_hop_sentence_list)}")
            redundant.append(len(error_joint)+len(delete_list)/len(one_hop_sentence_list))

            
            answer_check = self.graphrag.chat_without_stream_answer_check(data_list[i]['question'], res, data_list[i]['answer'])
            logger.log(f"-------------graph rag response check: {answer_check}")
            if "Insufficient information error" in answer_check:
                acc_refuse += 1
                logger.log(f"-------------graph rag response Insufficient information error----------------------------------")
            elif "True" in answer_check:
                acc_llm += 1
                logger.log(f"-------------graph rag response true (llM)----------------------------------")
            elif "Error" in answer_check:
                logger.log(f"-------------graph rag response false (llM)----------------------------------")
            else:
                logger.log(f"-------------graph rag response unknown (llM)----------------------------------")

            flag = False
            if isinstance(self.dataset.answer[i], list):
                instance = [j.lower() for j in self.dataset.answer[i]]
                for j in instance:
                    if j in res.lower():
                        flag = True
                        break
            else:
                instance = self.dataset.answer[i].lower()
                if instance in res.lower():
                    flag = True
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = data_list[i]['answer'] # 对于rgb而言一定是list
            reference_answer_item['answer_start'] = []
            for _ in range(len(reference_answer_item['text'])):
                reference_answer_item['answer_start'].append(0)
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")


    def baseline_redundancy_stage1_for_ragcache(self,): # 可以用来测ragcache的准确率
        import json

        # file_path = f'./logs/ragcache/{self.args.dataset_name}_ent10_pruning30_llama3.json'
        if self.args.dataset_name == 'rgb':
            file_path = f'./logs/ragcache/rgb_en_refine_ent10_pruning30.json'
            # file_path = f'./logs/ragcache/rgb_en_refine_ent16_pruning10.json'
        elif self.args.dataset_name == 'multihop':
            file_path = f'./logs/ragcache/multihop_inference_query_refine_ent10_pruning30.json'
            # file_path = f'./logs/ragcache/multihop_inference_query_refine_ent16_pruning10.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象
        
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.entity}_{self.args.pruning}_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        redundant = []
        data = []
        tokens_cost = []
        
        for i in range(len(data_list)):
            # if i >= 2:
            #     break
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['question']}")
            

            retrieve_result_list = [ item.strip() for sublist in data_list[i]['context'][:self.args.entity] for item in sublist[:self.args.pruning]]
            # retrieve_result = data_list[i]['context_str']
            logger.log(f"retrieve result : {json.dumps(retrieve_result_list, indent=2)}")
            # res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(data_list[i]['question'], retrieve_result_list)
            res, num_input_tokens, prompt = self.graphrag.chat_without_stream_with_triplets_llama_instruct(data_list[i]['question'], retrieve_result_list)

            self.response.append(res)
            data_list[i]['generated'] = res
            data_list[i]['tokens'] = num_input_tokens
            tokens_cost.append(num_input_tokens)

            flag = False
            flag_label = self.checkanswer(res, data_list[i]['answer'])
            flag = sum(flag_label) == len(flag_label)
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"answer: {data_list[i]['answer']}")
            logger.log(f"num_input_tokens: {num_input_tokens}")
            logger.log(f"graph rag response: {res}")
            data_list[i]['label'] = flag_label
            data_list[i]['flag'] = flag

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(data_list[i]['answer'], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")
        logger.log(f"\n\n-----------------------tokens cost avg: {sum(tokens_cost)/len(tokens_cost)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        # if self.llm.model_name == '':
        print(f"len(data_list): {len(data_list)}")

        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_{self.args.option}_{self.args.dataset_name}_{self.args.entity}_{self.args.pruning}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data_list, indent=2))

    def baseline_redundancy_stage2_for_ragcache(self,): # 使用qwen2.5:32b指令模型
        # return True
        import json
        #***
        # file_path = './logs/ragcache/rgb_ent10_pruning30_llama3.json'
        file_path = f"./logs/stage/Meta-Llama-3-8B-Instruct_baseline_redundancy_stage1_for_ragcache_{self.args.dataset_name}.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象

        # acc=0
        # for i in range(len(data_list)):
        #     if sum(data_list[i]['label']) == len(data_list[i]['label']):
        #         acc += 1
        # print(f"正确率 acc: {acc}/300")
        # assert False

        # 又不得不涉及到清洗数据

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{current_time}")

        logger2 = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_for_analysis_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        redundant = []
        data = []

        for i in range(len(data_list)):
            # if i >= 1:
            #     break
            data_tmp = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['question']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            data_tmp['index'] = i
            data_tmp['question'] = data_list[i]['question']

            logger2.log(f"-------------index : {i}")
            logger2.log(f"query: {data_list[i]['question']}")
            logger2.log(f"answer: {data_list[i]['answer']}")

            # keywords = data_list[i]['keyword'] # 问题相似度选
            # logger.log(f"keywords : {keywords}")

            retrieve_result_list = [ item.strip() for sublist in data_list[i]['context'] for item in sublist]
            # retrieve_result = data_list[i]['context_str']
            logger.log(f"retrieve result : {json.dumps(retrieve_result_list, indent=2)}")
            
            res = data_list[i]['generated']
            self.response.append(res)
            
            # retrieve_result -> retrieve_triple_3d
            retrieve_triple_3d = self.retriver.kg_seqs_to_triplets_for_ragcache(retrieve_result_list)
            # logger.log(f"retrieve_triple_3d : {json.dumps(retrieve_triple_3d, indent=2)}")

            redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d) # 使用v2没毛病
            # redundant_relationship_3d = []
            # all_relationship_group_str = ''
            # all_redundancy_triple_list = []
            # unique_number_count = 0
            logger.log(f"All relationship groups:\n{all_relationship_group_str}") #输出关系所有分组
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"Fliter relationship groups:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  # delete_list 2d    
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                    redundant.append(len(delete_list)/len(all_redundancy_triple_list))
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    data_tmp['redundancy_rate'] = len(delete_list)/len(all_redundancy_triple_list)
                else:
                    redundant.append(0)
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    delete_list = []
                    data_tmp['redundancy_rate'] = 0
                    logger.log(f"{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"---------{i} response for redundant relationships parse error!!!!!!!!!!----------")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                # 如果真出现了这种情况程序会出现异常
                redundant.append(0)
                data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                delete_list = []
                data_tmp['redundancy_rate'] = 0
                logger.log(f"{i} No redundant relationships")    
                print(f"---------{i} No redundant relationships!!!!!!!-------------") 

            
            # redundant.append(0)
            # data.append({})

            # 反馈处理，这里其实可以利用前面的处理结果
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retrieve_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                        triple_tmp = [triple[0], triple[1], triple[2]]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                        triple_tmp = [triple[2], triple[1], triple[0]]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple_tmp)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1

            result_triple = [sublist for sublist in one_hop_sentence_2d if sublist not in delete_list]
            result_triple_list = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in result_triple ]
            # logger.log(f"delete_list(filter):\n{json.dumps(delete_list, indent=2)}") 
            # logger.log(f"one_hop_sentence_2d:\n{json.dumps(one_hop_sentence_2d, indent=2)}") 
            # logger.log(f"result_triple(filter):\n{json.dumps(result_triple, indent=2)}") 
                    
            # logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            logger2.log(f"one_hop_sentence(filter):\n{json.dumps(result_triple_list, indent=2)}\n\n") 
            # num_list = self.graphrag.chat_without_stream_for_error_statistics(data_list[i]['question'], data_list[i]['answer'], one_hop_sentence_list, batch_size = 20)
            # logger.log(f"--------------error relationship:{len(num_list)} {len(num_list)/len(one_hop_sentence_list)}")
            # error_joint = []
            # error_disjoint = []
            # error_sentence = []
            # for index_tmp in num_list:
            #     row = one_hop_sentence_2d[index_tmp]
            #     error_sentence.append(one_hop_sentence_list[index_tmp])
            #     if row not in delete_list:
            #         error_disjoint.append(row)
            #     else:
            #         error_joint.append(row)
    
            # logger.log(f"error sentence:\n{json.dumps(error_sentence, indent=2)}")
            # logger.log(f"error disjoint sentence:\n{json.dumps(error_disjoint, indent=2)}")
            # logger.log(f"--------------error relationship disjoint:{len(error_disjoint)/len(one_hop_sentence_list)}")
            logger.log(f"--------------all_redundancy_triple_list:{len(all_redundancy_triple_list)}")
            logger.log(f"--------------one_hop_sentence_list:{len(one_hop_sentence_list)}")
            # redundant.append(len(error_disjoint)+len(delete_list)/len(one_hop_sentence_list))
            
            answer_check = self.graphrag.chat_without_stream_answer_check(data_list[i]['question'], res, data_list[i]['answer'])
            logger.log(f"-------------graph rag response check: {answer_check}")
            if "Insufficient information error" in answer_check:
                acc_refuse += 1
                logger.log(f"-------------graph rag response Insufficient information error----------------------------------")
            elif "True" in answer_check:
                acc_llm += 1
                logger.log(f"-------------graph rag response true (llM)----------------------------------")
            elif "Error" in answer_check:
                logger.log(f"-------------graph rag response false (llM)----------------------------------")
            else:
                logger.log(f"-------------graph rag response unknown (llM)----------------------------------")  

            # label = self.checkanswer(res, data_list[i]['answer'])
            # flag = False
            # if isinstance(data_list[i]['answer'], list):
            #     instance = [j.lower() for j in data_list[i]['answer']]
            #     for j in instance:
            #         if j in res.lower():
            #             flag = True
            #             break
            # else:
            #     instance = data_list[i]['answer'].lower()
            #     if instance in res.lower():
            #         flag = True
            
            
            label = data_list[i]['label']
            flag = sum(label) == len(label)
            data_tmp['label'] = flag
            data.append(data_tmp)
            
            if int(flag) == 1:
                acc +=1

            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")
        
        # 保存检索总数量与冗余统计
        with open(f"./logs/stage/Qwen-Qwen2.5-32B-Instruct_{self.args.option}_{self.args.dataset_name}_redundancy_rate.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def baseline_redundancy_stage2_for_ragcache_local(self,): # 使用qwen2.5:32b指令模型
        # return True
        import json
        #***
        # file_path = './logs/ragcache/rgb_ent10_pruning30_llama3.json'
        file_path = f"./logs/stage/Meta-Llama-3-8B-Instruct_baseline_redundancy_stage1_for_ragcache_{self.args.dataset_name}.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象

        # file_path = f'./logs/ragcache/{self.args.dataset_name}_ent10_pruning30_llama3.json'
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     data_list = json.load(file)  # 直接使用json.load读取文件对象

        # acc=0
        # for i in range(len(data_list)):
        #     if sum(data_list[i]['label']) == len(data_list[i]['label']):
        #         acc += 1
        # print(f"正确率 acc: {acc}/300")
        # assert False

        # 又不得不涉及到清洗数据

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"./local/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{current_time}")
        logger2 = Logger(f"./local/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_for_analysis_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        query_number = 0
        redundant = []
        data = []

        for i in range(len(data_list)):
            if i >= 1:
                break
            data_tmp = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['question']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            data_tmp['index'] = i
            data_tmp['question'] = data_list[i]['question']

            logger2.log(f"-------------index : {i}")
            logger2.log(f"query: {data_list[i]['question']}")
            logger2.log(f"answer: {data_list[i]['answer']}")

            # keywords = data_list[i]['keyword'] # 问题相似度选
            # logger.log(f"keywords : {keywords}")

            retrieve_result_list = [ item.strip() for sublist in data_list[i]['context'] for item in sublist]
            # retrieve_result = data_list[i]['context_str']
            logger.log(f"retrieve result : {json.dumps(retrieve_result_list, indent=2)}")
            
            res = data_list[i]['generated']
            self.response.append(res)
            data_tmp['generated'] = res
            # retrieve_result -> retrieve_triple_3d
            retrieve_triple_3d = self.retriver.kg_seqs_to_triplets_for_ragcache(retrieve_result_list)
            # logger.log(f"retrieve_triple_3d : {json.dumps(retrieve_triple_3d, indent=2)}")
            data_tmp['retrieve_triple_3d'] = retrieve_triple_3d

            distance_threshold = 0.15
            if self.args.dataset_name == 'rgb':
                distance_threshold = 0.15
            elif self.args.dataset_name == 'multihop':
                distance_threshold = 0.25

            redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d, distance_threshold = distance_threshold) # 使用v2没毛病
            # redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d) # 使用v2没毛病
            # redundant_relationship_3d = []
            # all_relationship_group_str = ''
            # all_redundancy_triple_list = []
            # unique_number_count = 0
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            logger.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            logger.log(f"-------All relationship groups-------:\n{all_relationship_group_str}") #输出关系所有分组

            logger2.log(f"-------------unique number count ------- : {unique_number_count}")
            logger2.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            # logger.log(f"{json.dumps(all_redundancy_triple_list)}")
            
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"-------Fliter relationship groups--------:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                # parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                # parsed_response_for_relationship = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api(redundant_relationship_3d)  
                enable_thinking = False
                parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3_local(redundant_relationship_3d, enable_thinking)    
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  # delete_list 2d    
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                    logger2.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")

                    redundant.append(len(delete_list)/len(all_redundancy_triple_list))
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    data_tmp['delete_relationship_number'] = len(delete_list)
                    data_tmp['redundancy_rate'] = len(delete_list)/len(all_redundancy_triple_list)
                else:
                    redundant.append(0)
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    delete_list = []
                    data_tmp['delete_relationship_number'] = 0
                    data_tmp['redundancy_rate'] = 0
                    logger.log(f"index{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"---------index{i} response for redundant relationships parse error!!!!!!!!!!----------")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                # 如果真出现了这种情况程序会出现异常
                redundant.append(0)
                data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                delete_list = []
                data_tmp['delete_relationship_number'] = 0
                data_tmp['redundancy_rate'] = 0
                logger.log(f"{i} No redundant relationships")    
                print(f"---------{i} No redundant relationships!!!!!!!-------------") 


            # 检索结果减去被认为是冗余的内容
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retrieve_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                        triple_tmp = [triple[0], triple[1], triple[2]]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                        triple_tmp = [triple[2], triple[1], triple[0]]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple_tmp)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1

            result_triple = [sublist for sublist in one_hop_sentence_2d if sublist not in delete_list]
            result_triple_list = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in result_triple ]
            # logger.log(f"delete_list(filter):\n{json.dumps(delete_list, indent=2)}") 
            # logger.log(f"one_hop_sentence_2d:\n{json.dumps(one_hop_sentence_2d, indent=2)}") 
            # logger.log(f"result_triple(filter):\n{json.dumps(result_triple, indent=2)}") 
                    
            # logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            logger2.log(f"one_hop_sentence(filter):\n{json.dumps(result_triple_list, indent=2)}\n\n") 
            delete_list_str = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in delete_list ]
            logger2.log(f"one_hop_sentence(delete):\n{json.dumps(delete_list_str, indent=2)}\n\n") 

            # 前一个阶段有答案就可以直接利用
            label = data_list[i]['label']
            flag = sum(label) == len(label)
            data_tmp['label'] = flag
            data.append(data_tmp)
            
            if int(flag) == 1:
                acc +=1

            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")
            logger2.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger2.log(f"graph rag response: {res}")

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")
        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")
        
        # 保存检索总数量与冗余统计
        with open(f"./logs/local/redundancy/Qwen3-32B-no-think_{self.args.option}_{self.args.dataset_name}_redundancy_rate.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def baseline_redundancy_stage2_for_ragcache_api(self,): # 使用qwen2.5:32b指令模型
        # return True
        import json
        #***
        # file_path = './logs/ragcache/rgb_ent10_pruning30_llama3.json'
        # file_path = f"./logs/stage/Meta-Llama-3-8B-Instruct_baseline_redundancy_stage1_for_ragcache_{self.args.dataset_name}.json"
        file_path = f"./logs/stage/Meta-Llama-3-8B-Instruct_baseline_redundancy_stage1_for_ragcache_{self.args.dataset_name}_10_30.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象

        # file_path = f'./logs/ragcache/{self.args.dataset_name}_ent10_pruning30_llama3.json'
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     data_list = json.load(file)  # 直接使用json.load读取文件对象

        # acc=0
        # for i in range(len(data_list)):
        #     if sum(data_list[i]['label']) == len(data_list[i]['label']):
        #         acc += 1
        # print(f"正确率 acc: {acc}/300")
        # assert False

        # 又不得不涉及到清洗数据

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"./api_qwen/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{current_time}")
        logger2 = Logger(f"./api_qwen/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_for_analysis_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        query_number = 0
        redundant = []
        data = []

        for i in range(len(data_list)):
            # if i >= 2:
            #     break
            data_tmp = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['question']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            data_tmp['index'] = i
            data_tmp['question'] = data_list[i]['question']

            logger2.log(f"-------------index : {i}")
            logger2.log(f"query: {data_list[i]['question']}")
            logger2.log(f"answer: {data_list[i]['answer']}")

            # keywords = data_list[i]['keyword'] # 问题相似度选
            # logger.log(f"keywords : {keywords}")

            retrieve_result_list = [ item.strip() for sublist in data_list[i]['context'] for item in sublist]
            # retrieve_result = data_list[i]['context_str']
            logger.log(f"retrieve result : {json.dumps(retrieve_result_list, indent=2)}")
            
            res = data_list[i]['generated']
            self.response.append(res)
            data_tmp['generated'] = res
            # retrieve_result -> retrieve_triple_3d
            retrieve_triple_3d = self.retriver.kg_seqs_to_triplets_for_ragcache(retrieve_result_list)
            # logger.log(f"retrieve_triple_3d : {json.dumps(retrieve_triple_3d, indent=2)}")
            data_tmp['retrieve_triple_3d'] = retrieve_triple_3d

            distance_threshold = 0.25
            if self.args.dataset_name == 'rgb':
                # distance_threshold = 0.15
                distance_threshold = 0.25
            elif self.args.dataset_name == 'multihop':
                distance_threshold = 0.25

            redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d, distance_threshold = distance_threshold) # 使用v2没毛病
            # redundant_relationship_3d = []
            # all_relationship_group_str = ''
            # all_redundancy_triple_list = []
            # unique_number_count = 0
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            logger.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            logger.log(f"-------All relationship groups-------:\n{all_relationship_group_str}") #输出关系所有分组

            logger2.log(f"-------------unique number count ------- : {unique_number_count}")
            logger2.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            # logger.log(f"{json.dumps(all_redundancy_triple_list)}")
            
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"-------Fliter relationship groups--------:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                # parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                # parsed_response_for_relationship = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api(redundant_relationship_3d)  
                enable_thinking = False
                parsed_response_for_relationship, think_list = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api_qwen(redundant_relationship_3d, enable_thinking)    
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  # delete_list 2d    
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                    logger2.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")

                    redundant.append(len(delete_list)/len(all_redundancy_triple_list))
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    data_tmp['delete_relationship_number'] = len(delete_list)
                    data_tmp['redundancy_rate'] = len(delete_list)/len(all_redundancy_triple_list)
                else:
                    redundant.append(0)
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    delete_list = []
                    data_tmp['delete_relationship_number'] = 0
                    data_tmp['redundancy_rate'] = 0
                    logger.log(f"index{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"---------index{i} response for redundant relationships parse error!!!!!!!!!!----------")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                # 如果真出现了这种情况程序会出现异常
                redundant.append(0)
                data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                delete_list = []
                data_tmp['delete_relationship_number'] = 0
                data_tmp['redundancy_rate'] = 0
                logger.log(f"{i} No redundant relationships")    
                print(f"---------{i} No redundant relationships!!!!!!!-------------") 


            # 检索结果减去被认为是冗余的内容
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retrieve_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                        triple_tmp = [triple[0], triple[1], triple[2]]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                        triple_tmp = [triple[2], triple[1], triple[0]]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple_tmp)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1

            result_triple = [sublist for sublist in one_hop_sentence_2d if sublist not in delete_list]
            result_triple_list = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in result_triple ]
            # logger.log(f"delete_list(filter):\n{json.dumps(delete_list, indent=2)}") 
            # logger.log(f"one_hop_sentence_2d:\n{json.dumps(one_hop_sentence_2d, indent=2)}") 
            # logger.log(f"result_triple(filter):\n{json.dumps(result_triple, indent=2)}") 
                    
            # logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            logger2.log(f"one_hop_sentence(filter):\n{json.dumps(result_triple_list, indent=2)}\n\n") 
            delete_list_str = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in delete_list ]
            logger2.log(f"one_hop_sentence(delete):\n{json.dumps(delete_list_str, indent=2)}\n\n") 

            # 前一个阶段有答案就可以直接利用
            label = data_list[i]['label']
            flag = sum(label) == len(label)
            data_tmp['label'] = flag
            data.append(data_tmp)
            
            if int(flag) == 1:
                acc +=1

            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")
            logger2.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger2.log(f"graph rag response: {res}")

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")
        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")
        
        # 保存检索总数量与冗余统计
        with open(f"./logs/api_qwen/redundancy/Qwen3-32B-no-think_{self.args.option}_{self.args.dataset_name}_redundancy_rate.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
    
    def baseline_redundancy_for_myoriginal_api(self,): # 也能统计我们方法的冗余
        # return True
        import json
        #***
        # file_path = './logs/ragcache/rgb_ent10_pruning30_llama3.json
        
        file_path = f"./logs/{self.args.dataset_name}/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_{self.args.dataset_name}_0.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象

        # file_path = f'./logs/ragcache/{self.args.dataset_name}_ent10_pruning30_llama3.json'
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     data_list = json.load(file)  # 直接使用json.load读取文件对象

        # acc=0
        # for i in range(len(data_list)):
        #     if sum(data_list[i]['label']) == len(data_list[i]['label']):
        #         acc += 1
        # print(f"正确率 acc: {acc}/300")
        # assert False

        # 又不得不涉及到清洗数据

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"./api_qwen/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{current_time}")
        logger2 = Logger(f"./api_qwen/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_for_analysis_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        query_number = 0
        redundant = []
        data = []

        for i in range(len(data_list)):
            # if i >= 10:
            #     break
            data_tmp = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['query']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            data_tmp['index'] = i
            data_tmp['question'] = data_list[i]['query']
            data_tmp['answer'] = data_list[i]['answer']

            logger2.log(f"-------------index : {i}")
            logger2.log(f"query: {data_list[i]['query']}")
            logger2.log(f"answer: {data_list[i]['answer']}")

            # keywords = data_list[i]['keyword'] # 问题相似度选
            # logger.log(f"keywords : {keywords}")

            retrieve_result_list = data_list[i]['filtered_retrieve_result']
            # retrieve_result = data_list[i]['context_str']
            logger.log(f"retrieve result : {json.dumps(retrieve_result_list, indent=2)}")
            
            res = data_list[i]['response']
            self.response.append(res)
            data_tmp['generated'] = res
            # retrieve_result -> retrieve_triple_3d
            retrieve_triple_3d = data_list[i]['filtered_triple_3d']
            # logger.log(f"retrieve_triple_3d : {json.dumps(retrieve_triple_3d, indent=2)}")
            data_tmp['retrieve_triple_3d'] = retrieve_triple_3d

            distance_threshold = 0.15
            if self.args.dataset_name == 'rgb':
                # distance_threshold = 0.15
                distance_threshold = 0.20
            elif self.args.dataset_name == 'multihop':
                distance_threshold = 0.25

            redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d, distance_threshold = distance_threshold) # 使用v2没毛病
            # redundant_relationship_3d = []
            # all_relationship_group_str = ''
            # all_redundancy_triple_list = []
            # unique_number_count = 0
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            logger.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            logger.log(f"-------All relationship groups-------:\n{all_relationship_group_str}") #输出关系所有分组

            logger2.log(f"-------------unique number count ------- : {unique_number_count}")
            logger2.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            # logger.log(f"{json.dumps(all_redundancy_triple_list)}")
            
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"-------Fliter relationship groups--------:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                # parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                # parsed_response_for_relationship = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api(redundant_relationship_3d)  
                enable_thinking = False
                parsed_response_for_relationship, think_list = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api_qwen(redundant_relationship_3d, enable_thinking)    
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  # delete_list 2d    
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                    logger2.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")

                    redundant.append(len(delete_list)/len(all_redundancy_triple_list))
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    data_tmp['delete_relationship_number'] = len(delete_list)
                    data_tmp['redundancy_rate'] = len(delete_list)/len(all_redundancy_triple_list)
                else:
                    redundant.append(0)
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    delete_list = []
                    data_tmp['delete_relationship_number'] = 0
                    data_tmp['redundancy_rate'] = 0
                    logger.log(f"index{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"---------index{i} response for redundant relationships parse error!!!!!!!!!!----------")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                # 如果真出现了这种情况程序会出现异常
                redundant.append(0)
                data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                delete_list = []
                data_tmp['delete_relationship_number'] = 0
                data_tmp['redundancy_rate'] = 0
                logger.log(f"{i} No redundant relationships")    
                print(f"---------{i} No redundant relationships!!!!!!!-------------") 


            # 检索结果减去被认为是冗余的内容
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retrieve_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                        triple_tmp = [triple[0], triple[1], triple[2]]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                        triple_tmp = [triple[2], triple[1], triple[0]]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple_tmp)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1

            result_triple = [sublist for sublist in one_hop_sentence_2d if sublist not in delete_list]
            result_triple_list = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in result_triple ]
            # logger.log(f"delete_list(filter):\n{json.dumps(delete_list, indent=2)}") 
            # logger.log(f"one_hop_sentence_2d:\n{json.dumps(one_hop_sentence_2d, indent=2)}") 
            # logger.log(f"result_triple(filter):\n{json.dumps(result_triple, indent=2)}") 
                    
            # logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            logger2.log(f"one_hop_sentence(filter):\n{json.dumps(result_triple_list, indent=2)}\n\n") 
            delete_list_str = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in delete_list ]
            logger2.log(f"one_hop_sentence(delete):\n{json.dumps(delete_list_str, indent=2)}\n\n") 

            # 前一个阶段有答案就可以直接利用
            flag = data_list[i]['label']
            data_tmp['label'] = flag
            data.append(data_tmp)
            
            if int(flag) == 1:
                acc +=1

            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")
            logger2.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger2.log(f"graph rag response: {res}")

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")
        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")
        
        # 保存检索总数量与冗余统计
        with open(f"./logs/api_qwen/redundancy/Qwen3-32B-no-think_{self.args.option}_{self.args.dataset_name}_redundancy_rate.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def baseline_redundancy_stage2_for_lightRAG(self,): 
        pass

    def graph_rag_baseline_redundancy(self,): # 使用llama3:8b指令模型
        # return True

        import json
        #***
        with open(f"./logs/triplets/{self.args.space_name}.json", "r", encoding="utf-8") as file:
            triplets_score_modify = json.load(file)
        
        # if stage:
        #     with open(f"./logs/{self.args.llmbackend}_two_stage_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}.json", "r", encoding="utf-8") as file:
        #         last_stage_list = json.load(file)

        import datetime
        # if not current_time:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        redundant = []
        
        for i in range(len(self.dataset.query)):
            # if i >= 5:
            #     break
        # for i in range(1):
            print(f"kg modify id : {i+1}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {self.dataset.query[i]}")
            logger.log(f"answer: {self.dataset.answer[i]}")
            # print(f"answer1: {self.dataset.answer[i]}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)
            
            # if stage == 0:
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i], 10) # 问题相似度选
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[i], keywords, 2)
            logger.log(f"keywords : {keywords}")
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")  
            
            retrieve_result, retrieve_triple_3d, sentences = self.retriver.retrieve_path_with_keywords_v1(self.dataset.query[i], keywords, path_depth = 2, pruning=30)
            logger.log(f"retrieve result : {json.dumps(retrieve_result, indent=2)}")
            res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[i], retrieve_result)

            redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d)
            logger.log(f"All relationship groups:\n{all_relationship_group_str}") #输出关系所有分组
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            self.response.append(res)
            # else:
            #     retrieve_result = last_stage_list
            two_delete_count = 0
            # redundant_relationship_3d = []
            # if stage == 1:
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"Fliter relationship groups:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                parsed_response_for_relationship = self.graphrag2.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)      
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"delete relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                    redundant.append(len(delete_list)/len(all_redundancy_triple_list))
                    two_delete_count += len(delete_list)
                else:
                    logger.log(f"{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"{i} response for redundant relationships parse error")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                logger.log(f"{i} No redundant relationships")    
                print(f"{i} No redundant relationships") 
            
            # all_redundancy_triple_list_two_satge = [sublist for sublist in all_redundancy_triple_list if sublist not in delete_list]
            # redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list_2 = self.retriver.find_redundant_relationship_v2_two_stage(all_redundancy_triple_list_two_satge)
            # logger.log(f"All relationship groups:\n{all_relationship_group_str}") #输出关系所有分组

            # if redundant_relationship_3d:
            #     fliter_relationship_group_str = ""
            #     index_tmp = 0
            #     for i_0, group in enumerate(redundant_relationship_3d):
            #         fliter_relationship_group_str += f"Group {i_0}:\n"
            #         for triple in group:
            #             fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
            #             index_tmp += 1
            #     logger.log(f"Fliter relationship groups:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
            #     parsed_response_for_relationship = self.graphrag2.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
            #     if parsed_response_for_relationship:      
            #         keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)      
            #         logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
            #         logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
            #         logger.log(f"delete relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
            #         two_delete_count += len(delete_list)
            #         logger.log(f"two stage delete relationship : {two_delete_count} : {two_delete_count/len(all_redundancy_triple_list)}")
            #         redundant.append(two_delete_count/len(all_redundancy_triple_list))
            #     else:
            #         logger.log(f"{i} response for redundant relationships parse error or llm think No redundant relationships")
            #         print(f"{i} response for redundant relationships parse error")
            #     # print(f"answer2_3: {self.dataset.answer[i]}")
            # else:
            #     logger.log(f"{i} No redundant relationships")    
            #     print(f"{i} No redundant relationships")   

            # print(f"answer3: {self.dataset.answer[i]}")
            
            # if self.graphrag2:        
            #     answer_check = self.graphrag2.chat_without_stream_answer_check(self.dataset_gene.query[i], res, self.dataset.answer[i])
            #     logger.log(f"-------------graph rag response check: {answer_check}")
            #     if "Insufficient information error" in answer_check:
            #         acc_refuse += 1
            #         logger.log(f"-------------graph rag response Insufficient information error----------------------------------")
            #     elif "True" in answer_check:
            #         acc_llm += 1
            #         logger.log(f"-------------graph rag response true (llM)----------------------------------")
            #     elif "Error" in answer_check:
            #         logger.log(f"-------------graph rag response false (llM)----------------------------------")
            #     else:
            #         logger.log(f"-------------graph rag response unknown (llM)----------------------------------")

            flag = False
            if isinstance(self.dataset.answer[i], list):
                instance = [j.lower() for j in self.dataset.answer[i]]
                # print(f"instance {instance}")
                for j in instance:
                    if j in res.lower():
                        flag = True
                        # print(f"True")
                        break
            else:
                # print("222")
                instance = self.dataset.answer[i].lower()
                if instance in res.lower():
                    flag = True
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")

            # logger.log(f"keywords : {keywords}")
            # logger.log(f"retrieve_result : {retrieve_result}")
            # logger.log(f"filtered_result : {filtered_retrieve_result}")
            logger.log(f"graph rag response: {res}")


        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.dataset.answer[i] # 对于rgb而言一定是list
            reference_answer_item['answer_start'] = []
            for _ in range(len(reference_answer_item['text'])):
                reference_answer_item['answer_start'].append(0)
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")

    def graph_rag_sys_redundancy(self,): # 我们系统的冗余率
        # return True
        import json
        
        if self.args.dataset_name == 'rgb':
            with open(f"/home/zhangyz/RAG_2025/logs/rgb_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_{self.args.dataset_name}_{self.args.iteration}.json", 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        elif self.args.space_name == 'hotpotqa' and self.args.iteration == '0':
            with open(f"/home/zhangyz/RAG_2025/logs/stage/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_hotpotqa_0.json", 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        elif self.args.space_name == 'hotpotqa' and self.args.iteration == '100':
            with open(f"/home/zhangyz/RAG_2025/logs/stage/huggingface_graph_rag_baseline_hotpotqa_10_30.json", 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        elif self.args.dataset_name == 'hotpotqa600' and self.args.iteration == '100':
            with open(f"/home/zhangyz/RAG_2025/logs/stage/huggingface_graph_rag_baseline_hotpotqa600_10_30.json", 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        elif self.args.dataset_name == 'hotpotqa600':
            with open(f"/home/zhangyz/RAG_2025/logs/hotpotqa600_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_{self.args.dataset_name}_{self.args.iteration}.json", 'r', encoding='utf-8') as f:
                data_list = json.load(f)

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = Logger(f"./a_logs/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.space_name}_{self.args.iteration}_{current_time}")
        logger2 = Logger(f"./a_logs/redundancy/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.space_name}_{self.args.iteration}_{current_time}_for_analysis")
        
        ##使用最终数据集测评
        acc = 0
        query_number = 0
        redundant = []
        data = []

        for i in range(len(data_list)):
            # if i >= 2:
            #     break
            data_tmp = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['query']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            data_tmp['index'] = i
            data_tmp['question'] = data_list[i]['query']

            logger2.log(f"-------------index : {i}")
            logger2.log(f"query: {data_list[i]['query']}")
            logger2.log(f"answer: {data_list[i]['answer']}")

            # keywords = data_list[i]['keyword'] # 问题相似度选
            # logger.log(f"keywords : {keywords}")
            if self.args.iteration == '100':
                logger.log(f"retrieve result : {json.dumps(data_list[i]['retrieve_result'], indent=2)}")
                res = data_list[i]['res']
                self.response.append(res)
                data_tmp['generated'] = res
                data_tmp['retrieve_result'] = data_list[i]['retrieve_result']
                retrieve_result = data_list[i]['retrieve_result']
                retrieve_triple_3d = data_list[i]['retrieve_result_triple']
                data_tmp['retrieve_triple_3d'] = data_list[i]['retrieve_result_triple']
            else:
                logger.log(f"retrieve result : {json.dumps(data_list[i]['filtered_retrieve_result'], indent=2)}")
                res = data_list[i]['response']
                self.response.append(res)
                data_tmp['generated'] = res
                data_tmp['retrieve_result'] = data_list[i]['filtered_retrieve_result']
                retrieve_result = data_list[i]['filtered_retrieve_result']
                retrieve_triple_3d = data_list[i]['filtered_triple_3d']
                data_tmp['retrieve_triple_3d'] = data_list[i]['filtered_triple_3d']
            
            

            # distance_threshold = 0.10
            distance_threshold = 0.28
            if self.args.iteration == '0' or self.args.iteration == '100':
                distance_threshold = 0.28

            path_list = list(set(retrieve_result))
            if len(path_list) < 2:
                redundant_relationship_3d = []
                all_relationship_group_str = ''
                all_redundancy_triple_list = []
                unique_number_count = 0
            else:
                redundant_relationship_3d, all_relationship_group_str, all_redundancy_triple_list, unique_number_count = self.retriver.find_redundant_relationship_v2(retrieve_triple_3d, distance_threshold = distance_threshold) # 使用v2没毛病
            # redundant_relationship_3d = []
            # all_relationship_group_str = ''
            # all_redundancy_triple_list = []
            # unique_number_count = 0
            logger.log(f"-------------unique number count ------- : {unique_number_count}")
            logger.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            logger.log(f"-------All relationship groups-------:\n{all_relationship_group_str}") #输出关系所有分组

            logger2.log(f"-------------unique number count ------- : {unique_number_count}")
            logger2.log(f"-------------len(all_redundancy_triple_list) ------- : {len(all_redundancy_triple_list)}")
            # logger.log(f"{json.dumps(all_redundancy_triple_list)}")
            
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i_0, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i_0}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"-------Fliter relationship groups--------:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                # parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                # parsed_response_for_relationship = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api(redundant_relationship_3d)  
                enable_thinking = False
                parsed_response_for_relationship, think_list = self.graphrag.chat_without_stream_for_redundant_relationship_v3_api_qwen(redundant_relationship_3d, enable_thinking)    
                if parsed_response_for_relationship:
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  # delete_list 2d    
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    logger.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")
                    logger2.log(f"redundant relationship : {len(delete_list)} : {len(delete_list)/len(all_redundancy_triple_list)}")

                    redundant.append(len(delete_list)/len(all_redundancy_triple_list))
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    data_tmp['delete_relationship_number'] = len(delete_list)
                    data_tmp['redundancy_rate'] = len(delete_list)/len(all_redundancy_triple_list)
                else:
                    redundant.append(0)
                    data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                    delete_list = []
                    data_tmp['delete_relationship_number'] = 0
                    data_tmp['redundancy_rate'] = 0
                    logger.log(f"index{i} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"---------index{i} response for redundant relationships parse error!!!!!!!!!!----------")
                # print(f"answer2_3: {self.dataset.answer[i]}")
            else:
                # 如果真出现了这种情况程序会出现异常
                redundant.append(0)
                data_tmp['relationship_number'] = len(all_redundancy_triple_list)
                delete_list = []
                data_tmp['delete_relationship_number'] = 0
                data_tmp['redundancy_rate'] = 0
                logger.log(f"{i} No redundant relationships")    
                print(f"---------{i} No redundant relationships!!!!!!!-------------") 


            # 检索结果减去被认为是冗余的内容
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retrieve_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                        triple_tmp = [triple[0], triple[1], triple[2]]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                        triple_tmp = [triple[2], triple[1], triple[0]]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple_tmp)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1

            result_triple = [sublist for sublist in one_hop_sentence_2d if sublist not in delete_list]
            result_triple_list = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in result_triple ]
            # logger.log(f"delete_list(filter):\n{json.dumps(delete_list, indent=2)}") 
            # logger.log(f"one_hop_sentence_2d:\n{json.dumps(one_hop_sentence_2d, indent=2)}") 
            # logger.log(f"result_triple(filter):\n{json.dumps(result_triple, indent=2)}") 
                    
            # logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            logger2.log(f"one_hop_sentence(filter):\n{json.dumps(result_triple_list, indent=2)}\n\n") 
            delete_list_str = [ triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2] for triple in delete_list ]
            logger2.log(f"one_hop_sentence(delete):\n{json.dumps(delete_list_str, indent=2)}\n\n") 

            # 前一个阶段有答案就可以直接利用
            if self.args.iteration == '100':
                flag = data_list[i]['flag']
            else:
                flag = data_list[i]['label']
            data_tmp['label'] = flag
            data.append(data_tmp)
            
            if int(flag) == 1:
                acc +=1

            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger.log(f"graph rag response: {res}")
            logger2.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            logger2.log(f"graph rag response: {res}")

        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        self.response = []
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")
        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of redundant: {sum(redundant) / len(redundant)}-----------------------")
        
        # 保存检索总数量与冗余统计
        with open(f"./logs/a_logs/redundancy/Qwen3-32B-no-think_{self.args.option}_{self.args.dataset_name}__{self.args.space_name}_redundancy_rate.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def graph_rag_baseline_analysis(self, current_time = None): # 可以复现ragcache，偏高
        # return True

        import json

        import datetime
        if not current_time:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        logger = Logger(f"./a_logs/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.space_name}_{self.args.entity}_{self.args.pruning}_{self.args.hop}_{current_time}")

        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        num_input_tokens_avg = []
        data = []
        labels_list = []
        select = [22, 26]
        for i in range(len(self.dataset.query)):
            if i not in select:
                continue
            if i > select[-1]:
                break
            # if i >= 1:
            #     break
            dict_data = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {self.dataset.query[i]}")
            logger.log(f"answer: {self.dataset.answer[i]}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)
            dict_data['index'] = i
            dict_data['query'] = self.dataset.query[i]
            dict_data['answer'] = self.dataset.answer[i]
            
            
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i], self.args.entity) # 问题相似度选
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[i], keywords, 2)
            logger.log(f"keywords : {keywords}")
            dict_data['keywords'] = keywords
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")  
            
            retrieve_result, retrieve_result_triple, retrieve_result_sentence = self.retriver.retrieve_path_with_keywords_v1(self.dataset.query[i], keywords, path_depth = self.args.hop, pruning=self.args.pruning)
            logger.log(f"retrieve result : {json.dumps(retrieve_result, indent=2)}")
            dict_data['retrieve_result'] = retrieve_result
            dict_data['retrieve_result_triple'] = retrieve_result_triple
            dict_data['retrieve_result_sentence'] = retrieve_result_sentence
            data.append(dict_data)
            # logger.log(f"retrieve_triple : {json.dumps(retrieve_triple, indent=2)}")
            # logger.log(f"retrieve_sentence : {json.dumps(retrieve_sentence, indent=2)}")
            # res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[i], retrieve_result)

        # with open(f"./logs/stage/{self.args.llmbackend}_{self.args.option}_{self.args.dataset_name}_{self.args.entity}_{self.args.pruning}.json", "w", encoding="utf-8") as f:
        with open(f"./logs/stage/{self.args.llmbackend}_{self.args.option}_{self.args.space_name}_{self.args.entity}_{self.args.pruning}_{self.args.hop}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def graph_rag_baseline(self, current_time = None): # 可以复现ragcache，偏高
        # return True

        import json

        import datetime
        if not current_time:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        logger = Logger(f"./a_logs/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.space_name}_baseline_{self.args.entity}_{self.args.pruning}_{current_time}")

        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        num_input_tokens_avg = []
        data = []
        labels_list = []
        
        for i in range(len(self.dataset.query)):
            # if i >= 1:
            #     break
            dict_data = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {self.dataset.query[i]}")
            logger.log(f"answer: {self.dataset.answer[i]}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)
            dict_data['index'] = i
            dict_data['query'] = self.dataset.query[i]
            dict_data['answer'] = self.dataset.answer[i]
            
            
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i], self.args.entity) # 问题相似度选
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[i], keywords, 2)
            logger.log(f"keywords : {keywords}")
            dict_data['keywords'] = keywords
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")  
            
            retrieve_result, retrieve_result_triple, retrieve_result_sentence = self.retriver.retrieve_path_with_keywords_v1(self.dataset.query[i], keywords, path_depth = 2, pruning=self.args.pruning)
            logger.log(f"retrieve result : {json.dumps(retrieve_result, indent=2)}")
            dict_data['retrieve_result'] = retrieve_result
            dict_data['retrieve_result_triple'] = retrieve_result_triple
            dict_data['retrieve_result_sentence'] = retrieve_result_sentence
            # logger.log(f"retrieve_triple : {json.dumps(retrieve_triple, indent=2)}")
            # logger.log(f"retrieve_sentence : {json.dumps(retrieve_sentence, indent=2)}")
            # res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[i], retrieve_result)


            # logger.log(f"retrieve sentence : {json.dumps(retrieve_sentence, indent=2)}")
            # res = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[i], retrieve_sentence)
            # res = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[i], retrieve_result)
            res, num_input_tokens, prompt = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset_gene.query[i], retrieve_result)
            logger.log(f"num_input_tokens : {num_input_tokens}")
            num_input_tokens_avg.append(num_input_tokens)
            dict_data['res'] = res
            dict_data['num_input_tokens'] = num_input_tokens

            self.response.append(res)
            # if self.graphrag2:        
            #     answer_check = self.graphrag2.chat_without_stream_answer_check(self.dataset_gene.query[i], res, self.dataset.answer[i])
            #     logger.log(f"-------------graph rag response check: {answer_check}")
            #     if "Insufficient information error" in answer_check:
            #         acc_refuse += 1
            #         logger.log(f"-------------graph rag response Insufficient information error----------------------------------")
            #     elif "True" in answer_check:
            #         acc_llm += 1
            #         logger.log(f"-------------graph rag response true (llM)----------------------------------")
            #     elif "Error" in answer_check:
            #         logger.log(f"-------------graph rag response false (llM)----------------------------------")
            #     else:
            #         logger.log(f"-------------graph rag response unknown (llM)----------------------------------")
            
            if self.args.dataset_name == 'dragonball':
                flag_label = self.checkanswer_rougel(res, self.dataset.answer[i])
                dict_data['label'] = flag_label
                labels_list.append(flag_label)
                logger.log(f"checkanswer rougel: {flag_label}")
                logger.log(f"graph rag response: {res}")
                data.append(dict_data)
            else:
                flag = False
                flag_label = self.checkanswer(res, self.dataset.answer[i])
                flag = sum(flag_label) == len(flag_label)
                if int(flag) == 1:
                    acc +=1
                logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
                dict_data['label'] = flag_label
                dict_data['flag'] = flag

                # logger.log(f"keywords : {keywords}")
                # logger.log(f"retrieve_result : {retrieve_result}")
                # logger.log(f"filtered_result : {filtered_retrieve_result}")
                logger.log(f"graph rag response: {res}")
                data.append(dict_data)

        if self.args.dataset_name == 'dragonball':
            accuracy = self.get_accuracy_rougel(labels_list)
            logger.log(f"----------accuracy rougel ...------------\n{accuracy}")
        else:    
            response_list = []
            answer_list = []
            for i in range(query_number):
                response_item = {}
                answer_item = {}
                response_item['id'] = str(i)
                response_item["prediction_text"] = self.response[i]
                response_item["no_answer_probability"] = 0.0 
                response_list.append(response_item)
                reference_answer_item = {}
                # reference_answer_item['text'] = self.dataset_gene.answer[i] # 对于rgb而言一定是list
                # reference_answer_item['answer_start'] = []
                # for _ in range(len(reference_answer_item['text'])):
                #     reference_answer_item['answer_start'].append(0)
                reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
                reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                answer_item['id'] = str(i)
                answer_item['answers'] = reference_answer_item
                answer_list.append(answer_item)
            
            
            squad_v2_metric = load("squad_v2")
            results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
            logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------num_input_tokens_avg: {sum(num_input_tokens_avg)/len(num_input_tokens_avg)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")

        # with open(f"./logs/stage/{self.args.llmbackend}_{self.args.option}_{self.args.dataset_name}_{self.args.entity}_{self.args.pruning}.json", "w", encoding="utf-8") as f:
        with open(f"./logs/stage/{self.args.llmbackend}_{self.args.option}_{self.args.space_name}_{self.args.entity}_{self.args.pruning}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def graph_rag_baseline_data(self, current_time = None): # 保留所有检索结果, 已经和node2的不再一样
        # return True

        import json
        import datetime
        import statistics
        if not current_time:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        # logger = Logger(f"./a_logs/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.space_name}_{self.args.entity}_{self.args.pruning}_{current_time}")
        logger = Logger(f"./a_logs/{self.args.option}_{self.args.space_name}_{self.args.entity}_{self.args.pruning}_{current_time}")

        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        num_input_tokens_avg = []
        data = []
        query_path_count = []
        entity_path_count = []
        
        for i in range(len(self.dataset.query)):
            # if i >= 2:
            #     break
            dict_data = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {self.dataset.query[i]}")
            logger.log(f"answer: {self.dataset.answer[i]}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)
            dict_data['index'] = i
            dict_data['query'] = self.dataset.query[i]
            dict_data['answer'] = self.dataset.answer[i] 
            
            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[i], self.args.entity) # 问题相似度选
            logger.log(f"keywords : {keywords}")
            dict_data['keywords'] = keywords
            
            retrieve_result_2d, retrieve_result_triple_4d, retrieve_result_sentence_2d = self.retriver.retrieve_path_with_keywords_v1_data(self.dataset.query[i], keywords, path_depth = 2, pruning=self.args.pruning)
            dict_data['retrieve_result_2d'] = retrieve_result_2d
            dict_data['retrieve_result_triple_4d'] = retrieve_result_triple_4d
            dict_data['retrieve_result_sentence_2d'] = retrieve_result_sentence_2d

            # retrieve_result = [item for sublist in retrieve_result_2d for item in sublist]  # flatten the 2d list to 1d
            lengths = [len(sublist) for sublist in retrieve_result_2d]
            dict_data['entity_path_count'] = lengths  # store the lengths of each sublist
            dict_data['query_path_count'] = sum(lengths)  # total number of paths for the query
            query_path_count.append(sum(lengths))
            entity_path_count.extend(lengths)  # flatten the list of lengths

            data.append(dict_data)

        # from pprint import pprint
        # print(data[0])
        logger.log(f"\n\n-----------------------entity_path_count_avg: {sum(entity_path_count)/len(entity_path_count)}-----------------------")
        logger.log(f"\n\n-----------------------entity_path_count median: {statistics.median(entity_path_count)}-----------------------")
        logger.log(f"\n\n-----------------------query_path_count_avg: {sum(query_path_count)/len(query_path_count)}-----------------------")
        logger.log(f"\n\n-----------------------query_path_count median: {statistics.median(query_path_count)}-----------------------")


        with open(f"./logs/stage/{self.args.option}_{self.args.dataset_name}_{self.args.entity}_{self.args.pruning}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def graph_rag_baseline_case(self, current_time = None): # 与node不再一致，通过
        import json

        # with open(f"./logs/stage/graph_rag_baseline_data_{self.args.dataset_name}_10_35.json", "r", encoding="utf-8") as file:
        #     data_list = json.load(file)
        with open(f"./logs/stage/graph_rag_baseline_data_{self.args.dataset_name}_10_10000.json", "r", encoding="utf-8") as file:
            data_list = json.load(file)

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_baseline_{current_time}")
        logger = Logger(f"./a_logs/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.space_name}_baseline_{self.args.entity}_{self.args.pruning}_{current_time}")

        
        ##使用最终数据集测评
        acc = 0
        acc_llm = 0
        acc_refuse = 0
        query_number = 0
        num_input_tokens_avg = []
        data = []
        query_rela_count = []
        
        for i in range(len(data_list)):
            # if i >= 1:
            #     break
            dict_data = {}
            print(f"kg modify id : {i}")
            query_number += 1
            logger.log("\n\n")
            logger.log(f"-------------index : {i}")
            logger.log(f"query: {data_list[i]['query']}")
            logger.log(f"answer: {data_list[i]['answer']}")
            # res = self.graphrag.chat_without_stream_with_llama_index(self.dataset.query[i], pruning=30)
            dict_data['index'] = i
            dict_data['query'] = data_list[i]['query']
            dict_data['answer'] = data_list[i]['answer']
            
            keywords = data_list[i]['keywords']
            logger.log(f"keywords : {keywords}")
            dict_data['keywords'] = keywords
            
            retrieve_result = [item for sublist in data_list[i]['retrieve_result_2d'][:self.args.entity] for item in sublist[:self.args.pruning]]
            retrieve_result_triple = [item for sublist in data_list[i]['retrieve_result_triple_4d'][:self.args.entity] for item in sublist[:self.args.pruning]]
            retrieve_result_sentence = [item for sublist in data_list[i]['retrieve_result_sentence_2d'][:self.args.entity] for item in sublist[:self.args.pruning]]

            logger.log(f"retrieve result : {json.dumps(retrieve_result, indent=2)}")
            dict_data['retrieve_result'] = retrieve_result
            dict_data['retrieve_result_triple'] = retrieve_result_triple
            dict_data['retrieve_result_sentence'] = retrieve_result_sentence
            query_rela_count.append(retrieve_result)
            
            res, num_input_tokens, prompt = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset_gene.query[i], retrieve_result)
            logger.log(f"num_input_tokens : {num_input_tokens}")
            num_input_tokens_avg.append(num_input_tokens)
            dict_data['res'] = res
            dict_data['num_input_tokens'] = num_input_tokens
            self.response.append(res)

            flag = False
            flag_label = self.checkanswer(res, self.dataset.answer[i])
            flag = sum(flag_label) == len(flag_label)
            if int(flag) == 1:
                acc +=1
            logger.log(f"Is the response from the LLM correct?: {'Yes' if int(flag) == 1 else 'No'}")
            dict_data['label'] = flag_label
            dict_data['flag'] = flag
            logger.log(f"graph rag response: {res}")
            data.append(dict_data)


        response_list = []
        answer_list = []
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0 
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        
        
        squad_v2_metric = load("squad_v2")
        results = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"----------em f1 ...------------\n{json.dumps(results, indent=2)}")

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------num_input_tokens_avg: {sum(num_input_tokens_avg)/len(num_input_tokens_avg)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------query_rela_count: {sum(query_rela_count)/len(query_rela_count)}-----------------------")

        with open(f"./logs/stage/{self.args.llmbackend}_{self.args.option}_{self.args.dataset_name}_{self.args.entity}_{self.args.pruning}_new.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))

    def create_replacement_map(self, similar_entities_list):
        """
        Creates a dictionary mapping entities to their canonical replacement.
        Example: [["A", "B", "C"], ["D", "E"]] -> {"B": "A", "C": "A", "E": "D"}
        """
        mapping = {}
        for group in similar_entities_list:
            if not group or len(group) < 2:
                continue
            canonical_entity = group[0]
            for i in range(1, len(group)):
                mapping[group[i]] = canonical_entity
        return mapping
    
    def update_json_data(self, source_json_data, similar_entities_list): # 返回一个新的字典
        """
        根据similar_entities_list替换实体，并通过保留得分较高的项来处理键冲突，
        更新JSON数据。

        参数：
            source_json_data (dict): 原始JSON数据，作为Python字典加载。
            similar_entities_list (list): 二维列表，每个子列表包含相似实体，
                                        第一个元素为规范表示。

        返回：
            dict: 包含更新后实体和解决键冲突的新字典。
        """
        replacement_map = self.create_replacement_map(similar_entities_list)
        
        # 这个字典将保存转换后的数据
        # 使用新字典有助于清晰地管理键冲突
        updated_json_data = {}
        
        # 假设source_json_data是一个字典，其中键是由三元组拼接而成的字符串
        # 值是一个包含"triplet"和"score"的字典
        for original_key, item_value in source_json_data.items():
            original_triplet = item_value["triplet"]
            score = item_value["score"]
            
            if len(original_triplet) != 3:
                # 处理格式错误的三元组（可选）
                print(f"警告：跳过格式错误的三元组，键 '{original_key}': {original_triplet}")
                continue
                
            h, r, t = original_triplet
            
            # 应用替换。如果某个实体不在替换表中，则保持不变。
            new_h = replacement_map.get(h, h)
            new_t = replacement_map.get(t, t)
            # 假设关系'r'不需要替换
            
            updated_triplet = [new_h, r, new_t]
            
            # 构造新的键。根据示例"NewKeyExample"，键由三元组元素以空格连接组成。
            # 如果需要无空格拼接，请使用以下代码：
            # new_key = f"{new_h}{r}{new_t}"
            new_key = f"{new_h} {r} {new_t}"
            
            # 处理可能的键冲突
            if new_key not in updated_json_data:
                updated_json_data[new_key] = {
                    "triplet": updated_triplet,
                    "score": score
                }
            else:
                # 检测到键冲突！保留得分较高的条目。
                # 如果得分相等，则保留已存在的条目（先处理的条目）
                existing_item_score = updated_json_data[new_key]["score"]
                if score > existing_item_score:
                    updated_json_data[new_key] = {
                        "triplet": updated_triplet,
                        "score": score
                    }
                # 如果score <= existing_item_score，则不做任何操作；保留已有条目
                
        return updated_json_data

    def combine_answer_formats(self, answer_groups, delimiter=', '): # 组合出答案的笛卡尔积
        from itertools import product
        answer_groups_2d = []
        if not isinstance(answer_groups, list):
            answer_groups_2d.append([answer_groups])
        else:
            for instance in answer_groups:
                if isinstance(instance, list):
                    answer_groups_2d.append(instance)
                else:
                    answer_groups_2d.append([instance])

        combinations = list(product(*answer_groups_2d))  # 笛卡尔积组合
        return [delimiter.join(combo) for combo in combinations]

    def checkanswer(self, prediction, ground_truth, verbose=False): # 特别针对RGB这种有多个答案的
        """
        检查预测答案是否与标准答案匹配。

        :param str prediction:
            预测答案，输入字符串将被转换为小写以进行比较。

        :param ground_truth:
            默认为列表，如果输入为str，将手动转为列表，其中列表中的元素表示为候选答案。
            如果是嵌套列表表示这个问题同时包括多个答案，需要同时回答正确。

        :return:
            二进制标签列表，1 表示匹配成功，0 表示匹配失败。
        :rtype: List[int]

        :示例:

        >>> prediction = "The cat sits on the mat"
        >>> ground_truth = [["cat", "CAT"]]
        >>> checkanswer("cat", ground_truth)
        [1]

        >>> checkanswer("cat and mat", [["cat"], ["MAT", "mat"]])
        [1, 1]
        """
        prediction = prediction.lower()
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        labels = []
        for instance in ground_truth:
            flag = True
            if isinstance(instance, list):
                flag = False
                instance = [i.lower() for i in instance]
                for i in instance:
                    if i in prediction:
                        flag = True
                        break
            else:
                instance = instance.lower()
                if instance not in prediction:
                    flag = False
            labels.append(int(flag))

        # if verbose:
        #     print_text(
        #         f"\nprediction: {prediction}, \nground_truth: {ground_truth}, \nlabels: {labels}\n",
        #         color="yellow",
        #     )

        return labels

    def get_accuracy(self, log : str = None, print=False):
        tt = 0
        labels = []
        for i in range(len(self.response)):
            # if i>=10:
            #     break
            # if i != 94:
            #     continue
            if i>= 5:
                break
            flag = False
            if isinstance(self.dataset.answer[i], list):
                instance = [j.lower() for j in self.dataset.answer[i]]
                for j in instance:
                    if j in self.response[i].lower():
                        flag = True
                        tt += 1
                        break
            else:
                instance = self.dataset.answer[i].lower()
                if instance in self.response[i].lower():
                    flag = True
                    tt += 1
            labels.append(int(flag))
        
        acc = tt / len(self.response)

        if log:
            logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{log}")
            logger.log(f"-----------------------accuracy: {acc}-----------------------")
            for i in range(len(self.dataset.query)):
                # if i<4:
                #     continue
                # if i != 94:
                #     break
                if i >= 5:
                    break
                if labels[i] == 1 and log == "only false":
                    continue
                logger.log(f"id : {i+1}")
                logger.log(f"query: {self.dataset.query[i]}")
                logger.log(f"response: {self.response[i]}")
                logger.log(f"answer: {self.dataset.answer[i]}")
                logger.log(f"Is the response from the LLM correct?: {'Yes' if labels[i] == 1 else 'No'}")
                if self.keywords:
                    logger.log(f"keywords: {self.keywords[i]}")
                if self.retrieve_result:
                    logger.log(f"retrieve_result: {self.retrieve_result[i]}")
                    # logger.log(f"retrieve_result: ")
                    # logger.log(f"{len(self.retrieve_result[i])}")
                    # for x in self.retrieve_result[i]:
                    #     logger.log(f"{len(x)}")
                    #     for y in x:
                    #         logger.log(f"{y}")
                    #     logger.log("\n")
                if self.args.option == "graph_rag_correction":
                    one_hop_tmp = self.retriver.one_hop_correction(self.retrieve_result[i]) 
                    logger.log(f"one_hop: ")  
                    for one_hop in one_hop_tmp:
                        logger.log(f"{one_hop}")                    
                    logger.log(f"correction_relation: {self.correction_relation[i]}")
                    logger.log(f"correction_entity: {self.correction_entity[i]}")
                    logger.log(f"correction_error: {self.correction_error[i]}")
                    logger.log(f"correction_generation: {self.correction_generation[i]}")
                logger.log("\n")

        import datetime
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.log(f'{current_time}\n')
        if print:
            print(f"id: {i + 1} user : {self.dataset.query[i]}")
            print(f"{self.llm.model_name}: \033[33m {self.response[i]}\033[0m")
            print()
        
    def checkanswer_rougel(self, prediction, ground_truth):
        m = RougeL(multiref="best")
        candidate = prediction.split()
        references = [ground_truth.split()]
        m.update(([candidate], [references]))
        return m.compute()
    
    def get_accuracy_rougel(self, labels):
        precision = np.array([x["Rouge-L-P"] for x in labels])
        recall = np.array([x["Rouge-L-R"] for x in labels])
        f1 = np.array([x["Rouge-L-F"] for x in labels])
        return {
            "precision": np.average(precision),
            "recall": np.average(recall),
            "f1": np.average(f1),
        }
        
    def kg_modify(self):

        # load score
        import json
        with open(f"./logs/triplets/{self.args.space_name}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)
        # if self.args.space_name == "multihop":
        #     with open("./logs/multihop_train_random70.json", 'r') as file:
        #         random_number = json.load(file)

        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_test")

        # index = 0
        # while(1):
        for index in range(len(self.dataset.query)):
            # if index not in random_number:
            #     continue
            # if index == 10:
            #     assert(False)
            print(f"kg modify id : {index+1}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {self.dataset.query[index]}")
            logger.log(f"answer: {self.dataset.answer[index]}")

            # response_one = self.graphrag.chat_without_stream(self.dataset.query[index], pruning=30)
            # retrieve_result_one = self.graphrag.get_triplets()

            keywords = self.retriver.extract_keyword(self.dataset.query[index])
            logger.log(f"keywords : {keywords}")


            # keywords = ['Carole king']
            retrieve_result_one = self.retriver.retrieve_2hop_with_keywords(self.dataset.query[index], keywords, pruning=30)
            # print(f"retrieve results : {retrieve_result_one}\n\n\n")
            # assert(False)
            logger.log(f"retrieve result : {retrieve_result_one}")
            # self.retrieve_result.append(retrieve_result_one)

            response_one = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[index], retrieve_result_one)

            logger.log(f"graph rag response: {response_one}")

            if self.check_one_response(response_one, self.dataset.answer[index]):
                self.response.append(response_one)
                logger.log(f"-------------graph rag response true----------------------------------")
            else:
                logger.log(f"-------------graph rag response false ~~~~~")
                response_without_rag_one = self.graphrag.chat_without_rag(self.dataset.query[index])
                logger.log(f" without rag response : {response_without_rag_one}")
                if self.check_one_response(response_without_rag_one, self.dataset.answer[index]):
                    # 证明LLM知道这个知识，但是三元组出了问题，利用LLM去修正KG
                    logger.log(f"------------- without rag response true %%%%%")
                else:
                    # humen in the loop: 人工修正KG，或者想办法修正
                    # 三元组错了，大模型也不知道这个知识
                    logger.log(f"------------- without rag response false ^^^^^")

            for i in range(len(retrieve_result_one)):
                triplet_one = retrieve_result_one[i]
                # list_triplet_one = [triplet_one]
                response_with_one_triplets = self.graphrag.chat_without_stream_with_one_triplet(self.dataset.query[index], triplet_one)
                if not triplet_one in triplets_score:
                    logger.log(f" {i}: graph rag with [{triplet_one}], the retrival result don't exit in triplets score &&&&&")
                    continue
                if self.check_one_response(response_with_one_triplets, self.dataset.answer[index]):
                    a = triplets_score[triplet_one]["score"] + 1
                    triplets_score[triplet_one]["score"] = a
                    logger.log(f" {i}: graph rag with [{triplet_one}], the response is {repr(response_with_one_triplets)}, true, score : {a}")
                else:
                    a = triplets_score[triplet_one]["score"] - 1
                    triplets_score[triplet_one]["score"] = a
                    logger.log(f" {i}: graph rag with [{triplet_one}], the response is {repr(response_with_one_triplets)}, false, score : {a}")
            # index += 1
        
        with open(f"./logs/triplets/{self.args.space_name}_after_modify.json", "w", encoding="utf-8") as file:
            json.dump(triplets_score, file, ensure_ascii=False, indent=4)

    def kg_modify_forward(self, iteration = 0): # 也不是真的前向，只是问答

        print(f"kg_modify_ignore iteration: {iteration}")

        # load score
        import json

        acc = 0
        acc_llm = 0
        acc_refuse = 0
        delete_relationship_num = 0
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        query_number = 0
        relationship_modify = []
        relationship_modify_index = []
        entity_modify = []
        entity_modify_index = []
        delete_entity_list = []
        insert_relationship_list_2d_all = []
        delete_triple_by_relationship_2d_all = []
        delete_triple_by_entity_2d_all = []
        triple_result_average = 0
        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []
        num_input_tokens_avg = []

        forward_data = []

        
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)

        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_test_{iteration}") # only_relationship_

        logger_triplet = Logger(f"triple_score_{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_test_{iteration}") # only_relationship_

        
        for index in range(len(self.dataset.query)):  #2196
            dict_tmp = {}
            # if index >= 20: 
            #     break
            # if index <= 256:
            #     continue
            query_number = query_number + 1

            print(f"kg modify id : {index+1}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {self.dataset.query[index]}")
            logger.log(f"answer: {self.dataset.answer[index]}")
            dict_tmp['question'] = self.dataset.query[index]
            dict_tmp['answer'] = self.dataset.answer[index]

            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[index], 10) # 问题相似度选
            logger.log(f"keywords : {keywords}")
            dict_tmp['keywords'] = keywords
            # 补充关键词
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[index], keywords, 2)
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")            

            # 使用带symbol的path过滤，但是给到大模型的依旧是sentence
            triple_result, filtered_retrieve_result, filtered_triple_3d, filtered_triple_3d_more, max_score, min_score, median_score, mean_score = self.retriver.retrieve_path_with_keywords_basic(self.dataset.query[index], triplets_score, keywords, path_depth = 2, pruning=10, pruning2=30, threshold=0.55, score_threshold = 71)
            max_score_avg.append(max_score)
            min_score_avg.append(min_score)
            median_score_avg.append(median_score)
            mean_score_avg.append(mean_score)
            dict_tmp['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子
            dict_tmp['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
            dict_tmp['filtered_triple_3d_more'] = filtered_triple_3d_more # 带symbol  len = 4
            
            # triple_result 是所有符合相似度阈值的路径 filtered_retrieve_result 是过滤后的4*10，用于回答问题 filtered_triple_3d 是留给冗余处理+feedback
            retrieve_result = []
            for triples_tmp in triple_result:
                retrieve_result.append(str(triples_tmp))
            logger.log(f"len(triple_result) : {len(triple_result)}")
            logger.log(f"max_score: {max_score}, min_score: {min_score}, median_score: {median_score}, mean_score: {mean_score}")
            triple_result_average += len(triple_result)
            logger.log(f"filtered_retrieve_result : {json.dumps(filtered_retrieve_result, indent=2)}")


            response_one, num_input_tokens, prompt = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[index], filtered_retrieve_result)

            self.response.append(response_one)
            dict_tmp['prompt'] = prompt
            dict_tmp['num_input_tokens'] = num_input_tokens
            num_input_tokens_avg.append(num_input_tokens)
            dict_tmp['response'] = response_one
            logger.log(f"num_input_tokens : {num_input_tokens}")
                       

            # 对错判断
            logger.log(f"graph rag response of paths: {response_one}")
            # logger.log(f"graph rag response of triples: {response_ofT}")
            # flag_TF = self.check_one_response(response_one, self.dataset.answer[index])

            flag_label = self.checkanswer(response_one, self.dataset.answer[index])
            flag_TF = sum(flag_label) == len(flag_label)

            if flag_TF:
                # self.response.append(response_one)
                logger.log(f"-------------graph rag response true----------------------------------")
                acc = acc + 1
            else:
                logger.log(f"-------------graph rag response false ~~~~~")
                # response_without_rag_one = self.graphrag.chat_without_rag(self.dataset.query[index])
                # if self.llm.model_name == "deepseek-r1:70b":
                #     response_without_rag_one = response_without_rag_one.split('</think>', 1)[1].strip()
                # # print(f"--------1----------")
                # logger.log(f" without rag response : {response_without_rag_one}")
                # # print(f"--------2----------")
                # # print(f"response_without_rag_one\n{response_without_rag_one}")
                # # print(f"--------3----------")
                # flag_label_tmp = self.checkanswer(response_one, self.dataset.answer[index])
                # flag_TF_tmp = sum(flag_label_tmp) == len(flag_label_tmp)
                # # if self.check_one_response(response_without_rag_one, self.dataset.answer[index]):
                # if flag_TF_tmp:
                #     # 证明LLM知道这个知识，但是三元组出了问题，利用LLM去修正KG
                #     logger.log(f"------------- without rag response true %%%%%")
                # else:
                #     # humen in the loop: 人工修正KG，或者想办法修正
                #     # 三元组错了，大模型也不知道这个知识
                #     logger.log(f"------------- without rag response false ^^^^^")
            dict_tmp['label'] = flag_TF
            forward_data.append(dict_tmp)

        response_list = []
        answer_list = []
        print(f"query_number: {query_number}")
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])

            # reference_answer_item['text'] = self.dataset.answer[i] # 对于rgb而言一定是list
            # reference_answer_item['answer_start'] = []
            # for _ in range(len(reference_answer_item['text'])):
            #     reference_answer_item['answer_start'].append(0)
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        squad_v2_metric = load("squad_v2")
        results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")
        self.response = [] # 每轮结束清空，不然分数不变，因为response累积
        
        # squad_metric = load("squad")
        # results = squad_metric.compute(predictions = response_list, references = answer_list)
        # logger.log(f"--------squad em f1 ...------------\n{json.dumps(results, indent=2)}")

        if max_score_avg:
            max_score = max(max_score_avg)
            min_score = min(min_score_avg)
            median_score = statistics.median(median_score_avg)
            mean_score = sum(mean_score_avg) / len(mean_score_avg)
        else:
            max_score = min_score = median_score = mean_score = None

        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------num_input_tokens_avg: {statistics.median(num_input_tokens_avg)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")
        logger.log(f"\n\n-----------------------len(triple_result): {triple_result_average/query_number}-----------------------")
        logger.log(f"\n\n-----------------------max_score: {max_score}-----------------------")
        logger.log(f"\n\n-----------------------min_score: {min_score}-----------------------")
        logger.log(f"\n\n-----------------------median_score: {median_score}-----------------------")
        logger.log(f"\n\n-----------------------mean_score: {mean_score}-----------------------")

        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_{self.args.option}_{self.args.dataset_name}_{iteration}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(forward_data, indent=2))

    def kg_modify_feedback(self, iteration = 0): # 处理实体

        # assert False

        print(f"kg_modify_ignore iteration: {iteration}")
        self.retriver.entity_embedding_bak(iteration) # 实体嵌入备份

        # load score
        import json

        acc = 0
        acc_llm = 0
        acc_refuse = 0
        delete_relationship_num = 0
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        query_number = 0
        relationship_modify = []
        relationship_modify_index = []
        entity_modify = []
        entity_modify_index = []
        delete_entity_list = []
        insert_relationship_list_2d_all = []
        delete_triple_by_relationship_2d_all = []
        delete_triple_by_entity_2d_all = []
        keep_and_delete_entity_list_2d_all = []
        entity_process_log = []
        triple_result_average = 0
        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []

        forward_data = []
        
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)

        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_kg_modify_forward_{self.args.dataset_name}_{iteration}.json", 'r', encoding='utf-8') as file:
            data_list = json.load(file)  # 直接使用json.load读取文件对象

        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{iteration}")

        logger_entity = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_entity_{iteration}")

        logger_triplet = Logger(f"triple_score_{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{iteration}")

        for index in range(len(data_list)):  #实体处理
            dict_tmp = {}
            entity_log_dict = {}
            if index >= 20: 
                break
            logger_entity.log(f"-------------index : {index}")
            print(f"kg modify id(redundant entity) : {index+1}")
            filtered_triple_3d_more = data_list[index]['filtered_triple_3d_more'] 

            if filtered_triple_3d_more and len(filtered_triple_3d_more)> 1:
                # redundant_relationship_3d, all_relationship_group_str, *rest = self.retriver.find_redundant_relationship_v2(filtered_triple_3d_more) # triple_result[:80] redundant_relationship_3d 无symbol 
                redundant_entity_2d, all_entity_group_str = self.retriver.find_redundant_entity(filtered_triple_3d_more)
                # redundant_entity_2d = []
                # all_entity_group_str = "---------------No entity operation------------"
            else:
                # redundant_relationship_3d = [] 
                # all_relationship_group_str = "--------No relationship group!!!---------"
                redundant_entity_2d = [] 
                all_entity_group_str = "-------------No entity group!!!------------"
                logger_entity.log(f"------------------filtered_retrieve_result is empty or length <= 1------------------")    
            
            logger_entity.log(f"All entity groups:\n{all_entity_group_str}") #输出实体所有分组
            if redundant_entity_2d:
                # fliter_entity_group_str = ""
                # for i, group in enumerate(redundant_entity_2d):
                #     fliter_entity_group_str += f"Group {i}:\n"
                #     fliter_entity_group_str = fliter_entity_group_str + str(group) + '\n'
                # logger.log(f"Fliter entity groups:\n{fliter_entity_group_str}") #输出实体所有分组中至少有两个关系的分组 

                # 存在二次检索，如果不同问题之前存在关联的实体被删除，则后续会报错，数组索引越界，可以改代码，可以直接过程中不修改数据库了
                sentences_of_entity_3d = self.retriver.retrieve_path_for_redundant_entity(redundant_entity_2d, path_depth = 2, pruning=5)   
                logger_entity.log(f"Fliter entity groups:\n{json.dumps(redundant_entity_2d, indent=2)}") 
                entity_log_dict['Fliter_entity_groups'] = redundant_entity_2d
                logger_entity.log(f"sentences_of_entity_3d:\n{json.dumps(sentences_of_entity_3d, indent=2)}")   
                entity_log_dict['sentences_of_entity_3d'] = sentences_of_entity_3d
                # parsed_response_for_entity = self.graphrag.chat_without_stream_for_redundant_entity_v2(redundant_entity_2d)
                parsed_response_for_entity = self.graphrag.chat_without_stream_for_redundant_entity_v3(redundant_entity_2d, sentences_of_entity_3d) # 添加三元组辅助
                if parsed_response_for_entity:# 正确解析
                    # parsed_response_for_entity_check = self.graphrag.chat_without_stream_for_redundant_entity_check(redundant_entity_2d, parsed_response_for_entity)

                    entity_modify_index.append(parsed_response_for_entity)
                    entity_modify.append(redundant_entity_2d)
                    # keep_and_delete_entity_str, delete_entity, insert_relationship_list_2d, delete_relationship_list_2d, insert_relationship_all, delete_relationship_all, keep_and_delete_entity_list_2d = self.retriver.process_redundant_entity(parsed_response_for_entity, redundant_entity_2d, True)  
                    keep_and_delete_entity_str, delete_entity, insert_relationship_list_2d, delete_relationship_list_2d, insert_relationship_all, delete_relationship_all, keep_and_delete_entity_list_2d = self.retriver.process_redundant_entity(parsed_response_for_entity, redundant_entity_2d, False)  
                    keep_and_delete_entity_list_2d_all.extend(keep_and_delete_entity_list_2d)
                    entity_log_dict['keep_and_delete_entity_list_2d'] = keep_and_delete_entity_list_2d

                    insert_relationship_list_2d_all.extend(insert_relationship_list_2d)
                    delete_triple_by_entity_2d_all.extend(delete_relationship_list_2d)
                    delete_relationship_by_entity_num += len(delete_relationship_list_2d)
                    insert_relationship_by_entity_num += len(insert_relationship_list_2d)
                    delete_entity_list.extend(delete_entity) 
                    delete_entity_num += len(delete_entity)
                    logger_entity.log(f"delete entity : \n{delete_entity}")     # 删除的实体
                    logger_entity.log(f"keep and delete entity : \n{keep_and_delete_entity_str}")     # 保留的实体
                    logger_entity.log(f"parsed_response_for_redundant_entity: \n{parsed_response_for_entity}")
                    # logger.log(f'insert relationship: \n{json.dumps(insert_relationship_all, indent=2)}')
                    # logger.log(f'dlete relationship: \n{json.dumps(delete_relationship_all, indent=2)}')
                    logger_entity.log(f'insert relationship: \n{json.dumps(insert_relationship_all, indent=2)}')
                    logger_entity.log(f'delete relationship: \n{json.dumps(delete_relationship_all, indent=2)}')
                    # --------------------------------------- 不再需要
                    # self.retriver.delete_redundant_entity(delete_entity, False) # False
                    
                    # 让删除实体和新增实体一一对应，好可以传递分数
                    # insert_relationship_all和 delete_relationship_all应该是一一对应的
                    # for triple_delete, triple_insert in zip(delete_relationship_list_2d, insert_relationship_list_2d):   
                    #     sentence_delete = f"{triple_delete[0]} {triple_delete[1]} {triple_delete[2]}"
                    #     score_delete = 100
                    #     if sentence_delete in triplets_score:
                    #         score_delete = triplets_score[sentence_delete]["score"]
                    #         value = triplets_score.pop(sentence_delete, {})
                    #     else:
                    #         logger_entity.log(f"{sentence_delete}(delete by entity) not exist in triple_score!")

                    #     sentence_insert = f"{triple_insert[0]} {triple_insert[1]} {triple_insert[2]}"
                    #     item = [triple_insert[0], triple_insert[1], triple_insert[2]]

                    #     if not sentence_insert in triplets_score:# 分数暂时简单初始化100
                    #         triplets_score[sentence_insert] = {}
                    #         triplets_score[sentence_insert]["triplet"] = item
                    #         triplets_score[sentence_insert]["score"] = score_delete

                    # for triple in insert_relationship_list_2d:        
                    #     sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                    #     item = [triple[0], triple[1], triple[2]]

                    #     if not sentence in triplets_score:# 分数暂时简单初始化100
                    #         triplets_score[sentence] = {}
                    #         triplets_score[sentence]["triplet"] = item
                    #         triplets_score[sentence]["score"] = 100
                    
                    # for triple in delete_relationship_list_2d:
                    #     sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                    #     value = triplets_score.pop(sentence, {})
                    #     if not value:
                    #         logger.log(f"{sentence}(delete by entity) not exist in triple_score!")
                else:
                    logger_entity.log(f"{index} response for redundant entities parse error or llm think No redundant entities")
                    print(f"{index} response for redundant entities parse error")
            else:
                logger_entity.log(f"{index} No redundant entities")    
                print(f"{index} No redundant entities")

            if "Fliter_entity_groups" not in entity_log_dict:
                entity_log_dict["Fliter_entity_groups"] = []
            if "sentences_of_entity_3d" not in entity_log_dict:
                entity_log_dict['sentences_of_entity_3d'] = []
            if "keep_and_delete_entity_list_2d" not in entity_log_dict:
                entity_log_dict['keep_and_delete_entity_list_2d'] = []

            entity_process_log.append(entity_log_dict)

        # 进行映射与实体替换
        replacement_map = self.create_replacement_map(keep_and_delete_entity_list_2d_all)
        triplets_score = self.update_json_data(triplets_score, keep_and_delete_entity_list_2d_all) # 进行实体替换，得到新的triplets_score
        
        for index in range(len(data_list)):  #2196
            dict_tmp = {}
            if index >= 20: 
                break
            # if index <= 256:
            #     continue
            query_number = query_number + 1

            print(f"kg modify id : {index+1}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {data_list[index]['question']}")
            logger.log(f"answer: {data_list[index]['answer']}")

            keywords = data_list[index]['keywords']
            logger.log(f"keywords : {keywords}")   

            filtered_retrieve_result = data_list[index]['filtered_retrieve_result']  # feedback用，因为是多轮对话
            filtered_triple_3d = data_list[index]['filtered_triple_3d'] # 处理冗余用，feedback也用 # 带symbol  len = 4
            filtered_triple_3d_more = data_list[index]['filtered_triple_3d_more']  # 带symbol  len = 4
            response_one = data_list[index]['response']
            self.response.append(response_one)
            logger.log(f"filtered retrieve result : {json.dumps(filtered_retrieve_result, indent=2)}")  

            # 实体处理log
            if entity_process_log[index]:
                logger.log(f"Fliter entity groups:\n{json.dumps(entity_process_log[index]['Fliter_entity_groups'], indent=2)}") 
                logger.log(f"sentences_of_entity_3d:\n{json.dumps(entity_process_log[index]['sentences_of_entity_3d'], indent=2)}") 
                logger.log(f"keep_and_delete_entity_list_2d:\n{json.dumps(entity_process_log[index]['keep_and_delete_entity_list_2d'], indent=2)}")   

            else:
                logger.log(f"----------------No entity process~~~~~~~~~~~~~~~~~~~")
                

            # 处理entity之后，在不重新检索的条件下，按照融合的实体重新组织关系
            if filtered_retrieve_result and len(filtered_retrieve_result)> 1:
                # redundant_relationship_3d, all_relationship_group_str, *rest = self.retriver.find_redundant_relationship_v2(filtered_triple_3d_more) # triple_result[:80] redundant_relationship_3d 无symbol 
                redundant_relationship_3d, all_relationship_group_str, triple_list, unique_number_count, seen_origin = self.retriver.find_redundant_relationship_v4(filtered_triple_3d_more, replacement_map) # triple_result[:80] redundant_relationship_3d 无symbol 
                # redundant_relationship_3d = []
                # all_relationship_group_str = ""
                # seen_origin = []

                # redundant_entity_2d, all_entity_group_str = self.retriver.find_redundant_entity(filtered_triple_3d_more)
                # redundant_entity_2d = []
                # all_entity_group_str = "---------------No entity operation------------"
            else:
                redundant_relationship_3d = [] 
                all_relationship_group_str = "--------No relationship group!!!---------"
                redundant_entity_2d = [] 
                all_entity_group_str = "-------------No entity group!!!------------"
                logger.log(f"------------------filtered_retrieve_result is empty or length <= 1------------------")    

            logger.log(f"Original All relationship groups:\n{json.dumps(list(seen_origin), indent=2)}") #输出关系所有分组，原始
            logger.log(f"All relationship groups:\n{all_relationship_group_str}") #输出关系所有分组，替换实体组织
            if redundant_relationship_3d:
                fliter_relationship_group_str = ""
                index_tmp = 0
                for i, group in enumerate(redundant_relationship_3d):
                    fliter_relationship_group_str += f"Group {i}:\n"
                    for triple in group:
                        fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                        index_tmp += 1
                logger.log(f"Fliter relationship groups:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_v3(redundant_relationship_3d)   
                if parsed_response_for_relationship:# 正确解析
                    relationship_modify_index.append(parsed_response_for_relationship)
                    relationship_modify.append(redundant_relationship_3d)             
                    # keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, True)  
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  

                    delete_relationship_num += len(delete_list)  
                    delete_triple_by_relationship_2d_all.extend(delete_list)            
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    
                    
                    for triple in delete_list: # 即时修改
                        sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                        value = triplets_score.pop(sentence, {})
                        if not value:
                            logger.log(f"{sentence}(delete by relationship) not exist in triple_score!")
                        else:
                            pass
                else:
                    logger.log(f"{index} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"{index} response for redundant relationships parse error")
            else:
                logger.log(f"{index} No redundant relationships")    
                print(f"{index} No redundant relationships")            
            
            
                              
            #### 还需要在实体列表实体嵌入中删除这个实体！！！ 在nebulagraph.py的process_redundant_entity中进行
                
            # # 原冗余处理无嵌入分组
            # if redundant_relationship:
            #     response_for_redundant_relationship = self.graphrag.chat_without_stream_for_redundant_relationship(redundant_relationship)
            #     ### 数据库处理冗余关系并记录更改返回
            #     process_redundant_relationship_log, delete_list = self.retriver.process_redundant_relationship(response_for_redundant_relationship, redundant_relationship_symbol)
            #     redundant_relationship_str = ""
            #     for relationship_group in redundant_relationship:
            #         for relationship in relationship_group:
            #             redundant_relationship_str = redundant_relationship_str + "\n" + relationship
            #     logger.log(f"response_for_redundant_relationship: {response_for_redundant_relationship}")
            #     logger.log(f"redundant relationship (Unprocessed) : {redundant_relationship_str}")   # 需要去冗余的关系
            #     logger.log(f"keep relationship : {process_redundant_relationship_log}")     # 保留的关系
            
            #     ### 并且对应的边在triple记录里需要被删除  先处理冗余会导致后续处理评分时有一些三元组不再存在
            #     delete_relationship_num += len(delete_list)
            #     for triple_str in delete_list:
            #         try:
            #             if triple_str in triplets_score:
            #                 logger.log(f"Redundant relationship: {triple_str} has been deleted in triplets score %%%%%")
            #             value = triplets_score.pop(triple_str)
            #         except KeyError as e:
            #             logger.log(f"Error: {e.args[0]}. {triple_str}, the redundant triple (relationship) don't exit in triplets score &&&&&")         
            # else:
            #     logger.log(f"redundant relationship lsit is empty")

            # response_ofT = self.graphrag.chat_without_stream_with_triplets(self.dataset_gene.query[index], retrieve_one_hop)
            # if self.llm.model_name == "deepseek-r1:70b":
            #     response_ofT = response_ofT.split('</think>', 1)[1].strip()

            # if self.graphrag2 and self.args.llm_chat == "llama3:8b-instruct-fp16":
            #     answer_check = self.graphrag2.chat_without_stream_answer_check_llama3_8b(self.dataset_gene.query[i], response_one, self.dataset.answer[i])
            # else:
            #     answer_check = self.graphrag.chat_without_stream_answer_check(self.dataset_gene.query[i], response_one, self.dataset.answer[i])

            answer_check = self.graphrag.chat_without_stream_answer_check(data_list[index]['question'], response_one, data_list[index]['answer'])
            # answer_check = self.graphrag.chat_without_stream_with_who_are_you(self.dataset_gene.query[index], [])
 
            # print(f"answer_check: {answer_check}")
            logger.log(f"-------------graph rag response check: {answer_check}")
            if "Insufficient information error" in answer_check:
                acc_refuse += 1
                logger.log(f"-------------graph rag response Insufficient information error----------------------------------")
            elif "True" in answer_check:
                acc_llm += 1
                logger.log(f"-------------graph rag response true (llM)----------------------------------")
            elif "Error" in answer_check:
                logger.log(f"-------------graph rag response false (llM)----------------------------------")
            else:
                logger.log(f"-------------graph rag response unknown (llM)----------------------------------")

            # 对错判断
            logger.log(f"graph rag response of paths: {response_one}")
            # logger.log(f"graph rag response of triples: {response_ofT}")
            # flag_TF = self.check_one_response(response_one, data_list[index]['answer'])
            flag_label = self.checkanswer(response_one, self.dataset.answer[index])
            flag_TF = sum(flag_label) == len(flag_label)
            
            if flag_TF:
                # self.response.append(response_one)
                logger.log(f"-------------graph rag response true----------------------------------")
                acc = acc + 1
            else:
                logger.log(f"-------------graph rag response false ~~~~~")
                # response_without_rag_one = self.graphrag.chat_without_rag_qwen(data_list[index]['question'])
                # if self.llm.model_name == "deepseek-r1:70b":
                #     response_without_rag_one = response_without_rag_one.split('</think>', 1)[1].strip()
                # logger.log(f" without rag response : {response_without_rag_one}")
                # # if self.check_one_response(response_without_rag_one, data_list[index]['answer']):
                # flag_label_tmp = self.checkanswer(response_one, self.dataset.answer[index])
                # flag_tmp = sum(flag_label) == len(flag_label_tmp)
                # if flag_tmp:
                #     # 证明LLM知道这个知识，但是三元组出了问题，利用LLM去修正KG
                #     logger.log(f"------------- without rag response true %%%%%")
                # else:
                #     # humen in the loop: 人工修正KG，或者想办法修正
                #     # 三元组错了，大模型也不知道这个知识
                #     logger.log(f"------------- without rag response false ^^^^^")
                    
            # 新的反馈处理
            
            # -------------反馈处理也需要进行实体替换-------------

            # continue
            # 反馈处理
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            seen_origin = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in filtered_triple_3d:
                for triple in path_triple:
                    triple_replace = [replacement_map.get(triple[0], triple[0]), triple[1], replacement_map.get(triple[2], triple[2]), triple[3]]
                    sentence = ""
                    sentence_origin = ''
                    if triple_replace[3] == '->':
                        sentence = triple_replace[0]+' '+triple_replace[1].replace("_"," ")+' '+triple_replace[2]
                        sentence_origin = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                    elif triple_replace[3] == '<-':
                        sentence = triple_replace[2]+' '+triple_replace[1].replace("_"," ")+' '+triple_replace[0]
                        sentence_origin = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple_replace)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n' # 有了id方便分析
                        sentence_id += 1
                    if sentence_origin not in seen_origin:
                        seen_origin.add(sentence_origin)

            logger.log(f"one_hop_sentence_origin:\n{json.dumps(list(seen_origin))}")         
            logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            response_insufficient, response_correct, response_error = self.graphrag.chat_without_stream_for_socre_feedback_latest_v4(data_list[index]['question'], response_one, filtered_retrieve_result, one_hop_sentence_list, flag_TF)
            logger.log(f"response_for_score : {response_insufficient}")
            logger.log(f"correct_numbers : {response_correct}")
            logger.log(f"error_numbers : {response_error}")
            logger.log(f"Score: ")
            if not (response_insufficient or response_correct or response_error) :
                logger.log(f"{index} Exceeded the corresponding number of times, all feedback is empty")
                print(f"{index} Exceeded the corresponding number of times, all feedback is empty")
            # elif response_insufficient == "Insufficient information" or response_insufficient == "No feedback":
            #     logger.log(f"---------------------feedback: {response_insufficient}----------------------------")
            elif response_correct or response_error :
                for i in range(len(one_hop_sentence_2d)):
                    if i in response_correct:
                        if one_hop_sentence_2d[i][3] == '->':                           
                            sentence = f"{one_hop_sentence_2d[i][0]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][2]}"
                        elif one_hop_sentence_2d[i][3] == '<-':                           
                            sentence = f"{one_hop_sentence_2d[i][2]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][0]}"
                        if not sentence in triplets_score:
                            logger.log(f" {i}: {one_hop_sentence_2d[i]}, the retrival result (good triple) don't exit in triplets score &&&&&") # 如果被误删了就再加进去？？？暂时不加
                            continue
                        else:
                            if response_correct[i] < 4 and response_correct[i] >0:
                                a = triplets_score[sentence]["score"] + response_correct[i]*5
                                triplets_score[sentence]["score"] = a
                                logger.log(f" {i}: {one_hop_sentence_2d[i]}, correct triple new score : {a}")
                                logger_triplet.log(f"[{sentence}] : {response_correct[i]*5}") 
                            else:
                                logger.log(f" {i}: {one_hop_sentence_2d[i]}, the correct triplets score error &&&&&")
                    elif i in response_error:
                        if one_hop_sentence_2d[i][3] == '->':                           
                            sentence = f"{one_hop_sentence_2d[i][0]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][2]}"
                        elif one_hop_sentence_2d[i][3] == '<-':                           
                            sentence = f"{one_hop_sentence_2d[i][2]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][0]}"
                        if not sentence in triplets_score:
                            logger.log(f" {i}: {one_hop_sentence_2d[i]}, the retrival result (bad triple) don't exit in triplets score &&&&&") # error不存在就不存在了
                            continue
                        else:
                            if response_error[i] < 4 and response_error[i] > 0 :
                                a = triplets_score[sentence]["score"] - response_error[i]*5
                                triplets_score[sentence]["score"] = a
                                logger.log(f" {i}: {one_hop_sentence_2d[i]}, error triple new score : {a}")
                                logger_triplet.log(f"[{sentence}] : -{response_error[i]*5}")
                            else:
                                logger.log(f" {i}: {one_hop_sentence_2d[i]}, the error triplets score error &&&&&") 

            # filtered_triple_3d
            # response_insufficient, response_correct, response_error = self.graphrag.chat_without_stream_for_socre_feedback_latest_v4(data_list[index]['question'], response_one, filtered_retrieve_result, flag_TF)
            # print(f"response_insufficient: {response_insufficient}")
            # print(f"response_correct: {response_correct}")
            # print(f"response_error: {response_error}")

            # # logger.log(f"response_for_score : {response_for_score}")
            # if self.llm.model_name == "deepseek-r1:70b":
            #     response_for_score = response_for_score.split('</think>', 1)[1].strip()
  
            # tmp = response_for_score.strip().split('\n')
            # correct_numbers = {}
            # error_numbers = {}
            # for res in tmp[:2]:
            #     res.strip()
            #     if res.startswith("Correct: "):
            #         correct_pairs = res.split("Correct:")[1].strip().split()
            #         for pair in correct_pairs:
            #             parts = pair.split(":")
            #             if len(parts) == 2:
            #                 try:
            #                     number, degree = map(int, parts)
            #                     correct_numbers[int(number)] = int(degree)
            #                 except ValueError as e:
            #                     logger.log(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
            #                 except Exception as e:
            #                     logger.log(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
            #             else:
            #                 logger.log(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
            #     elif res.startswith("Error: "):
            #         error_pairs = res.split("Error:")[1].strip().split()
            #         for pair in error_pairs:
            #             parts = pair.split(":")
            #             if len(parts) == 2:
            #                 try:
            #                     number, degree = map(int, parts)
            #                     error_numbers[int(number)] = int(degree)
            #                 except ValueError as e:
            #                     logger.log(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
            #                 except Exception as e:
            #                     logger.log(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
            #             else:
            #                 logger.log(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
            # logger.log(f"response_for_score : {response_for_score}")
            # logger.log(f"correct_numbers : {correct_numbers}")
            # logger.log(f"error_numbers : {error_numbers}")
            # logger.log(f"Score: ")
            # for i in range(len(retrieve_one_hop)):
            #     if i+1 in correct_numbers:
            #         sentence = self.retriver.triple_into_sentence(retrieve_one_hop[i])
            #         if not sentence in triplets_score:
            #             logger.log(f" {i+1}: {retrieve_one_hop[i]}, the retrival result don't exit in triplets score &&&&&")
            #             continue
            #         else:
            #             if correct_numbers[i+1] < 4 and correct_numbers[i+1] >0:
            #                 a = triplets_score[sentence]["score"] + correct_numbers[i+1]
            #                 triplets_score[sentence]["score"] = a
            #                 logger.log(f" {i+1}: {retrieve_one_hop[i]}, correct triple new score : {a}")
            #                 logger_triplet.log(f"[{sentence}] : {correct_numbers[i+1]}") 
            #             else:
            #                 logger.log(f" {i+1}: {retrieve_one_hop[i]}, the correct triplets score error &&&&&")
            #     elif i+1 in error_numbers:
            #         sentence = self.retriver.triple_into_sentence(retrieve_one_hop[i])
            #         if not sentence in triplets_score:
            #             logger.log(f" {i+1}: {retrieve_one_hop[i]}, the retrival result don't exit in triplets score &&&&&")
            #             continue
            #         else:
            #             if error_numbers[i+1] < 4 and error_numbers[i+1] > 0 :
            #                 a = triplets_score[sentence]["score"] - error_numbers[i+1]
            #                 triplets_score[sentence]["score"] = a
            #                 logger.log(f" {i+1}: {retrieve_one_hop[i]}, error triple new score : {a}")
            #                 logger_triplet.log(f"[{sentence}] : -{error_numbers[i+1]}")
            #             else:
            #                 logger.log(f" {i+1}: {retrieve_one_hop[i]}, the error triplets score error &&&&&")
            
        # for parsed_response_for_relationship, redundant_relationship_3d in zip(relationship_modify_index, relationship_modify):
        #     self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, True)
        
        # for parsed_response_for_entity, redundant_entity_2d in zip(entity_modify_index, entity_modify):
        #     self.retriver.process_redundant_entity(parsed_response_for_entity, redundant_entity_2d, True)
        # logger.log(f"all delete entity : \n{delete_entity_list}")
        # self.retriver.delete_redundant_entity(delete_entity_list)
        # self.retriver.delete_redundant_entity([], True)
        
        # -----------------update_triple_json----------------------- insert and delete
        # for triple in insert_relationship_list_2d_all:        
        #     sentence = f"{triple[0]} {triple[1]} {triple[2]}"
        #     item = [triple[0], triple[1], triple[2]]

        #     if not sentence in triplets_score:# 分数暂时简单初始化100
        #         triplets_score[sentence] = {}
        #         triplets_score[sentence]["triplet"] = item
        #         triplets_score[sentence]["score"] = 100
        
        # for triple in delete_triple_by_relationship_2d_all:
        #     sentence = f"{triple[0]} {triple[1]} {triple[2]}"
        #     value = triplets_score.pop(sentence, {})
        #     if not value:
        #         logger.log(f"{value} not exist in triple_score!")
            

        # for triple in delete_triple_by_entity_2d_all:
        #     sentence = f"{triple[0]} {triple[1]} {triple[2]}"
        #     value = triplets_score.pop(sentence, {})
        #     if not value:
        #         logger.log(f"{value} not exist in triple_score!")

        
        # --------------------------------
        # pass
        # self.retriver.delete_redundant_entity([], True) # 修改嵌入

        # EM F1
        response_list = []
        answer_list = []
        print(f"query_number: {query_number}")
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        squad_v2_metric = load("squad_v2")
        results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")
        self.response = [] # 每轮结束清空，不然分数不变，因为response累积
        
        # squad_metric = load("squad")
        # results = squad_metric.compute(predictions = response_list, references = answer_list)
        # logger.log(f"--------squad em f1 ...------------\n{json.dumps(results, indent=2)}")


        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")

        logger.log(f"\n\n-----------------------delete_relationship_num: {delete_relationship_num}-----------------------")
        logger.log(f"\n\n-----------------------delete_relationship_by_entity_num: {delete_relationship_by_entity_num}-----------------------")
        logger.log(f"\n\n-----------------------insert_relationship_by_entity_num: {insert_relationship_by_entity_num}-----------------------")
        logger.log(f"\n\n-----------------------delete_entity_num: {delete_entity_num}-----------------------")

        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration+1}.json", "w", encoding="utf-8") as file:
            json.dump(triplets_score, file, ensure_ascii=False, indent=4)

        # 根据triple_score重建entity_embedding
        entities_new = set()
        for value in triplets_score.values():
            entities_new.add(value['triplet'][0])
            entities_new.add(value['triplet'][2])
        self.retriver.generate_new_entity_embedding_standard(list(sorted(entities_new)))
    
    def kg_modify_api_qwen(self, iteration = 0): # 

        print(f"kg_modify_api iteration: {iteration}")

        ### 应该重新生成一份实体嵌入
        # self.retriver.entity_embedding_bak(iteration) # 实体嵌入备份

        # load score
        import json

        acc = 0
        acc_llm = 0
        acc_part = 0
        acc_llm_part = 0
        delete_relationship_num = []
        query_number = 0
        relationship_score_modify = 0
        num_input_tokens_avg = []

        forward_data = []
        test_data = []

        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)
        
        # 这轮开始重新生成嵌入
        entities_new = set()
        for value in triplets_score.values():
            if value['score'] > 70:
                entities_new.add(value['triplet'][0])
                entities_new.add(value['triplet'][2])
        self.retriver.generate_new_entity_embedding_standard(list(sorted(entities_new)))

        # logger = Logger(f"./api_qwen/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}") # only_relationship_
        logger = Logger(f"./api_qwen/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}") # only_relationship_

        # 写json
        # logger_triplet = f"./logs/api_qwen/triple_score/triple_score_{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}.json" # only_relationship_
        # logger_triplet_delete = f"./logs/api_qwen/triple_score/triple_score_delete_{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}.json" # only_relationship_
        # logger_data = f"./logs/api_qwen/stage/data_{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}.json" # only_relationship_
        logger_triplet = f"./logs/api_qwen/triple_score/triple_score_{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}.json" # only_relationship_
        logger_triplet_delete = f"./logs/api_qwen/triple_score/triple_score_delete_{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}.json" # only_relationship_
        logger_data = f"./logs/api_qwen/stage/data_{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_latest_{current_time}_{iteration}.json" # only_relationship_

        
        for index in range(len(self.dataset.query)):  #2196
            dict_tmp = {}
            test_dict_tmp = {}
            # if index >= 1: 
            #     break
            # if index <= 256:
            #     continue
            query_number = query_number + 1

            print(f"-----------kg modify id : {index}----------------")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {self.dataset.query[index]}")
            logger.log(f"answer: {self.dataset.answer[index]}")
            dict_tmp['index'] = index
            dict_tmp['question'] = self.dataset.query[index]
            dict_tmp['answer'] = self.dataset.answer[index]
            

            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[index], 10) # 问题相似度选
            logger.log(f"keywords : {keywords}")
            dict_tmp['keywords'] = keywords

            # 补充关键词
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[index], keywords, 2)
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")            

            # 使用带symbol的path过滤，但是给到大模型的依旧是sentence
            # 还差一点，所示检索结果按照分数排序
            
            triple_result, filtered_retrieve_result, filtered_triple_3d, filtered_triple_3d_more, max_score, min_score, median_score, mean_score = self.retriver.retrieve_path_with_keywords_v2(self.dataset.query[index], triplets_score, keywords, path_depth = 2, pruning=10, pruning2=30, threshold=0.55, score_threshold = 71)
        
            # triple_result 是所有符合相似度阈值的路径 filtered_retrieve_result 是过滤后的4*10，用于回答问题 filtered_triple_3d 是留给冗余处理+feedback
            dict_tmp['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子
            dict_tmp['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
            dict_tmp['filtered_triple_3d_more'] = filtered_triple_3d_more # 带symbol  len = 4
            
            
            logger.log(f"len(triple_result) : {len(triple_result)}")
            # logger.log(f"max_score: {max_score}, min_score: {min_score}, median_score: {median_score}, mean_score: {mean_score}")
            logger.log(f"filtered_retrieve_result : {json.dumps(filtered_retrieve_result, indent=2)}")

            # response_one, num_input_tokens = self.graphrag.chat_without_stream_with_triplets_llama_api(self.dataset.query[index], filtered_retrieve_result)
            response_one, num_input_tokens, prompt = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[index], filtered_retrieve_result)

            self.response.append(response_one)
            dict_tmp['num_input_tokens'] = num_input_tokens
            num_input_tokens_avg.append(num_input_tokens)
            dict_tmp['response'] = response_one
            logger.log(f"num_input_tokens : {num_input_tokens}")
            logger.log(f"response : {response_one}")
            


            if filtered_triple_3d_more and len(filtered_triple_3d_more)> 1:
                redundant_relationship_3d, relationship_group_sentence = self.retriver.find_redundant_relationship_by_entity(filtered_triple_3d_more) # triple_result[:80]
                # redundant_relationship_3d = [] 
                # relationship_group_sentence = ["--------No relationship group!!!---------"]
                # redundant_entity_2d, all_entity_group_str = self.retriver.find_redundant_entity(triple_result[:80])
                redundant_entity_2d = []
                all_entity_group_str = "---------------No entity operation------------"
            else:
                redundant_relationship_3d = [] 
                relationship_group_sentence = ["--------No relationship group!!!---------"]
                redundant_entity_2d = [] 
                all_entity_group_str = "-------------No entity group!!!------------"
                logger.log(f"------------------filtered_retrieve_result is empty or length <= 1------------------")

            # logger.log(f"Redundant relationship groups:\n{json.dumps(relationship_group_sentence, indent=2)}") #输出关系所有分组中至少有两个关系的分组
            dict_tmp['redundant_relationship_3d'] = redundant_relationship_3d
            dict_tmp['relationship_group_sentence'] = relationship_group_sentence

            delete_list = []
            if redundant_relationship_3d:
                enable_thinking = False
                parsed_response_for_relationship, think_content_list = self.graphrag_fb.chat_without_stream_for_redundant_relationship_v3_api_qwen(redundant_relationship_3d, enable_thinking)   
                if parsed_response_for_relationship:# 正确解析
                    # relationship_modify_index.append(parsed_response_for_relationship)
                    # relationship_modify.append(redundant_relationship_3d)
                    # 冗余可以立即处理，我特别想把这个process_redundant_relationship_v2方法优化   
                    fliter_relationship_group_str = ""
                    index_tmp = 0
                    for i_0, group in enumerate(redundant_relationship_3d):
                        fliter_relationship_group_str += f"Group {i_0}:\n"
                        for triple in group:
                            fliter_relationship_group_str = fliter_relationship_group_str + str(index_tmp) + str(triple) + '\n'
                            index_tmp += 1
                    logger.log(f"-------Fliter relationship groups--------:\n{fliter_relationship_group_str}") #输出关系所有分组中至少有两个关系的分组
                    keep_relationship_str, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  
                    delete_relationship_num.append(len(delete_list))  
                    # delete_triple_by_relationship_2d_all.extend(delete_list) 
                    logger.log(f"keep relationship : {keep_relationship_str}")     # 保留的关系
                    logger.log(f"parsed_response_for_redundant_relationship: {parsed_response_for_relationship}")
                    dict_tmp['keep_relationship_str'] = keep_relationship_str
                    dict_tmp['parsed_response_for_relationship'] = parsed_response_for_relationship
                    dict_tmp['delete_list'] = delete_list

                    for triple in delete_list: # 即时修改, len == 3
                        with open(logger_triplet_delete, 'a', encoding='utf-8') as f:
                            f.write(f"{json.dumps(triple, ensure_ascii=False)}\n")  
                            # f.write(f"{triple}\n")  

                        sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                        value = triplets_score.pop(sentence, {})
                        if not value:
                            logger.log(f"{sentence}(delete by relationship) not exist in triple_score!")
                else:
                    logger.log(f"{index} response for redundant relationships parse error or llm think No redundant relationships")
                    print(f"{index} response for redundant relationships parse error")
            else:
                logger.log(f"{index} No redundant relationships")    
                print(f"{index} No redundant relationships")
            
            logger.log(f"graph rag response: {response_one}")
            # 大模型准确率计算
            check_list, answer_think_content = self.graphrag_fb.chat_without_stream_answer_check_api_qwen(self.dataset.query[index], response_one, self.dataset.answer[index]) 
            dict_tmp['check_list'] = check_list
            dict_tmp['answer_think_content'] = answer_think_content
            answer_num = -1 # 答案主体是否为数字
            answer_acc = -1
            if check_list: # 有结果
                answer_num, answer_acc = check_list
                if answer_num == -1: # 响应超过三次
                    logger.log(f"-------------Large model responds more than three times: answer_num")
                elif answer_num == 1:
                    logger.log(f"-------------Answer is numeric")
                elif answer_num == 0:
                    logger.log(f"-------------Answer is not numeric")
                else:
                    logger.log(f"-------------Unknown return value: answer_num")

                if answer_acc == -1:
                    logger.log(f"-------------Large model responds more than three times: answer_acc")
                elif answer_acc == 1:
                    acc_llm += 1
                    logger.log(f"-------------Correct")
                elif answer_acc == 0:
                    logger.log(f"-------------Incorrect")
                elif answer_acc == 2:
                    acc_llm_part += 1
                    logger.log(f"-------------Partially Correct")
                else:
                    logger.log(f"-------------Unknown return value: answer_acc")
            else: # 信息不足，无需反馈
                logger.log(f"-------------Insufficient information, no feedback require")
            # logger.log(f"answer_think_content:\n {answer_think_content}")
            dict_tmp['answer_num'] = answer_num
            dict_tmp['answer_acc'] = answer_acc

            flag_label = self.checkanswer(response_one, self.dataset.answer[index])
            flag_TF = sum(flag_label) == len(flag_label)
            if flag_TF:
                logger.log(f"-------------graph rag response true----------------------------------")
                acc += 1
            elif sum(flag_label):
                logger.log(f"-------------graph rag response part_true----------------------------------")
                acc_part += 1
            else:
                logger.log(f"-------------graph rag response false ~~~~~")

            dict_tmp['label'] = flag_label
            dict_tmp['flag'] = flag_TF
            
            # 反馈处理
            delete_dict ={}
            for delete_triple in delete_list:
                delete_dict[f"{delete_triple[0]} {delete_triple[1].replace('_',' ')} {delete_triple[2]}"] = delete_triple

            # 在这里把delete的路径去除掉
            one_hop_sentence_2d = []
            one_hop_sentence = []
            seen = set()
            sentence_index = 0
            for path_triple in filtered_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                    if sentence not in seen and sentence not in delete_dict:
                        seen.add(sentence)                       
                        one_hop_sentence_2d.append(triple)
                        one_hop_sentence.append(str(sentence_index)+': '+sentence)
                        sentence_index += 1

            dict_tmp['one_hop_sentence_2d'] = one_hop_sentence_2d      
            # logger.log(f"one_hop_sentence_2d:\n{json.dumps(one_hop_sentence, indent=2)}")  
            sorted_triples = self.sort_triples_by_entity(one_hop_sentence_2d) # len == 4
            sorted_triples_sentence = []

            for evidence_index, value in enumerate(sorted_triples):
                if value[3] == '->':
                    sorted_triples_sentence.append(f"Path Evidence {evidence_index}: {value[0]} {value[1].replace('_', ' ')} {value[2]}")
                elif value[3] == '<-':
                    sorted_triples_sentence.append(f"Path Evidence {evidence_index}: {value[2]} {value[1].replace('_', ' ')} {value[0]}")
            logger.log(f"sorted_triples_sentence:\n{json.dumps(sorted_triples_sentence, indent=2)}")  

            # 暂时存在一个问题，多跳路径出现相同内容的话是重复加分的！！！未必是问题
            # answer_acc == -1
            if answer_acc == -1: # 信息不足或者解析失败
                logger.log(f"Because llm check response: Insufficient information or parsing failure, no feedback required")  
            else:
                output, path_evidence_numbers_2d, contribution_numbers_2d, final_result_list, reasoning_think_content = self.graphrag_fb.chat_without_stream_for_socre_feedback_Reasoning_Path_final_api(self.dataset.query[index], response_one, sorted_triples, keywords)
                dict_tmp['feedback'] = output
                dict_tmp['path_evidence_numbers_2d'] = path_evidence_numbers_2d
                dict_tmp['contribution_numbers_2d'] = contribution_numbers_2d
                logger.log(f"-----------feedback_Reasoning_Path_final_api llm output--------------\n{output}") 
                logger.log(f"-----------feedback_Reasoning_Path_final_api start--------------") 
                if path_evidence_numbers_2d:
                    for reasoning_group_index, (path_evidence_numbers_group, contribution_numbers_group) in enumerate(zip(path_evidence_numbers_2d, contribution_numbers_2d)):
                        logger.log(f"{reasoning_group_index} Group Path start")  
                        for path_index, (path_evidence_numbers, contribution_numbers) in enumerate(zip(path_evidence_numbers_group, contribution_numbers_group)):
                            if path_evidence_numbers < len(sorted_triples) and path_evidence_numbers >= 0 and contribution_numbers in [1,2,3]:
                                if sorted_triples[path_evidence_numbers][3] == '->':
                                    triple = [sorted_triples[path_evidence_numbers][0], sorted_triples[path_evidence_numbers][1], sorted_triples[path_evidence_numbers][2]]
                                elif sorted_triples[path_evidence_numbers][3] == '<-':
                                    triple = [sorted_triples[path_evidence_numbers][2], sorted_triples[path_evidence_numbers][1], sorted_triples[path_evidence_numbers][0]]
                                logger.log(f"{path_index} path(score:{contribution_numbers}): {triple[0]} {triple[1].replace('_',' ')} {triple[2]}") 
                                sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                                if answer_num == 1 and contribution_numbers == 3: # 是数字，且贡献度为3
                                    if answer_acc == 1 or answer_acc == 2:
                                        if not sentence in triplets_score:
                                            logger.log(f"---------------{sentence}: the retrival result don't exit in triplets score &&&&&")
                                        else:
                                            if triplets_score[sentence]["score"] <= 175: # 最大不超过200
                                                logger.log(f"---------------{sentence}: {triplets_score[sentence]['score']} --{contribution_numbers}--> {triplets_score[sentence]['score']+contribution_numbers*5}") 
                                                triplets_score[sentence]["score"] += contribution_numbers*5
                                                with open(logger_triplet, 'a', encoding='utf-8') as f:
                                                    f.write(f"{json.dumps([triple[0], triple[1], triple[2], 5*contribution_numbers], ensure_ascii=False)}\n")  
                                            else:
                                                logger.log(f"~~~~~~~~~~~~~~~{sentence}: {triplets_score[sentence]['score']} Score reaches threshold") 
                                            relationship_score_modify += 1
                                    elif answer_acc == 0:
                                        if not sentence in triplets_score:
                                            logger.log(f"---------------{sentence}: the retrival result don't exit in triplets score &&&&&")
                                        else:
                                            if triplets_score[sentence]["score"] >= 20: # 应该不会出现这种情况，因为小于70就检索不到了
                                                logger.log(f"---------------{sentence}: {triplets_score[sentence]['score']} --{contribution_numbers}--> {triplets_score[sentence]['score']-contribution_numbers*5}") 
                                                triplets_score[sentence]["score"] -= contribution_numbers*5
                                                with open(logger_triplet, 'a', encoding='utf-8') as f:
                                                    f.write(f"{json.dumps([triple[0], triple[1], triple[2], -5*contribution_numbers], ensure_ascii=False)}\n")  
                                            else:
                                                logger.log(f"~~~~~~~~~~~~~~~{sentence}: {triplets_score[sentence]['score']} Score reaches threshold") 
                                            relationship_score_modify += 1
                                else: # 不是数字
                                    if answer_acc == 1 or answer_acc == 2:
                                        if not sentence in triplets_score:
                                            logger.log(f"---------------{sentence}: the retrival result don't exit in triplets score &&&&&")
                                        else:
                                            if triplets_score[sentence]["score"] <= 175:
                                                logger.log(f"---------------{sentence}: {triplets_score[sentence]['score']} --{contribution_numbers}--> {triplets_score[sentence]['score']+contribution_numbers*5}") 
                                                triplets_score[sentence]["score"] += contribution_numbers*5
                                                with open(logger_triplet, 'a', encoding='utf-8') as f:
                                                    f.write(f"{json.dumps([triple[0], triple[1], triple[2], 5*contribution_numbers], ensure_ascii=False)}\n")  
                                            else:
                                                logger.log(f"~~~~~~~~~~~~~~~{sentence}: {triplets_score[sentence]['score']} Score reaches threshold") 
                                            relationship_score_modify += 1
                                    elif answer_acc == 0  and (contribution_numbers == 3 or contribution_numbers == 2):# 错误的话也只对贡献度为3的进行处理，是否要对错误程度为2的也进行处理
                                        if not sentence in triplets_score:
                                            logger.log(f"---------------{sentence}: the retrival result don't exit in triplets score &&&&&")
                                        else:
                                            if triplets_score[sentence]["score"] >= 20:
                                                logger.log(f"---------------{sentence}: {triplets_score[sentence]['score']} --{contribution_numbers}--> {triplets_score[sentence]['score']-contribution_numbers*5}") 
                                                triplets_score[sentence]["score"] -= contribution_numbers*5
                                                with open(logger_triplet, 'a', encoding='utf-8') as f:
                                                    f.write(f"{json.dumps([triple[0], triple[1], triple[2], -5*contribution_numbers], ensure_ascii=False)}\n")  
                                            else:
                                                logger.log(f"~~~~~~~~~~~~~~~{sentence}: {triplets_score[sentence]['score']} Score reaches threshold") 
                                            relationship_score_modify += 1
                            else:
                                pass
                else:
                    logger.log(f"feedback_Reasoning_Path_final_api: Large model responds more than three times")  
            with open(logger_data, 'a', encoding='utf-8') as f:
                f.write(f"{json.dumps(dict_tmp, ensure_ascii=False)}\n")  
            forward_data.append(dict_tmp)
       
                # if answer_acc == 1: # 回答正确
                #     output, path_evidence_numbers_2d_error, contribution_numbers_2d_error, final_result_list_error, reasoning_think_content_error = self.graphrag_fb.chat_without_stream_for_socre_feedback_Reasoning_Path_error_test_api(self.dataset.query[index], response_one, sorted_triples, keywords)
                #     logger.log(f"-----------feedback_Reasoning_Path_error_test_api llm output--------------\n{output}") 
                #     logger.log(f"-----------feedback_Reasoning_Path_error_test_api start--------------") 
                #     if path_evidence_numbers_2d:
                #         for reasoning_group_index, (path_evidence_numbers_group, contribution_numbers_group, error_result) in enumerate(zip(path_evidence_numbers_2d_error, contribution_numbers_2d_error, final_result_list_error)):
                #             logger.log(f"{reasoning_group_index} Group Path start (error_result: {error_result})")  
                #             for path_index, (path_evidence_numbers, contribution_numbers) in enumerate(zip(path_evidence_numbers_group, contribution_numbers_group)):
                #                 if path_evidence_numbers < len(sorted_triples) and path_evidence_numbers > 0 and contribution_numbers in [1,2,3]:
                #                     if sorted_triples[path_evidence_numbers][3] == '->':
                #                         triple = [sorted_triples[path_evidence_numbers][0], sorted_triples[path_evidence_numbers][1], sorted_triples[path_evidence_numbers][2]]
                #                     elif sorted_triples[path_evidence_numbers][3] == '<-':
                #                         triple = [sorted_triples[path_evidence_numbers][2], sorted_triples[path_evidence_numbers][1], sorted_triples[path_evidence_numbers][0]]
                #                     logger.log(f"{path_index} path(score:{contribution_numbers}): {triple[0]} {triple[1].replace('_',' ')} {triple[2]}") 
                #                     sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                #                     if contribution_numbers == 3: # 无论是不是数字，都只对贡献度为3的错误进行减分
                #                         if not sentence in triplets_score:
                #                             logger.log(f"---------------{sentence}: the retrival result don't exit in triplets score &&&&&")
                #                         else:
                #                             logger.log(f"---------------{sentence}: {triplets_score[sentence]['score']} --{contribution_numbers}--> {triplets_score[sentence]['score']-contribution_numbers*5}") 
                #                             triplets_score[sentence]["score"] -= contribution_numbers*5
                #                             relationship_score_modify += 1
                #                 else:
                #                     pass
                #     elif final_result_list: # 无错误路径
                #         logger.log(f"feedback_Reasoning_Path_error_test_api: No contradictory reasoning path found.")  
                #     else:
                #         logger.log(f"feedback_Reasoning_Path_error_test_api: Large model responds more than three times")  

        response_list = []
        answer_list = []
        print(f"query_number: {query_number}")
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        squad_v2_metric = load("squad_v2")
        results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")
        
        # squad_metric = load("squad")
        # results = squad_metric.compute(predictions = response_list, references = answer_list)
        # logger.log(f"--------squad em f1 ...------------\n{json.dumps(results, indent=2)}")


        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy part: {acc_part/query_number}-----------------------")
        logger.log(f"\n\n-----------------------num_input_tokens_avg: {sum(num_input_tokens_avg)/len(num_input_tokens_avg)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm part: {acc_llm_part/query_number}-----------------------")

        logger.log(f"\n\n-----------------------delete_relationship_num: {sum(delete_relationship_num)}-----------------------")
        logger.log(f"\n\n-----------------------relationship_score_modify: {relationship_score_modify}-----------------------")


        end_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"./logs/api_qwen/stage/API_{self.args.option}_{self.args.dataset_name}_{end_time}_{iteration}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(forward_data, indent=2))
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration+1}.json", "w", encoding="utf-8") as file:
            json.dump(triplets_score, file, ensure_ascii=False, indent=4)

        self.retriver.entity_embedding_bak(iteration) # 实体嵌入备份

        # 根据triple_score重建entity_embedding
        # entities_new = set()
        # for value in triplets_score.values():
        #     if value['score'] > 70:
        #         entities_new.add(value['triplet'][0])
        #         entities_new.add(value['triplet'][2])
        # self.retriver.generate_new_entity_embedding_standard(list(sorted(entities_new)))

    def kg_modify_llama_reproduce_forword_case(self, iteration):
        # self.retriver.entity_embedding_bak(iteration) # 实体嵌入备份, 不需要修改实体，因为只处理了相同实体

        print(f"kg_modify_ignore iteration: {iteration}")
        # assert False

        # load score
        import json

        acc = 0
        acc_llm = 0
        acc_refuse = 0
        delete_relationship_num = 0
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        query_number = 0
        relationship_modify = []
        relationship_modify_index = []
        entity_modify = []
        entity_modify_index = []
        delete_entity_list = []
        insert_relationship_list_2d_all = []
        delete_triple_by_relationship_2d_all = []
        delete_triple_by_entity_2d_all = []
        triple_result_average = 0
        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []
        forward_data = []
        num_input_tokens_avg = []
        
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)

        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_data_{self.args.dataset_name}_{self.args.iteration}.json", 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # 这轮开始重新生成嵌入
        # entities_new = set()
        # for value in triplets_score.values():
        #     if value['score'] > 70:
        #         entities_new.add(value['triplet'][0])
        #         entities_new.add(value['triplet'][2])
        # self.retriver.generate_new_entity_embedding_standard(list(sorted(entities_new)))

        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{iteration}_{self.args.entity}*{self.args.pruning}_new") # only_relationship_

        # 相似问题
        # with open("/home/zhangyz/RAG_2025/dataset/rgb/similar_questions.json", 'r', encoding='utf-8') as f:
        #     self.dataset.query = json.load(f)[0]

            # print(f"{json.dumps(self.dataset.query, indent=2)}")
            # assert False

        
        for index in range(len(data_list)):  #2196
            # if index >= 2: 
            #     break
            # if index <= 256:
            #     continue
            dict_tmp = {}
            query_number = query_number + 1

            print(f"kg modify id : {index}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {data_list[index]['query']}")
            logger.log(f"answer: {data_list[index]['answer']}")
            dict_tmp['index'] = index
            dict_tmp['query'] = data_list[index]['query']
            dict_tmp['answer'] = data_list[index]['answer']

            keywords = data_list[index]['keywords']
            logger.log(f"keywords : {keywords}")
            dict_tmp['keywords'] = keywords
            # 补充关键词
            # existing_keywords = self.retriver.extract_keyword(self.dataset_gene.query[index], keywords, 2)
            # logger.log(f"existing_keywords : {existing_keywords}")
            # for keyword in existing_keywords:
            #     keyword_link = self.retriver.extract_keywords_with_embedding_find_entity(keyword, 1)
            #     logger.log(f"Link Keywords : {keyword} -> {keyword_link}") 
            #     for item in keyword_link:
            #         if item not in keywords:
            #             keywords.append(item)
            # logger.log(f"Final Keywords : {keywords}")            

            filtered_retrieve_result = [item for sublist in data_list[index]['filtered_retrieve_result_2d'][:self.args.entity] for item in sublist[:self.args.pruning]]
            filtered_triple_3d = [item for sublist in data_list[index]['filtered_triple_4d'][:self.args.entity] for item in sublist[:self.args.pruning]]
            filtered_triple_3d_more = [item for sublist in data_list[index]['filtered_triple_4d_more'][:self.args.entity] for item in sublist[:self.args.pruning]]

            dict_tmp['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子
            dict_tmp['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
            dict_tmp['filtered_triple_3d_more'] = filtered_triple_3d_more # 带symbol  len = 4

            logger.log(f"filtered_retrieve_result : {json.dumps(filtered_retrieve_result, indent=2)}")

            # response_one = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[index], filtered_retrieve_result)
            response_one, num_input_tokens, prompt = self.graphrag.chat_without_stream_with_triplets_llama_instruct(self.dataset.query[index], filtered_retrieve_result)
            num_input_tokens_avg.append(num_input_tokens)
            logger.log(f"num_input_tokens: {num_input_tokens}")
            # if self.llm.model_name == "deepseek-r1:70b":
            #     response_one = response_one.split('</think>', 1)[1].strip()
            self.response.append(response_one)
            dict_tmp['prompt'] = prompt
            dict_tmp['num_input_tokens'] = num_input_tokens
            dict_tmp['response'] = response_one
            
            # 对错判断
            logger.log(f"graph rag response of paths: {response_one}")
            # logger.log(f"graph rag response of triples: {response_ofT}")
            flag_label = self.checkanswer(response_one, self.dataset.answer[index])
            flag_TF = sum(flag_label) == len(flag_label)
            if flag_TF:
                # self.response.append(response_one)
                logger.log(f"-------------graph rag response true----------------------------------")
                acc = acc + 1
            else:
                logger.log(f"-------------graph rag response false ~~~~~")
                    
            dict_tmp['label'] = flag_TF
            forward_data.append(dict_tmp)

        # EM F1
        response_list = []
        answer_list = []
        print(f"query_number: {query_number}")
        for i in range(query_number):
            response_item = {}
            answer_item = {}
            response_item['id'] = str(i)
            response_item["prediction_text"] = self.response[i]
            response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
            response_list.append(response_item)
            reference_answer_item = {}
            reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
            reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
            answer_item['id'] = str(i)
            answer_item['answers'] = reference_answer_item
            answer_list.append(answer_item)
        squad_v2_metric = load("squad_v2")
        results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
        logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")
        self.response = [] # 每轮结束清空，不然分数不变，因为response累积
        
        # squad_metric = load("squad")
        # results = squad_metric.compute(predictions = response_list, references = answer_list)
        # logger.log(f"--------squad em f1 ...------------\n{json.dumps(results, indent=2)}")


        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------num_input_tokens_avg: {sum(num_input_tokens_avg)/len(num_input_tokens_avg)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")

        logger.log(f"\n\n-----------------------delete_relationship_num: {delete_relationship_num}-----------------------")
        logger.log(f"\n\n-----------------------delete_relationship_by_entity_num: {delete_relationship_by_entity_num}-----------------------")
        logger.log(f"\n\n-----------------------insert_relationship_by_entity_num: {insert_relationship_by_entity_num}-----------------------")
        logger.log(f"\n\n-----------------------delete_entity_num: {delete_entity_num}-----------------------")

        # with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration+1}.json", "w", encoding="utf-8") as file:
        #     json.dump(triplets_score, file, ensure_ascii=False, indent=4)
        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_{self.args.option}_{self.args.dataset_name}_{iteration}_{self.args.entity}_{self.args.pruning}_new.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(forward_data, indent=2))

    def kg_modify_llama_reproduce_forword_data(self, iteration):
        # self.retriver.entity_embedding_bak(iteration) # 实体嵌入备份, 不需要修改实体，因为只处理了相同实体

        print(f"kg_modify_ignore iteration: {iteration}")
        # assert False

        # load score
        import json

        acc = 0
        acc_llm = 0
        acc_refuse = 0
        delete_relationship_num = 0
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        query_number = 0
        relationship_modify = []
        relationship_modify_index = []
        entity_modify = []
        entity_modify_index = []
        delete_entity_list = []
        insert_relationship_list_2d_all = []
        delete_triple_by_relationship_2d_all = []
        delete_triple_by_entity_2d_all = []
        triple_result_average = 0
        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []
        forward_data = []
        num_input_tokens_avg = []
        
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.type}_{iteration}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)


        logger = Logger(f"{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_llama_reproduce_{iteration}") # only_relationship_

        logger_triplet = Logger(f"triple_score_{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_{self.args.type}_llama_reproduce_{iteration}") # only_relationship_

        
        for index in range(len(self.dataset.query)):  #2196
            # if index >= 2: 
            #     break
            # if index <= 256:
            #     continue
            dict_tmp = {}
            query_number = query_number + 1

            print(f"kg modify id : {index}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {self.dataset.query[index]}")
            logger.log(f"answer: {self.dataset.answer[index]}")
            dict_tmp['index'] = index
            dict_tmp['query'] = self.dataset.query[index]
            dict_tmp['answer'] = self.dataset.answer[index]

            keywords = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[index], self.args.entity) # 问题相似度选 # 4
            logger.log(f"keywords : {keywords}")
            dict_tmp['keywords'] = keywords
            # 有一个问题就是这个triple_score是在删边的，数据库中可能会查到一些没有的东西
            triple_result, filtered_retrieve_result_2d, filtered_triple_4d, filtered_triple_4d_more, max_score, min_score, median_score, mean_score = self.retriver.retrieve_path_with_keywords_v2_data(self.dataset.query[index], triplets_score, keywords, path_depth = 2, pruning=self.args.pruning, pruning2=35, threshold=0.55, score_threshold = 71) # 10*30
            # triple_result 是所有符合相似度阈值的路径， filtered_retrieve_result 是过滤后的4*10，用于回答问题 filtered_triple_3d 是留给冗余处理+feedback，带symbol，len=4

            # dict_tmp['triple_result'] = triple_result
            dict_tmp['filtered_retrieve_result_2d'] = filtered_retrieve_result_2d # 给大模型的句子
            dict_tmp['filtered_triple_4d'] = filtered_triple_4d # 带symbol len = 4
            dict_tmp['filtered_triple_4d_more'] = filtered_triple_4d_more # 带symbol  len = 4
            forward_data.append(dict_tmp)
            
        with open(f"./logs/stage/Meta-Llama-3-8B-Instruct_{self.args.option}_{self.args.dataset_name}_{iteration}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(forward_data, indent=2))

    def kg_modify_llama_reproduce_forword_verify(self, iteration):
        # 使用相似问题的训练后图谱进行验证
        pass

    def select_10_percent(self, n, rate, excluded=None):
        import random
        if rate == 0.0:
            return []
        if excluded is None:
            excluded = set()
        else:
            excluded = set(excluded)
        count = max(1, round(n * rate))
        available = list(set(range(n)) - excluded)

        if len(available) < count:
            raise ValueError(f"没有足够的可用数字进行选择。需要 {count} 个，但只有 {len(available)} 个可用。")

        result = random.sample(available, count)
        return sorted(result)

    def copy_triplets_file(self, iteration: int): # 暂时没用上
        import os
        import shutil
        """
        当 iteration == 0 时，复制 triplets 文件到 use_local_model 对应的目录，
        并自动创建目标目录（如果不存在）。
        """
        if iteration == 0:
            # 源路径
            src_file = f"./logs/triplets/{self.args.space_name}_{iteration}.json"
            
            # 目标路径
            dest_dir = f"./logs/triplets"
            dest_file = f"{dest_dir}/{self.args.space_name}_{self.args.option}_{self.args.llm_fb}_{iteration}.json"
            
            # 关键：自动创建目标目录（如果不存在）
            os.makedirs(dest_dir, exist_ok=True)
            print(f"确保目录存在: {dest_dir}")

            # 检查源文件是否存在
            if not os.path.exists(src_file):
                print(f"源文件不存在: {src_file}")
                return

            # 执行复制
            try:
                shutil.copy2(src_file, dest_file)
                print(f"成功复制文件:\n    {src_file} \n    → {dest_file}")
            except Exception as e:
                print(f"复制失败: {e}")

    def evolve_basic_forward(self):

        print(f"evolve_basic iteration: {self.args.iteration}")
        # assert False

        # load score

        acc = 0
        acc_list = []
        acc_list_end = [] # 原图2hop错3hop对，反馈图2hop对
        acc_llm = 0
        acc_refuse = 0
        delete_relationship_num = 0
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        query_number = 0
        triple_result_average = 0
        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []
        forward_data = []
        num_input_tokens_avg = []
        labels_list = []

        print(type(self.args.iteration))
        print(f"self.args.iteration: {self.args.iteration}")
        
        if self.args.iteration == 0:
            with open(f"./logs/triplets/{self.args.space_name}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
                triplets_score = json.load(file)
        else:
            # with open(f"./logs/triplets/{self.args.space_name}_{self.args.llm_fb_name}_{self.args.algorithm}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
            #     triplets_score = json.load(file)
            with open(f"./logs/triplets/{self.args.space_name}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
                triplets_score = json.load(file)

        if self.args.similar: # 这个部分先不动
            if self.args.dataset_name == 'rgb':
                with open("./dataset/rgb/similar_questions_api_best.json", 'r', encoding='utf-8') as f:
                    self.dataset.query = json.load(f)
            elif self.args.dataset_name == 'multihop':
                with open("./dataset/multihop/dataset/similar_questions_api_best.json", 'r', encoding='utf-8') as f:
                    self.dataset.query = json.load(f)
            elif self.args.dataset_name == 'hotpotqa600':
                with open("./dataset/hotpotqa_graph/similar_questions_api_best.json", 'r', encoding='utf-8') as f:
                    self.dataset.query = json.load(f)

        if self.args.shouldRebuildDatabase:
            # 这轮开始重新生成嵌入
            entities_new = set()
            for value in triplets_score.values():
                if value['score'] > self.args.scoreThreshold:
                    entities_new.add(value['triplet'][0])
                    entities_new.add(value['triplet'][2])
            self.retriver.generate_new_entity_embedding_standard(list(sorted(entities_new)))
        
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.args.similar:
            logger = Logger(f"./data/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_similar_{self.args.iteration}")
        else:
            logger = Logger(f"./data/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_{self.args.iteration}")
        
        self.logger = logger
        self.retriver.triplets_score = triplets_score # 传递给retriver
        
        data_list = []
        data_list_end = []
        if self.args.dataset_name == 'rgb':
            # with open(f"/home/zhangyz/RAG_2025/logs/rgb/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_rgb_0.json", 'r', encoding='utf-8') as file:
            #     data_list = json.load(file)
            # with open(f"/home/zhangyz/RAG_2025/logs/rgb/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_rgb_5.json", 'r', encoding='utf-8') as file:
            #     data_list_end = json.load(file)    
            pass
        elif self.args.dataset_name == 'hotpotqa600':
            # with open(f"/home/zhangyz/RAG_2025/logs/hotpotqa600/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_hotpotqa600_0.json", 'r', encoding='utf-8') as file:
            #     data_list = json.load(file)
            # with open(f"/home/zhangyz/RAG_2025/logs/hotpotqa600/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_hotpotqa600_6.json", 'r', encoding='utf-8') as file:
            #     data_list_end = json.load(file)
            pass
        else:
            # data_list = []
            # data_list_end = []
            pass
        # with open(f"/home/zhangyz/RAG_2025/logs/multihop/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_multihop_0.json", 'r', encoding='utf-8') as file:
        #     data_list = json.load(file) 

        forward_total_time = []
        keywords_total_time = []
        nebula_retrieve_total_time = []
        filter_retrieve_total_time = []
        response_total_time = []
        response_2F3T = []

        forward_generate_time_total = []
        forward_prefill_time_total = []
        forward_decode_time_total = []
        forward_wait_scheduled_time_total = []
        forward_end2end_time_total = []
        prompt = []
        generate = []

        for index in range(len(self.dataset.query)):
            if index >= 1: 
                break
            # if index <= 165:
            #     continue
            if data_list and data_list[index]['label']:
                continue
            # if len(acc_list) > 0:
            #     break
            forward_start_time = time.perf_counter()
            
            dict_tmp = {}
            query_number = query_number + 1

            print(f"evolve basic id : {index}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {self.dataset.query[index]}")
            
            dict_tmp['index'] = index
            dict_tmp['query'] = self.dataset.query[index]
            dict_tmp['answer'] = self.dataset.answer[index]

            keywords, keyword_time = self.retriver.extract_keywords_with_embedding_find_entity(self.dataset.query[index], self.args.entity) # 问题相似度选 10
            keywords_total_time.append(keyword_time)
            logger.log(f"keywords : {keywords}")
            dict_tmp['keywords'] = keywords
            logger.log(f"keywords extraction time: {keyword_time} seconds")     


            # triple_result, filtered_retrieve_result, filtered_triple_3d, filtered_triple_3d_more, max_score, min_score, median_score, mean_score = self.retriver.retrieve_path_with_keywords_v2(self.dataset.query[index], triplets_score, keywords, path_depth = self.args.hop, pruning=self.args.pruning, pruning2=self.args.pruning, threshold=0.55, score_threshold = 71) 
            if self.args.retrieve_path == 'basic':
                filtered_retrieve_result, filtered_triple_3d, nebula_retrieve_time, filter_retrieve_time  = self.retrieve_path_with_keywords_module(method = self.args.retrieve_path, question=self.dataset.query[index], keywords=keywords, path_depth = self.args.hop, pruning=self.args.pruning, threshold=self.args.simThreshold, score_threshold = self.args.scoreThreshold, score_weight = self.args.score_weight, top_k_per_entity=self.args.top_k_per_entity) # 10*30

                dict_tmp['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子，带symbol
                dict_tmp['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
                # 一些用于分析的指标
                # logger.log(f"len(triple_result) : {len(triple_result)}")
                # logger.log(f"max_score: {max_score}, min_score: {min_score}, median_score: {median_score}, mean_score: {mean_score}")
                # triple_result_average += len(triple_result)
                logger.log(f"filtered_retrieve_result : {json.dumps(filtered_retrieve_result, indent=2)}")
            elif self.args.retrieve_path == 'standard':
                filtered_retrieve_result, filtered_triple_3d, filtered_probs, filtered_dict, pt_list_standard, sim_map, score_map, nebula_retrieve_time, filter_retrieve_time = self.retrieve_path_with_keywords_module(method = self.args.retrieve_path, question=self.dataset.query[index], keywords=keywords, path_depth = self.args.hop, pruning=self.args.pruning, threshold=self.args.simThreshold, score_threshold = self.args.scoreThreshold, score_weight = self.args.score_weight, top_k_per_entity=self.args.top_k_per_entity) # 10*30
                dict_tmp['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子，带symbol
                dict_tmp['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
                dict_tmp['filtered_probs'] = filtered_probs
                dict_tmp['filtered_dict'] = filtered_dict
                dict_tmp['pt_list_standard'] = pt_list_standard
                dict_tmp['sim_map'] = sim_map
                dict_tmp['score_map'] = score_map
                logger.log(f"filtered_retrieve_result : {json.dumps(filtered_retrieve_result, indent=2)}")
                # logger.log(f"filtered_probs : {json.dumps(filtered_probs, indent=2)}")
                # logger.log(f"filtered_dict : {json.dumps(filtered_dict, indent=2)}")
                # logger.log(f"pt_list_standard : {json.dumps(pt_list_standard, indent=2)}")
            nebula_retrieve_total_time.append(nebula_retrieve_time)
            filter_retrieve_total_time.append(filter_retrieve_time)
            logger.log(f"nebula retrieve time: {nebula_retrieve_time} seconds")
            logger.log(f"filter retrieve time: {filter_retrieve_time} seconds")

            if self.args.graphrag_response == "basic":
                response_one, num_input_tokens, response_time = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[index], filtered_retrieve_result)
            elif self.args.graphrag_response == "basic_shared_prefix" :
                response_one, num_input_tokens, generate_len, response_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self.graphrag.chat_without_stream_with_triplets_shared_prefix(self.dataset.query[index], filtered_retrieve_result)
                generate.append(generate_len)
                forward_generate_time_total.append(generate_time)
                forward_prefill_time_total.append(prefill_time)
                forward_decode_time_total.append(decode_time)
                forward_wait_scheduled_time_total.append(wait_scheduled_time)
                logger.log(f"generate_len: {generate_len}")
                logger.log(f"generate_time: {generate_time} seconds")
                logger.log(f"prefill_time: {prefill_time} seconds")
                logger.log(f"decode_time: {decode_time} seconds")
                logger.log(f"wait_scheduled_time: {wait_scheduled_time} seconds")

            response_total_time.append(response_time)
            num_input_tokens_avg.append(num_input_tokens)
            logger.log(f"response time: {response_time} seconds")
            logger.log(f"num_input_tokens: {num_input_tokens}")
            dict_tmp['num_input_tokens'] = num_input_tokens

            
            self.response.append(response_one)
            dict_tmp['response'] = response_one
            
            # 对错判断
            logger.log(f"answer: {self.dataset.answer[index]}")
            logger.log(f"graph rag response of paths: {response_one}")
            
            if self.args.dataset_name == 'dragonball':
                flag_label = self.checkanswer_rougel(response_one, self.dataset.answer[index])
                dict_tmp['label'] = flag_label
                labels_list.append(flag_label)
                logger.log(f"checkanswer rougel: {flag_label}")
            else:
                flag_label = self.checkanswer(response_one, self.dataset.answer[index])
                flag_TF = sum(flag_label) == len(flag_label)
                if flag_TF:
                    response_2F3T.append(index)
                    # self.response.append(response_one)
                    logger.log(f"-------------graph rag response true----------------------------------")
                    acc = acc + 1
                    acc_list.append(1)
                    if data_list_end and data_list_end[index]['label']:
                        acc_list_end.append(index)
                else:
                    logger.log(f"-------------graph rag response false ~~~~~")
                    acc_list.append(0)
                        
                dict_tmp['label'] = flag_TF
                forward_data.append(dict_tmp)
            forward_end_time = time.perf_counter()
            forward_total_time.append(forward_end_time - forward_start_time)
            logger.log(f"forward time: {forward_end_time - forward_start_time} seconds")
            # assert False

        if self.args.dataset_name == 'dragonball':
            accuracy = self.get_accuracy_rougel(labels_list)
            logger.log(f"----------accuracy rougel ...------------\n{accuracy}")
        elif self.args.dataset_name == 'metaqa':
            # 分 hop1 200 hop2 100 hop3 100
            # hop1 300 hop2 300
            
            print(f"query_number: {query_number}")
            for range_start, range_end in [[0,300],[300,600]]:
                response_list = []
                answer_list = []
                range_acc = []
                for i in range(range_start, min(query_number, range_end)):
                    response_item = {}
                    answer_item = {}
                    response_item['id'] = str(i)
                    response_item["prediction_text"] = self.response[i]
                    response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
                    response_list.append(response_item)
                    reference_answer_item = {}
                    reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
                    reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                    answer_item['id'] = str(i)
                    answer_item['answers'] = reference_answer_item
                    answer_list.append(answer_item)
                    range_acc.append(acc_list[i])
                squad_v2_metric = load("squad_v2")
                if response_list and answer_list:
                    results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
                    logger.log(f"--------squad_v2 em f1 ... range[{range_start},{range_end}]------------\n{json.dumps(results_v2, indent=2)}")
                # if range_acc:
                    logger.log(f"--------accuracy range[{range_start},{range_end}]: {sum(range_acc)/len(range_acc)}-----------------------")
        else:
            # EM F1
            response_list = []
            answer_list = []
            print(f"query_number: {query_number}")
            for i in range(query_number):
                response_item = {}
                answer_item = {}
                response_item['id'] = str(i)
                response_item["prediction_text"] = self.response[i]
                response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
                response_list.append(response_item)
                reference_answer_item = {}
                reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
                reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                answer_item['id'] = str(i)
                answer_item['answers'] = reference_answer_item
                answer_list.append(answer_item)
            squad_v2_metric = load("squad_v2")
            results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
            logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")


        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------num_input_tokens_avg: {sum(num_input_tokens_avg)/len(num_input_tokens_avg)}-----------------------")
        if generate:
            logger.log(f"\n\n-----------------------generate tokens avg: {sum(generate)/len(generate)}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")
        logger.log(f"\n\n-----------------------Probability of rejection: {acc_refuse/query_number}-----------------------")

        logger.log(f"\n\n-----------------------forward_total_time: {sum(forward_total_time)/len(forward_total_time)}-----------------------")
        
        logger.log(f"\n\n-----------------------keywords_total_time: {sum(keywords_total_time)/len(keywords_total_time)}-----------------------")
        logger.log(f"\n\n-----------------------nebula retrieve_total_time: {sum(nebula_retrieve_total_time)/len(nebula_retrieve_total_time)}-----------------------")
        logger.log(f"\n\n-----------------------filter retrieve_total_time: {sum(filter_retrieve_total_time)/len(filter_retrieve_total_time)}-----------------------")
        logger.log(f"\n\n-----------------------response_total_time: {sum(response_total_time)/len(response_total_time)}-----------------------")
        if forward_generate_time_total:
            logger.log(f"\n\n-----------------------forward_generate_time_total: {sum(forward_generate_time_total)/len(forward_generate_time_total)}-----------------------")
            logger.log(f"\n\n-----------------------forward_prefill_time_total: {sum(forward_prefill_time_total)/len(forward_prefill_time_total)}-----------------------")
            logger.log(f"\n\n-----------------------forward_decode_time_total: {sum(forward_decode_time_total)/len(forward_decode_time_total)}-----------------------")
            logger.log(f"\n\n-----------------------forward_wait_scheduled_time_total: {sum(forward_wait_scheduled_time_total)/len(forward_wait_scheduled_time_total)}-----------------------")


        logger.log(f"\n\n-----------------------acc_list_end: {len(acc_list_end)} --- {acc_list_end} -----------------------")
        
        # logger.log(f"\n\n-----------------------response_2F3T: {response_2F3T}-----------------------")
        


        # logger.log(f"\n\n-----------------------delete_relationship_num: {delete_relationship_num}-----------------------")
        # logger.log(f"\n\n-----------------------delete_relationship_by_entity_num: {delete_relationship_by_entity_num}-----------------------")
        # logger.log(f"\n\n-----------------------insert_relationship_by_entity_num: {insert_relationship_by_entity_num}-----------------------")
        # logger.log(f"\n\n-----------------------delete_entity_num: {delete_entity_num}-----------------------")

        if self.args.similar:
            with open(f"./logs/stage/{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_similar_{self.args.iteration}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(forward_data, indent=2))
        else:
            with open(f"./logs/stage/{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_{self.args.iteration}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(forward_data, indent=2))

    def retrieve_path_with_keywords_module(self, method, question, keywords = [], path_depth = 2, pruning = 10, threshold=0.55, score_threshold = 71, score_weight = 0.5, top_k_per_entity = True, build_node = False):
        if method == 'basic':
            return self.retriver.retrieve_path_with_keywords_basic(question, keywords, path_depth, pruning, threshold, score_threshold, build_node)
        elif method == 'standard':
            return self.retriver.retrieve_path_with_keywords_standard(question, keywords, path_depth, pruning, score_weight, top_k_per_entity, build_node)

    def redundant_process(self, method, filtered_triple_3d, index):
        redundancy_start_time = time.perf_counter()
        if method == 'basic_only_relationship_strict':
            if filtered_triple_3d and len(filtered_triple_3d)> 1:
                    redundant_relationship_3d, relationship_group_sentence = self.retriver.find_redundant_relationship_by_entity(filtered_triple_3d)
                    redundant_entity_2d = []
                    all_entity_group_str = "---------------No entity operation------------"
            else:
                redundant_relationship_3d = [] 
                relationship_group_sentence = ["--------No relationship group!!!---------"]
                redundant_entity_2d = [] 
                all_entity_group_str = "-------------No entity group!!!------------"
                self.logger.log(f"------------------filtered_retrieve_result is empty or length <= 1------------------")

            self.logger.log(f"Redundant relationship groups:\n{json.dumps(relationship_group_sentence, indent=2)}") #输出关系所有分组中至少有两个关系的分组
            if redundant_relationship_3d:
                parsed_response_for_relationship = self.graphrag.chat_without_stream_for_redundant_relationship_strict(redundant_relationship_3d)   
                if parsed_response_for_relationship:# 正确解析
                    # relationship_modify_index.append(parsed_response_for_relationship)
                    # relationship_modify.append(redundant_relationship_3d)             
                    keep_list, delete_list = self.retriver.process_redundant_relationship_v2(parsed_response_for_relationship, redundant_relationship_3d, False)  
                    # delete_relationship_num += len(delete_list)  
                    # delete_triple_by_relationship_2d_all.extend(delete_list) 
                    keep_relationship_str = [f"{triple[0]} {triple[1]} {triple[2]}" for triple in keep_list]
                    delete_relationship_list = [f"{triple[0]} {triple[1]} {triple[2]}" for triple in delete_list]
                    self.logger.log(f"parsed_response_for_redundant_relationship: {json.dumps(parsed_response_for_relationship, indent=2)}")
                    # self.logger.log(f"keep_relationship_str:\n {keep_relationship_str}")
                    # self.logger.log(f"delete_relationship_str:\n {delete_relationship_str}")
                    self.logger.log(f"delete_relationship_list:\n {json.dumps(delete_relationship_list, indent=2)}")
                    
                    for triple in delete_list: # 即时修改
                        sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                        value = self.triplets_score.pop(sentence, {})
                        if not value:
                            self.logger.log(f"{sentence}(delete by relationship) not exist in triple_score!")
                else:
                    self.logger.log(f"{index} response for redundant relationships parse error or llm think No redundant relationships")
                    # print(f"{index} response for redundant relationships parse error")
            else:
                self.logger.log(f"{index} No redundant relationships")    
                # print(f"{index} No redundant relationships")
        redundancy_end_time = time.perf_counter()

        return  len(delete_list), redundancy_end_time - redundancy_start_time

    # 不用因该这么传递参数，应该通过一个类传递，不然太乱
    # def feedback_process(self, method, filtered_retrieve_result, filtered_triple_3d, flag_TF, index, feedback_noise, question, response, filtered_probs, filtered_dict, pt_list_standard, score_weight, lr, sim_map, score_map):

    def feedback_process(self, method, filtered_retrieve_result, filtered_triple_3d, flag_TF, index, feedback_noise, question, response, filtered_probs, filtered_dict, pt_list_standard, args_param, sim_map, score_map):
        # feedback_process_start_time = time.perf_counter()
        num_input_tokens, generate_len, response_time, generate_time, prefill_time, decode_time, wait_scheduled_time = 0, 0, 0, 0, 0, 0, 0
        if method == 'basic_for_triplet':
            score_weight = args_param.score_weight
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen = set()
            sentence_id = 0
            for path_triple in filtered_triple_3d:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                    if sentence not in seen:
                        seen.add(sentence)
                        one_hop_sentence_2d.append(triple)
                        one_hop_sentence_list.append(sentence)
                        one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        sentence_id += 1
                    
            self.logger.log(f"one_hop_sentence:\n{one_hop_sentence_str}")  
            if index in feedback_noise:
                flag_TF = not flag_TF
                self.logger.log(f"----------------feedback_noise-------------: original: {not flag_TF}, now: {flag_TF}")  
            llm_feedback_start_time = time.perf_counter()
            response_insufficient, response_correct, response_error = self.graphrag.chat_without_stream_for_socre_feedback_basic(question, response, one_hop_sentence_list, flag_TF)
            llm_feedback_end_time = time.perf_counter()

            feedback_process_start_time = time.perf_counter()
            self.logger.log(f"response_for_score : {response_insufficient}")
            self.logger.log(f"correct_numbers : {response_correct}")
            self.logger.log(f"error_numbers : {response_error}")
            self.logger.log(f"Score: ")
            if not (response_insufficient or response_correct or response_error) :
                self.logger.log(f"{index} Exceeded the corresponding number of times, all feedback is empty")
                print(f"{index} Exceeded the corresponding number of times, all feedback is empty")
            elif response_correct or response_error :
                for i in range(len(one_hop_sentence_2d)):
                    if i in response_correct:
                        if one_hop_sentence_2d[i][3] == '->':                           
                            sentence = f"{one_hop_sentence_2d[i][0]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][2]}"
                        elif one_hop_sentence_2d[i][3] == '<-':                           
                            sentence = f"{one_hop_sentence_2d[i][2]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][0]}"
                        if not sentence in self.triplets_score:
                            self.logger.log(f" {i}: {one_hop_sentence_2d[i]}, the retrival result (good triple) don't exit in triplets score &&&&&") # 如果被误删了就再加进去？？？暂时不加
                            continue
                        else:
                            if response_correct[i] < 4 and response_correct[i] >0:
                                a = self.triplets_score[sentence]["score"] + response_correct[i] * 5
                                if a <= self.args.score_max:
                                    self.triplets_score[sentence]["score"] = a
                                self.logger.log(f" {i}: {one_hop_sentence_2d[i]}, correct triple new score : {a}")
                                # self.logger_triplet.log(f"[{sentence}] : {response_correct[i] * 5}") 
                            else:
                                self.logger.log(f" {i}: {one_hop_sentence_2d[i]}, the correct triplets score error &&&&&")
                    elif i in response_error:
                        if one_hop_sentence_2d[i][3] == '->':                           
                            sentence = f"{one_hop_sentence_2d[i][0]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][2]}"
                        elif one_hop_sentence_2d[i][3] == '<-':                           
                            sentence = f"{one_hop_sentence_2d[i][2]} {one_hop_sentence_2d[i][1]} {one_hop_sentence_2d[i][0]}"
                        if not sentence in self.triplets_score:
                            self.logger.log(f" {i}: {one_hop_sentence_2d[i]}, the retrival result (bad triple) don't exit in triplets score &&&&&") # error不存在就不存在了
                            continue
                        else:
                            if response_error[i] < 4 and response_error[i] > 0 :
                                a = self.triplets_score[sentence]["score"] - response_error[i] * 5
                                if a >= self.args.score_min: # 这种情况一般不会发生，因为小于70就检索不到了
                                    self.triplets_score[sentence]["score"] = a
                                self.logger.log(f" {i}: {one_hop_sentence_2d[i]}, error triple new score : {a}")
                                # self.logger_triplet.log(f"[{sentence}] : -{response_error[i] * 5}")
                            else:
                                self.logger.log(f" {i}: {one_hop_sentence_2d[i]}, the error triplets score error &&&&&") 

        elif method == 'standard_for_path':
            score_weight = args_param.score_weight
            lr = args_param.lr
            import math
            context = ""
            for idx, sentence in enumerate(filtered_retrieve_result, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            llm_feedback_start_time = time.perf_counter()
            response_dict = self.graphrag.chat_without_stream_for_socre_feedback_standard(question, response, filtered_retrieve_result, flag_TF)
            llm_feedback_end_time = time.perf_counter()

            feedback_process_start_time = time.perf_counter()
            self.logger.log(f"{context}")
            self.logger.log(f"{json.dumps(response_dict, indent=2)}")
            # assert False
            # print(type(response_dict))
            # print(response_dict)
            if response_dict['Insufficient_information']:
                pass
            else:
                if not flag_TF: # 根据对错反转反馈
                    new_dict = {key: -value for key, value in response_dict['Path_score'].items()}
                    response_dict['Path_score'] = new_dict
                # 计算期望
                EUL = 0
                AVGU = 0
                VLlist = {}
                # 所有都在计算，只不过有大量的反馈是0
                for path_id, (path, pl) in enumerate(zip(filtered_retrieve_result, filtered_probs)):
                    EUL += (response_dict['Path_score'].get(f"Path {path_id}", 0) + 1) / 2 * pl
                    AVGU += response_dict['Path_score'].get(f"Path {path_id}", 0) * pl
                for path_id, (path, pl) in enumerate(zip(filtered_retrieve_result, filtered_probs)):
                    # if f"Path {path_id}" in response_dict['Path_score']:
                    #     VLlist[f"Path {path_id}"] = pl * (response_dict['Path_score'].get(f"Path {path_id}") - AVGU)
                    # else:
                    #     VLlist[f"Path {path_id}"] = 0
                    VLlist[f"Path {path_id}"] = pl * (response_dict['Path_score'].get(f"Path {path_id}", 0) - AVGU)
                
                self.logger.log(f"Expected Utility of the LLM (EUL): {EUL}")
                self.logger.log(f"Average Utility of the LLM (AVGU): {AVGU}")
                # self.logger.log(f"Variance of the LLM (VL): {json.dumps(VLlist, indent=2)}")

                # 取负对数
                LEUL = -math.log(EUL) 
                triple_map = {}
                for idx, path in enumerate(filtered_triple_3d):
                    for triple in path:
                        if triple[3] == '->':
                            sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                        elif triple[3] == '<-':
                            sentence = f"{triple[2]} {triple[1]} {triple[0]}"
                        if sentence in triple_map:
                            triple_map[sentence].append(idx)
                        else:
                            triple_map[sentence] = [idx]
                grad_weight = 0
                for path_id, (path, pl) in enumerate(zip(filtered_triple_3d, filtered_probs)):
                    if f"Path {path_id}" in response_dict['Path_score']:
                        for triple in path:
                            if triple[3] == '->':
                                sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                                sentence_s = f"{triple[0]} {triple[1].replace('_',' ')} {triple[2]}"
                            elif triple[3] == '<-':
                                sentence = f"{triple[2]} {triple[1]} {triple[0]}"
                                sentence_s = f"{triple[2]} {triple[1].replace('_',' ')} {triple[0]}"
                            plset = 0
                            for path_index in triple_map.get(sentence, []): # 拿到包含t的所有L
                                # if f"Path {path_index}" in response_dict['Path_score']: # 反馈了才计算
                                #     if flag_TF:
                                #         plset *= filtered_probs[path_index] * VLlist.get(f"Path {path_index}", 0) # 不知道乘上Vi对不对
                                #     else:
                                #         plset *= filtered_probs[path_index] * -VLlist.get(f"Path {path_index}", 0) # 不知道乘上Vi对不对
                                plset += filtered_probs[path_index] * VLlist.get(f"Path {path_index}") 
                            grad_t = - (score_weight / (2 * EUL)) * (plset / pt_list_standard[filtered_dict[sentence]])
                            grad_weight_t = ((sim_map.get(sentence_s) - score_map.get(sentence_s)) / (2 * EUL)) * (plset / pt_list_standard[filtered_dict[sentence]])
                            grad_weight += grad_weight_t
                            # self.logger.log(f"sentence_s: {sentence_s}, sim: {sim_map.get(sentence_s, 0)}, score: {score_map.get(sentence_s, 0)}, grad_weight_t: {grad_weight_t}, grand_weight: {grad_weight}")
                            
                            if sentence in self.triplets_score:
                                old_score = self.triplets_score[sentence]['score']
                                # if flag_TF:
                                #     isNegative = 1
                                # else:
                                #     isNegative = -1
                                new_score = old_score - grad_t * lr
                                if new_score > self.args.score_max:
                                    new_score = self.args.score_max
                                if new_score < self.args.score_min:
                                    new_score = self.args.score_min
                                self.triplets_score[sentence]['score'] = new_score
                                self.logger.log(f" {sentence}, old score: {old_score}, grad: {grad_t}, new score : {new_score}")
                            else:
                                self.logger.log(f" {sentence}, the retrival result don't exit in triplets score &&&&&")

                # 处理学习率
                self.logger.log(f"old score_weight : {score_weight}, grad_weight : {grad_weight}, lr: {lr}")
                score_weight = score_weight - grad_weight * lr * 1e-3
                self.logger.log(f"new score_weight (before clamp): {score_weight}")
                if score_weight < 0.1:
                    score_weight = 0.1
                if score_weight > 0.9:    
                    score_weight = 0.9
                self.logger.log(f"new score_weight : {score_weight}")

        elif method == 'standard_for_path_shared_prefix':
            score_weight = args_param.score_weight
            lr = args_param.lr
            import math
            context = ""
            for idx, sentence in enumerate(filtered_retrieve_result, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            
            llm_feedback_start_time = time.perf_counter()
            response_dict, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self.graphrag.chat_without_stream_for_socre_feedback_standard_shared_prefix(question, response, filtered_retrieve_result, flag_TF)
            llm_feedback_end_time = time.perf_counter()

            feedback_process_start_time = time.perf_counter()
            self.logger.log(f"{context}")
            self.logger.log(f"{json.dumps(response_dict, indent=2)}")
            # assert False
            # print(type(response_dict))
            # print(response_dict)
            if response_dict['Insufficient_information']:
                pass
            else:
                if not flag_TF: # 根据对错反转反馈
                    new_dict = {key: -value * 0.1 for key, value in response_dict['Path_score'].items()}
                    response_dict['Path_score'] = new_dict
                else:
                    new_dict = {key: value * 0.1 for key, value in response_dict['Path_score'].items()}
                    response_dict['Path_score'] = new_dict
                # print(f"{json.dumps(response_dict, indent=2)}")
                # 计算期望
                EUL = 0
                AVGU = 0
                VLlist = {}
                # 所有都在计算，只不过有大量的反馈是0
                for path_id, (path, pl) in enumerate(zip(filtered_retrieve_result, filtered_probs)):
                    EUL += (response_dict['Path_score'].get(f"Path {path_id}", 0) + 1) / 2 * pl
                    # print(f"{response_dict['Path_score'].get(f'Path {path_id}', 0)} {(response_dict['Path_score'].get(f'Path {path_id}', 0) + 1) / 2 * pl}")
                    AVGU += response_dict['Path_score'].get(f"Path {path_id}", 0) * pl
                for path_id, (path, pl) in enumerate(zip(filtered_retrieve_result, filtered_probs)):
                    # if f"Path {path_id}" in response_dict['Path_score']:
                    #     VLlist[f"Path {path_id}"] = pl * (response_dict['Path_score'].get(f"Path {path_id}") - AVGU)
                    # else:
                    #     VLlist[f"Path {path_id}"] = 0
                    VLlist[f"Path {path_id}"] = pl * (response_dict['Path_score'].get(f"Path {path_id}", 0) - AVGU)
                
                self.logger.log(f"Expected Utility of the LLM (EUL): {EUL}")
                self.logger.log(f"Average Utility of the LLM (AVGU): {AVGU}")
                # self.logger.log(f"Variance of the LLM (VL): {json.dumps(VLlist, indent=2)}")
        
                # 取负对数
                LEUL = -math.log(EUL) 
                triple_map = {}
                for idx, path in enumerate(filtered_triple_3d):
                    for triple in path:
                        if triple[3] == '->':
                            sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                        elif triple[3] == '<-':
                            sentence = f"{triple[2]} {triple[1]} {triple[0]}"
                        if sentence in triple_map:
                            triple_map[sentence].append(idx)
                        else:
                            triple_map[sentence] = [idx]
                grad_weight = 0
                for path_id, (path, pl) in enumerate(zip(filtered_triple_3d, filtered_probs)):
                    if f"Path {path_id}" in response_dict['Path_score']:
                        for triple in path:
                            if triple[3] == '->':
                                sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                                sentence_s = f"{triple[0]} {triple[1].replace('_',' ')} {triple[2]}"
                            elif triple[3] == '<-':
                                sentence = f"{triple[2]} {triple[1]} {triple[0]}"
                                sentence_s = f"{triple[2]} {triple[1].replace('_',' ')} {triple[0]}"
                            plset = 0
                            for path_index in triple_map.get(sentence, []): # 拿到包含t的所有L
                                # if f"Path {path_index}" in response_dict['Path_score']: # 反馈了才计算
                                #     if flag_TF:
                                #         plset *= filtered_probs[path_index] * VLlist.get(f"Path {path_index}", 0) # 不知道乘上Vi对不对
                                #     else:
                                #         plset *= filtered_probs[path_index] * -VLlist.get(f"Path {path_index}", 0) # 不知道乘上Vi对不对
                                plset += filtered_probs[path_index] * VLlist.get(f"Path {path_index}") 
                            grad_t = - (score_weight / (2 * EUL)) * (plset / pt_list_standard[filtered_dict[sentence]])
                            grad_weight_t = ((sim_map.get(sentence_s) - score_map.get(sentence_s)) / (2 * EUL)) * (plset / pt_list_standard[filtered_dict[sentence]])
                            grad_weight += grad_weight_t
                            # self.logger.log(f"sentence_s: {sentence_s}, sim: {sim_map.get(sentence_s, 0)}, score: {score_map.get(sentence_s, 0)}, grad_weight_t: {grad_weight_t}, grand_weight: {grad_weight}")
                            
                            if sentence in self.triplets_score:
                                old_score = self.triplets_score[sentence]['score']
                                # if flag_TF:
                                #     isNegative = 1
                                # else:
                                #     isNegative = -1
                                new_score = old_score - grad_t * lr
                                if new_score > self.args.score_max:
                                    new_score = self.args.score_max
                                if new_score < self.args.score_min:
                                    new_score = self.args.score_min
                                self.triplets_score[sentence]['score'] = new_score
                                self.logger.log(f" {sentence}, old score: {old_score}, grad: {grad_t}, new score : {new_score}")
                            else:
                                self.logger.log(f" {sentence}, the retrival result don't exit in triplets score &&&&&")

                # 处理学习率
                self.logger.log(f"old score_weight : {score_weight}, grad_weight : {grad_weight}, lr: {lr}")
                score_weight = score_weight - grad_weight * lr * 1e-3
                self.logger.log(f"new score_weight (before clamp): {score_weight}")
                if score_weight < 0.1:
                    score_weight = 0.1
                if score_weight > 0.9:    
                    score_weight = 0.9
                self.logger.log(f"new score_weight : {score_weight}")

        feedback_process_end_time = time.perf_counter()
        if method == 'standard_for_path_shared_prefix':
            return prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time, feedback_process_end_time - feedback_process_start_time
        else:
            return llm_feedback_end_time - llm_feedback_start_time, feedback_process_end_time - feedback_process_start_time

    
    def feedback_process_one(self, method, filtered_retrieve_result, response_dict, filtered_triple_3d, flag_TF, feedback_noise, filtered_probs, filtered_dict, pt_list_standard, args_param, sim_map, score_map):
        # feedback_process_start_time = time.perf_counter()
        if method == 'standard_for_path_shared_prefix':
            score_weight = args_param.score_weight
            lr = args_param.lr
            import math
            context = ""
            for idx, sentence in enumerate(filtered_retrieve_result, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            self.logger.log(f"{context}")
            self.logger.log(f"{json.dumps(response_dict, indent=2)}")
            # assert False
            # print(type(response_dict))
            # print(response_dict)
            if feedback_noise:
                flag_TF = not flag_TF
                print(f"----------------feedback_noise-------------: original: {not flag_TF}, now: {flag_TF}")
            if response_dict['Insufficient_information']:
                self.logger.log(f"Insufficient_information is True, no feedback processed.")
                pass
            else:
                if not flag_TF: # 根据对错反转反馈
                    new_dict = {key: -value * 0.4 for key, value in response_dict['Path_score'].items()}
                    response_dict['Path_score'] = new_dict
                else:
                    new_dict = {key: value * 0.4 for key, value in response_dict['Path_score'].items()}
                    response_dict['Path_score'] = new_dict
                # print(f"{json.dumps(response_dict, indent=2)}")
                # 计算期望
                EUL = 0
                AVGU = 0
                VLlist = {}
                # 所有都在计算，只不过有大量的反馈是0
                # for path_id, (path, pl) in enumerate(zip(filtered_retrieve_result, filtered_probs)):
                for path_id, pl in enumerate(filtered_probs):
                    EUL += (response_dict['Path_score'].get(f"Path {path_id}", 0) + 1) / 2 * pl
                    # print(f"{response_dict['Path_score'].get(f'Path {path_id}', 0)} {(response_dict['Path_score'].get(f'Path {path_id}', 0) + 1) / 2 * pl}")
                    AVGU += response_dict['Path_score'].get(f"Path {path_id}", 0) * pl
                # for path_id, (path, pl) in enumerate(zip(filtered_retrieve_result, filtered_probs)):
                for path_id, pl in enumerate(filtered_probs):
                    VLlist[f"Path {path_id}"] = pl * (response_dict['Path_score'].get(f"Path {path_id}", 0) - AVGU) / len(filtered_triple_3d[path_id])
                
                self.logger.log(f"Expected Utility of the LLM (EUL): {EUL}")
                self.logger.log(f"Average Utility of the LLM (AVGU): {AVGU}")
                # self.logger.log(f"Variance of the LLM (VL): {json.dumps(VLlist, indent=2)}")
        
                # 取负对数
                LEUL = -math.log(EUL) 
                triple_map = {}
                for idx, path in enumerate(filtered_triple_3d):
                    for triple in path:
                        if triple[3] == '->':
                            sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                        elif triple[3] == '<-':
                            sentence = f"{triple[2]} {triple[1]} {triple[0]}"
                        if sentence in triple_map:
                            triple_map[sentence].append(idx)
                        else:
                            triple_map[sentence] = [idx]
                grad_weight = 0
                for path_id, (path, pl) in enumerate(zip(filtered_triple_3d, filtered_probs)):
                    # 对于未反馈的是否更新
                    if f"Path {path_id}" in response_dict['Path_score']:
                        for triple in path:
                            if triple[3] == '->':
                                sentence = f"{triple[0]} {triple[1]} {triple[2]}"
                                sentence_s = f"{triple[0]} {triple[1].replace('_',' ')} {triple[2]}"
                            elif triple[3] == '<-':
                                sentence = f"{triple[2]} {triple[1]} {triple[0]}"
                                sentence_s = f"{triple[2]} {triple[1].replace('_',' ')} {triple[0]}"
                            plset = 0
                            for path_index in triple_map.get(sentence, []): # 拿到包含t的所有L
                                # if f"Path {path_index}" in response_dict['Path_score']: # 反馈了才计算
                                #     if flag_TF:
                                #         plset *= filtered_probs[path_index] * VLlist.get(f"Path {path_index}", 0) # 不知道乘上Vi对不对
                                #     else:
                                #         plset *= filtered_probs[path_index] * -VLlist.get(f"Path {path_index}", 0) # 不知道乘上Vi对不对
                                plset += filtered_probs[path_index] * VLlist.get(f"Path {path_index}") 
                            grad_t = - (score_weight / (2 * EUL)) * (plset / pt_list_standard[filtered_dict[sentence]])
                            grad_weight_t = ((sim_map.get(sentence_s) - score_map.get(sentence_s)) / (2 * EUL)) * (plset / pt_list_standard[filtered_dict[sentence]])
                            grad_weight += grad_weight_t
                            # self.logger.log(f"sentence_s: {sentence_s}, sim: {sim_map.get(sentence_s, 0)}, score: {score_map.get(sentence_s, 0)}, grad_weight_t: {grad_weight_t}, grand_weight: {grad_weight}")
                            
                            if sentence in self.triplets_score:
                                old_score = self.triplets_score[sentence]['score']
                                # if flag_TF:
                                #     isNegative = 1
                                # else:
                                #     isNegative = -1
                                new_score = old_score - grad_t * lr
                                if new_score > self.args.score_max:
                                    new_score = self.args.score_max
                                if new_score < self.args.score_min:
                                    new_score = self.args.score_min
                                self.triplets_score[sentence]['score'] = new_score
                                self.logger.log(f" {sentence}, old score: {old_score}, grad: {grad_t}, new score : {new_score}")
                            else:
                                self.logger.log(f" {sentence}, the retrival result don't exit in triplets score &&&&&")

                # 处理学习率
                self.logger.log(f"old score_weight : {score_weight}, grad_weight : {grad_weight}, lr: {lr}")
                score_weight = score_weight - grad_weight * lr * 1e-3
                self.logger.log(f"new score_weight (before clamp): {score_weight}")
                if score_weight < 0.1:
                    score_weight = 0.1
                if score_weight > 0.9:    
                    score_weight = 0.9
                self.logger.log(f"new score_weight : {score_weight}")

        elif method == "basic_for_triplet_shared_prefix":
            # 此时 filtered_retrieve_result 是 triplet_unique
            score_weight = args_param.score_weight

            triplet_unique_list = []
            context = ""
            for idx, (h, r, t) in enumerate(filtered_retrieve_result, start=0):
                context += f"Path {idx}:\t{h} {r.replace('_',' ')} {t}\n"
                triplet_unique_list.append((h, r, t))
                            
            self.logger.log(f"triplet unique:\n{context}")
            self.logger.log(f"{json.dumps(response_dict, indent=2)}")

            if feedback_noise:
                flag_TF = not flag_TF
                print(f"----------------feedback_noise-------------: original: {not flag_TF}, now: {flag_TF}")
            if response_dict['Insufficient_information']:
                self.logger.log(f"Insufficient_information is True, no feedback processed.")
                pass
            else:
                for idx, (h, r, t) in enumerate(triplet_unique_list, start=0):
                    if f"Path {idx}" in response_dict['Path_score']:
                        score = response_dict['Path_score'][f"Path {idx}"] * 10
                        if not flag_TF:
                            score = -score
                        sentence = f"{h} {r} {t}"
                        if sentence in self.triplets_score:
                            old_score = self.triplets_score[sentence]['score']
                            new_score = old_score + score
                            if new_score > self.args.score_max:
                                new_score = self.args.score_max
                            if new_score < self.args.score_min:
                                new_score = self.args.score_min
                            self.triplets_score[sentence]['score'] = new_score
                            self.logger.log(f" {sentence}, old score: {old_score}, new score : {new_score}")
                        else:
                            self.logger.log(f" {sentence}, the retrival result don't exit in triplets score &&&&&")

    def evolve_basic_feedback(self):

        print(f"evolve_basic_feedback iteration: {self.args.iteration}")

        acc = 0
        acc_list = []
        acc_llm = 0
        acc_llm_part = 0
        delete_relationship_num = []
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        query_number = 0
        relationship_modify = []
        relationship_modify_index = []
        entity_modify = []
        entity_modify_index = []
        delete_entity_list = []
        insert_relationship_list_2d_all = []
        delete_triple_by_relationship_2d_all = []
        delete_triple_by_entity_2d_all = []
        triple_result_average = 0
        max_score_avg = []
        min_score_avg = []
        median_score_avg = []
        mean_score_avg = []
        
        if self.args.iteration == 0:
            with open(f"./logs/triplets/{self.args.space_name}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
                triplets_score = json.load(file)
        else:
            # with open(f"./logs/triplets/{self.args.space_name}_{self.args.option}_{self.args.llm}_{self.args.algorithm}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
            #     triplets_score = json.load(file)
            with open(f"./logs/triplets/{self.args.space_name}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
                triplets_score = json.load(file)
            
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.args.similar:
            with open(f"./logs/stage/{self.args.forward_llm}_evolve_basic_forward_{self.args.algorithm}_{self.args.space_name}_similar_{self.args.iteration}.json", 'r', encoding='utf-8') as file:
                data_list = json.load(file) 
        else:
            with open(f"./logs/stage/{self.args.forward_llm}_evolve_basic_forward_{self.args.algorithm}_{self.args.space_name}_{self.args.iteration}.json", 'r', encoding='utf-8') as file:
                data_list = json.load(file)

        if self.args.similar:
            logger = Logger(f"./data/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_similar_{self.args.iteration}")
        else:
            logger = Logger(f"./data/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_{self.args.iteration}")

        self.logger = logger
        self.triplets_score = triplets_score
        self.retriver.triplets_score = triplets_score # 传递给retriver

        if self.args.iteration == 0:
            feedback_random_last = []
        else:
            with open(f"./logs/data/{self.args.space_name}_error_feedback_random_{self.args.iteration-1}.json", 'r', encoding='utf-8') as file:
                feedback_random_last = json.load(file)
        feedback_noise = self.select_10_percent(len(data_list), self.args.rate, feedback_random_last)
        with open(f"./logs/data/{self.args.space_name}_error_feedback_random_{self.args.iteration}.json", 'w', encoding='utf-8') as file:
             file.write(json.dumps(feedback_noise, indent=2))
        logger.log(f"-------------feedback_noise last (rate: {self.args.rate}): {feedback_random_last}")
        logger.log(f"-------------feedback_noise (rate: {self.args.rate}): {feedback_noise}")
        # assert False

        llm_feedback_total_time = []
        feedback_total_time = []
        redundancy_total_time = []
        feedback_process_total_time = []
        
        feedback_generate_time_total = []
        feedback_prefill_time_total = []
        feedback_decode_time_total = []
        feedback_wait_scheduled_time_total = []
        feedback_end2end_time_total = []
        prompt = []
        generate = []
        for index in range(len(data_list)):
            # if index >= 10: 
            #     break
            # if index != 22:
            #     continue
            feedback_start_time = time.perf_counter()
            query_number = query_number + 1
            print(f"evolve basic feedback id : {index}")
            logger.log("\n\n")
            logger.log(f"-------------index : {index}")
            logger.log(f"query: {data_list[index]['query']}")
    
            keywords = data_list[index]['keywords'] # 问题相似度选 # 4
            logger.log(f"keywords : {keywords}")
            
            filtered_retrieve_result = data_list[index]['filtered_retrieve_result']  # 给大模型的句子
            filtered_triple_3d = data_list[index]['filtered_triple_3d']  # 带symbol len = 4
            if "filtered_probs" in data_list[index]:
                filtered_probs = data_list[index]['filtered_probs']
            else:
                filtered_probs = []
            if "filtered_dict" in data_list[index]:
                filtered_dict = data_list[index]['filtered_dict']
            else:
                filtered_dict = {}
            if "pt_list_standard" in data_list[index]:
                pt_list_standard = data_list[index]['pt_list_standard']
            else:
                pt_list_standard = []
            if "sim_map" in data_list[index]:
                sim_map = data_list[index]['sim_map']
            else:
                sim_map = {}
            if "score_map" in data_list[index]:
                score_map = data_list[index]['score_map']
            else:
                score_map = {}

            logger.log(f"filtered_retrieve_result : {json.dumps(filtered_retrieve_result, indent=2)}")

            response_one = data_list[index]['response']
            self.response.append(response_one)

            # 处理冗余
            if self.args.redundancy:
                delete_relationship_num_tmp, redundancy_time = self.redundant_process(method=self.args.redundant_process, filtered_triple_3d=filtered_triple_3d, index=index)
                delete_relationship_num.append(delete_relationship_num_tmp)
                redundancy_total_time.append(redundancy_time)
            
            # 对错判断
            logger.log(f"answer: {data_list[index]['answer']}")
            logger.log(f"graph rag response of paths: {response_one}")
            flag_TF = data_list[index]['label'] 
            if flag_TF:
                logger.log(f"-------------graph rag response true----------------------------------")
                acc = acc + 1
                acc_list.append(1)
            else:
                logger.log(f"-------------graph rag response false ~~~~~")
                acc_list.append(0)
            
            # 反馈处理
            # feedback_process_time =  self.feedback_process(method=self.args.feedback_process, filtered_retrieve_result = filtered_retrieve_result, filtered_triple_3d = filtered_triple_3d, flag_TF = flag_TF, index = index, feedback_noise=feedback_noise, question=data_list[index]['query'], response=response_one, filtered_probs=filtered_probs, filtered_dict=filtered_dict, pt_list_standard=pt_list_standard, score_weight = self.args.score_weight, lr=self.args.lr, sim_map=sim_map, score_map = score_map)
            if self.args.feedback_process == 'standard_for_path_shared_prefix':
                prompt_len, generate_len, llm_feedback_time, generate_time, prefill_time, decode_time, wait_scheduled_time, feedback_process_time =  self.feedback_process(method=self.args.feedback_process, filtered_retrieve_result = filtered_retrieve_result, filtered_triple_3d = filtered_triple_3d, flag_TF = flag_TF, index = index, feedback_noise=feedback_noise, question=data_list[index]['query'], response=response_one, filtered_probs=filtered_probs, filtered_dict=filtered_dict, pt_list_standard=pt_list_standard, args_param = self.args, sim_map=sim_map, score_map = score_map)
                prompt.append(prompt_len)
                generate.append(generate_len)
                feedback_generate_time_total.append(generate_time)
                feedback_prefill_time_total.append(prefill_time)
                feedback_decode_time_total.append(decode_time)
                feedback_wait_scheduled_time_total.append(wait_scheduled_time)
                logger.log(f"prompt_len: {prompt_len}")
                logger.log(f"generate_len: {generate_len}")
                logger.log(f"generate_time: {generate_time}")
                logger.log(f"prefill_time: {prefill_time}")
                logger.log(f"decode_time: {decode_time}")
                logger.log(f"wait_scheduled_time: {wait_scheduled_time}")
                
            else:
                llm_feedback_time, feedback_process_time =  self.feedback_process(method=self.args.feedback_process, filtered_retrieve_result = filtered_retrieve_result, filtered_triple_3d = filtered_triple_3d, flag_TF = flag_TF, index = index, feedback_noise=feedback_noise, question=data_list[index]['query'], response=response_one, filtered_probs=filtered_probs, filtered_dict=filtered_dict, pt_list_standard=pt_list_standard, args_param = self.args, sim_map=sim_map, score_map = score_map)
            
            llm_feedback_total_time.append(llm_feedback_time)
            feedback_process_total_time.append(feedback_process_time)
            logger.log(f"llm_feedback_time: {llm_feedback_time} seconds")
            logger.log(f"feedback_process_time: {feedback_process_time} seconds")
            # assert False
            feedback_end_time = time.perf_counter()
            feedback_total_time.append(feedback_end_time - feedback_start_time)
            logger.log(f"feedback total time: {feedback_end_time - feedback_start_time} seconds")
            

        if self.args.dataset_name == 'metaqa':
            # 分 hop1 200 hop2 100 hop3 100
            # hop1 300 hop2 300
            
            print(f"query_number: {query_number}")
            for range_start, range_end in [[0,300],[300,600]]:
                response_list = []
                answer_list = []
                range_acc = []
                for i in range(range_start, min(query_number, range_end)):
                    response_item = {}
                    answer_item = {}
                    response_item['id'] = str(i)
                    response_item["prediction_text"] = self.response[i]
                    response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
                    response_list.append(response_item)
                    reference_answer_item = {}
                    reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
                    reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                    answer_item['id'] = str(i)
                    answer_item['answers'] = reference_answer_item
                    answer_list.append(answer_item)
                    range_acc.append(acc_list[i])
                squad_v2_metric = load("squad_v2")
                if response_list and answer_list:
                    results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
                    logger.log(f"--------squad_v2 em f1 ... range[{range_start},{range_end}]------------\n{json.dumps(results_v2, indent=2)}")
                # if range_acc:
                    logger.log(f"--------accuracy range[{range_start},{range_end}]: {sum(range_acc)/len(range_acc)}-----------------------")
        else:
            # EM F1
            response_list = []
            answer_list = []
            print(f"query_number: {query_number}")
            for i in range(query_number):
                response_item = {}
                answer_item = {}
                response_item['id'] = str(i)
                response_item["prediction_text"] = self.response[i]
                response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
                response_list.append(response_item)
                reference_answer_item = {}
                reference_answer_item['text'] = self.combine_answer_formats(data_list[i]['answer'], delimiter=', ')
                reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                answer_item['id'] = str(i)
                answer_item['answers'] = reference_answer_item
                answer_list.append(answer_item)
            squad_v2_metric = load("squad_v2")
            results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
            logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")
        # squad_metric = load("squad")
        # results = squad_metric.compute(predictions = response_list, references = answer_list)
        # logger.log(f"--------squad em f1 ...------------\n{json.dumps(results, indent=2)}")


        logger.log(f"\n\n-----------------------accuracy: {acc/query_number}-----------------------")
        logger.log(f"\n\n-----------------------accuracy llm: {acc_llm/query_number}-----------------------")

        logger.log(f"\n\n-----------------------prompt len: {sum(prompt)/len(prompt)}-----------------------")
        logger.log(f"\n\n-----------------------generate len: {sum(generate)/len(generate)}-----------------------")
        

        logger.log(f"\n\n-----------------------feedback_total_time: {sum(feedback_total_time)/len(feedback_total_time)}-----------------------")

        if self.args.redundancy:
            logger.log(f"\n\n-----------------------redundancy_total_time: {sum(redundancy_total_time)/len(redundancy_total_time)}-----------------------")
        logger.log(f"\n\n-----------------------llm_feedback_total_time: {sum(llm_feedback_total_time)/len(llm_feedback_total_time)}-----------------------")
        if feedback_generate_time_total:
            logger.log(f"\n\n-----------------------feedback generate time: {sum(feedback_generate_time_total)/len(feedback_generate_time_total)}-----------------------")
            logger.log(f"\n\n-----------------------feedback prefill time: {sum(feedback_prefill_time_total)/len(feedback_prefill_time_total)}-----------------------")
            logger.log(f"\n\n-----------------------feedback decode time: {sum(feedback_decode_time_total)/len(feedback_decode_time_total)}-----------------------")
            logger.log(f"\n\n-----------------------feedback wait scheduled time: {sum(feedback_wait_scheduled_time_total)/len(feedback_wait_scheduled_time_total)}-----------------------")

        logger.log(f"\n\n-----------------------feedback_process_total_time: {sum(feedback_process_total_time)/len(feedback_process_total_time)}-----------------------")


        # logger.log(f"\n\n-----------------------delete_relationship_num: {sum(delete_relationship_num)/len(delete_relationship_num)}-----------------------")
        # logger.log(f"\n\n-----------------------delete_relationship_by_entity_num: {delete_relationship_by_entity_num}-----------------------")

        # with open(f"./logs/triplets/{self.args.space_name}_{self.args.llm}_{self.args.algorithm}_{self.args.iteration+1}.json", "w", encoding="utf-8") as file:
        #     json.dump(self.triplets_score, file, ensure_ascii=False, indent=2)
        with open(f"./logs/triplets/{self.args.space_name}_{self.args.iteration+1}.json", "w", encoding="utf-8") as file:
            json.dump(self.triplets_score, file, ensure_ascii=False, indent=2)
    

    def evolve_batch(self):

        print(f"evolve_basic iteration: {self.args.iteration} batch_size: {self.args.batch_size}")
        # assert False

        # load score

        acc_list = []
        acc_list_end = []
        acc_llm = 0
        acc_refuse = 0
        delete_relationship_num = 0
        insert_relationship_by_entity_num = 0
        delete_relationship_by_entity_num = 0
        delete_entity_num = 0
        data = []
        labels_list = []

        print(type(self.args.iteration))
        print(f"self.args.iteration: {self.args.iteration}")
        
        # 读取三元组分数
        
        with open(f"./logs/{self.args.algorithm}/triplets/{self.args.space_name}_{self.args.iteration}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)

        

        # 是否使用相似问题
        if self.args.similar:
            if self.args.dataset_name == 'rgb':
                with open("./dataset/rgb/similar_questions_api_best.json", 'r', encoding='utf-8') as f:
                    self.dataset.query = json.load(f)
            elif self.args.dataset_name == 'multihop':
                with open("./dataset/multihop/dataset/similar_questions_api_best.json", 'r', encoding='utf-8') as f:
                    self.dataset.query = json.load(f)
            elif self.args.dataset_name == 'hotpotqa600':
                with open("./dataset/hotpotqa_graph/similar_questions_api_best.json", 'r', encoding='utf-8') as f:
                    self.dataset.query = json.load(f)
        
        # 是否重新生成嵌入文件 未启用
        if self.args.shouldRebuildDatabase:
            # 这轮开始重新生成嵌入
            entities_new = set()
            for value in triplets_score.values():
                if value['score'] > self.args.scoreThreshold:
                    entities_new.add(value['triplet'][0])
                    entities_new.add(value['triplet'][2])
            self.retriver.generate_new_entity_embedding_standard(list(sorted(entities_new)))
        
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 生成log文件
        if self.args.similar:
            logger = Logger(f"./{self.args.algorithm}/data/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_similar_{self.args.iteration}")
        else:
            logger = Logger(f"./{self.args.algorithm}/data/{self.args.llmbackend}_{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_{self.args.iteration}")
        
        self.logger = logger
        self.retriver.triplets_score = triplets_score # 传递给retriver
        self.retriver.logger = logger # 传递给retriver
        self.triplets_score = triplets_score
    
        delete_relationship_num = []
        keywords_total_time = []
        nebula_retrieve_total_time = []
        filter_retrieve_total_time = []
        forward_response_total_time = []
        feedback_response_total_time = []
        forward_prompt = []
        redundancy_total_time = []
        feedback_process_total_time = []
        feedback_prompt = []

        
        # self.dataset.query = self.dataset.query[0:len(self.dataset.query)]
        # self.dataset.query = self.dataset.query[0:10]
        data = []

        # 加噪声
        if self.args.iteration == 0:
            feedback_random_last = []
        else:
            with open(f"./logs/{self.args.algorithm}/stage/{self.args.space_name}_error_feedback_random_{self.args.iteration-1}.json", 'r', encoding='utf-8') as file:
                feedback_random_last = json.load(file)
        feedback_noise = self.select_10_percent(len(self.dataset.query), self.args.rate, feedback_random_last)
        with open(f"./logs/{self.args.algorithm}/stage/{self.args.space_name}_error_feedback_random_{self.args.iteration}.json", 'w', encoding='utf-8') as file:
             file.write(json.dumps(feedback_noise, indent=2))
        logger.log(f"-------------feedback_noise last (rate: {self.args.rate}): {feedback_random_last}")
        logger.log(f"-------------feedback_noise (rate: {self.args.rate}): {feedback_noise}")

        for batch_idx, start_index in enumerate(range(0, len(self.dataset.query), self.args.batch_size)):
            end_index = min(start_index + self.args.batch_size, len(self.dataset.query))
            # if start_index >= 16: 
            #     break

            print(f"Batch {batch_idx}: start={start_index}, end={end_index}")
            logger.log(f"\n\n-------------Batch {batch_idx}: start={start_index}, end={end_index}------------")
            forward_start_time = time.perf_counter()
            
            data.extend([{} for _ in range(start_index, end_index)])

            # 获取问题+答案
            query_list = []
            for index in range(start_index, end_index):
                data[index]['index'] = index
                data[index]['query'] = self.dataset.query[index]
                query_list.append(self.dataset.query[index])
                data[index]['answer'] = self.dataset.answer[index]

            # 实体检索
            keywords_list, keyword_time = self.retriver.extract_keywords_with_embedding_find_entity(query_list, self.args.entity) # 问题相似度选 10
            for i, index in enumerate(range(start_index, end_index)):
                data[index]['keywords'] = keywords_list[i]
                data[index]['keyword_time'] = keyword_time/len(keywords_list)
            
            logger.log(f"keywords extraction time for batch: {keyword_time} seconds")   
            keywords_total_time.append(keyword_time)

            # 路径检索
            nebula_retrieve_batch_time = 0
            filter_retrieve_batch_time = 0
            filtered_retrieve_result_batch = []
            filtered_triple_3d_batch = []
            triplet_unique_batch = []
            for i, index in enumerate(range(start_index, end_index)):
                if self.args.retrieve_path == 'basic':
                    filtered_retrieve_result, filtered_triple_3d, nebula_retrieve_time, filter_retrieve_time  = self.retrieve_path_with_keywords_module(method = self.args.retrieve_path, question=self.dataset.query[index], keywords=data[index]['keywords'], path_depth = self.args.hop, pruning=self.args.pruning, threshold=self.args.simThreshold, score_threshold = self.args.scoreThreshold, score_weight = self.args.score_weight, top_k_per_entity=self.args.top_k_per_entity) # 10*30

                    data[index]['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子，带symbol
                    data[index]['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
                    data[index]['filtered_probs'] = []
                    data[index]['filtered_dict'] = []
                    data[index]['pt_list_standard'] = []
                    data[index]['sim_map'] = []
                    data[index]['score_map'] = []
                    # 一些用于分析的指标
                    # logger.log(f"len(triple_result) : {len(triple_result)}")
                    # logger.log(f"max_score: {max_score}, min_score: {min_score}, median_score: {median_score}, mean_score: {mean_score}")
                    # triple_result_average += len(triple_result)
                    
                elif self.args.retrieve_path == 'standard':
                    filtered_retrieve_result, filtered_triple_3d, filtered_probs, filtered_dict, pt_list_standard, sim_map, score_map, nebula_retrieve_time, filter_retrieve_time = self.retrieve_path_with_keywords_module(method = self.args.retrieve_path, question=self.dataset.query[index], keywords=data[index]['keywords'], path_depth = self.args.hop, pruning=self.args.pruning, threshold=self.args.simThreshold, score_threshold = self.args.scoreThreshold, score_weight = self.args.score_weight, top_k_per_entity=self.args.top_k_per_entity) # 10*30
                    data[index]['filtered_retrieve_result'] = filtered_retrieve_result # 给大模型的句子，带symbol
                    data[index]['filtered_triple_3d'] = filtered_triple_3d # 带symbol len = 4
                    data[index]['filtered_probs'] = filtered_probs
                    data[index]['filtered_dict'] = filtered_dict
                    data[index]['pt_list_standard'] = pt_list_standard
                    data[index]['sim_map'] = sim_map
                    data[index]['score_map'] = score_map
                    
                    # logger.log(f"filtered_probs : {json.dumps(filtered_probs, indent=2)}")
                    # logger.log(f"filtered_dict : {json.dumps(filtered_dict, indent=2)}")
                    # logger.log(f"pt_list_standard : {json.dumps(pt_list_standard, indent=2)}")

                # 一跳三元组
                seen = set()
                triplet_unique = []   
                for path_triple in filtered_triple_3d:
                    for triple in path_triple:
                        h, r, t, direction = triple

                        if direction == '->':
                            head, tail = h, t
                        elif direction == '<-':
                            head, tail = t, h
                        else:
                            continue
                        sentence_text = f"{head} {r.replace('_', ' ')} {tail}"
                        if sentence_text in seen:
                            continue
                        seen.add(sentence_text)
                        triplet_unique.append((head, r, tail))
                triplet_unique.sort(key=lambda x: (x[0], x[2])) 

                triplet_unique_batch.append(triplet_unique)
                filtered_retrieve_result_batch.append(filtered_retrieve_result)
                # filtered_triple_3d_batch.append(filtered_triple_3d)
                data[index]['nebula_retrieve_time'] = nebula_retrieve_time
                data[index]['filter_retrieve_time'] = filter_retrieve_time
                nebula_retrieve_total_time.append(nebula_retrieve_time)
                filter_retrieve_total_time.append(filter_retrieve_time)
                nebula_retrieve_batch_time += nebula_retrieve_time
                filter_retrieve_batch_time += filter_retrieve_time
            logger.log(f"nebula retrieve time for batch: {nebula_retrieve_batch_time} seconds")
            logger.log(f"filter retrieve time for batch: {filter_retrieve_batch_time} seconds")

            # 问答
            # if self.args.graphrag_response == "basic":
            #     response_one, num_input_tokens, response_time = self.graphrag.chat_without_stream_with_triplets(self.dataset.query[index], filtered_retrieve_result)
            if self.args.graphrag_response == "basic_shared_prefix" :
                response, prompt_len, response_time = self.graphrag.chat_without_stream_with_triplets_shared_prefix_batch(self.dataset.query[start_index: end_index], filtered_retrieve_result_batch)
                for i, index in enumerate(range(start_index, end_index)):
                    data[index]['response'] = response[i]
                    data[index]['prompt_tokens'] = prompt_len[i]
                    data[index]['response_time'] = response_time/len(response)
                    forward_prompt.append(prompt_len[i])
                    self.response.append(response[i])

                    # 对错判断
                    if self.args.dataset_name == 'dragonball':
                        flag_label = self.checkanswer_rougel(response[i], self.dataset.answer[index])
                        data[index]['label'] = flag_label
                        labels_list.append(flag_label)
                    else:
                        flag_label = self.checkanswer(response[i], self.dataset.answer[index])
                        flag_TF = sum(flag_label) == len(flag_label)
                        if flag_TF:
                            acc_list.append(1)
                        else:
                            acc_list.append(0)
                        data[index]['label'] = flag_TF

            
            logger.log(f"forward response time for batch: {response_time} seconds")
            forward_response_total_time.append(response_time)
            
            # 反馈问答 暂未反馈到三元组
            if self.args.feedback:
                if self.args.feedback_process == 'standard_for_path_shared_prefix':
                    feedback_response_batch, prompt_len, feedback_response_time = self.graphrag.chat_without_stream_for_socre_feedback_standard_shared_prefix_batch(self.dataset.query[start_index: end_index], self.response[start_index: end_index], filtered_retrieve_result_batch)

                    for i, index in enumerate(range(start_index, end_index)):
                        data[index]['feedback_response'] = feedback_response_batch[i]
                        data[index]['feedback_prompt_tokens'] = prompt_len[i]
                        data[index]['feedback_response_time'] = feedback_response_time/len(feedback_response_batch)
                        feedback_prompt.append(prompt_len[i])
                    feedback_response_total_time.append(feedback_response_time)
                elif self.args.feedback_process == "basic_for_triplet_shared_prefix":
                    feedback_response_batch, prompt_len, feedback_response_time = self.graphrag.chat_without_stream_for_socre_feedback_basic_shared_prefix_batch(self.dataset.query[start_index: end_index], self.response[start_index: end_index], triplet_unique_batch)

                    for i, index in enumerate(range(start_index, end_index)):
                        data[index]['feedback_response'] = feedback_response_batch[i]
                        data[index]['feedback_prompt_tokens'] = prompt_len[i]
                        data[index]['feedback_response_time'] = feedback_response_time/len(feedback_response_batch)
                        feedback_prompt.append(prompt_len[i])
                    feedback_response_total_time.append(feedback_response_time)
                logger.log(f"feedback response time for batch: {feedback_response_time} seconds")

            # 记录整个batch的log
            for i, index in enumerate(range(start_index, end_index)):
                print(f"Processing index: {index}")
                logger.log(f"\n\n-------------index : {index}")
                logger.log(f"query: {data[index]['query']}")
                logger.log(f"prompt_tokens: {data[index]['prompt_tokens']}")
                logger.log(f"keywords : {data[index]['keywords']}")
                logger.log(f"filtered_retrieve_result : {json.dumps(data[index]['filtered_retrieve_result'], indent=2)}")
                logger.log(f"answer: {data[index]['answer']}")
                logger.log(f"response: {data[index]['response']}")
                logger.log(f"graph rag reaponse {data[index]['label']}")

                # 处理冗余 涉及大模型问答
                if self.args.redundancy:
                    
                    delete_relationship_num_tmp, redundancy_time = self.redundant_process(method=self.args.redundant_process, filtered_triple_3d=data[index]["filtered_triple_3d"], index=index)
                    delete_relationship_num.append(delete_relationship_num_tmp)
                    redundancy_total_time.append(redundancy_time)
                    

                # 反馈处理
                if self.args.feedback:
                    if self.args.feedback_process == 'standard_for_path_shared_prefix' or self.args.feedback_process == 'basic_for_triplet_shared_prefix':
                        feedback_process_start_time = time.perf_counter()
                        self.feedback_process_one(
                            method = self.args.feedback_process,
                            filtered_retrieve_result = filtered_retrieve_result_batch[i] if self.args.feedback_process == "standard_for_path_shared_prefix" else triplet_unique_batch[i],
                            response_dict = data[index]["feedback_response"],
                            filtered_triple_3d = data[index]['filtered_triple_3d'],
                            flag_TF = data[index]['label'], 
                            feedback_noise = index in feedback_noise, 
                            filtered_probs = data[index]['filtered_probs'], 
                            filtered_dict = data[index]['filtered_dict'], 
                            pt_list_standard = data[index]['pt_list_standard'], 
                            args_param = self.args, 
                            sim_map = data[index]['sim_map'], 
                            score_map = data[index]['score_map'],)
                        feedback_process_end_time = time.perf_counter()
                        feedback_process_time = feedback_process_end_time - feedback_process_start_time
                        logger.log(f"feedback_process_time: {feedback_process_time} seconds")
                        feedback_process_total_time.append(feedback_process_time)

        self.short_cut_for_graphdatabase()

        if self.args.dataset_name == 'dragonball':
            accuracy = self.get_accuracy_rougel(labels_list)
            logger.log(f"----------accuracy rougel ...------------\n{accuracy}")
        elif self.args.dataset_name == 'metaqa':
            # 分 hop1 200 hop2 100 hop3 100
            # hop1 300 hop2 300
            
            print(f"query_number: {len(data)}")
            for range_start, range_end in [[0,300],[300,600]]:
                response_list = []
                answer_list = []
                range_acc = []
                for i in range(range_start, min(len(data), range_end)):
                    response_item = {}
                    answer_item = {}
                    response_item['id'] = str(i)
                    response_item["prediction_text"] = self.response[i]
                    response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
                    response_list.append(response_item)
                    reference_answer_item = {}
                    reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
                    reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                    answer_item['id'] = str(i)
                    answer_item['answers'] = reference_answer_item
                    answer_list.append(answer_item)
                    range_acc.append(acc_list[i])
                squad_v2_metric = load("squad_v2")
                if response_list and answer_list:
                    results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
                    logger.log(f"--------squad_v2 em f1 ... range[{range_start},{range_end}]------------\n{json.dumps(results_v2, indent=2)}")
                # if range_acc:
                    logger.log(f"--------accuracy range[{range_start},{range_end}]: {sum(range_acc)/len(range_acc)}-----------------------")
        else:
            # EM F1
            response_list = []
            answer_list = []
            print(f"query_number: {len(data)}")
            for i in range(len(data)):
                response_item = {}
                answer_item = {}
                response_item['id'] = str(i)
                response_item["prediction_text"] = self.response[i]
                response_item["no_answer_probability"] = 0.0   # 'no_answer_probability'
                response_list.append(response_item)
                reference_answer_item = {}
                reference_answer_item['text'] = self.combine_answer_formats(self.dataset.answer[i], delimiter=', ')
                reference_answer_item['answer_start'] = [0] * len(reference_answer_item['text'])
                answer_item['id'] = str(i)
                answer_item['answers'] = reference_answer_item
                answer_list.append(answer_item)
            squad_v2_metric = load("squad_v2")
            results_v2 = squad_v2_metric.compute(predictions = response_list, references = answer_list)
            logger.log(f"--------squad_v2 em f1 ...------------\n{json.dumps(results_v2, indent=2)}")


        logger.log(f"\n\n-----------------------accuracy : {sum(acc_list)/len(data)} acc query: {sum(acc_list)} / total query: {len(data)}-----------------------")
        logger.log(f"-----------------------forward prompt: {sum(forward_prompt)/len(data)}-----------------------")
        if self.args.feedback:
            logger.log(f"-----------------------feedback prompt: {sum(feedback_prompt)/len(data)}-----------------------")

        logger.log(f"\n\n-----------------------keywords_total_time: {sum(keywords_total_time)/len(data)}-----------------------")
        logger.log(f"-----------------------nebula retrieve_total_time: {sum(nebula_retrieve_total_time)/len(data)}-----------------------")
        logger.log(f"-----------------------filter retrieve_total_time: {sum(filter_retrieve_total_time)/len(data)}-----------------------")
        logger.log(f"-----------------------forward_response_total_time: {sum(forward_response_total_time)/len(data)}-----------------------")
        if self.args.feedback:
            logger.log(f"-----------------------feedback_response_total_time: {sum(feedback_response_total_time)/len(data)}-----------------------")
            logger.log(f"-----------------------feedback_process_total_time: {sum(feedback_process_total_time)/len(data)}-----------------------")

        if self.args.similar:
            with open(f"./logs/{self.args.algorithm}/stage/{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_similar_{self.args.iteration}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2))
        else:
            with open(f"./logs/{self.args.algorithm}/stage/{self.args.llm}_{self.args.option}_{self.args.algorithm}_{self.args.space_name}_{self.args.iteration}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2))
        
        with open(f"./logs/{self.args.algorithm}/triplets/{self.args.space_name}_{self.args.iteration+1}.json", "w", encoding="utf-8") as file:
            json.dump(self.triplets_score, file, ensure_ascii=False, indent=2)

    def check_one_response(self, response, answer):
        # print(f"response\n{response}")
        flag = False
        if isinstance(answer, list):
            instance = [j.lower() for j in answer]
            for j in instance:
                if j in response.lower():
                    flag = True
                    break
        else:
            instance = answer.lower()
            if instance in response.lower():
                flag = True

        return flag

    def check_one_response_ignore(self, response, answer):
        flag = 0
        if "The context do not contain answer" in response:
            return 2
        if isinstance(answer, list):
            instance = [j.lower() for j in answer]
            for j in instance:
                if j in response.lower():
                    flag = 1
                    break
        else:
            instance = answer.lower()
            if instance in response.lower():
                flag = 1

        return flag

    def test(self):
        # python kg_modify.py --dataset_name rgb_int --llm qwen3_32b --graphdb nebulagraph --space_name integrationrgb --option test --llmbackend api
        # python kg_modify.py --dataset_name metaqa --llm qwen3_32b --graphdb nebulagraph --space_name metaqa --option test --llmbackend api
        # get triplets from graph db
        print("start get triplets")
        all_triplets = self.graph_database.get_triplets()
        index = 0
        import json
        my_dict = {}
        for item in all_triplets:
            assert len(item) == 3
            result = " ".join(map(str, item))
            my_dict[result] = {}
            my_dict[result]["triplet"] = item
            my_dict[result]["score"] = 100
        with open(f"./logs/triplets/{self.args.space_name}.json", "w", encoding="utf-8") as file:
            json.dump(my_dict, file, ensure_ascii=False, indent=4)
    
    def transE(self):
        # get triplets from graph db
        print("start get transE triplets")
        all_triplets = self.graph_database.get_triplets() # 与下面方法二选一

        # all_triplets = []
        # with open(f"./logs/triplets/{self.args.space_name}.json", "r", encoding="utf-8") as f:
        #     triplets_score = json.load(f)
        # for key, value in triplets_score.items():
        #     x, y, z = value["triplet"]
        #     if x and y and z:
        #         all_triplets.append((str(x), str(y), str(z)))

        with open(f"./logs/stage/{self.args.space_name}_triplets.txt", "w", encoding="utf-8") as f:
            for triplet in all_triplets:
                f.write(triplet[0] + "\t" + triplet[1] + '\t' + triplet[2] +"\n")

        left_entities = [triplet[0] for triplet in all_triplets]
        right_entities = [triplet[2] for triplet in all_triplets]
        entities = set(left_entities + right_entities)
        relationship_all = [triplet[1] for triplet in all_triplets]
        relationship = set(relationship_all)

        with open(f"./logs/stage/{self.args.space_name}_entity.txt", "w", encoding="utf-8") as f:
            for idx, entity in enumerate(list(entities)):
                f.write(entity + "\t" + str(idx) +"\n")

        with open(f"./logs/stage/{self.args.space_name}_relationship.txt", "w", encoding="utf-8") as f:
            for idx, item in enumerate(list(relationship)):
                f.write(item + "\t" + str(idx) +"\n")

        print(f'triplets: {len(all_triplets)}, entities: {len(entities)}, relationship: {len(relationship)}')
        # print(list(entities)[:10])


    def random_numbers(self, train_file="./logs/multihop_train_random70.json", test_file="./logs/multihop_test_random30.json", total_range=2556, ratio=0.7):
        # import random
        # import json
        # all_numbers = list(range(total_range ))  # 1到2556共2556个数字
        # train_numbers = random.sample(all_numbers, int(total_range * ratio))
        # test_numbers = list(set(all_numbers) - set(train_numbers))
        # with open(train_file, 'w') as train_f:
        #     json.dump(train_numbers, train_f)
        # with open(test_file, 'w') as test_f:
        #     json.dump(test_numbers, test_f)
        # print(f"Training data saved to {train_file}")
        # print(f"Testing data saved to {test_file}")
        
        # input_text = "how are you?"
        # response_of_KG_list_path = "Path-based Evidence 1('I'->'went to'->'seaside'. Path-based Evidence 2('I'->'contracted'->'cold'). Path-based Evidence 3('seaside'->'had occurrence of'->'rainfall')"
        # response_of_KG_neighbor = "Neighbor-based Evidence 1('cold'->'need'->'Paracetamol'). Neighbor-based Evidence 2('Paracetamol'->'medical'->'Analgesic'). Neighbor-based Evidence 3('Paracetamol'->'moa:inhibits'->'COX-2'). Neighbor-based Evidence 4('Paracetamol'->'med:treats'->'cond:CommonCold'). Neighbor-based Evidence 5('Paracetamol'->'med:relieves'->'headache'). Neighbor-based Evidence 6('Paracetamol'->'med:relieves->'fever')"
                
        # history = [
        #     {"role": "system", "content": "You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation."},
        #     {"role": "user", "content": f"Patient input: {input_text}"},
        #     {"role": "assistant", "content": f"You have some medical knowledge information in the following:\n\n### {response_of_KG_list_path}\n\n### {response_of_KG_neighbor}"},
        #     {"role": "user", "content": (
        #         "What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommended medications can cure the disease? Think step by step.\n\n\n"
        #         "Output1: The answer includes disease and tests and recommended medications.\n\n"
        #         "Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighbor-based Evidence, and in the end infer what result. \n"
        #         "TranspRAG_2025/KGModify/__pycache__ort the inference process into the following format:\n"
        #         "Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->"
        #         "Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->"
        #         "result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...).\n\n"
        #         "Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, "
        #         "which is followed by the entity in parentheses.\n\n"
        #         "There is a sample:\n"
        #         "Output 1:\n"
        #         "Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, "
        #         "the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. "
        #         "Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. "
        #         "It is also recommended to rest the voice and avoid smoking and irritants.\n\n"
        #         "Output 2:\n"
        #         "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')"
        #         "->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->"
        #         "Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')"
        #         "->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')"
        #         "->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').\n\n"
        #         "Output 3:\n"
        #         "Patient(Path-based Evidence 1)\n"
        #         "└── has been experiencing(Path-based Evidence 1)\n"
        #         "    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)\n"
        #         "        └── could be caused by(Path-based Evidence 2)\n"
        #         "            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)\n"
        #         "                ├── requires(Neighbor-based Evidence 1)\n"
        #         "                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)\n"
        #         "                │       └── may include(Neighbor-based Evidence 2)\n"
        #         "                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)\n"
        #         "                ├── can be treated with(Path-based Evidence 3)\n"
        #         "                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)\n"
        #         "                └── should be accompanied by(Neighbor-based Evidence 3)\n"
        #         "                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)"
        #     )}
        # ]
        
        # input_text = "What is the price for a 30-second spot during the Super Bowl 2022?"
        # response_of_KG_list_path = ("Path-based Evidence 1('Super bowl 2022'->'Has price'->'$115,000')\n"
        # "Path-based Evidence 2('Super bowl'->'Received sponsorship from'->'Advertisers') -> Path-based Evidence 3('Advertisers'->'Spent on ad'->'$7 million')\n"
        # "Path-based Evidence 4('Super bowl'->'halftime show Produced by'->'Nfl') -> Path-based Evidence 5('Nfl'->'Partners with'->'Youtube tv')\n")

        # response_of_KG_neighbor = ("Neighbor-based Evidence 1('Super bowl 2022'->'Takes place on'->'February 13'); Neighbor-based Evidence 2('Super bowl 2022'->'Took place on'->'Jan 31, 2022')\n"
        # "Neighbor-based Evidence 3('Super bowl'->'Generated_revenue'->'$7 million')\n"
        # "Neighbor-based Evidence 4('Advertisers'->'Paid'->'$7 million for a thirty-second spot during super bowl lvi in 2022')\n"
        # "Neighbor-based Evidence 5('Nfl'->'Recorded highest price'->'$6.5 million'); Neighbor-based Evidence 6('Nfl'->'Seeks higher payments for'->'Super bowl halftime show sponsorship rights'); "
        # "Neighbor-based Evidence 7('Nfl'->'Enhanced live performance with'->'Roc nation')")
        
        # last_response = "$6.5 million"
        # flag_TF = "Correct"
        
        # history = [
        #     {"role": "system", "content": "You are an advanced AI assistant capable of reasoning over structured knowledge graphs. You help users answer complex questions by extracting relevant information from structured triples and evaluating how well the retrieved knowledge supports the generated response."},
        #     {"role": "user", "content": f"User question: {input_text}"},
        #     {"role": "assistant", "content": f"I have some knowledge information related to the question in the following:\n\n### {response_of_KG_list_path}\n\n### {response_of_KG_neighbor}. I think the answer to the question is {last_response}"},
        #     {"role": "user", "content": (
        #         f"The correctness of your previous response: {flag_TF}. Now, evaluate the retrieved knowledge based on the correctness of your previous response. Think step by step.\n\n\n"
                
        #         "Output1: If your response was correct: Assign a score (1-3) to each Path-based Evidence or Neighbor-based Evidence supporting the correct answer (higher score = more relevant).  Assign a score (1-3) to each Path-based Evidence or Neighbor-based Evidence contributing to the incorrect answer (higher score = more misleading).\n"
        #         "If your response was incorrect: Assign a score (1-3) to each Path-based Evidence or Neighbor-based Evidence contributing to the incorrect answer (higher score = more misleading).  No need to find correct Path-based Evidence or Neighbor-based Evidence and output rating\n"
        #         "Regardless of whether the last answer was correct, if the last response did not directly answer the question but instead indicated that the retrieved information was insufficient, irrelevant, or that more information was needed "
        #         "(e.g., responses like 'without more specific information' or 'need a bit more context or details about the xxx').\n" 
        #         "Do not score any retrieved knowledge.Just output the sentence 'Insufficient search information'. And skip the outputs of output 1 and output 2 afterwards.\n\n"

        #         "Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighbor-based Evidence, and in the end infer what result. "
        #         "The final result is compared with the previous answer, and the correctness of the previous answer is used to determine whether the inference result is reasonable. \n"
        #         "Transport the inference process into the following format:\n"
        #         "Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->"
        #         "Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->"
        #         "result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...).\n"

        #         "Output3: Draw a decision tree.(You can have multiple) The entity or relation in single quotes in the inference process is added as a node with the source of evidence, "
        #         "which is followed by the entity in parentheses.\n"
        #         "If the evidence supports the last answer and the last answer is correct, it is marked as correct; if the evidence supports the last answer but the last answer is wrong or opposes the last answer" 
        #         "but the last answer is correct, it is marked as wrong. Other irrelevant ones do not need to be marked.\n\n"
        #         "There is a sample:\n"
        #         "Output 1:\n"
        #         "Correct Path-based Evidence: \n"
        #         "Correct Neighbor-based Evidence: 3:3\n"
        #         "Error Path-based Evidence: 3:2\n"
        #         "Error Neighbor-based Evidence: 2:3\n\n"
        #         "Output 2:\n"
        #         "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')"
        #         "->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->"
        #         "Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')"
        #         "->result 1('laryngitis')(Correct)->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')"
        #         "->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').\n\n"
        #         "Output 3:\n"
        #         "Patient(Path-based Evidence 1)\n"
        #         "└── has been experiencing(Path-based Evidence 1)\n"
        #         "    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)\n"
        #         "        └── could be caused by(Path-based Evidence 2)\n"
        #         "            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)\n"
        #         "                ├── requires(Neighbor-based Evidence 1)\n"
        #         "                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)\n"
        #         "                │       └── may include(Neighbor-based Evidence 2)\n"
        #         "                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)(Error)\n"
        #         "                ├── can be treated with(Path-based Evidence 3)\n"
        #         "                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)(Error)\n"
        #         "                └── should be accompanied by(Neighbor-based Evidence 3)\n"
        #         "                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)(Correct)"
        #     )}
        # ]
        input_0 = (
            "Group 0:\n"
            "1. December 18, 2022\n"
            "2. November 20, 2022\n"
            "3. November 30, 2022\n"
            "4. 18 december 2022\n"
            "5. 29 november 2022\n"
            "Group 1:\n"
            "6. Sam griffith\n"
            "7. Dr. rob griffith\n"
            "8. Rob griffith\n"
            "Group 2:\n"
            "9. 936,000 cars delivered in 2021\n"
            "10. 308,600 vehicle deliveries in the fourth quarter of 2021\n"
            "Group 3:\n"
            "11. the 310,000 vehicles\n"
            "12. 310,000 vehicles\n"
            "13. Over 310,000 vehicles\n"
            "14. 305,000 vehicles"
        )
        response_0 = (
            "4, 1\n"
            "7, 8\n"
            "12, 11"
        )

        promot = (
            "You will be given multiple groups of entities from a knowledge graph. Each entity in a group has a unique ID. Your task is to identify sets of entities within each group that have exactly the same meaning.\n"
            "Two entities are considered to have the same meaning only if they can always be used interchangeably in any context, without changing the meaning of the sentence. Equivalence must only be considered within each group. Do not compare entities across different groups.\n"
            "Please follow these specific rules:\n"
            "1. For numbers, including dates, quantities, and monetary values, two entities are equivalent only if they are exactly the same.\n"
            "2. Prefer full spellings over abbreviations (e.g., 'December 18, 2022' is preferred over '18 Dec 2022').\n"
            "3. For each equivalence set, choose the most complete and grammatically correct entity to appear first in the output list.\n"
            "4. Output only entity IDs, not entity names or text.\n"
            "5. Separate IDs in the same equivalence set using commas: ID1, ID2, ID3\n"
            "6. Output one equivalence set per line.\n"
            "7. If no entities in a group are equivalent, output nothing for that group.\n"
            "8. Output only the result—no explanation or extra text.\n\n"
            "Output Format Example:\n"
            "5, 3, 6\n"
            "12, 11\n" 
            "15, 14\n\n"
            "Entities:\n{input}"
        )

        input_1 = (
            "Group 0:\n"
            "0. Good sam\n"
            "1. The good sam\n"
            "2. Sam\n"
            "Group 1:\n"
            "3. Medical drama\n"
            "4. Medical k-dramas\n"
            "Group 2:\n"
            "5. Cbs television studios\n"
            "6. Cbs\n"
            "Group 3:\n"
            "7. Other current medical shows\n"
            "8. Medical show\n"
            "Group 4:\n"
            "9. Sam griffith\n"
            "10. Dr. rob griffith\n"
            "11. Rob griffith\n"
            "Group 5:\n"
            "12. 2022\n"
            "13. 2024\n"
            "Group 6:\n"
            "14. May 14, 2021\n"
            "15. May 2022\n"
            "Group 7:\n"
            "16. November, december\n"
            "17. 20 november to 18 december\n"
            "Group 8:\n"
            "18. January 2, 2022\n"
            "19. January 2\n"
            "Group 9:\n"
            "20. Dec 1, 2022\n"
            "21. 18 december 2022\n"
            "22. November 20, 2022\n"
            "23. November 30, 2022\n"
            "24. December 18, 2022\n"
            "25. 29 november 2022"
        )

        history = [
            {"role": "user", "content": promot.format(input = input_0)},
            {"role": "assistant", "content": response_0},
            {"role": "user", "content": promot.format(input = input_1)},
        ]
        
        output = self.graphrag.chat_without_stream_for_socre_feedback_multi_round_dialogue("", history, "")
        
        print(output)

    def demo(self):
        pass
        input_text = "Who won the 2022 Citrus Bowl?"
        last_answer = ""
        statement_list = ""
        with open(f"./KGModify/data.txt") as fin:
            for line in fin:
                statement_list += f"{line}\n"
        print(f"statement_list: {statement_list}")
        response_of_KG_list_path = statement_list


        history = [
            {
                "role": "system",
                "content": (
                    "You are an expert in reasoning over knowledge graphs. "
                    "You can infer answers based on the question and the triple-based declarative entity knowledge."
                )
            },
            {"role": "user", "content": f"User Input: {input_text}"},
            {
                "role": "assistant",
                "content": (
                    f"You have some potentially relevant triple-based declarative knowledge for the question as follows:\n\n"
                    f"### {response_of_KG_list_path}\n\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Please answer the user's question. Think step by step.\n\n\n"
                    "Output 1: The answer to the question.\n\n"
                    "Output 2: Show the inference process as a string. Extract which knowledge is used from which Path-based Evidence, "
                    "and indicate what final result is inferred. \n"
                    "Represent the inference process in the following format:\n"
                    "Path-based Evidence number('entity name'->'relation name'->...)->"
                    "Path-based Evidence number('entity name'->'relation name'->...)->"
                    "result number('entity name')->"
                    "Path-based Evidence number('entity name'->'relation name'->...)->...\n\n"
                    "Output 3: Draw a decision tree. Use the entities and relations in single quotes from the inference process as nodes. "
                    "Each node should include the evidence source, shown in parentheses.\n\n"
                    "Here is a sample:\n"
                    "Output 1:\n"
                    "Laryngitis.\n\n"
                    "Output 2:\n"
                    "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->"
                    "Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->"
                    "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')->"
                    "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')->"
                    "result 1('laryngitis').\n\n"
                    "Output 3:\n"
                    "Patient (Path-based Evidence 1)\n"
                    "└── has been experiencing (Path-based Evidence 1)\n"
                    "    └── hoarse voice (Path-based Evidence 1)(Path-based Evidence 2)\n"
                    "        └── could be caused by (Path-based Evidence 2)\n"
                    "            └── laryngitis (Path-based Evidence 2)(Path-based Evidence 4)\n"
                    "                ├── requires (Path-based Evidence 4)\n"
                    "                │   └── physical examination of the throat (Path-based Evidence 4)(Path-based Evidence 5)\n"
                    "                │       └── may include (Path-based Evidence 5)\n"
                    "                │           └── laryngoscopy (Path-based Evidence 5)(result 1)(Path-based Evidence 3)\n"
                    "                ├── can be treated with (Path-based Evidence 3)\n"
                    "                │   └── anti-inflammatory drugs and steroids (Path-based Evidence 3)(Path-based Evidence 6)\n"
                    "                └── should be accompanied by (Path-based Evidence 6)\n"
                    "                    └── resting the voice and avoiding irritants (Path-based Evidence 6)"
                )
            }
        ]
        output = self.graphrag.chat_without_stream_for_socre_feedback_multi_round_dialogue_test("", history)
        
        print(output)

    def mindmap_test(self): 

        with open(f"./logs/stage/mindmap_test_data.json", "r", encoding="utf-8") as file:
            test_data = json.load(file)
        pass
        # input_text = "Who won the 2022 Citrus Bowl?"
        # statement_list = ""
        # with open(f"./KGModify/data.txt") as fin:
        #     for line in fin:
        #         statement_list += f"{line}\n"
        # print(f"statement_list: {statement_list}")
        # response_of_KG_list_path = statement_list
        for index, dict_item in enumerate(test_data):
            print(f"----{index}----")
            # if index >= 7:
            #     break
            # if index <= 5:
            #     continue
            
            user_question = dict_item['question']
            retrieved_knowledge = ""
            evidence_list = dict_item['one_hop_sentence_list']
            for evidence in evidence_list:
                retrieved_knowledge += evidence
            final_answer = dict_item["response"]
            answer = dict_item['answer']
            answer_list = self.combine_answer_formats(answer) # 只取组合后的第一个答案
            print(f"answer_list: {json.dumps(answer_list, indent=2)}")
            answer_str = answer_list[0]
            print(f"answer_str: {answer_str}")
            triples = dict_item['one_hop_sentence_2d']
            sorted_triples = self.sort_triples_by_entity(triples)
            keywords_list = []
            for index, keyword in enumerate(dict_item['keywords']):
                keywords_list.append(f"Keywords {index}: {keyword}")
                
            history_all = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in reasoning over knowledge graphs. Your primary task is to construct a clear reasoning path "
                        "that supports a given answer, based on a user's question and a set of provided knowledge evidence. "
                        "You must strictly use only the provided evidence to build the reasoning chain."
                    )
                },
                {
                    "role": "user",
                    "content": f"User Question: \"{user_question}\""
                },
                {
                    "role": "assistant",
                    "content": (
                        "Based on the user's question, I have gathered the following potentially relevant evidence and formulated an answer.\n\n"
                        f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n"
                        f"### Final Answer:\n{final_answer}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Excellent. Now, based on the user's question, the `Retrieved Evidence`, and the `Final Answer` you provided in the last turn, "
                        "please identify and show the exact reasoning path that supports your answer. Think step by step.\n\n"
                        "Your output must have three parts:\n\n"
                        
                        "Output 1: First, restate the `Final Answer` from the previous turn that your reasoning path will justify.\n\n"
                        
                        "Output 2: Second, show the inference process as a string. Extract which knowledge is used from which `Path-based Evidence`, "
                        "and indicate how they connect to infer the final result. Represent the inference process in this exact format:\n"
                        "Path-based Evidence number('entity name'->'relation name'->...)->"
                        "Path-based Evidence number('entity name'->'relation name'->...)->"
                        "result number('entity name').\n\n"

                        "Output 3: Third, draw a decision tree based on the inference process. Use the entities and relations in single quotes "
                        "from the string in Output 2 as nodes. Each node must include the evidence source number in parentheses.\n\n"

                        "Here is a sample to strictly guide your output format:\n"
                        "--- SAMPLE START ---\n"
                        "Output 1:\n"
                        "Laryngitis.\n\n"

                        "Output 2:\n"
                        "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->"
                        "Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->"
                        "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')->"
                        "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')->"
                        "result 1('laryngitis').\n\n"

                        "Output 3:\n"
                        "Patient (Path-based Evidence 1)\n"
                        "└── has been experiencing (Path-based Evidence 1)\n"
                        "    └── hoarse voice (Path-based Evidence 1)(Path-based Evidence 2)\n"
                        "        └── could be caused by (Path-based Evidence 2)\n"
                        "            └── laryngitis (Path-based Evidence 2)(Path-based Evidence 4)(result 1)\n"
                        "                ├── requires (Path-based Evidence 4)\n"
                        "                │   └── physical examination of the throat (Path-based Evidence 4)(Path-based Evidence 5)\n"
                        "                │       └── may include (Path-based Evidence 5)\n"
                        "                │           └── laryngoscopy (Path-based Evidence 5)\n"
                        "                └── can be treated with (Path-based Evidence 3)\n"
                        "                    └── anti-inflammatory drugs and steroids (Path-based Evidence 3)\n"
                        "--- SAMPLE END ---"
                    )
                }
            ]

            history = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in reasoning over knowledge graphs. Your primary task is to construct a clear reasoning path "
                        "that supports a given answer, based on a user's question and a set of provided knowledge evidence. "
                        "You must strictly use only the provided evidence to build the reasoning chain."
                    )
                },
                {
                    "role": "user",
                    "content": f"User Question: \"{user_question}\""
                },
                {
                    "role": "assistant",
                    "content": (
                        "Based on the user's question, I have gathered the following potentially relevant evidence and formulated an answer.\n\n"
                        f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n"
                        f"### Final Answer:\n{final_answer}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Excellent. Now, based on the user's question, the `Retrieved Evidence`, and the `Final Answer` you provided in the last turn, "
                        "please identify and show the exact reasoning path that supports your answer. Think step by step.\n\n"
                        
                        "Output 1: Show the inference process as a string. For useful knowledge extracted from the `Retrieved Evidence`, "
                        "you must connect them to infer the final result and provide two scores in a `[Relevance, Contribution]` format.\n"
                        "Relevance to the Question (Score 1-3): How relevant the evidence is to the user's original question.\n"
                        "Contribution to the Answer (Score 1-3): How critical the evidence is for constructing the final answer.\n"
                        "(A higher score means greater relevance or contribution).\n"
                        "Represent the inference process in this exact format:\n"
                        "('entity name'->'relation name'->...)[Relevance, Contribution]->\n"
                        "('entity name'->'relation name'->...)[Relevance, Contribution]->\n"
                        "Result('the `Final Answer` you provided in the last turn').\n"

                        "Here is a sample to strictly guide your output format:\n"
                        "--- SAMPLE START ---\n"

                        "Output 1:\n"
                        "('Patient'->'has been experiencing'->'hoarse voice')->"
                        "('hoarse voice'->'could be caused by'->'laryngitis')->"
                        "('laryngitis'->'requires'->'physical examination of the throat')->"
                        "'physical examination of the throat'->'may include'->'laryngoscopy')->"
                        "Result('laryngitis').\n\n"

                        "--- SAMPLE END ---"
                    )
                }
            ]

            

            # output = self.graphrag.chat_without_stream_for_socre_feedback_Reasoning_Path_api(user_question, final_answer, evidence_list)
            # output = self.graphrag.chat_without_stream_for_socre_feedback_Reasoning_Path_direct_api(user_question, final_answer, evidence_list)
            # output = self.graphrag.chat_without_stream_for_socre_feedback_Reasoning_Path_Keywords_api(user_question, final_answer, evidence_list, keywords_list)
            # print(f"{len(triple)}{json.dumps(triple, indent=2)}")
            # assert False
            # output = self.graphrag.chat_without_stream_for_socre_feedback_Reasoning_Path_Reverse_api(user_question, final_answer, triple, keywords_list)
            output = self.graphrag.chat_without_stream_for_socre_feedback_Reasoning_Path_final_api(user_question, final_answer, sorted_triples, keywords_list)

            output = self.graphrag.chat_without_stream_for_socre_feedback_Reasoning_Path_error_test_api(user_question, final_answer, sorted_triples, keywords_list)

            # output = self.graphrag.chat_without_stream_answer_check_api_qwen3(user_question, final_answer, answer_str)

            # self.graphrag.chat_without_stream_for_redundant_relationship_v3_api_qwen() # 处理冗余



            
            print(f"----output----\n{output}")

    def sort_triples_by_entity(self, triples):
        # 定义排序的 key 函数
        def get_sort_key(triple):
            if triple[3] == '->':
                return triple[0]  # 使用实体1排序
            else:  # '<-'
                return triple[2]  # 使用实体2排序
        # 使用 sorted() 并基于 get_sort_key 排序
        return sorted(triples, key=get_sort_key)
    
    def iteration_redundancy_statistics(self,):
        # iteration 从1开始，计算与前一代的差距
        # if self.args.dataset_name == 'rgb':
        #     with open(f"/home/zhangyz/RAG_2025/logs/rgb_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_rgb_{int(self.args.iteration)-1}.json", 'r', encoding='utf-8') as f:
        #         retriver_data_last = json.load(f)
        #     with open(f"/home/zhangyz/RAG_2025/logs/rgb_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_rgb_{self.args.iteration}.json", 'r', encoding='utf-8') as f:
        #         retriver_data = json.load(f)

        with open(f"/home/zhangyz/RAG_2025/logs/{self.args.dataset_name}_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_{self.args.dataset_name}_{int(self.args.iteration)-1}.json", 'r', encoding='utf-8') as f:
            retriver_data_last = json.load(f)
        with open(f"/home/zhangyz/RAG_2025/logs/{self.args.dataset_name}_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_{self.args.dataset_name}_{self.args.iteration}.json", 'r', encoding='utf-8') as f:
            retriver_data = json.load(f)

        logger = Logger(f"./a_logs/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}_iteration_redundancy_{self.args.iteration}") # only_relationship_

        seen_last_num = []
        redundancy_reduction = []
        redundancy_reduction_rate = []

        for index in range(len(retriver_data_last)):
            # if index > 2:
            #     break
            one_hop_sentence_2d = []
            one_hop_sentence_list = []
            one_hop_sentence_str = ""
            seen_last = set()
            # print(f"filtered_triple_3d: {json.dumps(filtered_triple_3d, indent=2)}")
            sentence_id = 0
            for path_triple in retriver_data_last[index]['filtered_triple_3d']:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                    if sentence not in seen_last:
                        seen_last.add(sentence)
                        # one_hop_sentence_2d.append(triple)
                        # one_hop_sentence_list.append(sentence)
                        # one_hop_sentence_str = one_hop_sentence_str + str(sentence_id) + ': ' + sentence + '\n'
                        # sentence_id += 1

            seen = set()
            sentence_id = 0
            for path_triple in retriver_data[index]['filtered_triple_3d']:
                for triple in path_triple:
                    sentence = ""
                    if triple[3] == '->':
                        sentence = triple[0]+' '+triple[1].replace("_"," ")+' '+triple[2]
                    elif triple[3] == '<-':
                        sentence = triple[2]+' '+triple[1].replace("_"," ")+' '+triple[0]
                    if sentence not in seen:
                        seen.add(sentence)

            seen_last_str = ''
            for s in seen_last:
                seen_last_str += f'{s}\n'
            seen_str = ''
            for s in seen:
                seen_str += f'{s}\n'
            diff = seen_last - seen

            diff_str = ''
            for s in diff_str:
                diff_str += f'{s}\n'
            seen_last_num.append(len(seen_last))
            redundancy_reduction.append(len(diff))
            redundancy_reduction_rate.append(len(diff)/len(seen_last))

            logger.log(f"--------retriver one hop last {len(seen_last)}------------\n{json.dumps(list(seen_last), indent=2)}")
            logger.log(f"--------retriver one hop {len(seen)}------------\n{json.dumps(list(seen), indent=2)}")
            logger.log(f"--------diff(in last not in now) {len(diff)}------------\n{json.dumps(list(diff), indent=2)}")
            logger.log(f"--------redundancy reduction------------\n{len(diff)/len(seen_last)}")
        logger.log(f"\n\n-----end------\n--------retriver one hop last avg------------\n{sum(seen_last_num)/len(seen_last_num)}")
        logger.log(f"--------redundancy reduction avg------------\n{sum(redundancy_reduction)/len(redundancy_reduction)}")
        logger.log(f"--------redundancy reduction rate avg------------\n{sum(redundancy_reduction_rate)/len(redundancy_reduction_rate)}")
        logger.log(f"--------redundancy reduction rate avg 2 ------------\n{sum(redundancy_reduction)/sum(seen_last_num)}")
    
    def iteration_score_statistics(self,):
        # python kg_modify.py --dataset_name rgb --llm qwen3_32b --graphdb nebulagraph --space_name rgb_zyz --option iteration_score_statistics --llmbackend api --iteration 9
        logger = Logger(f"./a_logs/{self.args.llmbackend}_{self.llm.model_name}_{self.args.option}_{self.args.dataset_name}") # only_relationship_
        for index in range(self.args.iteration):
            with open(f"/home/zhangyz/RAG_2025/logs/{self.args.dataset_name}_redundancy/{self.args.space_name}_{index}.json", 'r', encoding='utf-8') as f:
                triplets_score = json.load(f)
            
            score_avg = []

            for key, value in triplets_score.items():
                # 获取 triplet 字段中的三个短语
                if value["score"] > 200:
                    score_avg.append(200)
                elif value["score"] > 0:
                    score_avg.append(value["score"])
            logger.log(f"--------score avg {index}------------{sum(score_avg)/len(score_avg)}")


    def iteration_error_statistics(self,):
        # python kg_modify.py --dataset_name multihop --llm qwen3_32b --graphdb nebulagraph --space_name multihop_zyz --option iteration_error_statistics --llmbackend api --iteration 0
        # 找错的问题，也可以找两轮之间的错题
        with open(f"/home/zhangyz/RAG_2025/logs/{self.args.dataset_name}_redundancy/Meta-Llama-3-8B-Instruct_kg_modify_llama_reproduce_forword_{self.args.dataset_name}_{self.args.iteration}.json", 'r', encoding='utf-8') as f:
            retriver_data = json.load(f)
        index_list = []
        for index in range(len(retriver_data)):
            # if index > 2:
            #     break
            if not retriver_data[index]['label']:
                index_list.append(index)
        print(index_list)

    def test1(self,):
        print("test1")
        # python kg_modify.py --dataset_name rgb_int --llm qwen3_32b --graphdb nebulagraph --space_name integrationrgb --option test --llmbackend api
        # get triplets from graph db
        print("start get triplets")
        all_triplets = self.graph_database.get_triplets()
        self.dataset
        index = 0
        import json
        my_dict = {}
        for item in all_triplets:
            assert len(item) == 3
            result = " ".join(map(str, item))
            my_dict[result] = {}
            my_dict[result]["triplet"] = item
            my_dict[result]["score"] = 100
        with open(f"./logs/triplets/{self.args.space_name}.json", "w", encoding="utf-8") as file:
            json.dump(my_dict, file, ensure_ascii=False, indent=4)

    def short_cut_test(self):
        triplets = {
            "three_hop_path": [
                { "source_node": "Tim Cook", "relationship": "is the CEO of", "target_node": "Apple Inc"},
                { "source_node": "Apple Inc", "relationship": "developed", "target_node": "Apple Vision Pro"},
                { "source_node": "Apple Vision Pro", "relationship": "is a type of", "target_node": "Spatial Computer"}
            ]
        }
        triplets2 = {
            "three_hop_path": [
                { "source_node": "Alice", "relationship": "joined community", "target_node": "VIP Skincare Group"},
                { "source_node": "VIP Skincare Group", "relationship": "participated in", "target_node": "New Serum Launch Event"},
                { "source_node": "New Serum Launch Event", "relationship": "promotes", "target_node": "Revitalizing Serum X"}
            ]
        }
        output = self.graphrag.chat_without_stream_short_cut(triplets)
        print(json.dumps(output, indent = 2))
        output2 = self.graphrag.chat_without_stream_short_cut(triplets2)
        print(json.dumps(output2, indent = 2))

    def short_cut_for_graphdatabase(self):
        pass

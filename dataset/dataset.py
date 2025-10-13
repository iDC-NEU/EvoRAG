'''
Author: fzb0316 fzb0316@163.com
Date: 2024-10-18 14:57:50
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-04 16:04:32
FilePath: /BigModel/RAGWebUi_demo/dataset/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from pathlib import Path
import json
from utils.file_util import file_exist
from dataset.Hotpotqa_process import Hotpot_dataset
from dataset.Concurrent_process import Concurrent_dataset


RGB_PATH = os.path.join(Path(__file__).parent, "rgb", "en.json")
RGB_PATH_INT = os.path.join(Path(__file__).parent, "rgb", "en_int.json")
RGB_PATH_FACT = os.path.join(Path(__file__).parent, "rgb", "en_fact.json")
# RGB_PATH = os.path.join(Path(__file__).parent, "rgb", "en_refine.json")
MULTIHOP_PATH = os.path.join(Path(__file__).parent, "multihop", "dataset", "MultiHopRAG.json")
MULTIHOP_CONTEXT_PATH = os.path.join(Path(__file__).parent, "multihop", "dataset", "corpus.json")
HOTPOTQA600_PATH = os.path.join(Path(__file__).parent, "hotpotqa_graph", "hotpotqa_600.json")
MULTIHOP_INFERENCE_PATH = os.path.join(Path(__file__).parent, "multihop", "multihop_inference.json")\

DRAGONBALL_PATH = os.path.join(Path(__file__).parent, "dragonball", "dragonball_queries.jsonl")
DRAGONBALL_CONTEXT_PATH = os.path.join(Path(__file__).parent, "dragonball", "dragonball_docs.jsonl")
METAPATH_HOP1_PATH = os.path.join(Path(__file__).parent, "metaqa", "hop1_qa_dev.txt")
METAPATH_HOP2_PATH = os.path.join(Path(__file__).parent, "metaqa", "hop2_qa_dev.txt")
METAPATH_HOP3_PATH = os.path.join(Path(__file__).parent, "metaqa", "hop3_qa_dev.txt")

class Dataset:
    def __init__(self, name : str):
        self.dataset_name = name
        self.query = []
        self.answer = []
        self.corpus = []
        self.get_data()


    def get_data(self):
        # print(f"-----{self.dataset_name}-------")
        if self.dataset_name == 'rgb':
            rgb_data_path = RGB_PATH
            assert file_exist(rgb_data_path), f"File {rgb_data_path} does not exist"
            with open(rgb_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # print(json.loads(line).keys())
                    self.query.append(json.loads(line)['query'])
                    # self.answer.append(json.loads(line)['answer'])
                    # answer = json.loads(line)['answer']
                    # if len(answer) == 1 and isinstance(answer[0], list):
                    #     self.answer.append(answer[0])
                    # else:
                    #     self.answer.append(answer)
                    self.answer.append(json.loads(line)['answer'])
        elif self.dataset_name == 'rgb_int':
            rgb_data_path = RGB_PATH_INT
            assert file_exist(rgb_data_path), f"File {rgb_data_path} does not exist"
            with open(rgb_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.query.append(json.loads(line)['query'])
                    self.answer.append(json.loads(line)['answer'])
        elif self.dataset_name == 'rgb_fact':
            rgb_data_path = RGB_PATH_FACT
            assert file_exist(rgb_data_path), f"File {rgb_data_path} does not exist"
            with open(rgb_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.query.append(json.loads(line)['query'])
                    self.answer.append(json.loads(line)['answer'])
        elif self.dataset_name == 'multihop':
            multihop_data_path = MULTIHOP_PATH
            # multihop_data_path = os.path.join(Path(__file__).parent, "multihop", "dataset", "process.json")
            assert file_exist(multihop_data_path), f"File {multihop_data_path} does not exist"
            with open(multihop_data_path, 'r', encoding='utf-8') as f:
                # for line in f:
                #     self.query.append(json.loads(line)['query'])
                #     self.answer.append(json.loads(line)['answer'])
                data = json.load(f)
            inference_query = []
            for item in data:
                if item['question_type'] == "inference_query":# 816
                    self.query.append(item['query'])
                    self.answer.append(item['answer'])
            #         item_dict = {}
            #         item_dict['query'] = item['query']
            #         item_dict['answer'] = item['answer']
            #         inference_query.append(item_dict)

            # with open(MULTIHOP_INFERENCE_PATH, "w", encoding="utf-8") as f:
            #     f.write(json.dumps(inference_query, indent=2))
            

        elif self.dataset_name == "hotpotqa":
            data = Hotpot_dataset()
            self.query = data.query
            self.answer = data.answer
            
        elif self.dataset_name == "hotpotqa600":
            with open(HOTPOTQA600_PATH , 'r', encoding='utf-8') as f:
                # for line in f:
                #     self.query.append(json.loads(line)['query'])
                #     self.answer.append(json.loads(line)['answer'])
                data = json.load(f)
            for item in data:
                self.query.append(item['question'])
                self.answer.append(item['answer'])
        elif self.dataset_name == "concurrentqa":
            data = Concurrent_dataset()
            self.query = data.query
            self.answer = data.answer
        elif self.dataset_name == "dragonball":
            language = 'en'
            # query_type = "Summary Question"
            query_type = "Summarization Question"
            with open(DRAGONBALL_PATH, "r") as file:
                data = [json.loads(line.strip()) for line in file if line.strip()]
            for item in data:

                domain_tmp = item["domain"]
                language_tmp = item["language"]
                query_type_tmp = item["query"]["query_type"]

                if language is not None and language != language_tmp:
                    continue
                if query_type is not None and query_type != query_type_tmp:
                    continue

                question = item["query"]["content"]
                answer = item["ground_truth"]["content"]

                assert isinstance(question, str)
                assert isinstance(answer, str)

                self.query.append(question)
                self.answer.append(answer)
                
                # languages.append(language_tmp)
                # domains.append(domain_tmp)
                # question_types.append(query_type_tmp)
            print(f"-query num--{len(self.query)}----")
        elif self.dataset_name == "metaqa":
            q_hop1 = []
            a_hop1 = []
            with open(METAPATH_HOP1_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        question, answers = line.split('\t', 1)
                        # 去除问题中的[]
                        question = question.replace('[', '').replace(']', '')
                        # 分割答案
                        answer_list = [ans.strip() for ans in answers.split('|')]
                        q_hop1.append(question)
                        a_hop1.append(answer_list)
            q_hop2 = []
            a_hop2 = []
            with open(METAPATH_HOP2_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        question, answers = line.split('\t', 1)
                        # 去除问题中的[]
                        question = question.replace('[', '').replace(']', '')
                        # 分割答案
                        answer_list = [ans.strip() for ans in answers.split('|')]
                        q_hop2.append(question)
                        a_hop2.append(answer_list)
            q_hop3 = []
            a_hop3 = []
            with open(METAPATH_HOP3_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        question, answers = line.split('\t', 1)
                        # 去除问题中的[]
                        question = question.replace('[', '').replace(']', '')
                        # 分割答案
                        answer_list = [ans.strip() for ans in answers.split('|')]
                        q_hop3.append(question)
                        a_hop3.append(answer_list)
            # print(q_hop1[:10])
            # print(a_hop1[:10])
            
            # 根据答案个数进行排序
            # sorted_pairs_hop1 = sorted(zip(q_hop1, a_hop1), key=lambda x: len(x[1]))
            # # 解压为两个排序后的列表
            # sorted_question_hop1, sorted_answer_list_hop1 = zip(*sorted_pairs_hop1)
            # # 转换回 list
            # sorted_q_hop1 = list(sorted_question_hop1)
            # sorted_a_hop1 = list(sorted_answer_list_hop1)
            
            # sorted_pairs_hop2 = sorted(zip(q_hop2, a_hop2), key=lambda x: len(x[1]))
            # sorted_question_hop2, sorted_answer_list_hop2 = zip(*sorted_pairs_hop2)
            # sorted_q_hop2 = list(sorted_question_hop2)
            # sorted_a_hop2 = list(sorted_answer_list_hop2)
            
            # sorted_pairs_hop3 = sorted(zip(q_hop3, a_hop3), key=lambda x: len(x[1]))
            # sorted_question_hop3, sorted_answer_list_hop3 = zip(*sorted_pairs_hop3)
            # sorted_q_hop3 = list(sorted_question_hop3)
            # sorted_a_hop3 = list(sorted_answer_list_hop3)
            
            # self.query.extend(sorted_q_hop1[:200])
            # self.query.extend(sorted_q_hop2[:200])
            self.query.extend(q_hop1[:300])
            self.query.extend(q_hop2[:300])
            # self.query.extend(q_hop3[:100])
            
            # self.answer.extend(sorted_a_hop1[:200])
            # self.answer.extend(sorted_a_hop2[:200])
            self.answer.extend(a_hop1[:300])
            self.answer.extend(a_hop2[:300])
            # self.answer.extend(a_hop3[:100])
            # assert False
            

        else:
            raise ValueError("Invalid dataset name")

    def get_corpus(self, option : str = "positive"):
        if self.dataset_name == 'rgb':
            rgb_data_path = RGB_PATH
            assert file_exist(rgb_data_path), f"File {rgb_data_path} does not exist"
            with open(rgb_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # print(json.loads(line).keys())
                    if option == "positive":
                        self.corpus.append(json.loads(line)['positive'])
                    elif option == "negative":
                        self.corpus.append(json.loads(line)['negative'])
                    elif option == "full":
                        self.corpus.append(json.loads(line)['positive'])
                        self.corpus.append(json.loads(line)['negative'])
                    else:
                        raise ValueError("Invalid option")
        elif self.dataset_name == 'multihop':
            multihop_data_path = MULTIHOP_PATH
            multihop_context_path = MULTIHOP_CONTEXT_PATH
            assert file_exist(multihop_data_path), f"File {multihop_data_path} does not exist"
            assert file_exist(multihop_context_path), f"File {multihop_context_path} does not exist"
            context_dict = {}
            with open(multihop_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(multihop_context_path, 'r', encoding='utf-8') as f:
                context = json.load(f)
                for item in context:
                    context_dict[item['url']] = item['body'] #这块谁写的？能跑起来吗？
            for item in data:
                if item['question_type'] == "inference_query":# 816
                    context_list = []
                    for evidence in item['evidence_list']:
                        if evidence['url'] in context_dict:
                            context_list.append(context_dict[evidence['url']])
                        else:
                            print("the evidence url {evidence.url} not exist")  # 或者处理键不存在的情况
                    self.corpus.append(context_list)
        elif self.dataset_name == "hotpotqa600":
            with open(HOTPOTQA600_PATH , 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                context_list = []
                for paragraph in item['context']:
                    sentence_tmp = ""
                    for sentence in paragraph[1]:
                        sentence_tmp += sentence
                    context_list.append(sentence_tmp)
                    self.corpus.append(context_list)
        elif self.dataset_name == 'dragonball':# 这个现在有问题，不是按照每个问题对应的文本块加载的，而是加载了所有的英文问题
            language = 'en'
            with open(DRAGONBALL_CONTEXT_PATH, "r") as file:
                corpus = [json.loads(line.strip()) for line in file if line.strip()]
            for chunk in corpus:
                if language is not None and language != chunk["language"]:
                    continue
                self.corpus.append(chunk["content"])
                

if __name__ == "__main__":
    Dataset("metaqa")
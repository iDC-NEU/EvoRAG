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
    Dataset("multihop")
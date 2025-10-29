from llmragenv.LLM.llm_factory import ClientFactory
from database.graph.graph_dbfactory import GraphDBFactory
from dataset.dataset import Dataset
from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv
from chat.chat_graphrag import ChatGraphRAG

import argparse
import json
import os
import numpy as np
from pathlib import Path



RGB_PATH = os.path.join(Path(__file__).parent, "rgb", "similar_questions.json")
MULTIHOP_PATH = os.path.join(Path(__file__).parent, "multihop", "dataset", "similar_questions.json")


RGB_PATH_2 = os.path.join(Path(__file__).parent, "rgb", "field_questions")
MULTIHOP_PATH_2 = os.path.join(Path(__file__).parent, "multihop", "dataset", "field_questions")


generate_question_prompt = ( # 同答案不同问题
    "Using the provided question, answer, and two background information paragraphs, generate {num} new questions that approach the answer from different perspectives or rephrase the original question. Follow these guidelines:\n"
    "1. The answers to these questions must align with the original question's answer.\n"
    "2. Please do not directly include the answer in the generated questions.\n"
    "3. All questions must be directly supported by facts from the provided context.\n"
    "Format your output strictly as shown in the example below. Do not include any additional explanations or content:\n"
    "Example:\n"
    "Question: What is the main focus of the discussion?\n"
    "Question: What topic is being prioritized in the meeting?\n\n"
    "###\n"
    "Question: {query}\n"
    "Answer: {answer}\n"
    "Context: {context}\n"
)

generate_question_prompt_1 = ( # 同答案不同问题
    "Using the provided question, answer, and two background information paragraphs, generate {num} new questions that approach the answer from different perspectives or rephrase the original question. Follow these guidelines:\n"
    "1. The answers to these questions must align with the original question's answer.\n"
    "2. Please do not directly include the answer in the generated questions\n"
    "Format your output strictly as shown in the example below. Do not include any additional explanations or content:\n"
    "Example:\n"
    "Question: What is the main focus of the discussion?\n"
    "Question: What topic is being prioritized in the meeting?\n\n"
    "###\n"
    "Question: {query}\n"
    "Answer: {answer}\n"
)




generate_question_prompt_2 = ( # 不同答案不同问题
    "Using the provided question, answer, and two background information paragraphs, generate 5 new questions that are similar to the original question. Follow these guidelines:\n"
    "1. The expected answers for these questions must be either YES, NO, or a few words.\n"
    "2. All questions and answers must be directly supported by facts from the provided context.\n"
    "3. If the answer consists of a few words, provide multiple possible variations of the answer to cover all potential responses.\n"
    "Format your output strictly as shown in the example below. Do not include any additional explanations or content:\n"
    "Example:\n"
    "Question: What is the primary topic of the discussion?\n"
    "Answer: Budget planning ;;; Financial planning ;;; Planning the budget\n"
    "Question: What time does the event start?\n"
    "Answer: 10:00 AM ;;; 10 AM ;;; Ten in the morning\n"
    "###\n"
    "Question: {query}\n"
    "Answer: {answer}\n"
    "Context: {context}\n"
)

class Generate_dataset:
    def __init__(self, args):
        self.args = args
        # try:
        self.llm = ClientFactory(model_name=args.llm, llmbackend=args.llmbackend).get_client()
        # print("________________1______________________")
        self.graph_db = GraphDBFactory(args.graphdb).get_graphdb(space_name=args.space_name)
        # print("________________2______________________")
        self.dataset = Dataset(args.dataset_name)
        self.dataset.get_corpus()
        # print("________________3______________________")
        self.embed_model = EmbeddingEnv("BAAI/bge-large-en-v1.5")
        # print("________________4______________________")
        self.graphrag = ChatGraphRAG(self.llm, self.graph_db)
        # print("________________5______________________")
        self.retriver = self.graphrag.retriver_graph
        # print("________________6______________________")
        # except ImportError as e:
        # print("________________7______________________")
            # raise ImportError(f"Failed to init Class: {e}") from e
        # print("________________8______________________")
        # print("________________9______________________")
        self.PATH = ""
        self.PATH_2 = ""
        self.dataset_name = args.dataset_name
        self.num = 5 #生成相似问题数量
        if args.dataset_name == "rgb":
            self.PATH = RGB_PATH
            self.PATH_2 = RGB_PATH_2
        elif args.dataset_name == "multihop":
            self.PATH = MULTIHOP_PATH
            self.PATH_2 = MULTIHOP_PATH_2
        # self.read_similar_question() 

    def parse_response(self, response):
        import re
        pattern = r"Question:\s*(.*)"
        # response.strip().split('\n')
        extracted_questions = [re.search(pattern, q).group(1).strip() for q in response.strip().split('\n')]
        return extracted_questions
    
    def parse_response2(self, response):
        import re
        pattern_q = r"Question:\s*(.*)"
        pattern_a = r"Answer:\s*(.*)"
        
        response_list = response.strip().split('\n')
        extracted_questions = []
        extracted_answers = []

        for i in range(0, len(response_list), 2):
            if i + 1 >= len(response_list):  # 确保不会越界
                break

            question_match = re.search(pattern_q, response_list[i].strip())
            answer_match = re.search(pattern_a, response_list[i + 1].strip())

            if question_match and answer_match:
                extracted_questions.append(question_match.group(1).strip())
                
                answer_str = answer_match.group(1).strip()
                extracted_answers.append([ans.strip() for ans in answer_str.split(";;;")])
        
        return extracted_questions, extracted_answers

    def parse_response_2(self, response):
        import re
        pattern_q = r"Question:\s*(.*)"
        pattern_a = r"Answer:\s*(.*)"
        
        response_list = response.strip().split('\n')
        extracted_questions = []
        extracted_answers = []

        for i in range(0, len(response_list), 2):
            if i + 1 >= len(response_list):  # 确保不会越界
                break

            try:
                question_match = re.search(pattern_q, response_list[i].strip())
                answer_match = re.search(pattern_a, response_list[i + 1].strip())

                if question_match and answer_match:
                    answer_str = answer_match.group(1).strip()
                    extracted_questions.append(question_match.group(1).strip())
                    extracted_answers.append([ans.strip() for ans in answer_str.split(";;;")])
                else:
                    raise ValueError(f"parse response fail: {response_list[i]} or {response_list[i+1]}")

            except Exception as e:
                print(f"parse response fail ( {i//2+1} group): {e}")
        
        return extracted_questions, extracted_answers 

    
    def calculate_similarity(self, embedding1, embedding2):# 可以批量计算一对多相似度
        # 计算向量的点积
        dot_product = np.dot(embedding1, embedding2)

        # 计算向量的欧几里得范数（长度）
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)

        similarity_rounded = round(similarity, 6)

        return similarity_rounded # [-1,1]
    
    # 嵌入相似度大于50%，实体重合度大于50%，不检查三元组
    def check_similarity(self, query, query_embedding, question):
        label = False
        embedding = self.embed_model.get_embedding(question)
        similarity_score = self.calculate_similarity(query_embedding, embedding)
        entities = self.retriver.extract_keywords_with_embedding_find_entity(query,12) 
        entities_sim = self.retriver.extract_keywords_with_embedding_find_entity(question,12) 
        # print(f"entities: {entities}")
        # print(f"entities_sim: {entities_sim}")
        entities_set = set(entities)
        entities_sim_set = set(entities_sim)
        common_elements = entities_set & entities_sim_set
        common_elements_count = len(common_elements)
        query_entities_length = len(entities)
        rito = common_elements_count / query_entities_length
        # print(f"similarity_score:{similarity_score },rito:{rito}")

        if similarity_score >= 0.5 and rito >= 0.5:
            label = True
        elif rito >= 0.70:
            label = True
        return label

    # 生成同答案不同问题
    # 3次迭代，第一次第二次给context（最多3个），最后一次只给问题和答案并且不做检查
    # 生成问题检查嵌入相似、实体交集、不重复,5个
    # current 是已经成功生成的问题数， num是需要生成的问题数， iteration 是当前迭代次数， max_iteration 最大迭代次数
    def generate_similar_question(self, query, answer, context, current = 0, num = 5, iteration = 0, max_iteration = 3):
        query_embedding = self.embed_model.get_embedding(query)
        # print(f"generate_similar_question iteration: {iteration}")
        if iteration < max_iteration - 1:
            prompt = generate_question_prompt.format(num = str(num), query = query, answer = answer, context = context[:3])
        else:
            prompt = generate_question_prompt_1.format(num = str(num), query = query, answer = answer)
        response = self.llm.chat_with_ai(prompt)
        question_list = self.parse_response(response)
        # print(f"question_list {question_list}")
        res_list = []
        if iteration == max_iteration - 1:
            res_list += question_list[:num-current]
        else:
            for question in question_list:
                if self.check_similarity(query, query_embedding, question) and question not in res_list:
                    if current < num:
                        res_list.append(question)
                        current += 1
                    else:
                        break
            if current < num:
                res = self.generate_similar_question(query, answer, context, current, num, iteration+1)
                res_list += res[:num-current]
        
        return res_list

    def generate_similar_questions(self, ):
        similar_questions = []
        for i in range(5):
            similar_questions.append([])
        
        for i in range(len(self.dataset.query)):
            # if i >= 2 :
            #     break
            res = self.generate_similar_question(self.dataset.query[i], self.dataset.answer[i], self.dataset.corpus[i], current = 0, num = 5, iteration = 0, max_iteration = 3)
            for j in range(5):
                similar_questions[j].append(res[j])        

        if not os.path.exists(self.PATH):
            with open(self.PATH, 'w', encoding='utf-8') as f:
                json.dump(similar_questions, f, ensure_ascii=False, indent=4)
            print(f"generate similar questions save in  {self.PATH}")
        else:
            print(f"{self.PATH} already exists")

    # 生成领域问题 不同答案不同问题
    def generate_field_question(self, query, answer, context, current = 0, num = 5, iteration = 0, max_iteration = 3):
        query_embedding = self.embed_model.get_embedding(query)
        # print(f"generate_similar_question iteration: {iteration}")

        prompt = generate_question_prompt_2.format(num = str(num), query = query, answer = answer, context = context[:5])
        response = self.llm.chat_with_ai(prompt)
        question_list, answer_list = self.parse_response_2(response)
        # print(f"question_list {question_list}")
        res_question_list = []
        res_answer_list = []
        if iteration == max_iteration - 1:
            res_question_list += question_list[:num-current]
            res_answer_list += answer_list[:num-current]
        else:
            for question, generate_answer in zip(question_list, answer_list):
                if self.check_similarity(query, query_embedding, question) and question not in res_question_list:
                    if current < num:
                        res_question_list.append(question)
                        res_answer_list.append(generate_answer)
                        current += 1
                    else:
                        break
            if current < num:
                res_question, res_answer = self.generate_field_question(query, answer, context, current, num, iteration+1)
                res_question_list += res_question[:num-current]
                res_answer_list += res_answer[:num-current]
        
        return res_question_list, res_answer_list
    

    def generate_field_questions(self):
        file_paths = [f"{self.PATH_2}_{j+1}.json" for j in range(self.num)]  # 生成5个不同的文件路径

        for path in file_paths:
            if os.path.exists(path):
                print(f"field question already exists: {path} skip")
                continue          
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    pass
                print(f"mkfile field question success {path}")
            except IOError as e:
                print(f"mkfile field question fail {path}: {e}")

        for i in range(len(self.dataset.query)):
            if i >=5:
                break
            res_question, res_answer = self.generate_field_question(
                self.dataset.query[i], 
                self.dataset.answer[i], 
                self.dataset.corpus[i], 
                current=0, num=self.num, iteration=0, max_iteration=3
            )

            for j in range(5):
                data = {
                    "query": res_question[j],
                    "answer": res_answer[j]
                }

                with open(file_paths[j], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"Generated field questions saved in: {', '.join(file_paths)}")
    
    def read_similar_question(self,):
        if os.path.exists(self.PATH):
            with open(self.PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.dataset.query = data[0]
            # return data
        else:
            print(f"{self.PATH} not exist")
            self.dataset.query = []
            # return None

    def read_field_question(self, num = 0 ): # 返回指定一代问题
        if self.dataset_name == 'rgb':
            rgb_file_paths = [f"{self.PATH_2}_{j+1}.json" for j in range(self.num)]

            rgb_data_path = rgb_file_paths[num]
            if os.path.exists(rgb_data_path):
                query = []
                answer = []
                with open(rgb_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        query.append(json.loads(line)['query'])
                        answer.append(json.loads(line)['answer'])
                self.dataset.query = query
                self.dataset.answer = answer
                print(f"read field question from {rgb_data_path}")
            else:
                print(f"{rgb_data_path} not exist")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLMRag Workload")
    parser.add_argument("--dataset_name", type=str, help="dataset name", default='rgb')
    parser.add_argument("--llm", type=str, help="llm env (e.g., qwen0.5b, llama2:7b, llama2:13b, llama2:70b)", default='llama3.3')
    parser.add_argument("--graphdb", type=str, help="graph database baskend (e.g., neo4j, ) ", default='nebulagraph')
    parser.add_argument("--space_name", type=str, help="graph database space name (e.g., rgb, ) ", default='rgb_zyz')
    parser.add_argument("--llmbackend", type=str, help="openai or llama_index", default="llama_index")
    args = parser.parse_args()
    ob = Generate_dataset(args)
    ob.generate_field_questions()
    tmp = ob.read_field_question(0)
    print(tmp)

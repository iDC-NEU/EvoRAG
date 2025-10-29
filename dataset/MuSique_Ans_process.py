
import json
import os
from pathlib import Path
import random


MUSIQUE_ANS_PATH = os.path.join(Path(__file__).parent, "musique", "musique_ans_v1.0_dev.jsonl")
MUSIQUE_ANS_RANDOM_1200_PATH = os.path.join(Path(__file__).parent, "musique", "musique_ans_v1.0_dev_1200.jsonl")

class Musique_ans_dataset:
    def __init__(self):
        self.corpus = []
        self.query = []
        self.answer = []
        # self.read_context(CONCURRENTQA_CONTEXT_PATH)
        self.qa = {}
        # self.read_qa(CONCURRENTQA_PATH)
        self.random_1200()
        self.read()

    def random_1200(self):
        if os.path.exists(MUSIQUE_ANS_RANDOM_1200_PATH):
            return
        
        all_data = []
        with open(MUSIQUE_ANS_PATH, 'r', encoding='utf-8') as file:
            count = 0
            for line in file:
                # if count >= 1:
                #     break
                # count += 1
                # print(json.loads(line).keys())
                all_data.append(json.loads(line))

        if len(all_data) < 1200:
            raise ValueError("The number of elements in the raw data is less than 1200")
        
        random_indices = random.sample(range(len(all_data)), 1200)
        self.random_1200 = [all_data[i] for i in random_indices]
        
        with open(MUSIQUE_ANS_RANDOM_1200_PATH, 'w', encoding='utf-8') as file:
            for item in self.random_1200:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
     
            
    def read(self):
        count = 0
        with open(MUSIQUE_ANS_RANDOM_1200_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                dict_item = json.loads(line)
                self.query.append(dict_item["question"])
                # if count == 0:
                #     print(dict_item["answer_aliases"])
                #     print(type(dict_item["answer_aliases"]))
                #     print(dict_item['answer'])

                answer_list = dict_item["answer_aliases"] + [dict_item["answer"]]
                self.answer.append(answer_list)
                # self.answer.append((dict_item["answer_aliases"]).append(dict_item['answer'])) # 值为None，因为append没有返回值
                # 固定20个 text
                merge_count = 0
                paragraph_2_str = ''
                for paragraph_dict in dict_item['paragraphs']: 
                    merge_count += 1
                    paragraph_2_str = paragraph_2_str + ' ' + paragraph_dict['paragraph_text']
                    if merge_count%2 == 0:
                        self.corpus.append(paragraph_2_str.strip())
                        paragraph_2_str = ""
                if paragraph_2_str:
                    self.corpus.append(paragraph_2_str.strip())
            
        #         if count <= 0:
        #             print(self.query[0])
        #             print(self.answer[0])
        #             print(len(self.corpus))
        #             print(json.dumps(self.corpus, indent=2))
        #         count += 1
        # print(f"self.query: {len(self.query)}")
        # print(f"self.answer: {len(self.answer)}")
        # print(f"self.corpus: {len(self.corpus)}")

            
if __name__ == '__main__':
    Musique_ans_dataset()
    
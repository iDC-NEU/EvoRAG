
import json
import os
from pathlib import Path
import random


CONCURRENTQA_CONTEXT_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "corpora", "title2sent_map.json")
HOTPOTQA_FULLWIKI_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_fullwiki_v1.json")
HOTPOTQA_DISTRACTOR_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_distractor_v1.json")
HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "random_1200.json")
HOTPOTQA_DISTRACTOR_RANDOM_600_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "random_600.json")


class Hotpot_dataset:
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
        if os.path.exists(HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH):
            with open(HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH, 'r', encoding='utf-8') as file:
                self.random_1200 = json.load(file)
            return
        
        with open(HOTPOTQA_DISTRACTOR_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file) 
        
        if len(data) < 1200:
            raise ValueError("The number of elements in the raw data is less than 1200")

        random_indices = random.sample(range(len(data)), 1200)
        self.random_1200 = [data[i] for i in random_indices]
        
        with open(HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH, 'w', encoding='utf-8') as file:
            json.dump(self.random_1200, file, indent=2, ensure_ascii=False)

    def read(self):
        # with open(HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH, 'r', encoding='utf-8') as file:
        #     self.random_1200 = json.load(file)
        count = 0
        for item_dict in self.random_1200:
            # count += 1
            # if count >= 3:
            #     break
            self.query.append(item_dict["question"])
            self.answer.append(item_dict["answer"])
            for paragraph in item_dict['context']:
                sentence_tmp = ""
                for sentence in paragraph[1]:
                    sentence_tmp += sentence
                self.corpus.append(sentence_tmp)
        # print(self.query)
        # print(self.answer)
        # print(self.corpus)
            
if __name__ == '__main__':
    Hotpot_dataset()
    
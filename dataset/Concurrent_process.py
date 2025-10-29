import json
import os
from pathlib import Path

CONCURRENTQA_CONTEXT_PATH = os.path.join(Path(__file__).parent, "concurrentqa", "corpora", "title2sent_map.json")
CONCURRENTQA_PATH = os.path.join(Path(__file__).parent, "concurrentqa", "CQA_test_all.json")  
CONCURRENTQA_RETRIEVER_PATH = os.path.join(Path(__file__).parent, "concurrentqa", "Retriever_CQA_test_all_original.json")  

class Concurrent_dataset:
    def __init__(self):
        self.corpus = []
        self.query = []
        self.answer = []
        # self.read_context(CONCURRENTQA_CONTEXT_PATH)
        # self.read_qa(CONCURRENTQA_PATH)
        self.read_retriever(CONCURRENTQA_RETRIEVER_PATH)

    
    def read_context(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            self.corpus = json.load(file)

        titles = list(self.corpus.keys())
        unique_titles = set(titles)

        if len(titles) == len(unique_titles):
            print("All titles are unique.")
        else:
            print("There are duplicate titles.")

        count = 0   
        for key, value in self.corpus.items():
            count += 1
            if count >= 3:
                break
            print(f"ID: {count}")
            print(f"Title: {key}")
            print(f"Sentences: {value}")

    def read_qa(self, path):
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # count += 1
                # if count >= 3:
                #     break
                sp = json.loads(line)['sp']
                question = json.loads(line)['question']
                answer = json.loads(line)['answer']
                for context_dict in sp:
                    context = ""
                    for sentence in context_dict['sents']:
                        context = context + ' ' + sentence
                    self.corpus.append(context)
                self.query.append(question)
                self.answer.append(answer)
                # print(f"question: {question}")
                # print(f"answer: {answer}")
            # for item in self.corpus:
            #     print(f"{item}")
    def read_retriever(self, path):
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # print(json.loads(line).keys())                
                # count += 1
                # if count >= 3:
                #     break
                self.query.append(json.loads(line)['question'])
                self.answer.append(json.loads(line)['answers'])
                for pos_dict in json.loads(line)['pos_paras']:
                    self.corpus.append(pos_dict["text"])
                for neg_dict in json.loads(line)['neg_paras']:
                    self.corpus.append(neg_dict["text"])
                    
            # print(self.query)
            # print(self.answer)
            # print(self.corpus)
                

if __name__ == '__main__':
    Concurrent_dataset()
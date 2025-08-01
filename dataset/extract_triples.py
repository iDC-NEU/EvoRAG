# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain.document_loaders import PyPDFLoader
# from itext2kg import iText2KG
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from pydantic import BaseModel, Field
# import time
import argparse
import json
from typing import Dict, List

from tqdm import tqdm
import time
import multiprocessing
# from data.rgb_info import get_rgb_docs_str
# from utils.base import create_dir, extract_json_str, get_date_now

# import datetime
import os
import re

# from utils.llm_env import LLMEnv, OllamaEnv
from llmragenv.LLM.llm_factory import ClientFactory
from logger import Logger
# from utils.logger import Logger
# from utils.timer import Timer

from dataset.Concurrent_process import Concurrent_dataset
from dataset.Hotpotqa_process import Hotpot_dataset
from dataset.MuSique_Ans_process import Musique_ans_dataset
from dataset.MuSique_Full_process import Musique_full_dataset
from datetime import datetime
from database.graph.nebulagraph.nebulagraph import NebulaDB, NebulaClient

prompt_template_str = """
#### Process
**Identify Named Entities**: Extract entities based on the given entity types, ensuring they appear in the order they are mentioned in the text.
**Establish Triplets**: Form triples with reference to the provided predicates, again in the order they appear in the text.

Your final response should follow this format:

**Output:**
```json
{{
    "entities": # type: Dict
    {{
        "Entity Type": ["entity_name"]
    }},
    "triplets": # type: List
    [
        ["subject", "predicate", "object"]
    ]
}}
```

### Example:
**Entity Types:**
ORGANIZATION
COMPANY
CITY
STATE
COUNTRY
OTHER
PERSON
YEAR
MONTH
DAY
OTHER
QUANTITY
EVENT

**Predicates:**
FOUNDED_BY
HEADQUARTERED_IN
OPERATES_IN
OWNED_BY
ACQUIRED_BY
HAS_EMPLOYEE_COUNT
GENERATED_REVENUE
LISTED_ON
INCORPORATED
HAS_DIVISION
ALIAS
ANNOUNCED
HAS_QUANTITY
AS_OF


**Input:**
Walmart Inc. (formerly Wal-Mart Stores, Inc.) is an American multinational retail corporation that operates a chain of hypermarkets (also called supercenters), discount department stores, and grocery stores in the United States, headquartered in Bentonville, Arkansas.[10] The company was founded by brothers Sam and James "Bud" Walton in nearby Rogers, Arkansas in 1962 and incorporated under Delaware General Corporation Law on October 31, 1969. It also owns and operates Sam's Club retail warehouses.[11][12]

As of October 31, 2022, Walmart has 10,586 stores and clubs in 24 countries, operating under 46 different names.[2][3][4] The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, and as Flipkart Wholesale in India.

**Output:**
```json
{{
"entities": {{
    "COMPANY": ["Walmart Inc.", "Sam's Club", "Flipkart Wholesale"],
    "PERSON": ["Sam Walton", "James 'Bud' Walton"],
    "COUNTRY": ["United States", "Canada", "Mexico", "Central America", "India"],
    "CITY": ["Bentonville", "Rogers"],
    "STATE": ["Arkansas"],
    "DATE": ["1962", "October 31, 1969", "October 31, 2022"],
    "ORGANIZATION": ["Delaware General Corporation Law"]
}},
"triplets": [
    ["Walmart Inc.", "FOUNDED_BY", "Sam Walton"],
    ["Walmart Inc.", "FOUNDED_BY", "James 'Bud' Walton"],
    ["Walmart Inc.", "HEADQUARTERED_IN", "Bentonville, Arkansas"],
    ["Walmart Inc.", "FOUNDED_IN", "1962"],
    ["Walmart Inc.", "INCORPORATED", "October 31, 1969"],
    ["Sam Walton", "FOUNDED", "Walmart Inc."],
    ["James \"Bud\" Walton", "CO-FOUNDED", "Walmart Inc."],
    ["Walmart Inc.", "OWNS", "Sam's Club"],
    ["Flipkart Wholesale", "OWNED_BY", "Walmart Inc."],
    ["Walmart Inc.", "OPERATES_IN", "United States"],
    ["Walmart Inc.", "OPERATES_IN", "Canada"],
    ["Walmart Inc.", "OPERATES_IN", "Mexico"],
    ["Walmart Inc.", "OPERATES_IN", "Central America"],
    ["Walmart Inc.", "OPERATES_IN", "India"]
]
}}
```

### Task:
Your task is to perform Named Entity Recognition (NER) and knowledge graph triplet extraction on the text that follows below.

**Input:**
{context}

**Output:**
"""

def get_date_now():
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_string

def file_exist(path):
    return os.path.exists(path)

def create_dir(path=None):
    if path and not file_exist(path):
        os.makedirs(path, exist_ok=True)
        
def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()
        


class Extract_model:
    def __init__(self, args):
        self.llm = ClientFactory(model_name=args.llm, llmbackend=args.llmbackend ).get_client()
        self.corpus = []
        self.dataset = args.dataset_name
        # self.get_corpus()
        self.triplets = []
        
    def get_corpus(self):
        if self.dataset == "concurrentqa":
            current_dataset = Concurrent_dataset()
            self.corpus = current_dataset.corpus
        elif self.dataset == "hotpotqa":
            current_dataset = Hotpot_dataset()
            self.corpus = current_dataset.corpus
        elif self.dataset == "musique_ans":
            current_dataset = Musique_ans_dataset()
            self.corpus = current_dataset.corpus
        elif self.dataset == "musique_full":
            current_dataset = Musique_full_dataset()
            self.corpus = current_dataset.corpus
            
    def llm_extract(self, context):
        output = None
        retry = 3
        while retry > 0:
            try:
                output = self.llm.chat_with_ai(prompt_template_str.format(context=context))
                output = extract_json_str(output)
                parsed_output = json.loads(output)
                assert "entities" in parsed_output and "triplets" in parsed_output
                return parsed_output
            except Exception as e:
                print(f"JSON format error: {e}")
                retry -= 1  # Decrement the retry counter
        return output
    
    def extract_triplets(self,):

        print(f"process {len(self.corpus)} texts...")

        create_dir(f"./dataset/logs/{self.dataset}")  
        logger = Logger(f"./build_kg/{self.dataset}/build_kg_{get_date_now()}.json")
        start = datetime.now()
        logger.log(f"start time: {start.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        output_path = f"./dataset/logs/{self.dataset}/triplets.json"
        
        # res_list = []
        with open(output_path, "a", encoding="utf-8") as f:  
            for i, text in enumerate(tqdm(self.corpus, desc="Building KG")):
                if i <= 16351:
                    continue
                if i != 0 and i % 100 == 0:
                    current_time = datetime.now()
                    time_diff_current = current_time - start
                    total_seconds_current = time_diff_current.total_seconds()
                    hours_current = int(total_seconds_current // 3600)
                    remaining_seconds_current = total_seconds_current % 3600
                    minutes_current = int(remaining_seconds_current // 60)  
                    print(f"spend time: {hours_current} hours {minutes_current} minutes ")
                    
                output = self.llm_extract(context=text)
                if not output:
                    print(f"-------------------{i} llm extract is None-----------------------")
                else:
                    if isinstance(output, str):
                        logger.log(f"doc_{i}_text {None}")
                        logger.log(f"doc_{i}_entity {None}")
                        logger.log(f"doc_{i}_triplet {None}")
                        continue

                    entity_label = {}
                    if isinstance(output["entities"], List):
                        for entities in output["entities"]:
                            for entity_type, names in entities.items():
                                for name in names:
                                    if not isinstance(name, str) or not isinstance(
                                        entity_type, str
                                    ):
                                        continue
                                    entity_label[name.capitalize()] = entity_type.capitalize()

                    else:
                        assert isinstance(output["entities"], Dict)
                        for entity_type, names in output["entities"].items():
                            for name in names:
                                if not isinstance(name, str) or not isinstance(entity_type, str):
                                    continue
                                entity_label[name.capitalize()] = entity_type.capitalize()

                    triplets = [
                        [
                            phrase.capitalize() if isinstance(phrase, str) else phrase
                            for phrase in triplet
                        ]
                        for triplet in output["triplets"]
                    ]
                    # res_list.extend(triplets)
                    logger.log(f"doc_{i}_text {text}")
                    logger.log(f"doc_{i}_entity {json.dumps(entity_label, indent=4, ensure_ascii=False)}")
                    logger.log(f"doc_{i}_triplet {json.dumps(triplets, indent=4, ensure_ascii=False)}")
                    
                    # 增量写入
                    for triplet in triplets:
                        if len(triplet) != 3:
                            logger.log(f"The length of a triplet is not equal to 3. triple: {triplet}")
                            continue
                        try:
                            entry = {
                                "head": triplet[0],
                                "relationship": triplet[1],
                                "tail": triplet[2]
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            f.flush()
                        except (IOError, json.JSONEncodeError) as e:
                            logger.log(f"write triple fail: {str(e)}. triple: {triplet}")
                            continue
                                  
            
        # with open(f"./dataset/logs/{self.dataset}/triplets.json", "w", encoding="utf-8") as f:
        #     json.dump(res_list, f, indent=4, ensure_ascii=False)  # ensure_ascii=False 支持非ASCII字符
        print(f"triplets have been saved in ./logs/{self.dataset}/triplets.json")
        
        end = datetime.now()
        logger.log(f"end time: {end.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        time_diff = end - start
        total_seconds = time_diff.total_seconds()
        hours = int(total_seconds // 3600)
        remaining_seconds = total_seconds % 3600
        minutes = int(remaining_seconds // 60)  
        logger.log(f"spend time: {hours} hours {minutes} minutes ")

    def read(self,):
        with open("triplets.json", "r", encoding="utf-8") as f:
            loaded_triplets = json.load(f)
            
    def read_triplets(self,):
        read_path = f"./dataset/logs/{self.dataset}/triplets.json"
        with open(read_path, "r", encoding="utf-8") as f:
            count = 0
            for line_num, line in enumerate(f, 1):
                # count += 1
                # if count >=3 :
                #     break
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # yield (
                    #     data["head"],
                    #     data["relationship"],
                    #     data["tail"]
                    # )
                    # print(data)
                    self.triplets.append(data)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"第{line_num}行数据异常: {str(e)}")
                    continue

    def read_triplets_rebuild(self,):
        with open(f"./logs/triplets/{self.dataset}_unchanged_0.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)

        for key, value in triplets_score.items():
            # 获取 triplet 字段中的三个短语
            x, y, z = value["triplet"]
            data = {
                "head": str(x),
                "relationship": str(y),
                "tail": str(z)
            }
            self.triplets.append(data)


    def insert_triple(self, pid, triplets, graph_db: NebulaDB, verbose=False, log=False):
        start_time = time.time()
        # for i, triplet in tqdm(enumerate(triplets), f"insert triplets in {db_names}"):
        for i, triplet in enumerate(triplets):
            if i and i % 10000 == 0 and verbose:
                print(
                    f'processor {pid} insert {i}/{len(triplets)} triplets, cost {time.time() - start_time : .3f}s.'
                )
            # assert triplet["head"] and triplet["relationship"] and triplet["tail"]
            if not (triplet["head"] and triplet["relationship"] and triplet["tail"]):
                continue
            graph_db.upsert_triplet([triplet["head"], triplet["relationship"], triplet["tail"]])
        end_time = time.time()
        print(
            f'pid {pid} insert {len(triplets)} triplets cost {end_time - start_time : .3f}s.'
        )


    def parallel_insert(self, db_name, nproc=1, reuse=False):
        start_time = time.time()
        processes = []
        n_triplets = len(self.triplets)
        if n_triplets < 100:
            nproc = 1
        step = (n_triplets + nproc - 1) // nproc

        print(n_triplets, db_name, nproc, step)
        print(f'\ninsert {n_triplets} triplets in {db_name}, nproc={nproc}')

        nebula_db = NebulaDB(space_name = db_name)
        if not reuse:
            nebula_db.clear()

        # assert(False)

        for i in range(nproc):
            start = i * step
            end = min(start + step, n_triplets)
            print(f'pid {i} take {start}-{end}')
            p = multiprocessing.Process(target=self.insert_triple,
                                        args=(
                                            i,
                                            self.triplets[start:end],
                                            nebula_db,
                                            True,
                                        ))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        end_time = time.time()

        print(f'insert_triple_parallel cost {end_time - start_time:.3f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process some entities and triplets for knowledge extraction."
    )

    parser.add_argument("--llm", type=str, default="llama3.3", help="llm model.")

    parser.add_argument("--dataset_name", type=str, default="concurrentqa", help="dataset")
    parser.add_argument("--llmbackend", type=str, help="openai or llama_index", default="llama_index")
    parser.add_argument("--option", type=str, help="execution way (e.g., extract_triplets, insert_nebulagraph ) ", default='extract_triplets')

    args = parser.parse_args()
    # print(args)

    print(f"Starting to build a knowledge graph")
    
    build = Extract_model(args)
    if args.option == "extract_triplets":
        build.get_corpus()
        build.extract_triplets()
    elif args.option == "insert_nebulagraph":
        client = NebulaClient()
        client.create_space(args.dataset_name)
        build.read_triplets()
        build.parallel_insert(args.dataset_name)
    elif args.option == "rebuild_nebulagraph":
        client = NebulaClient()
        client.create_space(args.dataset_name)
        build.read_triplets_rebuild()
        build.parallel_insert(args.dataset_name)
    # build.read_triplets()


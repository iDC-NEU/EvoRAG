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

from data.rgb_info import get_rgb_docs_str
from utils.base import create_dir, extract_json_str, get_date_now

# from pprint import pprint
# from llama_index.core import (
#     Document,
#     Settings,
# )
from utils.llm_env import LLMEnv, OllamaEnv
from llmragenv.LLM.llm_factory import ClientFactory
from utils.logger import Logger
from utils.timer import Timer

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


def llm_extract(llm: LLMEnv, context):
    retry = 3
    while retry > 0:
        try:
            output = llm.complete(prompt_template_str.format(context=context))
            output = extract_json_str(output)
            parsed_output = json.loads(output)
            assert "entities" in parsed_output and "triplets" in parsed_output
            return parsed_output
        except Exception as e:
            print(f"JSON format error: {e}")
            retry -= 1  # Decrement the retry counter
    return output


def extract_triplets(llm, dataset, texts, pid, start, end):

    texts = texts[start:end]
    print(f"process [{start}:{end}] texts...")

    create_dir(f"./logs/{dataset}")
    logger = Logger(log_name=f"./logs/{dataset}/build_kg_{pid}-{get_date_now()}.json")

    timer = Timer()

    for i, text in enumerate(tqdm(texts, desc="Building KG")):

        with timer.timing(f"doc_{start+i}_extract"):
            output = llm_extract(llm, context=text)

        if isinstance(output, str):
            logger.add(f"doc_{start+i}_text", None)
            logger.add(f"doc_{start+i}_entity", None)
            logger.add(f"doc_{start+i}_triplet", None)
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

        logger.add(f"doc_{start+i}_text", text)
        logger.add(f"doc_{start+i}_entity", entity_label)
        logger.add(f"doc_{start+i}_triplet", triplets)
        logger.update(timer.duration_dict)
        logger.save()

    logger.update(timer.duration_dict)
    logger.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process some entities and triplets for knowledge extraction."
    )

    parser.add_argument("--llm", type=str, default="llama3.1:70b", help="llm model.")

    parser.add_argument(
        "--pid", type=int, required=True, help="Specify the process ID for tracking."
    )

    parser.add_argument(
        "--start", type=int, required=True, help="document start position."
    )

    parser.add_argument("--end", type=int, required=True, help="document end position.")

    parser.add_argument("--data", type=str, default="rgb", help="dataset")

    args = parser.parse_args()
    print(args)

    if args.data == "rgb":
        docs = get_rgb_docs_str()
    elif args.data == "squad":
        from data.squad_info import get_squad_info

        docs, _, _ = get_squad_info()
    elif args.data == "musique":
        from data.musique_info import get_musique_docs

        docs = get_musique_docs(name="train", version="ans", words=1024)
        # from llama_index.core import Document
        # docs = [Document(text=t) for t in docs]
        # text = node.get_content(metadata_mode=MetadataMode.LLM)

    else:
        raise NotImplementedError

    llm = OllamaEnv(args.llm, port=11434 + args.pid)
    llm = ClientFactory(model_name=args.llm, llmbackend=args.llmbackend).get_client()

    print(f"documents has {len(docs)} paragraphs")
    # exit(0)

    # transformations = Settings.transformations
    # from llama_index.core.ingestion import run_transformations
    # nodes = run_transformations(docs, transformations, show_progress=True)
    # print("nodes: ", len(nodes))

    extract_triplets(llm, args.data, docs, args.pid, args.start, args.end)

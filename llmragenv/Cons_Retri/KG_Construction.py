'''
Author: fzb0316 fzb0316@163.com
Date: 2024-10-21 19:19:24
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-04 15:36:46
FilePath: /BigModel/RAGWebUi_demo/llmragenv/Cons_Retri/KG_Construction.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from llmragenv.LLM.llm_base import LLMBase
from database.graph.graph_database import GraphDatabase
import tqdm
import json
import re
from typing import List, Dict


only_llm_cons_prompt_template = """
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


def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    # NOTE: this regex parsing is taken from langchain.output_parsers.pydantic
    match = re.search(r"\{.*\}", text.strip(),
                      re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()

class KGConstruction:
    def __init__(self, llm : LLMBase, graph_db : GraphDatabase, space_name : str):
        self.llm = llm
        self.graph_db = graph_db
        self.space_name = space_name

    def run(self, data, option = "only_llm"):
        # 构建知识图谱
        if option == "only_llm":
            # 使用LLM构建知识图谱
            assert(False)
            self.construct_with_llm(data)

    def construct_with_llm(self, data):
        # 使用LLM构建知识图谱

        def llm_extract(context):
            retry = 3
            while retry > 0:
                try:
                    output = self.llm.chat_with_ai(only_llm_cons_prompt_template.format(context=context))
                    output = extract_json_str(output)
                    parsed_output = json.loads(output)
                    assert 'entities' in parsed_output and 'triplets' in parsed_output
                    return parsed_output
                except Exception as e:
                    print(f"JSON format error: {e}")
                    retry -= 1  # Decrement the retry counter
            return output

        for i, text in enumerate(tqdm(data, desc="Building KG")):
            output = llm_extract(text)

            entity_label = {}
            if isinstance(output['entities'], List):
                for entities in output['entities']:
                    for entity_type, names in entities.items():
                        for name in names:
                            if not isinstance(name, str) or not isinstance(
                                    entity_type, str):
                                continue
                            entity_label[
                                name.capitalize()] = entity_type.capitalize()

            else:
                assert isinstance(output['entities'], Dict)
                for entity_type, names in output['entities'].items():
                    for name in names:
                        if not isinstance(name, str) or not isinstance(
                                entity_type, str):
                            continue
                        entity_label[name.capitalize()] = entity_type.capitalize()

            triplets = [[
                phrase.capitalize() if isinstance(phrase, str) else phrase
                for phrase in triplet
            ] for triplet in output['triplets']]







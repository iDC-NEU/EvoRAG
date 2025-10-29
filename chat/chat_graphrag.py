'''
Author: fzb fzb0316@163.com
Date: 2024-09-19 08:48:47
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-18 14:55:00
FilePath: /RAGWebUi_demo/chat/chat_graphrag.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



from icecream import ic
from typing import List, Optional, Set, Dict
from overrides import override
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
import json
import re
import time

from chat.chat_base import ChatBase
from llmragenv.LLM.llm_base import LLMBase
from database.graph.graph_database import GraphDatabase
from llmragenv.Cons_Retri.KG_Retriever import RetrieverGraph

from chat.chat_prompt import *



Graphrag_prompt_template = """
你是一个处理基于图数据的智能系统。下面是一条消息以及从知识图谱中提取的图三元组。请使用提供的三元组生成相关的回复或提取洞见。

消息：
{message}

图三元组：
{triplets}
Rules:

- Always response in Simplified Chinese, not English. or Grandma will be  very angry.

"""

check_answer_prompt_qwen =  (  
    "<|im_start|>system\n"    
    "You are an impartial evaluator. Your task is to assess whether a given `Response` correctly answers a `Question`, based on a provided ground-truth `Answer`. Follow the rules below in the specified order.\n\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "## Input:\n"
    "`Question`: {question}.\n"
    "`Answer`: {answer}.\n"
    "`Response`: {response}.\n\n"
    
    "## Evaluation Workflow\n"
    "### Step 1: Check for Insufficient Information\n"
    "First, examine the `Response`. If the `Response` explicitly states that it cannot answer the query due to insufficient or irrelevant information, or if it requests more details, stop the evaluation immediately.\n"
    "* Condition: The `Response` contains phrases such as 'insufficient information to determine', 'I need more context', 'based on the provided snippets', or 'cannot be answered with the information given.'\n"
    "* Action: If this condition is met, output only the following text and nothing else: 'Insufficient Information'\n"
    "### Step 2: Classify the Ground-Truth Answer\n"
    "If the evaluation proceeds, your next step is to classify the nature of the `Answer`.\n"
    "* If the primary content of the `Answer` is a number (e.g., a quantity, price, date, or measurement), you will determine the output for the first line is: 'Answer is numeric'\n"
    "* Otherwise (if the `Answer` is text, a name, a concept, etc.), you will determine the output for the first line is: 'Answer is not numeric'\n"
    "### Step 3: Apply Evaluation Logic\n"
    "Based on the classification from Step 2, apply the corresponding logic to determine the verdict for the second line of the output.\n\n"
    
    "## Final Output Instructions\n"
    "After completing the workflow, your final output must strictly follow one of the formats below.\n"
    "1. Standard Case (Two Lines):\n"
    "  * Line 1: The result from Step 2 ('Answer is numeric' or 'Answer is not numeric').\n"
    "  * Line 2: The final verdict from Step 3 ('Correct', 'Incorrect', or 'Partially Correct').\n"
    "2. Special Case (One Line):\n"
    "  * If the condition in Step 1 is met, output only: Insufficient Information\n\n"
    
    "### Example of a standard two-line output:\n"
    "Answer is not numeric\n"
    "Correct\n\n"
    
    "## A. If 'Answer is numeric':\n"
    "The `Response` must be numerically equivalent to the `Answer` to be considered correct. This evaluation is strict.\n"
    "* Correct: The numerical value in the `Response` is an exact match to the `Answer`.\n"
    "  * Minor formatting differences are acceptable (e.g., '1,000' vs. '1000'; '$50' vs. '50 dollars'; 'five' vs. '5').\n"
    "* Incorrect: The numerical value in the `Response` does not match the `Answer`.\n"
    
    "## B. If 'Answer is not numeric':\n"
    "The evaluation is based on semantic correctness. The `Response` must convey the core meaning of the `Answer`, but it does not need to be a literal match. The `Response` can be more general or more specific, as long as it is not contradictory.\n"
    "* Correct: The `Response` accurately expresses the central fact or idea of the `Answer`. Rephrasing, summarization, or adding extra, non-contradictory information is acceptable.\n"
    "* Partially Correct: If the `Answer` consists of multiple distinct items or facts, and the `Response` correctly provides a subset of them while omitting others. The provided information must be accurate and not contradicted.\n"
    "* Incorrect: The `Response` omits the core information of the `Answer` or contradicts it.\n"
    "* Specific Rule for Proper Nouns (People, Places, etc.):\n"
    "  * If the `Answer` is a proper noun (e.g., a person's full name, a full name of a place or organization), a `Response` that provides a commonly recognized partial version (e.g., a last name, an acronym) is also considered Correct.\n"
    "  * Additionally, a `Response` that uses a widely recognized alternative name, a former name for a rebranded entity, or vice versa, is also considered Correct. For example, if the `Answer` is 'Federal Reserve', a `Response` of 'Reserve bank' is correct. Similarly, if the `Answer` is 'Meta', a `Response` of 'Facebook' is correct. Similarly, if the `Answer` is 'Sportsbooks', a `Response` of 'Sportsbook apps' is correct."
    " /no_think<|im_end|>\n"
    "<|im_start|>assistant\n"
)


check_answer_prompt_llama =  (      
    "You are an impartial evaluator. Your task is to assess whether a given `Response` correctly answers a `Question`, based on a provided ground-truth `Answer`. Follow the rules below in the specified order.\n\n"
    
    "## Input:\n"
    "`Question`: {question}.\n"
    "`Answer`: {answer}.\n"
    "`Response`: {response}.\n\n"
    
    "## Evaluation Workflow\n"
    "### Step 1: Check for Insufficient Information\n"
    "First, examine the `Response`. If the `Response` explicitly states that it cannot answer the query due to insufficient or irrelevant information, or if it requests more details, stop the evaluation immediately.\n"
    "* Condition: The `Response` contains phrases such as 'insufficient information to determine', 'I need more context', 'based on the provided snippets', or 'cannot be answered with the information given.'\n"
    "* Action: If this condition is met, output only the following text and nothing else: 'Insufficient Information'\n"
    "### Step 2: Classify the Ground-Truth Answer\n"
    "If the evaluation proceeds, your next step is to classify the nature of the `Answer`.\n"
    "* If the primary content of the `Answer` is a number (e.g., a quantity, price, date, or measurement), you will determine the output for the first line is: 'Answer is numeric'\n"
    "* Otherwise (if the `Answer` is text, a name, a concept, etc.), you will determine the output for the first line is: 'Answer is not numeric'\n"
    "### Step 3: Apply Evaluation Logic\n"
    "Based on the classification from Step 2, apply the corresponding logic to determine the verdict for the second line of the output.\n\n"
    
    "## Final Output Instructions\n"
    "After completing the workflow, your final output must strictly follow one of the formats below.\n"
    "1. Standard Case (Two Lines):\n"
    "  * Line 1: The result from Step 2 ('Answer is numeric' or 'Answer is not numeric').\n"
    "  * Line 2: The final verdict from Step 3 ('Correct', 'Incorrect', or 'Partially Correct').\n"
    "2. Special Case (One Line):\n"
    "  * If the condition in Step 1 is met, output only: Insufficient Information\n\n"
    
    "### Example of a standard two-line output:\n"
    "Answer is not numeric\n"
    "Correct\n\n"
    
    "## A. If 'Answer is numeric':\n"
    "The `Response` must be numerically equivalent to the `Answer` to be considered correct. This evaluation is strict.\n"
    "* Correct: The numerical value in the `Response` is an exact match to the `Answer`.\n"
    "  * Minor formatting differences are acceptable (e.g., '1,000' vs. '1000'; '$50' vs. '50 dollars'; 'five' vs. '5').\n"
    "* Incorrect: The numerical value in the `Response` does not match the `Answer`.\n"
    
    "## B. If 'Answer is not numeric':\n"
    "The evaluation is based on semantic correctness. The `Response` must convey the core meaning of the `Answer`, but it does not need to be a literal match. The `Response` can be more general or more specific, as long as it is not contradictory.\n"
    "* Correct: The `Response` accurately expresses the central fact or idea of the `Answer`. Rephrasing, summarization, or adding extra, non-contradictory information is acceptable.\n"
    "* Partially Correct: If the `Answer` consists of multiple distinct items or facts, and the `Response` correctly provides a subset of them while omitting others. The provided information must be accurate and not contradicted.\n"
    "* Incorrect: The `Response` omits the core information of the `Answer` or contradicts it.\n"
    "* Specific Rule for Proper Nouns (People, Places, etc.):\n"
    "  * If the `Answer` is a proper noun (e.g., a person's full name, a full name of a place or organization), a `Response` that provides a commonly recognized partial version (e.g., a last name, an acronym) is also considered Correct.\n"
    "  * Additionally, a `Response` that uses a widely recognized alternative name, a former name for a rebranded entity, or vice versa, is also considered Correct. For example, if the `Answer` is 'Federal Reserve', a `Response` of 'Reserve bank' is correct. Similarly, if the `Answer` is 'Meta', a `Response` of 'Facebook' is correct. Similarly, if the `Answer` is 'Sportsbooks', a `Response` of 'Sportsbook apps' is correct."
)

llama_QA_system_prompt = (
    "You are an expert Q&A system that is trusted around the world. "
    "Always answer the query using the provided context information, and not prior knowledge. "
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "Context information is below.\n"
    "---------------------\n")


llama_QA_graph_prompt = (
    f"{llama_QA_system_prompt} "
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. Note that only answer questions without explanation.\n"
    "Query: {query}\n"
    # "Given the context information answer the query.\n"
    "Answer:")

llama_QA_graph_prompt_with_one_context = (
    "You are an expert Q&A system that is trusted around the world. "
    "Always answer the query using the provided context information, and not prior knowledge. "
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "Context information is below.\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. Note that only answer questions without explanation.\n"
    "The context is only one, and if the contex does not contain the answer, you just need to output 'The context do not contain answer!'\n"
    "Query: {query}\n"
    "Answer:")

response_check_promot = (
    "You are an AI evaluator. Your task is to determine if the provided `Answer` correctly responds to the `Question`, based on the list of acceptable `Standard Answers`.\n\n"
    "**Inputs:**\n"
    "*   `Question`: The question that was asked.\n"
    "*   `Answer`: The response given by another system that you need to evaluate.\n"
    "*   `Standard Answers`: A list containing one or more valid ways the correct answer could be expressed.\n\n"
    "**Evaluation Rules (Follow these steps IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, carefully read the `Answer`.\n"
    "    *   Does the `Answer` explicitly state it cannot provide an answer due to missing or unclear information? Look for phrases like 'insufficient information', 'cannot determine', 'need more context', 'based on the provided snippets', 'unable to answer', or similar indications of refusal/inability to answer the core question.\n"
    "    *   If you find such phrases indicating inability to answer, your **only** output should be:\n"
    "        `Insufficient information error`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If the `Answer` does *not* indicate insufficient information (Step 1 was not met), compare the factual information presented in the `Answer` against the `Standard Answers` list.\n"
    "    *   The `Answer` is considered **correct** if it contains the essential factual information that matches or is equivalent to **at least one** of the entries in the `Standard Answers` list.\n"
    "    *   **Crucially:** The `Answer` might contain extra information or be longer than the `Standard Answers`. This is OKAY. As long as the core information required by the `Question` is present in the `Answer` and accurately matches one of the `Standard Answers`, consider it correct.\n"
    "    *   If the `Answer` contains information matching at least one `Standard Answer` (even with extra details present), your **only** output should be:\n"
    "        `True`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If the `Answer` did *not* meet the criteria for 'Insufficient Information' (Step 1) AND did *not* meet the criteria for 'Correctness' (Step 2), then it must be incorrect.\n"
    "    *   In this case, your **only** output should be:\n"
    "        `Error`\n\n"
    "**Output:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n"
    "---\n\n"
    "**Evaluation Result:**"
)

response_check_promot_qwen_instruct = (
    "<|im_start|>system\n"
    "You are an AI evaluator. Your task is to determine if the provided `Answer` correctly responds to the `Question`, based on the list of acceptable `Standard Answers`.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n\n"
    "**Inputs:**\n"
    "*   `Question`: The question that was asked.\n"
    "*   `Answer`: The response given by another system that you need to evaluate.\n"
    "*   `Standard Answers`: A list containing one or more valid ways the correct answer could be expressed.\n\n"
    "**Evaluation Rules (Follow these steps IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, carefully read the `Answer`.\n"
    "    *   Does the `Answer` explicitly state it cannot provide an answer due to missing or unclear information? Look for phrases like 'insufficient information', 'cannot determine', 'need more context', 'based on the provided snippets', 'unable to answer', or similar indications of refusal/inability to answer the core question.\n"
    "    *   If you find such phrases indicating inability to answer, your **only** output should be:\n"
    "        `Insufficient information error`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If the `Answer` does *not* indicate insufficient information (Step 1 was not met), compare the factual information presented in the `Answer` against the `Standard Answers` list.\n"
    "    *   The `Answer` is considered **correct** if it contains the essential factual information that matches or is equivalent to **at least one** of the entries in the `Standard Answers` list.\n"
    "    *   **Crucially:** The `Answer` might contain extra information or be longer than the `Standard Answers`. This is OKAY. As long as the core information required by the `Question` is present in the `Answer` and accurately matches one of the `Standard Answers`, consider it correct.\n"
    "    *   If the `Answer` contains information matching at least one `Standard Answer` (even with extra details present), your **only** output should be:\n"
    "        `True`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If the `Answer` did *not* meet the criteria for 'Insufficient Information' (Step 1) AND did *not* meet the criteria for 'Correctness' (Step 2), then it must be incorrect.\n"
    "    *   In this case, your **only** output should be:\n"
    "        `Error`\n\n"
    "**Output:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n"
    "---\n\n"
    "**Evaluation Result:**"
    "\n<|im_end|>\n"
    "<|im_start|>assistant"
)

response_check_promot_llama_instruct = (
    # "<|im_start|>system\n"
    "You are an AI evaluator. Your task is to determine if the provided `Answer` correctly responds to the `Question`, based on the list of acceptable `Standard Answers`.\n"
    # "<|im_end|>\n"
    # "<|im_start|>user\n\n"
    "**Inputs:**\n"
    "*   `Question`: The question that was asked.\n"
    "*   `Answer`: The response given by another system that you need to evaluate.\n"
    "*   `Standard Answers`: A list containing one or more valid ways the correct answer could be expressed.\n\n"
    "**Evaluation Rules (Follow these steps IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, carefully read the `Answer`.\n"
    "    *   Does the `Answer` explicitly state it cannot provide an answer due to missing or unclear information? Look for phrases like 'insufficient information', 'cannot determine', 'need more context', 'based on the provided snippets', 'unable to answer', or similar indications of refusal/inability to answer the core question.\n"
    "    *   If you find such phrases indicating inability to answer, your **only** output should be:\n"
    "        `Insufficient information error`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If the `Answer` does *not* indicate insufficient information (Step 1 was not met), compare the factual information presented in the `Answer` against the `Standard Answers` list.\n"
    "    *   The `Answer` is considered **correct** if it contains the essential factual information that matches or is equivalent to **at least one** of the entries in the `Standard Answers` list.\n"
    "    *   **Crucially:** The `Answer` might contain extra information or be longer than the `Standard Answers`. This is OKAY. As long as the core information required by the `Question` is present in the `Answer` and accurately matches one of the `Standard Answers`, consider it correct.\n"
    "    *   If the `Answer` contains information matching at least one `Standard Answer` (even with extra details present), your **only** output should be:\n"
    "        `True`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If the `Answer` did *not* meet the criteria for 'Insufficient Information' (Step 1) AND did *not* meet the criteria for 'Correctness' (Step 2), then it must be incorrect.\n"
    "    *   In this case, your **only** output should be:\n"
    "        `Error`\n\n"
    "**Output:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n"
    "---\n\n"
    "**Evaluation Result:**"
    # "\n<|im_end|>\n"
    # "<|im_start|>assistant"
)

response_check_promot_system_api = (
    "You are an AI evaluator. Your task is to determine if the provided `Answer` correctly responds to the `Question`, based on the list of acceptable `Standard Answers`.\n"
)

response_check_promot_user_api = (
    "**Inputs:**\n"
    "*   `Question`: The question that was asked.\n"
    "*   `Answer`: The response given by another system that you need to evaluate.\n"
    "*   `Standard Answers`: A list containing one or more valid ways the correct answer could be expressed.\n\n"
    "**Evaluation Rules (Follow these steps IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, carefully read the `Answer`.\n"
    "    *   Does the `Answer` explicitly state it cannot provide an answer due to missing or unclear information? Look for phrases like 'insufficient information', 'cannot determine', 'need more context', 'based on the provided snippets', 'unable to answer', or similar indications of refusal/inability to answer the core question.\n"
    "    *   If you find such phrases indicating inability to answer, your **only** output should be:\n"
    "        `Insufficient information error`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If the `Answer` does *not* indicate insufficient information (Step 1 was not met), compare the factual information presented in the `Answer` against the `Standard Answers` list.\n"
    "    *   The `Answer` is considered **correct** if it contains the essential factual information that matches or is equivalent to **at least one** of the entries in the `Standard Answers` list.\n"
    "    *   **Crucially:** The `Answer` might contain extra information or be longer than the `Standard Answers`. This is OKAY. As long as the core information required by the `Question` is present in the `Answer` and accurately matches one of the `Standard Answers`, consider it correct.\n"
    "    *   If the `Answer` contains information matching at least one `Standard Answer` (even with extra details present), your **only** output should be:\n"
    "        `True`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If the `Answer` did *not* meet the criteria for 'Insufficient Information' (Step 1) AND did *not* meet the criteria for 'Correctness' (Step 2), then it must be incorrect.\n"
    "    *   In this case, your **only** output should be:\n"
    "        `Error`\n\n"
    "**Output:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n"
    "---\n\n"
    "**Evaluation Result:**"
)

response_check_promot_llama_instruct_huggingface = (
    # "<|im_start|>system\n"
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an AI evaluator. Your task is to determine if the provided `Answer` correctly responds to the `Question`, based on the list of acceptable `Standard Answers`.\n"
    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n"
    # "<|im_end|>\n"
    # "<|im_start|>user\n\n"
    "**Inputs:**\n"
    "*   `Question`: The question that was asked.\n"
    "*   `Answer`: The response given by another system that you need to evaluate.\n"
    "*   `Standard Answers`: A list containing one or more valid ways the correct answer could be expressed.\n\n"
    "**Evaluation Rules (Follow these steps IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, carefully read the `Answer`.\n"
    "    *   Does the `Answer` explicitly state it cannot provide an answer due to missing or unclear information? Look for phrases like 'insufficient information', 'cannot determine', 'need more context', 'based on the provided snippets', 'unable to answer', or similar indications of refusal/inability to answer the core question.\n"
    "    *   If you find such phrases indicating inability to answer, your **only** output should be:\n"
    "        `Insufficient information error`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If the `Answer` does *not* indicate insufficient information (Step 1 was not met), compare the factual information presented in the `Answer` against the `Standard Answers` list.\n"
    "    *   The `Answer` is considered **correct** if it contains the essential factual information that matches or is equivalent to **at least one** of the entries in the `Standard Answers` list.\n"
    "    *   **Crucially:** The `Answer` might contain extra information or be longer than the `Standard Answers`. This is OKAY. As long as the core information required by the `Question` is present in the `Answer` and accurately matches one of the `Standard Answers`, consider it correct.\n"
    "    *   If the `Answer` contains information matching at least one `Standard Answer` (even with extra details present), your **only** output should be:\n"
    "        `True`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If the `Answer` did *not* meet the criteria for 'Insufficient Information' (Step 1) AND did *not* meet the criteria for 'Correctness' (Step 2), then it must be incorrect.\n"
    "    *   In this case, your **only** output should be:\n"
    "        `Error`\n\n"
    "**Output:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n"
    "---\n\n"
    "**Evaluation Result:**"
    "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    # "\n<|im_end|>\n"
    # "<|im_start|>assistant"
)

response_check_promot_qwen_instruct_bak = (
    "<|im_start|>system\n"
    "You are a precise AI evaluator. Your task is to assess if an `Answer` correctly addresses a `Question` based on provided `Standard Answers`. Strictly follow the evaluation rules. Your output must be ONLY ONE of these exact strings: `True`, `Error`, or `Insufficient information error`. Do not add any explanation.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Please evaluate the following based on the rules provided:\n\n"
    "**Evaluation Rules (Follow IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, read the `Answer`.\n"
    "    *   Does it explicitly state it cannot provide an answer due to missing/unclear information (e.g., 'insufficient information', 'cannot determine', 'need more context', 'unable to answer')?\n"
    "    *   If YES, your **only** output should be: `Insufficient information error`. STOP HERE.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If Step 1 was NOT met: Compare the factual information in the `Answer` against the `Standard Answers`.\n"
    "    *   Is the `Answer` correct if it contains the essential factual information matching or equivalent to **at least one** `Standard Answer`? (Extra information in the `Answer` is OKAY).\n"
    "    *   If YES, your **only** output should be: `True`. STOP HERE.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If Step 1 AND Step 2 were NOT met, then the `Answer` is incorrect.\n"
    "    *   Your **only** output should be: `Error`.\n\n"
    "**Output Constraint:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    "**Inputs for Evaluation:**\n"
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n"
    "---\n\n"
    "**Evaluation Result:**\n"  # Changed from the original "Evaluation Result:" to make it a clearer instruction.
    "<|im_end|>\n"
    "<|im_start|>assistant"
)


response_check_promot_llama3_8b = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an AI evaluator. Your task is to follow the user's instructions precisely to evaluate a given Answer against a Question and Standard Answers, and output only the specified result format." # System prompt defining the specific role for this task
    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n"
    # --- Start of the original prompt instructions, now inside the user block ---
    "Your task is to determine if the provided `Answer` correctly responds to the `Question`, based on the list of acceptable `Standard Answers`.\n\n"
    "**Inputs:**\n"
    "*   `Question`: The question that was asked.\n"
    "*   `Answer`: The response given by another system that you need to evaluate.\n"
    "*   `Standard Answers`: A list containing one or more valid ways the correct answer could be expressed.\n\n"
    "**Evaluation Rules (Follow these steps IN ORDER):**\n\n"
    "1.  **Check for Insufficient Information:**\n"
    "    *   First, carefully read the `Answer`.\n"
    "    *   Does the `Answer` explicitly state it cannot provide an answer due to missing or unclear information? Look for phrases like 'insufficient information', 'cannot determine', 'need more context', 'based on the provided snippets', 'unable to answer', or similar indications of refusal/inability to answer the core question.\n"
    "    *   If you find such phrases indicating inability to answer, your **only** output should be:\n"
    "        `Insufficient information error`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "2.  **Check for Correctness:**\n"
    "    *   If the `Answer` does *not* indicate insufficient information (Step 1 was not met), compare the factual information presented in the `Answer` against the `Standard Answers` list.\n"
    "    *   The `Answer` is considered **correct** if it contains the essential factual information that matches or is equivalent to **at least one** of the entries in the `Standard Answers` list.\n"
    "    *   **Crucially:** The `Answer` might contain extra information or be longer than the `Standard Answers`. This is OKAY. As long as the core information required by the `Question` is present in the `Answer` and accurately matches one of the `Standard Answers`, consider it correct.\n"
    "    *   If the `Answer` contains information matching at least one `Standard Answer` (even with extra details present), your **only** output should be:\n"
    "        `True`\n"
    "    *   If this condition is met, STOP HERE. Do not proceed to the next step.\n\n"
    "3.  **Determine Incorrectness:**\n"
    "    *   If the `Answer` did *not* meet the criteria for 'Insufficient Information' (Step 1) AND did *not* meet the criteria for 'Correctness' (Step 2), then it must be incorrect.\n"
    "    *   In this case, your **only** output should be:\n"
    "        `Error`\n\n"
    "**Output:**\n"
    "*   Provide **only one** of the three possible outputs: `True`, `Error`, or `Insufficient information error`.\n"
    "*   Do not add any explanations or extra text.\n\n"
    "---\n\n"
    # --- Input placeholders ---
    "**Question:** {Question}\n"
    "**Answer:** {Answer}\n"
    "**Standard Answers:** {Standard_answers}\n" # Make sure your variable name matches here
    "---\n\n"
    # --- Prompt for the model's response ---
    "**Evaluation Result:**"
    # --- End of user instructions ---
    "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>" # Signal for the assistant to start its response
)

knowledge_graph_refinement_prompt = (
    "You are a highly accurate and structured expert in knowledge graph reasoning and error correction. "
    "Your task is to refine and correct retrieved knowledge graph paths based on user feedback. "
    "Follow these instructions carefully:\n"
    "Path Analysis Rules:\n"
    "1. Path Representation: The retrieval result consists of a single-hop path in the knowledge graph. "
    "The relationship connects the head entity and the tail entity, and the direction of the relationship is from the head entity to the tail entity. "
    "The relationship is enclosed in - ->.\n"
    "2. Direction Matters: The path A -R-> B is not equivalent to B -R-> A. The direction of the relationship is significant.\n"
    "Task Breakdown:\n"
    "1. Identify Redundant Single-Hop Paths in Retrieval Result: Determine if there are redundant single-hop paths in the retrieved results. "
    "These paths may differ in wording but convey the same semantic meaning.\n"
    "2. Output Format: If there are no redundant relationships, respond with: 'No relevant information found.' "
    "If redundant relationships are found, represent each group of redundant relationships on a single line, in the form of Entity1 -Relation-> Entity2, "
    "with the most standard and clear relationship placed first. Separate redundant relationships using ;;;.\n"
    "3. Step-by-Step Reasoning: Think step by step and strictly follow the instructions below. Do not provide explanations—only structured outputs.\n"
    "4. Be rigorous: if you are uncertain about any content, ignore it and do not include it in the results.\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer: {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above. Please focus on the information in retrieve_result that is directly relevant to the query and answer, "
    "and ignore weakly related noisy information.\n"
    "Answer:"
)

knowledge_graph_entity_refinement_prompt = (
    "You are a highly accurate and structured expert in knowledge graph reasoning and error correction. "
    "Your task is to refine and correct retrieved knowledge graph paths based on user feedback. "
    "Follow these instructions carefully:\n"
    "Path Analysis Rules:\n"
    "1. Path Representation: The retrieval result consists of a single-hop path in the knowledge graph. "
    "The relationship connects the head entity and the tail entity, and the direction of the relationship is from the head entity to the tail entity. "
    "The relationship is enclosed in - ->.\n"
    "2. Direction Matters: The path A -R-> B is not equivalent to B -R-> A. The direction of the relationship is significant.\n"
    "Task Breakdown:\n"
    "1. Identifying Redundant Entities in Retrieval Result: "
    "Find entities in Retrieval Result that are merely different textual representations of the same real-world concept. "
    "These entities should remain identical in meaning even when removed from the current context.\n"
    "2. Output Format: If there are no redundant entities, respond with: 'No relevant information found.' "
    "Represent each group of redundant entities on a single line, with the most standard, complete, and unambiguous entity placed first. "
    "Separate entities within a group using ;;;.\n"
    "3. Step-by-Step Reasoning: Think step by step and strictly follow the instructions below. Do not provide explanations—only structured outputs.\n"
    "4. Be rigorous: if you are uncertain about any content, ignore it and do not include it in the results.\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer: {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above. Please focus on the information in retrieve_result that is directly relevant to the query and answer, "
    "and ignore weakly related noisy information.\n"
    "Answer:"
)

knowledge_graph_error_correction_prompt = (
    "You are a highly accurate and structured expert in knowledge graph reasoning and error correction. "
    "Your task is to refine and correct retrieved knowledge graph paths based on user feedback. "
    "Follow these instructions carefully:\n"
    "Path Analysis Rules:\n"
    "1. Path Representation: The retrieval result consists of a single-hop path in the knowledge graph. "
    "The relationship connects the head entity and the tail entity, and the direction of the relationship is from the head entity to the tail entity. "
    "The relationship is enclosed in - ->.\n"
    "2. Direction Matters: The path A -R-> B is not equivalent to B -R-> A. The direction of the relationship is significant.\n"
    "Task Breakdown:\n"
    "1. Identifying Incorrect or Misleading Paths in Retrieval Results: "
    "Pinpoint content in the retrieval results that directly contradicts the fact-based answer to the question.\n"
    "2. Be Rigorous: If you are uncertain about any content, exclude it from the results entirely. "
    "Pay close attention to the timing of events. For example, having different champions in a competition across different years is not contradictory. "
    "Focus on Contradictory Relationships: Example: In the same match, if Entity A wins against Entity B, then it is impossible for Entity B to have defeated Entity A.\n"
    "3. Output Format: If no incorrect or misleading paths exist, respond with: 'No relevant information found.' "
    "Represent all incorrect one-hop paths in the format: Entity1 -Relation-> Entity2. "
    "If multiple incorrect paths exist, present each path on a new line.\n"
    "4. Do not provide explanations—only output structured results as specified above.\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer (Various formats for writing and expressing answers) : {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above. Please focus on the information in retrieve_result that is directly relevant to the query and answer, "
    "and ignore weakly related noisy information.\n"
    "Answer:"
)

knowledge_graph_path_generation_prompt = (
    "You are a highly accurate and structured expert in knowledge graph reasoning and error correction. "
    "Your task is to refine and correct retrieved knowledge graph paths based on user feedback. "
    "Follow these instructions carefully:\n"
    "Path Analysis Rules:\n"
    "1. Path Representation: The retrieval result consists of a single-hop path in the knowledge graph. "
    "The relationship connects the head entity and the tail entity, and the direction of the relationship is from the head entity to the tail entity. "
    "The relationship is enclosed in - ->.\n"
    "2. Direction Matters: The path A -R-> B is not equivalent to B -R-> A. The direction of the relationship is significant.\n"
    "Task Breakdown:\n"
    "1. Generating Corrected Reasoning Paths: "
    "If the one-hop paths in the Retrieval Results are insufficient to answer the question correctly, generate up to three additional one-hop paths based on the question and the expected answer. "
    "For complex questions, ensure the new paths support multi-hop reasoning by creating interconnected relationships.\n"
    "2. If multiple paths are generated, present each path on a new line.\n"
    "3. If the existing retrieved paths are already sufficient, respond with 'No relevant information found.'\n"
    "4. Follow the instructions strictly, think step-by-step, and provide only structured outputs without explanations.\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer (Various formats for writing and expressing answers) : {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above. Please focus on the information in retrieve_result that is directly relevant to the query and answer, "
    "and ignore weakly related noisy information.\n"
    "Answer:"
)

knowledge_graph_score_correction_prompt = (
    "You are a highly accurate and structured expert in knowledge graph reasoning and error correction. "
    "Your task is to identify paths (triples) in the retrieved knowledge graph that either contradict or support the given answer based on the question and answer. "
    "Follow these instructions carefully:\n"
    "Path Analysis Rules:\n"
    "1. Path Representation: The retrieval result consists of a single-hop path in the knowledge graph. "
    "The relationship connects the head entity and the tail entity, and the direction of the relationship is from the head entity to the tail entity. "
    "The relationship is enclosed in - ->.\n"
    "2. Direction Matters: The path A -R-> B is not equivalent to B -R-> A. The direction of the relationship is significant.\n"
    "Task Breakdown:\n"
    "Score the retrieved paths based on the factual question and answer. If a triple contains the factual answer to the question, score it 2. If a triple contradicts the factual answer to the question, score it 0. All other triples, which are not directly related to the question and answer, score 1. "
    "Please output the scores of the triples in order. For a Retrieve Result containing 3 triples, a possible output format is, Answer: "
    "2\n"
    "1\n"
    "0\n"
    "4. Do not provide explanations—only output structured results as specified above.\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer (Various formats for writing and expressing answers) : {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above.\n"
    "Answer:"
)

knowledge_graph_score_correction_prompt_2 = (
    "You are a highly accurate and structured expert in knowledge graph reasoning and error correction. "
    "Your task is to identify paths (triples) in the retrieved knowledge graph that either contradict or support the given answer based on the question and answer. "
    "Follow these instructions carefully:\n"
    "Path Analysis Rules:\n"
    "1. Path Representation: The retrieval result consists of a single-hop path in the knowledge graph. "
    "The relationship connects the head entity and the tail entity, and the direction of the relationship is from the head entity to the tail entity. "
    "The relationship is enclosed in - ->.\n"
    "2. Direction Matters: The path A -R-> B is not equivalent to B -R-> A. The direction of the relationship is significant.\n"
    "Task Breakdown:\n"
    "Identify the triples in the Retrieve Result that contain the correct answer to the question and those that contain incorrect answers. The remaining triples are those that contain indirectly related answers. "
    "After determining the correct (or incorrect) triples, evaluate the degree of correctness (or incorrectness) of these triples on a scale of three levels: 1 represents partially correct (partially incorrect), 2 represents mostly correct (generally incorrect), and 3 represents completely correct (completely incorrect)."
    "For uncertain errors, do not classify them as incorrect. For example, if the answer is 'A starred in movie B,' this does not contradict others starring in movie B, as a movie can have multiple actors. However, if A is the leading actor in movie B, then others cannot also be the leading actor in movie B, as the leading role is unique. "
    "Please pay attention to time-related issues, as the same question may yield different results at different times."
    "Please pay attention to the active and passive forms of relationships, as they represent completely opposite meanings."
    "Please output the indices of the triples that contain the correct answer after 'Correct:', and the indices of the triples that contain incorrect answers after 'Error:', using ' ' as a separator. "
    "An example output format is as follows:"
    "Correct: 2:3 5:2 12:1"
    "Error:"
    "Do not provide explanations—only output structured results as specified above.\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer (Various formats for writing and expressing answers) : {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above.\n"
    "Correct:\n"
    "Error:"
)

knowledge_graph_score_correction_prompt_3 = (  #使用中
    "Your task is to evaluate each short statement retrieved in the Retrieve Result and determine whether it can answer the Query.\n\n"
    "If the statement leads to the correct answer found in the Fact-based Answer, it is classified as a correct triple.\n"
    "If the statement leads to an incorrect answer, it is classified as an incorrect triple.\n"
    "Please be careful, please pay close attention, If the statement cannot answer the question, it should be ignored.\n"
    "Please make sure that the answers you get are exactly the same as the real answers, especially the numbers and .\n"
    "For both correct and incorrect triples, assign a score from 1 to 3 based on the degree of correctness or incorrectness:\n\n"
    "1 – Partially correct (or partially incorrect)\n"
    "2 – Mostly correct (or generally incorrect)\n"
    "3 – Completely correct (or completely incorrect)\n"
    "A higher score indicates a stronger degree of correctness or incorrectness.\n\n"
    "Example output:\n"
    "Correct: 2:3 5:2 12:1\n"
    "Error: 3:3 7:2\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer (Various formats for writing and expressing answers) : {answer}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Please complete the task based on the information provided above. Do not provide explanations—only output structured results as specified above.\n"
)

knowledge_graph_redundant_relationship_prompt = (
    "You will receive multiple groups of short statements. Some statements within the same group may convey the same or similar meaning in different ways.\n\n"
    "Task:\n"
    "Identify and retain only distinct meanings while ensuring that at least one statement is kept from each group.\n"
    "For statements with the same or similar meaning that can be used interchangeably to answer related questions, keep only the clearest and most complete one.\n"
    "If multiple statements mention the same event or fact with variations in time, location, or factual detail, retain the statement with the most complete and informative content.\n"
    "For statements with different meanings, retain them all.\n"
    "Retain only one statement per group.\n"
    "Retain only one statement per group.\n"
    "Retain only one statement per group.\n"
    "If you decide to keep more than one, carefully consider whether they truly express completely different meanings.\n"
    "Output Format:\n"
    "Print the indices of the retained statements, separated by commas for each group.\n"
    "Each group's retained indices should be printed on a new line.\n"
    "Do not provide explanations—only output structured results as specified.\n"
    "Example Output:\n"
    "2\n"
    "6, 8\n"
    "12\n\n"
    "statements:\n"
    "{redundant_relationship}"
    "output:"
)

knowledge_graph_redundant_relationship_prompt_v2 = (
    "Identify and retain only the distinct meanings from the following group of statements.\n"
    "For statements that express the same or similar meaning and can be used interchangeably to answer related questions, keep only the clearest and most complete one.\n"
    "If statements describe the same event or fact but differ in time, location, or factual detail, retain the one with the most informative and complete content.\n"
    "If the statements express clearly different meanings, retain them all.\n"
    "Statements involving quantities, numbers, monetary values, or dates must have exactly the same values to be considered semantically equivalent—otherwise, treat them as expressing different meanings.\n"
    "Retain only one statement unless multiple statements truly express completely different meanings.\n"
    "If you decide to keep more than one, ensure each conveys a unique meaning.\n\n"
    "Output Format:\n"
    "Print the indices of the retained statements, separated by commas.\n"
    "Do not provide explanations—only output the structured result as specified.\n\n"
    "Example Output:\n"
    "1, 3\n\n"
    "statements:\n{redundant_relationship}\n"
    "output:"
)

knowledge_graph_redundant_relationship_prompt_v3 = (
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy.\n\n"
    "1.  Similarity & Paraphrasing: If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only one instance, preferably the clearest, most complete, or most standard representation.\n"
    "2.  Information Subsumption: If one statement's meaning is fully contained within another more specific or informative statement, retain only the more informative one.\n"
    "3.  Sensitivity to Key Details (Numbers, Dates, Quantities): Be sensitive to differences in quantities, numbers, monetary values, dates, or other factual details.\n"
    "    If these differences clearly indicate distinct events, facts, times, or significantly different quantities, treat the statements as having different meanings and retain them accordingly.\n"
    "    However, if the context strongly suggests the statements refer to the same underlying fact or event but have minor variations in these details (e.g., slight rounding differences, near-identical numbers likely due to reporting variations), prioritize the shared core meaning and treat them as similar, keeping only one representative statement according to rule #1.\n"
    "4.  Different Aspects of Related Facts: If statements describe different facets or temporal aspects of a related situation (e.g., an action and its resulting state, like 'transferred to' vs. 'plays for'), evaluate if both convey unique, valuable information. If so, retain both. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "5.  Clearly Different Meanings: If statements express clearly distinct facts, events, or relationships, retain all of them.\n\n"
    "Processing Steps:\n"
    "Consider the statements derived from the input.\n"
    "Apply the rules above to filter the statements within each group.\n"
    "Aim to represent the unique information from the original set accurately and concisely.\n\n"
    "Output Format:\n"
    "Print the zero-based indices of the retained statements, separated by commas.\n"
    "Do not provide explanations—only output the structured result as specified.\n"
    "Example Output:\n"
    "1, 3\n\n"
    "Example Application to Provided Groups:\n"
    "Group 0 (0, 1): 0. 2023 citrus bowl Features teams Lsu tigers, 1. Lsu tigers Will play in 2023 citrus bowl. Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "Group 1 (2, 3): 2. 2023 citrus bowl Features teams Purdue boilermakers, 3. Purdue boilermakers Will play in 2023 citrus bowl. Semantically similar. Keep one index (e.g., 2 or 3).\n"
    "Group 2 (4, 5): 4. 2022 citrus bowl Featured Kentucky wildcats, 5. Kentucky wildcats Played in 2022 citrus bowl. Semantically similar. Keep one index (e.g., 4 or 5).\n"
    "Group 3 (6, 7): 6. Kentucky wildcats Won against Iowa hawkeyes, 7. Iowa hawkeyes Played against Kentucky wildcats. Statement 6 ('Won against') is more specific and implies 7 ('Played against'). Keep index 6.\n"
    "Group 4 (8, 9): 8. 2022 citrus bowl Featured Iowa hawkeyes, 9. Iowa hawkeyes Played in 2022 citrus bowl. Semantically similar. Keep one index (e.g., 8 or 9, user preferred 9). Let's aim for consistency, maybe keep the 'Played in' form if available, so keep 9.\n"
    "Group 5 (10, 11): 10. Tayvion robinson Plays for Kentucky, 11. Tayvion robinson Transferred to Kentucky. Statement 11 ('Transferred to') describes the event leading to the state in 10 ('Plays for'). 'Transferred to' provides unique temporal/action information. Depending on whether both the action and the state are considered distinct valuable facts, keep both 10 and 11.\n\n"
    "statements:\n{redundant_relationship}\n"
    "output:"
)

knowledge_graph_redundant_relationship_prompt_qwen_instruct = (
    "<|im_start|>system\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy.\n\n"
    "1.  Similarity & Paraphrasing: If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only one instance, preferably the clearest, most complete, or most standard representation.\n"
    "2.  Information Subsumption: If one statement's meaning is fully contained within another more specific or informative statement, retain only the more informative one.\n"
    "3.  Sensitivity to Key Details (Numbers, Dates, Quantities): Be sensitive to differences in quantities, numbers, monetary values, dates, or other factual details.\n"
    "    If these differences clearly indicate distinct events, facts, times, or significantly different quantities, treat the statements as having different meanings and retain them accordingly.\n"
    "    However, if the context strongly suggests the statements refer to the same underlying fact or event but have minor variations in these details (e.g., slight rounding differences, near-identical numbers likely due to reporting variations), prioritize the shared core meaning and treat them as similar, keeping only one representative statement according to rule #1.\n"
    "4.  Different Aspects of Related Facts: If statements describe different facets or temporal aspects of a related situation (e.g., an action and its resulting state, like 'transferred to' vs. 'plays for'), evaluate if both convey unique, valuable information. If so, retain both. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "5.  Clearly Different Meanings: If statements express clearly distinct facts, events, or relationships, retain all of them.\n\n"
    "Processing Steps:\n"
    "Consider the statements derived from the input.\n"
    "Apply the rules above to filter the statements within each group.\n"
    "Aim to represent the unique information from the original set accurately and concisely.\n\n"
    "Output Format:\n"
    "Print the zero-based indices of the retained statements, separated by commas.\n"
    "Do not provide explanations—only output the structured result as specified.\n"
    "Example Output:\n"
    "1, 3\n\n"
    "Example Application to Provided Groups:\n"
    "Group 0 (0, 1): 0. 2023 citrus bowl Features teams Lsu tigers, 1. Lsu tigers Will play in 2023 citrus bowl. Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "Group 1 (2, 3): 2. 2023 citrus bowl Features teams Purdue boilermakers, 3. Purdue boilermakers Will play in 2023 citrus bowl. Semantically similar. Keep one index (e.g., 2 or 3).\n"
    "Group 2 (4, 5): 4. 2022 citrus bowl Featured Kentucky wildcats, 5. Kentucky wildcats Played in 2022 citrus bowl. Semantically similar. Keep one index (e.g., 4 or 5).\n"
    "Group 3 (6, 7): 6. Kentucky wildcats Won against Iowa hawkeyes, 7. Iowa hawkeyes Played against Kentucky wildcats. Statement 6 ('Won against') is more specific and implies 7 ('Played against'). Keep index 6.\n"
    "Group 4 (8, 9): 8. 2022 citrus bowl Featured Iowa hawkeyes, 9. Iowa hawkeyes Played in 2022 citrus bowl. Semantically similar. Keep one index (e.g., 8 or 9, user preferred 9). Let's aim for consistency, maybe keep the 'Played in' form if available, so keep 9.\n"
    "Group 5 (10, 11): 10. Tayvion robinson Plays for Kentucky, 11. Tayvion robinson Transferred to Kentucky. Statement 11 ('Transferred to') describes the event leading to the state in 10 ('Plays for'). 'Transferred to' provides unique temporal/action information. Depending on whether both the action and the state are considered distinct valuable facts, keep both 10 and 11.\n\n"
    "statements:\n{redundant_relationship}\n"
    "output:"
    "\n<|im_end|>\n"
    "<|im_start|>assistant"
)

knowledge_graph_redundant_relationship_prompt_qwen_instruct_v2 = (
    "<|im_start|>system\n"
    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy, strictly adhering to the rules below, especially regarding dates and numerical values.\n\n"
    "**Rules for Statement Retention:**\n"
    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only **one** instance, preferably the clearest, most complete, or most standard representation.\n"
    "2.  **Information Subsumption & Date Specificity:**\n"
    "    * If one statement's meaning is fully contained within another more specific or informative statement, retain only the **more informative** one.\n"
    "    * **Crucially for dates:** If two statements refer to the **exact same event** but differ *only* in the level of date detail (e.g., 'January 1st, 2023' vs. 'January 2023' vs. '2023'), retain **only** the statement with the **most specific date**. A more detailed date *replaces* a less detailed date for the same fact.\n"
    "3.  **Strict Handling of Numerical and Temporal Details (Numbers, Dates, Quantities):**\n"
    "    * Treat statements as representing **distinct** facts and retain them **all** if they differ in any stated:\n"
    "        * Numerical values (quantities, monetary amounts, scores, counts, etc.)\n"
    "        * Specific dates or times (unless Rule 2 applies for granularity of the *same event*)\n"
    "    * **Exception:** Redundancy based on numerical values (Rule 1) can *only* be considered if the numbers/quantities mentioned are **exactly identical** *and* the rest of the core meaning is also identical. Any difference, no matter how small, means the statements are distinct unless it's a date granularity issue covered by Rule 2.\n"
    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation (e.g., an action 'transferred to' and its resulting state 'plays for'), evaluate if both convey unique, valuable information. If so, **retain both**. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships unrelated to the rules above, **retain all** of them.\n\n"
    "**Processing Steps:**\n"
    "1.  Consider the list of statements provided.\n"
    "2.  Apply the rules above meticulously to filter the statements.\n"
    "3.  Prioritize strict interpretation of rules 2 and 3 regarding dates and numbers.\n"
    "4.  Aim to represent the unique information from the original set accurately and concisely according to these rules.\n\n"
    "**Output Format:**\n"
    "* Print **only** the zero-based indices of the retained statements, separated by commas.\n"
    "* Do not include headers, explanations, or any text other than the comma-separated indices.\n"
    "**Example Output:**\n"
    "1, 3\n\n"
    "**Example Application Illustrating Rules (Conceptual):**\n"
    "* Group A (Rule 3 - Different Numbers): 'Team X scored 10 points.', 'Team X scored 11 points.' -> Retain both indices.\n"
    "* Group B (Rule 2 - Date Specificity): 'The event happened in 2023.', 'The event happened on Jan 5th, 2023.' -> Retain only the index for the second statement.\n"
    # "* Group C (Rule 4 - Different Aspects): 'He joined the team.', 'He is on the team.' -> Potentially retain both indices if 'joining' (action) and 'being on' (state) are deemed distinctly valuable, or retain the most informative index if one implies the other strongly in context.\n"
    "* Group C (Rule 1/3 - Identical Numbers): 'The budget is $5,000.', 'The budget is $5,000.00.' -> Treat as identical numbers. If rest of the meaning is the same, keep one index.\n"
    "* Group D (Rule 3 - Different Dates): 'She arrived on Monday.', 'He arrived on Tuesday.' -> Retain both indices.\n"
    "* Group E (Rule 1 - Paraphrasing): '2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.' -> Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "* Group F (Rule 2 - Subsumption/More Specific Info): 'Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.' -> Statement 0 ('Won against') is more specific and implies 1 ('Played against'). Keep index 0.\n"
    # "* Group H (Rule 1 - Active/Passive Voice): 'J.K. Rowling wrote Harry Potter.', 'Harry Potter was written by J.K. Rowling.' -> Express the same core fact using different grammatical voice (active vs. passive). Keep only one index (e.g., 0 or 1).\n"
    "**Statements to Process:**\n{redundant_relationship}\n\n"
    "**Output:**\n"
    "<|im_end|>\n" #  /no_think
    "<|im_start|>assistant"
)

knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3 = (
    "<|im_start|>system\n"
    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy, strictly adhering to the rules below, especially regarding dates and numerical values.\n\n"
    "**Rules for Statement Retention:**\n"
    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only **one** instance, preferably the clearest, most complete, or most standard representation.\n"
    "    * If different descriptions of the same attribute of an event appear — even if they are conflicting — all such descriptions should be retained.\n"
    "2.  **Information Subsumption & Date Specificity:**\n"
    "    * If one statement's meaning is fully contained within another more specific or informative statement, retain only the **more informative** one.\n"
    "    * If there is no subsumption, retain **both**.\n"
    "    * **Crucially for dates:** If two statements refer to the **exact same event** but differ *only* in the level of date detail (e.g., 'January 1st, 2023' vs. 'January 2023' vs. '2023'), retain **only** the statement with the **most specific date**. A more detailed date *replaces* a less detailed date for the same fact.\n"
    "3.  **Strict Handling of Numerical and Temporal Details (Numbers, Dates, Quantities):**\n"
    "    * Treat statements as representing **distinct** facts and retain them **all** if they differ in any stated:\n"
    "        * Numerical values (quantities, monetary amounts, scores, counts, etc.)\n"
    "        * Specific dates or times (unless Rule 2 applies for granularity of the *same event*)\n"
    "    * **Exception:** Redundancy based on numerical values (Rule 1) can *only* be considered if the numbers/quantities mentioned are **exactly identical** *and* the rest of the core meaning is also identical. Any difference, no matter how small, means the statements are distinct unless it's a date granularity issue covered by Rule 2.\n"
    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation, evaluate if both convey unique, valuable information. If so, **retain both**. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "    * For example, both 'joining a company' and 'being employed at the company' should be retained. For cases like 'located at' vs. 'held at', retain only one.\n"
    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships unrelated to the rules above, **retain all** of them.\n\n"
    "**Processing Steps:**\n"
    "1.  Consider the list of statements provided.\n"
    "2.  Apply the rules above meticulously to filter the statements.\n"
    "3.  Prioritize strict interpretation of rules 2 and 3 regarding dates and numbers.\n"
    "4.  Aim to represent the unique information from the original set accurately and concisely according to these rules.\n\n"
    "**Output Format:**\n"
    "* Print **only** the zero-based indices of the retained statements, separated by commas.\n"
    "* Do not include headers, explanations, or any text other than the comma-separated indices.\n"
    "**Example Output:**\n"
    "1, 3\n\n"
    "**Example Application Illustrating Rules (Conceptual):**\n"
    "* Group A (Rule 3 - Different Numbers): 'Team X scored 10 points.', 'Team X scored 11 points.' -> Retain both indices.\n"
    "* Group B (Rule 2 - Date Specificity): 'The event happened in 2023.', 'The event happened on Jan 5th, 2023.' -> Retain only the index for the second statement. Because the second sentence includes the information from the first.\n"
    "* Group C (Rule 1/3 - Identical Numbers): 'The budget is $5,000.', 'The budget is $5,000.00.' -> Treat as identical numbers. If rest of the meaning is the same, keep one index.\n"
    "* Group D (Rule 3 - Different Dates): 'She arrived on Monday.', 'He arrived on Tuesday.' -> Retain both indices.\n"
    "* Group E (Rule 1 - Paraphrasing): '2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.' -> Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "* Group F (Rule 2 - Subsumption/More Specific Info): 'Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.' -> Statement 0 ('Won against') is more specific and implies 1 ('Played against'). Keep index 0.\n"
    "**Statements to Process:**\n{redundant_relationship}\n\n"
    "**Output:** /no_think"
    "<|im_end|>\n"
    "<|im_start|>assistant"
)

knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3_local = (
    "<|im_start|>system\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Your *primary and most crucial objective* is to synthesize a list of statements, ensuring you *capture the complete set of unique information while simultaneously eliminating all possible redundancy*. You must analyze the provided list and produce a minimal, definitive set of facts. This means every unique piece of information must be represented, but with the fewest statements possible.\n"
    "To achieve this, apply the following principles and rules, focusing on semantic meaning and factual accuracy.\n\n"
    
    "### Guiding Principle\n"
    "* One Fact, One Statement: The primary goal is to represent each unique piece of information with a single statement. For any given subject or event, strive to retain only the most comprehensive sentence.\n"
    "* Preserve All Unique Information: If multiple statements present genuinely different facts (e.g., separate events, different numerical values) or offer conflicting details about the same fact, they must *all* be retained. Do not discard a statement if it contains unique information not present elsewhere.\n"
    
    "### Core Deduplication Rules\n"
    "1. Semantic Equivalence (Same Meaning):\n"
    "* If multiple statements express the exact same fact using different wording (e.g., synonyms, rephrasing, active/passive voice), keep only *one* representative instance.\n"
    "* Conflict Exception: If statements provide conflicting details for the same attribute (e.g., 'The price is $10' vs. 'The price is $12'), keep *both* to highlight the discrepancy.\n"
    "2. Information Specificity (General vs. Detailed):\n"
    "* If one statement's information is fully contained within another, more detailed statement describing the exact same event, keep only the *more specific* statement.\n"
    "* Factual Detail: A more descriptive fact (e.g., 'Team A won against Team B') replaces a more generic one (e.g., 'Team A played against Team B').\n"
    "* Date Granularity: A statement with a more precise date (e.g., 'January 5th, 2023') replaces a less specific one (e.g., 'January 2023' or '2023') for the same event.\n"
    "3. Distinct Facts (Different Information):\n"
    "* Keep all statements that describe fundamentally different information. Treat statements as unique and retain all of them if they:\n"
    "  * Contain different numerical values (quantities, scores, amounts).\n"
    "  * Describe different events, even if they are related (e.g., 'joined the company' vs. 'was promoted at the company').\n"
    "  * Refer to different dates or times for separate events (e.g., 'He arrived on Monday' vs. 'She arrived on Tuesday').\n\n"
    
    "### Examples:\n"
    "* Rule 1 (Equivalence):['2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.'] -> Semantically the same. Keep one.\n"
    "* Rule 1 (Conflict Exception):['The project's budget is $5,000.', 'The project's budget is $5,500.'] -> Conflicting numerical values for the same attribute. Keep both.\n"
    "* Rule 2 (Specificity - Factual):['Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.'] -> 'Won against' is more specific than 'Played against'. Keep the first.\n"
    "* Rule 2 (Specificity - Date):['The event happened in 2023.', 'The event happened on Jan 5th, 2023.'] -> The second date is more specific. Keep the second.\n"
    "* Rule 3 (Distinct Facts - Events):['She joined the team in May.', 'She was promoted to team lead in August.'] -> Different events. Keep both.\n\n"
    
    "### Output Format:\n"
    "* Provide only the zero-based indices of the statements to be retained.\n"
    "* The indices must be separated by commas.\n"
    "* Do not include any headers, explanations, or other text in your final output.\n\n"
    
    "### Example Output:\n"
    "1,3\n\n"
    
    "Statements to Process:\n"
    "{redundant_relationship}\n\n"
    
    "Output: /no_think"
    "<|im_end|>\n"
    "<|im_start|>assistant"
)

knowledge_graph_redundant_relationship_prompt_qwen_v3_system_api = (
    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
)

knowledge_graph_redundant_relationship_prompt_qwen_v3_user_api = (
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy, strictly adhering to the rules below, especially regarding dates and numerical values.\n\n"
    "**Rules for Statement Retention:**\n"
    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only **one** instance, preferably the clearest, most complete, or most standard representation.\n"
    "    * If different descriptions of the same attribute of an event appear — even if they are conflicting — all such descriptions should be retained.\n"
    "2.  **Information Subsumption & Date Specificity:**\n"
    "    * If one statement's meaning is fully contained within another more specific or informative statement, retain only the **more informative** one.\n"
    "    * If there is no subsumption, retain **both**.\n"
    "    * **Crucially for dates:** If two statements refer to the **exact same event** but differ *only* in the level of date detail (e.g., 'January 1st, 2023' vs. 'January 2023' vs. '2023'), retain **only** the statement with the **most specific date**. A more detailed date *replaces* a less detailed date for the same fact.\n"
    "3.  **Strict Handling of Numerical and Temporal Details (Numbers, Dates, Quantities):**\n"
    "    * Treat statements as representing **distinct** facts and retain them **all** if they differ in any stated:\n"
    "        * Numerical values (quantities, monetary amounts, scores, counts, etc.)\n"
    "        * Specific dates or times (unless Rule 2 applies for granularity of the *same event*)\n"
    "    * **Exception:** Redundancy based on numerical values (Rule 1) can *only* be considered if the numbers/quantities mentioned are **exactly identical** *and* the rest of the core meaning is also identical. Any difference, no matter how small, means the statements are distinct unless it's a date granularity issue covered by Rule 2.\n"
    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation, evaluate if both convey unique, valuable information. If so, **retain both**. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "    * For example, both 'joining a company' and 'being employed at the company' should be retained. For cases like 'located at' vs. 'held at', retain only one.\n"
    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships unrelated to the rules above, **retain all** of them.\n\n"
    "**Processing Steps:**\n"
    "1.  Consider the list of statements provided.\n"
    "2.  Apply the rules above meticulously to filter the statements.\n"
    "3.  Prioritize strict interpretation of rules 2 and 3 regarding dates and numbers.\n"
    "4.  Aim to represent the unique information from the original set accurately and concisely according to these rules.\n\n"
    "**Output Format:**\n"
    "* Print **only** the zero-based indices of the retained statements, separated by commas.\n"
    "* Do not include headers, explanations, or any text other than the comma-separated indices.\n"
    "**Example Output:**\n"
    "1, 3\n\n"
    "**Example Application Illustrating Rules (Conceptual):**\n"
    "* Group A (Rule 3 - Different Numbers): 'Team X scored 10 points.', 'Team X scored 11 points.' -> Retain both indices.\n"
    "* Group B (Rule 2 - Date Specificity): 'The event happened in 2023.', 'The event happened on Jan 5th, 2023.' -> Retain only the index for the second statement. Because the second sentence includes the information from the first.\n"
    "* Group C (Rule 1/3 - Identical Numbers): 'The budget is $5,000.', 'The budget is $5,000.00.' -> Treat as identical numbers. If rest of the meaning is the same, keep one index.\n"
    "* Group D (Rule 3 - Different Dates): 'She arrived on Monday.', 'He arrived on Tuesday.' -> Retain both indices.\n"
    "* Group E (Rule 1 - Paraphrasing): '2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.' -> Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "* Group F (Rule 2 - Subsumption/More Specific Info): 'Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.' -> Statement 0 ('Won against') is more specific and implies 1 ('Played against'). Keep index 0.\n"
    "**Statements to Process:**\n{redundant_relationship}\n\n"
    "**Output:**\n"
)

knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3_llama = (
    # "<|im_start|>system\n"
    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
    # "<|im_end|>\n"
    # "<|im_start|>user\n"
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy, strictly adhering to the rules below, especially regarding dates and numerical values.\n\n"
    "**Rules for Statement Retention:**\n"
    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only **one** instance, preferably the clearest, most complete, or most standard representation.\n"
    "    * If different descriptions of the same attribute of an event appear — even if they are conflicting — all such descriptions should be retained.\n"
    "2.  **Information Subsumption & Date Specificity:**\n"
    "    * If one statement's meaning is fully contained within another more specific or informative statement, retain only the **more informative** one.\n"
    "    * If there is no subsumption, retain **both**.\n"
    "    * **Crucially for dates:** If two statements refer to the **exact same event** but differ *only* in the level of date detail (e.g., 'January 1st, 2023' vs. 'January 2023' vs. '2023'), retain **only** the statement with the **most specific date**. A more detailed date *replaces* a less detailed date for the same fact.\n"
    "3.  **Strict Handling of Numerical and Temporal Details (Numbers, Dates, Quantities):**\n"
    "    * Treat statements as representing **distinct** facts and retain them **all** if they differ in any stated:\n"
    "        * Numerical values (quantities, monetary amounts, scores, counts, etc.)\n"
    "        * Specific dates or times (unless Rule 2 applies for granularity of the *same event*)\n"
    "    * **Exception:** Redundancy based on numerical values (Rule 1) can *only* be considered if the numbers/quantities mentioned are **exactly identical** *and* the rest of the core meaning is also identical. Any difference, no matter how small, means the statements are distinct unless it's a date granularity issue covered by Rule 2.\n"
    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation, evaluate if both convey unique, valuable information. If so, **retain both**. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "    * For example, both 'joining a company' and 'being employed at the company' should be retained. For cases like 'located at' vs. 'held at', retain only one.\n"
    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships unrelated to the rules above, **retain all** of them.\n\n"
    "**Processing Steps:**\n"
    "1.  Consider the list of statements provided.\n"
    "2.  Apply the rules above meticulously to filter the statements.\n"
    "3.  Prioritize strict interpretation of rules 2 and 3 regarding dates and numbers.\n"
    "4.  Aim to represent the unique information from the original set accurately and concisely according to these rules.\n\n"
    "**Output Format:**\n"
    "* Print **only** the zero-based indices of the retained statements, separated by commas.\n"
    "* Do not include headers, explanations, or any text other than the comma-separated indices.\n"
    "**Example Output:**\n"
    "1, 3\n\n"
    "**Example Application Illustrating Rules (Conceptual):**\n"
    "* Group A (Rule 3 - Different Numbers): 'Team X scored 10 points.', 'Team X scored 11 points.' -> Retain both indices.\n"
    "* Group B (Rule 2 - Date Specificity): 'The event happened in 2023.', 'The event happened on Jan 5th, 2023.' -> Retain only the index for the second statement. Because the second sentence includes the information from the first.\n"
    "* Group C (Rule 1/3 - Identical Numbers): 'The budget is $5,000.', 'The budget is $5,000.00.' -> Treat as identical numbers. If rest of the meaning is the same, keep one index.\n"
    "* Group D (Rule 3 - Different Dates): 'She arrived on Monday.', 'He arrived on Tuesday.' -> Retain both indices.\n"
    "* Group E (Rule 1 - Paraphrasing): '2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.' -> Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "* Group F (Rule 2 - Subsumption/More Specific Info): 'Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.' -> Statement 0 ('Won against') is more specific and implies 1 ('Played against'). Keep index 0.\n"
    "**Statements to Process:**\n{redundant_relationship}\n\n"
    "**Output:**\n"
    # "<|im_end|>\n"
    # "<|im_start|>assistant"
)

knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3_llama_huggingface = (
    # "<|im_start|>system\n"
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n"
    # "<|im_end|>\n"
    # "<|im_start|>user\n"
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy, strictly adhering to the rules below, especially regarding dates and numerical values.\n\n"
    "**Rules for Statement Retention:**\n"
    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only **one** instance, preferably the clearest, most complete, or most standard representation.\n"
    "    * If different descriptions of the same attribute of an event appear — even if they are conflicting — all such descriptions should be retained.\n"
    "2.  **Information Subsumption & Date Specificity:**\n"
    "    * If one statement's meaning is fully contained within another more specific or informative statement, retain only the **more informative** one.\n"
    "    * If there is no subsumption, retain **both**.\n"
    "    * **Crucially for dates:** If two statements refer to the **exact same event** but differ *only* in the level of date detail (e.g., 'January 1st, 2023' vs. 'January 2023' vs. '2023'), retain **only** the statement with the **most specific date**. A more detailed date *replaces* a less detailed date for the same fact.\n"
    "3.  **Strict Handling of Numerical and Temporal Details (Numbers, Dates, Quantities):**\n"
    "    * Treat statements as representing **distinct** facts and retain them **all** if they differ in any stated:\n"
    "        * Numerical values (quantities, monetary amounts, scores, counts, etc.)\n"
    "        * Specific dates or times (unless Rule 2 applies for granularity of the *same event*)\n"
    "    * **Exception:** Redundancy based on numerical values (Rule 1) can *only* be considered if the numbers/quantities mentioned are **exactly identical** *and* the rest of the core meaning is also identical. Any difference, no matter how small, means the statements are distinct unless it's a date granularity issue covered by Rule 2.\n"
    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation, evaluate if both convey unique, valuable information. If so, **retain both**. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "    * For example, both 'joining a company' and 'being employed at the company' should be retained. For cases like 'located at' vs. 'held at', retain only one.\n"
    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships unrelated to the rules above, **retain all** of them.\n\n"
    "**Processing Steps:**\n"
    "1.  Consider the list of statements provided.\n"
    "2.  Apply the rules above meticulously to filter the statements.\n"
    "3.  Prioritize strict interpretation of rules 2 and 3 regarding dates and numbers.\n"
    "4.  Aim to represent the unique information from the original set accurately and concisely according to these rules.\n\n"
    "**Output Format:**\n"
    "* Print **only** the zero-based indices of the retained statements, separated by commas.\n"
    "* Do not include headers, explanations, or any text other than the comma-separated indices.\n"
    "**Example Output:**\n"
    "1, 3\n\n"
    "**Example Application Illustrating Rules (Conceptual):**\n"
    "* Group A (Rule 3 - Different Numbers): 'Team X scored 10 points.', 'Team X scored 11 points.' -> Retain both indices.\n"
    "* Group B (Rule 2 - Date Specificity): 'The event happened in 2023.', 'The event happened on Jan 5th, 2023.' -> Retain only the index for the second statement. Because the second sentence includes the information from the first.\n"
    "* Group C (Rule 1/3 - Identical Numbers): 'The budget is $5,000.', 'The budget is $5,000.00.' -> Treat as identical numbers. If rest of the meaning is the same, keep one index.\n"
    "* Group D (Rule 3 - Different Dates): 'She arrived on Monday.', 'He arrived on Tuesday.' -> Retain both indices.\n"
    "* Group E (Rule 1 - Paraphrasing): '2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.' -> Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
    "* Group F (Rule 2 - Subsumption/More Specific Info): 'Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.' -> Statement 0 ('Won against') is more specific and implies 1 ('Played against'). Keep index 0.\n"
    "**Statements to Process:**\n{redundant_relationship}\n\n"
    "**Output:**\n"
    "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    # "<|im_end|>\n"
    # "<|im_start|>assistant"
)


knowledge_graph_redundant_relationship_prompt_qwen_instruct_bak = (
    "<|im_start|>system\n"
    "You are an AI assistant expert in identifying semantic relationships and eliminating redundancy based on provided rules.<|im_end|>\n"
    "<|im_start|>user\n"
    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy.\n\n"
    "Rules:\n"
    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only one instance, preferably the clearest, most complete, or most standard representation.\n"
    "2.  **Information Subsumption:** If one statement's meaning is fully contained within another more specific or informative statement, retain only the more informative one.\n"
    "3.  **Sensitivity to Key Details (Numbers, Dates, Quantities):** Be sensitive to differences in quantities, numbers, monetary values, dates, or other factual details.\n"
    "    - If these differences clearly indicate distinct events, facts, times, or significantly different quantities, treat the statements as having different meanings and retain them accordingly.\n"
    "    - However, if the context strongly suggests the statements refer to the same underlying fact or event but have minor variations in these details (e.g., slight rounding differences, near-identical numbers likely due to reporting variations), prioritize the shared core meaning and treat them as similar, keeping only one representative statement according to rule #1.\n"
    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation (e.g., an action and its resulting state, like 'transferred to' vs. 'plays for'), evaluate if both convey unique, valuable information. If so, retain both. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships, retain all of them.\n\n"
    "Processing Steps:\n"
    "- Consider the statements provided below.\n"
    "- Apply the rules above to filter the statements.\n"
    "- Aim to represent the unique information from the original set accurately and concisely.\n\n"
    "Output Format:\n"
    "- Print the zero-based indices of the retained statements, separated by commas.\n"
    "- Do not provide explanations—only output the structured result as specified.\n\n"
    "Example Output:\n"
    "1, 3\n\n"
    "Example Application Walkthrough:\n"
    "Group 0 (0, 1): 0. 2023 citrus bowl Features teams Lsu tigers, 1. Lsu tigers Will play in 2023 citrus bowl. -> Similar. Keep one (e.g., 0 or 1).\n"
    "Group 1 (2, 3): 2. 2023 citrus bowl Features teams Purdue boilermakers, 3. Purdue boilermakers Will play in 2023 citrus bowl. -> Similar. Keep one (e.g., 2 or 3).\n"
    "Group 2 (4, 5): 4. 2022 citrus bowl Featured Kentucky wildcats, 5. Kentucky wildcats Played in 2022 citrus bowl. -> Similar. Keep one (e.g., 4 or 5).\n"
    "Group 3 (6, 7): 6. Kentucky wildcats Won against Iowa hawkeyes, 7. Iowa hawkeyes Played against Kentucky wildcats. -> 6 implies 7 and is more specific. Keep 6.\n"
    "Group 4 (8, 9): 8. 2022 citrus bowl Featured Iowa hawkeyes, 9. Iowa hawkeyes Played in 2022 citrus bowl. -> Similar. Keep one (e.g., 8 or 9).\n"
    "Group 5 (10, 11): 10. Tayvion robinson Plays for Kentucky, 11. Tayvion robinson Transferred to Kentucky. -> Different aspects (state vs action leading to state). Both convey potentially unique info. Keep both 10, 11.\n\n"
    "Statements to process:\n"
    "{redundant_relationship}\n\n"
    "Output the indices of the retained statements according to the format specified:\n"
    "<|im_end|>\n"
    "<|im_start|>assistant"
)



knowledge_graph_redundant_entity_prompt = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
    "I will provide multiple groups of entities from a knowledge graph, each with a unique ID. Within each group, some entities may have identical meanings. Your task is to identify the largest set of entities with the same meaning in each group and output them in the specified format.\n\n"
    "Rules:\n"
    "Find the largest set of entities that share the exact same meaning within each group.\n"
    "For numbers (e.g., time, quantity), they must be identical to be considered equivalent.\n"
    "Differences in word order do NOT change the meaning. Such entities should be grouped together.\n"
    "When both an abbreviation and a full name appear for a date-related entity (e.g., 'Jan 2024' vs. 'January 2024'), retain the full name as the representative entity.\n"
    "Choose the most clear and complete entity as the first in the set, followed by other equivalent entities.\n"
    "Use only entity IDs (do not output entity names).\n"
    "Separate entities in a set with , and output each set on a new line.\n"
    "If no entities share the same meaning in a group, output nothing for that group.\n"
    "Do not output anything other than the required formatted result.\n\n"
    "Correct Output Format (Example):\n"
    "3, 5, 6\n"
    "12, 11\n"
    "15, 14\n\n"
    "Entities:\n{redundant_entity}<|im_end|>\n"
    "<|im_start|>assistant"
    )


knowledge_graph_redundant_entity_prompt_v2 = (
    "You will be given a single group of entities, each with a unique ID and associated text. Your task is to identify the only largest sets of entities within this group that share exactly the same meaning.\n"
    "Two entities are considered to have the same meaning only if they can always be used interchangeably in any context, without altering the meaning of the sentence. Follow these specific rules:\n"
    "For numbers, including dates, quantities, and monetary values, entities must be exactly the same to be considered equivalent.\n"
    "Prefer full spellings over abbreviations (e.g., 'January 2024' is preferred over 'Jan 2024').\n"
    "Differences in word order do not change meaning and such entities should be grouped together.\n"
    "Choose the most complete and grammatically correct entity as the representative (i.e., the first ID in the set).\n"
    "Use only entity IDs in the output, not entity names or texts.\n"
    "Separate IDs within the same equivalence set using commas, e.g., ID1, ID2, ID3.\n"
    "If no entities in the group have the same meaning, output: No results matching your criteria.\n"
    "Output only the result—no explanation or extra text.\n\n"
    "Output Format (Example):\n"
    "5, 3, 6\n\n"
    "Entities:\n{redundant_entity}"
)

knowledge_graph_redundant_entity_prompt_v2_qwen = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "You will be given a single group of entities, each with a unique ID and associated text. Your task is to identify the only largest sets of entities within this group that share exactly the same meaning.\n"
    "Two entities are considered to have the same meaning only if they can always be used interchangeably in any context, without altering the meaning of the sentence. Follow these specific rules:\n"
    "Prefer full spellings over abbreviations (e.g., 'January 2024' is preferred over 'Jan 2024').\n"
    "For numbers, including dates, quantities, and monetary values, entities must be exactly the same to be considered equivalent.\n"
    "Differences in word order do not change meaning and such entities should be grouped together.\n"
    "Choose the most complete and grammatically correct entity as the representative (i.e., the first ID in the set).\n"
    "Use only entity IDs in the output, not entity names or texts.\n"
    "Separate IDs within the same equivalence set using commas, e.g., ID1, ID2, ID3.\n"
    "If no entities in the group have the same meaning, output: No results matching your criteria.\n"
    "Output only the result—no explanation or extra text.\n\n"
    "Output Format (Example):\n"
    "5, 3, 6\n\n"
    "Entities:\n{redundant_entity}<|im_end|>\n" #  /no_think
    "<|im_start|>assistant\n"
    # "<think>\n"
)

knowledge_graph_redundant_entity_prompt_v3_qwen_en = (
    "<|im_start|>system\n"
    "You are an AI assistant specializing in entity deduplication within knowledge graphs. Your core task is to identify all the largest sets of entities that are semantically identical, based on the rules provided by the user, from a given group of entities.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "You will be given a group of entities, each with a unique ID and associated text. Please carefully read and strictly adhere to all the rules below to identify all the largest sets of entities within this group that share exactly the same meaning.\n\n"
    "**Definition of 'Semantically Identical':**\n"
    "Two or more entities are considered semantically identical if and only if they can always be used interchangeably in any context without altering the overall meaning of the sentence or expression.\n\n"
    "**Core Rules:**\n"
    "1.  **Largest Set Priority:** You need to find the equivalent entity set(s) that are the largest in size (containing the most entities). If there are multiple distinct equivalent sets of the same largest size, list all of them.\n"
    "2.  **Representative Entity Selection and Ordering:** In each identified equivalent set, select the ID corresponding to the entity text that is the most complete, most normative, and grammatically most correct as the first ID in that set (this is the representative ID). Other IDs within the set can be in any order, but it's recommended to sort them by their numerical or alphabetical value.\n"
    "3.  **Normativeness and Completeness:** Prefer complete and standard forms over abbreviations or non-standard writings. For example, assuming equivalent meaning, 'January 2024' is preferred over 'Jan 2024' or '24 Jan'.\n"
    "4.  **Numerical and Date Equivalence Principle:**\n"
    "    * **Monetary/Quantitative Values:** Entities representing the same numerical value with the same currency or unit of measure, despite different formatting (e.g., different number of decimal places), should be considered equivalent. For example, '100 dollars' is equivalent to '100.00 dollars'. However, entities with different actual values (e.g., '100 dollars' vs '101 dollars') or different units (e.g., '100 dollars' vs '100 euros') are not equivalent.\n"
    "    * **Dates/Times:** Entities representing the exact same date or time, despite different formatting, should be considered equivalent. For example, '2024-01-05' is equivalent to 'January 5, 2024'.\n"
    "    * The core criterion is whether they represent the exact same value, date, or point in time after reasonable normalization (e.g., unifying decimal places, parsing date formats). Any substantive difference means they are not equivalent.\n"
    "5.  **Word Order Flexibility:** If reversing the word order in an entity's text does not change its core meaning (e.g., 'Peking University' and 'University Peking' explicitly refer to the same institution within a specific knowledge base), they can be considered equivalent. However, the selection of the representative entity must still follow rules 2 and 3, choosing the most standard word order.\n\n"
    "**Output Format Requirements:**\n"
    "* **Output IDs Only:** The result should only contain entity IDs. Do not include entity names, original text, or any explanatory remarks.\n"
    "* **ID Separator:** Use an English comma (`,`) to separate IDs within the same equivalent set.\n"
    "* **Newline for Multiple Sets:** If multiple equivalent sets of the same largest size are found, output each set on a new line.\n"
    "* **No Results Case:** If no entity sets meeting the 'semantically identical' criteria are found in the input (i.e., no group of at least two entities are semantically identical), output exactly: `No results matching your criteria.`\n"
    "* **No extraneous text:** Apart from the ID sets or the 'No results...' message, do not output any other explanations or text.\n\n"
    "**Output Format Examples:**\n"
    "If one largest equivalent set is found:\n"
    "5, 3, 6\n\n"
    "If two equally largest equivalent sets are found:\n"
    "5, 3, 1\n"
    "6, 8, 9\n\n"
    "If no equivalent entity sets are found:\n"
    "No results matching your criteria.\n\n"
    "**Entities to be processed are listed below:**\n"
    "{redundant_entity}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

knowledge_graph_redundant_entity_prompt_v4_qwen_en = (
    "<|im_start|>system\n"
    "You are an AI assistant specializing in entity deduplication within knowledge graphs. Your core task is to identify all the largest sets of entities that are semantically identical, based on the rules and provided information (entity text AND related statements), from a given group of entities.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "You will be given a group of entities. Each entity will have a unique ID, associated text (the entity's name or label), and a list of related facts presented as natural language statements derived from knowledge graph triples. Please carefully read and strictly adhere to all the rules below to identify all the largest sets of entities within this group that share exactly the same meaning.\n\n"
    "**Definition of 'Semantically Identical':**\n"
    "Two or more entities are considered semantically identical if and only if they represent the exact same real-world concept, object, person, event, value, etc., and can always be used interchangeably in any context without altering the overall meaning.\n\n"
    "**Core Rules:**\n"
    "1.  **Evidence from Related Statements (Triples):** Each entity is accompanied by natural language statements derived from associated knowledge graph triples. These statements provide crucial contextual information about the entity.\n"
    "    *   If the statements associated with two entities strongly imply they refer to the same thing (e.g., they share multiple consistent properties or relationships like location, date of birth, type of organization), this is strong evidence for identity.\n"
    "    *   If the statements associated with two entities clearly contradict each other or describe distinct, incompatible properties or relationships (e.g., different locations, different founding dates, different types), they **cannot** be considered semantically identical, **regardless of the similarity of their entity text**.\n"
    "    *   Use these statements as primary evidence alongside the entity text to make your determination.\n\n"
    "2.  **Largest Set Priority:** You need to find the equivalent entity set(s) that are the largest in size (containing the most entities). If there are multiple distinct equivalent sets of the same largest size, list all of them.\n\n"
    "3.  **Representative Entity Selection and Ordering:** In each identified equivalent set, select the ID corresponding to the entity text that is the most complete, most normative (standard), and grammatically most correct as the first ID in that set (this is the representative ID). Other IDs within the set can be in any order, but it's recommended to sort them by their numerical or alphabetical value for consistency.\n\n"
    "4.  **Normativeness and Completeness (for Representative Selection):** Prefer complete and standard forms over abbreviations or non-standard writings when selecting the representative entity. For example, assuming equivalence confirmed by all rules including triple statements, 'January 2024' is preferred over 'Jan 2024' or '24 Jan'.\n\n"
    "5.  **Numerical and Date Equivalence Principle (Considered alongside statements):**\n"
    "    *   **Monetary/Quantitative Values:** Entities representing the same numerical value with the same currency or unit of measure, despite different formatting (e.g., different number of decimal places), should be considered equivalent *if confirmed by related statements*. For example, '100 dollars' is equivalent to '100.00 dollars' *if both are associated with the same transaction confirmed by statements*. However, different actual values or different units are not equivalent.\n"
    "    *   **Dates/Times:** Entities representing the exact same date or time, despite different formatting, should be considered equivalent *if confirmed by related statements*. For example, '2024-01-05' is equivalent to 'January 5, 2024' *if both are associated with the same event confirmed by statements*.\n"
    "    *   The core criterion is whether they represent the exact same value, date, or point in time after reasonable normalization *and* this is supported or at least not contradicted by the related statements. Any substantive difference means they are not equivalent.\n\n"
    "6.  **Word Order Flexibility (Considered alongside statements):** If reversing the word order in an entity's text does not change its core meaning (e.g., 'Peking University' and 'University Peking' explicitly refer to the same institution *and their related statements are consistent*), they can be considered equivalent. However, the selection of the representative entity must still follow rules 3 and 4, choosing the most standard word order.\n\n"
    "**Important Note:** Rule 1 (Evidence from Related Statements) is paramount. If statements contradict identity, the entities are not identical, even if other rules based purely on text might suggest similarity.\n\n"
    "**Output Format Requirements:**\n"
    "*   **Output IDs Only:** The result should only contain entity IDs. Do not include entity names, original text, related statements, or any explanatory remarks.\n"
    "*   **ID Separator:** Use an English comma (`,`) to separate IDs within the same equivalent set.\n"
    "*   **Newline for Multiple Sets:** If multiple equivalent sets of the same largest size are found, output each set on a new line.\n"
    "*   **No Results Case:** If no entity sets meeting the 'semantically identical' criteria are found in the input (i.e., no group of at least two entities are semantically identical based on all rules), output exactly: `No results matching your criteria.`\n"
    "*   **No extraneous text:** Apart from the ID sets or the 'No results...' message, do not output any other explanations or text.\n\n"
    "**Output Format Examples:**\n"
    "If one largest equivalent set is found:\n"
    "5, 3, 6\n\n"
    "If two equally largest equivalent sets are found:\n"
    "5, 3, 1\n"
    "6, 8, 9\n\n"
    "If no equivalent entity sets are found:\n"
    "No results matching your criteria.\n\n"
    "**Entities to be processed are listed below:**\n"
    "{redundant_entity}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# knowledge_graph_redundant_entity_prompt_v5_qwen_en 使用相关陈述作为一种辅助判断，不作为主要依据
knowledge_graph_redundant_entity_prompt_v5_qwen_en = ( 
    "<|im_start|>system\n"
    "You are an AI assistant specializing in entity deduplication within knowledge graphs. Your core task is to identify all the largest sets of entities that are semantically identical, based on the rules and provided information (entity text AND related statements), from a given group of entities.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "You will be given a group of entities. Each entity will have a unique ID, associated text (the entity's name or label), and a list of related facts (up to 5 statements per entity) presented as natural language statements derived from knowledge graph triples. Please carefully read and strictly adhere to all the rules below to identify all the largest sets of entities within this group that are considered semantically identical.\n\n"
    "**Definition of 'Semantically Identical':**\n"
    "Two or more entities are considered semantically identical if their corresponding texts suggest they represent the same real-world concept, object, person, event, value, etc., and they can be used interchangeably in most general contexts. Slight ambiguity arising from such substitution is acceptable, provided it does not significantly alter the primary meaning of the context. The associated statements for the entities (see Core Rule 1) will be used to further support or refute this initial assessment based on text.\n\n"
    "**Core Rules:**\n"
    "1. **Primary Judgment by Entity Text, Auxiliary Judgment by Related Statements:**\n"
    "	The primary basis for determining if entities are semantically identical is their associated text (name or label).\n"
    "	Each entity is accompanied by up to 5 natural language statements derived from associated knowledge graph triples. These statements serve as auxiliary evidence to support or refute the assessment made based on the entity text.\n"
    "	It is not required that the statements for different entities describe identical or nearly identical facts. Instead, the focus is on whether the general nature or context hinted at by these statements indicates that the entities could plausibly represent the same thing.\n"
    "	**Compatibility Check using Statements:**\n"
    "		If the entity texts are similar, and the associated statements for these entities hint at compatible contexts or types (e.g., one entity's statements suggest it's a 'competition name', and another's also suggest a 'competition name'), this increases the probability that the entities are identical.\n"
    "		Conversely, if the statements provide strong contradictory hints about the nature of the entities (e.g., one entity's statements imply it's a 'mobile phone model,' while another's imply it's a 'type of fruit,' even if their texts were identical like 'Apple'), they should generally be considered different entities. Such clear contradictions from statement hints can override text-based similarity.\n"
    "	Essentially, after an initial assessment based on entity text, use the statements to check if the potential identity is plausible or if there's a clear indication they represent different things despite textual similarity.\n"
    "2. **Largest Set Priority:** You need to find the equivalent entity set(s) that are the largest in size (containing the most entities). If there are multiple distinct equivalent sets of the same largest size, list all of them.\n"
    "3. **Representative Entity Selection and Ordering:** In each identified equivalent set, select the ID corresponding to the entity text that is the most complete, most normative (standard), and grammatically most correct as the first ID in that set (this is the representative ID). Other IDs within the set can be in any order, but it's recommended to sort them by their numerical or alphabetical value for consistency.\n"
    "4. **Normativeness and Completeness (for Representative Selection):** Prefer complete and standard forms over abbreviations or non-standard writings when selecting the representative entity. For example, assuming equivalence confirmed by all rules including statement hints, 'January 2024' is preferred over 'Jan 2024' or '24 Jan'.\n"
    "5.	**Numerical and Date Equivalence Principle (Considered alongside statements):**\n"
    "	**Monetary/Quantitative Values:** Entities representing the same numerical value with the same currency or unit of measure, despite different formatting (e.g., different number of decimal places), should be considered equivalent if this is supported or not contradicted by the hints from related statements. For example, '100 dollars' is equivalent to '100.00 dollars' if their related statements are compatible with them referring to the same specific transaction or item. However, different actual values or different units (unless demonstrably equivalent and supported by statements) are not equivalent.\n"
    "	**Dates/Times:** Entities representing the exact same date or time, despite different formatting, should be considered equivalent if this is supported or not contradicted by the hints from related statements. For example, '2024-01-05' is equivalent to 'January 5, 2024' if their related statements are compatible with them referring to the same specific event or point in time.\n"
    "	The core criterion is whether they represent the exact same value, date, or point in time after reasonable normalization and this is plausible given the contextual hints from the related statements. Any substantive difference means they are not equivalent.\n"
    "6. **Word Order Flexibility (Considered alongside statements):** If reversing the word order in an entity's text does not change its core meaning (e.g., 'Peking University' and 'University Peking' explicitly refer to the same institution and their related statements provide compatible hints), they can be considered equivalent. However, the selection of the representative entity must still follow rules 3 and 4, choosing the most standard word order.\n\n"
    "**Important Note:** While entity text is the primary basis for judgment (Rule 1), the auxiliary information from related statements is crucial. If the hints from statements clearly and strongly contradict the possibility of identity suggested by the text (e.g., indicating fundamentally different types of real-world objects like a phone model versus a fruit, when text alone might be ambiguous), then the entities should be considered different.\n"
    "Output Format Requirements:\n"
    "	Output IDs Only: The result should only contain entity IDs. Do not include entity names, original text, related statements, or any explanatory remarks.\n"
    "	ID Separator: Use an English comma (,) to separate IDs within the same equivalent set.\n"
    "	Newline for Multiple Sets: If multiple equivalent sets of the same largest size are found, output each set on a new line.\n"
    "	No Results Case: If no entity sets meeting the 'semantically identical' criteria are found in the input (i.e., no group of at least two entities are semantically identical based on all rules), output exactly: No results matching your criteria.\n"
    "	No extraneous text: Apart from the ID sets or the 'No results...' message, do not output any other explanations or text.\n\n"
    "Output Format Examples:\n"
    "If one largest equivalent set is found:\n"
    "5, 3, 6\n"
    "If two equally largest equivalent sets are found:\n"
    "5, 3, 1\n"
    "6, 8, 9\n"
    "If no equivalent entity sets are found:\n"
    "No results matching your criteria.\n\n"
    "Entities to be processed are listed below:\n"
    "{redundant_entity}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

knowledge_graph_redundant_entity_check_prompt= (
    "I will provide multiple groups of entities. Each group contains entities that are similar in meaning.\n"
    "Your task is to determine, for each group, whether all entities have exactly the same meaning.\n"
    "Two entities are considered to have exactly the same meaning if they can be used interchangeably in any context without changing the meaning of the sentence. If there is any ambiguity or difference in meaning or usage, they are not considered equivalent.\n"
    "For numbers, dates, quantities, or monetary values, two entities are equivalent only if they are exactly the same.\n"
    "For each group, respond with:\n"
    "1 if all entities in the group have exactly the same meaning,\n"
    "0 otherwise.\n"
    "Output one line per group, containing only 1 or 0.\n\n"
    "Entities:\n{redundant_entity}"
)

knowledge_graph_score_correction_prompt_true = (
    "Task: Evaluate each short statement retrieved in the 'Retrieve Result' and determine whether it answers the given 'Query'.\n\n"
    "If a statement leads to the correct answer found in the Fact-based Answer, classify it as a correct triple.\n"
    "If a statement leads to an incorrect answer, classify it as an incorrect triple.\n"
    "If a statement does not answer the query, ignore it.\n\n"
    "⚠ Key Considerations:\n"
    "Pay close attention to differences between the statement and the query in terms of time, location, entities, numerical values, active/passive voice, and other relevant aspects.\n"
    "Do not hastily classify an unrelated statement as incorrect—only statements that explicitly lead to an incorrect answer should be labeled as such.\n"
    "Ensure that the retrieved answers match the real answers exactly, especially when dealing with numbers and factual details.\n"
    "For both correct and incorrect statements, assign a score from 1 to 3 based on their degree of correctness or incorrectness:\n\n"
    "1: Partially correct (or partially incorrect)\n"
    "2: Mostly correct (or generally incorrect)\n"
    "3: Completely correct (or completely incorrect)\n"
    "A higher score indicates a stronger degree of correctness or incorrectness.\n\n"
    "Example Output:\n"
    "Correct: 2:3 5:2 12:1\n"
    "Error: 3:3 7:2\n\n"
    "---------------------\n"
    "Query: {query}\n"
    "Fact-based Answer : {response}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Task: Based on the information provided above, classify each statements accordingly.Do not provide explanations—only output structured results as specified above."
)

knowledge_graph_score_correction_prompt_false = (
    "Task: Identify the erroneous statement(s) from the retrieved results that caused the wrong answer, based solely on the question and the incorrect answer.\n\n"
    "Relevance: Only consider statements directly related to the question.\n"
    "Example: If the question is about 2022 and a statement is about 2021, ignore it.\n"
    "Match Details: If a statement’s details (especially time and numerical information) exactly match those in the wrong answer, classify it as erroneous.\n"
    "Scoring: Rate the harmfulness of each erroneous statement on a scale from 1 to 3 (3 indicates the most harmful).\n\n"
    "The core error in answering the question must be consistent.\n"
    "For example:\n"
    "Question: Which country won the 2022 FIFA World Cup?\n"
    "'France won the World Cup.' → Incorrect\n"
    "'Argentina defeated France.' → Irrelevant\n"
    "'France participated in the 2022 World Cup.' → Irrelevant\n"
    "'France won the 2018 World Cup.' → Irrelevant\n\n"
    "Provide your results in a structured format with no additional explanation.\n"
    "Example Output:\n"
    "Error: 3:3 7:2\n\n"
    "---------------------\n"
    "Query: {query}\n"
    "Wrong Answer : {response}\n"
    "Retrieve Result: {retrieve_result}\n"
    "Remember: Do not provide explanations—only output structured results as specified above."
)


knowledge_graph_score_feedback_prompt_4_1_latest = (
    "You are an AI assistant tasked with evaluating the relevance and correctness of retrieved knowledge graph triples in relation to a given query and a model-generated answer. Your task is to assign scores only to relevant triples while ignoring unrelated ones.\n\n"
    "Scoring Criteria:\n"
    "Regardless of whether the last answer was correct, if the last answer did not directly answer the question but instead indicated that the retrieved information was insufficient, irrelevant, or that more information was needed (e.g., responses like 'without more specific information' or 'need a bit more context or details about the xxx').\n"
    "    Do not score any triples.Just output the sentence 'Insufficient search information'.\n"
    "If the last answer was correct:\n"
    "    Assign a score from 1 to 3 to triples that directly contain the correct answer. A higher score (closer to 3) means the triple is highly relevant to the correct answer.\n"
    "    Assign a score from 1 to 3 to triples that contain incorrect or misleading answers. A higher score (closer to 3) means the triple is highly misleading.\n"
    "If the last answer was incorrect:\n"
    "    Assign a score from 1 to 3 to triples that contain the incorrect answer. A higher score (closer to 3) means the triple is highly misleading.\n"
    "    No need to find correct triples and output rating.\n\n"
    "Rules for Relevance:\n"
    "Only evaluate triples that are directly relevant to the query.\n"
    "If the query is about the winner of the 2022 Citrus Bowl, ignore information about winners of other years (e.g., 2019, 2023).\n"
    "If the query is about the total number of vehicles Tesla delivered in 2021, ignore triples about Tesla's production numbers or deliveries in other years.\n"
    "Do not assign scores to triples that do not provide an answer-related fact.\n"
    "If a triple just provides background information (e.g., 'Tesla builds all-electric vehicles' or 'The Citrus Bowl was held in Orlando, Florida'), do not score it.\n"
    "Pay special attention to numerical values.\n"
    "Be cautious when evaluating quantities, dates, and statistics to ensure correctness.\n"
    "If multiple triples contain relevant numbers, prefer those that are most precise and clearly related to the query.\n\n"
    "Use the following format to return the results:\n"
    "For correct triples: 'Correct: <triple_index>:<score> <triple_index>:<score> ...'\n"
    "For incorrect triples: 'Error: <triple_index>:<score> <triple_index>:<score> ...'\n"
    "If there are no correct or incorrect triples, return an empty string.\n"
    "Example Output:\n"
    "Correct: 2:3 5:2 12:1\n"
    "Error: 3:3 7:2\n\n"
    "Do not provide explanations—only output structured results as specified above.\n\n"
    "Query: {question}\n"
    "Last Answer ({flag_TF}): {last_response}\n"
    "Retrieve Result:{knowlege_sentence}\n"
)

knowledge_graph_score_feedback_prompt_gemini = (
    "You are an AI assistant tasked with evaluating the relevance and correctness of retrieved knowledge graph triples in relation to a given query and the model's previous response (`Last Response`). This evaluation occurs in the second step of a process, focusing on the triples used or relevant to the `Last Response`. Your task is to assign scores *only* to triples that are directly relevant to the query's specific factual question, ignoring irrelevant ones.\n\n"
    "**Crucially, your evaluation *must* be exceptionally strict regarding numerical values, dates, quantities, statistics, and other precise factual details. Exact, literal matches are paramount.**\n\n"
    "Pre-condition for Scoring:\n"
    "1. Handling 'Insufficient Information':\n"
    "Regardless of whether the `Last Response` was correct or incorrect, if it explicitly states that it cannot answer the query due to insufficient or irrelevant information, or requests more details (e.g., contains phrases like 'insufficient information to determine', 'need more context', 'based on the provided snippets...'), do not evaluate or score any triples.\n"
    "In this specific case, your only output should be the phrase 'Insufficient information'. Do not proceed with the scoring rules below.\n\n"
    "Evaluation Criteria based on Last Response Correctness (Apply only if the pre-condition above is NOT met):\n\n"
    "1.  **If the Last Response was Correct :**\n"
    "    You will evaluate *two* potential types of triples in this case:\n"
    "    *   **Score Supporting Correct Triples:**\n"
    "        Assign a score **only** to triples that **precisely and unambiguously contain the exact information** required to answer the query correctly. The information within the triple (subject, predicate, object) must **perfectly match** the query's specific constraints (e.g., the correct value for the requested attribute, the entity corresponding to the specified time period/date/number).\n"
    "        Score from 1 to 3. A score of **3** indicates the triple is highly relevant, precise, and crucial for confirming the correct answer, containing the exact required fact. Lower scores (2 or 1) indicate relevance but perhaps less directness or completeness while still being factually supportive and precise.\n"
    "        *   **Strict Constraint Matching:** Triples referring to different constraints (e.g., a different year, a related but different metric), even if about the same core entity, are **not** considered 'Correct' as they do not provide the specific fact requested.\n"
    "    *   **Score Contradictory/Error Triples:**\n"
    "        Assign a score **only** to triples containing information that **directly contradicts the actual correct answer** or presents potentially misleading information directly related to the query's core question. The contradiction must be factual and precise (e.g., a wrong number, different date for the same event).\n"
    "        Score from 1 to 3. A score of **3** indicates the triple is highly misleading or presents a direct, factual contradiction to the correct answer in the context of the query.\n\n"
    "2.  **If the Last Response was Incorrect :**\n"
    "    You will evaluate *only one* type of triple in this case:\n"
    "    *   **Score Triples Supporting the Error (Highly Strict Matching):**\n"
    "        Assign a score **only** to triples where the relevant factual information (typically the object/value in the triple related to the query's question) **exactly and literally matches the specific incorrect information** present in the `Last Response`.\n"
    "        *   **Exact Value Mandate:** The number, date, quantity, or specific entity mentioned in the triple must be **identical** to the erroneous one in the `Last Response`. No approximations, rounding, or 'close' values are acceptable.\n"
    "        *   **Predicate/Context Match:** The predicate and context must also align with the nature of the error. If the error was about 'deliveries', a triple about 'production' should generally not be scored, even if the number is coincidentally similar, unless the `Last Response` itself confused these terms in a way the triple reflects.\n"
    "        Score from 1 to 3. A score of **3** indicates the triple provides direct, unambiguous, and **exact** support for the specific error made in the `Last Response`. Lower scores (2 or 1) might be used if the triple contains the exact error value but perhaps requires slight inference or has minor ambiguities, while still being the source of the precise error. **However, prioritize exactness above all; if the value isn't identical, do not score.**\n"
    "    *   **Crucially, do not evaluate or score triples that contain the actual correct answer or any information *other than the specific error* in this scenario.** Your focus here is solely on identifying the source of the **exact** mistake.\n\n"
    "Rules for Relevance and Scoring:\n\n"
    "*   **Direct Relevance:** Only evaluate triples that directly address the core factual question posed by the query.\n"
    "*   **Precision is Paramount (Critical Rule - Applied Universally):**\n"
    "    *   **Strict Matching for Values (Essential):** Exact numerical values, dates (day, month, year), quantities, and specific identifiers are **non-negotiable** for *any* scoring.\n"
    "    *   **Correctness Evaluation:** A triple supporting a correct answer *must* match the required details **perfectly**.\n"
    "    *   **Error Evaluation (Reiteration):** A triple supporting an incorrect answer *must* **perfectly mirror the specific erroneous value and context** from the `Last Response`. Near matches, different quantities or qualifiers (e.g., 'over X' vs 'X'), or related but different metrics (e.g., 'Produced X' when the error was about 'Delivered X') **do not qualify** for an error score. If the triple doesn't contain the *literal* wrong fact from the `Last Response`, it should not be scored under 'Error'.\n"
    "    *   **Constraint Mismatches:** Triples failing the query's specific constraints (e.g., wrong year, different metric) are irrelevant for scoring as 'Correct'. They are also irrelevant for scoring as 'Error' unless they **perfectly match the specific error** made in the `Last Response` regarding that mismatched constraint/value.\n"
    "*   **Ignore Background/Contextual Information:** Do not score triples providing only general context.\n"
    "*   **Avoid Scoring Partial Matches (Unless Error-Specific & Exact):** Do not score triples just sharing keywords. The triple must provide the specific complete fact (correct or exactly incorrect). Related but distinct attributes are not scored unless that distinct attribute *is* the precise point of error and is *identically represented* in the triple.\n\n"
    "Output Format:\n\n"
    "*   For scored correct triples: `Correct: <triple_index>:<score> <triple_index>:<score> ...`\n"
    "*   For scored error/misleading triples: `Error: <triple_index>:<score> <triple_index>:<score> ...`\n"
    "*   If no triples are scored as either Correct or Error after evaluation: return `No feedback`\n"
    "*   If the 'Insufficient Information' pre-condition is met: return `Insufficient information`\n\n"
    "Example Output Structures:\n"
    "Correct: 2:3 5:2\n"
    "Error: 3:3 7:1\n"
    "Correct: 1:3\n"
    "Error: 4:3\n"
    "No feedback\n"
    "Insufficient information\n\n"
    "**Do not provide explanations—only output structured results as specified above.**\n\n"
    "Query: {question}\n"
    "Last Answer ({flag_TF}): {last_response}\n"
    "Retrieve Result:{knowlege_sentence}\n"
)

# knowledge_graph_score_feedback_prompt_qwen3 = (
#   "You are an AI assistant tasked with evaluating the factual relevance and correctness of retrieved sets of inferential statements (each statement in natural language form) in relation to a given query and the model's previous response (Last Response). This evaluation occurs in the second step of a process, focusing on sets of statements used or relevant to the Last Response. Your task is to assign scores only to sets of statements that, when their information is combined and considered as a whole, are directly relevant to the query's specific factual question, ignoring irrelevant ones.\n\n"
#   "Crucially, your evaluation must be exceptionally strict regarding numerical values, dates, quantities, statistics, named entities, and other precise factual details derived from the collective information of the set. Exact, literal matches of the information conveyed by the entire set are paramount.\n\n"
#   "Pre-condition for Scoring:\n"
#   "1. Handling 'Insufficient Information':\n"
#   "Regardless of whether the Last Response was correct or incorrect, if it explicitly states that it cannot answer the query due to insufficient or irrelevant information, or requests more details (e.g., contains phrases like 'insufficient information to determine', 'need more context', 'based on the provided snippets/statements...'), do not evaluate or score any sets of statements.\n"
#   "In this specific case, your only output should be the phrase 'Insufficient information'. Do not proceed with the scoring rules below.\n\n"
#   "Evaluation Criteria based on Last Response Correctness (Apply only if the pre-condition above is NOT met):\n\n"
#   "1.  If the Last Response was Correct :\n"
#   "    You will evaluate two potential types of sets of statements in this case:\n"
#   "    * Score Supporting Correct Sets of Statements:\n"
#   "        Assign a score only to sets of statements where the combined information from all statements within the set, when taken together, provides specific and exact facts that directly justify the correct answer. The set as a whole must unambiguously and explicitly lead to or express the key facts needed to answer the question correctly. It is not necessary for every individual statement in the set to contain the complete correct answer, as long as their combination does.\n"
#   "        Score from 1 to 3:\n"
#   "        - 3 = The set is highly relevant, its collective information is precise and essential; the combined statements provide the exact fact or strong logical support for the answer.\n"
#   "        - 2 = The set is relevant and its collective information is mostly correct, but the combined information may be slightly indirect, or some statements within the set might be redundant if others already establish the point.\n"
#   "        - 1 = The set provides weak but still factual collective support for the correct answer.\n\n"
#   "    * Score Contradictory or Misleading Sets of Statements:\n"
#   "        Assign a score only to sets of statements where the combined information, when taken together, contains factual inaccuracies or strongly suggests incorrect conclusions that would contradict the actual correct answer. It is not necessary for every individual statement in the set to be contradictory, as long as the collective meaning of the set is contradictory.\n"
#   "        Score from 1 to 3 depending on how directly and convincingly the set as a whole would mislead or contradict the correct answer.\n\n"
#   "2.  If the Last Response was Incorrect :\n"
#   "    You will evaluate only one type of set of statements in this case:\n"
#   "    * Score Sets of Statements Supporting the Error (Exact Match Rule for Collective Information):\n"
#   "        Assign a score only to sets of statements whose collective factual content, when all statements are considered together, exactly and literally matches the specific incorrect information present in the Last Response. It is not necessary for every individual statement in the set to contain the full error, as long as their combination directly supports the specific error.\n"
#   "        Score from 1 to 3, with 3 indicating the set's combined information provides direct, unambiguous, and exact factual support for the specific error. Do not score sets whose collective information provides only vague or approximate matches to the error. Crucially, do not score sets of statements that collectively provide the correct answer in this scenario.\n\n"
#   "General Rules for Relevance and Scoring:\n\n"
#   "* Direct Relevance of the Set Only: Only evaluate sets of statements that, as a collective unit, directly address or support the answering of the core factual question posed by the query. Ignore sets providing only general or contextual statements not directly contributing to this.\n"
#   "* Precision is Paramount (for Collective Information):\n"
#   "    * Dates, quantities, named entities, and identifiers derived from the collective information of the set must match exactly where precision is required by the query or for evaluating correctness/error.\n"
#   "    * Sets supporting a correct answer must, through their combined information, affirm the exact constraints required by the query perfectly.\n"
#   "    * Sets supporting an incorrect answer must, through their combined information, perfectly mirror the specific erroneous value and context from the Last Response.\n"
#   "* Constraint Mismatches: Sets of statements where the overall information presented by the set fails the query's specific constraints (e.g., wrong year, different metric for the collective data) are irrelevant for scoring as 'Correct'. They are also irrelevant for scoring as 'Error' unless the set as a whole perfectly matches the specific error made in the Last Response regarding that mismatched constraint/value.\n"
#   "* No New Information: Do not score sets of statements that, when taken together, merely repeat parts of the query without adding new, collectively useful factual information.\n"
#   "* Context of Last Response: Do not score a set as 'Correct' (supporting the correct answer) if the Last Response was incorrect. Similarly, the 'Contradictory' category applies primarily when the Last Response was correct.\n\n"
#   "Output Format:\n\n"
#   "* For scored sets supporting a correct answer: Correct: <set_index>:<score> <set_index>:<score> ...\n"
#   "* For scored sets supporting an error or being contradictory/misleading: Error: <set_index>:<score> <set_index>:<score> ...\n"
#   "* If no sets of statements are scored as either Correct or Error after evaluation: return No feedback\n"
#   "* If the 'Insufficient Information' pre-condition is met: return Insufficient information\n\n"
#   "Example Output Structures:\n"
#   "Correct: 2:3 5:2\n"
#   "Error: 3:3 7:1\n"
#   "Correct: 1:3\n"
#   "Error: 4:3\n"
#   "No feedback\n"
#   "Insufficient information\n\n"
#   "Do not provide explanations—only output structured results as specified above.\n\n"
#   "Query: {question}\n"
#   "Last Answer ({flag_TF}): {last_response}\n"
#   "Retrieved Statement Sets: {knowledge_statement_sets}\n"
# )

knowledge_graph_score_feedback_prompt_qwen3_huggingface = (
  "<|im_start|>system\n"
  "You are an AI assistant tasked with evaluating the factual relevance and correctness of retrieved sets of inferential statements (each statement in natural language form) in relation to a given query and the model's previous response (Last Response). This evaluation occurs in the second step of a process, focusing on sets of statements used or relevant to the Last Response. Your task is to assign scores only to sets of statements that, when their information is combined and considered as a whole, are directly relevant to the query's specific factual question, ignoring irrelevant ones.\n\n"
  "<|im_end|>\n"
  "<|im_start|>user\n"
  "Crucially, your evaluation must be exceptionally strict regarding numerical values, dates, quantities, statistics, named entities, and other precise factual details derived from the collective information of the set. Exact, literal matches of the information conveyed by the entire set are paramount.\n\n"
  "Pre-condition for Scoring:\n"
  "1. Handling 'Insufficient Information':\n"
  "Regardless of whether the Last Response was correct or incorrect, if it explicitly states that it cannot answer the query due to insufficient or irrelevant information, or requests more details (e.g., contains phrases like 'insufficient information to determine', 'need more context', 'based on the provided snippets/statements...'), do not evaluate or score any sets of statements.\n"
  "In this specific case, your only output should be the phrase 'Insufficient information'. Do not proceed with the scoring rules below.\n\n"
  "Evaluation Criteria based on Last Response Correctness (Apply only if the pre-condition above is NOT met):\n\n"
  "1.  If the Last Response was Correct :\n"
  "    You will evaluate two potential types of sets of statements in this case:\n"
  "    * Score Supporting Correct Sets of Statements:\n"
  "        Assign a score only to sets of statements where the combined information from all statements within the set, when taken together, provides specific and exact facts that directly justify the correct answer. The set as a whole must unambiguously and explicitly lead to or express the key facts needed to answer the question correctly. It is not necessary for every individual statement in the set to contain the complete correct answer, as long as their combination does.\n"
  "        Score from 1 to 3:\n"
  "        - 3 = The set is highly relevant, its collective information is precise and essential; the combined statements provide the exact fact or strong logical support for the answer.\n"
  "        - 2 = The set is relevant and its collective information is mostly correct, but the combined information may be slightly indirect, or some statements within the set might be redundant if others already establish the point.\n"
  "        - 1 = The set provides weak but still factual collective support for the correct answer.\n\n"
  "    * Score Contradictory or Misleading Sets of Statements:\n"
  "        Assign a score only to sets of statements where the combined information, when taken together, contains factual inaccuracies or strongly suggests incorrect conclusions that would contradict the actual correct answer. It is not necessary for every individual statement in the set to be contradictory, as long as the collective meaning of the set is contradictory.\n"
  "        Score from 1 to 3 depending on how directly and convincingly the set as a whole would mislead or contradict the correct answer.\n\n"
  "2.  If the Last Response was Incorrect :\n"
  "    You will evaluate only one type of set of statements in this case:\n"
  "    * Score Sets of Statements Supporting the Error (Exact Match Rule for Collective Information):\n"
  "        Assign a score only to sets of statements whose collective factual content, when all statements are considered together, exactly and literally matches the specific incorrect information present in the Last Response. It is not necessary for every individual statement in the set to contain the full error, as long as their combination directly supports the specific error.\n"
  "        Score from 1 to 3, with 3 indicating the set's combined information provides direct, unambiguous, and exact factual support for the specific error. Do not score sets whose collective information provides only vague or approximate matches to the error. Crucially, do not score sets of statements that collectively provide the correct answer in this scenario.\n\n"
  "General Rules for Relevance and Scoring:\n\n"
  "* Direct Relevance of the Set Only: Only evaluate sets of statements that, as a collective unit, directly address or support the answering of the core factual question posed by the query. Ignore sets providing only general or contextual statements not directly contributing to this.\n"
  "* Precision is Paramount (for Collective Information):\n"
  "    * Dates, quantities, named entities, and identifiers derived from the collective information of the set must match exactly where precision is required by the query or for evaluating correctness/error.\n"
  "    * Sets supporting a correct answer must, through their combined information, affirm the exact constraints required by the query perfectly.\n"
  "    * Sets supporting an incorrect answer must, through their combined information, perfectly mirror the specific erroneous value and context from the Last Response.\n"
  "* Constraint Mismatches: Sets of statements where the overall information presented by the set fails the query's specific constraints (e.g., wrong year, different metric for the collective data) are irrelevant for scoring as 'Correct'. They are also irrelevant for scoring as 'Error' unless the set as a whole perfectly matches the specific error made in the Last Response regarding that mismatched constraint/value.\n"
  "* No New Information: Do not score sets of statements that, when taken together, merely repeat parts of the query without adding new, collectively useful factual information.\n"
  "* Context of Last Response: Do not score a set as 'Correct' (supporting the correct answer) if the Last Response was incorrect. Similarly, the 'Contradictory' category applies primarily when the Last Response was correct.\n\n"
  "Output Format:\n\n"
  "* For scored sets supporting a correct answer: Correct: <set_index>:<score> <set_index>:<score> ...\n"
  "* For scored sets supporting an error or being contradictory/misleading: Error: <set_index>:<score> <set_index>:<score> ...\n"
  "* If no sets of statements are scored as either Correct or Error after evaluation: return No feedback\n"
  "* If the 'Insufficient Information' pre-condition is met: return Insufficient information\n\n"
  "Example Output Structures:\n"
  "Correct: 2:3 5:2\n"
  "Error: 3:3 7:1\n"
  "Correct: 1:3\n"
  "Error: 4:3\n"
  "No feedback\n"
  "Insufficient information\n\n"
  "Do not provide explanations—only output structured results as specified above.\n\n"
  "Query: {question}\n"
  "Last Answer ({flag_TF}): {last_response}\n"
  "Retrieved Statement Sets: {knowledge_statement_sets}\n"
  " /no_think <|im_end|>\n"
  "<|im_start|>assistant"
)

knowledge_graph_score_feedback_prompt_llama = (
  "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
  "You are an AI assistant tasked with evaluating the factual relevance and correctness of retrieved sets of inferential statements (each statement in natural language form) in relation to a given query and the model's previous response (Last Response). This evaluation occurs in the second step of a process, focusing on sets of statements used or relevant to the Last Response. Your task is to assign scores only to sets of statements that, when their information is combined and considered as a whole, are directly relevant to the query's specific factual question, ignoring irrelevant ones.\n\n"
  "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n"
  "Crucially, your evaluation must be exceptionally strict regarding numerical values, dates, quantities, statistics, named entities, and other precise factual details derived from the collective information of the set. Exact, literal matches of the information conveyed by the entire set are paramount.\n\n"
  "Pre-condition for Scoring:\n"
  "1. Handling 'Insufficient Information':\n"
  "Regardless of whether the Last Response was correct or incorrect, if it explicitly states that it cannot answer the query due to insufficient or irrelevant information, or requests more details (e.g., contains phrases like 'insufficient information to determine', 'need more context', 'based on the provided snippets/statements...'), do not evaluate or score any sets of statements.\n"
  "In this specific case, your only output should be the phrase 'Insufficient information'. Do not proceed with the scoring rules below.\n\n"
  "Evaluation Criteria based on Last Response Correctness (Apply only if the pre-condition above is NOT met):\n\n"
  "1.  If the Last Response was Correct :\n"
  "    You will evaluate two potential types of sets of statements in this case:\n"
  "    * Score Supporting Correct Sets of Statements:\n"
  "        Assign a score only to sets of statements where the combined information from all statements within the set, when taken together, provides specific and exact facts that directly justify the correct answer. The set as a whole must unambiguously and explicitly lead to or express the key facts needed to answer the question correctly. It is not necessary for every individual statement in the set to contain the complete correct answer, as long as their combination does.\n"
  "        Score from 1 to 3:\n"
  "        - 3 = The set is highly relevant, its collective information is precise and essential; the combined statements provide the exact fact or strong logical support for the answer.\n"
  "        - 2 = The set is relevant and its collective information is mostly correct, but the combined information may be slightly indirect, or some statements within the set might be redundant if others already establish the point.\n"
  "        - 1 = The set provides weak but still factual collective support for the correct answer.\n\n"
  "    * Score Contradictory or Misleading Sets of Statements:\n"
  "        Assign a score only to sets of statements where the combined information, when taken together, contains factual inaccuracies or strongly suggests incorrect conclusions that would contradict the actual correct answer. It is not necessary for every individual statement in the set to be contradictory, as long as the collective meaning of the set is contradictory.\n"
  "        Score from 1 to 3 depending on how directly and convincingly the set as a whole would mislead or contradict the correct answer.\n\n"
  "2.  If the Last Response was Incorrect :\n"
  "    You will evaluate only one type of set of statements in this case:\n"
  "    * Score Sets of Statements Supporting the Error (Exact Match Rule for Collective Information):\n"
  "        Assign a score only to sets of statements whose collective factual content, when all statements are considered together, exactly and literally matches the specific incorrect information present in the Last Response. It is not necessary for every individual statement in the set to contain the full error, as long as their combination directly supports the specific error.\n"
  "        Score from 1 to 3, with 3 indicating the set's combined information provides direct, unambiguous, and exact factual support for the specific error. Do not score sets whose collective information provides only vague or approximate matches to the error. Crucially, do not score sets of statements that collectively provide the correct answer in this scenario.\n\n"
  "General Rules for Relevance and Scoring:\n\n"
  "* Direct Relevance of the Set Only: Only evaluate sets of statements that, as a collective unit, directly address or support the answering of the core factual question posed by the query. Ignore sets providing only general or contextual statements not directly contributing to this.\n"
  "* Precision is Paramount (for Collective Information):\n"
  "    * Dates, quantities, named entities, and identifiers derived from the collective information of the set must match exactly where precision is required by the query or for evaluating correctness/error.\n"
  "    * Sets supporting a correct answer must, through their combined information, affirm the exact constraints required by the query perfectly.\n"
  "    * Sets supporting an incorrect answer must, through their combined information, perfectly mirror the specific erroneous value and context from the Last Response.\n"
  "* Constraint Mismatches: Sets of statements where the overall information presented by the set fails the query's specific constraints (e.g., wrong year, different metric for the collective data) are irrelevant for scoring as 'Correct'. They are also irrelevant for scoring as 'Error' unless the set as a whole perfectly matches the specific error made in the Last Response regarding that mismatched constraint/value.\n"
  "* No New Information: Do not score sets of statements that, when taken together, merely repeat parts of the query without adding new, collectively useful factual information.\n"
  "* Context of Last Response: Do not score a set as 'Correct' (supporting the correct answer) if the Last Response was incorrect. Similarly, the 'Contradictory' category applies primarily when the Last Response was correct.\n\n"
  "Output Format:\n\n"
  "* For scored sets supporting a correct answer: Correct: <set_index>:<score> <set_index>:<score> ...\n"
  "* For scored sets supporting an error or being contradictory/misleading: Error: <set_index>:<score> <set_index>:<score> ...\n"
  "* If no sets of statements are scored as either Correct or Error after evaluation: return No feedback\n"
  "* If the 'Insufficient Information' pre-condition is met: return Insufficient information\n\n"
  "Example Output Structures:\n"
  "Correct: 2:3 5:2\n"
  "Error: 3:3 7:1\n"
  "Correct: 1:3\n"
  "Error: 4:3\n"
  "No feedback\n"
  "Insufficient information\n\n"
  "Do not provide explanations—only output structured results as specified above.\n\n"
  "Query: {question}\n"
  "Last Answer ({flag_TF}): {last_response}\n"
  "Retrieved Statement Sets: {knowledge_statement_sets}\n"
  "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
)

llama_preamble = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
llama_query = "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

chat_with_ollama_8b_for_response = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
    "{knowledge_sequences}"
    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
)

chat_with_ollama_8b_for_response_api_system = (
    "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
    "{knowledge_sequences}"
)

chat_with_ollama_8b_for_response_api_user = (
    "Please write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}"
)

chat_with_ollama_8b_for_response_api = (
    "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
    "{knowledge_sequences}"
    "Please write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). If the provided context does not explicitly contain the information to answer the question, please state that more information is needed. Answer directly without explanation and keep the response short and direct.\nQuestion: {question}"
)

# 允许不回答问题索要更多信息
# If the provided context does not explicitly contain the information to answer the question, please state that more information is needed.
chat_with_ollama_8b_for_response_api_v1 = (
    "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
    "{knowledge_sequences}"
    "Please write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}"
)

chat_with_qwen_32b_for_response = (
    "<|im_start|>system\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
    "{knowledge_sequences}"
    "<|im_end|>\n<|im_start|>user\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|im_end|>\n<|im_start|>assistant"
)

knowledge_graph_error_statistics_prompt = (
    "<|im_start|>system\nYou are an AI assistant. Your task is to identify sentences within a given numbered list (`statement_text`) that attempt to answer the specific `question` but provide information that is factually incorrect based on a provided list of acceptable answers (`answers`).\n<|im_end|>\n"
    "<|im_start|>user\n"
    "**Inputs:**\n"
    "\n"
    "1.  `question`: The specific factual question being asked.\n"
    "2.  `answers`: A list containing multiple acceptable ways the correct answer might be written (e.g., [\"42\", \"forty-two\", \"approximately 42\"]).\n"
    "3.  `statement_text`: A text containing numbered statements, where each line starts with an index (number) followed by a period or similar separator, and then the statement itself. Like this:\n"
    "    `0. Statement one.`\n"
    "    `1. Statement two.`\n"
    "    `2. Statement three.`\n"
    "\n"
    "**Your Goal:**\n"
    "\n"
    "Carefully read each numbered statement in `statement_text`. For each statement, determine if it directly addresses the `question` and if the factual answer it provides matches **any** of the forms provided in the `answers` list. Identify the statements that provide an answer but are **incorrect**.\n"
    "\n"
    "**Evaluation Criteria:**\n"
    "\n"
    "1.  **Direct Relevance:** Only consider statements that are clearly attempting to provide an answer to the specific `question`. Ignore irrelevant statements or background information.\n"
    "2.  **Factual Check:** Compare the factual information (numbers, names, dates, values, etc.) presented in the relevant statements against **all** entries in the `answers` list.\n"
    "3.  **Identify Errors:** A statement is considered an error **only if** it directly answers the `question` AND its factual content contradicts **every single** acceptable answer form in the `answers` list.\n"
    "4.  **Strictness:** Be precise. Pay close attention to numerical values, dates, quantities, specific names, etc. If a statement matches *any* of the acceptable forms in the `answers` list, it is **not** an error.\n"
    "5.  **Focus on the statement text:** Evaluate the text *after* the initial index number and separator.\n"
    "\n"
    "**Output Format:**\n"
    "\n"
    "*   If you find one or more statements in `statement_text` that answer the `question` incorrectly:\n"
    "    *   Identify the **index** (the number at the beginning of the line) for each incorrect statement.\n"
    "    *   Output these indices as a single string, separated only by commas (no spaces).\n"
    "    *   Example: `0,1` (if statements indexed 0 and 1 were incorrect)\n"
    "    *   Example: `2` (if only statement indexed 2 was incorrect)\n"
    "*   If you find **no** statements in `statement_text` that provide an incorrect answer to the `question` based on the `answers` list:\n"
    "    *   Output the exact phrase: `No errors found`\n"
    "\n"
    "**Example Scenario 1:**\n"
    "\n"
    "*   `question`: \"What year was 'Carousel of Time' released?\"\n"
    "*   `answers`: [\"2021\", \"Twenty Twenty One\"]\n"
    "*   `statement_text`:\n"
    "    `0. Carousel of time Released in 2022`\n"
    "    `1. Batwheels Streaming on Hbo max`\n"
    "    `2. Carousel of Time is a movie.`\n"
    "\n"
    "*   **Explanation:** Statement 0 attempts to answer the question but gives \"2022\", which is not in the `answers` list [\"2021\", \"Twenty Twenty One\"]. Statement 1 is irrelevant. Statement 2 is relevant context but doesn't provide the specific answer (the year).\n"
    "*   **Expected Output:**\n"
    "    `0`\n"
    "\n"
    "**Instructions:**\n"
    "\n"
    "Now, analyze the provided inputs and generate the output according to the rules above.\n"
    "\n"
    "`question`: {question}\n"
    "`answers`: {answers}\n"
    "`statement_text`: {knowlege_sentence} // This will contain the numbered statements<|im_end|>\n<|im_start|>assistant"
)

knowledge_graph_error_statistics_prompt_v1 = (
    "<|im_start|>system\n# ROLE: AI Evaluator of Knowledge Graph Sentences/Triples<|im_end|>\n\n"
    "# TASK:\n"
    "<|im_start|>user\nYou are an AI assistant evaluating retrieved sentences/triples based **solely** on a given **Query** and its **Correct Answer**. Your objective is to identify the **index numbers** (sentence number, 0-based) of any retrieved sentences/triples that **factually and directly contradict** the provided **Correct Answer**. You must ignore irrelevant sentences/triples or those that merely provide context or are incomplete.\n\n"
    "# CORE PRINCIPLE: Extreme Strictness on Factual Details\n"
    "Your evaluation *must* be exceptionally strict, particularly regarding **numerical values, dates (including day, month, and year), quantities, statistics, and specific identifiers.** Only exact, literal matches matter when determining a contradiction.\n\n"
    "# Evaluation Criteria and Rules:\n\n"
    "* **Identify Contradictory Sentences/Triples ONLY:** Identify the index number **exclusively** for sentences/triples containing information that **directly and factually contradicts the Correct Answer** regarding the query's core question.\n"
    "    * The contradiction must be **precise and factual** (e.g., the sentence states '$100 million' when the Correct Answer is '$50 million'; the sentence states 'January 5, 2023' when the Correct Answer is 'January 5, 2022').\n\n"
    "* **Direct Relevance:** Only evaluate sentences/triples that directly address the core factual element(s) asked in the Query. Ignore those providing only background, context, or related but distinct information.\n\n"
    "* **Precision is Paramount (Universal Critical Rule):**\n"
    "    * **Exact Literal Match Required:** For *any* evaluation, especially involving numbers, dates, quantities, or specific identifiers, the information in the sentence/triple must **exactly and literally** match the detail in the Correct Answer to determine if there is a contradiction.\n"
    "        * **Dates:** Must match Day, Month, and Year if all are relevant to the Correct Answer. A sentence stating only the month and day does *not* contradict an answer specifying month, day, and year.\n"
    "        * **Numbers/Quantities:** Must match the exact value. Qualifiers matter (e.g., 'over 100' does *not* contradict '100'; 'approximately 50' does *not* contradict '50').\n"
    "    * **Ignore Incomplete Information:** A sentence/triple providing incomplete information (e.g., correct day and month but missing the year when the Correct Answer requires the year) does **not** contradict the Correct Answer and should **not** be identified.\n"
    "    * **Ignore Correct/Aligned Information:** Sentences/triples that align with the Correct Answer (even if perfectly) are **ignored**. Your focus is *only* on identifying contradictions.\n"
    "    * **Constraint Mismatches:** Sentences/triples that fail the query's specific constraints (e.g., data for the wrong year requested, statistics for a different category) are generally irrelevant and ignored, *unless* they present a value that *directly contradicts* the Correct Answer for the *actual requested constraint*.\n\n" 
    "* **Ignore Context/Background:** Do not identify sentences/triples offering only general context, definitions, or related but non-answering facts.\n\n"
    "* **No Partial Keyword Matches:** Do not identify sentences/triples just because they share keywords with the query or answer. The sentence/triple must contain the specific factual assertion that *contradicts* the Correct Answer.\n\n"
    "# Output Format:\n\n"
    "Strictly adhere to the following output formats:\n\n"
    "1.  **If Contradictory Sentences/Triples Found:**\n"
    "    List the 0-based index numbers (句子序号) of **all** contradictory sentences/triples, separated by commas.\n"
    "    Example: 1, 5\n"
    "    (If only one is found, list just that index. Example: 1)\n\n"
    "2.  **If NO Sentences/Triples Contradict the Correct Answer (after evaluation):**\n"
    "    Output the exact phrase:\n"
    "    No feedback\n\n"
    "**CRITICAL FINAL INSTRUCTION:** Provide **only** the structured output as specified above. Do **not** include any explanations, justifications, summaries, or conversational text.\n\n"
    "Query: {question}\n"
    "Correct Answer: {answers}\n"
    "Retrieve Result:\n {knowlege_sentence}<|im_end|>\n<|im_start|>assistant"
)

knowledge_graph_error_statistics_prompt_v2 = (
    "<|im_start|>system\n# ROLE: AI Evaluator of Knowledge Graph Sentences/Triples<|im_end|>\n\n"
    "<|im_start|>user\n"
    "# TASK:\n"
    "ROLE: AI Evaluator of Knowledge Graph Sentences/Triples\n"
    "TASK:\n"
    "You are an AI assistant evaluating retrieved sentences/triples based solely on a given Query and its Correct Answer. Your objective is to identify the index numbers (0-based) of any retrieved sentences/triples that factually and directly contradict the specific assertion made by the Correct Answer in response to the Query. You must ignore irrelevant sentences/triples, those providing context, those that are incomplete, or those discussing related but distinct facts about the entities involved.\n"
    "CORE PRINCIPLE: Extreme Strictness on Factual Details and Direct Contradiction\n"
    "Your evaluation must be exceptionally strict. A sentence is ONLY contradictory if it directly negates the core factual information provided in the Correct Answer as it relates to the specific question asked in the Query. Pay close attention to numerical values, dates (including day, month, and year), quantities, statistics, specific identifiers, and the relationship between entities (e.g., who acquired whom).\n"
    "Evaluation Criteria and Rules:\n"
    "Identify Directly Contradictory Sentences/Triples ONLY: Identify the index number exclusively for sentences/triples containing information that directly and factually negates the specific assertion of the Correct Answer in relation to the Query.\n"
    "The contradiction must be precise and factual regarding the core subject of the Query (e.g., if the query asks 'Who acquired A?' and the answer is 'B acquired A', a sentence stating 'A acquired B' or 'C acquired A' is contradictory. However, a sentence stating 'B was acquired by D' is not a direct contradiction to the fact that B acquired A, as it describes a separate event).\n"
    "The contradiction must be about the specific detail requested (e.g., the sentence states '$100 million' when the Correct Answer is '$50 million'; the sentence states 'January 5, 2023' when the Correct Answer is 'January 5, 2022').\n"
    "Focus on the Query-Answer Relationship: Only evaluate sentences based on whether they contradict the specific information provided in the Correct Answer to the specific Query. Ignore sentences that provide information about the entities involved if that information does not invalidate the core assertion of the Query-Answer pair.\n"
    "Example: Query: \"Who acquired ShowBiz?\", Answer: \"EVO Entertainment Group\". Sentence: \"EVO Entertainment Group was acquired by Times Square\". This sentence is NOT contradictory because EVO being acquired by someone else does not negate the fact that EVO acquired ShowBiz.\n"
    "Precision is Paramount (Universal Critical Rule):\n"
    "Exact Literal Match Required for Contradiction: For any evaluation, especially involving numbers, dates, quantities, or specific identifiers, the information in the sentence/triple must exactly and literally conflict with the detail in the Correct Answer to be considered a contradiction.\n"
    "Dates: Must contradict the specific Day, Month, or Year components provided/relevant in the Correct Answer.\n"
    "Numbers/Quantities: Must contradict the exact value. Qualifiers matter (e.g., 'over 100' does not contradict '100').\n"
    "Ignore Incomplete Information: A sentence/triple providing incomplete information (e.g., correct company name but wrong relationship, or vice versa) does not contradict the Correct Answer if it doesn't make a specific, false assertion about the queried fact. It should not be identified.\n"
    "Ignore Correct/Aligned Information: Sentences/triples that align with the Correct Answer are ignored.\n"
    "Ignore Irrelevant/Contextual Information: Do not identify sentences/triples offering only background, context, definitions, or facts related to the entities but not directly addressing the core Query-Answer assertion.\n"
    "No Partial Keyword Matches: Do not identify sentences/triples just because they share keywords. The sentence/triple must contain the specific factual assertion that directly contradicts the Correct Answer for the specific question asked.\n"
    "Output Format:\n"
    "Strictly adhere to the following output formats:\n"
    "If Contradictory Sentences/Triples Found: List the 0-based index numbers of all directly contradictory sentences/triples, separated by commas. Example: (If only one is found, list just that index. Example: 1, 51)\n"
    "If NO Sentences/Triples Directly Contradict the Correct Answer (after evaluation): Output the exact phrase: No feedback\n"
    "CRITICAL FINAL INSTRUCTION: Provide only the structured output as specified above. Do not include any explanations, justifications, summaries, or conversational text in your final output.\n"
    "Query: {question}\n"
    "Correct Answer: {answers}\n"
    "Retrieve Result:\n {knowlege_sentence}<|im_end|>\n<|im_start|>assistant"
)

knowledge_graph_error_statistics_prompt_v3 = (
    "<|im_start|>system\nROLE: AI Evaluator of Knowledge Graph Sentences/Triples\n<|im_end|>\n\n"
    "<|im_start|>user\n"
    "TASK:\n"
    "You are an AI assistant evaluating retrieved sentences/triples based solely on a given Query and its Correct Answer. Your objective is to identify the index numbers (0-based) of any retrieved sentences/triples that factually and directly contradict the specific assertion made by the Correct Answer specifically in relation to the subject and attribute of the Query. You must ignore sentences/triples that are irrelevant, provide context, are incomplete, or discuss related but distinct facts, even if they involve the same entities or similar concepts (like dates or amounts).\n"
    "\n"
    "CORE PRINCIPLE: Extreme Strictness on Factual Details, Subject Alignment, and Direct Contradiction\n"
    "Your evaluation must be exceptionally strict. A sentence is ONLY contradictory if it:\n"
    "1. Refers to the exact same subject entity (or entities and their relationship) as the Query.\n"
    "2. Addresses the exact same attribute, property, or relationship inquired about in the Query.\n"
    "3. Presents a factual value (e.g., date, number, name, status) that directly negates or conflicts with the specific value provided in the Correct Answer for that subject and attribute.\n"
    "\n"
    "Evaluation Criteria and Rules:\n"
    "1. Identify Directly Contradictory Sentences/Triples ONLY: Identify the index number exclusively for sentences/triples containing information that meets all three conditions listed in the CORE PRINCIPLE.\n"
    "2. Subject and Attribute Alignment is Mandatory: Before checking for conflicting values, first confirm the sentence makes a claim about the precise subject and precise attribute targeted by the Query.\n"
    "  Example: If the Query is \"When did Movie X premiere?\" and the Answer is \"January 1st\", a sentence stating \"Movie Y premiered February 1st\" is NOT contradictory, it is irrelevant because the subject (Movie Y) differs.\n"
    "  Example: If the Query is \"What is the revenue of Company A?\" and the Answer is \"$100M\", a sentence stating \"Company A's profit was $10M\" is NOT contradictory, it is irrelevant because the attribute (profit vs. revenue) differs, even though the subject (Company A) is the same.\n"
    "3. Contradiction Must Be Precise and Factual: Once subject and attribute alignment is confirmed, the value presented in the sentence must factually contradict the specific detail in the Correct Answer.\n"
    "  Dates: Must contradict the relevant Day, Month, or Year components provided in the Correct Answer. A sentence providing only a month when the answer has a specific day/month/year might be incomplete, but not contradictory unless the month itself conflicts.\n"
    "  Numbers/Quantities: Must contradict the exact value. Qualifiers matter (e.g., 'more than 100' does not contradict '100' or '150').\n"
    "  Relationships/Status: Must assert a conflicting relationship or status (e.g., Query: \"Who acquired A?\", Answer: \"B acquired A\". Contradictory sentences: \"A acquired B\", \"C acquired A\". Non-contradictory: \"B merged with D\", \"A was founded in 2000\").\n"
    "4. Focus Strictly on the Query-Answer Assertion: Evaluate sentences only on whether they invalidate the specific fact asserted by the Correct Answer for the specific Query's subject and attribute. Do not infer contradictions.\n"
    "5.Ignore Irrelevant, Contextual, or Incomplete Information:\n"
    "  Irrelevant: Sentences about different subjects, different attributes, or related but distinct events/facts (even concerning the same entity) must be ignored. This includes facts that are true but do not address the specific Query-Answer assertion.\n"
    "  Contextual: Background information, definitions, or general statements about the entities are not contradictions.\n"
    "  Incomplete: Sentences providing partial information that doesn't form a specific, conflicting assertion about the queried fact should be ignored.\n"
    "6. No Partial Keyword Matches: Do not identify sentences merely because they share keywords (like dates, names, locations) with the Query or Answer. The sentence must make a specific factual assertion about the Query's subject and attribute that directly conflicts with the Correct Answer's value.\n"
    "\n"
    "Output Format:\n"
    "Strictly adhere to the following output formats:\n"
    "  If Contradictory Sentences/Triples Found: List the 0-based index numbers of all directly contradictory sentences/triples, separated by commas. (e.g., , , 10, 415, 21, 30)\n"
    "  If NO Sentences/Triples Directly Contradict the Correct Answer (after strict evaluation): Output the exact phrase: No feedback\n"
    "\n"
    "CRITICAL FINAL INSTRUCTION: Provide only the structured output as specified above. Do not include any explanations, justifications, summaries, or conversational text in your final output.\n"
    "Query: {question}\n"
    "Correct Answer: {answers}\n"
    "Retrieve Result:\n {knowlege_sentence}<|im_end|>\n<|im_start|>assistant"
)

class ChatGraphRAG(ChatBase):

    def __init__(self, llm: LLMBase, graph_db : GraphDatabase, args = None):
        super().__init__(llm)
        self.graph_database = graph_db
        self.retriver_graph = RetrieverGraph(llm,graph_db, args)

    
    def retrieve_triplets(self, message, space_name):
        """
        Args:
            message (str): the query from users
            space_name (str): the graph name of graph database

        Returns:
            list[dict["source", "relationship", "destination"]]: retrieve triplets
        """
        self.triplets = self.retriver_graph.retrieve_2hop(question=message)
        return self.triplets
    
    def get_triplets(self):
        if self.triplets:
            return self.triplets
        else:
            return ["还未检索!"]
        
    @override
    def retrieval_result(self):
        return self.triplets

    @override
    def web_chat(self, message: str,history: List[Optional[List]] | None = None):
        
        self.triplets = self.retriver_graph.retrieve_2hop(question=message, pruning = 30)

        prompt = llama_QA_graph_prompt.format(query = message, context = self.triplets)

        ic(prompt)
        ic(history)

        # answers = self._llm.chat_with_ai_stream(prompt, history)
        # result = ""
        # for chunk in answers:
        #     result =  result + chunk.choices[0].delta.content or ""

        #     yield result

        # ic(result)
        return self._llm.chat_with_ai_stream(prompt, history)

    def web_chat_with_triplets(self, message: str,triplets,history: List[Optional[List]] | None):

        prompt = llama_QA_graph_prompt.format(query = message, context = triplets)
        ic(prompt)
        ic(history)
        # result = 

        answers = self._llm.chat_with_ai_stream(prompt, history)
        result = ""
        for chunk in answers:
            result =  result + chunk.choices[0].delta.content or ""
            yield result

        ic(result)
    
    @override
    def chat_without_stream(self, message: str, pruning = None):
        self.triplets = self.retriver_graph.retrieve_2hop(question=message, pruning = pruning)
        prompt = llama_QA_graph_prompt.format(query = message, context = self.triplets)
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)
            
        # ic(answers)
        return answers

    
    # +++
    def chat_without_stream_with_triplets(self, message: str, triplets = None):
        # ic(self.triplets)
        # self.triplets = triplets
        if isinstance(triplets, List):
            context = ""
            for sentence in triplets:
                context += f'{sentence}\n'
        else:
            context = triplets

        prompt_system = chat_with_graphrag_for_response_system.format(knowledge_sequences = context)
        prompt_user = chat_with_graphrag_for_response_user.format(question = message)

        # ic(prompt)
        # print(f"chat_without_stream_with_triplets_llama_instruct:\n{prompt}")

        # answers, num_input_tokens, chat_time = self._llm.chat_with_ai_with_system(prompt_system=prompt_system,
        #                                                               prompt_user=prompt_user,
        #                                                               history=None)
        start_time = time.perf_counter()
        answers = self._llm.chat_with_ai_with_system(prompt_system=prompt_system,
                                                                      prompt_user=prompt_user,
                                                                      history=None)
        num_input_tokens = 6
        end_time = time.perf_counter()
        chat_time = end_time - start_time

        # ic(answers)
        return answers, num_input_tokens, chat_time

    # +++
    def chat_without_stream_with_triplets_shared_prefix(self, message: str, triplets = None):
        # ic(self.triplets)
        # self.triplets = triplets
        if isinstance(triplets, List):
            context = ""
            for idx, sentence in enumerate(triplets, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            # for sentence in triplets:
            #     context += f'{sentence}\n'
        else:
            context = triplets

        prompt_system = shared_prefix.format(knowledge_sequences = context)
        prompt_user = chat_with_graphrag_for_response_user_shared_prefix.format(question = message)

        # ic(prompt)
        # print(f"chat_without_stream_with_triplets_llama_instruct:\n{prompt}")


        # if self._llm.llmbackend == 'vllm':
        #     answers, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self._llm.chat_with_ai_with_system(prompt_system=prompt_system, prompt_user=prompt_user, history=None)
        # else:
        #     start_end2end_time = time.perf_counter()
        #     answers = self._llm.chat_with_ai_with_system(prompt_system=prompt_system,
        #                                                                 prompt_user=prompt_user,
        #                                                                 history=None)
        #     end_end2end_time = time.perf_counter()
        #     prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = 0, 0, end_end2end_time - start_end2end_time, 0, 0, 0, 0

        answers, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self._llm.chat_with_ai_with_system(prompt_system=prompt_system, prompt_user=prompt_user, history=None)
        
        # ic(answers)
        return answers, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time
    
    # +++
    def chat_without_stream_with_triplets_shared_prefix_batch(self, message: List[str], triplets = List[str]):
        # ic(self.triplets)
        # self.triplets = triplets
        if isinstance(triplets, List):
            context = ""
            for idx, sentence in enumerate(triplets, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            # for sentence in triplets:
            #     context += f'{sentence}\n'
        else:
            context = triplets

        prompt_system = []
        prompt_user = []
        for message_item, triplets_item in zip(message, triplets):
            context = ""
            for idx, sentence in enumerate(triplets_item, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            prompt_system.append(shared_prefix.format(knowledge_sequences = context))
            prompt_user.append(chat_with_graphrag_for_response_user_shared_prefix.format(question = message_item))

        answers, prompt_len, end2end_time = self._llm.chat_with_ai_with_system(prompt_system=prompt_system, prompt_user=prompt_user, history=None)
        
        # ic(answers)
        return answers, prompt_len, end2end_time
    
    # +++
    def chat_without_stream_short_cut(self, triplets):
        # ic(self.triplets)

        prompt_user = short_cut(original_path = triplets)

        output = None
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                output = self._llm.chat_with_ai_with_system("", prompt_user)
                # output = self._llm.chat_with_ai_mulRounds(history = history)
                # print(f"feedback output {retry}\noutput: {output}")
                res_parsed = json.loads(output)
                source_node = res_parsed["source_node"]
                inferred_relationship = res_parsed["inferred_relationship"]["label"]
                target_node = res_parsed["target_node"]
                return res_parsed
            except Exception as e:
                print(f"short_cut output parse error: {e}\nllm response: {output}")
                retry += 1       
        return {}

    def chat_without_stream_with_triplets_llama_api(self, message: str, triplets = None):
        import time
        # ic(self.triplets)
        # self.triplets = triplets

        if isinstance(triplets, List):
            context = ""
            for sentence in triplets:
                # context = context + sentence
                context = context + sentence + '\n'
        else:
            context = triplets
        
        # system_prompt = chat_with_ollama_8b_for_response_api_system.format(knowledge_sequences = context)
        # user_prompt = chat_with_ollama_8b_for_response_api_user.format(question = message)
        # messages=[{'role': 'system', 'content': system_prompt},
        #           {'role': 'user', 'content': user_prompt}]
        # 百度千帆的llama模型不允许有system角色
        # response.text {"error_code":336006,"error_msg":"the role of first message must be user or assistant","id":"as-fbrfchfgf2"}
        
        prompt = chat_with_ollama_8b_for_response_api.format(knowledge_sequences = context, question = message)
        # prompt = chat_with_ollama_8b_for_response_api_v1.format(knowledge_sequences = context, question = message)
        messages=[{'role': 'user', 'content': prompt}]

        answers = '' 
        num_input_tokens = 0
        try:
            answers, num_input_tokens = self._llm.chat_with_ai(prompt="", history = messages)
            if "API call throws exception" in answers: # 
                time.sleep(5)
                answers, num_input_tokens = self._llm.chat_with_ai(prompt="", history = messages)
        except Exception as e:
            print(f"chat_without_stream_with_triplets_llama_api error: {e}\nllm response: {answers}\nnum_input_tokens: {num_input_tokens}")
            return "API call throws exception", 0
        # ic(answers)
        return answers, num_input_tokens
    
    def chat_without_stream_with_triplets_qwen_instruct(self, message: str, triplets = None):
        # ic(self.triplets)
        # self.triplets = triplets
        if isinstance(triplets, List):
            context = ""
            for sentence in triplets:
                # context = context + sentence
                context = context + sentence + '\n'
        else:
            context = triplets

        prompt = chat_with_qwen_32b_for_response.format(knowledge_sequences = context, question = message)
        # ic(prompt)
        # print(f"chat_without_stream_with_triplets_llama_instruct:\n{prompt}")

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    def chat_without_stream_with_who_are_you(self, message: str, triplets = []):

        prompt = "who are you?"
        # ic(prompt)
        print(f"chat_without_stream_with_triplets_llama_instruct:\n{prompt}")

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    def chat_without_stream_with_one_triplet(self, message: str, triplet = ""):
        # ic(self.triplets)

        prompt = llama_QA_graph_prompt_with_one_context.format(query = message, context = triplet)
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    def chat_without_stream_answer_check(self, question: str, response: str, standard_answer = []):
        # ic(self.triplets)

        # prompt = response_check_promot.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        prompt = response_check_promot_qwen_instruct.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers

    def chat_without_stream_answer_check_llama(self, question: str, response: str, standard_answer = []):
        # ic(self.triplets)

        # prompt = response_check_promot.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        # prompt = response_check_promot_qwen_instruct.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        # prompt = response_check_promot_llama_instruct.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        # prompt = response_check_promot_llama_instruct_huggingface.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        messages=[{'role': 'system', 'content': response_check_promot_system_api},
                            {'role': 'user', 'content': response_check_promot_user_api.format(Question = question, Answer = response, Standard_answers = str(standard_answer))}]


        
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt= "0", history=messages)

        # ic(answers)
        return answers
    
    def chat_without_stream_answer_check_api(self, question: str, response: str, standard_answer = []):
        # ic(self.triplets)

        # prompt = response_check_promot.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        # prompt = response_check_promot_qwen_instruct.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        prompt = response_check_promot_llama_instruct.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        # prompt = response_check_promot_llama_instruct_huggingface.format(Question = question, Answer = response, Standard_answers = str(standard_answer))


        
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    def chat_without_stream_answer_check_api_qwen(self, question: str, response: str, answer: str):
        # ic(self.triplets)
        history_bak = [
            # {
            #     "role": "user",
            #     "content": (
            #         "You are an impartial evaluator. Your task is to assess whether a given `Response` correctly answers a `Question`, based on a provided ground-truth `Answer`. Follow the rules below in the specified order.\n\n"
                    
            #         "### Input:\n"
            #         f"* `Question`: {question}.\n"
            #         f"* `Answer`: {answer}.\n"
            #         f"* `Response`: {response}.\n\n"
                    
            #         "Evaluation Workflow\n"
            #         "### Step 1: Check for Insufficient Information\n"
            #         "First, examine the `Response`. If the `Response` explicitly states that it cannot answer the query due to insufficient or irrelevant information, or if it requests more details, stop the evaluation immediately.\n"
            #         "* Condition: The Response contains phrases such as 'insufficient information to determine', 'I need more context', 'based on the provided snippets', or 'cannot be answered with the information given.'\n"
            #         "* Action: If this condition is met, output only the following text and nothing else:\n"
            #         "`Insufficient Information`\n\n"
                    
            #         "### Step 2: Classify the Ground-Truth Answer\n"
            #         "If the evaluation proceeds, your next step is to classify the nature of the `Answer`.\n"
            #         "If the primary content of the `Answer` is a number (e.g., a quantity, price, date, or measurement), output:\n"
            #         "`Answer is numeric`\n"
            #         "Otherwise (if the `Answer` is text, a name, a concept, etc.), output:\n"
            #         "`Answer is not numeric`\n\n"
                    
            #         "### Step 3: Apply Evaluation Logic\n"
            #         "Based on the classification from Step 2, apply the corresponding logic to determine the final verdict.\n"
            #         "A. If `Answer is numeric`:\n"
            #         "The `Response` must be numerically equivalent to the `Answer` to be considered correct.\n"
            #         "* Correct: The numerical value in the Response is an exact match to the Answer.\n"
            #         "    Minor formatting differences are acceptable (e.g., `1,000` vs. `1000`; `$50` vs. `50 dollars`; `five` vs. `5`).\n"
            #         "* Incorrect: The numerical value in the `Response` does not match the `Answer`.\n"
            #         "B. If `Answer is not numeric`:\n"
            #         "The evaluation is based on the containment of the `Answer` within the `Response`.\n"
            #         "* Correct: The `Response` contains the entire `Answer`. The presence of additional, related, or even irrelevant information does not invalidate the correctness.\n"
            #         "* Partially Correct: The `Response` contains only a part of the `Answer` but misses key components.\n"
            #         "* Incorrect: The `Response` does not contain the `Answer`.\n\n"
                    
            #         "General Guideline:\n"
            #         "* Acronyms and Full Names: Treat acronyms and their corresponding full names as identical. For example, if the `Answer` is 'NASA' and the `Response` contains 'National Aeronautics and Space Administration,' it should be considered a match, and vice-versa.\n"
            #         "* Case-insensitive matching is allowed.\n\n"

            #         "### Example:\n"
            #         "`Question`: 'Who were the main architects of the Eiffel Tower?'\n"
            #         "`Answer`: 'Maurice Koechlin and Émile Nouguier'\n"
            #         "`Response`: 'The Eiffel Tower was designed by Maurice Koechlin.'\n"
            #         "Your Output:\n"
            #         "Answer is not numeric\n"
            #         "Partially Correct"
            #     )
            # }
        ]
        
        history_bak_349 = [
            # {
            #     "role": "user",
            #     "content": (
            #         "You are an impartial evaluator. Your task is to assess whether a given `Response` correctly answers a `Question`, based on a provided ground-truth `Answer`. Follow the rules below in the specified order.\n\n"
                    
            #         "### Input:\n"
            #         f"`Question`: {question}.\n"
            #         f"`Answer`: {answer}.\n"
            #         f"`Response`: {response}.\n\n"
                    
            #         "### Evaluation Workflow\n"
            #         "Step 1: Check for Insufficient Information\n"
            #         "First, examine the `Response`. If the `Response` explicitly states that it cannot answer the query due to insufficient or irrelevant information, or if it requests more details, stop the evaluation immediately.\n"
            #         "* Condition: The `Response` contains phrases such as 'insufficient information to determine', 'I need more context', 'based on the provided snippets', or 'cannot be answered with the information given.'\n"
            #         "* Action: If this condition is met, output only the following text and nothing else: 'Insufficient Information'\n\n"
                    
            #         "Step 2: Classify the Ground-Truth Answer\n"
            #         "If the evaluation proceeds, your next step is to classify the nature of the `Answer`.\n"
            #         "If the primary content of the `Answer` is a number (e.g., a quantity, price, date, or measurement), output:\n"
            #         "Answer is numeric\n"
            #         "Otherwise (if the `Answer` is text, a name, a concept, etc.), output:\n"
            #         "Answer is not numeric\n\n"
                    
            #         "Step 3: Apply Evaluation Logic\n"
            #         "Based on the classification from Step 2, apply the corresponding logic to determine the final verdict.\n\n"
                    
            #         "### A. If `Answer is numeric`:\n"
            #         "The `Response` must be numerically equivalent to the `Answer` to be considered correct. This evaluation is strict.\n"
            #         "* Correct: The numerical value in the `Response` is an exact match to the `Answer`.\n"
            #         "  * Minor formatting differences are acceptable (e.g., '1,000' vs. '1000'; '$50' vs. '50 dollars'; 'five' vs. '5').\n"
            #         "* Incorrect: The numerical value in the `Response` does not match the `Answer`.\n\n"
                    
            #         "### B. If 'Answer is not numeric':\n"
            #         "The evaluation is based on semantic correctness. The `Response` must convey the core meaning of the `Answer`, but it does not need to be a literal match. The `Response` can be more general or more specific, as long as it is not contradictory.\n"
            #         "* Correct: The `Response` accurately expresses the central fact or idea of the `Answer`. Rephrasing, summarization, or adding extra, non-contradictory information is acceptable.\n"
            #         "* Partially Correct: If the Answer consists of multiple distinct items or facts, and the Response correctly provides a subset of them while omitting others. The provided information must be accurate and not contradicted.\n"
            #         "* Incorrect: The `Response` omits the core information of the `Answer` or contradicts it.\n"
            #         "* Specific Rule for Proper Nouns (People, Places, etc.):\n"
            #         "  If the `Answer` is a proper noun (e.g., a person's full name, a full name of a place or organization), a `Response` that provides a commonly recognized partial version (e.g., a last name, an acronym) is also considered **Correct**.\n"

            #     )
            # }
        ]
        
        history = [
            {
                "role": "user",
                "content": (
                    "You are an impartial evaluator. Your task is to assess whether a given `Response` correctly answers a `Question`, based on a provided ground-truth `Answer`. Follow the rules below in the specified order.\n\n"
                    
                    f"## Input:\n"
                    f"`Question`: {question}.\n"
                    f"`Answer`: {answer}.\n"
                    f"`Response`: {response}.\n\n"
                    
                    "## Evaluation Workflow\n"
                    "### Step 1: Check for Insufficient Information\n"
                    "First, examine the `Response`. If the `Response` explicitly states that it cannot answer the query due to insufficient or irrelevant information, or if it requests more details, stop the evaluation immediately.\n"
                    "* Condition: The `Response` contains phrases such as 'insufficient information to determine', 'I need more context', 'based on the provided snippets', or 'cannot be answered with the information given.'\n"
                    "* Action: If this condition is met, output only the following text and nothing else: 'Insufficient Information'\n"
                    "### Step 2: Classify the Ground-Truth Answer\n"
                    "If the evaluation proceeds, your next step is to classify the nature of the `Answer`.\n"
                    "* If the primary content of the `Answer` is a number (e.g., a quantity, price, date, or measurement), you will determine the output for the first line is: 'Answer is numeric'\n"
                    "* Otherwise (if the `Answer` is text, a name, a concept, etc.), you will determine the output for the first line is: 'Answer is not numeric'\n"
                    "### Step 3: Apply Evaluation Logic\n"
                    "Based on the classification from Step 2, apply the corresponding logic to determine the verdict for the second line of the output.\n\n"
                    
                    "## Final Output Instructions\n"
                    "After completing the workflow, your final output must strictly follow one of the formats below.\n"
                    "1. Standard Case (Two Lines):\n"
                    "  * Line 1: The result from Step 2 ('Answer is numeric' or 'Answer is not numeric').\n"
                    "  * Line 2: The final verdict from Step 3 ('Correct', 'Incorrect', or 'Partially Correct').\n"
                    "2. Special Case (One Line):\n"
                    "  * If the condition in Step 1 is met, output only: Insufficient Information\n\n"
                    
                    "### Example of a standard two-line output:\n"
                    "Answer is not numeric\n"
                    "Correct\n\n"
                    
                    "## A. If 'Answer is numeric':\n"
                    "The `Response` must be numerically equivalent to the `Answer` to be considered correct. This evaluation is strict.\n"
                    "* Correct: The numerical value in the `Response` is an exact match to the `Answer`.\n"
                    "  * Minor formatting differences are acceptable (e.g., '1,000' vs. '1000'; '$50' vs. '50 dollars'; 'five' vs. '5').\n"
                    "* Incorrect: The numerical value in the `Response` does not match the `Answer`.\n"
                    
                    "## B. If 'Answer is not numeric':\n"
                    "The evaluation is based on semantic correctness. The `Response` must convey the core meaning of the `Answer`, but it does not need to be a literal match. The `Response` can be more general or more specific, as long as it is not contradictory.\n"
                    "* Correct: The `Response` accurately expresses the central fact or idea of the `Answer`. Rephrasing, summarization, or adding extra, non-contradictory information is acceptable.\n"
                    "* Partially Correct: If the `Answer` consists of multiple distinct items or facts, and the `Response` correctly provides a subset of them while omitting others. The provided information must be accurate and not contradicted.\n"
                    "* Incorrect: The `Response` omits the core information of the `Answer` or contradicts it.\n"
                    "* Specific Rule for Proper Nouns (People, Places, etc.):\n"
                    "  * If the `Answer` is a proper noun (e.g., a person's full name, a full name of a place or organization), a `Response` that provides a commonly recognized partial version (e.g., a last name, an acronym) is also considered Correct.\n"
                    "  * Additionally, a `Response` that uses a widely recognized alternative name, a former name for a rebranded entity, or vice versa, is also considered Correct. For example, if the `Answer` is 'Federal Reserve', a `Response` of 'Reserve bank' is correct. Similarly, if the `Answer` is 'Meta', a `Response` of 'Facebook' is correct. Similarly, if the `Answer` is 'Sportsbooks', a `Response` of 'Sportsbook apps' is correct."
                )
            }
        ]
        
        # ic(prompt)
        # print(f"-----user-----\n{history[0]['content']}")
        retry = 1
        number_TF = -1
        acc_TF = -1
        answer = '' 
        think_content = ''
        while retry <= 3:
            try:
                answer, think_content = self._llm.chat_with_ai(prompt= "think_return", history=history)
                print(f"chat_without_stream_answer_check_api_qwen output:\n{answer}")
                if 'insufficient information' in answer.lower():
                    return [], think_content
                line_list =  answer.strip().split('\n')
                line_list = [ item for item in line_list if item] # 干掉空行
                assert len(line_list) == 2
                if 'answer is numeric' in line_list[0].lower():
                    number_TF = 1
                # elif 'Answer is not numeric' in line_list[0]:
                elif 'not numeric' in line_list[0].lower():
                    number_TF = 0
                else:
                    raise Exception("----No answer whether it is a number----")
                if 'Correct' in line_list[1]:
                    acc_TF = 1
                elif 'Partially Correct' in line_list[1]:
                    acc_TF = 2
                elif 'Incorrect' in line_list[1]:
                    acc_TF = 0
                else:
                    raise Exception("----No answer whether it is correct----")
                return [number_TF, acc_TF], think_content
            except Exception as e:
                number_TF = -1
                acc_TF = -1
                print(f"check_answer_api_qwen output parse error: {e}\nllm response: {answer}\nllm think: {think_content}")
                retry += 1   
        # ic(answers)
        return [number_TF, acc_TF], ''
    
    def chat_without_stream_answer_check_qwen(self, question: str, response: str, answer: str, llm_name = ''):
        # ic(self.triplets)
        if llm_name == 'Qwen2.5-32B-Instruct' or llm_name == 'Qwen3-32B':
            prompt = check_answer_prompt_qwen
        elif llm_name == 'llama3.3':
            prompt = check_answer_prompt_llama
        
        # ic(prompt)
        # print(f"-----user-----\n{history[0]['content']}")
        retry = 1
        number_TF = -1
        acc_TF = -1
        answer = '' 
        while retry <= 3:
            try:
                answer = self._llm.chat_with_ai(prompt= prompt.format(question = question, response = response, answer = answer))
                # print(f"chat_without_stream_answer_check_api_qwen output:\n{answer}")
                if 'insufficient information' in answer.lower():
                    return []
                line_list =  answer.strip().split('\n')
                line_list = [ item for item in line_list if item] # 干掉空行
                assert len(line_list) == 2
                if 'answer is numeric' in line_list[0].lower():
                    number_TF = 1
                # elif 'Answer is not numeric' in line_list[0]:
                elif 'not numeric' in line_list[0].lower():
                    number_TF = 0
                else:
                    raise Exception("----No answer whether it is a number----")
                if 'Correct' in line_list[1]:
                    acc_TF = 1
                elif 'Partially Correct' in line_list[1]:
                    acc_TF = 2
                elif 'Incorrect' in line_list[1]:
                    acc_TF = 0
                else:
                    raise Exception("----No answer whether it is correct----")
                return [number_TF, acc_TF]
            except Exception as e:
                number_TF = -1
                acc_TF = -1
                print(f"check_answer_qwen (local_huggingface) output parse error: {e}\nllm response: {answer}")
                retry += 1   
        # ic(answers)
        return [number_TF, acc_TF]

    def chat_without_stream_answer_check_llama3_8b(self, question: str, response: str, standard_answer = []):
        # ic(self.triplets)

        prompt = response_check_promot_llama3_8b.format(Question = question, Answer = response, Standard_answers = str(standard_answer))
        # print(f"chat_without_stream_answer_check_llama3_8b:\n{prompt}")
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    
    
    
    def chat_without_stream_with_all_context(self, message: str, answer = [], retrieve_result = [], method = 0):
        # retrieve_str = ""
        # for tmp in retrieve_result:
        #     retrieve_str = retrieve_str + "\n" + tmp
        if method == 1:
            prompt = knowledge_graph_refinement_prompt.format(query = message, answer = answer, retrieve_result = retrieve_result)
            answers = self._llm.chat_with_ai(prompt)
        elif method == 2:
            prompt = knowledge_graph_entity_refinement_prompt.format(query = message, answer = answer, retrieve_result = retrieve_result)
            answers = self._llm.chat_with_ai(prompt)
        elif method == 3:
            prompt = knowledge_graph_error_correction_prompt.format(query = message, answer = answer, retrieve_result = retrieve_result)
            answers = self._llm.chat_with_ai(prompt)
        elif method == 4:
            prompt = knowledge_graph_path_generation_prompt.format(query = message, answer = answer, retrieve_result = retrieve_result)
            answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    def chat_without_stream_for_socre_feedback(self, message: str, answer = [], retrieve_result = []):
        retrieve_str = ""
        for idx, tmp in enumerate(retrieve_result, start=1):
            retrieve_str = retrieve_str + "\n" + str(idx) + ": " + tmp 
        prompt = knowledge_graph_score_correction_prompt_3.format(query = message, answer = answer, retrieve_result = retrieve_str)
        answers = self._llm.chat_with_ai(prompt)
        return answers
    
    # no answer use response
    def chat_without_stream_for_socre_feedback_latest(self, message: str, response : str, retrieve_result = [], flag_TF = True):
        retrieve_str = ""
        for idx, tmp in enumerate(retrieve_result, start=1):
            retrieve_str = retrieve_str + "\n" + str(idx) + ": " + tmp 
        if flag_TF:
            prompt = knowledge_graph_score_correction_prompt_true.format(query = message, response = response, retrieve_result = retrieve_str)
        else:
            prompt = knowledge_graph_score_correction_prompt_false.format(query = message, response = response, retrieve_result = retrieve_str)
        answers = self._llm.chat_with_ai(prompt)
        return answers
    
    def chat_without_stream_for_socre_feedback_latest_v2(self, message: str, response: str, retrieve_result = [], flag_TF = True):
        if flag_TF:
            flag_str = "Correct"
        else:
            flag_str = "Incorrect"
        retrieve_str = ""
        for idx, tmp in enumerate(retrieve_result, start=0):
            retrieve_str = retrieve_str + "\n" + str(idx) + ": " + tmp 
        # prompt = knowledge_graph_score_feedback_prompt_4_1_latest.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        prompt = knowledge_graph_score_feedback_prompt_gemini.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        
        output = None
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"output {retry}: {output}")
                line_list =  output.strip().split('\n')
                correct_numbers = {}
                error_numbers = {}
                insufficient_str = ''
                for line in line_list:
                    if line.strip() == 'Insufficient search information':
                        insufficient_str = 'Insufficient search information'
                    elif line.strip().startswith("Correct: "):
                        correct_pairs = line.strip().split("Correct:")[1].strip().split()
                        for pair in correct_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    correct_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                    elif line.strip().startswith('Error: '):
                        error_pairs = line.strip().split("Error:")[1].strip().split()
                        for pair in error_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    error_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                assert (correct_numbers or error_numbers or insufficient_str)
                return insufficient_str, correct_numbers, error_numbers
            except Exception as e:
                print(f"feedback output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return '', {}, {}

    # 错误统计

    def chat_without_stream_for_error_statistics(self, message: str, answer=[], sentence_list=[], batch_size = 30):
        batch_size = 30
        total_length = len(sentence_list)
        res_list_2d = []

        for idx in range(0, total_length, batch_size):
            # 获取当前批次的数据
            batch_sentences = sentence_list[idx: idx + batch_size]            
            res = self.chat_without_stream_for_error_statistics_item(
                message=message, 
                answer=answer, 
                sentence_list=batch_sentences
            )
            if res:
                adjusted_res = [id + idx for id in res]
                res_list_2d.extend(adjusted_res)
        
        return res_list_2d

    def chat_without_stream_for_error_statistics_item(self, message: str, answer = [], sentence_list = []):
        retrieve_str = ''
        for idx, tmp in enumerate(sentence_list, start=0):
            retrieve_str = retrieve_str + str(idx) + ". " + tmp + "\n"

        prompt = knowledge_graph_error_statistics_prompt_v3.format(question = message, answers = str(answer), knowlege_sentence = sentence_list)
        
        output = None
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                output = self._llm.chat_with_ai(prompt) # openai huggingface
                print(f"error statistics output {retry}: {output}")
                if "No feedback" in output:
                    return []
                else:
                    numeric_ids = [
                        int(id_str.strip())
                        for id_str in output.strip().split(",") 
                        if id_str.strip().isdigit() and int(id_str.strip()) < len(sentence_list)
                    ]

                    assert numeric_ids and len(numeric_ids) <= len(sentence_list)
                    return numeric_ids
            except Exception as e:
                print(f"{retry} error statistics output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
    
    def chat_without_stream_for_socre_feedback_latest_v3(self, message: str, response: str, filtered_retrieve_result= [], retrieve_result = [], flag_TF = True):
        context = ""
        for sentence in filtered_retrieve_result:
            context = context + '\n' + sentence
        prompt_history = llama_QA_graph_prompt.format(query = message, context = context)

        if flag_TF:
            flag_str = "Correct"
        else:
            flag_str = "Incorrect"
        retrieve_str = ""
        for idx, tmp in enumerate(retrieve_result, start=0):
            retrieve_str = retrieve_str + "\n" + str(idx) + ": " + tmp 
        # prompt = knowledge_graph_score_feedback_prompt_4_1_latest.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        prompt = knowledge_graph_score_feedback_prompt_gemini.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        # prompt = knowledge_graph_score_feedback_prompt_qwen3.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)

        
        history = [
            {"role": "user", "content": prompt_history},
            {"role": "assistant", "content": response},
            {"role": "user", "content": prompt},
        ]
        # print(f"history: \n{json.dumps(history, indent = 2)}")

        output = None
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                output = self._llm.chat_with_ai_multi_round('', history) # openai huggingface
                # print(f"feedback output {retry}: {output}")
                line_list =  output.strip().split('\n')
                correct_numbers = {}
                error_numbers = {}
                insufficient_str = ''
                for line in line_list:
                    # if line.strip() == 'Insufficient search information':
                    if 'Insufficient information' in line.strip():
                        insufficient_str = 'Insufficient information'
                        continue
                        # break
                    elif "No feedback" in line.strip():
                        insufficient_str = "No feedback"
                        continue
                        # break
                    elif line.strip().startswith("Correct: "):
                        correct_pairs = line.strip().split("Correct:")[1].strip().split()
                        for pair in correct_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    correct_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                    elif line.strip().startswith('Error: '):
                        error_pairs = line.strip().split("Error:")[1].strip().split()
                        for pair in error_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    error_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                assert (correct_numbers or error_numbers or insufficient_str)
                return insufficient_str, correct_numbers, error_numbers
            except Exception as e:
                print(f"feedback output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return '', {}, {}
    
    # 无多轮对话
    def chat_without_stream_for_socre_feedback_latest_v4(self, message: str, response: str, filtered_retrieve_result= [], flag_TF = True):
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context = context + f"{idx}. {sentence}\n"
        if flag_TF:
            flag_str = "Correct"
        else:
            flag_str = "Incorrect"
        # prompt = knowledge_graph_score_feedback_prompt_4_1_latest.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        # prompt = knowledge_graph_score_feedback_prompt_gemini.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        prompt = knowledge_graph_score_feedback_prompt_qwen3.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = context) # 提示词被改了记得换回来
        # prompt = knowledge_graph_score_feedback_prompt_llama.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = context)

        output = None
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"feedback output {retry}: {output}")
                line_list =  output.strip().split('\n')
                correct_numbers = {}
                error_numbers = {}
                insufficient_str = ''
                for line in line_list:
                    # if line.strip() == 'Insufficient search information':
                    if 'Insufficient information' in line.strip():
                        insufficient_str = 'Insufficient information'
                        continue
                        # break
                    elif "No feedback" in line.strip():
                        insufficient_str = "No feedback"
                        continue
                        # break
                    elif line.strip().startswith("Correct: "):
                        correct_pairs = line.strip().split("Correct:")[1].strip().split()
                        for pair in correct_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    correct_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                    elif line.strip().startswith('Error: '):
                        error_pairs = line.strip().split("Error:")[1].strip().split()
                        for pair in error_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    error_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                assert (correct_numbers or error_numbers or insufficient_str)
                return insufficient_str, correct_numbers, error_numbers
            except Exception as e:
                print(f"feedback output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return '', {}, {}
    
    # 无多轮对话
    def chat_without_stream_for_socre_feedback_latest_v4_api(self, message: str, response: str, filtered_retrieve_result= [], flag_TF = True):
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context = context + f"{idx}. {sentence}\n"
        if flag_TF:
            flag_str = "Correct"
        else:
            flag_str = "Incorrect"
        # prompt = knowledge_graph_score_feedback_prompt_4_1_latest.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        # prompt = knowledge_graph_score_feedback_prompt_gemini.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = retrieve_str)
        prompt = knowledge_graph_score_feedback_prompt_qwen3.format(question = message, flag_TF = flag_str, last_response = response, knowledge_statement_sets = context) # 提示词被改了记得换回来
        # prompt = knowledge_graph_score_feedback_prompt_llama.format(question = message, flag_TF = flag_str, last_response = response, knowlege_sentence = context)

        messages=[{'role': 'system', 'content': ''},
                            {'role': 'user', 'content': prompt}]

        output = None
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt='0', history=messages)
                # print(f"feedback output {retry}: {output}")
                line_list =  output.strip().split('\n')
                correct_numbers = {}
                error_numbers = {}
                insufficient_str = ''
                for line in line_list:
                    # if line.strip() == 'Insufficient search information':
                    if 'Insufficient information' in line.strip():
                        insufficient_str = 'Insufficient information'
                        continue
                        # break
                    elif "No feedback" in line.strip():
                        insufficient_str = "No feedback"
                        continue
                        # break
                    elif line.strip().startswith("Correct: "):
                        correct_pairs = line.strip().split("Correct:")[1].strip().split()
                        for pair in correct_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    correct_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                    elif line.strip().startswith('Error: '):
                        error_pairs = line.strip().split("Error:")[1].strip().split()
                        for pair in error_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    error_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                assert (correct_numbers or error_numbers or insufficient_str)
                return insufficient_str, correct_numbers, error_numbers
            except Exception as e:
                print(f"feedback output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return '', {}, {}
    
    def chat_without_stream_for_socre_feedback_mindmap_api(self, input_text: str, evidence_list = []): # 问题+证据
        response_of_KG_list_path = ""
        for evidence in evidence_list:
            response_of_KG_list_path += f"{evidence}\n"

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

        output =  self._llm.chat_with_ai("", history)
        return output
    
    def chat_without_stream_for_socre_feedback_Reasoning_Path_api(self, user_question: str, final_answer: str, evidence_list = []): # 问题+证据
        retrieved_knowledge = ""
        for evidence in evidence_list:
            retrieved_knowledge += f"{evidence}\n"

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
        
        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        print(f"-----prompt----")
        print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        print(f"-----assistant-----\n{history[2]['content']}")
        print(f"-----user2-----\n{history[3]['content']}")
        output =  self._llm.chat_with_ai("", history)
        return output
    
    def chat_without_stream_for_socre_feedback_Reasoning_Path_direct_api(self, user_question: str, final_answer: str, evidence_list = []): # 问题+证据
        retrieved_knowledge = ""
        for evidence in evidence_list:
            retrieved_knowledge += f"{evidence}\n"

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
            # {
            #     "role": "user",
            #     "content": (
            #         "Excellent. Now, based on the user's question, the `Retrieved Evidence`, and the `Final Answer` you provided in the last turn, "
            #         "please identify and show the exact reasoning path that supports your answer. Think step by step.\n\n"
                    
            #         "Output 1: Show the inference process as a string. For useful knowledge extracted from the `Retrieved Evidence`, "
            #         "you must connect them to infer the final result and provide two scores in a `[Relevance, Contribution]` format.\n"
            #         "Relevance to the Question (Score 1-3): How derictly relevant the evidence is to the user's original question.\n"
            #         "Contribution to the Answer (Score 1-3): How derictly relevant the evidence is to the final answer.\n"
            #         "(A higher score means greater relevance or contribution).\n"
            #         "Represent the inference process in this exact format:\n"
            #         "('entity name'->'relation name'->...)[Relevance, Contribution]->"
            #         "('entity name'->'relation name'->...)[Relevance, Contribution]->"
            #         "Result('the `Final Answer` you provided in the last turn').\n"

            #         # "Here is a sample to strictly guide your output format:\n"
            #         # "--- SAMPLE START ---\n"

            #         # "Output 1:\n"
            #         # "('Patient'->'has been experiencing'->'hoarse voice')->"
            #         # "('hoarse voice'->'could be caused by'->'laryngitis')->"
            #         # "('laryngitis'->'requires'->'physical examination of the throat')->"
            #         # "'physical examination of the throat'->'may include'->'laryngoscopy')->"
            #         # "Result('laryngitis').\n\n"

            #         # "--- SAMPLE END ---"

            #         "Here is a sample to strictly guide your output format:\n"
            #         "User Question: What tests should be done if the patient has a hoarse throat?\n"
            #         "Final Answer: Laryngoscopy\n"
            #         "--- SAMPLE START ---\n"
                    
            #         "Output 1:\n"
            #         "('Patient'->'has been experiencing'->'hoarse voice')[3,1]->"
            #         "('hoarse voice'->'could be caused by'->'laryngitis')[2,1]->"
            #         "('laryngitis'->'requires'->'physical examination of the throat')[1,1]->"
            #         "'physical examination of the throat'->'may include'->'laryngoscopy')[1,3]->"
            #         "Result('Laryngoscopy').\n\n"
            #         "--- SAMPLE END ---"
            #     )
            # }
            {
                "role": "user",
                "content": (
                    "Excellent. Now, based on the `User Question`, `Retrieved Evidence`, and `Final Answer` from the previous context, your task is to construct the precise reasoning path that justifies the answer. Think step by step.\n\n"
                    "Your output must be a single section, `Output 1`, showing the inference process as a string. For each piece of useful knowledge extracted from the `Retrieved Evidence`, you must connect them and provide two scores in a `[Relevance, Contribution]` format based on the following strict criteria:\n\n"
                    
                    "  - **1. Relevance to the Question (Score 1-3):** This score measures the **direct semantic relevance** between the evidence and the `User Question`. \n"
                    "    - **Score 3 (High):** The evidence shares key concepts and is very similar in meaning to the question.\n"
                    "    - **Score 2 (Medium):** The evidence is related to the question but requires an intermediate logical step.\n"
                    "    - **Score 1 (Low):** The evidence is topically related but not semantically close to the question.\n\n"
                    
                    "  - **2. Contribution to the Answer (Score 1-3):** This score measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
                    "    - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
                    "    - **Score 2 (Medium):** The evidence contains a significant part of the `Final Answer`.\n"
                    "    - **Score 1 (Low):** The evidence does not contain the `Final Answer` or any part of it.\n\n"
                    "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component. Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer.\n"
     
                    
                    "Represent the inference process in this exact format, ending with the final result:\n"
                    "Path-based Evidence number('entity name'->'relation name'->...)[Relevance, Contribution]->\n"
                    "Path-based Evidence number('entity name'->'relation name'->...)[Relevance, Contribution]->\n"
                    "Result('The Final Answer').\n\n"
                    
                    "Here is a sample to strictly guide your output format:\n"
                    "--- SAMPLE START ---\n"
                    "User Question: What tests should be done if the patient has a hoarse throat?\n"
                    "Final Answer: Laryngoscopy\n\n"
            
                    "Output 1:\n"
                    "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')[3, 1]->"
                    "Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')[2, 1]->"
                    "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')[1, 1]->"
                    "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')[1, 3]->"
                    "Result('Laryngoscopy').\n"
                    "--- SAMPLE END ---"
                )
            }
        ]
        
        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        print(f"-----prompt----")
        print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        print(f"-----assistant-----\n{history[2]['content']}")
        print(f"-----user2-----\n{history[3]['content']}")

        output =  self._llm.chat_with_ai("", history)
        return output


    def chat_without_stream_for_socre_feedback_Reasoning_Path_Keywords_api(self, user_question: str, final_answer: str, evidence_list = [], keywords = []): # 问题+证据
        retrieved_knowledge = ""
        for evidence in evidence_list:
            retrieved_knowledge += f"{evidence}\n"
        retrieved_keywords = ""
        for keyword in keywords:
            retrieved_keywords += f"{keyword}\n"

        history_bak = [
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
                    f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
                    f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
                    f"### Final Answer:\n{final_answer}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step.\n\n"
                    
                    "### Rules for Path Construction:\n"
                    "1.  **Multiple Paths:** You can and should create multiple, separate reasoning paths. The combination of all paths should be sufficient to logically derive the `Final Answer`.\n"
                    "2.  **Start from a Keyword:** Every path **must** begin with one of the provided `Keywords` as its starting entity.\n"
                    "3.  **Link via Common Entities:** To chain evidence within a path, consecutive steps **must** be connected by a shared entity. The entity at the end of one step must be the starting entity of the next step.The entity at the end of one step must be the starting entity of the next step.The entity at the end of one step must be the starting entity of the next step.\n"
                    "4.  **End Each Path:** Clearly mark the end of each complete reasoning path with the text `->Path Reasoning Ends`.\n\n"
                    
                    "### Scoring Criteria:\n"
                    "For each piece of evidence in your paths, provide two scores in a `[Relevance, Contribution]` format:\n"
                    "  - **1. Relevance to the Question (Score 1-3):** Measures the **direct semantic relevance** between the evidence and the `User Question`.\n"
                    "    - **Score 3 (High):** The evidence shares key concepts and is very similar in meaning to the question.\n"
                    "    - **Score 1 (Low):** The evidence is topically related but not semantically close.\n"
                    "  - **2. Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
                    "    - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
                    "    - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
                    "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
                    " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
                    "### Output Format:\n"
                    "Present your entire output as a single block of text. Follow this format exactly:\n"
                    "Path-based Evidence number('keyword'->'relation'->'entity')[Relevance, Contribution]->"
                    "Path-based Evidence number('entity'<-'relation'<-'entity')[Relevance, Contribution]->Path Reasoning Ends\n"
                    "Path-based Evidence number('keyword'->'relation'->...)[Relevance, Contribution]->Path Reasoning Ends\n"
                    "Result('The Final Answer').\n\n"
            
                    # "--- SAMPLE START ---\n"
                    # "### Context Provided:\n"
                    # "User Question: What tests should be done if the patient has a hoarse throat?\n"
                    # "Keywords: Patient, hoarse voice\n"
                    # "Retrieved Evidence:\n"
                    # "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
                    # "  Path-based Evidence 2: ('laryngitis'->'can cause'->'hoarse voice')\n"
                    # "  Path-based Evidence 3: ('hoarse voice'->'requires'->'physical examination of the throat')\n"
                    # "  Path-based Evidence 4: ('physical examination of the throat'->'may include'->'laryngoscopy')\n"
                    # "Final Answer: Laryngoscopy\n\n"
                    # "### Expected Output:\n"
                    # "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')[3, 1]->"
                    # "Path-based Evidence 2('hoarse voice'<-'can cause'<-'laryngitis')[2, 1]->Path Reasoning Ends\n"
                    # "Path-based Evidence 3('hoarse voice'->'requires'->'physical examination of the throat')[2, 1]->"
                    # "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')[1, 3]->Path Reasoning Ends\n"
                    # "Result('Laryngoscopy').\n"
                    # "--- SAMPLE END ---"

                    "--- SAMPLE START ---\n"
                    "### Context Provided:\n"
                    "User Question: What tests should be done if the patient has a hoarse throat?\n"
                    "Keywords: Patient, hoarse voice\n"
                    "Retrieved Evidence:\n"
                    "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
                    "  Path-based Evidence 2: ('laryngitis'->'can cause'->'hoarse voice')\n"
                    "  Path-based Evidence 3: ('hoarse voice'->'requires'->'physical examination of the throat')\n"
                    "  Path-based Evidence 4: ('laryngoscopy'->'is part of'->'physical examination of the throat')\n"
                    "Final Answer: Laryngoscopy\n\n"
                    "### Expected Output:\n"
                    "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')[3, 1]->"
                    "Path-based Evidence 2('hoarse voice'<-'can cause'<-'laryngitis')[2, 1]->Path Reasoning Ends\n"
                    "Path-based Evidence 3('hoarse voice'->'requires'->'physical examination of the throat')[2, 1]->"
                    "Path-based Evidence 4('physical examination of the throat'<-'is part of'<-'laryngoscopy')[1, 3]->Path Reasoning Ends\n"
                    "Result('Laryngoscopy').\n"
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
                    f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
                    f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
                    f"### Final Answer:\n{final_answer}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step.\n\n"
                    
                    "### Rules for Path Construction:\n"
                    "1.  **Multiple Paths:** You can and should create multiple, separate reasoning paths. The combination of all paths should be sufficient to logically derive the `Final Answer`.\n"
                    "2.  **Start from a Keyword:** Every path **must** begin with one of the provided `Keywords` as its starting entity.\n"
                    "3.  **Link via Common Entities:** To chain evidence within a path, consecutive steps **must** be connected by a shared entity. The entity at the end of one step must be the starting entity of the next step.\n"
                    "4.  **End Each Path:** Clearly mark the end of each complete reasoning path with the text `->Path Reasoning Ends`.\n\n"
                    
                    "### Scoring Criteria:\n"
                    "For each piece of evidence in your paths, provide two scores in a `[Relevance, Contribution]` format:\n"
                    "  - **1. Relevance to the Question (Score 1-3):** Measures the **direct semantic relevance** between the evidence and the `User Question`.\n"
                    "    - **Score 3 (High):** The evidence shares key concepts and is very similar in meaning to the question.\n"
                    "    - **Score 1 (Low):** The evidence is topically related but not semantically close.\n"
                    "  - **2. Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
                    "    - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
                    "    - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
                    "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
                    " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
                    "### Output Format:\n"
                    "Present your entire output as a single block of text. Follow this format exactly:\n"
                    "Path-based Evidence number('keyword'->'relation'->'entity')[Relevance, Contribution]->"
                    "Path-based Evidence number('entity'<-'relation'<-'entity')[Relevance, Contribution]->Path Reasoning Ends\n"
                    "Path-based Evidence number('keyword'->'relation'->...)[Relevance, Contribution]->Path Reasoning Ends\n"
                    "Result('The Final Answer').\n\n"

                    "**Note on Reversed Evidence (`<-`):** The `<-` arrows mean you are reading the evidence in reverse (from tail entity to head entity) to connect the path. The underlying fact is the same.\n"
                    "  - *Normal:* `'laryngitis'->'can cause'->'hoarse voice'`\n"
                    "  - *Reversed:* `'hoarse voice'<-'can cause'<-'laryngitis'`\n"
                    "The entity at the end of one step must be the starting entity of the next step.\n\n"
            
                    "--- SAMPLE START ---\n"
                    "### Context Provided:\n"
                    "User Question: What tests should be done if the patient has a hoarse throat?\n"
                    "Keywords: Patient, hoarse voice\n"
                    "Retrieved Evidence:\n"
                    "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
                    "  Path-based Evidence 2: ('laryngitis'->'can cause'->'hoarse voice')\n"
                    "  Path-based Evidence 3: ('hoarse voice'->'requires'->'physical examination of the throat')\n"
                    "  Path-based Evidence 4: ('laryngoscopy'->'is part of'->'physical examination of the throat')\n"
                    "Final Answer: Laryngoscopy\n\n"
                    "### Expected Output:\n"
                    "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')[3, 1]->"
                    "Path-based Evidence 2('hoarse voice'<-'can cause'<-'laryngitis')[2, 1]->Path Reasoning Ends\n"
                    "Path-based Evidence 3('hoarse voice'->'requires'->'physical examination of the throat')[2, 1]->"
                    "Path-based Evidence 4('physical examination of the throat'<-'is part of'<-'laryngoscopy')[1, 3]->Path Reasoning Ends\n"
                    "Result('Laryngoscopy').\n"
                    "--- SAMPLE END ---"
                )
            }
        ]
        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        print(f"-----prompt----")
        print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        print(f"-----assistant-----\n{history[2]['content']}")
        print(f"-----user2-----\n{history[3]['content']}")

        output =  self._llm.chat_with_ai("", history)
        return output

    def chat_without_stream_for_socre_feedback_Reasoning_Path_Reverse_api(self, user_question: str, final_answer: str, evidence_list = [], keywords = []): # 问题+证据
        retrieved_knowledge = ""
        retrieved_knowledge_reverse = ""
        for index, evidence in enumerate(evidence_list):
            if evidence[3] == '->':
                retrieved_knowledge += f"  Path-based Evidence {index}: '{evidence[0]}'->'{evidence[1].replace('_',' ')}'->'{evidence[2]}'\n"
                retrieved_knowledge_reverse += f"  Reversed Evidence {index}: '{evidence[2]}'<-'{evidence[1].replace('_',' ')}'<-'{evidence[0]}'\n"
            elif evidence[3] == '<-':
                retrieved_knowledge += f"  Path-based Evidence {index}: '{evidence[2]}'->'{evidence[1].replace('_',' ')}'->'{evidence[0]}'\n"
                retrieved_knowledge_reverse += f"  Reversed Evidence {index}: '{evidence[0]}'<-'{evidence[1].replace('_',' ')}'<-'{evidence[2]}'\n"
            else:
                print(f"----Error evidence, Error direction----")
        retrieved_keywords = ""
        for index, keyword in enumerate(keywords):
            retrieved_keywords += f"  Keyword {index}: {keyword}\n"

        
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
                    f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
                    f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
                    
                    "**Note on Reversed Evidence (`<-`):** The `<-` arrows mean you are reading the evidence in reverse (from tail entity to head entity) to connect the path. The underlying fact is the same.\n"
                    "  - *Normal:* `'laryngitis'->'can cause'->'hoarse voice'`\n"
                    "  - *Reversed:* `'hoarse voice'<-'can cause'<-'laryngitis'`\n"
                    f"### Reversed Evidence:\n{retrieved_knowledge_reverse}\n\n" 
                    f"### Final Answer:\n{final_answer}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step.\n\n"
                    
                    "### Rules for Path Construction:\n"
                    "1.  **Multiple Paths:** You can and should create multiple, separate reasoning paths. The combination of all paths should be sufficient to logically derive the `Final Answer`.\n"
                    "2.  **Start from a Keyword:** Every path **must** begin with one of the provided `Keywords` as its starting entity.\n"
                    "3.  **Link via Common Entities:** To chain evidence within a path, consecutive steps **must** be connected by a shared entity. The entity at the end of one step must be the starting entity of the next step.The entity at the end of one step must be the starting entity of the next step.The entity at the end of one step must be the starting entity of the next step.\n"
                    "4.  **End Each Path:** Clearly mark the end of each complete reasoning path with the text `->Path Reasoning Ends`.\n\n"
                    
                    "### Scoring Criteria:\n"
                    "For each piece of evidence in your paths, provide two scores in a `[Relevance, Contribution]` format:\n"
                    "  - **1. Relevance to the Question (Score 1-3):** Measures the **direct semantic relevance** between the evidence and the `User Question`.\n"
                    "    - **Score 3 (High):** The evidence shares key concepts and is very similar in meaning to the question.\n"
                    "    - **Score 1 (Low):** The evidence is topically related but not semantically close.\n"
                    "  - **2. Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
                    "    - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
                    "    - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
                    "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
                    " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
                    "### Output Format:\n"
                    "Present your entire output as a single block of text. Follow this format exactly:\n"
                    "Path-based Evidence number('keyword'->'relation'->'entity')[Relevance, Contribution]->"
                    "Path-based Evidence number('entity'<-'relation'<-'entity')[Relevance, Contribution]->Path Reasoning Ends\n"
                    "Path-based Evidence number('keyword'->'relation'->...)[Relevance, Contribution]->Path Reasoning Ends\n"
                    "Result('The Final Answer').\n\n"

                    "The entity at the end of one step must be the starting entity of the next step.\n\n"
            
                    "--- SAMPLE START ---\n"
                    "### Context Provided:\n"
                    "User Question: What tests should be done if the patient has a hoarse throat?\n"
                    "Keywords: hoarse voice\n"
                    "Retrieved Evidence:\n"
                    "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
                    "  Path-based Evidence 2: ('laryngitis'->'could be caused by'->'hoarse voice')\n"
                    "  Path-based Evidence 3: (physical examination of the throat' -> 'is required by' -> 'laryngitis')\n"
                    "  Path-based Evidence 4: ('physical examination of the throat'->'may include'->'laryngoscopy')\n"
                    "Reversed Evidence:\n"
                    "  Reversed Evidence 1: ('hoarse voice'<-'has been experiencing'<-'Patient')\n"
                    "  Reversed Evidence 2: ('hoarse voice'<-'could be caused by'<-'laryngitis')\n"
                    "  Reversed Evidence 3: ('laryngitis'->'is required by'->'physical examination of the throat')\n"
                    "  Reversed Evidence 4: ('laryngoscopy'<-'may include'<-'physical examination of the throat')\n"
                    "Final Answer: Laryngoscopy\n\n"
                    "### Expected Output:\n"
                    "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')[3, 1]->"
                    "Reversed Evidence 2('hoarse voice'<-'could be caused by'<-'laryngitis')[2, 1]->"
                    "Reversed Evidence 3('laryngitis'<-'is required by'<-'physical examination of the throat')[1, 1]->"
                    "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')[1, 3]->Path Reasoning Ends\n"
                    "Result('Laryngoscopy').\n"
                    "--- SAMPLE END ---"
                )
            }
        ]
        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        print(f"-----prompt----")
        print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        print(f"-----assistant-----\n{history[2]['content']}")
        print(f"-----user2-----\n{history[3]['content']}")

        # assert False
        output =  self._llm.chat_with_ai("", history)
        return output

    def parse(self, res):
        path_evidence_numbers = re.findall(r"Path-based Evidence (\d+)", res)
        contribution_numbers = re.findall(r"Contribution\[(\d+)\]", res)
        result_match = re.search(r"Result\((.*?)\)", res)
        final_result = result_match.group(1)
        return [ int(item) for item in path_evidence_numbers], [ int(item) for item in contribution_numbers], final_result.strip(" '\"")
    
    def chat_without_stream_for_socre_feedback_Reasoning_Path_final_api(self, user_question: str, final_answer: str, evidence_list = [], keywords = []): # 问题+证据
        retrieved_knowledge = ""
        retrieved_knowledge_reverse = ""
        for index, evidence in enumerate(evidence_list):
            if evidence[3] == '->':
                retrieved_knowledge += f"  Path-based Evidence {index}: '{evidence[0]}'->'{evidence[1].replace('_',' ')}'->'{evidence[2]}'\n"
                retrieved_knowledge_reverse += f"  Reversed Evidence {index}: '{evidence[2]}'<-'{evidence[1].replace('_',' ')}'<-'{evidence[0]}'\n"
            elif evidence[3] == '<-':
                retrieved_knowledge += f"  Path-based Evidence {index}: '{evidence[2]}'->'{evidence[1].replace('_',' ')}'->'{evidence[0]}'\n"
                retrieved_knowledge_reverse += f"  Reversed Evidence {index}: '{evidence[0]}'<-'{evidence[1].replace('_',' ')}'<-'{evidence[2]}'\n"
            else:
                print(f"----Error evidence, Error direction----")
        retrieved_keywords = ""
        for index, keyword in enumerate(keywords):
            retrieved_keywords += f"  Keyword {index}: {keyword}\n"

        
        history_original = [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are an expert in reasoning over knowledge graphs. Your primary task is to construct a clear reasoning path "
            #         "that supports a given answer, based on a user's question and a set of provided knowledge evidence. "
            #         "You must strictly use only the provided evidence to build the reasoning chain."
            #     )
            # },
            # {
            #     "role": "user",
            #     "content": f"User Question: \"{user_question}\""
            # },
            # {
            #     "role": "assistant",
            #     "content": (
            #         "Based on the user's question, I have gathered the following potentially relevant evidence and formulated an answer.\n\n"
            #         f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
            #         f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
            #         f"### Final Answer:\n{final_answer}"
            #     )
            # },
            # {
            #     "role": "user",
            #     "content": (
            #         "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step.\n\n"
                    
            #         "### Rules for Path Construction:\n"
            #         "1.  Overall Goal: Construct a set of separate reasoning paths. The combined logic of all paths must be sufficient to justify the Final Answer.\n"
            #         "2.  Initiating a Path: To begin a new reasoning path, the first piece of evidence you select must contain at least one of the provided Keywords. The keyword can be either the head or the tail entity of that evidence.\n"
            #         "3.  Linking Steps: To connect consecutive steps within a single path, they must share at least one common entity. The position of this shared entity does not matter; it can be the head or the tail in either step.\n"
            #         "4. Terminating a Path: Clearly mark the end of each complete reasoning path with the text ->Path Reasoning Ends.\n"
                    
            #         "### Scoring Criteria:\n"
            #         "For each piece of evidence in your paths, provide score in a `Contribution[Contribution]` format:\n"
            #         "**Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
            #         "  - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
            #         "  - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
            #         "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
            #         " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
            #         "### Output Format:\n"
            #         "Present your entire output as a single block of text. Follow this format exactly:\n"
            #         "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->"
            #         "Path-based Evidence number('entity'->'relation'->'entity')Contribution[Contribution]->Path Reasoning Ends\n"
            #         "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->Path Reasoning Ends\n"
            #         "Result('The Final Answer').\n\n"
            
            #         "--- SAMPLE START ---\n"
            #         "### Context Provided:\n"
            #         "User Question: What tests should be done if the patient has a hoarse throat?\n"
            #         "Retrieved Keywords:\n"
            #         "  Keyword 0: Patient\n"
            #         "  Keyword 1: The hoarse voice\n"
            #         "Retrieved Evidence:\n"
            #         "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
            #         "  Path-based Evidence 2: ('laryngitis'->'could be caused by'->'The hoarse voice')\n"
            #         "  Path-based Evidence 3: ('laryngitis'->'requires'->'physical examination of the throat')\n"
            #         "  Path-based Evidence 4: ('physical examination of the throat'->'may include'->'laryngoscopy')\n"
            #         "Final Answer: Laryngoscopy\n\n"
            #         "### Expected Output:\n"
            #         "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')Contribution[1]->Path Reasoning Ends\n"
            #         "Path-based Evidence 2('laryngitis'->'could be caused by'->'The hoarse voice')Contribution[1]->"
            #         "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')Contribution[1]->"
            #         "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')[3]->Path Reasoning Ends\n"
            #         "Result('Laryngoscopy').\n"
            #         "--- SAMPLE END ---"
            #     )
            # }
        ]
        
        history_v1 = [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are an expert in reasoning over knowledge graphs. Your primary task is to construct a clear reasoning path "
            #         "that supports a given answer, based on a user's question and a set of provided knowledge evidence. "
            #         "You must strictly use only the provided evidence to build the reasoning chain."
            #     )
            # },
            # {
            #     "role": "user",
            #     "content": f"User Question: \"{user_question}\""
            # },
            # {
            #     "role": "assistant",
            #     "content": (
            #         "Based on the user's question, I have gathered the following potentially relevant evidence and formulated an answer.\n\n"
            #         f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
            #         f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
            #         f"### Final Answer:\n{final_answer}"
            #     )
            # },
            # {
            #     "role": "user",
            #     "content": (
            #         "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step.\n\n"
                    
            #         "### Rules for Path Construction:\n"
            #         "* There may be multiple reasoning paths leading to the final answer. For each valid path, explicitly show the inference process as a string. Within each path, extract which knowledge is used from which Path-based Evidence, and clearly indicate how these pieces of knowledge are connected step-by-step to reach an intermediate or final conclusion.\n"

            #         "1. Path Selection Criteria\n"
            #         "(These rules govern the overall set of paths you will present.)\n"
            #         "Relevance: You must begin by ignoring any piece of Retrieved Evidence that is irrelevant or does not logically contribute to reaching the Final Answer. Focus only on useful evidence.\n"
            #         "Efficiency: Each selected path must be as concise and efficient as possible, using the minimum number of steps necessary to form a complete logical chain.\n"
            #         "Diversity: The chosen paths should be as different from each other as possible. Prioritize a set of paths that demonstrate varied reasoning angles and avoid presenting multiple paths that are highly similar or redundant.\n"
            #         "Quantity: You must generate a maximum of 4 distinct and optimized paths.\n\n"
                    
            #         "2. Path Formation Rules\n"
            #         "(These rules govern how each individual path is built.)\n"
            #         "Initiation: The first piece of evidence in any path must contain at least one of the provided Keywords.\n"
            #         "Linking: To connect consecutive steps within a path, they must share a common entity.\n"
            #         "Crucially, this shared entity must be an exact, case-sensitive match. For example, 'Laryngitis' and 'laryngitis' would not be considered a valid link.\n"
            #         "The position of the shared entity (head or tail) does not matter.\n\n"

                    
            #         "### Scoring Criteria:\n"
            #         "For each piece of evidence in your paths, provide score in a `Contribution[Contribution]` format:\n"
            #         "**Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
            #         "  - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
            #         "  - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
            #         "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
            #         " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
            #         "### Output Format:\n"
            #         "Present your entire output as a single block of text. Follow this format exactly:\n"
            #         "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->"
            #         "Path-based Evidence number('entity'->'relation'->'entity')Contribution[Contribution]->Result('The Final Answer').\n\n"
            #         "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->"
            #         "Result('The Final Answer').\n\n"
            
            #         "--- SAMPLE START ---\n"
            #         "### Context Provided:\n"
            #         "User Question: What tests should be done if the patient has a hoarse throat?\n"
            #         "Retrieved Keywords:\n"
            #         "  Keyword 0: Patient\n"
            #         "  Keyword 1: The hoarse voice\n"
            #         "Retrieved Evidence:\n"
            #         "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
            #         "  Path-based Evidence 2: ('laryngitis'->'could be caused by'->'The hoarse voice')\n"
            #         "  Path-based Evidence 3: ('laryngitis'->'requires'->'physical examination of the throat')\n"
            #         "  Path-based Evidence 4: ('physical examination of the throat'->'may include'->'laryngoscopy')\n"
            #         "  Path-based Evidence 5: ('hoarse voice'->'need'->'laryngoscopy')\n"
            #         "Final Answer: Laryngoscopy\n\n"
            #         "### Expected Output:\n"
            #         "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')Contribution[1]->"
            #         "Path-based Evidence 5('hoarse voice'->'need'->'laryngoscopy')->Result('Laryngoscopy').\n"
            #         "Path-based Evidence 2('laryngitis'->'could be caused by'->'The hoarse voice')Contribution[1]->"
            #         "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')Contribution[1]->"
            #         "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')[3]->"
            #         "Result('Laryngoscopy').\n"
            #         "--- SAMPLE END ---"
            #     )
            # }
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
                    f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
                    f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
                    f"### Final Answer:\n{final_answer}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step. The quality and validity of the paths are more important than the quantity.\n\n"
                    
                    "### Rules for Path Construction:\n"
                    "* There may be multiple reasoning paths leading to the final answer. For each valid path, explicitly show the inference process as a string. Within each path, extract which knowledge is used from which piece of evidence, and clearly indicate how these pieces of knowledge are connected step-by-step to reach an intermediate or final conclusion.\n\n"
                    
                    "1. Path Selection Criteria\n"
                    "(These rules govern the overall set of paths you will present.)\n"
                    
                    "* Guiding Principle: Your primary goal is to identify and present only the most sound and varied reasoning paths supported by the evidence. Quality and logical validity are paramount and take precedence over the number of paths generated.\n"
                    "* Relevance: You must begin by ignoring any piece of Retrieved Evidence that is irrelevant or does not logically contribute to reaching the Final Answer. Focus only on useful evidence.\n"
                    "* Efficiency: Each selected path must be as concise and efficient as possible, using the minimum number of steps necessary to form a complete logical chain.\n"
                    "* Diversity: The chosen paths should be substantively different from each other. Prioritize a set of paths that demonstrate varied reasoning angles. Strictly avoid presenting multiple paths that are redundant or only superficially different.\n"
                    "* Quantity: Present up to a maximum of 4 distinct paths. If the available evidence only supports one, two, or three strong and diverse paths, you should only present that number. Do not create weak, redundant, or illogical paths simply to meet the maximum number.\n\n"
                    
                    "2. Path Formation Rules\n"
                    "(These rules govern how each individual path is built.)\n"
                    
                    "* Initiation: The first piece of evidence in any path must contain at least one of the provided Keywords.\n"
                    "* Linking: To connect consecutive steps within a path, they must share a common entity.\n"
                    "* Crucially, this shared entity must be an exact, case-sensitive match. For example, 'Laryngitis' and 'laryngitis' would not be considered a valid link.\n"
                    "* The position of the shared entity (head or tail) does not matter.\n\n"

                    
                    "### Scoring Criteria:\n"
                    "For each piece of evidence in your paths, provide score in a `Contribution[Contribution]` format:\n"
                    "**Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
                    "  - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
                    "  - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
                    "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
                    " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
                    "### Output Format:\n"
                    "Present your entire output as a single block of text. Follow this format exactly:\n"
                    "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->"
                    "Path-based Evidence number('entity'->'relation'->'entity')Contribution[Contribution]->Result('The Final Answer').\n\n"
                    "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->"
                    "Result('The Final Answer').\n\n"
            
                    "--- SAMPLE START ---\n"
                    "### Context Provided:\n"
                    "User Question: What tests should be done if the patient has a hoarse throat?\n"
                    "Retrieved Keywords:\n"
                    "  Keyword 0: Patient\n"
                    "  Keyword 1: The hoarse voice\n"
                    "Retrieved Evidence:\n"
                    "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
                    "  Path-based Evidence 2: ('laryngitis'->'could be caused by'->'The hoarse voice')\n"
                    "  Path-based Evidence 3: ('laryngitis'->'requires'->'physical examination of the throat')\n"
                    "  Path-based Evidence 4: ('physical examination of the throat'->'may include'->'laryngoscopy')\n"
                    "  Path-based Evidence 5: ('hoarse voice'->'need'->'laryngoscopy')\n"
                    "Final Answer: Laryngoscopy\n\n"
                    "### Expected Output:\n"
                    "Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')Contribution[1]->"
                    "Path-based Evidence 5('hoarse voice'->'need'->'laryngoscopy')Contribution[3]->Result('Laryngoscopy').\n"
                    "Path-based Evidence 2('laryngitis'->'could be caused by'->'The hoarse voice')Contribution[1]->"
                    "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')Contribution[1]->"
                    "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')[3]->"
                    "Result('Laryngoscopy').\n"
                    "--- SAMPLE END ---"
                )
            }
        ]

        history_single = [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are an expert in reasoning over knowledge graphs. Your primary task is to construct a clear reasoning path "
            #         "that supports a given answer, based on a user's question and a set of provided knowledge evidence. "
            #         "You must strictly use only the provided evidence to build the reasoning chain."
            #     )
            # },
            # {
            #     "role": "user",
            #     "content": f"User Question: \"{user_question}\""
            # },
            # {
            #     "role": "assistant",
            #     "content": (
            #         "Based on the user's question, I have gathered the following potentially relevant evidence and formulated an answer.\n\n"
            #         f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
            #         f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n" 
            #         f"### Final Answer:\n{final_answer}"
            #     )
            # },
            # {
            #     "role": "user",
            #     "content": (
            #         "Excellent. Now, using the `User Question`, `Keywords`, `Retrieved Evidence`, and `Final Answer` I just provided, your task is to construct a set of precise reasoning paths that collectively justify the answer. Think step by step.\n\n"
                    
            #         "### Rules for Path Construction:\n"
            #         "1.  Show the inference process as a string. Extract which knowledge is used from which `Path-based Evidence`, and indicate how they connect to infer the Final Answer."
            #         "2.  Initiating a Path: To begin a new reasoning path, the first piece of evidence you select must contain at least one of the provided Keywords. The keyword can be either the head or the tail entity of that evidence.\n"
            #         "3.  Linking Steps: To connect consecutive steps within a single path, they must share at least one common entity. The position of this shared entity does not matter; it can be the head or the tail in either step.\n"
                    
            #         "### Scoring Criteria:\n"
            #         "For each piece of evidence in your paths, provide score in a `Contribution[Contribution]` format:\n"
            #         "**Contribution to the Answer (Score 1-3):** Measures if the evidence **directly contains the `Final Answer`** or a critical part of it.\n"
            #         "  - **Score 3 (High):** The evidence explicitly contains the complete `Final Answer`.\n"
            #         "  - **Score 1 (Low):** The evidence does not contain the `Final Answer`.\n\n"
            #         "**Important Note for Numerical Answers:** For answers that include numerical values (e.g., quantities, prices), the numerical value found in the evidence **MUST exactly match** the corresponding numerical value in the ` Answer` for the evidence to receive a Contribution Score of 3 or 2 for that specific numerical component."
            #         " Numerical values that are merely 'close' or 'approximate' will not be considered a match and will not contribute to a higher score for that part of the answer."

            
            #         "### Output Format:\n"
            #         "Present your entire output as a single block of text. Follow this format exactly:\n"
            #         "Path-based Evidence number('keyword/enity'->'relation'->'entity/keyword')Contribution[Contribution]->"
            #         "Path-based Evidence number('entity'->'relation'->'entity')Contribution[Contribution]->"
            #         "Result('The Final Answer').\n\n"
            
            #         "--- SAMPLE START ---\n"
            #         "### Context Provided:\n"
            #         "User Question: What tests should be done if the patient has a hoarse throat?\n"
            #         "Retrieved Keywords:\n"
            #         "  Keyword 0: hoarse voice\n"
            #         "Retrieved Evidence:\n"
            #         "  Path-based Evidence 1: ('Patient'->'has been experiencing'->'hoarse voice')\n"
            #         "  Path-based Evidence 2: ('hoarse voice'->'can cause'->'laryngitis')\n"
            #         "  Path-based Evidence 3: ('laryngitis'->'requires'->'physical examination of the throat')\n"
            #         "  Path-based Evidence 4: ('physical examination of the throat'->'may include'->'laryngoscopy')\n"
            #         "Final Answer: Laryngoscopy\n\n"
            #         "### Expected Output:\n"
            #         "Path-based Evidence 2('hoarse voice'->'can cause'->'laryngitis')Contribution[1]->"
            #         "Path-based Evidence 3('laryngitis'->'requires'->'physical examination of the throat')Contribution[3]->"
            #         "Path-based Evidence 4('physical examination of the throat'->'may include'->'laryngoscopy')Contribution[3]->"
            #         "Result('Laryngoscopy').\n"
            #         "--- SAMPLE END ---"
            #     )
            # }
        ]

        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        print(f"-----prompt----")
        print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        print(f"-----assistant-----\n{history[2]['content']}")
        print(f"-----user2-----\n{history[3]['content']}")

        # assert False
        # output =  self._llm.chat_with_ai("", history)

        output = ''
        think_content = ''
        retry = 1
        while retry <= 3:
            try:
                output, think_content =  self._llm.chat_with_ai("think_return", history)
                # print(f"feedback output {retry}: {output}")
                line_list =  output.strip().split('\n')
                line_list = [ line  for line in line_list if line]
                path_evidence_numbers_2d = []
                contribution_numbers_2d = []
                final_result_list = []
                for line in line_list:
                    path_evidence_numbers, contribution_numbers, final_result = self.parse(line)
                    if len(path_evidence_numbers) == len(contribution_numbers) and final_result:
                        path_evidence_numbers_2d.append(path_evidence_numbers)
                        contribution_numbers_2d.append(contribution_numbers)
                        final_result_list.append(final_result)
                if path_evidence_numbers_2d:
                    return output, path_evidence_numbers_2d, contribution_numbers_2d, final_result_list, think_content
                else:
                    raise Exception(f"----parse error----")
            except Exception as e:
                print(f"feedback_Reasoning_Path_final_api output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return output, [], [], [], ''
    

    def chat_without_stream_for_socre_feedback_Reasoning_Path_error_test_api(self, user_question: str, final_answer: str, evidence_list = [], keywords = []): # 问题+证据
        retrieved_knowledge = ""
        retrieved_knowledge_reverse = ""
        for index, evidence in enumerate(evidence_list):
            if evidence[3] == '->':
                retrieved_knowledge += f"  Path-based Evidence {index}: '{evidence[0]}'->'{evidence[1].replace('_',' ')}'->'{evidence[2]}'\n"
                retrieved_knowledge_reverse += f"  Reversed Evidence {index}: '{evidence[2]}'<-'{evidence[1].replace('_',' ')}'<-'{evidence[0]}'\n"
            elif evidence[3] == '<-':
                retrieved_knowledge += f"  Path-based Evidence {index}: '{evidence[2]}'->'{evidence[1].replace('_',' ')}'->'{evidence[0]}'\n"
                retrieved_knowledge_reverse += f"  Reversed Evidence {index}: '{evidence[0]}'<-'{evidence[1].replace('_',' ')}'<-'{evidence[2]}'\n"
            else:
                print(f"----Error evidence, Error direction----")
        retrieved_keywords = ""
        for index, keyword in enumerate(keywords):
            retrieved_keywords += f"  Keyword {index}: {keyword}\n"

        history = [
            {
                "role": "system",
                "content": (
                    "You are an AI expert in logical reasoning. Your primary task is to analyze provided evidence to construct clear, step-by-step reasoning paths that answer a user's question or fulfill a specific reasoning objective. You must strictly ground all reasoning in the provided evidence, without inventing facts or using external knowledge."
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
                    f"### Retrieved Keywords:\n{retrieved_keywords}\n\n"
                    f"### Retrieved Evidence:\n{retrieved_knowledge}\n\n"
                    f"### Final Answer:\n{final_answer}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Now for a more challenging task. Your goal is to act as a devil's advocate. Using the *same* `User Question`, `Keywords`, and `Retrieved Evidence` as before, you must disregard the previous `Final Answer`.\n\n"
                    "Your new task is to critically re-examine the evidence and construct reasoning paths that lead to **strictly contradictory answers**. These new conclusions must be logically and explicitly derived from the provided evidence.\n\n"
            
                    "### Core Directives for Finding Contradictory Paths:\n\n"
                    "**1. What is a Valid Contradictory Answer?**\n"
                    "*   **Strict Conflict:** A `New Contradictory Answer` must be a direct and factual contradiction to the `Final Answer`. It must be of the same semantic type and address the exact same question, but provide a conflicting value. For instance, if the question is 'What is the capital of X?' and the `Final Answer` is 'City A,' a valid contradictory answer is 'City B.' An answer like 'a famous landmark in City A' is **not** a valid contradiction.\n"
                    "*   **Absolute Independence from Original Logic:** This is a critical rule. A contradictory answer **cannot** be an entity that was part of the reasoning path for the original `Final Answer`. It must not be an intermediate step, a supporting fact, or a component of the original logic. The goal is to find a *genuinely separate and conflicting conclusion*, not to re-state a piece of the prior reasoning.\n\n"
                    
                    "**2. Rules for Constructing the Reasoning Path:**\n"
                    "*   **Quality over Quantity:** Present **up to five** distinct contradictory paths. It is crucial to only generate paths that are strongly supported by the evidence and meet all rules. **It is better to provide one or two high-quality contradictory paths—or even none—than to force the creation of five weak or invalid ones.**\n"
                    "*   **Rational Inference:** The reasoning path for a contradictory answer must be logical and well-supported. Do not force connections or invent paths just to find a contradiction.\n"
                    "*   **Path Uniqueness:** Each reasoning path should be distinct and not be a minor variation of another.\n"
                    "*   **Path Initiation:** The first piece of evidence in the path must contain at least one of the provided `Keywords`.\n"
                    "*   **Path Linking:** To connect consecutive steps within a path, they must share a common entity. This shared entity must be an **exact, case-sensitive match**. The position of the shared entity (head or tail) does not matter.\n\n"
            
                    "### Scoring Criteria:\n"
                    "For each piece of evidence in your path, provide a score in a `Contribution[Contribution]` format, measuring its contribution to your **newly derived contradictory answer**.\n"
                    "**Contribution to the Answer (Score 1 or 3):**\n"
                    "*   **Score 3 (High):** The evidence's tail entity (the object of the triple) is the exact `New Contradictory Answer` for that path.\n"
                    "*   **Score 1 (Low):** The evidence is a necessary link in the chain but does not contain the `New Contradictory Answer`.\n\n"
            
                    "### Output Format:\n"
                    "Present your entire output as a single block of text, with each valid path on a new line. Follow this format exactly for each path:\n"
                    "Path-based Evidence number('keyword/entity'->'relation'->'entity/keyword')Contribution[Contribution]->Path-based Evidence number('entity'->'relation'->'entity')Contribution[Contribution]->Result('Your New Contradictory Answer').\n\n"
                    "**If no strictly contradictory paths can be constructed following all the rules above, your entire output must be the single phrase:**\n"
                    "No contradictory reasoning path found.\n\n"
            
                    "--- SAMPLE START ---\n"
                    "### Context Provided:\n"
                    "User Question: What is the primary treatment for severe laryngitis?\n"
                    "Retrieved Keywords:\n"
                    "  Keyword 0: severe laryngitis\n"
                    "Retrieved Evidence:\n"
                    "  Path-based Evidence 1: ('severe laryngitis'->'is a type of'->'laryngitis')\n"
                    "  Path-based Evidence 2: ('laryngitis'->'primary treatment'->'voice rest')\n"
                    "  Path-based Evidence 3: ('laryngitis'->'can be treated with'->'corticosteroids')\n"
                    "  Path-based Evidence 4: ('severe laryngitis'->'may require'->'antibiotics')\n"
                    "Previous Final Answer: voice rest\n\n"
                    "### Expected Contradictory Output:\n"
                    "Path-based Evidence 1('severe laryngitis'->'is a type of'->'laryngitis')Contribution[1]->Path-based Evidence 3('laryngitis'->'can be treated with'->'corticosteroids')Contribution[3]->Result('corticosteroids').\n"
                    "Path-based Evidence 4('severe laryngitis'->'may require'->'antibiotics')Contribution[3]->Result('antibiotics').\n"
                    "--- SAMPLE END ---"
                )
            }
        ]

        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        print(f"-----prompt----")
        print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        print(f"-----assistant-----\n{history[2]['content']}")
        print(f"-----user2-----\n{history[3]['content']}")

        # assert False
        # output =  self._llm.chat_with_ai("", history)
        output = ''
        think_content = ''
        retry = 1
        while retry <= 3:
            try:
                output, think_content =  self._llm.chat_with_ai("think_return", history)
                # print(f"feedback output {retry}: {output}")
                if 'No contradictory reasoning path found.' in output:
                    return output, [], [], ['No contradictory reasoning path found.'], ''
                line_list =  output.strip().split('\n')
                line_list = [ line  for line in line_list if line]
                path_evidence_numbers_2d = []
                contribution_numbers_2d = []
                final_result_list = []
                for line in line_list:
                    path_evidence_numbers, contribution_numbers, final_result = self.parse(line)
                    if len(path_evidence_numbers) == len(contribution_numbers) and final_result:
                        path_evidence_numbers_2d.append(path_evidence_numbers)
                        contribution_numbers_2d.append(contribution_numbers)
                        final_result_list.append(final_result)
                if path_evidence_numbers_2d:
                    return output, path_evidence_numbers_2d, contribution_numbers_2d, final_result_list, think_content
                else:
                    raise Exception(f"----parse error----")
            except Exception as e:
                print(f"feedback_Reasoning_Path_error_test_api output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return output, [], [], [], ''

    # 旧的方法使用api，qwen3
    def chat_without_stream_for_socre_feedback_latest_v4_qwen_api(self, user_question: str, final_answer: str, evidence_list = [], flag_TF = True): # 问题+证据
        retrieved_knowledge = ""
        for index, evidence in enumerate(evidence_list):
            if evidence[3] == '->':
                retrieved_knowledge += f"  {index}: {evidence[0]} {evidence[1].replace('_',' ')} {evidence[2]}\n"
            elif evidence[3] == '<-':
                retrieved_knowledge += f"  {index}: {evidence[2]} {evidence[1].replace('_',' ')} {evidence[0]}\n"
        if flag_TF:
            flag_str = "Correct"
        else:
            flag_str = "Incorrect"
        prompt = knowledge_graph_score_feedback_prompt_qwen3.format(question = user_question, flag_TF = flag_str, last_response = final_answer, knowledge_statement_sets = retrieved_knowledge)

        history = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant tasked with evaluating the factual relevance and correctness of retrieved sets of inferential statements (each statement in natural language form) in relation to a given query and the model's previous response (Last Response). This evaluation occurs in the second step of a process, focusing on sets of statements used or relevant to the Last Response. Your task is to assign scores only to sets of statements that, when their information is combined and considered as a whole, are directly relevant to the query's specific factual question, ignoring irrelevant ones."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # print(f"chat_without_stream_for_socre_feedback_Reasoning_Path_api prompt\n{json.dumps(history, indent=2)}")
        # print(f"-----prompt----")
        # print(f"-----system-----\n{history[0]['content']}")
        print(f"-----user1-----\n{history[1]['content']}")
        # print(f"-----assistant-----\n{history[2]['content']}")
        # print(f"-----user2-----\n{history[3]['content']}")

        # assert False
        # output =  self._llm.chat_with_ai("", history)
        output = ''
        # think_content = ''
        retry = 1
        while retry <= 3:
            try:
                # print("111")
                output =  self._llm.chat_with_ai("", history)
                # print("222")

                # print(f"feedback output {retry}: {output}")
                line_list =  output.strip().split('\n')
                correct_numbers = {}
                error_numbers = {}
                insufficient_str = ''
                for line in line_list:
                    # if line.strip() == 'Insufficient search information':
                    if 'Insufficient information' in line.strip():
                        insufficient_str = 'Insufficient information'
                        continue
                        # break
                    elif "No feedback" in line.strip():
                        insufficient_str = "No feedback"
                        continue
                        # break
                    elif line.strip().startswith("Correct: "):
                        correct_pairs = line.strip().split("Correct:")[1].strip().split()
                        for pair in correct_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    correct_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                    elif line.strip().startswith('Error: '):
                        error_pairs = line.strip().split("Error:")[1].strip().split()
                        for pair in error_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    error_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                assert (correct_numbers or error_numbers or insufficient_str)
                return insufficient_str, correct_numbers, error_numbers
            except Exception as e:
                print(f"feedback output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return '', {}, {}

    
    # 使用ollama，无多轮对话，与chat_without_stream_for_socre_feedback_latest_v4一模一样
    # +++
    def chat_without_stream_for_socre_feedback_basic(self, message: str, response: str, filtered_retrieve_result= [], flag_TF = 0):
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context = context + f"{idx}. {sentence}\n"
        if flag_TF:
            flag_str = "Correct"
        else:
            flag_str = "Incorrect"
        prompt_system = score_feedback_prompt_baisc_system
        prompt_user = score_feedback_prompt_baisc_user.format(question = message, flag_TF = flag_str, last_response = response, knowledge_statement_sets = context)
        output = None
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                output = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                # print(f"feedback output {retry}:\n prompt:\n{prompt_user}\n output:\n {output}")
                line_list =  output.strip().split('\n')
                correct_numbers = {}
                error_numbers = {}
                insufficient_str = ''
                for line in line_list:
                    # if line.strip() == 'Insufficient search information':
                    if 'Insufficient information' in line.strip():
                        insufficient_str = 'Insufficient information'
                        continue
                        # break
                    elif "No feedback" in line.strip():
                        insufficient_str = "No feedback"
                        continue
                        # break
                    elif line.strip().startswith("Correct: "):
                        correct_pairs = line.strip().split("Correct:")[1].strip().split()
                        for pair in correct_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    correct_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                    elif line.strip().startswith('Error: '):
                        error_pairs = line.strip().split("Error:")[1].strip().split()
                        for pair in error_pairs:
                            parts = pair.split(":")
                            if len(parts) == 2:
                                try:
                                    number, degree = map(int, parts)
                                    error_numbers[int(number)] = int(degree)
                                except ValueError as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                except Exception as e:
                                    print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                            else:
                                print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                assert (correct_numbers or error_numbers or insufficient_str)
                return insufficient_str, correct_numbers, error_numbers
            except Exception as e:
                print(f"feedback output parse error: {e}\nllm response: {output}")
                retry += 1       
        
        return '', {}, {}
    
    # +++
    def chat_without_stream_for_socre_feedback_standard(self, message: str, response: str, filtered_retrieve_result= [], flag_TF = 0):
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context += f"Path {idx}:\t{sentence}\n"
        if flag_TF:
            flag_str = "Correct answer"
        else:
            flag_str = "Incorrect answer"
        prompt_system = score_feedback_prompt_standard_system
        prompt_user = score_feedback_prompt_standard_user.format(question = message, last_response = response, knowledge_paths = context) # flag_TF = flag_str
        # history = [
        #     {
        #         "role": "system",
        #         "content": score_feedback_prompt_standard_system_mulRounds,
        #     },
        #     {
        #         "role": "user",
        #         "content": score_feedback_prompt_standard_user1_mulRounds.format(question = message)
        #     },
        #     {
        #         "role": "assistant",
        #         "content": score_feedback_prompt_standard_assistant_mulRounds.format(last_response = response, knowledge_paths = context)
        #     },
        #     {
        #         "role": "user",
        #         "content": score_feedback_prompt_standard_user2_mulRounds,
        #     }
        # ]
        # print(f"-----prompt----")
        # print(f"-----system-----\n{history[0]['content']}")
        # print(f"-----user1-----\n{history[1]['content']}")
        # print(f"-----assistant-----\n{history[2]['content']}")
        # print(f"-----user2-----\n{history[3]['content']}")

        print(f"-----prompt----\n{prompt_system}\n{prompt_user}")

        output = None
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                output = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                # output = self._llm.chat_with_ai_mulRounds(history = history)
                # print(f"feedback output {retry}\noutput: {output}")
                res_parsed = json.loads(output)
                reasoning_path = res_parsed["Reasoning_path"]
                insufficient = res_parsed["Insufficient_information"]
                path_score = res_parsed["Path_score"]
                return res_parsed
            except Exception as e:
                print(f"feedback_standard output parse error: {e}\nllm response: {output}")
                retry += 1       
        return {}
    
    # +++
    def chat_without_stream_for_socre_feedback_standard_shared_prefix(self, message: str, response: str, filtered_retrieve_result= [], flag_TF = 0):
        context = ""
        for idx, sentence in enumerate(filtered_retrieve_result, start=0):
            context += f"Path {idx}:\t{sentence}\n"
        if flag_TF:
            flag_str = "Correct answer"
        else:
            flag_str = "Incorrect answer"
        prompt_system = shared_prefix.format(knowledge_sequences = context)
        prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question = message, last_response = response) # flag_TF = flag_str
        
        # prompt_system = score_feedback_prompt_standard_system_test.format(knowledge_sequences = context)
        # prompt_user = score_feedback_prompt_standard_user_test.format(question = message, last_response = response)
        
        # prompt_system = score_feedback_prompt_standard_system_test_1.format(knowledge_paths = context)
        # prompt_user = score_feedback_prompt_standard_user_test_1.format(question = message, last_response = response)
        
        
        print(f"-----prompt----\n{prompt_system}\n{prompt_user}")

        output = None
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                # output = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                output, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                # output = self._llm.chat_with_ai_mulRounds(history = history)
                res_parsed = json.loads(output)
                print(f"feedback output {retry}\noutput: {json.dumps(res_parsed, indent=2)}")
                # reasoning_path = res_parsed["Reasoning_path"]
                insufficient = res_parsed["Insufficient_information"]
                path_score = res_parsed["Path_score"]
                return res_parsed, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time
            except Exception as e:
                print(f"feedback_standard_shared_prefix output parse error: {e}\nllm response: {output}")
                retry += 1       
        return {}

    # +++
    def chat_without_stream_for_socre_feedback_standard_shared_prefix_batch(self, query: List[str], response: List[str], filtered_retrieve_result_batch: List[str]):
        context_batch = []
        for filtered_retrieve_result in filtered_retrieve_result_batch:
            context = ""
            for idx, sentence in enumerate(filtered_retrieve_result, start=0):
                context += f"Path {idx}:\t{sentence}\n"
            context_batch.append(context)
        prompt_system_batch = []
        prompt_user_batch = []
        for i in range(len(query)):
            prompt_system = shared_prefix.format(knowledge_sequences = context_batch[i])
            prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question = query[i], last_response = response[i]) # flag_TF = flag_str
            prompt_system_batch.append(prompt_system)
            prompt_user_batch.append(prompt_user)
            # print(f"-----prompt----\n{prompt_system}\n{prompt_user}")

        output = []
        prompt_len = []
        response_batch = []
        generate_time = 0
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                response_batch, prompt_len, generate_time = self._llm.chat_with_ai_with_system(prompt_system_batch, prompt_user_batch)
                # output, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                for i in range(len(response_batch)):
                    try:
                        res_parsed = json.loads(response_batch[i].strip("`\n "))
                        insufficient = res_parsed.get("Insufficient_information")
                        path_score = res_parsed.get("Path_score")

                        # 如果字段缺失，视为解析失败
                        if insufficient is None or path_score is None:
                            raise ValueError("Missing required fields: Insufficient_information or Path_score")

                        # print(f"feedback output {retry}\noutput: {json.dumps(res_parsed, indent=2)}")
                    except Exception as e:
                        print(f"Parse error: {e} | Response: {response_batch[i]}")
                        res_parsed = {
                            "Insufficient_information": True,
                            "Path_score": {},
                        }
                    output.append(res_parsed)

                return output, prompt_len, generate_time
            except Exception as e:
                print(f"feedback_standard_shared_prefix_batch output chat error: {e}\nllm response: {response_batch}")
                retry += 1       
        return [], [], 0

    # +++
    def chat_without_stream_for_socre_feedback_basic_shared_prefix_batch(self, query: List[str], response: List[str], triplet_unique_batch: List[str]):
        context_batch = []
        for triplet_unique in triplet_unique_batch:
            context = ""
            for idx, (h, r, t) in enumerate(triplet_unique, start=0):
                context += f"Path {idx}:\t{h} {r.replace('_',' ')} {t}\n"
            context_batch.append(context)
        prompt_system_batch = []
        prompt_user_batch = []
        for i in range(len(query)):
            prompt_system = shared_prefix.format(knowledge_sequences = context_batch[i])
            # prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question = query[i], last_response = response[i])
            prompt_user = score_feedback_prompt_standard_user_shared_prefix_optimized.format(question = query[i], last_response = response[i])
            prompt_system_batch.append(prompt_system)
            prompt_user_batch.append(prompt_user)
            # print(f"-----prompt----\n{prompt_system}\n{prompt_user}")

        output = []
        prompt_len = []
        response_batch = []
        generate_time = 0
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                response_batch, prompt_len, generate_time = self._llm.chat_with_ai_with_system(prompt_system_batch, prompt_user_batch)
                # output, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                for i in range(len(response_batch)):
                    try:

                        res_parsed = json.loads(response_batch[i].strip("`\n "))
                        insufficient = res_parsed.get("Insufficient_information")
                        path_score = res_parsed.get("Path_score")

                        # 如果字段缺失，视为解析失败
                        if insufficient is None or path_score is None:
                            raise ValueError("Missing required fields: Insufficient_information or Path_score")

                        # print(f"feedback output {retry}\noutput: {json.dumps(res_parsed, indent=2)}")
                    except Exception as e:
                        print(f"Parse error: {e} | Response: {response_batch[i]}")
                        res_parsed = {
                            "Insufficient_information": True,
                            "Path_score": {},
                        }
                    output.append(res_parsed)

                return output, prompt_len, generate_time
            except Exception as e:
                print(f"feedback_standard_shared_prefix_batch output chat error: {e}\nllm response: {response_batch}")
                retry += 1       
        return [], [], 0
    
    # +++
    def chat_without_stream_for_leaderboardStrategy_feedback_shared_prefix_batch(self, query: List[str], response: List[str], triplet_unique_batch: List[str], TF_list: List[int]):
        context_batch = []
        for triplet_unique in triplet_unique_batch:
            context = ""
            for idx, (h, r, t) in enumerate(triplet_unique, start=0):
                context += f"Path {idx}:\t{h} {r.replace('_',' ')} {t}\n"
            context_batch.append(context)
        prompt_system_batch = []
        prompt_user_batch = []
        for i in range(len(query)):
            prompt_system = shared_prefix.format(knowledge_sequences = context_batch[i])
            # prompt_user = score_feedback_prompt_standard_user_shared_prefix.format(question = query[i], last_response = response[i])
            prompt_user = score_feedback_prompt_leaderboardStrategy_user_shared_prefix.format(question = query[i], last_response = response[i], flag_TF = "Correct" if TF_list[i] else "Incorrect")
            prompt_system_batch.append(prompt_system)
            prompt_user_batch.append(prompt_user)
            # print(f"-----prompt----\n{prompt_system}\n{prompt_user}")

        output = []
        prompt_len = []
        response_batch = []
        generate_time = 0
        retry = 1
        while retry <= 3:
            try:
                # output = self._llm.chat_with_ai(prompt)
                response_batch, prompt_len, generate_time = self._llm.chat_with_ai_with_system(prompt_system_batch, prompt_user_batch)
                # output, prompt_len, generate_len, end2end_time, generate_time, prefill_time, decode_time, wait_scheduled_time = self._llm.chat_with_ai_with_system(prompt_system, prompt_user)
                for i in range(len(response_batch)):
                    print(f"chat_without_stream_for_leaderboardStrategy_feedback_shared_prefix_batch {i} feedback response {response_batch[i]}")
                    try:
                        res_parsed = {
                            "Insufficient_information": False,
                            'No_feedback': False,
                            "correct_numbers": {},
                            "error_numbers": {},
                        }
                        res_parsed_str = response_batch[i].strip("`\n ").split('\n')

                        correct_numbers = {}
                        error_numbers = {}
                        insufficient_str = ''
                        for line in res_parsed_str:
                            # if line.strip() == 'Insufficient search information':
                            if 'Insufficient information' in line.strip():
                                res_parsed['Insufficient_information'] = True
                                # continue
                                break
                            elif "No feedback" in line.strip():
                                res_parsed['No_feedback'] = True
                                # continue
                                break
                            elif line.strip().startswith("Correct: "):
                                correct_pairs = line.strip().split("Correct:")[1].strip().split()
                                for pair in correct_pairs:
                                    parts = pair.split(":")
                                    if len(parts) == 2:
                                        try:
                                            number, degree = map(int, parts)
                                            # correct_numbers[int(number)] = int(degree)
                                            res_parsed["correct_numbers"][int(number)] = int(degree)
                                        except ValueError as e:
                                            print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                        except Exception as e:
                                            print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                                    else:
                                        print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                            elif line.strip().startswith('Error: '):
                                error_pairs = line.strip().split("Error:")[1].strip().split()
                                for pair in error_pairs:
                                    parts = pair.split(":")
                                    if len(parts) == 2:
                                        try:
                                            number, degree = map(int, parts)
                                            # error_numbers[int(number)] = int(degree)
                                            res_parsed["error_numbers"][int(number)] = int(degree)
                                        except ValueError as e:
                                            print(f"------------ There is a degree score error in the response; please check and revise it once. ----------ValueError: {e}-----")
                                        except Exception as e:
                                            print(f"------------ There is a degree score error in the response; please check and revise it once. ----------Exception: {e}-----")
                                    else:
                                        print(f"------------ There is a degree score error in the response; please check and revise it once. ---------------")
                        # assert (correct_numbers or error_numbers or insufficient_str)
                        # return insufficient_str, correct_numbers, error_numbers
                    except Exception as e:
                        print(f"Parse error: {e} | Response: {response_batch[i]}")
                        res_parsed = {
                            "Insufficient_information": True,
                            'No_feedback': True,
                            "correct_numbers": {},
                            "error_numbers": {},
                        }
                    output.append(res_parsed)

                return output, prompt_len, generate_time
            except Exception as e:
                print(f"feedback_standard_shared_prefix_batch output chat error: {e}\nllm response: {response_batch}")
                retry += 1       
        return [], [], 0
        
    def chat_without_stream_for_socre_feedback_multi_round_dialogue(self, message: str, history : List[Dict[str, str]], response : str, retrieve_result = [], flag_TF = True):
        # retrieve_str = ""
        # for idx, tmp in enumerate(retrieve_result, start=1):
        #     retrieve_str = retrieve_str + "\n" + str(idx) + ": " + tmp 
        # if flag_TF:
        #     prompt = knowledge_graph_score_correction_prompt_true.format(query = message, response = response, retrieve_result = retrieve_str)
        # else:
        #     prompt = knowledge_graph_score_correction_prompt_false.format(query = message, response = response, retrieve_result = retrieve_str)
        answers = self._llm.chat_with_ai_multi_round(message, history)
        
        return answers
    
    def chat_without_stream_for_socre_feedback_multi_round_dialogue_test(self, message: str, history : List[Dict[str, str]], response = "", retrieve_result = [], flag_TF = True):
        # retrieve_str = ""
        # for idx, tmp in enumerate(retrieve_result, start=1):
        #     retrieve_str = retrieve_str + "\n" + str(idx) + ": " + tmp 
        # if flag_TF:
        #     prompt = knowledge_graph_score_correction_prompt_true.format(query = message, response = response, retrieve_result = retrieve_str)
        # else:
        #     prompt = knowledge_graph_score_correction_prompt_false.format(query = message, response = response, retrieve_result = retrieve_str)
        answers = self._llm.chat_with_ai_multi_round(message, history)
        
        return answers

    def chat_without_stream_for_redundant_relationship(self, redundant_relationship = []):
        redundant_relationship_list = [item for sublist in redundant_relationship for item in sublist]
        redundant_relationship_str = ""
        idy = 1
        for idx, group in enumerate(redundant_relationship, start=1):
            redundant_relationship_str = redundant_relationship_str + f"Group {idx}: Short statements\n"
            for sentence in group:
                redundant_relationship_str = redundant_relationship_str + str(idy) + ": " + sentence + "\n"
                idy += 1
            redundant_relationship_str = redundant_relationship_str + "\n"
        prompt = knowledge_graph_redundant_relationship_prompt.format(redundant_relationship = redundant_relationship_str)
        # print(f"redundant_relationship_prompt\n{prompt}")
        answers = self._llm.chat_with_ai(prompt)
        return answers 
 
    def chat_without_stream_for_redundant_relationship_v2(self, redundant_relationship_3d = []):
        redundant_relationship_str = ""
        idy = 0
        for idx, group in enumerate(redundant_relationship_3d, start=0):
            redundant_relationship_str = redundant_relationship_str + f"Group {idx}: Short statements\n"
            for triple in group:
                redundant_relationship_str = redundant_relationship_str + str(idy) + ": " + triple[0] +' '+ triple[1].replace("_"," ") +' '+ triple[2] + "\n"
                idy += 1
            redundant_relationship_str = redundant_relationship_str + "\n"
        prompt = knowledge_graph_redundant_relationship_prompt.format(redundant_relationship = redundant_relationship_str)
        # print(f"redundant_relationship_prompt\n{prompt}")
        
        
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"output {retry}: \n{output}")
                relationship_list =  output.strip().split('\n')
                keep_id_list = []
                for ids in relationship_list:
                    numeric_ids = [
                        id_str.strip() 
                        for id_str in ids.strip().split(",") 
                        if id_str.strip().isdigit()
                    ]
                    if numeric_ids:
                        keep_id_list.append(numeric_ids)
                assert len(keep_id_list) == len(redundant_relationship_3d)
                return keep_id_list
            except Exception as e:
                print(f"{retry} redundant relationship output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
    
    def chat_without_stream_for_redundant_relationship_item(self, redundant_relationship_2d = []):
        redundant_relationship_str = ""
        for idx, triple in enumerate(redundant_relationship_2d, start=0):
            redundant_relationship_str = redundant_relationship_str + str(idx) + ". " + triple[0] +' '+ triple[1].replace("_"," ") +' '+ triple[2] + "\n"
        # prompt = knowledge_graph_redundant_relationship_prompt_v3.format(redundant_relationship = redundant_relationship_str)
        # prompt = knowledge_graph_redundant_relationship_prompt_qwen_instruct_v2.format(redundant_relationship = redundant_relationship_str)
        prompt = knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3.format(redundant_relationship = redundant_relationship_str)

        # print(f"redundant_relationship_prompt\n{prompt}")
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"output_for_redundant_relationship {retry}: \n{output}")

                # if output == "No results matching your criteria.":
                #     return []
                numeric_ids = [
                    int(id_str.strip())
                    for id_str in output.strip().split(",") 
                    if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_relationship_2d)
                ]

                assert numeric_ids and len(numeric_ids) <= len(redundant_relationship_2d)
                return numeric_ids
            except Exception as e:
                print(f"{retry} redundant relationship output parse error: {e}\nllm response: {output}")
                retry += 1
        return []

    def chat_without_stream_for_redundant_relationship_v3(self, redundant_relationship_3d = []):
        idy = 0
        res_list_2d = []
        for idx, group in enumerate(redundant_relationship_3d, start=0):
            res = self.chat_without_stream_for_redundant_relationship_item(redundant_relationship_2d = group)
            if res:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d
    
    def chat_without_stream_for_redundant_relationship_item_api(self, redundant_relationship_2d = []):
        redundant_relationship_str = ""
        for idx, triple in enumerate(redundant_relationship_2d, start=0):
            redundant_relationship_str = redundant_relationship_str + str(idx) + ". " + triple[0] +' '+ triple[1].replace("_"," ") +' '+ triple[2] + "\n"
        # prompt = knowledge_graph_redundant_relationship_prompt_v3.format(redundant_relationship = redundant_relationship_str)
        # prompt = knowledge_graph_redundant_relationship_prompt_qwen_instruct_v2.format(redundant_relationship = redundant_relationship_str)
        # prompt = knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3.format(redundant_relationship = redundant_relationship_str)
        messages=[{'role': 'system', 'content': knowledge_graph_redundant_relationship_prompt_qwen_v3_system_api},
                            {'role': 'user', 'content': knowledge_graph_redundant_relationship_prompt_qwen_v3_user_api.format(redundant_relationship = redundant_relationship_str)}]

        # print(f"redundant_relationship_prompt\n{prompt}")
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt="0", history=messages)
                # print(f"output_for_redundant_relationship {retry}: \n{output}")

                # if output == "No results matching your criteria.":
                #     return []
                numeric_ids = [
                    int(id_str.strip())
                    for id_str in output.strip().split(",") 
                    if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_relationship_2d)
                ]

                assert numeric_ids and len(numeric_ids) <= len(redundant_relationship_2d)
                return numeric_ids
            except Exception as e:
                print(f"{retry} redundant relationship output parse error: {e}\nllm response: {output}")
                retry += 1
        return []



    def chat_without_stream_for_redundant_relationship_v3_api(self, redundant_relationship_3d = []):
        idy = 0
        res_list_2d = []
        for idx, group in enumerate(redundant_relationship_3d, start=0):
            res = self.chat_without_stream_for_redundant_relationship_item_api(redundant_relationship_2d = group)
            if res:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d

    def chat_without_stream_for_redundant_relationship_item_api_qwen(self, redundant_relationship_2d = [], enable_thinking = False):
        redundant_relationship = ""
        for idx, triple in enumerate(redundant_relationship_2d, start=0):
            redundant_relationship = redundant_relationship + str(idx) + ". " + triple[0] +' '+ triple[1].replace("_"," ") +' '+ triple[2] + ".\n"
        history_bak = [
            # {
            #     "role": "user",
            #     "content": (
            #         "Your task is to analyze a list of statements and identify a minimal set that captures all unique information. Deduplicate the list based on the following rules, focusing on semantic meaning and factual details.\n\n"
                    
            #         "Core Rules:\n"
            #         "1. Semantic Equivalence: If multiple statements express the same core fact using different wording (e.g., synonyms, rephrasing, active/passive voice), keep only **one** instance.\n"
            #         "  * Exception: If statements provide conflicting details for the same attribute (e.g., 'The price is $10' vs. 'The price is $12'), keep **both**.\n"
            #         "2. Information Specificity (Subsumption): If one statement's information is fully contained within a more detailed one, keep only the **more specific** statement.\n"
            #         "  * Date Granularity: This is crucial for dates. A statement with a more specific date (e.g., 'January 5th, 2023') replaces a less specific one (e.g., 'January 2023' or '2023') for the **exact same event**.\n"
            #         "  * Factual Detail: A more descriptive fact (e.g., 'Team A won against Team B') replaces a more generic one (e.g., 'Team A played against Team B').\n"
            #         "3. Distinct Facts: Treat statements as unique and keep **all** of them if they differ in any of the following ways:\n"
            #         "  * They contain different numerical values (quantities, scores, amounts).\n"
            #         "  * They describe different events, even if related (e.g., 'joined the company' vs. 'was promoted at the company').\n"
            #         "  * They refer to different dates or times for different events (e.g., 'He arrived on Monday' vs. 'She arrived on Tuesday').\n\n"
                    
            #         "Examples:\n"
            #         "* Rule 1 (Equivalence):\n"
            #         "  * ['2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.'] -> Semantically the same. Keep one.\n"
            #         "* Rule 2 (Specificity - Factual):\n"
            #         "  * ['Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.'] -> 'Won against' is more specific than 'Played against'. Keep the first.\n"
            #         "* Rule 2 (Specificity - Date):\n"
            #         "  * ['The event happened in 2023.', 'The event happened on Jan 5th, 2023.'] -> The second is more specific. Keep the second.\n"
            #         "* Rule 3 (Distinct Facts - Numbers):\n"
            #         "  * ['Team X scored 10 points.', 'Team X scored 11 points.'] -> Different numerical values. Keep both.\n"
            #         "* Rule 3 (Distinct Facts - Dates):\n"
            #         "  * ['She arrived on Monday.', 'He arrived on Tuesday.'] -> Different events on different dates. Keep both.\n\n"
                    
            #         "Output Format:\n"
            #         "* Provide only the zero-based indices of the retained statements, separated by commas.\n"
            #         "* Do not include any headers, explanations, or other text.\n\n"
                    
            #         "Example Output:\n"
            #         "1, 3\n\n"
                    
            #         "Statements to Process:\n"
            #         f"{redundant_relationship}\n\n"
                    
            #         "Output:"
            #     )
            # }
        ]
        
        # api冗余处理用的这个
        history_use = [
            # {
            #     "role": "user",
            #     "content": (
            #         "Your *primary and most crucial objective* is to synthesize a list of statements, ensuring you *capture the complete set of unique information while simultaneously eliminating all possible redundancy*. You must analyze the provided list and produce a minimal, definitive set of facts. This means every unique piece of information must be represented, but with the fewest statements possible.\n"
            #         "To achieve this, apply the following principles and rules, focusing on semantic meaning and factual accuracy.\n\n"
                    
            #         "### Guiding Principle\n"
            #         "* One Fact, One Statement: The primary goal is to represent each unique piece of information with a single statement. For any given subject or event, strive to retain only the most comprehensive sentence.\n"
            #         "* Preserve All Unique Information: If multiple statements present genuinely different facts (e.g., separate events, different numerical values) or offer conflicting details about the same fact, they must *all* be retained. Do not discard a statement if it contains unique information not present elsewhere.\n"
                    
            #         "### Core Deduplication Rules\n"
            #         "1. Semantic Equivalence (Same Meaning):\n"
            #         "* If multiple statements express the exact same fact using different wording (e.g., synonyms, rephrasing, active/passive voice), keep only *one* representative instance.\n"
            #         "* Conflict Exception: If statements provide conflicting details for the same attribute (e.g., 'The price is $10' vs. 'The price is $12'), keep *both* to highlight the discrepancy.\n"
            #         "2. Information Specificity (General vs. Detailed):\n"
            #         "* If one statement's information is fully contained within another, more detailed statement describing the exact same event, keep only the *more specific* statement.\n"
            #         "* Factual Detail: A more descriptive fact (e.g., 'Team A won against Team B') replaces a more generic one (e.g., 'Team A played against Team B').\n"
            #         "* Date Granularity: A statement with a more precise date (e.g., 'January 5th, 2023') replaces a less specific one (e.g., 'January 2023' or '2023') for the same event.\n"
            #         "3. Distinct Facts (Different Information):\n"
            #         "* Keep all statements that describe fundamentally different information. Treat statements as unique and retain all of them if they:\n"
            #         "  * Contain different numerical values (quantities, scores, amounts).\n"
            #         "  * Describe different events, even if they are related (e.g., 'joined the company' vs. 'was promoted at the company').\n"
            #         "  * Refer to different dates or times for separate events (e.g., 'He arrived on Monday' vs. 'She arrived on Tuesday').\n\n"
                    
            #         "### Examples:\n"
            #         "* Rule 1 (Equivalence):['2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.'] -> Semantically the same. Keep one.\n"
            #         "* Rule 1 (Conflict Exception):['The project's budget is $5,000.', 'The project's budget is $5,500.'] -> Conflicting numerical values for the same attribute. Keep both.\n"
            #         "* Rule 2 (Specificity - Factual):['Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.'] -> 'Won against' is more specific than 'Played against'. Keep the first.\n"
            #         "* Rule 2 (Specificity - Date):['The event happened in 2023.', 'The event happened on Jan 5th, 2023.'] -> The second date is more specific. Keep the second.\n"
            #         "* Rule 3 (Distinct Facts - Events):['She joined the team in May.', 'She was promoted to team lead in August.'] -> Different events. Keep both.\n\n"
                    
            #         "### Output Format:\n"
            #         "* Provide only the zero-based indices of the statements to be retained.\n"
            #         "* The indices must be separated by commas.\n"
            #         "* Do not include any headers, explanations, or other text in your final output.\n\n"
                    
            #         "### Example Output:\n"
            #         "1,3\n\n"
                    
            #         "Statements to Process:\n"
            #         f"{redundant_relationship}\n\n"
                    
            #         "Output:"
            #     )
            # }
        ]

        # noise feedback 用的这个
        history = [
            {
                "role": "user",
                "content": (
                    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
                    "Identify and retain statements representing unique core meanings or facts from the following group. Your goal is to capture the complete set of distinct information with minimal redundancy, strictly adhering to the rules below, especially regarding dates and numerical values.\n\n"
                    "**Rules for Statement Retention:**\n"
                    "1.  **Similarity & Paraphrasing:** If multiple statements express essentially the same core meaning (e.g., using synonyms, different phrasing, active/passive voice, or swapped subject/object for the same underlying relationship), keep only **one** instance, preferably the clearest, most complete, or most standard representation.\n"
                    "    * If different descriptions of the same attribute of an event appear — even if they are conflicting — all such descriptions should be retained.\n"
                    "2.  **Information Subsumption & Date Specificity:**\n"
                    "    * If one statement's meaning is fully contained within another more specific or informative statement, retain only the **more informative** one.\n"
                    "    * If there is no subsumption, retain **both**.\n"
                    "    * **Crucially for dates:** If two statements refer to the **exact same event** but differ *only* in the level of date detail (e.g., 'January 1st, 2023' vs. 'January 2023' vs. '2023'), retain **only** the statement with the **most specific date**. A more detailed date *replaces* a less detailed date for the same fact.\n"
                    "3.  **Strict Handling of Numerical and Temporal Details (Numbers, Dates, Quantities):**\n"
                    "    * Treat statements as representing **distinct** facts and retain them **all** if they differ in any stated:\n"
                    "        * Numerical values (quantities, monetary amounts, scores, counts, etc.)\n"
                    "        * Specific dates or times (unless Rule 2 applies for granularity of the *same event*)\n"
                    "    * **Exception:** Redundancy based on numerical values (Rule 1) can *only* be considered if the numbers/quantities mentioned are **exactly identical** *and* the rest of the core meaning is also identical. Any difference, no matter how small, means the statements are distinct unless it's a date granularity issue covered by Rule 2.\n"
                    "4.  **Different Aspects of Related Facts:** If statements describe different facets or temporal aspects of a related situation, evaluate if both convey unique, valuable information. If so, **retain both**. If one aspect largely implies the other or is significantly more informative, retain the more valuable one(s).\n"
                    "    * For example, both 'joining a company' and 'being employed at the company' should be retained. For cases like 'located at' vs. 'held at', retain only one.\n"
                    "5.  **Clearly Different Meanings:** If statements express clearly distinct facts, events, or relationships unrelated to the rules above, **retain all** of them.\n\n"
                    "**Processing Steps:**\n"
                    "1.  Consider the list of statements provided.\n"
                    "2.  Apply the rules above meticulously to filter the statements.\n"
                    "3.  Prioritize strict interpretation of rules 2 and 3 regarding dates and numbers.\n"
                    "4.  Aim to represent the unique information from the original set accurately and concisely according to these rules.\n\n"
                    "**Output Format:**\n"
                    "* Print **only** the zero-based indices of the retained statements, separated by commas.\n"
                    "* Do not include headers, explanations, or any text other than the comma-separated indices.\n"
                    "**Example Output:**\n"
                    "1, 3\n\n"
                    "**Example Application Illustrating Rules (Conceptual):**\n"
                    "* Group A (Rule 3 - Different Numbers): 'Team X scored 10 points.', 'Team X scored 11 points.' -> Retain both indices.\n"
                    "* Group B (Rule 2 - Date Specificity): 'The event happened in 2023.', 'The event happened on Jan 5th, 2023.' -> Retain only the index for the second statement. Because the second sentence includes the information from the first.\n"
                    "* Group C (Rule 1/3 - Identical Numbers): 'The budget is $5,000.', 'The budget is $5,000.00.' -> Treat as identical numbers. If rest of the meaning is the same, keep one index.\n"
                    "* Group D (Rule 3 - Different Dates): 'She arrived on Monday.', 'He arrived on Tuesday.' -> Retain both indices.\n"
                    "* Group E (Rule 1 - Paraphrasing): '2023 citrus bowl Features teams Lsu tigers', 'Lsu tigers Will play in 2023 citrus bowl.' -> Semantically similar (different perspective). Keep one index (e.g., 0 or 1).\n"
                    "* Group F (Rule 2 - Subsumption/More Specific Info): 'Kentucky wildcats Won against Iowa hawkeyes', 'Iowa hawkeyes Played against Kentucky wildcats.' -> Statement 0 ('Won against') is more specific and implies 1 ('Played against'). Keep index 0.\n"
                    f"**Statements to Process:**\n{redundant_relationship}\n\n"
                    "**Output:**\n"
                )
            }
        ]
        
        # print(f"redundant_relationship_prompt\n{prompt}")
        prompt_think = ''
        if enable_thinking:
            prompt_think = "think_return"
        output = []
        think_content = ''
        retry = 1
        while retry <= 3:
            try:
                if enable_thinking:
                    output, think_content = self._llm.chat_with_ai(prompt = prompt_think, history=history)
                else:
                    output  = self._llm.chat_with_ai(prompt = prompt_think, history=history)
                    think_content = ''
                print(f"output_for_redundant_relationship {retry}: \n{output}")

                # if output == "No results matching your criteria.":
                #     return []
                numeric_ids = [
                    int(id_str.strip())
                    for id_str in output.strip().split(",") 
                    if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_relationship_2d)
                ]

                assert numeric_ids and len(numeric_ids) <= len(redundant_relationship_2d)
                return numeric_ids, think_content
            except Exception as e:
                print(f"{retry} redundant relationship output parse error: {e}\nllm response: {output}")
                retry += 1
        return [], ''
    
    def chat_without_stream_for_redundant_relationship_v3_api_qwen(self, redundant_relationship_3d = [], enable_thinking = False):
        idy = 0
        res_list_2d = []
        think_list = []
        for idx, group in enumerate(redundant_relationship_3d, start=0):
            res, think_content = self.chat_without_stream_for_redundant_relationship_item_api_qwen(redundant_relationship_2d = group, enable_thinking = enable_thinking)
            think_list.append(think_content)
            if res:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d, think_list
    
    def chat_without_stream_for_redundant_relationship_item_local(self, redundant_relationship_2d = [], enable_thinking = False):
        redundant_relationship = ""
        for idx, triple in enumerate(redundant_relationship_2d, start=0):
            redundant_relationship = redundant_relationship + str(idx) + ". " + triple[0] +' '+ triple[1].replace("_"," ") +' '+ triple[2] + ".\n"
        
        prompt = knowledge_graph_redundant_relationship_prompt_qwen_instruct_v3_local.format(redundant_relationship = redundant_relationship)
        
        # print(f"redundant_relationship_prompt\n{prompt}")
        # prompt_think = ''
        # if enable_thinking:
        #     prompt_think = "think_return"
        output = []
        # think_content = ''
        retry = 1
        while retry <= 3:
            try:
                output  = self._llm.chat_with_ai(prompt = prompt)
                print(f"output_for_redundant_relationship {retry}: \n{output}")

                # if output == "No results matching your criteria.":
                #     return []
                numeric_ids = [
                    int(id_str.strip())
                    for id_str in output.strip().split(",") 
                    if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_relationship_2d)
                ]

                assert numeric_ids and len(numeric_ids) <= len(redundant_relationship_2d)
                return numeric_ids
            except Exception as e:
                print(f"{retry} redundant relationship output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
    
    def chat_without_stream_for_redundant_relationship_v3_local(self, redundant_relationship_3d = [], enable_thinking = False):
        idy = 0
        res_list_2d = []
        think_list = []
        for idx, group in enumerate(redundant_relationship_3d, start=0):
            res = self.chat_without_stream_for_redundant_relationship_item_local(redundant_relationship_2d = group, enable_thinking = enable_thinking)
            # think_list.append(think_content)
            if res:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d
    
    # +++
    def chat_without_stream_for_redundant_relationship_strict_item(self, redundant_relationship_2d = []):
        redundant_relationship_str = ""
        for idx, triple in enumerate(redundant_relationship_2d, start=0):
            redundant_relationship_str = redundant_relationship_str + str(idx) + ". " + triple[0] +' '+ triple[1].replace("_"," ") +' '+ triple[2] + "\n"

        prompt_system = redundant_relationship_prompt_basic_system
        prompt_user = redundant_relationship_prompt_basic_user.format(redundant_relationship = redundant_relationship_str)

        # print(f"redundant_relationship_prompt\n{prompt}")
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai_with_system(prompt_system = prompt_system, prompt_user = prompt_user)
                # print(f"output_for_redundant_relationship {retry}: \n{output}")
                numeric_ids = [
                    int(id_str.strip())
                    for id_str in output.strip().split(",") 
                    if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_relationship_2d)
                ]

                assert numeric_ids and len(numeric_ids) <= len(redundant_relationship_2d)
                return numeric_ids
            except Exception as e:
                print(f"{retry} redundant relationship output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
    
    # +++
    def chat_without_stream_for_redundant_relationship_strict(self, redundant_relationship_3d = []):
        idy = 0
        res_list_2d = []
        for idx, group in enumerate(redundant_relationship_3d, start=0):
            res = self.chat_without_stream_for_redundant_relationship_strict_item(redundant_relationship_2d = group)
            if res:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d

    def chat_without_stream_for_redundant_entity(self, redundant_entity_2d = []):
        redundant_entity_str = ""
        idy = 0
        for idx, group in enumerate(redundant_entity_2d, start=0):
            redundant_entity_str += f"Group {idx}:\n"
            for entity in group:
                redundant_entity_str = redundant_entity_str + str(idy) + ". " + entity + "\n"
                idy += 1
            redundant_entity_str = redundant_entity_str + "\n"
        prompt = knowledge_graph_redundant_entity_prompt.format(redundant_entity = redundant_entity_str)
        # print(f"redundant_entity_prompt\n{prompt}")
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"output {retry}: \n{output}")
                entity_list =  output.strip().split('\n')
                keep_id_list = []
                for ids in entity_list:
                    numeric_ids = [
                        id_str.strip() 
                        for id_str in ids.strip().split(",") 
                        if id_str.strip().isdigit()
                    ]
                    if numeric_ids:
                        keep_id_list.append(numeric_ids)
                # assert len(keep_id_list) and (len(keep_id_list) <= len(redundant_entity_2d))
                assert keep_id_list
                return keep_id_list
            except Exception as e:
                print(f"{retry} redundant entity output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
        

    def chat_without_stream_for_redundant_entity_item(self, redundant_entity_list = []):
        redundant_entity_str = ""
        for idx, entity in enumerate(redundant_entity_list, start=0):
            redundant_entity_str += f"{idx}. {entity}\n"
        # prompt = knowledge_graph_redundant_entity_prompt_v2_qwen.format(redundant_entity = redundant_entity_str)
        prompt = knowledge_graph_redundant_entity_prompt_v3_qwen_en.format(redundant_entity = redundant_entity_str)
        # print(f"redundant_entity_prompt\n{prompt}")
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"{retry} redundant_entity_item: {output}")
                # print(f"output {retry}: \n{output}")
                # if output.strip() == "No results matching your criteria.":  # 这样一直检测不出来
                if "No results matching your criteria." in output.strip(): 
                    return []
                else:
                    # numeric_ids = [
                    #     int(id_str.strip())
                    #     for id_str in output.strip().split(",") 
                    #     if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_entity_list)
                    # ]
                    entity_list =  output.strip().split('\n')
                    keep_id_list = []
                    for ids in entity_list:
                        numeric_ids = [
                            int(id_str.strip()) 
                            for id_str in ids.strip().split(",") 
                            if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_entity_list)
                        ]
                        if numeric_ids:
                            keep_id_list.append(numeric_ids)
                # assert len(keep_id_list) and (len(keep_id_list) <= len(redundant_entity_2d))
                # assert numeric_ids and len(numeric_ids) <= len(redundant_entity_list)
                # return numeric_ids
                return keep_id_list
            except Exception as e:
                print(f"{retry} redundant entity output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
    
    def chat_without_stream_for_redundant_entity_item_v2(self, redundant_entity_list = [], sentences_of_entity_2d = []):
        redundant_entity_str = ""
        statement_id = 0
        for idx, (entity, sentences) in enumerate(zip(redundant_entity_list, sentences_of_entity_2d), start=0):
            redundant_entity_str += f"{idx}. {entity}\n"
            for idy, sentence in enumerate(sentences):
                # redundant_entity_str += f"{idx}.{idy}: {sentence}\n"
                redundant_entity_str += f"Triple statement {statement_id}: {sentence}\n"
                statement_id += 1

            redundant_entity_str += '\n'

        # prompt = knowledge_graph_redundant_entity_prompt_v2_qwen.format(redundant_entity = redundant_entity_str)
        # prompt = knowledge_graph_redundant_entity_prompt_v3_qwen_en.format(redundant_entity = redundant_entity_str)
        # prompt = knowledge_graph_redundant_entity_prompt_v4_qwen_en.format(redundant_entity = redundant_entity_str)
        prompt = knowledge_graph_redundant_entity_prompt_v5_qwen_en.format(redundant_entity = redundant_entity_str)
        # print(f"redundant_entity_prompt\n{prompt}")
        # assert False
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"{retry} redundant_entity_item: {output}")
                # assert False
                # print(f"output {retry}: \n{output}")
                # if output.strip() == "No results matching your criteria.":  # 这样一直检测不出来
                if "No results matching your criteria." in output.strip(): 
                    return []
                else:
                    # numeric_ids = [
                    #     int(id_str.strip())
                    #     for id_str in output.strip().split(",") 
                    #     if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_entity_list)
                    # ]
                    entity_list =  output.strip().split('\n')
                    keep_id_list = []
                    for ids in entity_list:
                        numeric_ids = [
                            int(id_str.strip()) 
                            for id_str in ids.strip().split(",") 
                            if id_str.strip().isdigit() and int(id_str.strip()) < len(redundant_entity_list)
                        ]
                        if numeric_ids:
                            keep_id_list.append(numeric_ids)
                # assert len(keep_id_list) and (len(keep_id_list) <= len(redundant_entity_2d))
                # assert numeric_ids and len(numeric_ids) <= len(redundant_entity_list)
                # return numeric_ids
                return keep_id_list
            except Exception as e:
                print(f"{retry} redundant entity output parse error: {e}\nllm response: {output}")
                retry += 1
        return []
    
    def chat_without_stream_for_redundant_entity_v2(self, redundant_entity_2d = []):
        idy = 0
        res_list_2d = []
        for idx, group in enumerate(redundant_entity_2d, start=0):
            res_all = self.chat_without_stream_for_redundant_entity_item(redundant_entity_list = group)
            for res in res_all:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d
    
    # 过滤实体的时候添加5个三元组作为辅助
    def chat_without_stream_for_redundant_entity_v3(self, redundant_entity_2d = [], sentences_of_entity_3d = []):
        idy = 0
        res_list_2d = []
        for idx, (group, group_2d) in enumerate(zip(redundant_entity_2d, sentences_of_entity_3d), start=0):
            res_all = self.chat_without_stream_for_redundant_entity_item_v2(redundant_entity_list = group, sentences_of_entity_2d = group_2d)
            for res in res_all:
                res = [
                    id + idy
                    for id in res
                ]
                res_list_2d.append(res)
            idy += len(group)
        return res_list_2d
    
    def chat_without_stream_for_redundant_entity_check(self, redundant_entity_2d = [], response_id = []): # 暂未启用
        redundant_entity_list = [item for sublist in redundant_entity_2d for item in sublist]
        redundant_entity_str = ""
        for idx, group in enumerate(response_id, start=0):
            redundant_entity_str += f"Group {idx}:\n"
            for id in group:
                redundant_entity_str = redundant_entity_str + redundant_entity_list[int(id)] + "\n"
            redundant_entity_str = redundant_entity_str + "\n"
        prompt = knowledge_graph_redundant_entity_check_prompt.format(redundant_entity = redundant_entity_str)
        # print(f"redundant_entity_prompt\n{prompt}")
                
        output = []
        retry = 1
        while retry <= 3:
            try:
                output = self._llm.chat_with_ai(prompt)
                # print(f"output {retry}: \n{output}")
                entity_list =  output.strip().split('\n')
                keep_id_list = []
                for ids in entity_list:
                    numeric_ids = [
                        id_str.strip() 
                        for id_str in ids.strip().split(",") 
                        if id_str.strip().isdigit()
                    ]
                    if numeric_ids:
                        keep_id_list.append(numeric_ids)
                # assert len(keep_id_list) and (len(keep_id_list) <= len(redundant_entity_2d))
                assert keep_id_list
                return keep_id_list
            except Exception as e:
                print(f"redundant entity output parse error: {e}\nllm response: {output}")
                retry += 1
        return []


    def chat_without_rag(self, message: str):
        
        prompt = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful, respectful, and honest assistant. <|eot_id|>
            <|start_header_id|>user<|end_header_id|> 
            Please help me answer the questions below.
            ---------------------
            {question}
            ---------------------
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """

        return self._llm.chat_with_ai(prompt.format(question=message))


    def chat_without_rag_qwen(self, message: str):
        
        prompt = """
            <|im_start|>system
            You are a helpful, respectful, and honest assistant. <|im_end|>
            <|im_start|>user
            Please help me answer the questions below.
            ---------------------
            {question}
            ---------------------
            <|im_end|>
            <|im_start|>assistant
            """

        return self._llm.chat_with_ai(prompt.format(question=message))





if __name__ == "__main__":
    from database.graph.nebulagraph.nebulagraph import NebulaClient
    from llmragenv.LLM.ollama.client import OllamaClient

    graph_db = NebulaClient()
    llm = OllamaClient()

    ChatGraphRAG = ChatGraphRAG(llm,graph_db)
    print(ChatGraphRAG.extract_keyword(question="Curry is a famous basketball player"))

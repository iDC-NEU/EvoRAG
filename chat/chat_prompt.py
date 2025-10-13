import json

chat_with_graphrag_for_response_system = (
    "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
    "{knowledge_sequences}"
)

chat_with_graphrag_for_response_user = (
    "Please write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}"
)

shared_prefix = (
    "You are an expert AI assistant specializing in knowledge graph question answering. "
    "Your response must be based *only* on the information contained within the provided knowledge sequences below. "
    "Follow the user's specific instructions for the task at hand.\n\n"
    "### Retrieved Knowledge Paths\n"
    "{knowledge_sequences}"
)

chat_with_graphrag_for_response_user_shared_prefix = (
    "### Task: Generate Answer\n"
    "Based on the provided knowledge paths, write a high-quality, concise answer for the following question. "
    "Use only the provided information. Answer directly without explanation.\n\n"
    "**Question**: {question}"
)

score_feedback_prompt_standard_user_shared_prefix = '''
### Task: Score Key Evidence Paths (Max 6)

You are an efficient Evidence Triage Agent. Your task is to swiftly identify and score a **maximum of 6** of the most relevant knowledge paths related to the **Question** and the **Knowledge-derived Answer**. Your primary goal is speed. **Only evaluate paths that are directly relevant (score 2, 1, or -2). Do not feel compelled to score 6 paths if fewer are truly relevant.**

Your output must be a **single, valid JSON object** and nothing else.

### Scoring Guide

Assign a score **only to the most critical paths**. Ignore all others.

*   **2 (Crucial Support)**: The path provides direct, critical evidence that confirms the answer.
*   **1 (Helpful Context)**: The path offers a necessary logical step or important background to understand the answer.
*   **-2 (Direct Contradiction)**: The path states a fact that directly and unambiguously contradicts the `Knowledge-derived Answer`.

### Output Format (Strictly Required)

Provide a JSON object. The `Path_score` dictionary must contain **a maximum of 6 entries**. It should only include paths that you have actively scored (i.e., those truly relevant to the answer).

{{
  "Insufficient_information": boolean, // True if no path provides meaningful support for the answer.
  "Path_score": {{                  // Key: Path ID, Value: integer (2, 1, or -2). At most 6 entries. Only include truly relevant paths.
    "Path X": integer
  }}
}}

### Example

**Input**:
Question: Who is on trial for fraud related to FTX?
Retrieved Knowledge Paths:
  - Path 1: (FTX) -[founded by]-> (Sam Bankman-Fried)
  - Path 2: (Sam Bankman-Fried) -[accused of]-> (Orchestrating financial fraud)
  - Path 3: (FTX) -[sponsored]-> (Mercedes-AMG Petronas F1 Team)
  - Path 4: (John Ray III) -[took over as CEO of]-> (FTX)
Knowledge-derived Answer: Sam Bankman-Fried

**Output**:

{{
  "Insufficient_information": false,
  "Path_score": {{
    "Path 1": 1,
    "Path 2": 2
  }}
}}

---

### Input for Scoring

**Question**: {question}
**Knowledge-derived Answer**: {last_response}

Quickly identify and score a **maximum of 6** key supporting or contradicting paths. Only score paths that are genuinely relevant. Ensure your output is a single, valid JSON object.
**Output**:
'''

score_feedback_prompt_standard_user_shared_prefix_optimized = '''
### Task: Score Key Evidence Paths (Max 6)

You are an efficient Evidence Triage Agent. Your task is to swiftly identify and score a **maximum of 6** of the most relevant knowledge paths related to the **Question** and the **Knowledge-derived Answer**. Your primary goal is speed. **Only evaluate paths that are directly relevant (score 2, 1, or -2). Do not feel compelled to score 6 paths if fewer are truly relevant.**

Your output must be a **single, valid JSON object** and nothing else.

### Scoring Guide

Your evaluation must be **exceptionally strict regarding numerical values, dates, quantities, and statistics**. These must match the `Knowledge-derived Answer` *exactly*.
For non-numerical facts (like names or relationships), focus on direct logical support.

Assign a score **only to the most critical paths**. Ignore all others.

* **2 (Crucial Support)**: The path provides direct, critical evidence that confirms the `Knowledge-derived Answer`.
    * **For numerical/date/quantity answers**: This path must contain the ***exact, literal*** number, date, or quantity mentioned in the answer. (e.g., `936,172` matches `936,172`).
    * **For entity/concept answers**: This path directly states the core fact (e.g., "LSU won") or provides an unambiguous logical equivalent (e.g., "Evo holds 100% stock" is crucial support for "Evo acquired").

* **1 (Helpful Context)**: The path offers a necessary logical step or important background to understand the answer, but isn't the single piece of crucial evidence.

* **-2 (Direct Contradiction)**: The path states a fact that directly and unambiguously contradicts the `Knowledge-derived Answer` (e.g., states a *different* exact number, or a different company/person).

**CRUCIAL RULE FOR NUMBERS**: If the answer is or contains a specific number (e.g., `936,172`), paths with *approximate* or *different* numbers (e.g., `936,000` or `930,000`) are **not** relevant support and must **not** be scored (neither positive nor negative). Only paths with the ***exact*** number are eligible for scoring.

### Output Format (Strictly Required)

Provide a JSON object. The `Path_score` dictionary must contain **a maximum of 6 entries**. It should only include paths that you have actively scored (i.e., those truly relevant to the answer).

{{
  "Insufficient_information": boolean, // True if no path provides meaningful support (score 1 or 2) for the answer.
  "Path_score": {{              // Key: Path ID, Value: integer (2, 1, or -2). At most 6 entries. Only include truly relevant paths.
    "Path X": integer
  }}
}}

### Example

**Input**:
Question: Who is on trial for fraud related to FTX?
Retrieved Knowledge Paths:
  - Path 1: (FTX) -[founded by]-> (Sam Bankman-Fried)
  - Path 2: (Sam Bankman-Fried) -[accused of]-> (Orchestrating financial fraud)
  - Path 3: (FTX) -[sponsored]-> (Mercedes-AMG Petronas F1 Team)
  - Path 4: (John Ray III) -[took over as CEO of]-> (FTX)
Knowledge-derived Answer: Sam Bankman-Fried

**Output**:

{{
  "Insufficient_information": false,
  "Path_score": {{
    "Path 1": 1,
    "Path 2": 2
  }}
}}

---

### Input for Scoring

**Question**: {question}
**Knowledge-derived Answer**: {last_response}

Quickly identify and score a **maximum of 6** key supporting or contradicting paths. Only score paths that are genuinely relevant. Ensure your output is a single, valid JSON object.
**Output**:
'''

score_feedback_prompt_standard_user_shared_prefix_bak = '''
### Task: Identify Key Evidence Paths

You are an efficient Evidence Triage Agent. Your task is to quickly identify which of the retrieved knowledge paths provide significant evidence related to the **Question** and the **Knowledge-derived Answer**. Your goal is speed and efficiency. **Do not evaluate every path.** Focus only on the most impactful ones.

Your output must be a **valid JSON object** and nothing else.

### Simplified Scoring Guide

Assign a score **only to the paths that are highly relevant**. Ignore all other paths.

*   **2 (Crucial Support)**: The path provides direct, critical evidence that confirms the answer.
*   **1 (Helpful Context)**: The path offers a necessary logical step or important background to understand the answer.
*   **-2 (Direct Contradiction)**: The path states a fact that directly and unambiguously contradicts the `Knowledge-derived Answer`.

### Output Format (strictly required)

Provide a JSON object with the following structure. The `Path_score` dictionary should **only include entries for the paths you have scored**.

{{
  "Insufficient_information": boolean, // True if no path provides meaningful support for the answer.
  "Path_score": {{                  // Key: Path ID, Value: integer (2, 1, or -2). Only include the most relevant paths.
    "Path X": integer
  }}
}}

### Example

**Input**:
Question: Who is on trial for fraud related to FTX?
Retrieved Knowledge Paths:
  - Path 1: (FTX) -[founded by]-> (Sam Bankman-Fried)
  - Path 2: (Sam Bankman-Fried) -[accused of]-> (Orchestrating financial fraud)
  - Path 3: (FTX) -[sponsored]-> (Mercedes-AMG Petronas F1 Team)
  - Path 4: (John Ray III) -[took over as CEO of]-> (FTX)
Knowledge-derived Answer: Sam Bankman-Fried

**Output**:

{{
  "Insufficient_information": false,
  "Path_score": {{
    "Path 1": 1,
    "Path 2": 2
  }}
}}
**Reasoning for the example scores (for your understanding only, do not include in output)**:
- Path 2 gets a 2 as it's the most direct evidence.
- Path 1 gets a 1 as it's a helpful logical link.
- Paths 3 and 4 are irrelevant, so they are completely ignored and not included in the `Path_score` dictionary.

---

### Input for Scoring

**Question**: {question}
**Knowledge-derived Answer**: {last_response}

Quickly identify the key supporting or contradicting paths based on the simplified guide. Ensure your output is a single, valid JSON object.
**Output**:
'''

# 这是两组任务将要共同使用的 System Prompt
score_feedback_prompt_standard_system_test = (
    "You are an expert AI assistant specializing in knowledge graph question answering. "
    "Your response must be based *only* on the information contained within the provided knowledge sequences below. "
    "Follow the user's specific instructions for the task at hand.\n\n"
    "### Retrieved Knowledge Paths\n"
    "{knowledge_sequences}"
)
# User Prompt 2: 用于生成打分JSON
# User Prompt 2 (Optimized): 用于生成更精确的打分JSON
score_feedback_prompt_standard_user_test = '''
### Task: Score and Analyze Paths

You are a meticulous and logical Knowledge Analyst. Your primary task is to critically evaluate a set of retrieved knowledge paths and score their actual contribution to answering a given **Question**. You must output a **valid JSON object** and nothing else.

### Guiding Principles for Scoring

1.  **Semantic Relevance over Keyword Matching**: A path's score depends on whether the **relationship** it describes helps answer the **Question**. Do not assign a high score simply because a path contains keywords from the question or answer. The context and meaning are paramount.
2.  **Directness and Completeness**: A path that directly connects the subject of the **Question** to the core information in the **Knowledge-derived Answer** should receive the highest score.
3.  **Logical Contribution**: A path might not contain the final answer but could be a crucial logical step. Evaluate its contribution to the overall reasoning chain.
4.  **Strict Contradiction**: A path should only be scored negatively if it presents information that directly contradicts the **Knowledge-derived Answer**.

### Detailed Scoring Rubric [-5, 5]

All scores must be **integers** in the range `[-5, 5]`.

* **5 (Direct & Complete Evidence)**: The path single-handedly provides or confirms the answer. It explicitly links the entities in the question to the answer.
* **3 to 4 (Strong Inferential Support)**: The path provides a critical piece of information that is a necessary logical step to reach the answer. For example, it establishes a key relationship like `(Entity A) -> [is the founder of] -> (Entity B)`.
* **1 to 2 (Relevant Context)**: The path provides useful background information that is related to the question, but it is not sufficient to infer the answer on its own.
* **0 (Irrelevant / Non-contributory)**: The path is topically related (e.g., mentions an entity from the question) but provides no information that helps in answering the specific question asked. **This is a common trap; a path mentioning "FTX" in the context of its marketing budget is irrelevant to a question about its founder's fraud trial.**
* **-1 to -3 (Misleading / Weak Contradiction)**: The path provides information that, while not a direct contradiction, might lead a user to an incorrect conclusion.
* **-4 to -5 (Direct Contradiction)**: The path states a fact that directly and unambiguously contradicts the `Knowledge-derived Answer`.

### Output Format (strictly required)

{{
  "Reasoning_path": string,          // A reasoning chain (<= 3 hops) logically explaining how the high-scoring paths lead to the answer. If no path is useful, this should be "".
  "Insufficient_information": boolean, // True if no path provides meaningful support for the answer.
  "Path_score": {{                  // Key: Path ID (e.g. "Path 0"), Value: integer ∈ [-5, 5]. Score every path provided.
    "Path X": integer
  }}
}}

### Example

**Input**:
Question: Who is on trial for fraud related to FTX?
Retrieved Knowledge Paths:
  - Path 1: (FTX) -[founded by]-> (Sam Bankman-Fried)
  - Path 2: (Sam Bankman-Fried) -[accused of]-> (Orchestrating financial fraud)
  - Path 3: (FTX) -[sponsored]-> (Mercedes-AMG Petronas F1 Team)
  - Path 4: (John Ray III) -[took over as CEO of]-> (FTX)
Knowledge-derived Answer: Sam Bankman-Fried

**Output**:

{{
  "Reasoning_path": "Path 1 establishes that Sam Bankman-Fried is the founder of FTX. Path 2 directly links him to accusations of financial fraud. Together, they strongly support the answer.",
  "Insufficient_information": false,
  "Path_score": {{
    "Path 1": 4,
    "Path 2": 5,
    "Path 3": 0,
    "Path 4": 1
  }}
}}
**Reasoning for the example scores**:
- Path 2 gets a 5 because it's the most direct evidence linking the person to the crime.
- Path 1 gets a 4 because establishing the founder relationship is a critical step.
- Path 3 gets a 0 because FTX's F1 sponsorship is completely irrelevant to a fraud trial. This is a classic example of keyword-bait.
- Path 4 gets a 1 because it provides relevant context about FTX's leadership but doesn't directly help identify who is on trial for the fraud that occurred before the new CEO took over.

---

### Input for Scoring

**Question**: {question}
**Knowledge-derived Answer**: {last_response}

Critically evaluate each path based on the principles and rubric provided. Ensure your output is a single, valid JSON object.
**Output**:
'''

# User Prompt 2 (Optimized for Speed)
score_feedback_prompt_standard_user_test_optimized = '''
### Task: Identify Key Evidence Paths

You are an efficient Evidence Triage Agent. Your task is to quickly identify which of the retrieved knowledge paths provide significant evidence related to the **Question** and the **Knowledge-derived Answer**. Your goal is speed and efficiency. **Do not evaluate every path.** Focus only on the most impactful ones.

Your output must be a **valid JSON object** and nothing else.

### Simplified Scoring Guide

Assign a score **only to the paths that are highly relevant**. Ignore all other paths.

*   **2 (Crucial Support)**: The path provides direct, critical evidence that confirms the answer.
*   **1 (Helpful Context)**: The path offers a necessary logical step or important background to understand the answer.
*   **-2 (Direct Contradiction)**: The path states a fact that directly and unambiguously contradicts the `Knowledge-derived Answer`.

### Output Format (strictly required)

Provide a JSON object with the following structure. The `Path_score` dictionary should **only include entries for the paths you have scored**.

{{
  "Insufficient_information": boolean, // True if no path provides meaningful support for the answer.
  "Path_score": {{                  // Key: Path ID, Value: integer (2, 1, or -2). Only include the most relevant paths.
    "Path X": integer
  }}
}}

### Example

**Input**:
Question: Who is on trial for fraud related to FTX?
Retrieved Knowledge Paths:
  - Path 1: (FTX) -[founded by]-> (Sam Bankman-Fried)
  - Path 2: (Sam Bankman-Fried) -[accused of]-> (Orchestrating financial fraud)
  - Path 3: (FTX) -[sponsored]-> (Mercedes-AMG Petronas F1 Team)
  - Path 4: (John Ray III) -[took over as CEO of]-> (FTX)
Knowledge-derived Answer: Sam Bankman-Fried

**Output**:

{{
  "Insufficient_information": false,
  "Path_score": {{
    "Path 1": 1,
    "Path 2": 2
  }}
}}
**Reasoning for the example scores (for your understanding only, do not include in output)**:
- Path 2 gets a 2 as it's the most direct evidence.
- Path 1 gets a 1 as it's a helpful logical link.
- Paths 3 and 4 are irrelevant, so they are completely ignored and not included in the `Path_score` dictionary.

---

### Input for Scoring

**Question**: {question}
**Knowledge-derived Answer**: {last_response}

Quickly identify the key supporting or contradicting paths based on the simplified guide. Ensure your output is a single, valid JSON object.
**Output**:
'''

# 这个东西应该交给huggingface client处理
llama_instruct_system = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_instruction}"
)

llama_instruct_user = (
    "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{user_instruction}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
)

qwen2_instruct_system = (
    "<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
)
qwen2_instruct_user = (
    "<|im_start|>user\n{user_instruction}\n<|im_end|>\n<|im_start|>assistant"
)

qwen3_instruct_system = (
    "<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
)
qwen3_instruct_user_think = (
    "<|im_start|>user\n{user_instruction}\n<|im_end|>\n<|im_start|>assistant"
)

qwen3_instruct_user_noThink = (
    "<|im_start|>user\n{user_instruction}\n  /no_think <|im_end|>\n<|im_start|>assistant"
)

redundant_relationship_prompt_basic_system = (
    "You are an AI assistant specialized in analyzing text and identifying unique statements based on specific rules. Focus on semantic meaning, factual details, and minimizing redundancy according to the user's instructions.\n"
)

# basic 分离 冗余关系提示词
redundant_relationship_prompt_basic_user = (
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
    "**Output:**"
)



score_feedback_prompt_standard_system_test_1 = """
You are an expert in knowledge graph question answering and retrieval evaluation.
Your sole task is to output a **valid JSON object** that conforms exactly to the required structure.
Do NOT output any explanations, markdown, or extra text outside JSON.

---

## Retrieved Knowledge (to be evaluated)

Below are the retrieved graph-based knowledge paths used for answering:

{knowledge_paths}

---

You must treat these as the **sole evidence source** for reasoning about the question and its knowledge-derived answer.
All reasoning, scoring, and judgments must be grounded in these paths and their internal triples.
If the paths fail to provide sufficient or relevant information, explicitly mark `Insufficient_information: true`.

Follow the subsequent instructions strictly.
"""

score_feedback_prompt_standard_user_test_1 = """
### Task

You are evaluating how effectively the retrieved **knowledge graph paths** and their **triples (hops)** support or contradict the *knowledge-derived answer* for a given question.

You must:
1. Reconstruct a short reasoning chain (≤3 hops) **based only on the retrieved paths** that could logically lead to the given answer.
2. Assign scores to both:
   - The entire **path** (coarse-grained)
   - Each **triple within the path** (fine-grained)

### Evaluation Logic

Think in two steps:
1. **Top-down**: Identify which paths most plausibly explain or justify the knowledge-derived answer.
2. **Bottom-up**: Evaluate how each triple contributes (positively, negatively, or neutrally) to that reasoning chain.

### Output Format (STRICTLY REQUIRED)

Output must be a **single JSON object** in the following form:

{{
  "Reasoning_path": string,                  // A concise reasoning chain (≤3 hops) showing how retrieved knowledge leads to the answer
  "Insufficient_information": boolean,      // True if retrieved knowledge cannot support or refute the answer
  "Path_score": {{                          // Coarse-grained evaluation of each full path
    "Path X": integer                       // integer ∈ [-5, 5], higher = stronger support
  }},
  "Triple_score": {{                        // Fine-grained evaluation per triple within each path
    "Path X": {{
      "Triple 1": integer,                  // integer ∈ [-5, 5]
      "Triple 2": integer
    }}
  }}
}}

### Scoring Rules

- All scores must be integers in the closed range [-5, 5].
- **Path_score (coarse-grained):**
  - >0 → Path supports or helps infer the answer
  - <0 → Path contradicts or misleads the answer
  - 0 → Path is irrelevant or neutral
- **Triple_score (fine-grained):**
  - >0 → Triple provides factual or logical support for the knowledge-derived answer
  - <0 → Triple directly conflicts with or undermines it
  - 0 → Triple is unrelated or marginally relevant
- If knowledge is insufficient or irrelevant overall, set:
  - `Insufficient_information: true`
  - `Path_score` and `Triple_score` to empty `{{}}`

### Constraints

- Evaluate **up to 5** retrieved paths.
- Each path must be considered, even if all scores = 0.
- The reasoning chain and scores must be **consistent** with the knowledge-derived answer.
- Numerical, temporal, or named-entity facts must match exactly for a positive score.

### Example

**Input**
Question: Who is on trial for fraud related to FTX?
Knowledge-derived Answer: Sam Bankman-Fried
User Feedback: Correct

**Output**
{{
  "Reasoning_path": "FTX - founded_by -> Sam Bankman-Fried --- Sam Bankman-Fried - accused_of -> Fraud",
  "Insufficient_information": false,
  "Path_score": {{
    "Path 0": 5,
    "Path 5": 3
  }},
  "Triple_score": {{
    "Path 0": {{
      "Triple 1": 5
    }},
    "Path 5": {{
      "Triple 1": 1,
      "Triple 2": 3
    }}
  }}
}}

---

**Now evaluate the following input:**

**Question**: {question}  
**Knowledge-derived Answer**: {last_response}  

Your output must be a single JSON object as specified above.
Do not output explanations.

**Output:**
"""

score_feedback_prompt_baisc_system = (
  "You are an AI assistant tasked with evaluating the factual relevance and correctness of retrieved sets of inferential statements (each statement in natural language form) in relation to a given query and the model's previous response (Last Response). This evaluation occurs in the second step of a process, focusing on sets of statements used or relevant to the Last Response. Your task is to assign scores only to sets of statements that, when their information is combined and considered as a whole, are directly relevant to the query's specific factual question, ignoring irrelevant ones.\n\n"
)

# basic 分离 反馈提示词
score_feedback_prompt_baisc_user = (
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
)

# score_feedback_prompt_standard_system = (
#     "You are an expert in the field of knowledge graph question answering. You specialize in analyzing questions, retrieved knowledge paths, model-generated answers, and user feedback to evaluate reasoning consistency and path relevance.\n"
# )

# ```json
# ```

# **User Feedback**: {flag_TF} # 这个值没有用处

# * Positive score (>0): Path supports the answer (higher = stronger support).
# * Negative score (<0): Path contradicts the answer.
# * Zero (0): Path irrelevant or weak and should not be included in the output.
# * For numeric answers: only exact matches count as supportive.
# * Note: Every retrieved path must be evaluated and assigned a score according to the rules above (including assigning 0 where appropriate). HOWEVER, the final JSON output's "Path_score" object must only include paths with non-zero scores (i.e., exclude any paths assigned a score of 0). Scores should be numeric (float or int) within [-1, 1].

#   "Path_score": {{
#     "Path 0": 0.9,
#     "Path 5": 0.6,
#     "Path 6": -0.3,
#     "Path 9": 0.3
#   }}

#   "Path_score": {{
#     "Path 0": 0.9,
#     "Path 1": 0.25,
#     "Path 2": 0.6,
#     "Path 3": -0.3,
#     "Path 4": 0.3
#   }}

# * Zero (0): Path irrelevant or weak.

score_feedback_prompt_standard_system = '''You are an expert in knowledge graph question answering. Your sole task is to output a **valid JSON object** that conforms to the required structure.
You must **not** output explanations, markdown, or any text outside the JSON.'''

score_feedback_prompt_standard_user = '''
### Revised Prompt

You are an expert in knowledge graph question answering. Your sole task is to output a **valid JSON object** that conforms to the required structure.
You must **not** output explanations, markdown, or any text outside the JSON.

### Output Format (strictly required)

The output must be a single JSON object with the following keys:

{{
  "Reasoning_path": string,        // A reasoning chain (<= 3 hops) explaining the answer, ultimately pointing to the Knowledge-derived Answer, or "" if insufficient info
  "Insufficient_information": boolean, // True if the derived answer is "insufficient information", else False
  "Path_score": {{                // Key: Path ID (e.g. "Path 0"), Value: number in [-1, 1], or {{}} if insufficient info
    "Path X": number
  }}
}}

### Scoring Rules

* Positive score (>0): Path supports the answer (higher = stronger support).
* Negative score (<0): Path contradicts the answer.
* Zero (0): The path is irrelevant or only weakly related to the question and the knowledge-derived answer.
* For numeric answers: only exact matches count as supportive.
* Note: Every retrieved path must be evaluated and assigned a score according to the rules above (including assigning 0 where appropriate). Scores should be numeric (float or int) within [-1, 1].

### Examples

**Input**:
Question: Who is on trial for fraud related to FTX?
Knowledge-derived Answer: Sam Bankman-Fried
User Feedback: Correct

**Output**:

{{
  "Reasoning_path": "One hop(Founder of failed cryptocurrency exchange ftx <- Is - Sam bankman-fried) --- One hop(Sam bankman-fried - Accused of -> Orchestrating financial fraud) --- Knowledge-derived Answer(Sam Bankman-Fried)",
  "Insufficient_information": false,
  "Path_score": {{
    "Path 0": 0.9,
    "Path 5": 0.6,
    "Path 6": -0.3,
    "Path 9": 0.3
  }}
}}

---

### Input

**Question**: {question}
**Retrieved Knowledge Paths**: \n{knowledge_paths}
**Knowledge-derived Answer**: {last_response}

This is very important to my career, You'd better be sure.
**Output**:
'''

# This is very important to my career, You'd better be sure.


score_feedback_prompt_standard_system_mulRounds = '''You are an expert in knowledge graph question answering. Your sole task is to output a **valid JSON object** that conforms to the required structure.
You must **not** output explanations, markdown, or any text outside the JSON.'''

score_feedback_prompt_standard_user1_mulRounds = "**Question**: {question}"

score_feedback_prompt_standard_assistant_mulRounds = (
    "Based on the question, I have gathered the following potentially relevant knowledge paths and formulated an answer.\n\n"
    "**Retrieved Knowledge Paths**: \n{knowledge_paths}\n"
    "**Knowledge-derived Answer**: {last_response}"
)

score_feedback_prompt_standard_user2_mulRounds = '''
### Output Format (strictly required)

The output must be a single JSON object with the following keys:

{
  "Reasoning_path": string,        // A reasoning chain (<= 3 hops) explaining the answer, ultimately pointing to the Knowledge-derived Answer, or "" if insufficient info
  "Insufficient_information": boolean, // True if the derived answer is "insufficient information", else False
  "Path_score": {                // Key: Path ID (e.g. "Path 0"), Value: number in [-1, 1], or {} if insufficient info
    "Path X": number
  }
}

### Scoring Rules

* Positive score (>0): Path supports the answer (higher = stronger support).
* Negative score (<0): Path contradicts the answer.
* Zero (0): Path irrelevant or weak and should not be included in the output.
* For numeric answers: only exact matches count as supportive.
* Note: Every retrieved path must be evaluated and assigned a score according to the rules above (including assigning 0 where appropriate). HOWEVER, the final JSON output's "Path_score" object must only include paths with non-zero scores (i.e., exclude any paths assigned a score of 0). Scores should be numeric (float or int) within [-1, 1].

### Examples

**Input**:
Question: Who is on trial for fraud related to FTX?
Knowledge-derived Answer: Sam Bankman-Fried
User Feedback: Correct

**Output**:

{
  "Reasoning_path": "One hop(Founder of failed cryptocurrency exchange ftx <- Is - Sam bankman-fried) --- One hop(Sam bankman-fried - Accused of -> Orchestrating financial fraud) --- Knowledge-derived Answer(Sam Bankman-Fried)",
  "Insufficient_information": false,
  "Path_score": {
    "Path 0": 0.9,
    "Path 5": 0.6,
    "Path 6": -0.3,
    "Path 9": 0.3
  }
}

---

This is very important to my career, You'd better be sure.
**Output**:
'''

def short_cut(original_path):
  return f'''
You are an expert Knowledge Graph Analyst. Your task is to analyze a complete three-hop path (A -> B -> C -> D) and compress the most logical two-hop segment into a single, direct, and meaningful relationship.

**Your task has two steps:**
1.  **Analyze and Decide**: Examine the entire three-hop path. You must decide which segment is more suitable for compression: the first two hops (A -> B -> C) or the last two hops (B -> C -> D). Choose the segment that results in the most informative and semantically coherent new relationship.
2.  **Infer and Compress**: For the segment you selected, infer a direct relationship between its start and end nodes.

**Guiding Principles for Your Decision:**
* **Semantic Cohesion**: Choose the segment where the three nodes form a strong, logical narrative (e.g., a person creating a product, a city being part of a continent).
* **Information Value**: The compressed relationship should be a valuable shortcut, not a trivial or overly generic fact. A compression from `(Company -> Produces -> Product)` to `(Founder -> Masterminded -> Product)` is highly valuable.
* **Avoid Weak Links**: If one hop is a very generic or weak link (like "has type" or "is related to"), compressing the other, more specific segment is often better.

You will be given a JSON object describing the full three-hop path. You MUST respond ONLY with a single JSON object representing the compressed path.

**In your output, the `explanation` field is critical.** It must clearly state:
1.  Which segment you chose to compress (e.g., "I chose the last two hops...").
2.  Why you chose that segment over the other one.
3.  Why the new inferred relationship is appropriate.

**Output JSON Format:**
{{
  "source_node": "...",
  "inferred_relationship": {{
    "label": "human-readable label for the inferred relationship",
    "explanation": "Your detailed reasoning for segment selection and relationship inference.",
    "confidence_score": 0.0
  }},
  "target_node": "..."
}}

---
**Example 1: Compressing the first two hops (A -> B -> C)**

**Input:**
{{
  "three_hop_path": [
    {{ "source_node": "J.K. Rowling", "relationship": "wrote", "target_node": "Harry Potter"}},
    {{ "source_node": "Harry Potter", "relationship": "features character", "target_node": "Hermione Granger"}},
    {{ "source_node": "Hermione Granger", "relationship": "portrayed by", "target_node": "Emma Watson"}}
  ]
}}

**Output:**
{{
  "source_node": "J.K. Rowling",
  "inferred_relationship": {{
    "label": "created the character of",
    "explanation": "Selected the first two hops (J.K. Rowling -> Harry Potter -> Hermione Granger). This is the most logical segment because an author writing a book that features a character is a direct act of creation. Compressing this to 'created the character of' is a powerful and accurate summary. The second segment (Book -> Character -> Actor) is a weaker link for compression.",
    "confidence_score": 0.98
  }},
  "target_node": "Hermione Granger"
}}

---
**Example 2: Compressing the last two hops (B -> C -> D)**

**Input:**
{{
  "three_hop_path": [
    {{ "source_node": "NASA", "relationship": "operates", "target_node": "Jet Propulsion Laboratory"}},
    {{ "source_node": "Jet Propulsion Laboratory", "relationship": "managed", "target_node": "Mars Rover Perseverance"}},
    {{ "source_node": "Mars Rover Perseverance", "relationship": "landed on", "target_node": "Mars"}}
  ]
}}

**Output:**
{{
  "source_node": "Jet Propulsion Laboratory",
  "inferred_relationship": {{
    "label": "sent a rover to",
    "explanation": "Selected the last two hops (JPL -> Mars Rover Perseverance -> Mars). This segment represents a clear mission objective. JPL managing a rover that landed on Mars can be logically compressed to JPL 'sent a rover to' Mars. The first segment (NASA -> JPL -> Rover) is more about operational hierarchy rather than a distinct event.",
    "confidence_score": 0.95
  }},
  "target_node": "Mars"
}}

---
**Now, process the following input:**

**Input:**
{json.dumps(original_path, indent = 2)}
'''
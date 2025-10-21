import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class Hop:
    subject: str
    relation: str
    object: str
    direction: str  # "->" or "<-"

@dataclass
class ParsedOutput:
    reasoning_path_raw: str
    insufficient_information: bool
    path_scores: Dict[str, float]
    hops: List[Hop]

# --- core parsing helpers ---

_SPLIT_PATTERN = re.compile(r"\s-{-}{2,}\s*|\s*---+\s*", re.IGNORECASE)  
# 解释：
# - 上面的模式宽松匹配多连字符的段间分隔（如 '---', '--->>>' 等），两边可有空白。

_TRIPLE_FWD = re.compile(r"^\s*(.*?)\s*-\s*(.*?)\s*->\s*(.*?)\s*$")
_TRIPLE_BWD = re.compile(r"^\s*(.*?)\s*<-\s*(.*?)\s*-\s*(.*?)\s*$")

def _parse_single_hop(chunk: str) -> Hop:
    """
    支持两种格式：
      1) A - Rel -> B
      2) A <- Rel - B
    返回 Hop(subject, relation, object, direction)
    """
    m = _TRIPLE_FWD.match(chunk)
    if m:
        s, r, o = m.groups()
        return Hop(s.strip(), r.strip(), o.strip(), "->")
    m = _TRIPLE_BWD.match(chunk)
    if m:
        s, r, o = m.groups()
        # 注意：当匹配 'A <- Rel - B' 时，语义是 B -(Rel)-> A
        # 为了统一输出，把 subject 设置为 B，object 设置为 A，方向仍记为 "<-" 以保留原始形态
        return Hop(s.strip(), r.strip(), o.strip(), "<-")
    raise ValueError(f"Unrecognized hop format: {chunk!r}")

def parse_reasoning_path(path: str) -> List[Hop]:
    if not path:
        return []
    # 先把各种“长连字符分隔”标准化为 '---'，随后按 '---' 切分
    normalized = re.sub(r"-{3,}>*", "---", path)
    chunks = [c for c in normalized.split("---") if c.strip()]
    hops = []
    for c in chunks:
        hop = _parse_single_hop(c)
        hops.append(hop)
    return hops

def validate_struct(obj: Dict[str, Any]) -> Tuple[bool, str]:
    # 1) 必要键存在
    required = ["Reasoning_path", "Insufficient_information", "Path_score"]
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"

    # 2) 类型检查
    if not isinstance(obj["Reasoning_path"], str):
        return False, "Reasoning_path must be a string"
    if not isinstance(obj["Insufficient_information"], bool):
        return False, "Insufficient_information must be a boolean"
    if not isinstance(obj["Path_score"], dict):
        return False, "Path_score must be an object (dict)"

    # 3) 分数范围检查
    for k, v in obj["Path_score"].items():
        if not isinstance(v, (int, float)):
            return False, f"Path_score['{k}'] must be a number"
        if not (-1 <= float(v) <= 1):
            return False, f"Path_score['{k}'] must be in [-1, 1], got {v}"

    # 4) 若 insufficient 信息为 True，则 Reasoning_path 应为空串（可按需放宽）
    if obj["Insufficient_information"] and obj["Reasoning_path"].strip() != "":
        return False, "When Insufficient_information is true, Reasoning_path must be empty"

    return True, "ok"

def parse_output(json_input: str | Dict[str, Any]) -> ParsedOutput:
    # 支持传入 JSON 字符串或字典
    data = json.loads(json_input) if isinstance(json_input, str) else json_input

    ok, msg = validate_struct(data)
    if not ok:
        raise ValueError(f"Invalid structure: {msg}")

    hops = parse_reasoning_path(data["Reasoning_path"])

    return ParsedOutput(
        reasoning_path_raw=data["Reasoning_path"],
        insufficient_information=data["Insufficient_information"],
        path_scores={k: float(v) for k, v in data["Path_score"].items()},
        hops=hops
    )

# --- demo with your sample ---

if __name__ == "__main__":
    sample = r'''
    {
      "Reasoning_path": "Carole king & james taylor: just call out my name - Premieres on -> January 2 --->>> Carole king <- Showcasing - Carole king & james taylor: just call out my name - Premiered on -> January 2, 2022",
      "Insufficient_information": false,
      "Path_score": {
        "Path 0": 0.8,
        "Path 1": 0.7,
        "Path 36": 0.9,
        "Path 37": 1
      }
    }
    '''
    parsed = parse_output(sample)
    print("Insufficient_information:", parsed.insufficient_information)
    print("Path_score:", parsed.path_scores)
    print("Hops parsed:")
    for i, h in enumerate(parsed.hops, 1):
        # 统一展示：当 direction 为 "<-"，语义等价为 object -(relation)-> subject
        if h.direction == "->":
            print(f"  Hop {i}: {h.subject} - {h.relation} -> {h.object}")
        else:
            print(f"  Hop {i}: {h.subject} <- {h.relation} - {h.object}  (i.e., {h.object} - {h.relation} -> {h.subject})")

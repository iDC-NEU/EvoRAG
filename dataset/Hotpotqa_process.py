
import json
import os
from pathlib import Path
import random
from collections import defaultdict

HOTPOTQA600_PATH = os.path.join(Path(__file__).parent, "hotpotqa_graph", "hotpotqa_600.json")

HOTPOTQA_TRAIN_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_train_v1.1.json")
HOTPOTQA_FULLWIKI_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_fullwiki_v1.json")
HOTPOTQA_TEST_FULLWIKI_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_test_fullwiki_v1.json")
HOTPOTQA_DISTRACTOR_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_distractor_v1.json")


HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_distractor_v1_random_1200.json")
HOTPOTQA_DISTRACTOR_RANDOM_200_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_distractor_v1_random_200.json")
HOTPOTQA_FULLWIKI_RANDOM_200_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_dev_fullwiki_v1_random_200.json")
HOTPOTQA_TRAIN_RANDOM_200_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_train_v1.1_random_200.json")
HOTPOTQA_TEST_FULLWIKI_RANDOM_200_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "hotpot_test_fullwiki_v1_random_200.json")
HOTPOTQA_DISTRACTOR_RANDOM_600_PATH = os.path.join(Path(__file__).parent, "hotpotqa", "random_600.json")


# 现在有的random1200来自于HOTPOTQA_DISTRACTOR_PATH

class Hotpot_dataset:
    def __init__(self):
        self.corpus = []
        self.query = []
        self.answer = []
        self.qa = {}
        self.random_1200 = None
        # 随机提取一些文件
        # self.random(origin_file = HOTPOTQA_DISTRACTOR_PATH, target_file = HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH)
        # self.random(origin_file = HOTPOTQA_FULLWIKI_PATH, target_file = HOTPOTQA_FULLWIKI_RANDOM_200_PATH)
        # self.random(origin_file = HOTPOTQA_TEST_FULLWIKI_PATH, target_file = HOTPOTQA_TEST_FULLWIKI_RANDOM_200_PATH)
        # self.random(origin_file = HOTPOTQA_TRAIN_PATH, target_file = HOTPOTQA_TRAIN_RANDOM_200_PATH)
        
        # self.simple_context_analysis(HOTPOTQA_DISTRACTOR_PATH)
        # self.simple_context_analysis(HOTPOTQA_FULLWIKI_PATH)
        # self.simple_context_analysis(HOTPOTQA_TRAIN_PATH)
        # self.simple_context_analysis(HOTPOTQA_TEST_FULLWIKI_PATH)
        
        # self.analyze(HOTPOTQA600_PATH, HOTPOTQA_DISTRACTOR_PATH)
        # self.analyze_v2(HOTPOTQA_DISTRACTOR_PATH, HOTPOTQA_TEST_FULLWIKI_PATH)
        self.analyze_v2(HOTPOTQA_DISTRACTOR_PATH, HOTPOTQA_TRAIN_PATH)
        
        # self.test_exist_simple()
        # self.simple_overlap_test()
        # self.read()
        
    def random(self, origin_file = HOTPOTQA_DISTRACTOR_PATH, target_file = HOTPOTQA_DISTRACTOR_RANDOM_1200_PATH):
        if os.path.exists(target_file):
            with open(target_file, 'r', encoding='utf-8') as file:
                self.random_1200 = json.load(file)
            return
        
        with open(origin_file, 'r', encoding='utf-8') as file:
            data = json.load(file) 
            print(f"Loaded {len(data)} items from {origin_file}")
        
        if len(data) < 1200:
            raise ValueError("The number of elements in the raw data is less than 1200")

        # random_indices = random.sample(range(len(data)), 200)
        # self.random_1200 = [data[i] for i in random_indices]
        self.random_1200 = data[:200]
        
        with open(target_file, 'w', encoding='utf-8') as file:
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
        
    def test_exist_simple(self):
        """简化版：只统计数量"""
        # 1. 读取600条数据
        with open(HOTPOTQA600_PATH, 'r', encoding='utf-8') as f:
            data_600 = json.load(f)
        
        questions_600 = set(item['question'] for item in data_600)
        
        # 2. 统计三个数据集中的存在情况
        datasets = [
            ("FULLWIKI", HOTPOTQA_FULLWIKI_PATH),
            ("TEST_FULLWIKI", HOTPOTQA_TEST_FULLWIKI_PATH),
            ("DISTRACTOR", HOTPOTQA_DISTRACTOR_PATH)
        ]
        
        print(f"HOTPOTQA600_PATH 问题总数: {len(questions_600)}")
        
        for name, path in datasets:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                
                source_questions = set(item['question'] for item in source_data)
                common_count = len(questions_600.intersection(source_questions))
                
                print(f"{name}: {common_count} 个问题存在 (覆盖率: {common_count/len(questions_600)*100:.1f}%)")
                
            except Exception as e:
                print(f"{name}: 读取失败 - {e}")
        
        # 3. 计算在至少一个数据集中存在的问题数
        all_source_questions = set()
        for _, path in datasets:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                    all_source_questions.update(item['question'] for item in source_data)
            except:
                continue
        
        found_in_any = len(questions_600.intersection(all_source_questions))
        print(f"\n在任何源数据集中存在的问题数: {found_in_any}/{len(questions_600)} ({found_in_any/len(questions_600)*100:.1f}%)")
    
    def simple_overlap_test(self):
        """简化的重合度测试"""
        try:
            with open(HOTPOTQA_FULLWIKI_PATH, 'r', encoding='utf-8') as f:
                fullwiki = json.load(f)
            print(f"FULLWIKI: {len(fullwiki)} 个问题")
            fullwiki_qs = set(item['question'] for item in fullwiki)
            
            with open(HOTPOTQA_DISTRACTOR_PATH, 'r', encoding='utf-8') as f:
                distractor = json.load(f)
            print(f"DISTRACTOR: {len(distractor)} 个问题")
            distractor_qs = set(item['question'] for item in distractor)
            
            common = fullwiki_qs.intersection(distractor_qs)
            
            print(f"数据集统计:")
            print(f"FULLWIKI set: {len(fullwiki_qs)} 个问题")
            print(f"DISTRACTOR set: {len(distractor_qs)} 个问题")
            print(f"共同问题: {len(common)} 个")
            print(f"重合率: {len(common)/len(fullwiki_qs)*100:.1f}% (对FULLWIKI)")
            print(f"重合率: {len(common)/len(distractor_qs)*100:.1f}% (对DISTRACTOR)")
            
        except Exception as e:
            print(f"错误: {e}")
            
            
    def simple_context_analysis(self, json_file):
        """最简单的上下文分析"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        contexts = set()
        type_count = defaultdict(int)
        level_count = defaultdict(int)

        lengths = []
        
        for item in data:
            for paragraph in item.get('context', []):
                if len(paragraph) > 1:
                    text = "".join(paragraph[1])
                    contexts.add(text)
                    lengths.append(len(text))
            if 'type' in item:
                type_count[item['type']] += 1
            if 'level' in item:
                level_count[item['level']] += 1
        
        unique_count = len(contexts)
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        
        print(f"Question count in file: {len(data)}")
        print(f"Path: {json_file}")
        print(f"不重复上下文数: {unique_count}")
        print(f"平均长度: {avg_length:.1f} 字符")
        print(f"类型统计: {dict(type_count)}")
        print(f"难度统计: {dict(level_count)}")
        
        # return unique_count, avg_length
        
    def analyze(self, source_file, target_file):
        """执行分析"""
        source_questions = set()
        type_count = defaultdict(int)
        level_count = defaultdict(int)

        try:
            # 读取源文件
            with open(source_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            
            # 提取问题
            for item in source_data:
                if 'question' in item:
                    source_questions.add(item['question'])
            
            # 读取目标文件并统计
            with open(target_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
            
            contexts = set()
            lengths = []
            
            for item in target_data:
                if 'question' in item and item['question'] in source_questions:
                    if 'context' in item:
                        for paragraph in item['context']:
                            if len(paragraph) > 1:
                                text = "".join(paragraph[1])
                                contexts.add(text)
                                lengths.append(len(text))
                    if 'type' in item:
                        type_count[item['type']] += 1

                    if 'level' in item:
                        level_count[item['level']] += 1
            
            # 保存统计结果
            print(f"source_file: {source_file}\t\ttarget_file: {target_file}")
            print(f"匹配到的问题数: {len(set(item['question'] for item in target_data if 'question' in item and item['question'] in source_questions))}")
            print(f"不重复上下文数: {len(contexts)}")
            print(f"平均长度: {sum(lengths) / len(lengths) if lengths else 0:.1f}")
            print(f"类型统计: {dict(type_count)}")
            print(f"难度统计: {dict(level_count)}")
            
        except Exception as e:
            print(f"分析失败: {e}")

    def analyze_v2(self, source_file, target_file):
        """执行分析"""
        source_questions = set()
        target_questions = set()
        type_count = defaultdict(int)
        level_count = defaultdict(int)

        try:
            # 读取源文件
            with open(source_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            
            # 提取问题
            for item in source_data:
                if 'question' in item:
                    source_questions.add(item['question'])
            
            # 读取目标文件并统计
            with open(target_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
            for item in target_data:
                if 'question' in item:
                    target_questions.add(item['question'])
            results = {
                # 基础统计
                'source_count': len(source_questions),
                'target_count': len(target_questions),
                
                # 重合度统计
                'intersection_count': len(source_questions & target_questions),
                'union_count': len(source_questions | target_questions),
                'source_only_count': len(source_questions - target_questions),
                'target_only_count': len(target_questions - source_questions),
                
                # 重合度百分比
                'intersection_rate': (
                    len(source_questions & target_questions) / len(source_questions) * 100
                    if source_questions else 0
                ),
            }
            print(f"source_file: {source_file}\t\ttarget_file: {target_file}")
            print(f"分析结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            print(f"分析失败: {e}")


if __name__ == '__main__':
    Hotpot_dataset()
    
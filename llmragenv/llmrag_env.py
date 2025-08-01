'''
Author: lpz 1565561624@qq.com
Date: 2024-09-20 14:35:55
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-11-15 15:20:20
FilePath: /lipz/fzb_rag_demo/RAGWebUi_demo/llmragenv/llmrag_env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import List, Optional

from chat.chat_base import ChatBase
from chat.chat_withoutrag import ChatWithoutRAG
from chat.chat_vectorrag import ChatVectorRAG
from chat.chat_graphrag import ChatGraphRAG
from chat.chat_unionrag import ChatUnionRAG
from database.vector.Milvus.milvus import MilvusDB
from llmragenv.LLM.llm_factory import ClientFactory
from database.graph.graph_dbfactory import GraphDBFactory
from llmragenv.Cons_Retri.KG_Construction import KGConstruction

from dataset.dataset import Dataset
from logger import Logger



def triples_to_json(triples):

    colors = [
        "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
        "#B5EAD7", "#ECC5FB", "#FFC3A0", "#FF9AA2", "#FFDAC1",
        "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2",
        "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA",
        "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1",
        "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2",
        "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA",
        "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1",
        "#E2F0CB", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#FF9AA2",
        "#FFDAC1", "#C7CEEA", "#FFB3BA", "#FFDFBA", "#FFFFBA",
        "#BAFFC9", "#BAE1FF", "#FFC3A0", "#FF9AA2", "#FFDAC1"
    ]

    json_result = {'edges': [], 'nodes': []}
    node_set = set()  # 用于追踪已经添加的节点

    import random

    for triple in triples:
        source = triple["source"]
        relationship = triple["relationship"]
        destination = triple["destination"]

        # 添加边
        json_result['edges'].append({
            'data': {
                'label': relationship,
                'source': source,
                'target': destination,
                'color': colors[random.randint(0,54)] # 可以根据需要自定义颜色
            }
        })

        # 添加源节点和目标节点（避免重复）
        if source not in node_set:
            json_result['nodes'].append({
                'data': {
                    'id': source,
                    'label': source,
                    'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                }
            })
            node_set.add(source)

        if destination not in node_set:
            json_result['nodes'].append({
                'data': {
                    'id': destination,
                    'label': destination,
                    'color': colors[random.randint(0,54)]  # 可以根据需要自定义颜色
                }
            })
            node_set.add(destination)

    return json_result



WITHOUTRAG = "Without RAG"
GRAPHRAG = "Graph RAG"
VECTORRAG = "Vector RAG"
VECTORGRAPHRAG = "Union"


class LLMRAGEnv(object):
    """
    可以聊任何东西的机器人
    """


    def __init__(self):
        pass
        
    def get_function(self, op1 : str, op2 : str, op3 : str) -> ChatBase:
        if op1 == WITHOUTRAG:
            return ChatWithoutRAG(self.llm)

        elif op1 == VECTORRAG:
            vector_db = MilvusDB('rgb', 1024, overwrite=False, store=True,retriever=True) #差一个供用户选择的下拉框，选择知识库
            return ChatVectorRAG(self.llm,vector_db)

        elif op1 == GRAPHRAG:
            graph_db = GraphDBFactory(op2).get_graphdb(space_name='rgb')
            return ChatGraphRAG(self.llm, graph_db)

        elif op1 == VECTORGRAPHRAG:
            return ChatUnionRAG(self.llm)
        
            
        else:
            print(f"{op1} is not supported, switch chat Cithout RAG")
            return ChatWithoutRAG(self.llm)

    def get_retrieve_result(self):
        return self.llm_func.retrieval_result()

    #将检索到的结果返回到前端的容器中
    def get_resulturl(self,op1:str):
        """According to the different methods of RAG, the retrieval results are returned to the frontend. 
           In the case of Graph RAG, a new webpage rendering graph topology will be generated.

        Args:
            op1 (str): The RAG way

        Returns:
            str: The text of vector rag or url of graph topology
        """
        result = self.get_retrieve_result()
        if result is None:
            content = "<h4>No Retrieve:</h4><ul>"
        else:
            if op1 == VECTORRAG:
                content = "<h4>Text Results:</h4><ul>"
                for text in result:
                    content += f"<li>{text}</li>"
                content += "</ul>"
            elif op1 == GRAPHRAG:

                import json
                with open('./templates/triplet.json', 'w', encoding='utf-8') as f:
                    json.dump(triples_to_json(result), f, ensure_ascii=False, indent=4)  # 使用 ensure_ascii=False 保持非 ASCII 字符的正确编码，indent=4 美化输出

                import time
                timestamp = str(int(time.time()*1000))
                content = f'''
                    <iframe src="http://202.199.13.88:8085?{timestamp}" width="1300" height="800" style="border:none;"></iframe>
                    '''
        return content

    def web_chat(self, message : str, history: List[Optional[List]] | None, op0 : str,op1 : str, op2 : str, op3 : str):
        
        """Used to chat with LLM from ./webchat.py

        Args:
            message (list[str]): The query list
            op0 (str): The llm name ("qwen:0.5b", "llama2:7b", "llama2:13b", "llama2:70b","qwen:7b","qwen:14b","qwen:72b", etc)
            op1 (str): The RAG way ("graph rag", "vector rag", "without rag", "union rag")
            op2 (str): The name of graph database ("nebulagraph", "neo4j", ect)
            op3 (str): The name of vector database ("MilvusDB")
        """

        print("model name : {}".format(op0))
        self.llm = ClientFactory(model_name=op0, llmbackend="openai").get_client()

        self.llm_func = self.get_function(op1, op2, op3)

        if history is None:
            history = []  # 确保历史记录初始化为一个空列表

        yield from self.llm_func.web_chat(message, history)

    def low_chat(self, message : str, history: List[Optional[List]],op1 : str):
        
        """Default chat with llm ("qwen:14b") and db name ("nebulagraph", "MilvusDB")

        Args:
            message (list[str]): The query list
            op1 (str): The RAG way ("graph rag", "vector rag", "without rag", "union rag")
        """

        
        print("model name : {}".format("qwen:14b"))
        self.llm = ClientFactory(model_name="qwen:14b", llmbackend="openai").get_client()

        self.llm_func = self.get_function(op1, "nebulagraph", "MilvusDB")

        if history is None:
            history = []  # 确保历史记录初始化为一个空列表

        yield from self.llm_func.web_chat(message, history)
        
    # 用于从后端启动大模型，主要测试数据集中的数据
    def backend_chat(self, message : list[str], op0 : str, op1 : str, op2 : str, op3 : str, op4 : str):
        """Used to chat with LLM from ./backend_chat.py

        Args:
            message (list[str]): The query list
            op0 (str): The llm name ("qwen:0.5b", "llama2:7b", "llama2:13b", "llama2:70b","qwen:7b","qwen:14b","qwen:72b", etc)
            op1 (str): The RAG way ("graph rag", "vector rag", "without rag", "union rag")
            op2 (str): The name of graph database ("nebulagraph", "neo4j", ect)
            op3 (str): The name of vector database ("MilvusDB")
            op4 (str): The llm backend ("openai", "llama_index")
        """
        
        print("model name : {}".format(op0))
        self.llm = ClientFactory(model_name=op0, llmbackend=op4).get_client()

        llm_func = self.get_function(op1, op2, op3)

        for query in message:
            print(f"user : {query}")
            print(f"{self.llm.model_name}: \033[33m {llm_func.chat_without_stream(query)}\033[0m")

    def chat_with_dataset(self, dataset : Dataset, args):
        """used to chat with dataset, from ./backend_chat.py

        Args:
            Dataset (Dataset): The dataset include dataset.query and dataset.answer
            op0 (str): The llm name ("qwen:0.5b", "llama2:7b", "llama2:13b", "llama2:70b","qwen:7b","qwen:14b","qwen:72b", etc)
            op1 (str): The RAG way ("graph rag", "vector rag", "without rag", "union rag")
            op2 (str): The name of graph database ("nebulagraph", "neo4j", ect)
            op3 (str): The name of vector database ("MilvusDB")
            op4 (str): The llm backend ("openai", "llama_index")
        """

        self.llm = ClientFactory(model_name=args.llm, llmbackend=args.llmbackend).get_client()
        
        from icecream import ic
        ic(self.llm.model_name)

        # llm_func = self.get_function(op1, op2, op3)
        if args.func == WITHOUTRAG:
            logger = Logger("chat_with_dataset_without_rag")
            # logger = Logger(f"{args.llmbackend}_{args.model_name}_{args.func}_{self.args.dataset_name}")
            llm_func = ChatWithoutRAG(self.llm)
        elif args.func == VECTORRAG:
            logger = Logger("chat_with_dataset_vector_rag")
            vector_db = MilvusDB(args.space_name, 1024, overwrite=False, store=True,retriever=True, similarity_top_k = 5)
            llm_func = ChatVectorRAG(self.llm,vector_db)
        elif args.func == GRAPHRAG:
            logger = Logger("chat_with_dataset_graph_rag")
            graph_db = GraphDBFactory(args.graphdb).get_graphdb(space_name=args.space_name)
            llm_func = ChatGraphRAG(self.llm, graph_db)
        elif args.func == VECTORGRAPHRAG:
            logger = Logger("chat_with_dataset_union_rag")
            llm_func = ChatUnionRAG(self.llm)
        
        response = []
        retrieve_result = []

        for i in range(len(dataset.query)):
            ic(i)
            response.append(llm_func.chat_without_stream(dataset.query[i]))
            ic(dataset.answer[i])
            ic("\n\n")
            if llm_func.retrieval_result() is not None:
                retrieve_result.append(llm_func.retrieval_result())
            else:
                retrieve_result.append("None")
        
        tt = 0
        labels = []
        for i in range(len(response)):
            flag = False
            if isinstance(dataset.answer[i], list):
                instance = [j.lower() for j in dataset.answer[i]]
                for j in instance:
                    if j in response[i].lower():
                        flag = True
                        tt += 1
                        break
            else:
                instance = dataset.answer[i].lower()
                if instance in response[i].lower():
                    flag = True
                    tt += 1
            labels.append(int(flag))
        
        acc = tt / len(response)

        logger.log(f"-----------------------accuracy: {acc}---------------------")

        for i in range(len(dataset.query)):
            # print(f"id: {i + 1} user : {dataset.query[i]}")
            # print(f"{self.llm.model_name}: \033[33m {response}\033[0m")
            # print()
            logger.log(f"id : {i+1}")
            logger.log(f"query: {dataset.query[i]}")
            logger.log(f"response: {response[i]}")
            # logger.log(f"response type: {type(response[i])}")
            logger.log(f"answer: {dataset.answer[i]}")
            logger.log(f"Is the response from the LLM correct?: {'Yes' if labels[i] == 1 else 'No'}")
            # 可以判断是否是graph或者vector rag，然后把检索结果返回，并写入log里面
            if retrieve_result:
                logger.log(f"retrieve result: {retrieve_result[i]}")
            logger.log("\n\n")
        

    def chat_to_KG_construction(self, data, args):
        """
        Args:
            data (_type_): dataset to construct kg
            op0 (str): llm name
            op2 (str): graph database name
            space_name (str): space name of graph database
        """
        
        print("model name : {}".format(args.llm))
        self.llm = ClientFactory(model_name=args.llm, llmbackend=args.llmbackend).get_client()
        graph_db = GraphDBFactory(args.graphdb).get_graphdb(space_name=args.space_name)
        

        KGConstruction(self.llm, graph_db, args.space_name).run(data, option="only_llm")
    
    def chat_to_KG_modify(self, args):
        try:
            from KGModify.KGModify import KGModify
            if args.llm == 'Meta-Llama-3-8B-Instruct':
                self.llm = ClientFactory(model_name=args.llm, llmbackend='huggingface').get_client()
            else:
                self.llm = ClientFactory(model_name=args.llm, llmbackend=args.llmbackend).get_client()
            graph_db = GraphDBFactory(args.graphdb).get_graphdb(space_name=args.space_name)
            # graph_db2 = GraphDBFactory(args.graphdb).get_graphdb(space_name=args.space_name)
            if args.llm_fb == "None":
                print(f"-----------1个模型------------------")
                KGModify(ChatGraphRAG(self.llm, graph_db), None, args).run()
            else:
                print(f"-----------2个模型------------------{args.llm_fb}")
                self.llm_fb = ClientFactory(model_name=args.llm_fb, llmbackend=args.llmbackend).get_client()
                graph_db_null = GraphDBFactory(args.graphdb).get_graphdb(space_name="api")
                # print("lalala")
                # assert(False)
                KGModify(ChatGraphRAG(self.llm, graph_db), ChatGraphRAG(self.llm_fb, graph_db_null), args).run()
        except ImportError as e:
            # 如果发生 ImportError，抛出带有详细信息的异常
            print("KGModify is not exit, please ask for author!")
            raise ImportError(f"Failed to import KGModify: {e}") from e
            
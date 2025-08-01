
from logger import Logger
from database.graph.graph_database import GraphDatabase
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from database.graph.nebulagraph.FormatResp import print_resp
# from nebula3.common import *
import os
import json
import numpy as np
import time
from utils.file_util import file_exist


class NebulaClient:

    def __init__(self):
        config = Config()
        config.max_connection_pool_size = 10

        self.connection_pool = ConnectionPool()
        ok = self.connection_pool.init([('127.0.0.1', 9669)], config)
        assert ok

        self.session = self.connection_pool.get_session('root', 'nebula')

    def __del__(self):
        if self.connection_pool:
            self.connection_pool.close()
        if self.session:
            try:
                self.session.release()
            except Exception as e:
                print(f"Unexpected error: {e}")

    def create_space(self, space_name):
        self.session.execute(
            f'CREATE SPACE IF NOT EXISTS {space_name}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);'
        )
        time.sleep(10)
        self.session.execute(
            f'USE {space_name}; CREATE TAG IF NOT EXISTS entity(name string);')
        self.session.execute(
            f'USE {space_name}; CREATE EDGE IF NOT EXISTS relationship(relationship string);'
        )
        self.session.execute(
            f'USE {space_name}; CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));'
        )
        time.sleep(10)

    def drop_space(self, space_name):
        if not isinstance(space_name, list):
            space_name = [space_name]
        for space in space_name:
            self.session.execute(f'drop space {space}')

    def info(self, space_name):
        result = self.session.execute(
            f'use {space_name}; submit job stats; show stats;')
        print(result)
        print_resp(result)

    def count_edges(self, space_name):
        result = self.session.execute(
            f'use {space_name}; MATCH (m)-[e]->(n) RETURN COUNT(*);')
        print_resp(result)

    def show_space(self):
        result = self.session.execute('SHOW SPACES;')
        print_resp(result)

    def show_edges(self, space_name, limits):
        result = self.session.execute(
            f'use {space_name}; MATCH ()-[e]->() RETURN e LIMIT {limits};')
        print_resp(result)

    def clear(self, space_name):
        query = f'CLEAR SPACE {space_name};'
        self.session.execute(query)

    def save_triplets(self, space_name, file_path=None):
        if not file_path:
            file_path = space_name + '_triplets.json'

        all_triples = self.get_triplets(space_name=space_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(all_triples, file, ensure_ascii=False, indent=2)
            print(f'save {len(all_triples)} triples to {file_path}.')

    def get_triplets(self, space_name):
        result = self.session.execute(
            f'use {space_name}; MATCH (n1)-[e]->(n2) RETURN n1, e, n2;')

        all_triples = []

        if result.row_size() > 0:
            for row in result.rows():
                values = row.values

                head, relation, tail = '', '', ''

                for value in values:
                    if value.field == 9:  # 对应 Vertex
                        vertex = value.get_vVal()
                        if not head:
                            head = vertex.vid.get_sVal().decode('utf-8')
                        else:
                            tail = vertex.vid.get_sVal().decode('utf-8')

                    elif value.field == 10:  # 对应 Edge
                        edge = value.get_eVal()
                        relation = edge.props.get(
                            b'relationship').get_sVal().decode('utf-8')

                triplet = [head, relation, tail]
                all_triples.append(triplet)
        else:
            print(f'No triplets found {space_name}.')

        all_triples = set(tuple(triplet) for triplet in all_triples)
        all_triples = [list(triplet) for triplet in all_triples]

        return all_triples
    

    def get_retrieve_triplets_1hop(self, space_name, entities: list[str]):
        self.session.execute(f'USE {space_name}')

        # 将 entities 列表中的字符串拼接为查询条件
        entities_str = ", ".join([f'"{entity}"' for entity in entities])

        # 修改查询语句，根据传入的 entities 列表查询三元组
        query = f'''
        MATCH (n)-[e1]->(o)
        WHERE id(n) IN [{entities_str}] OR id(o) IN [{entities_str}]
        RETURN n, e1, o LIMIT 30;
        '''

        result = self.session.execute(query)

        # 检查查询是否成功
        if not result.is_succeeded():
            print(f"Query failed: {result.error_msg()}")
            return []
        else:
            # 打印查询结果
            print("Query succeeded. Results:")
            
            # 用于存储三元组的列表
            triplets = []

            for row in result.rows():
                # 提取源节点信息
                node_source = row.values[0].get_vVal()  # 获取第一个 Value 对象的 Vertex
                source_name = node_source.tags[0].props[b'name'].get_sVal().decode('utf-8')  # 解码 name 属性

                # 提取关系边信息
                relationship = row.values[1].get_eVal()  # 获取第二个 Value 对象的 Edge
                relationship_name = relationship.props[b'relationship'].get_sVal().decode('utf-8')  # 解码关系属性

                # 提取目标节点信息
                node_destination = row.values[2].get_vVal()  # 获取第三个 Value 对象的 Vertex
                destination_name = node_destination.tags[0].props[b'name'].get_sVal().decode('utf-8')  # 解码 name 属性

                # 构造一个三元组字典
                # triple = {
                #     "source": source_name,
                #     "relationship": relationship_name,
                #     "destination": destination_name
                # }

                triple = f"{source_name}-{relationship_name}->{destination_name}"

                # 将三元组添加到列表中
                triplets.append(triple)
            
            return triplets

    def get_retrieve_triplets_2hop(self, space_name, entities: list[str]):
        self.session.execute(f'USE {space_name}')

        # 将 entities 列表中的字符串拼接为查询条件
        entities_str = ", ".join([f'"{entity}"' for entity in entities])

        # 修改查询语句，根据传入的 entities 列表查询三元组
        query = f'''
        MATCH (n)-[e1]->(o1)-[e2]->(o2)
        WHERE id(n) IN [{entities_str}] OR id(o2) IN [{entities_str}]
        RETURN n, e1, o1, e2, o2 LIMIT 30;
        '''

        result = self.session.execute(query)

        # 用于存储三元组的列表
        triplets = []

        # 检查查询是否成功
        if not result.is_succeeded():
            print(f"Query failed: {result.error_msg()}")
            return triplets
        else:
            # 打印查询结果
            print("Query succeeded. Results:")

            
            return triplets


from llama_index.legacy.graph_stores.nebulagraph import NebulaGraphStore
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.retrievers import (
    KnowledgeGraphRAGRetriever, )
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.utils import print_text
from llama_index.core.schema import (
    # BaseNode,
    # MetadataMode,
    NodeWithScore,
    # QueryBundle,
    TextNode,
)
import re
from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv
# from sentence_transformers import SentenceTransformer


class NebulaDB(GraphDatabase):

    def __init__(self,
                 server_url="127.0.0.1:9669",
                 server_username = "root",
                 server_password = "nebula",
                 space_name = "rgb",
                #  log_file='./database/nebula.log',
                 create=False,
                 verbose=False):
        #  verbose=False, retriever=False, llm_env=None):
        # self.log_file = log_file
        # self.server_ip = server_ip
        # self.server_port = server_port

        os.environ["NEBULA_ADDRESS"] = server_url
        os.environ["NEBULA_USER"] = server_username
        os.environ["NEBULA_PASSWORD"] = server_password  # default is "nebula"

        self.space_name = space_name
        self.edge_types = ['relationship']
        self.rel_prop_names = ['relationship']
        self.tags = ['entity']
        self.client = NebulaClient()
        self.verbose = verbose
        self.store: NebulaGraphStore = None

        try:
            self.store, self.storage_context = self.init_nebula_store()
        except Exception as e:
            print(e)
            print(
                f'please use NebulaClient().create() to create space {self.space_name}!!!\n\n\n'
            )
        self.graph_schema = self.store.get_schema(refresh=None)

        self.retriever = None

        # self.triplet2id, self.triplet_embeddings = self.generate_embedding()
        
        # self.entity2id, self.entity_embeddings = self.generate_entity_embedding()
        # self.entities = self.get_all_entities() 
        
        # Do not use get_all_dentities to obtain entities, as the modified entities are out of order
        self.entities, self.entity_embeddings = self.generate_entity_embedding_standard()

    def __del__(self):
        del self.client
        
    def init_nebula_store(self):
        nebula_store = NebulaGraphStore(
            space_name=self.space_name,
            edge_types=self.edge_types,
            rel_prop_names=self.rel_prop_names,
            tags=self.tags,
        )
        storage_context = StorageContext.from_defaults(
            graph_store=nebula_store)
        return nebula_store, storage_context

    # def init_nebula_store(self):

    #     print(self.space_name)
    #     print(self.edge_types)
    #     print(self.rel_prop_names)
    #     print(self.tags)


    #     try:
    #         nebula_store = NebulaGraphStore(
    #         space_name=self.space_name,
    #         edge_types=self.edge_types,
    #         rel_prop_names=self.rel_prop_names,
    #         tags=self.tags,
    #     )
    #     except Exception as e:
    #         print(e)
    #         nebula_store = None
        

       

    #     print('nebula_store', nebula_store)
    #     print(type(nebula_store))
    #     storage_context = StorageContext.from_defaults(
    #         graph_store=nebula_store)
    #     return nebula_store, storage_context

    def upsert_triplet(self, triplet: Tuple[str, str, str]):
        self.store.upsert_triplet(*triplet)

    def get_storage_context(self):
        return self.storage_context

    def get_space_name(self):
        return self.space_name

    def get_index(self):
        return load_index_from_storage(self.storage_context)

    def process_docs(self,
                     documents,
                     triplets_per_chunk=10,
                     include_embeddings=True,
                     data_dir='./storage_graph',
                     extract_fn=None,
                     cache=True):

        # TODO: use rebel to extract the kg elements.

        # filter documents
        # filter_documents = [doc for doc in documents if not is_file_processed(self.log_file, doc.id_)]
        # print(f'filter {len(documents) - len(filter_documents)} documents, last {len(filter_documents)} documents.')
        # documents = filter_documents

        # if len(documents) == 0:
        #     return

        # kg_index = KnowledgeGraphIndex.from_documents(
        #     documents,
        #     storage_context=self.storage_context,
        #     max_triplets_per_chunk=triplets_per_chunk,
        #     space_name=self.space_name,
        #     edge_types=self.edge_types,
        #     rel_prop_names=self.rel_prop_names,
        #     tags=self.tags,
        #     include_embeddings=True,
        #     show_progress=True,
        # )
        print(os.getcwd(), data_dir)
        index_loaded = False
        if cache:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=data_dir, graph_store=self.store)
                kg_index = load_index_from_storage(
                    storage_context=storage_context,
                    # service_context=service_context,
                    max_triplets_per_chunk=triplets_per_chunk,
                    space_name=self.space_name,
                    edge_types=self.edge_types,
                    rel_prop_names=self.rel_prop_names,
                    tags=self.tags,
                    verbose=True,
                    show_progress=True,
                )
                index_loaded = True
                print(f"graph index load from {data_dir}.")
                return kg_index
            except Exception:
                index_loaded = False

        if not index_loaded:
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                kg_triplet_extract_fn=extract_fn,
                # service_context=service_context,
                max_triplets_per_chunk=triplets_per_chunk,
                space_name=self.space_name,
                edge_types=self.edge_types,
                rel_prop_names=self.rel_prop_names,
                tags=self.tags,
                include_embeddings=True,
                show_progress=True,
            )
        if cache:
            kg_index.storage_context.persist(persist_dir=data_dir)
            print(f"kg index store to {data_dir}.")
        # for doc in documents:
        #     append_log(self.log_file, doc.id_)
        return kg_index

    def get_rel_map(self, entities, depth=2, limit=30):
        rel_map: Optional[Dict] = self.store.get_rel_map(entities,
                                                         depth=depth,
                                                         limit=limit)
        
        return rel_map
    
    def delete(self, subj: str, rel: str, obj: str):
        self.store.delete(subj, rel, obj)

    def set_retriever(self, llm_env, limit=30):
        self.retriever = KnowledgeGraphRAGRetriever(
            storage_context=self.storage_context,
            graph_traversal_depth=2,
            retriever_mode='keyword',
            verbose=False,
            entity_extract_template=llm_env.keyword_extract_prompt_template,
            synonym_expand_template=llm_env.synonym_expand_prompt_template,
            # clean_kg_sequences_fn=self.clean_kg_sequences,
            max_knowledge_sequence=limit)

    def get_entities(self, query_str: str) -> List[str]:
        """Get entities from query string."""
        return self.retriever._get_entities(query_str)

    def _get_knowledge_sequence(
            self,
            entities: List[str]) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        return self.retriever._get_knowledge_sequence(entities)

    def build_nodes(
            self,
            knowledge_sequence: List[str],
            rel_map: Optional[Dict[Any, Any]] = None) -> List[NodeWithScore]:
        if len(knowledge_sequence) == 0:
            print_text("> No knowledge sequence extracted from entities.\n",
                       color='red')
            return []

        _new_line_char = ", "
        context_string = (
            # f"The following are knowledge sequence in max depth"
            # f" {self._graph_traversal_depth} "
            # f"in the form of directed graph like:\n"
            # f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
            # f" object_next_hop ...`"
            # f" extracted based on key entities as subject:\n"
            # f"{_new_line_char.join(knowledge_sequence)}"
            f"{_new_line_char.join(knowledge_sequence)}")

        if self.verbose:
            print_text(f"Graph RAG context:\n{context_string}\n", color="blue")

        rel_node_info = {
            "kg_rel_map": rel_map,
            "kg_rel_text": knowledge_sequence,
        }
        metadata_keys = ["kg_rel_map", "kg_rel_text"]

        if self.graph_schema != "":
            rel_node_info["kg_schema"] = {"schema": self.graph_schema}
            metadata_keys.append("kg_schema")
        node = NodeWithScore(node=TextNode(
            text=context_string,
            score=1.0,
            metadata=rel_node_info,
            excluded_embed_metadata_keys=metadata_keys,
            excluded_llm_metadata_keys=metadata_keys,
        ))
        return [node]

    def get_knowledge_sequence(self, rel_map):
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend([
                str(rel_obj) for rel_objs in rel_map.values()
                for rel_obj in rel_objs
            ])
        else:
            print("> No knowledge sequence extracted from entities.")
            return []
        return knowledge_sequence

    def clean_sequence(self,
                       sequence,
                       name_pattern=r'(?<=\{name: )([^{}]+)(?=\})',
                       edge_pattern=r'(?<=\{relationship: )([^{}]+)(?=\})'):
        '''
        kg result: 'James{name: James} -[relationship:{relationship: Joined}]-> Michael jordan{name: Michael jordan}'

        clean the kg result above to James -Joined-> Michael jordan
        '''
        names = re.findall(name_pattern, sequence)
        edges = re.findall(edge_pattern, sequence)
        direction_pattern = r'(..)\[relationship:\{relationship:'
        matches = re.findall(direction_pattern, sequence)
        assert len(names) == sequence.count('{name:'), sequence
        assert len(edges) == sequence.count('{relationship:')
        assert len(edges) == len(matches)
        assert len(names) == len(edges) + 1, \
            f"The number of entities ({len (names)}) should be equal to the number of relationships ({len (edges)})+1"
        for name in names:
            sequence = sequence.replace(f'{{name: {name}}}', '')
        for edge in edges:
            sequence = sequence.replace(
                f'[relationship:{{relationship: {edge}}}]', f" {edge.replace('_',' ')} ")
        # direction_list = []    
        # for direction in matches:
        #     if direction == ' -':
        #         direction_list.append("->")
        #         # print(f"Relationship direction to the right `->`, Match to string: '{direction}'")
        #     elif direction == '<-':
        #         direction_list.append("<-")
        #         # print(f"Relationship direction to the light `<-`, Match to string: '{direction}'")
        #     else:
        #         # print(f"Unknown directional symbol: '{direction}'")
        #         assert True , f"Unknown directional symbol: '{direction}'"
        triples = []
        sentence = ""
        for i in range(len(edges)):
            if matches[i] == ' -':
                triples.append((names[i], edges[i], names[i+1], "->"))
                if not sentence:
                    sentence = sentence + names[i] + ' ' + edges[i].replace("_", " ") + ' ' +  names[i+1]
                else:
                    sentence = sentence + ", " + names[i] + ' ' + edges[i].replace("_", " ") + ' ' + names[i+1]
            elif matches[i] == '<-':
                triples.append((names[i], edges[i], names[i+1], "<-"))
                if not sentence:
                    sentence = sentence + names[i+1] + ' ' + edges[i].replace("_", " ") + ' ' + names[i]
                else:
                    sentence = sentence + ", " + names[i+1] + ' ' + edges[i].replace("_", " ") + ' ' + names[i]
            else:
                assert True , f"Unknown directional symbol: {matches[i]}"
        sentence += "."
            
        return sequence, triples, sentence

    def clean_kg_sequences(self, knowledge_sequence):
        exit(0)  # remove this function, any dependency?
        # clean_knowledge_sequence = [
        #     self.clean_sequence(seq) for seq in knowledge_sequence
        # ]
        # return clean_knowledge_sequence
        
    def remove_prefix_strings(self, strs):
        sorted_strs = sorted(strs, key=lambda x: -len(x))
        result = []
        for s in sorted_strs:
            if not any(t.startswith(s) for t in result):
                result.append(s)
        return result
    
    def clean_rel_map(self, rel_map, flag = False, keywords = []): # 加一个参数keyword
        # relmap是一个字典，没有有序没序一说，可以传入关键词控制顺序，或者返回后再按照关键词排序
        name_pattern = r'(?<=\{name: )([^{}]+)(?=\})'
        clean_rel_map = {}
        sentence_to_triple_4d = []
        sentences_2d = []
        if keywords:
            for keyword in keywords:
                # entity = f"{keyword}" + "{name: " + f"{keyword}" + "}"
                entity = f"{keyword}{{name: {keyword}}}"
                if entity in rel_map:
                    sequences = rel_map[entity]
                else:
                    continue
                name = re.findall(name_pattern, entity)[0]
                clean_ent = entity.replace(f'{{name: {name}}}', '')
                # clean_seq = [self.clean_sequence(seq) for seq in sequences]
                clean_seq = []
                sentence_to_triple = []
                sentences = []
                # sequences = self.remove_prefix_strings(sequences) # 去除已经被长字符串包含的短字符串
                if flag:
                    sequences = sorted(sequences, key=len) # 从短到长
                else:
                    sequences = self.remove_prefix_strings(sequences)
                for seq in sequences:
                    sentence_with_symbol, sentence_with_triple, sentence = self.clean_sequence(seq)
                    clean_seq.append(sentence_with_symbol)
                    sentence_to_triple.append(sentence_with_triple)
                    sentences.append(sentence)
                clean_rel_map[clean_ent] = clean_seq
                sentence_to_triple_4d.append(sentence_to_triple)
                sentences_2d.append(sentences)
        else:
            for entity, sequences in rel_map.items():
                name = re.findall(name_pattern, entity)[0]
                clean_ent = entity.replace(f'{{name: {name}}}', '')
                # clean_seq = [self.clean_sequence(seq) for seq in sequences]
                clean_seq = []
                sentence_to_triple = []
                sentences = []
                # sequences = self.remove_prefix_strings(sequences) # 去除已经被长字符串包含的短字符串
                if flag:
                    sequences = sorted(sequences, key=len) # 从短到长
                else:
                    sequences = self.remove_prefix_strings(sequences)
                for seq in sequences:
                    sentence_with_symbol, sentence_with_triple, sentence = self.clean_sequence(seq)
                    clean_seq.append(sentence_with_symbol)
                    sentence_to_triple.append(sentence_with_triple)
                    sentences.append(sentence)
                clean_rel_map[clean_ent] = clean_seq
                sentence_to_triple_4d.append(sentence_to_triple)
                sentences_2d.append(sentences)
        return clean_rel_map, sentence_to_triple_4d, sentences_2d
    
    def clean_rel_map_bak(self, rel_map, flag = False):
        name_pattern = r'(?<=\{name: )([^{}]+)(?=\})'
        clean_rel_map = {}
        sentence_to_triple_4d = []
        sentences_2d = []
        for entity, sequences in rel_map.items():
            name = re.findall(name_pattern, entity)[0]
            clean_ent = entity.replace(f'{{name: {name}}}', '')
            # clean_seq = [self.clean_sequence(seq) for seq in sequences]
            clean_seq = []
            sentence_to_triple = []
            sentences = []
            # sequences = self.remove_prefix_strings(sequences) # 去除已经被长字符串包含的短字符串
            if flag:
                sequences = sorted(sequences, key=len) # 从短到长
            else:
                sequences = self.remove_prefix_strings(sequences)
            for seq in sequences:
                sentence_with_symbol, sentence_with_triple, sentence = self.clean_sequence(seq)
                clean_seq.append(sentence_with_symbol)
                sentence_to_triple.append(sentence_with_triple)
                sentences.append(sentence)
            clean_rel_map[clean_ent] = clean_seq
            sentence_to_triple_4d.append(sentence_to_triple)
            sentences_2d.append(sentences)
        return clean_rel_map, sentence_to_triple_4d, sentences_2d

    def drop(self):
        self.client.drop_space(self.space_name)

    def info(self):
        self.client.info(self.space_name)

    def count_edges(self):
        self.client.count_edges(self.space_name)

    def show_edges(self, limits=10):
        self.client.show_edges(self.space_name, limits)

    def clear(self):
        self.client.clear(self.space_name)

    def show_space(self):
        return self.client.show_space()

    def get_triplets(self):
        return self.client.get_triplets(self.space_name)

    def save_triplets(self, file_path=None):
        self.client.save_triplets(self.space_name, file_path)

    def get_all_entities(self):
        all_triplets = self.get_triplets()

        left_entities = [triplet[0] for triplet in all_triplets]
        right_entities = [triplet[2] for triplet in all_triplets]
        entities = set(left_entities + right_entities)

        print(f'triplets: {len(all_triplets)}, entities: {len(entities)}')
        # print(list(entities)[:10])
        return entities

    def generate_embedding(self):
        # file_path = f'/home/hdd/dataset/rag-data/{self.space_name}-triplet-embedding.npz'
        # file_path = f'/home/hdd/dataset/rag-data/rgb-triplet-embedding.npz'
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-triplet-embedding.npz'

        if file_exist(file_path):
            print(f"load embedding from {file_path}")
            loaded_data = np.load(file_path, allow_pickle=True)
            triplet2id = loaded_data['triplet2id'].item()
            triplet_embeddings = loaded_data['triplet_embeddings']

            return triplet2id, triplet_embeddings

        all_triplets = self.get_triplets()
        triplet2id = {}
        all_triplets_str = []
        for i, triplet in enumerate(all_triplets):
            triplet_str = ' '.join(triplet)
            triplet2id[triplet_str] = i
            all_triplets_str.append(triplet_str)

        embed_model = EmbeddingEnv(embed_name="BAAI/bge-large-en-v1.5",# small
                                   embed_batch_size=10)

        all_embeddings = []

        step = 10
        n_triplets = len(all_triplets_str)
        for start in range(0, n_triplets, step):
            input_texts = all_triplets_str[start:min(start + step, n_triplets)]
            # print(input_texts)
            embeddings = embed_model.get_embeddings(input_texts)
            all_embeddings += embeddings
            # break

        # for i, triplet in enumerate(all_triplets_str):
        #     print(i, triplet)
        #     embedding = embed_model.get_embedding(triplet)
        #     assert np.allclose(embedding, all_embeddings[i], atol=1e-4), i

        all_embeddings_np = np.array(all_embeddings, dtype=float)

        np.savez(file_path,
                 triplet2id=triplet2id,
                 triplet_embeddings=all_embeddings_np)

        print(
            f'triplet embeddings ({all_embeddings_np.shape}) saved to {file_path}'
        )
        return triplet2id, all_embeddings_np

    # def generate_entity_embedding(self):
    #     file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding.npz'
    #     # file_path = f'/home/hdd/zhangyz/rag-data/rgb-entity-embedding.npz'

    #     if file_exist(file_path):
    #         print(f"load embedding from {file_path}")
    #         loaded_data = np.load(file_path, allow_pickle=True)
    #         entity2id = loaded_data['entity2id'].item()
    #         entity_embeddings = loaded_data['entity_embeddings']

    #         return entity2id, entity_embeddings

    #     all_entities = self.get_all_entities()
    #     # all_entities_list = [' '.join(entity) for entity in sorted(all_entities)]
    #     entity2id = {}
    #     all_entities_str = []
    #     for i, entity in enumerate(sorted(all_entities)):
    #         # entity_str = ' '.join(entity)
    #         entity_str = str(entity)
    #         entity2id[entity_str] = i
    #         all_entities_str.append(entity_str)

    #     # embed_model = EmbeddingEnv(embed_name="BAAI/bge-large-en-v1.5",
    #     #                            embed_batch_size=10)
    #     embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # device = "cuda:2"

    #     all_embeddings = []
    #     import torch
    #     step = 10
    #     n_entities = len(all_entities_str)
    #     for start in range(0, n_entities, step):
    #         input_texts = all_entities_str[start:min(start + step, n_entities)]
    #         # embeddings = embed_model.get_embeddings(input_texts)
    #         with torch.no_grad(): 
    #             embeddings = embed_model.encode(input_texts, convert_to_numpy=True)
    #         torch.cuda.empty_cache()
    #         all_embeddings.extend(embeddings)

    #     all_embeddings_np = np.array(all_embeddings, dtype=float)

    #     np.savez(file_path,
    #              entity2id=entity2id,
    #              entity_embeddings=all_embeddings_np)

    #     print(
    #         f'entity embeddings ({all_embeddings_np.shape}) saved to {file_path}'
    #     )
    #     return entity2id, all_embeddings_np
    
    def remove_entities(self, delete_entities_list):
        new_entity2id = {}
        new_embeddings = []
        new_index = 0

        # for entity, old_index in sorted(self.entity2id.items(), key=lambda x: x[1]):
        for entity, old_index in self.entity2id.items():
            if old_index < 200:
                print(f"entity2id entity: {entity}")
            if entity not in delete_entities_list:
                new_entity2id[entity] = new_index
                new_embeddings.append(self.entity_embeddings[old_index])
                new_index += 1
        self.entity2id = new_entity2id
        self.entity_embeddings = np.array(new_embeddings, dtype=float)
        
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding.npz'
        np.savez(file_path, 
                 entity2id=self.entity2id, 
                 entity_embeddings=self.entity_embeddings)
        print(f'Updated entity embeddings ({self.entity_embeddings.shape}) saved to {file_path}')
            
    def remove_entities_bak(self, delete_entities_list):
        new_entity2id = {}
        new_embeddings = []
        new_index = 0

        for entity, old_index in sorted(self.entity2id.items(), key=lambda x: x[1]):
            if old_index < 3:
                print(f"entity2id entity: {entity}")
            if entity not in delete_entities_list:
                new_entity2id[entity] = new_index
                new_embeddings.append(self.entity_embeddings[old_index])
                new_index += 1
        self.entity2id = new_entity2id
        self.entity_embeddings = np.array(new_embeddings, dtype=float)
        
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding.npz'
        np.savez(file_path, 
                 entity2id=self.entity2id, 
                 entity_embeddings=self.entity_embeddings)
        print(f'Updated entity embeddings ({self.entity_embeddings.shape}) saved to {file_path}')
    
    def generate_new_entity_embedding_standard(self, entities_new):
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard.npz'

        entities_new_np = np.array(entities_new, dtype=str)
        indices = np.searchsorted(self.entities, entities_new_np)
        selected_embeddings = self.entity_embeddings[indices]
        np.savez(file_path, entities=entities_new_np, embeddings=selected_embeddings)

        print(f'Modify Entity embeddings ({selected_embeddings.shape}) saved to {file_path}')
    
    def generate_entity_embedding_standard(self):
        # 定义文件路径
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard.npz'
        
        # 如果文件存在，则加载嵌入和实体列表
        if file_exist(file_path):
        # if os.path.exists(file_path):
            print(f"Load embedding from {file_path}")
            loaded_data = np.load(file_path, allow_pickle=True)
            entities = loaded_data['entities']  # 加载实体列表
            entity_embeddings = loaded_data['embeddings']  # 加载嵌入矩阵
            print(f"npz contain entity number '{len(entities)}' with embedding number {len(entity_embeddings)}.")
            return entities, entity_embeddings
        
        # 如果文件不存在，则生成实体和嵌入
        all_entities = self.get_all_entities()  # 假设此方法获取所有实体

        # 将实体列表转化为字符串数组
        entities = np.array([str(entity) for entity in sorted(all_entities)], dtype=str)

        # 使用嵌入模型生成嵌入
        embed_model = EmbeddingEnv(embed_name="BAAI/bge-large-en-v1.5", embed_batch_size=10)
        
        # embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # device = "cuda:2"

        all_embeddings = []
        step = 10
        n_entities = len(entities)
        
        # 批量生成嵌入
        import torch
        for start in range(0, n_entities, step):
            input_texts = entities[start:min(start + step, n_entities)]
            # embeddings = embed_model.get_embeddings(input_texts)
            with torch.no_grad(): 
                embeddings = embed_model.get_embeddings(input_texts)
                # embeddings = embed_model.encode(input_texts, convert_to_numpy=True)
            # torch.cuda.empty_cache()
            # all_embeddings += embeddings
            all_embeddings.extend(embeddings)
        
        embeddings_np = np.array(all_embeddings, dtype=float)

        # 保存实体和嵌入到文件
        np.savez(file_path, entities=entities, embeddings=embeddings_np)

        print(f'Entity embeddings ({embeddings_np.shape}) saved to {file_path}')
        
        return entities, embeddings_np

    def add_entity_embedding(self, new_entities, new_embeddings): # new_embedding type: np.array
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard.npz'

        if not self.entities or not self.entity_embeddings:
            raise FileNotFoundError(f"self.entities or self.entity_embeddings does not exist.")
        


        self.entities = np.concatenate((self._entities, new_entities))
        self.entity_embeddings = np.vstack((self.entity_embeddings, new_embeddings))

        # 更新文件
        np.savez(file_path, entities=self.entities, embeddings=self.entity_embeddings)

        print(f"Added new entity number '{len(self.entities)}' with embedding number {len(self.entity_embeddings)}.")
        return self.entities, self.entity_embeddings
    
    def entity_embedding_bak(self, iteration):
        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard_{iteration}.npz'
        np.savez(file_path, entities=self.entities, embeddings=self.entity_embeddings)
        print(f"The backup entity embeds data in path: /home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard_{iteration}.npz")

    def remove_entity_embedding(self, remove_entities, flag = False): # 添加False是为了不修改一次回写一次文件，最后统一回写

        file_path = f'/home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard.npz'
        
        if flag:
            np.savez(file_path, entities=self.entities, embeddings=self.entity_embeddings)
            print(f"The modify entity embeds data in path: /home/hdd/zhangyz/rag-data/{self.space_name}-entity-embedding-standard.npz")
            return self.entities, self.entity_embeddings
        
        # if not self.entities or not self.entity_embeddings:
        if self.entities is None or self.entity_embeddings is None:
            raise FileNotFoundError(f"self.entities or self.entity_embeddings does not exist.")

        
        # entity_index = np.where(entities == entity_str)[0][0]
        # entities = np.delete(entities, entity_index)
        # embeddings = np.delete(embeddings, entity_index, axis=0)

        remove_entities_set = set(map(str, remove_entities))
        mask = np.isin(self.entities, list(remove_entities_set), invert=True)
        self.entities = self.entities[mask]
        self.entity_embeddings = self.entity_embeddings[mask]
        

        print(f"Removed entity number '{len(remove_entities)}' and its embedding.")
        return self.entities, self.entity_embeddings

    def load_triplets_embedding(self, file_path):

        self.client.save_triplets(self.space_name, file_path)

    def execute(self, query):
        result = self.store.execute(query)
        return result

    def two_hop_parse_triplets(self, query):
        # 定义正则表达式模式
        two_hop_pattern1 = re.compile(
            r'(.+) <-(?<! )(.+?)(?<! )- (.+) -(?<! )(.+?)(?<! )-> (.+)')
        two_hop_pattern2 = re.compile(
            r'(.+) <-(?<! )(.+?)(?<! )- (.+) <-(?<! )(.+?)(?<! )- (.+)')
        two_hop_pattern3 = re.compile(
            r'(.+) -(?<! )(.+?)(?<! )-> (.+) -(?<! )(.+?)(?<! )-> (.+)')
        two_hop_pattern4 = re.compile(
            r'(.+) -(?<! )(.+?)(?<! )-> (.+) <-(?<! )(.+?)(?<! )- (.+)')

        one_hop_pattern5 = re.compile(r'(.+) -(?<! )(.+?)(?<! )-> (.+)')
        one_hop_pattern6 = re.compile(r'(.+) <-(?<! )(.+?)(?<! )- (.+)')

        match = two_hop_pattern1.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1),
                    (entity2, relation2, entity3)]

        match = two_hop_pattern2.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1),
                    (entity3, relation2, entity2)]

        match = two_hop_pattern3.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2),
                    (entity2, relation2, entity3)]

        match = two_hop_pattern4.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2),
                    (entity3, relation2, entity2)]

        match = one_hop_pattern5.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity1, relation1, entity2)]

        match = one_hop_pattern6.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity2, relation1, entity1)]

        assert False, query

    def two_hop_parse_triplets_zyz(self, query):
        # 定义正则表达式模式
        two_hop_pattern1 = re.compile(
            r'(.+) <-(?<! )(.+?)(?<! )- (.+) -(?<! )(.+?)(?<! )-> (.+)')
        two_hop_pattern2 = re.compile(
            r'(.+) <-(?<! )(.+?)(?<! )- (.+) <-(?<! )(.+?)(?<! )- (.+)')
        two_hop_pattern3 = re.compile(
            r'(.+) -(?<! )(.+?)(?<! )-> (.+) -(?<! )(.+?)(?<! )-> (.+)')
        two_hop_pattern4 = re.compile(
            r'(.+) -(?<! )(.+?)(?<! )-> (.+) <-(?<! )(.+?)(?<! )- (.+)')

        one_hop_pattern5 = re.compile(r'(.+) -(?<! )(.+?)(?<! )-> (.+)')
        one_hop_pattern6 = re.compile(r'(.+) <-(?<! )(.+?)(?<! )- (.+)')

        match = two_hop_pattern1.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1, '->'),
                    (entity2, relation2, entity3, '->')]

        match = two_hop_pattern2.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1, '->'),
                    (entity3, relation2, entity2, '->')]

        match = two_hop_pattern3.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2, '->'),
                    (entity2, relation2, entity3, '->')]

        match = two_hop_pattern4.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2, '->'),
                    (entity3, relation2, entity2, '->')]

        match = one_hop_pattern5.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity1, relation1, entity2, '->')]

        match = one_hop_pattern6.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity2, relation1, entity1, '->')]

        # assert False, query
        print(f"Error path parse[nebulagraph.py]: {query}")

    def triple_into_sentence(self, path):
        pattern = r'\s*(<-[\s\S]*?-|-[\s\S]*?->)\s*'
        split_result = re.split(pattern, path)
        nodes = split_result[::2] 
        relations = split_result[1::2]
        
        hops = []
        for i in range(len(relations)):
            if relations[i].startswith("<-"):
                relations[i] = relations[i].strip("<-").strip("-") 
                hop = f"{nodes[i+1]} {relations[i]} {nodes[i]}"
            else:
                relations[i] = relations[i].strip("->").strip("-") 
                hop = f"{nodes[i]} {relations[i]} {nodes[i+1]}"
            hops.append(hop)    
        return hops[0]

    def split_into_hops(self, path):
        import re
        # pattern = r'\s*(-[a-zA-Z_]+->|<-[a-zA-Z_]+-)\s*'
        # pattern = r'\s*(<-[a-zA-Z_-]+-|-[a-zA-Z_-]+->)\s*'
        # pattern = r'\s*(<-[\s\S]*?-|-[\s\S]*?->)\s*'
        # pattern = r'\s*( <-[^>\s]+?- | -[^<\s]+?-> )\s*'
        pattern = r'\s*( <-[^>]+?- | -[^<]+?-> )\s*'
        split_result = re.split(pattern, path)
        split_result = [part.strip() for part in split_result if part.strip()]
        # split_result = [part.strip() for part in split_result if part is not None and part.strip()]
        
        nodes = split_result[::2] 
        relations = split_result[1::2]
        
        hops = []
        for i in range(len(relations)):
            if relations[i].startswith("<-"):
                relations[i] = relations[i].strip("<-").strip("-") # 保留 - ->
                hop = f"{nodes[i+1]} -{relations[i]}-> {nodes[i]}"
            else:
                relations[i] = relations[i].strip("->").strip("-")  # 保留 - ->
                hop = f"{nodes[i]} -{relations[i]}-> {nodes[i+1]}"
            hops.append(hop)
        
        return hops

    # def get_head_entity(self, path):
    #     parts = path.split(" -")
    #     return parts[0].strip()
        
    def path_to_triples(self, paths):
        all_results = []
        for path in paths:
            all_results.extend(self.split_into_hops(path))
        unique_results = set(all_results)
        # sorted_results = sorted(unique_results, key=self.get_head_entity)
        sorted_results = sorted(unique_results)
        # for result in sorted_results:
        #     print(result)
        return sorted_results
    
    def hop_to_path(self, elements, score):
        if not elements:
            return [], []
        
        result = []
        result_score = []
        current_str = f"{elements[0]['head']} {elements[0]['relation']} {elements[0]['tail']}"
        current_tail = elements[0]['tail']
        current_score = [score[0]]
        
        for elem, score_tmp in zip(elements[1:], score[1:]):
            if elem['head'] == current_tail:
                current_str += f" {elem['relation']} {elem['tail']}"
                current_tail = elem['tail']
                current_score.append(score_tmp)
            else:
                result.append(current_str)
                result_score.append(sum(current_score)/len(current_score))
                current_str = f"{elem['head']} {elem['relation']} {elem['tail']}"
                current_score = [score_tmp]
                current_tail = elem['tail']
        
        result.append(current_str)
        result_score.append(sum(current_score)/len(current_score))
    
        return result, result_score
    
    def filter_path_by_score(self, path, score_dict):
        import re
        # pattern = r'\s*(<-[\s\S]*?-|-[\s\S]*?->)\s*'
        pattern = r'\s*( <-[^>]+?- | -[^<]+?-> )\s*'
        split_result = re.split(pattern, path)
        split_result = [part.strip() for part in split_result if part.strip()]
        # split_result = [part.strip() for part in split_result if part is not None and part.strip()]
        
        nodes = split_result[::2] 
        relations = split_result[1::2]
        
        hops = []
        score = []
        for i in range(len(relations)):
            if relations[i].startswith("<-"):
                relations[i] = relations[i].strip("<-").strip("-") 
                key = f"{nodes[i+1]} {relations[i]} {nodes[i]}"
                if key not in score_dict:
                    print(f"triple not exist: {nodes[i+1]} -{relations[i]}-> {nodes[i]}")
                if key in score_dict and score_dict[key]["score"] >= 96: # 对应迭代2*8=16分，迭代8-12次
                    tmp = {} # 头和尾并非头实体与尾实体，而是检索路径上的头和尾
                    tmp["head"] = f"{nodes[i]}"
                    tmp["tail"] = f"{nodes[i+1]}"
                    tmp["relation"] =  f"<-{relations[i]}-"
                    hops.append(tmp)
                    score.append(score_dict[key]["score"])
                    # print(f"filter function one hop score: {tmp} {score_dict[key]['score']}")
            else:
                relations[i] = relations[i].strip("->").strip("-") 
                key = f"{nodes[i]} {relations[i]} {nodes[i+1]}"
                if key not in score_dict:
                    print(f"triple not exist: {nodes[i]} -{relations[i]}-> {nodes[i+1]}")
                if key in score_dict and score_dict[key]["score"] >= 96:
                    tmp = {}
                    tmp["head"] = f"{nodes[i]}"
                    tmp["tail"] = f"{nodes[i+1]}"
                    tmp["relation"] =  f"-{relations[i]}->"
                    hops.append(tmp)
                    score.append(score_dict[key]["score"])
                    # print(f"filter function one hop score: {tmp} {score_dict[key]['score']}")
        if hops:
            return self.hop_to_path(hops, score)
        else:
            return hops, score
        
    def filter_paths_by_score(self, paths, score_dict):
        filter_list = []
        filter_list_score = []
        for path in paths:
            res, score = self.filter_path_by_score(path, score_dict)
            filter_list += res
            filter_list_score += score
        # for a,b in zip(filter_list,filter_list_score):
        #     print(f"triple with score: {a}: {b}")
        return filter_list, filter_list_score
    
    def filter_paths_by_score_v2(self, triples_3d, score_dict): # path中出现分数小于阈值的三元组，这个path直接不要了
        score_threshold = 95
        filter_list = []
        filter_list_score = []
        for i, triples_2d in enumerate(triples_3d):
            score = 0
            count = 0
            triples_tmp = []
            flag = 0
            for triple in triples_2d:
                if not (len(triple) == 4):
                    print(f"The length of the triple is not 4: {triple}")
                    continue
                if triple[3] == "->":
                    key = f"{triple[0]} {triple[1]} {triple[2]}"
                elif triple[3] == "<-":
                    key = f"{triple[2]} {triple[1]} {triple[0]}"
                else:
                    print(f"The appearance of position and direction symbols in triple: {triple}")
                    continue
                if key not in score_dict:
                    print(f"triple not exist: {triple}")
                    continue
                if key in score_dict and score_dict[key]["score"] < score_threshold:
                    flag = 1
                    break
                if key in score_dict and score_dict[key]["score"] >= score_threshold:
                    score += score_dict[key]["score"]
                    count += 1
                    triples_tmp.append(triple)
            if flag:
                continue
            else:
                filter_list.append(triples_tmp)
            if count == 0:
                filter_list_score.append(0)
            else:
                filter_list_score.append(score/count)
            
        return filter_list, filter_list_score
        

    def split_into_hops_for_redundant_relationship(self, path):
        import re
        pattern = r'\s*( <-[^>]+?- | -[^<]+?-> )\s*'
        split_result = re.split(pattern, path)
        split_result = [part.strip() for part in split_result if part.strip()]
        
        nodes = split_result[::2] 
        relations = split_result[1::2]
        
        hops = []
        for i in range(len(relations)):
            hop_list = []
            if relations[i].startswith("<-"):
                relation = relations[i].strip("<-").strip("-")
                relation_no_underline = relation.replace("_", " ")
                hop = f"{nodes[i+1]} {relation_no_underline} {nodes[i]}" # 去掉 - ->  与下划线
                hop2 = f"{nodes[i+1]} -{relation}-> {nodes[i]}"
                hop_list.append(nodes[i+1])
                hop_list.append(nodes[i])
                hop_list.append(relation)
                hop_list.append(hop)
                hop_list.append(hop2)
            else:
                relation = relations[i].strip("->").strip("-")
                relation_no_underline = relation.replace("_", " ")
                hop = f"{nodes[i]} {relation_no_underline} {nodes[i+1]}" # 去掉 - ->   与下划线
                hop2 = f"{nodes[i]} -{relation}-> {nodes[i+1]}"
                hop_list.append(nodes[i])
                hop_list.append(nodes[i+1])
                hop_list.append(relation)
                hop_list.append(hop)
                hop_list.append(hop2)
            if hop_list:
                hops.append(hop_list)
        
        return hops
    # 未测试
    def path_to_triples_grouped(self, paths): # find_redundant_relationship
        group_dict = {}
        group_dict_copy = {}
        for path in paths:
            hops = self.split_into_hops_for_redundant_relationship(path)
            for hop in hops:
                key = (hop[0], hop[1])
                if key not in group_dict:
                    group_dict[key] = []
                    group_dict_copy[key] = []
                if hop[3] not in group_dict[key]:
                    group_dict[key].append(hop[3])
                    group_dict_copy[key].append(hop[4])

        group_list = list(group_dict.values())
        group_list_copy = list(group_dict_copy.values())
        # result = [tmp for tmp in group_list if len(tmp) > 1]
        result = [sublist for sublist in group_list if len(sublist) > 1]
        result_copy = [sublist for sublist in group_list_copy if len(sublist) > 1]
        return result, result_copy
    # 未测试
    def parse_keep_relationship(self,response_for_redundant_relationship):
        keep_id_list = []
        relationship_list =  response_for_redundant_relationship.strip().split('\n')
        for ids in relationship_list:
            keep_id_list.append([id.strip() for id in ids.strip().split(",")])
        return keep_id_list, len(relationship_list)

    def process_redundant_relationship(self, response_for_redundant_relationship, redundant_relationship, TF):
        keep_relationship, group_num = self.parse_keep_relationship(response_for_redundant_relationship)
        if group_num != len(redundant_relationship):
            return "For the processing of redundant relations, if the number of answer groups is different from the question array, skip the processing"
        redundant_relationship_list = [item for sublist in redundant_relationship for item in sublist]
        keep_relationship_list = [item for sublist in keep_relationship for item in sublist]
        log_keep = ""
        delele_list = []
        for idx, relationship in enumerate(redundant_relationship_list, start= 1): # 是否要捕捉错误？？？
            triple_item = self.split_into_hops_for_redundant_relationship(relationship)[0]
            if str(idx) in keep_relationship_list:
                log_keep = log_keep + '\n' + triple_item[3]
            else:
                if TF:
                    self.delete(triple_item[0], triple_item[2], triple_item[1])
                delele_list.append(triple_item[3])
                pass
        return log_keep, delele_list
    
    def process_redundant_relationship_v2(self, parsed_response_for_relationship, redundant_relationship_3d, TF):
        redundant_relationship_list = [item for sublist in redundant_relationship_3d for item in sublist]
        keep_id_list = [item for sublist in parsed_response_for_relationship for item in sublist]
        # keep_id_list = []
        # for sublist in parsed_response_for_relationship:
        #     if len(sublist) > 1:
        #         keep_id_list.extend(sublist[1:])
        # print(f"process_redundant_relationship_v2 111")
        keep_str = ""
        delele_list = []
        for idx, triple in enumerate(redundant_relationship_list, start= 0): # 是否要捕捉错误？？？
            if idx in keep_id_list:
                keep_str = keep_str + '\n' + str(triple)
            else:
                if TF:
                    # print(f"process_redundant_relationship_v2 222")
                    # print(f"{triple[0]}, {triple[1]}, {triple[2]}")
                    self.delete(triple[0], triple[1], triple[2])
                delele_list.append(triple)
                # pass
        return keep_str, delele_list
    
    def process_redundant_entity(self, parsed_response_for_entity, redundant_entity_2d, TF):
        redundant_entity_list = [item for sublist in redundant_entity_2d for item in sublist]
        keep_and_delete_str = ""
        delete_entity = []
        insert_relationship_list_2d = []
        delete_relationship_list_2d = []
        insert_relationship_all = []
        delete_relationship_all = []
        keep_and_delete_entity_list_2d = []
        # print(f"parsed_response_for_entity : {json.dumps(parsed_response_for_entity, indent=2)}")
        for group in parsed_response_for_entity: 
            similar_entity = [ redundant_entity_list[int(i)] for i in group ]
            keep_and_delete_entity_list_2d.append(similar_entity)
            keep_and_delete_str += str(similar_entity)
            # 查一跳邻居、将这些关系插入第一个实体、删除原有的关系
            insert_relationship = []
            delete_relationship = []
            entity_keep = similar_entity[0]
            if len(similar_entity) <= 1:
                continue
            delete_entity.extend(similar_entity[1:])
            rel_map = self.get_rel_map(similar_entity[1:], depth=1, limit=100000)
            clean_real_map, sentence_to_triple_4d, sentences_2d = self.clean_rel_map(rel_map)
            for list_3d in sentence_to_triple_4d:
                for sentence_to_triple_2d in list_3d:
                    for sentence_to_triple in sentence_to_triple_2d:
                        if sentence_to_triple[3] == '->':
                            insert_relationship_all.append(f"{entity_keep} {sentence_to_triple[1]} {sentence_to_triple[2]}")
                            insert_relationship.append([entity_keep, sentence_to_triple[1], sentence_to_triple[2]])
                            delete_relationship.append(f"{sentence_to_triple[0]} {sentence_to_triple[1]} {sentence_to_triple[2]}")
                            delete_relationship_list_2d.append([sentence_to_triple[0], sentence_to_triple[1], sentence_to_triple[2]])
                            if TF:
                                self.delete(sentence_to_triple[0], sentence_to_triple[1], sentence_to_triple[2])
                        elif sentence_to_triple[3] == '<-':
                            insert_relationship_all.append(f"{sentence_to_triple[2]} {sentence_to_triple[1]} {entity_keep}")
                            insert_relationship.append([sentence_to_triple[2], sentence_to_triple[1], entity_keep])                            
                            delete_relationship.append(f"{sentence_to_triple[2]} {sentence_to_triple[1]} {sentence_to_triple[0]}")
                            delete_relationship_list_2d.append([sentence_to_triple[2], sentence_to_triple[1], sentence_to_triple[0]])
                            if TF:
                                self.delete(sentence_to_triple[2], sentence_to_triple[1], sentence_to_triple[0])  
            if TF:
                for triple in insert_relationship:
                    self.upsert_triplet([triple[0], triple[1], triple[2]]) 
            # insert_relationship_all.extend(insert_relationship)
            insert_relationship_list_2d.extend(insert_relationship)
            delete_relationship_all.extend(delete_relationship)
                
        ### 删除实体与对应的嵌入，并保存到文件中，但不重新读取 (本轮迭代结束统一删除)   remove_entities                  
        # keep_and_delete_str 保存与删除的实体, delete_entity 删除的实体, insert_relationship_list_2d 插入关系三元组, delete_relationship_list_2d 删除关系三元组, insert_relationship_all 插入关系句子, delete_relationship_all 删除关系句子                                           
        return keep_and_delete_str, delete_entity, insert_relationship_list_2d, delete_relationship_list_2d, insert_relationship_all, delete_relationship_all, keep_and_delete_entity_list_2d

    def rel_map_to_triplets(self, clean_map):
        all_triplets = set()
        for rels in clean_map.values():
            triplets, _ = self.two_hop_parse_multi_triplets(rels)
            all_triplets.update(triplets)
        return all_triplets

    def kg_seqs_to_triplets(self, kg_seqs):
        all_triplets = []
        for rel in kg_seqs:
            for triplet in self.two_hop_parse_triplets(rel):
                all_triplets.append(triplet)
        all_triplets = set(all_triplets)

        return all_triplets

    def kg_seqs_to_triplets_for_ragcache(self, kg_seqs): # 处理两跳以内
        all_triplets = []
        for rel in kg_seqs:
            # for triplet in self.two_hop_parse_triplets_zyz(rel.replace('\n','')):
            #     all_triplets.append(triplet)
            all_triplets.append(self.two_hop_parse_triplets_zyz(rel.replace('\n','')))
        # all_triplets = set(all_triplets)

        return all_triplets

    def two_hop_parse_multi_triplets(self, queries):
        triplets = []
        rel_to_entities = {}
        for query in queries:
            query_triplets = self.two_hop_parse_triplets(query)
            triplets += query_triplets
            if query not in rel_to_entities:
                rel_to_entities[query] = set()
            for triplet in query_triplets:
                rel_to_entities[query].add(triplet[0])
                rel_to_entities[query].add(triplet[2])
        return triplets, rel_to_entities



if __name__ == '__main__':
    # export PYTHONPATH=/home/zhangyz/KGModify/RAGWebUi_demo:$PYTHONPATH
    # space_name = 'hotpotqa'
    space_name = 'multihop_zyz'
    client = NebulaClient()
    client.show_space()
    client.info(space_name)
    db = NebulaDB(space_name = space_name)
    
    # db.clear()
    
    # rel_map = db.get_rel_map(['Truth social'], depth=4,limit=30000)
    # print("rel_map",db.clean_rel_map(rel_map))

    # res = db.delete("Mook animation", "Animated", "")
    # print(f"delete res: {res}")
    
    # res = db.upsert_triplet(["Zyz", "Test", "Nebulagraph upsert triplet"])
    # print(f"upsert triplet res: {res}")
    
    # ['Carole king & james taylor: just call out my name', 'Carole king and james taylor: just call out my name', 'Aug.', 'August', 'U.s. open women’s singles draw', "Us open women's singles"]
    rel_map = db.get_rel_map(['$1000'], depth=1, limit=1000000)
    print("rel_map", json.dumps(rel_map, indent=2))
    clean_rel_map, sentence_to_triple_4d, sentences_2d = db.clean_rel_map(rel_map)
    print("clean_rel_map", json.dumps(clean_rel_map, indent=2))
    # print("sentence_to_triple_4d", json.dumps(sentence_to_triple_4d, indent=2))
    # print("sentences_2d", json.dumps(sentences_2d, indent=2))
    # print("clean_rel_map",db.clean_rel_map(rel_map))
    
    
    # client.create_space("rgb_zyz2")

    print("\n\n\n finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
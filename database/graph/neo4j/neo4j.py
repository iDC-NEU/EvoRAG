'''
Author: fzb fzb0316@163.com
Date: 2024-09-18 17:26:28
LastEditors: fzb fzb0316@163.com
LastEditTime: 2024-09-20 14:12:30
FilePath: /RAGWebUi_demo/database/graph/graph_dbfactory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


from datetime import datetime
from pathlib import Path
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher, ConnectionUnavailable
from overrides import override

from logger import Logger
from database.graph.graph_database import GraphDatabase


# Neo4j图数据库的支持，包含图数据库的连接、增删改查等方法
class MyNeo4j(GraphDatabase):
    """
    连接图数据库neo4j, 并设置label(ENTITY)中, name具有唯一性约束
    triple = {
            "source": "bob",
            "relationship": "know",
            "destination" : "Alice"
        }
    """

    def __init__(self, url : str, usrname : str, pd : str):

        
        self._logger: Logger = Logger(Path(__file__).name)
        self.baseurl = url
        self.username = usrname
        self.password = pd

        self.connect_graphdb()

        self.node_matcher = NodeMatcher(self.graph) if self.graph else None
        self.relationship_matcher = RelationshipMatcher(self.graph) if  self.graph else None

        self.label = "ENTITY"
        self.relation = "REALTION"

        self.graph.run(f"CREATE CONSTRAINT unique_{self.label}_name IF NOT EXISTS FOR (p:{self.label}) REQUIRE p.name IS UNIQUE;")
    
    @staticmethod
    def ensure_connection(function):
        def wrapper(*args, **kwargs):
            if not args[0].graph:
                args[0]._logger.warning("Graph database connection is not available.")
                return None
            return function(*args, **kwargs)

        return wrapper

    @override
    def connect_graphdb(self):
        try:
            self.graph = Graph(self.baseurl, auth=(self.username, self.password))
        except ConnectionUnavailable as e:
            self._logger.warning(f"Failed to connect to Neo4j graph database: {e}")
            self.graph = None

    @ensure_connection
    def create_node(self, label, properties):
        node = Node(label, **properties)
        self.graph.create(node)
        print(f" successfully create node: {node}")
    
    @ensure_connection
    def query_all_nodes(self):
        all_nodes = self.node_matcher.match()
        if all_nodes:
            print("All node are as follow:")
            for node in all_nodes:
                print(node)
        else:
            print("Database no node!")


    @ensure_connection
    def query_node(self, label, properties, limit = 10):
        nodes = self.node_matcher.match(label, **properties).limit(limit)
        if nodes:
            print(f"All node by {label} {properties} when limit({limit}) are as follow:")
            for node in nodes:
                print(node)
        else:
            print(f"No node: {label} {properties}")

    @ensure_connection
    def delete_node(self, label, properties):
        nodes = self.node_matcher.match(label, **properties)
        if nodes:
            for node in nodes:
                self.graph.delete(node)
                print(f"Node {node} deleted.")
        else:
            print("Node not found")

    @ensure_connection
    def query_all_relationships(self):
        rels = self.relationship_matcher.match()

        if rels:
            print("All relationships are as follow:")
            for rel in rels:
                print(rel)
        else:
            print("Database no relationship!")

    @ensure_connection
    def find_or_create_node(self, label, identifier) -> Node:
        # 查找节点
        node = self.node_matcher.match(label, name=identifier).first()
        
        if not node:
            # 节点不存在，创建节点
            node = Node(label, name=identifier)
            self.graph.create(node)
        return node
    
    @ensure_connection
    def query_relationship(self, triple : dict):
        self.node1_id = triple["source"]
        self.node2_id = triple["destination"]
        self.relationship_type = triple["relationship"]

        self.node1 = self.find_or_create_node(self.label, self.node1_id)
        self.node2 = self.find_or_create_node(self.label, self.node2_id)

        existing_relationship = self.relationship_matcher.match(nodes=(self.node1, self.node2), r_type=self.relationship_type).first()
        # existing_relationship = self.relationship_matcher.match(nodes=(self.node1, self.node2)).first()

        return existing_relationship
    
    
    @ensure_connection
    def create_relationship(self, triple : list[dict]):
        for triple_one in triple:
            existing_relationship = self.query_relationship(triple_one)

            if existing_relationship is None:
                # 创建关系
                relationship = Relationship(self.node1, self.relationship_type, self.node2)
                self.graph.create(relationship)
                print(f"Created relationship '{self.relationship_type}' between {self.node1_id} and {self.node2_id}.")
            else:
                print(f"Relationship '{self.relationship_type}' already exists between {self.node1_id} and {self.node2_id}.")

    @ensure_connection
    def delete_relationship(self, triple : list[dict]):
        for triple_one in triple: 
            existing_relationship = self.query_relationship(triple_one)

            # 如果找到了关系，删除它
            if existing_relationship is not None:
                self.graph.separate(existing_relationship)
                if self.relationship_matcher.match((existing_relationship.start_node, None)).first() is None \
                            and self.relationship_matcher.match((None, existing_relationship.start_node)).first() is None:
                    self.graph.delete(existing_relationship.start_node)
                if self.relationship_matcher.match((existing_relationship.end_node, None)).first() is None \
                            and self.relationship_matcher.match((None, existing_relationship.end_node)).first() is None:
                    self.graph.delete(existing_relationship.end_node)
                print("Relationship deleted.")
            else:
                print("No relationship found.")
    
    def query_all(self):
        self.query_all_nodes()
        self.query_all_relationships()


    











    
        
        
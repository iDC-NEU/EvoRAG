'''
Author: fzb fzb0316@163.com
Date: 2024-09-18 17:26:28
LastEditors: fzb0316 fzb0316@163.com
LastEditTime: 2024-10-24 19:31:59
FilePath: /RAGWebUi_demo/database/graph/graph_dbfactory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



from config.config import Config
from utils.url_paser import is_valid_url
from database.database_error import DatabaseUrlFormatError, DatabaseAPIUnsupportedError
from database.graph.graph_database import GraphDatabase
from database.graph.neo4j.neo4j import MyNeo4j
from database.graph.nebulagraph.nebulagraph import NebulaDB


NEO4J = "neo4j"
NEBULA = "nebulagraph"

class GraphDBFactory(object):

    def __init__(self, DBName : str):

        self.dbname = DBName

        self.dburl = Config.get_instance().get_with_nested_params("database", f"{DBName}", "url")
        self.dbusrname = Config.get_instance().get_with_nested_params("database", f"{DBName}", "username")
        self.dbpasswd = Config.get_instance().get_with_nested_params("database", f"{DBName}", "password")
        # self._sanity_check()

    
    def _sanity_check(self):
        if not is_valid_url(self.dburl):
            raise DatabaseUrlFormatError("client url provided is not a url string")
        

    def get_graphdb(self, space_name : str) -> GraphDatabase:
        if self.dbname == NEO4J:
            return MyNeo4j(self.dburl, self.dbusrname, self.dbpasswd)
        elif self.dbname == NEBULA:
            return NebulaDB(self.dburl, self.dbusrname, self.dbpasswd, space_name)

        else:
            raise DatabaseAPIUnsupportedError(self.dbname)
        


        
if __name__ == '__main__':

    '''
    export PYTHONPATH=/home/lipz/fuzb_rag_demo/RAGWebUi_demo/:$PYTHONPATH
    python database/graph/graph_dbfactory.py
    '''

    graphdb = GraphDBFactory("nebulagraph").get_graphdb()
    # triple = [
    #     {"source": "bob", "relationship": "Know", "destination" : "Tom"},
    #     {"source": "bob", "relationship": "Love", "destination" : "Alice"},
    #     {"source": "bob", "relationship": "Know", "destination" : "Andy"},
    #     {"source": "bob", "relationship": "Know", "destination" : "Red"},
    # ]

    # result_json = triples_to_json(triple)
    # print(result_json)

    graphdb.show_space()
    graphdb.show_edges("rgb",limits= 30)
    
    graphdb.get_retrieve_triplets("rgb", ["Apple"])




    
    # graphdb.create_node("Person", properties)
    
    # graphdb.query_all_nodes()
    # graphdb.query_node("Person", properties, 15)
    # graphdb.delete_node("ENTITY", {"name" : "Tom"})
    # graphdb.create_relationship(triple)
    # graphdb.delete_relationship(triple)



from abc import abstractmethod





class VectorDatabase(object):

    @abstractmethod
    def connect_graphdb(self):
        raise NotImplementedError()

import numpy as np
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class Document:
    name: str
    embedding: np.array

    def __str__(self):
        return f"doc '{self.name}'"
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

@dataclass
class Query:
    name: str
    embedding: np.array
    gold_doc: Document

    def __str__(self):
        return f"query '{self.name}'"
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"

class QuerySearch:
    """
    A class representing a Query.

    Attributes:
        name (str): Identification of the query.
        embedding (np.array): The pre-computed embedding of the document.
        messages (list[MessageQuery]): List of all messages related to the query. Useful to know the simulation status.
    """

    def __init__(self, query: Query):
        """
        Creates a Query.

        Arguments:
            name (str): Identification of the query.
            embedding (np.array): The pre-computed embedding of the document.
        """
        self.search_id = uuid4()

        self.query = query
        self.messages = []

    def spawn_message(self, ttl):
        """
        Spawns a message object carrying a query.

        Arguments:
            ttl (int): Time to live field.
        """
        message = QueryMessage(query = self.query,
                               ttl=ttl,
                               search_id=self.search_id,
                               register_message_callback=self.register)
        self.register(message)
        return message
    
    def register(self, message):
        self.messages.append(message)

    @property
    def candidate_doc(self):
        """
        Determines the best candidate document for retrieval accross all messages spawned by the same query.

        Returns:
            Document: The candidate document.
        """
        candidate_doc, max_similarity = None, -float("inf")
        for message in self.messages:
            if message.candidate_doc_similarity > max_similarity:
                max_similarity = message.candidate_doc_similarity
                candidate_doc = message.candidate_doc
        return candidate_doc

    @property
    def hops_to_reach_candidate_doc(self):
        """
        Determines the hop count until the best candidate document for retrieval accross all messages spawned by the same query.

        Returns:
            int: The hop count to the candidate document.
        """
        candidate_hops, max_similarity = None, -float("inf")
        for message in self.messages:
            if message.candidate_doc_similarity > max_similarity:
                max_similarity = message.candidate_doc_similarity
                candidate_hops = message.hops_to_reach_doc
        return candidate_hops

    @property
    def visited_tree(self):
        """
        Determines the nodes traversed by all messages spawned by the same query.

        Returns:
            list[Node]: A list of all traversed nodes.
        """
        tree = set()
        for message in self.messages:
            tree.update(set(message.visited_edges))
        return list(tree)

    def __str__(self):
        return f"search '{self.search_id}' ({self.query}, {len(self.messages)} spawned message{'s' if len(self.messages)!=1 else ''})"
    
    def __repr__(self):
        return f"{self.__class__.__name__} ('{self.search_id}')"


class QueryMessage:
    """
    A class representing a message carrying a query.

    Although node embeddings are also diffused via message passing,
    message in the code refers to query messages unless specified otherwise.

    Parameters:
        query (Query): The query carried by the message.
        ttl (int): The time-to-live field of the message.
        name (string): Identification of the message. If not given, creates a unique name with the help of an internal counter.
        hops (int): The hops that the message has travelled so far.
        hops_to_reach_doc (int): The hops to reach the candidate document for this message.
        candidate_doc (Document): The best candidate document for this query message.
        candidate_doc_similarity (float): The score of the best candidate document for this query message.
        visited_edges (list[tuple]): The edges visited by this query message in correct order.
    """

    counter = 0

    def __init__(self, query, ttl, search_id, register_message_callback):
        """
        Constructs a QueryMessage.

        Arguments:
            query (Query): The query carried by the message.
            ttl (int): The time-to-live field of the message.
            name (string): Identification of the message. If None, creates a unique name with the help of an internal counter.
        """
        self.name = f"qm{self.__class__.counter}"
        self.__class__.counter += 1
        
        self.query = query
        self.ttl = ttl
        self.search_id = search_id
        self.register_message_callback = register_message_callback
        
        self.hops = 0
        self.hops_to_reach_doc = 0
        self.candidate_doc = None
        self.candidate_doc_similarity = -float("inf")
        self.visited_edges = []

    @property
    def embedding(self):
        """
        The embedding of the query carried by the message.
        """
        return self.query.embedding
    
    @property
    def visited_nodes(self):
        """
        A list of nodes visited by the query message in correct order.
        """
        if len(self.visited_edges) == 0:
            return []
        nodes = [edge[0] for edge in self.visited_edges]
        nodes.append(self.visited_edges[-1][1])
        return nodes

    def is_alive(self):
        """
        Checks if the time-to-live field has not expired.
        """
        return self.hops < self.ttl

    def clone(self):
        copy = QueryMessage(self.query, self.ttl, self.search_id, self.register_message_callback)
        copy.hops = self.hops
        copy.hops_to_reach_doc = self.hops_to_reach_doc
        copy.candidate_doc = self.candidate_doc
        copy.candidate_doc_similarity = self.candidate_doc_similarity
        copy.visited_edges = [edge for edge in self.visited_edges] # should it start anew?
        self.register_message_callback(copy)
        return copy

    def send(self, from_node, to_node):
        """
        Updates message parameters when a message is transferred from a sending to a receiving node.
        Does *not* actually transfer the message.

        Returns:
            QueryMessage: The query message itself.
        """
        self.hops += 1
        self.visited_edges.append((from_node.name, to_node.name))
        return self

    def receive(self, other):
        """
        Merges messages of the same query that arrive at the same node.
        Triggers updating the candidate documents.
        """
        assert self.name == other.name
        if other.candidate_doc_similarity > self.candidate_doc_similarity:
            self.candidate_doc = other.candidate_doc
            self.hops_to_reach_doc = other.hops_to_reach_doc
            self.candidate_doc_similarity = other.candidate_doc_similarity
        self.hops = max(self.hops, other.hops)  # min could be another option

    def retrieve(self, docs):
        """
        Executes the message query against a collection of documents. 
        Triggers updating the candidate document.This function is delegated to the MessageQuery
        instead of the Node object to contain all retrieval operations.

        Arguments:
            docs (Iterable[Document]): A collection of documents.
        """
        for doc in docs:
            score = np.sum(doc.embedding * self.embedding)
            # score = -np.linalg.norm(docs[doc].embedding - self.embedding)
            if score > self.candidate_doc_similarity:
                self.candidate_doc_similarity = score
                self.candidate_doc = doc
                self.hops_to_reach_doc = self.hops

    def __str__(self):
        return f"message '{self.name}' (search '{self.search_id}', {self.query}, {self.ttl-self.hops} hops remaining)"
    
    def __repr__(self):
        return f"{self.__class__.__name__} ('{self.name}', {self.ttl-self.hops} hops remaining)"

if __name__ == "__main__":
    dim = 5
    ttl = 4
    doc1 = Document("doc1", np.random.random(dim))
    doc2 = Document("doc2", np.random.random(dim))
    doc3 = Document("doc3", np.random.random(dim))
    docs = [doc1, doc2, doc3]

    query = Query("que0", np.random.random(dim), None)

    node1 = Document("node1", None)
    node2 = Document("node2", None)
    node3 = Document("node3", None)
    node4 = Document("node4", None)

    search = QuerySearch(query)
    message = search.spawn_message(ttl)
    print(search)

    message.send(from_node=node1, to_node=node3)
    message.send(from_node=node3, to_node=node2)
    message.send(from_node=node2, to_node=node1)
    message.send(from_node=node1, to_node=node2)
    message.send(from_node=node2, to_node=node3)
    message.send(from_node=node3, to_node=node2)
 
    print(message.visited_nodes)
    
    print(message.visited_edges)
    print(search.visited_tree)




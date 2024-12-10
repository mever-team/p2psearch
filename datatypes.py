import numpy as np


class Document:
    """
    A class representing a document.

    Attributes:
        name (str): Identification of the document. Should be unique.
        embedding (np.array): The pre-computed embedding of the document.
    """

    def __init__(self, name, embedding):
        """
        Constructs a Document.
        """
        self.name = name
        self.embedding = embedding

    def __repr__(self):
        return f"{self.__class__.__name__} ('{self.name}')"


class Query:
    """
    A class representing a Query.

    Attributes:
        name (str): Identification of the query.
        embedding (np.array): The pre-computed embedding of the document.
        messages (list[MessageQuery]): List of all messages related to the query. Useful to know the simulation status.
    """

    def __init__(self, name, embedding, _gold_doc=None):
        """
        Creates a Query.
        """
        self.name = name
        self.embedding = embedding
        self.messages = []
        self._gold_doc = _gold_doc

    def spawn(self, ttl):
        """
        Spawns a message object carrying a query.

        Arguments:
            ttl (int): Time to live field.
        """
        message = MessageQuery(self, ttl)
        return message

    def register(self, message):
        # TODO: does it need to be a separate function? why not through spawn??
        """
        Registers a message to the query.

        Arguments:
            message (MessageQuery): A message related to the query.
        """
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

    def __repr__(self):
        return f"{self.__class__.__name__} ('{self.name}')"


class MessageQuery:
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

    def __init__(self, query, ttl, name=None):
        """
        Constructs a QueryMessage.

        Arguments:
            query (Query): The query carried by the message.
            ttl (int): The time-to-live field of the message.
            name (string): Identification of the message. If None, creates a unique name with the help of an internal counter.
        """
        if name is None:
            name = f"qm{self.__class__.counter}({query.name})"
            self.__class__.counter += 1
        self.name = name
        # refers to all messages cloned from the same initial message query
        # messages added to different nodes will have diffrent message_names EVEN if they point to the same query obj

        self.query = query
        self.ttl = ttl
        self.hops = 0
        self.hops_to_reach_doc = 0
        self.candidate_doc = None
        self.candidate_doc_similarity = -float("inf")
        self.visited_edges = []

        # notify original query so that self can be monitored
        self.query.register(self)

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

    @property
    def query_name(self):
        # TODO: is it used anywhere???
        """
        The name of the query, distinct from the name identifying the message.
        """
        return self.query.name

    @property
    def embedding(self):
        """
        The embedding of the query carried by the message.
        """
        return self.query.embedding

    def is_alive(self):
        """
        Checks if the time-to-live field has not expired.
        """
        return self.hops < self.ttl

    def kill(self, at_node, reason=""):
        # TODO: is it used anywhere???
        pass
        # print(f"Query {self.query.name} died at node {at_node.name} because {reason}")
        # TODO notify query

    def clone(self):
        # TODO is it used anywhere??? or message are only created via spawn?
        # does not register the message
        copy = MessageQuery(self.query, self.ttl, name=self.name)
        copy.hops = self.hops
        copy.hops_to_reach_doc = self.hops_to_reach_doc
        copy.candidate_doc = self.candidate_doc
        copy.candidate_doc_similarity = self.candidate_doc_similarity
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

    def check_now(self, docs):
        """
        Executes the message query against a collection of documents. 
        Triggers updating the candidate document.This function is delegated to the MessageQuery
        instead of the Node object to contain all retrieval operations.

        Arguments:
            docs (Iterable[Document]): A collection of documents.
        """
        for doc in docs:
            score = np.sum(docs[doc].embedding * self.embedding)
            # score = -np.linalg.norm(docs[doc].embedding - self.embedding)
            if score > self.candidate_doc_similarity:
                self.candidate_doc_similarity = score
                self.candidate_doc = doc
                self.hops_to_reach_doc = self.hops

    def __repr__(self):
        return f"{self.__class__.__name__} ('{self.name}', {self.ttl-self.hops} hops remaining)"

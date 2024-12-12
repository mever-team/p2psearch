import numpy as np

from uuid import uuid4
from typing import Sequence
from dataclasses import dataclass

@dataclass
class Document:
    """
    A class representing a document from an information retrieval dataset.

    Attributes:
        name (str): The name of the document, taken from a retrieval dataset. Should be unique.
        embedding (np.array): A pre-computed embedding vector for the document.
    """

    name: str
    embedding: np.array

    def __str__(self):
        return f"document '{self.name}'"

    def __repr__(self):
        return f"document '{self.name}'"


@dataclass
class Query:
    """
    A class representing a query from an information retrieval dataset.

    Attributes:
        name (str): The name of the query, taken from a retrieval dataset. Should be unique.
        embedding (np.array): A pre-computed embedding vector for the query.
    """

    name: str
    embedding: np.array
    # gold_doc: Document # could be useful

    def __str__(self):
        return f"query '{self.name}'"

    def __repr__(self):
        return f"query '{self.name}'"


class QuerySearch:
    """
    A class representing the search for a document inside a P2P network.

    The search object contains one query and can spawn multiple messages
    inside the network. It also contains utility functions to monitor these
    messages and determine the best document for retrieval at any time.
    This is for convenience, as in practice the retrieval results would be
    collected and aggregated at the querying node when all associated messages
    finish their walks.

    Note that search objects and their messages are unique but they can carry
    the same query. They are treated independently by the network.

    Attributes:
        search_id (str): A unique field identifying the search. Nodes can
            determine which messages belong to the same search through this field.
        query (Query): The query object associated with the search.
        messages (list[QueryMessage]): A list of all messages spawned by this search.
            Useful to centrally monitor all messages associated with this search.
            The list also tracks messages that are cloned during the simulation
            via a callback.
    """

    def __init__(self, query: Query):
        """
        Creates a search object.

        Arguments:
            query (Query): A query object.
        """
        self.search_id = str(uuid4())

        self.query = query
        self.messages = []

    def spawn_message(self, ttl: int):
        """
        Spawns a message carrying a query.

        Arguments:
            ttl (int): Time-to-live field.

        Returns:
            QueryMessage: A message carrying the query associated with the search.
        """
        message = QueryMessage(
            query=self.query,
            ttl=ttl,
            search_id=self.search_id,
            register_message_callback=self.register,
        )
        self.register(message)
        return message

    def register(self, message):
        """
        Registers a message to be tracked by the search object.

        Arguments:
            message (QueryMessage): A message associated with this search.
        """
        self.messages.append(message)

    @property
    def candidate_doc(self):
        """
        Computes the best candidate document for retrieval across all messages
        spawned by this search.

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
        Computes the hop count until the best candidate document for retrieval
        across all messages spawned by this search.

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
        Computes the edges travelled by all messages spawned by this search.

        Returns:
            list[Node]: A list of all travelled edges (without duplicates).
        """
        tree = set()
        for message in self.messages:
            tree.update(set(message.visited_edges))
        return list(tree)

    def __str__(self):
        return f"search '{self.search_id}' ({self.query})"

    def __repr__(self):
        return f"search '{self.search_id}' ({self.query})"


class QueryMessage:
    """
    A class representing a message carrying a query.

    All query messages are associated with a search object. They should *not* be initialized
    directly but they should be spawned by the search object or cloned by another message.

    Note that node embeddings are also diffused via message passing but message objects refer
    only to query messages in this simulation.

    Attributes:
        name (string): A name identifying the message, unique in the network.
        query (Query): The query carried by the message.
        embedding (np.array): The query embedding carried by this message.
        search_id (str): A field identifying the search that this message is associated with.
        ttl (int): The time-to-live field of the message.
        hops (int): The hops that the message has travelled so far.
        hops_to_reach_doc (int): The hops to reach the candidate document for this message.
        candidate_doc (Document): The best candidate document for this query message.
        candidate_doc_similarity (float): The score of the best candidate document for this query message.
        visited_edges (list[tuple]): The edges visited by this query message in correct order.
        visited_nodes (list[Node]): The nodes visited by this query message in correct order.
    """

    counter = 0

    def __init__(
        self, query: Query, ttl: int, search_id: str, register_message_callback
    ):
        """
        Constructs a QueryMessage.

        Note that the message constructor should not be called directly.
        Query messages should be created either via spawning from a search object
        or cloning from another message.

        Arguments:
            query (Query): A query object.
            ttl (int): A time-to-live value.
            search_id (string): The identification of a search to associate the message with.
            register_message_callback (lambda): A callback function to register the message
                to a search object.
        """
        self.name = f"mesg{self.__class__.counter}"
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
        return self.query.embedding

    @property
    def visited_nodes(self):
        if len(self.visited_edges) == 0:
            return []
        nodes = [edge[0] for edge in self.visited_edges]
        nodes.append(self.visited_edges[-1][1])
        return nodes

    def is_alive(self):
        """
        Utility function to check if the message has not run out of hops.
        """
        return self.hops < self.ttl

    def clone(self):
        """
        Clones a message to perform a parallel walk.

        Returns:
            QueryMessage: A copy of this query message. It also copies its walk history.
        """

        copy = QueryMessage(
            self.query, self.ttl, self.search_id, self.register_message_callback
        )
        copy.hops = self.hops
        copy.hops_to_reach_doc = self.hops_to_reach_doc
        copy.candidate_doc = self.candidate_doc
        copy.candidate_doc_similarity = self.candidate_doc_similarity
        copy.visited_edges = [
            edge for edge in self.visited_edges
        ]  # should it start anew?
        self.register_message_callback(copy)
        return copy

    def send(self, from_node, to_node):  # use node names?
        """
        Updates the fields of the message when it is transmitted in a link.
        The actual transmission is performed by the network object.

        Returns:
            QueryMessage: The query message itself.
        """
        self.hops += 1
        self.visited_edges.append((from_node.name, to_node.name))
        return self

    def receive(self, other):
        """
        Merges messages of the same search that arrive at the same node.
        Triggers updating the candidate documents.

        Arguments:
            other (QueryMessage): Another message of the same search.
        """
        assert self.search_id == other.search_id
        if other.candidate_doc_similarity > self.candidate_doc_similarity:
            self.candidate_doc = other.candidate_doc
            self.hops_to_reach_doc = other.hops_to_reach_doc
            self.candidate_doc_similarity = other.candidate_doc_similarity
        self.hops = max(self.hops, other.hops)  # min could be another option

    def retrieve(self, docs: Sequence[Document]):
        """
        Executes the message query against a sequence of documents.
        Triggers updating the candidate document. This function is delegated to the message
        instead of the node object to contain all retrieval operations.

        Arguments:
            docs (Sequence[Document]): A sequence of documents.
        """
        for doc in docs:
            score = np.sum(doc.embedding * self.embedding)
            if score > self.candidate_doc_similarity:
                self.candidate_doc_similarity = score
                self.candidate_doc = doc
                self.hops_to_reach_doc = self.hops

    def kill(self, at_node, reason):
        # could potentially notify the querying node
        pass

    def __str__(self):
        return f"message '{self.name}' (search '{self.search_id}', {self.query})"
        # return f"message '{self.name}' (search '{self.search_id}', {self.query}, {self.ttl-self.hops} hops remaining)"

    def __repr__(self):
        return f"message '{self.name}' (search '{self.search_id}', {self.query})"

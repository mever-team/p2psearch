from abc import abstractmethod
from datatypes import Document, MessageQuery
from collections import defaultdict


class Node:

    """
    Abstract base class representing an embedding-aware P2P network node.
    The node can store pre-computed document embeddings,
    as well as calculate and diffuse node embeddings via personalized page rank.

    The base class is extended by other node classes in the nodes package
    that implement specific ways to calculate personalization embeddings
    and to forward query messages to neighboring nodes. 

    Class attributes:
        ppr_a (float): The teleport probability of the personalized page rank diffusion.
    
    Instance attributes:
        name (str): Identification of the node in the network. Should be unique.
        neighbors (dict[Node, np.array]): Dictionary of nodes and their learnt embeddings. Both are updated when a query is received.
        emb_dim (int): The embedding dimension, which applies to node, query, and document embeddings.
        docs (dict[str, Document]): Dictionary of stored documents indexed by their names.
        query_queue (dict[str, MessageQuery]): Dictionary of messages indexed by their name. Represents processed message queries waiting to be forwarded.
        seen_from (dict[str, Set[Node]]): Dictionary of nodes indexed by message names. Remembers from which nodes a message was received to avoid reforwarding.
        sent_to (dict[str, Set[Node]]): Dictionary of nodes indexed by message names. Remembers to which nodes a message has been sent to avoid reforwarding.
        personalization (np.array): Initial node embedding generated via the stored document embeddings.
        embedding (np.array): Node embedding generated via diffusion of the personalization embeddings.
    """

    ppr_a = 0.1

    @classmethod
    def set_ppr_a(cls, ppr_a):
        cls.ppr_a = ppr_a

    def __init__(self, name, emb_dim):

        """
        Constructs a Node object.

        Arguments:
            name (str): The name identifying the node.
            emb_dim (int): The embedding dimension.
        """

        self.name = name
        self.neighbors = dict()
        self.emb_dim = emb_dim

        self.docs = dict()
        self.query_queue = dict()
        self.seen_from = defaultdict(lambda: set())
        self.sent_to = defaultdict(lambda: set())

        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def clear(self):

        """
        Resets the node by clearing all node structures and the node embeddings.
        """

        self.neighbors.clear()
        self.docs.clear()
        self.query_queue.clear()
        self.seen_from.clear()
        self.sent_to.clear()
        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def add_doc(self, doc: Document):
        
        """
        Adds a document to the node. Triggers the update of the node embeddings.

        Arguments:
            doc (Document): The document to add to the node.
        """
        
        self.docs[doc.name] = doc

        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def add_query(self, query: MessageQuery):

        """
        Adds a query message to the node at the start of a simulation.

        Arguments:
            query (MessageQuery): A message representing a query.
        """
                
        assert query.ttl >= 0, f"{query}, ttl should be >= 0"
        query.check_now(self.docs)
        if query.is_alive():
            if query.name in self.query_queue:
                self.query_queue[query.name].receive(query)
                query.kill(self, reason=f"query merged with queued clone")
            else:
                self.query_queue[query.name] = query
        else:
            query.kill(self, reason="ttl was initialized to 0")

    def filter_seen_from(self, nodes, query, as_type=list):
        '''Utility function that filters the nodes from which a message was seen.'''
        return as_type(set(nodes).difference(self.seen_from[query.name]))

    def filter_sent_to(self, nodes, query, as_type=list):
        '''Utility function that filters the nodes to which a message was sent in the past.'''
        return as_type(set(nodes).difference(self.sent_to[query.name]))

    def filter_query_history(self, nodes, query, as_type=list):
        '''Utility function that filters the nodes from which a message has passed as recorded in its history.'''
        nodes = {node.name: node for node in nodes}
        for visited_node_name in query.visited_nodes:
            if visited_node_name in nodes:
                nodes.pop(visited_node_name)
        return as_type(nodes.values())

    def has_queries_to_send(self):
        '''Utility function that checks if node has pending messages to forward.'''
        return len(self.query_queue) > 0

    def send_embedding(self):

        """
        Returns an embedding to be sent to neighboring nodes.
        The actual transfer is done by the simulation object.
        Shares the burden of computing the personalized page rank diffusion with the receiving nodes.
        """
        
        return self.embedding / max(1, len(self.neighbors)) ** 0.5

    @DeprecationWarning
    def update_embedding(self):
        ppr_a = self.__class__.ppr_a
        embedding = sum([emb for emb in self.neighbors.values()])
        self.embedding = (
            ppr_a * self.personalization
            + (1 - ppr_a) * embedding / len(self.neighbors) ** 0.5
        )

    def receive_embedding(self, neighbor, neighbor_embedding):
        
        """
        Receives an update from a neighboring node.
        Shares the burden of computing the personalized page rank diffusion with the sending node.
        If the neighbor is known, for efficiency the node embedding is updated based on the previously stored value.
        Otherwise, the embedding is calculated from scratch and the neighbor is stored.

        Arguments:
            neighbor (Node): A node from which an embedding advertisement is received.
            neighbor_embedding (np.array): The embedding advertised by the neighboring node.
        """

        ppr_a = self.__class__.ppr_a
        N = len(self.neighbors)
        if neighbor in self.neighbors:
            self.embedding += (
                (neighbor_embedding - self.neighbors[neighbor]) / N**0.5 * (1 - ppr_a)
            )
        else:
            self.embedding = (
                (self.embedding - ppr_a * self.personalization) * N**0.5
                + neighbor_embedding * (1 - ppr_a)
            ) / (N + 1) ** 0.5 + ppr_a * self.personalization
        self.neighbors[neighbor] = neighbor_embedding
        # self.update_embedding()

    def send_queries(self):

        """
        Decides the next hops to forward the processed queued messages.
        Empties the message queue and returns the next hops.
        The actual forwarding is done by the simulation object.

        Returns:
            dict[MessageQuery, List[Node]]: The next hops to forward the messages.
        """
        
        assert all(
            [query.is_alive() for query in self.query_queue.values()]
        ), "queries in query queue should have been alive"
        to_send = defaultdict(lambda: [])
        for query in self.query_queue.values():
            next_hops = self.get_next_hops(query)
            if len(next_hops) > 0:
                clones = [query.clone() for _ in range(len(next_hops) - 1)]
                outgoing_queries = [query]
                outgoing_queries.extend(clones)
                for next_hop, outgoing_query in zip(next_hops, outgoing_queries):
                    outgoing_query.send(self, next_hop)
                    self.sent_to[outgoing_query.name].add(next_hop)
                    to_send[next_hop].append(outgoing_query)
            else:
                query.kill(self, reason="no next hops to forward")
        self.query_queue.clear()
        return to_send

    def receive_queries(self, queries, from_node, kill_seen=False):

        """
        Receives and processes messages from a node.
        Processed messages are discarded or enter the message queue for forwarding.

        Arguments:
            queries (Iterable[QueryMessage]): An iterable of received messages.
            from_node (Node): The node from which messages were received.
            kill_seen (bool): Specifies if previously seen messages should be discarded.
        """
          
        for query in queries:
            if query.name not in self.seen_from:
                query.check_now(self.docs)  # performs retrieval against the node's documents
            elif query.name in self.seen_from and kill_seen:
                query.kill(self, reason="query has already been seen")

            if query.is_alive():
                if query.name in self.query_queue:
                    self.query_queue[query.name].receive(query)
                    query.kill(self, reason=f"query merged at node {self.name}")
                else:
                    self.query_queue[query.name] = query
            else:
                query.kill(self, reason="query reached its ttl limit")
            self.seen_from[query.name].add(from_node)

    @abstractmethod
    def get_next_hops(self, query):

        """
        Determines the next hops for a message. Abstract method, implemented in subclasses.

        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """
        
        pass

    @abstractmethod
    def get_personalization(self):
        
        """
        Calculates the personalization embedding.  Abstract method, implemented in subclasses.

        Returns:
            np.array: The personalization embedding.
        """
        
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

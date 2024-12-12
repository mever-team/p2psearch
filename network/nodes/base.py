import numpy as np

from abc import abstractmethod
from datatypes import Document, QueryMessage
from collections import defaultdict


class Node:
    """
    Abstract base class representing an embedding-aware P2P network node.
    The node can store pre-computed document embeddings, calculate node embeddings,
    and diffuse the latter via personalized page rank.

    The base class is extended by other classes in the nodes package
    that implement different ways of forwarding query messages to neighbors.

    Attributes:
        name (str): A name identifying the node in the network. Should be unique.
        neighbors (dict[Node, np.array]): Dictionary of nodes and their learnt embeddings.
            Starts empty and is updated when embeddings are diffused.
        emb_dim (int): The embedding dimension, same for node, query, and document embeddings.
        docs (dict[str, Document]): Dictionary of stored documents indexed by their names.
        query_queue (dict[str, MessageQuery]): Dictionary of messages indexed by their name. Represents processed message queries waiting to be forwarded.
        seen_from (dict[str, Set[Node]]): Dictionary of nodes indexed by message names. Remembers from which nodes a message was received to avoid reforwarding.
        sent_to (dict[str, Set[Node]]): Dictionary of nodes indexed by message names. Remembers to which nodes a message has been sent to avoid reforwarding.
        personalization (np.array): Initial node embedding generated via the stored document embeddings.
        embedding (np.array): Node embedding generated via diffusion of the personalization embeddings.
    """

    def __init__(self, name, emb_dim):
        """
        Constructs a Node object.

        Arguments:
            name (str): The name identifying the node.
            emb_dim (int): The embedding dimension.
        """

        self.name = name
        self.emb_dim = emb_dim
        self.neighbors_index = dict()
        self.docs_index = dict()
        self.messages_queue = dict()
        self.messages_seen_from = defaultdict(lambda: set())
        self.messages_sent_to = defaultdict(lambda: set())

        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def clear(self):
        """
        Resets the node by clearing all node structures and the node embeddings.
        """

        self.neighbors_index.clear()
        self.docs_index.clear()
        self.messages_queue.clear()
        self.messages_seen_from.clear()
        self.messages_sent_to.clear()
        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def add_doc(self, doc: Document):
        """
        Adds a document to the node. Triggers the update of the node embeddings.

        Arguments:
            doc (Document): The document to add to the node.
        """

        self.docs_index[doc.name] = doc

        self.personalization = self.get_personalization()
        self.embedding = self.personalization

    def add_message(self, message: QueryMessage):
        """
        Adds a query message to the node at the start of a simulation.

        Arguments:
            message (MessageQuery): A message representing a query.
        """

        assert message.ttl >= 0, f"{message}, ttl should be >= 0"
        message.retrieve(list(self.docs_index.values()))
        if message.is_alive():
            if message.name in self.messages_queue:
                self.messages_queue[message.name].receive(message)
                message.kill(self, reason=f"message merged with queued clone")
            else:
                self.messages_queue[message.name] = message
        else:
            message.kill(self, reason="ttl was initialized to 0")

    def get_personalization(self):
        """
        Computes the personalization embedding from all stored documents.

        Returns:
            np.array: The personalization embedding.
        """

        personalization = np.zeros(self.emb_dim)
        for doc in self.docs_index.values():
            personalization += doc.embedding
        return personalization

    def send_embedding(self):
        """
        Advertises an embedding to neighboring nodes (the actual transfer is done by the network object).
        Shares the burden of computing the personalized page rank diffusion with the neighboring nodes.

        Returns:
            (np.array): An embedding to be advertised.
        """

        return self.embedding / max(1, len(self.neighbors_index)) ** 0.5

    def receive_embedding(self, neighbor, neighbor_embedding: np.array, ppr_a: float):
        """
        Receives an embedding advertised by a neighboring node.
        Shares the burden of computing the personalized page rank diffusion with the neighboring node.
        If the neighbor is known, the embedding is updated efficiently via its previous value.
        Otherwise, it is computed from scratch and the neighbor is stored.

        Arguments:
            neighbor (Node): A neighboring node.
            neighbor_embedding (np.array): An embedding advertised by the neighboring node.
            ppr_a (float): The diffusion parameter of personalized page rank.
        """
        N = len(self.neighbors_index)
        if neighbor in self.neighbors_index:
            self.embedding += (
                (neighbor_embedding - self.neighbors_index[neighbor])
                / N**0.5
                * (1 - ppr_a)
            )
        else:
            self.embedding = (
                (self.embedding - ppr_a * self.personalization) * N**0.5
                + neighbor_embedding * (1 - ppr_a)
            ) / (N + 1) ** 0.5 + ppr_a * self.personalization
        self.neighbors_index[neighbor] = neighbor_embedding

    def filter_seen_from(self, nodes, message, as_type=list):
        """Utility function that filters the nodes from which a message was seen."""
        return as_type(set(nodes).difference(self.messages_seen_from[message.name]))

    def filter_sent_to(self, nodes, message, as_type=list):
        """Utility function that filters the nodes to which a message was sent in the past."""
        return as_type(set(nodes).difference(self.messages_sent_to[message.name]))

    def filter_message_history(self, nodes, message, as_type=list):
        """Utility function that filters the nodes from which a message has passed as recorded in its history."""
        nodes = {node.name: node for node in nodes}
        for visited_node_name in message.visited_nodes:
            if visited_node_name in nodes:
                nodes.pop(visited_node_name)
        return as_type(nodes.values())

    def has_messages_to_send(self):
        """Utility function that checks if node has pending messages to forward."""
        return len(self.messages_queue) > 0

    def send_messages(self):
        """
        Decides the next hops to forward the processed queued messages.
        Empties the message queue and returns the next hops.
        The actual forwarding is done by the simulation object.

        Returns:
            dict[QueryMessage, List[Node]]: The next hops to forward the messages.
        """

        assert all(
            [message.is_alive() for message in self.messages_queue.values()]
        ), "messages in message queue should have been alive"
        to_send = defaultdict(lambda: [])
        for message in self.messages_queue.values():
            next_hops = self.get_next_hops(message)
            if len(next_hops) > 0:
                clones = [message.clone() for _ in range(len(next_hops) - 1)]
                outgoing_messages = [message]
                outgoing_messages.extend(clones)
                for next_hop, outgoing_message in zip(next_hops, outgoing_messages):
                    outgoing_message.send(self, next_hop)
                    self.messages_sent_to[outgoing_message.name].add(next_hop)
                    to_send[next_hop].append(outgoing_message)
            else:
                message.kill(self, reason="no next hops to forward")
        self.messages_queue.clear()
        return to_send

    def receive_messages(self, messages, from_node, kill_seen=False):
        """
        Receives and processes messages from a node.
        Processed messages are discarded or enter the message queue for forwarding.

        Arguments:
            queries (Iterable[QueryMessage]): An iterable of received messages.
            from_node (Node): The node from which messages were received.
            kill_seen (bool): Specifies if previously seen messages should be discarded.
        """

        for message in messages:
            if message.name not in self.messages_seen_from:
                message.retrieve(
                    list(self.docs_index.values())
                )
            elif message.name in self.messages_seen_from and kill_seen:
                message.kill(self, reason="message has already been seen")

            if message.is_alive():
                if message.name in self.messages_queue:
                    self.messages_queue[message.name].receive(message)
                    message.kill(self, reason=f"message merged at node {self.name}")
                else:
                    self.messages_queue[message.name] = message
            else:
                message.kill(self, reason="message reached its ttl limit")
            self.messages_seen_from[message.name].add(from_node)

    @abstractmethod
    def get_next_hops(self, message):
        """
        Determines the next hops for a message. Abstract method that is implemented in subclasses.

        Arguments:
            message (QueryMessage): The message to be forwarded.

        Returns:
            Sequence[Node]: The nodes to forward the message.
        """

        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

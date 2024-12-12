import random
import numpy as np

from .base import Node


class WalkerNode(Node):
    """
    A class representing a P2P network node that forwards messages via an unbiased random walk.
    Implements the Node abstract class.

    WalkerNode forwards a query message to a randomly chosen neighbor.
    It does not discard previously seen messages
    but tries to avoid forwarding to neighbors that have already seen the message.

    Attributes:
        --> refer to Node.
    """

    def receive_messages(self, messages, from_node):
        """
        Overrides receive_messages by Node.
        Does not discard seen messages as reforwarding *does* make sense with random walks.

        Arguments:
            messages (Sequence[QueryMessage]): A sequence of received messages.
            from_node (Node): The node from which the messages are received.
        """

        super().receive_messages(messages, from_node, kill_seen=False)

    def get_next_hops(self, message):
        """
        Implements get_next_hops by Node.
        Samples a node uniformly from the node's neighbors (unbiased walk).
        Tries to filter nodes that have already seen the message but reverts if no options are available.

        Arguments:
            message (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors_index)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        return random.sample(neighbors, k=1)


class HardSumEmbeddingNode(WalkerNode):
    """
    A class representing a P2P network node that forwards messages via a biased random walk.
    The walker selects the neighbor whose embedding has the highest dot product with the query embedding.
    It does not discard seen messages and tries to avoid forwarding to nodes that have already seen the message.

    Attributes:
        --> for other attributes, refer to Node.
    """


    def get_next_hops(self, message):
        """
        Implements get_next_hops by Node and overrides Walker.
        Selects the neighbor whose embedding has the highest dot product with the query embedding.
        Tries to filter nodes that have already seen the message but reverts if no options are available.

        Arguments:
            message (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors_index)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors_index[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                np.sum(message.embedding * neighbor_embedding)
                for neighbor_embedding in neighbor_embeddings
            ]
        )
        idx = np.argmax(scores)
        return [neighbors[idx]]


class HardSumL2EmbeddingNodeWithSpawn(WalkerNode):
    """
    %% EXPERIMENTAL, USE WITH CARE %%

    A class representing a P2P network node that forwards messages via a biased random walk.
    The walker selects the neighbor whose embedding has the highest dot product with the query embedding,
    Every spawn_interval, it also samples two nodes instead of one.
    It does not discard seen messages and tries to avoid forwarding to nodes that have already seen the message.

    Attributes:
        spawn_interval (int): Hop interval at which to spawn walkers.
        --> for other attributes, refer to Node.
    """

    def __init__(self, spawn_interval=5, *args, **kwargs):
        """
        Constructs a HardSumL2EmbeddingNodeWithSpawn.
        """

        self.spawn_interval = spawn_interval
        super(HardSumL2EmbeddingNodeWithSpawn, self).__init__(*args, **kwargs)

    def get_next_hops(self, message):
        """
        Implements get_next_hops by Node and overrides Walker.
        Similar to HardSumEmbeddingNode but also spawns two walkers every spawn_interval.
        Tries to filter nodes that have already seen the message but reverts if no options are available.

        Arguments:
            message (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors_index)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors_index[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                np.linalg.norm(message.embedding - neighbor_embedding)
                for neighbor_embedding in neighbor_embeddings
            ]
        )
        if len(message.visited_nodes) % self.spawn_interval == 0:
            idxs = np.argsort(scores)[:2]
            return [neighbors[idx] for idx in idxs]
        else:
            idx = np.argmax(scores)
            return [neighbors[idx]]


class HardSumL2EmbeddingNode(WalkerNode):
    """
    %% EXPERIMENTAL, USE WITH CARE %%

    A class representing a P2P network node that forwards messages via a biased random walk.
    The walker selects the neighbor whose embedding has the lowest L2 distance from the query embedding.
    It does not discard seen messages and tries to avoid forwarding to nodes that have already seen the message.

    Attributes:
        spawn_interval (int): Hop interval at which to spawn walkers.
        --> for other attributes, refer to Node.
    """

    def get_next_hops(self, message):
        """
        Implements get_next_hops by Node and overrides Walker.
        Selects the neighbor whose embedding has the lowest L2 distance with the query embedding.
        Tries to filter nodes that have already seen the message but reverts if no options are available.

        Arguments:
            message (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors_index)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, message, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors_index[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                -np.linalg.norm(message.embedding - neighbor_embedding)
                for neighbor_embedding in neighbor_embeddings
            ]
        )
        idx = np.argmax(scores)
        return [neighbors[idx]]


class SoftSumEmbeddingNode(WalkerNode):
    """
    %% EXPERIMENTAL, USE WITH CARE %%

    A class representing a P2P network node that forwards messages via a biased random walk.
    The walker samples a node from the top 3 neighbors with the highest dot product with the query embedding.
    Meant for robustness and variety.
    It does not discard seen messages and tries to avoid forwarding to nodes that have already seen the message.

    Attributes:
        --> refer to Node.
    """

    def get_next_hops(self, message):
        """
        Implements get_next_hops by Node and overrides Walker.
        Samples a node from the top 3 neighbors whose embeddings have the highest dot product with the query embedding.
        Avoids sampling always the top node for variety.
        Tries to filter nodes that have already seen the message but reverts if no options are available.

        Arguments:
            message (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors_index)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, message)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, message)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors_index[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                np.sum(message.embedding * neighbor_embedding)
                for neighbor_embedding in neighbor_embeddings
            ]
        )
        idx = np.random.choice(
            np.argsort(-scores)[:3]
        )  # choose randomly from top scored documents

        return [neighbors[idx]]

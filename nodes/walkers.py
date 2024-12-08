import random
import numpy as np

from nodes.base import Node


class WalkerNode(Node):

    """
    A class representing a P2P network node that forwards messages via random walk.
    Implements the Node abstract class.

    WalkerNode forwards a query message to a randomly chosen neighbor.
    It does not discard previously seen messages
    but tries to avoid neighbors that have previously sent the message.
    
    Instance attributes:
        --> refer to Node.
    """

    def receive_queries(self, queries, from_node):
        
        """
        Overrides receive_queries by Node.
        Does not discard seen messages as reforwarding makes sense with random walks.

        Arguments:
            --> refer to Node.
        """
        
        super().receive_queries(queries, from_node, kill_seen=False)

    def get_next_hops(self, query):

        """
        Implements get_next_hops by Node.
        Tries to sample a neighbor from the nodes that haven't received the message.
        If no nodes are available, it samples from the full list of neighbors.
        Sampling is uniform.

        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        candidates = self.filter_seen_from(neighbors, query, as_type=list)
        if len(candidates) > 0:
            return random.sample(candidates, k=1)
        else:
            return random.sample(neighbors, k=1)

    def get_personalization(self):

        """
        Implements get_personalization by Node.
        Of no importance to WalkerNode as it does not bias the walk.

        Returns:
            np.array: Zeros personalization embedding.
        """

        return np.zeros(self.emb_dim)
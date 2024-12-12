from .base import Node


class FlooderNode(Node):
    """
    A class representing a P2P network node that forwards messages via flooding.
    Implements the Node abstract class.

    FlooderNode forwards query messages the first time and discards them upon subsequent receptions.
    It does not use personalization embeddings.

    Instance attributes:
        --> refer to Node.
    """

    def receive_messages(self, queries, from_node):
        """
        Overrides receive_queries by Node.
        Discards seen messages as reforwarding makes no sense with flooding.

        Arguments:
            queries (Iterable[QueryMessage]): An iterable of received messages.
            from_node (Node): The node from which messages were received.
        """

        super().receive_messages(queries, from_node, kill_seen=True)

    def get_next_hops(self, query):
        """
        Implements get_next_hops by Node.
        Selects all neighbors except for the node that sent the message.

        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors_index)
        if len(neighbors) == 0:
            return []

        next_hops = self.filter_seen_from(neighbors, query, as_type=list)
        return next_hops

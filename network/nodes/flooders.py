from .base import Node


class FlooderNode(Node):
    """
    A class representing a P2P network node that forwards messages via flooding.
    Implements the Node abstract class.

    FlooderNode forwards query messages the first time and discards them the next time it resees them.
    It does not use personalization embeddings.

    Attributes:
        --> refer to Node.
    """

    def receive_messages(self, messages, from_node):
        """
        Overrides receive_messages by Node.
        Discards seen messages as reforwarding makes no sense with flooding.

        Arguments:
            queries (Sequence[QueryMessage]): A sequence of received messages.
            from_node (Node): The node from which the messages are received.
        """

        super().receive_messages(messages, from_node, kill_seen=True)

    def get_next_hops(self, message):
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

        next_hops = self.filter_seen_from(neighbors, message, as_type=list)
        return next_hops

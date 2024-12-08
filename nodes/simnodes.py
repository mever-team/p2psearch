import numpy as np
from nodes import WalkerNode


class HardSumEmbeddingNode(WalkerNode):

    """
    A class representing a P2P network node that forwards messages via a biased random walk.
    The walker selects the neighbor whose embedding has the highest dot product with the query embedding.
    It does not discard seen messages and tries to avoid sending the message to nodes that have already seen it.
    
    Instance attributes:
        remove_succesful_queries (bool): Discards queries that have found the golden document.
            In practice, the node does not know which is the golden document,
            this is meant as a computational shortcut for the hit count and the hop count length,
            which are not affected by the next hops.
        --> for other attributes, refer to Node.
    """


    def __init__(self, name, dim, remove_successful_queries=False):

        """
        Constructs a HardSumEmbeddingNode.
        """

        self.remove_successful_queries = remove_successful_queries
        super(HardSumEmbeddingNode, self).__init__(name, dim)

    def get_personalization(self):

        if len(self.docs) == 0:
            return np.zeros(self.emb_dim)
        personalization = 0
        for doc in self.docs.values():
            personalization += doc.embedding
        return personalization

    def get_next_hops(self, query):

        """
        Implements get_next_hops by Node and overrides Walker.
        Selects the neighbor whose embedding has the highest dot product with the query embedding.
        Tries to filter nodes that have already seen the message but reverts if no options remain.
        
        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """
                
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        if (
            self.remove_successful_queries
            and query.candidate_doc == query.query._gold_doc
        ):
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        # filtered_neighbors = self.filter_query_history(neighbors, query, as_type=list)
        # if len(filtered_neighbors) > 0:
        #     neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                np.sum(query.embedding * neighbor_embedding)
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
    It does not discard seen messages and tries to avoid sending the message to nodes that have already seen it.
    
    Instance attributes:
        spawn_interval (int): Hop interval at which to spawn walkers.
        --> for other attributes, refer to Node.
    """


    def __init__(self, spawn_interval=5, *args, **kwargs):

        """
        Constructs a HardSumL2EmbeddingNodeWithSpawn.
        """

        self.spawn_interval = spawn_interval
        super(HardSumL2EmbeddingNodeWithSpawn, self).__init__(*args, **kwargs)

    def get_personalization(self):
        if len(self.docs) == 0:
            return np.zeros(self.emb_dim)
        personalization = 0
        for doc in self.docs.values():
            personalization += doc.embedding
        return personalization

    def get_next_hops(self, query):
        
        """
        Implements get_next_hops by Node and overrides Walker.
        Similar to HardSumEmbeddingNode but also spawns two walkers every spawn_interval.
        Tries to filter nodes that have already seen the message but reverts if no options remain.
        
        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """
        
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        # filtered_neighbors = self.filter_query_history(neighbors, query, as_type=list)
        # if len(filtered_neighbors) > 0:
        #     neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                np.linalg.norm(query.embedding - neighbor_embedding)
                for neighbor_embedding in neighbor_embeddings
            ]
        )
        if len(query.visited_nodes) % self.spawn_interval == 0:
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
    It does not discard seen messages and tries to avoid sending the message to nodes that have already seen it.
    
    Instance attributes:
        spawn_interval (int): Hop interval at which to spawn walkers.
        --> for other attributes, refer to Node.
    """

    def get_personalization(self):
        if len(self.docs) == 0:
            return np.zeros(self.emb_dim)
        personalization = 0
        for doc in self.docs.values():
            personalization += doc.embedding
        return personalization

    def get_next_hops(self, query):

        """
        Implements get_next_hops by Node and overrides Walker.
        Selects the neighbor whose embedding has the lowest L2 distance with the query embedding.
        Tries to filter nodes that have already seen the message but reverts if no options remain.
        
        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """

        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, query, as_type=list)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        # filtered_neighbors = self.filter_query_history(neighbors, query, as_type=list)
        # if len(filtered_neighbors) > 0:
        #     neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                -np.linalg.norm(query.embedding - neighbor_embedding)
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
    It does not discard seen messages and tries to avoid sending the message to nodes that have already seen it.
    
    Instance attributes:
        --> refer to Node.
    """

    def get_personalization(self):
        return np.sum(doc.embedding for doc in self.docs.values())

    def get_next_hops(self, query):

        """
        Implements get_next_hops by Node and overrides Walker.
        Samples a node from the top 3 neighbors whose embeddings have the highest dot product with the query embedding.
        Avoids sampling always the top node for variety.
        Tries to filter nodes that have already seen the message but reverts if no options remain.
        
        Arguments:
            query (QueryMessage): The message to be forwarded.

        Returns:
            List[Node]: The nodes to forward the message.
        """
                
        neighbors = list(self.neighbors)
        if len(neighbors) == 0:
            return []

        filtered_neighbors = self.filter_seen_from(neighbors, query)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        filtered_neighbors = self.filter_sent_to(neighbors, query)
        if len(filtered_neighbors) > 0:
            neighbors = filtered_neighbors

        neighbor_embeddings = [self.neighbors[neighbor] for neighbor in neighbors]
        scores = np.array(
            [
                np.sum(query.embedding * neighbor_embedding)
                for neighbor_embedding in neighbor_embeddings
            ]
        )
        idx = np.random.choice(
            np.argsort(-scores)[:3]
        )  # choose randomly from top scored documents

        return [neighbors[idx]]

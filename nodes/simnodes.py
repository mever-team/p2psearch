import numpy as np
from nodes import WalkerNode


class HardSumEmbeddingNode(WalkerNode):

    def __init__(self, name, dim, remove_successful_queries=False):
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

    def __init__(self, spawn_interval=5, *args, **kwargs):
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

    def get_personalization(self):
        if len(self.docs) == 0:
            return np.zeros(self.emb_dim)
        personalization = 0
        for doc in self.docs.values():
            personalization += doc.embedding
        return personalization

    def get_next_hops(self, query):
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

    def get_personalization(self):
        return np.sum(doc.embedding for doc in self.docs.values())

    def get_next_hops(self, query):
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

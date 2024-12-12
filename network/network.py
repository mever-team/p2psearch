import random
import numpy as np
import networkx as nx

from datatypes import Document, QueryMessage
from typing import List
from .loader import load_ppr_matrix, load_graph


def load_network(dataset, init_node, ppr_a):
    graph = load_graph(dataset)
    return P2PNetwork(dataset, graph, init_node, ppr_a)


class P2PNetwork:
    """
    A class representing a simulation of a retrieval application over a P2P network.

    Attributes:
        nodes (list[networkx.Node]): A list of the network's nodes.
        edges (list[tuple]): The edgelist of the network.
    """

    def __init__(self, name: str, graph: nx.Graph, init_node, ppr_a: float):
        """
        Constructs a Simulation.

        Arguments:
            graph (networkx.Graph): The graph of a network.
            _graph_name: The name of the graph used for retrieving cached data.
        """

        self.name = name

        self.nodes = [init_node(name) for name in graph.nodes]
        node_dict = {name: node for name, node in zip(graph.nodes, self.nodes)}
        self.edges = [(node_dict[u], node_dict[v]) for u, v in graph.edges]
        self.adj = nx.adjacency_matrix(graph)

        self.set_ppr_a(ppr_a)

    def set_ppr_a(self, ppr_a):
        self.ppr_a = ppr_a

        self.ppr_mat = load_ppr_matrix(
            dataset=self.name, ppr_a=ppr_a, symmetric=True, _adjacency_matrix=self.adj
        )

    def sample_node(self):
        """Utility function to sample a random node from the graph."""
        return random.choice(self.nodes)

    def sample_nodes(self, k):
        """Utility function to sample k random nodes from the graph (with replacement)."""
        return random.choices(self.nodes, k=k)

    def scatter_doc(self, document: Document):
        self.scatter_docs([document])

    def scatter_docs(self, documents: List[Document]):
        """Stores documents to nodes sampled randonly from the graph (with replacement)."""
        for node, doc in zip(random.choices(self.nodes, k=len(documents)), documents):
            node.add_doc(doc)

    def scatter_queries(self, queries: List[QueryMessage]):
        """Stores queries to nodes sampled randonly from the graph (with replacement)."""
        for node, query in zip(random.choices(self.nodes, k=len(queries)), queries):
            node.add_query(query)

    def diffuse_embeddings(self, epochs, monitor=None):
        """
        Diffuse node embeddings asynchronously for multiple epochs.
        This will be slow. If speed is important, run diffuse_fast_embeddings().

        Arguments:
            epochs: The number of epochs to run the diffusion.
            monitor: Utility object to stop early due to convergence.
        """

        for time in range(epochs):
            print(f"EPOCH {time+1}")
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.5:
                    v.receive_embedding(u, u.send_embedding(), self.ppr_a)
                    u.receive_embedding(v, v.send_embedding(), self.ppr_a)

            if monitor is not None and not monitor():
                break

    def diffuse_fast_embeddings(self):
        """
        Diffuse node embeddings quickly by calculating them analytically.
        The calculations require the personalized page rank diffusion matrix
        which is cached and even passed directly in some simulations to save time.
        The convergence of asynchronous and analytical diffusion is confirmed in check_ppr_convergence.py,
        if complete accuracy is however desired, run diffuse_embeddings().

        Arguments:
            _ppr_mat (np.array): The personalized page rank diffusion matrix, if passed directly.
        """

        personalizations = np.array([node.personalization for node in self.nodes])
        if personalizations.ndim > 2:
            embeddings = np.zeros_like(personalizations)
            for i in range(personalizations.shape[1]):
                embeddings[:, i, :] = self.ppr_mat @ personalizations[:, i, :]
        else:
            embeddings = self.ppr_mat @ personalizations

        for node, embedding in zip(self.nodes, embeddings):
            node.embedding = embedding

        for u, v in self.edges:
            u.neighbors[v] = v.embedding
            v.neighbors[u] = u.embedding

    def forward_queries(self, epochs, monitor):
        """
        Forward queries for multiple epochs.

        Arguments:
            epochs: The number of epochs to run queries.
            monitor: Utility object to stop early.

        Returns:
            int: The epoch the simulation finished.
        """
        nodes_to_check = self.nodes
        for time in range(epochs):
            outgoing = {}
            for u in nodes_to_check:
                if u.has_queries_to_send():
                    outgoing[u] = u.send_queries()

            nodes_to_check = []
            for u, to_send in outgoing.items():
                for v, queries in to_send.items():
                    v.receive_queries(queries, u)
                    nodes_to_check.append(v)  # focus only on nodes with queries to send

            if len(nodes_to_check) == 0:
                break

            if monitor is not None and not monitor():
                break
        return time

    def __call__(self, epochs, monitor=None):
        """
        Runs a complete simulation where advertisements and queries are sent jointly at each epoch.

        Arguments:
            epochs: The number of epochs to run the simulation.
            monitor: Utility object to stop early.

        Returns:
            int: The epoch the simulation finished.
        """
        time = 0
        for time in range(epochs):
            random.shuffle(self.edges)
            for u, v in self.edges:
                if random.random() < 0.1:
                    mesg_to_v, mesg_to_u = u.send_embedding(), v.send_embedding()
                    v.receive_embedding(u, mesg_to_v, self.ppr_a)
                    u.receive_embedding(v, mesg_to_u, self.ppr_a)

            random.shuffle(self.nodes)
            outgoing = {}
            for u in self.nodes:
                if u.has_queries_to_send() and random.random() < 1.0:
                    outgoing[u] = u.send_queries()
            for u, to_send in outgoing.items():
                for v, queries in to_send.items():
                    v.receive_queries(queries, u)

            if monitor is not None and not monitor():
                break
        return time

    def clear(self):
        """
        Clears the nodes from embeddings, queries, and documents.
        """
        for node in self.nodes:
            node.clear()

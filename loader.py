import os
import numpy as np
import networkx as nx
import utils

from data import network, ir
from stubs import StubNode



def load_graph(node_init, dataset="fb"):
    """
    Loads a dataset graph as an networkx.Graph object.
    The first time the dataset is downloaded and
    then cached locally to speed up subsequent loadings.

    Arguments:
        node_init (cls): The class of the node type from the nodes package.
        dataset (str): The name of the dataset.

    Returns:
        networkx.Graph: An object representing the dataset graph.
    """

    graph = nx.Graph()
    node_dict = dict()
    filepath = network.get_edgelist_path(dataset)
    if not os.path.exists(filepath):
        network.download(dataset, filepath)
    with open(filepath) as file:
        for line in file:
            node_names = line[:-1].split(";")
            for node_name in node_names:
                if node_name not in node_dict:
                    node_dict[node_name] = node_init(node_name)
            graph.add_edge(node_dict[node_names[0]], node_dict[node_names[1]])
            graph.add_edge(node_dict[node_names[1]], node_dict[node_names[0]])
    return graph


def load_ppr_matrix(dataset, alpha, symmetric=True, _graph=None):
    """
    Loads the personalized page rank diffusion matrix.
    The first time the matrix is calculated and
    then cached locally to speed up subsequent loadings.

    Arguments:
        dataset (str): The name of the graph dataset.
        alpha (fload): The teleport probability of the personalized page rank diffusion.
        symmetric (bool): Selects the symmetric or asymmetric form of the matrix.
        _graph (networkx.Graph): Passes the graph object directly to avoid loading from file system.

    Returns:
        networkx.Graph: An object representing the dataset graph.
    """

    filepath = network.get_ppr_matrix_path(dataset, alpha, symmetric)
    if os.path.exists(filepath):
        return np.load(filepath)

    graph = _graph or load_graph(StubNode, dataset)
    ppr_matrix = utils.analytic_ppr(nx.adjacency_matrix(graph), alpha, symmetric)
    np.save(filepath, ppr_matrix)
    return ppr_matrix


def load_query_results(dataset="glove"):
    """
    Loads the golden truth of a retrieval dataset.
    If the dataset does not exist locally, it is downloaded and cached,
    hence the first time may be slow.

    Arguments:
        dataset (str): The name of a retrieval dataset.

    Returns:
        dict[str, str]: A dictionary of queries and their gold documents.
    """
    filepath = ir.get_qrels_path(dataset)
    print(filepath)
    if not os.path.exists(filepath):
        ir.download(dataset)
    with open(filepath, "r", encoding="utf8") as f:
        results = dict()
        for line in f:
            que_id, doc_id, _ = line.strip().split("\t")
            results[que_id] = doc_id
    return results


def load_embeddings(dataset="glove", type="docs"):
    """
    Loads the embeddings of a retrieval dataset.
    If the dataset does not exist locally, it is downloaded and cached,
    hence the first time may be slow.

    Arguments:
        dataset (str): The name of a retrieval dataset.
        type (str): The type of the embeddings. Available types are "queries", "documents", "other_docs",
            respesenting queries, relevant / gold documents, irrelevant / other documents.
    Returns:
        dict[str, np.array]: A dictionary of embeddings indexed by query or document names.
    """
    filepath = ir.get_embeddings_path(dataset, type)
    if not os.path.exists(filepath):
        ir.download(dataset)
    many_arrays = np.load(filepath)
    ids = many_arrays["ids"]
    embs = many_arrays["embs"]
    return {idx: emb for idx, emb in zip(ids, embs)}


def load_all(dataset):
    """
    Utility function that loads all useful data of a retrieval dataset that are needed in a simulation.
    If the dataset does not exist locally, it is downloaded and cached, hence the first time may be slow.

    Arguments:
        dataset (str): The name of a retrieval dataset.

    Returns:
        tuple: A tuple containing the embedding dimension, the ground truth dictionary,
            the query, document, and other document embeddings.
    """
    query_results = load_query_results()
    que_embs = load_embeddings(dataset=dataset, type="queries")
    doc_embs = load_embeddings(dataset=dataset, type="docs")
    other_doc_embs = load_embeddings(dataset=dataset, type="other_docs")
    dim = len(next(iter(other_doc_embs.values())))
    return dim, query_results, que_embs, doc_embs, other_doc_embs


def load_texts(dataset="glove", type="docs"):
    """
    Loads the texts of a retrieval dataset.
    If the dataset does not exist locally, it is downloaded and cached,
    hence the first time may be slow.

    Arguments:
        dataset (str): The name of a retrieval dataset.
        type (str): The type of the embeddings. Available types are "queries", "documents", "other_docs",
            respesenting queries, relevant / gold documents, irrelevant / other documents.
    Returns:
        dict[str, str]: A dictionary of texts indexed by the query or document name.
    """
    filepath = ir.get_texts_path(dataset, type)
    if not os.path.exists(filepath):
        ir.download(dataset)
    with open(filepath, "r", encoding="utf8") as f:
        texts = dict()
        for line in f:
            idx, text = line.strip().split("\t")
            texts[idx] = text
    return texts

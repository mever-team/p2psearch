import gzip
import requests
import networkx as nx
import numpy as np

from pathlib import Path
from utils import analytic_ppr
from .network import P2PNetwork

URLS = {
    "gnutella": "http://snap.stanford.edu/data/p2p-Gnutella08.txt.gz",
    "fb": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "internet": "https://snap.stanford.edu/data/as20000102.txt.gz",
}

TOY_GRAPH_PARAMETERS = {
    "toy_erdos": {"n": 50, "p": 0.2},
    "toy_watts_strogatz": {"n": 50, "k": 3, "p": 0.2},
}

def load_network(dataset, node_init, ppr_a):
    graph = load_graph(dataset)
    return P2PNetwork(dataset, graph, node_init, ppr_a)


def load_graph(dataset="fb"):
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

    dset_path = Path(__file__).parent / ".cache" / dataset
    dset_path.mkdir(parents=True, exist_ok=True)
    edgelist_path = dset_path / "edgelist.csv"
    if not edgelist_path.exists():
        print(f'generating and caching graph "{dataset}"')
        generate(dataset, edgelist_path)
    return nx.read_edgelist(edgelist_path)



def load_ppr_matrix(dataset, alpha, symmetric=True, _adjacency_matrix=None):
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
    dset_path = Path(__file__).parent / ".cache" / dataset
    dset_path.mkdir(exist_ok=True)
    pprmat_path = (
        dset_path / f"pprmat_alpha{alpha}_{'symm' if symmetric else 'asymm'}.npy"
    )

    if pprmat_path.exists():
        return np.load(pprmat_path)

    print("[ppr matrix loader]: diffusion matrix is not cached, will compute")

    adj = (
        nx.adjacency_matrix(load_graph(dataset))
        if _adjacency_matrix is None
        else _adjacency_matrix
    )
    ppr_matrix = analytic_ppr(adj, alpha, symmetric)

    np.save(pprmat_path, ppr_matrix)
    return ppr_matrix


def generate(dataset, edgelist_path):
    """
    Downloads the edgelist of a graph dataset to filepath.

    Arguments:
        dataset (str): The name of the dataset. Available names are found in METADATA.
        filepath (str): The path where to download the dataset edgelist. The standard path is given by get_edgelist_path().
    """

    if dataset in URLS:
        url = URLS[dataset]
        print(f'[graph generation script]: downloading graph "{dataset}" from {url}')
        res = requests.get(url, allow_redirects=True)
        print(f"[graph generation script]: decompressing graph")
        data = gzip.decompress(res.content)
        print(f"[graph generation script]: caching graph as edgelist")
        with open(edgelist_path, "wb") as f:
            f.write(data)
        print(f"[graph generation script]: finished!")

    elif dataset == "toy_erdos":
        n, p = TOY_GRAPH_PARAMETERS[dataset]["n"], TOY_GRAPH_PARAMETERS[dataset]["p"]
        g = nx.gnp_random_graph(n, p)
        while not nx.is_connected(g):
            p = min(1, 1.01 * p)
            g = nx.gnp_random_graph(n, p)

        with open(edgelist_path, "w") as f:
            for e in g.edges:
                f.write(f"{e[0]} {e[1]}\n")

    elif dataset == "toy_watts_strogatz":
        n, k, p = TOY_GRAPH_PARAMETERS[dataset]["n"], TOY_GRAPH_PARAMETERS[dataset]["k"], TOY_GRAPH_PARAMETERS[dataset]["p"]
        g = nx.connected_watts_strogatz_graph(n, k, p)

        with open(edgelist_path, "w") as f:
            for e in g.edges:
                f.write(f"{e[0]} {e[1]}\n")

    else:
        raise Exception(
            f"unknown graph \"{dataset}\", try from {list(URLS)} or {list(TOY_GRAPH_PARAMETERS)}"
        )

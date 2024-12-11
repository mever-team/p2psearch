import gzip
import requests
import networkx as nx
import numpy as np
from pathlib import Path
from utils import analytic_ppr


METADATA = {
    "gnutella": {
        "url": "http://snap.stanford.edu/data/p2p-Gnutella08.txt.gz",
        "delimiter": "\t",
    },
    "fb": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "delimiter": " ",
    },
    "internet": {
        "url": "https://snap.stanford.edu/data/as20000102.txt.gz",
        "delimiter": "\t",
    },
    "toy_erdos": {"n": 50, "p": 0.2},
    "toy_watts_strogatz": {"n": 50, "k": 3, "p": 0.2},
}
COMMON_DELIMITER = ";"

def load_graph(dataset="fb", node_init=None):
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
    if node_init is None:
        node_init = lambda x:x

    graph = nx.Graph()
    node_dict = dict()

    dset_path =  Path(__file__).parent / ".cache" / dataset
    dset_path.mkdir(parents=True, exist_ok=True)
    edgelist_path = dset_path / "edgelist.csv"
    if not edgelist_path.exists():
        print("[graph loader]: graph is not cached, will download")
        download(dataset, edgelist_path)

    with open(edgelist_path) as file:
        for line in file:
            node_names = line[:-1].split(";")
            for node_name in node_names:
                if node_name not in node_dict:
                    node_dict[node_name] = node_init(node_name)
            graph.add_edge(node_dict[node_names[0]], node_dict[node_names[1]])
            graph.add_edge(node_dict[node_names[1]], node_dict[node_names[0]])
    return graph


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
    dset_path =  Path(__file__).parent / ".cache" / dataset
    dset_path.mkdir(exist_ok=True)
    pprmat_path = dset_path / f"pprmat_alpha{alpha}_{'symm' if symmetric else 'asymm'}.npy"

    if pprmat_path.exists():
        return np.load(pprmat_path)

    print("[ppr matrix loader]: diffusion matrix is not cached, will compute")
    
    adj = nx.adjacency_matrix(load_graph(dataset)) if _adjacency_matrix is None else _adjacency_matrix
    ppr_matrix = analytic_ppr(adj, alpha, symmetric)

    np.save(pprmat_path, ppr_matrix)
    return ppr_matrix


def download(dataset, edgelist_path):
    """
    Downloads the edgelist of a graph dataset to filepath.

    Arguments:
        dataset (str): The name of the dataset. Available names are found in METADATA.
        filepath (str): The path where to download the dataset edgelist. The standard path is given by get_edgelist_path().
    """
    
    if dataset in ["gnutella", "fb", "internet"]:
        url = METADATA[dataset]["url"]
        print(f"[graph generation script]: downloading {dataset} graph from {url}")
        res = requests.get(url, allow_redirects=True)
        print(f"[graph generation script]: decompressing graph")
        data = gzip.decompress(res.content)
        print(f"[graph generation script]: saving graph in common format")
        with open(edgelist_path, "wb") as f:
            f.write(data)

        f = open(edgelist_path, "r", encoding="utf8")
        lines = f.readlines()
        f.close()

        with open(edgelist_path, "w") as f:
            for line in lines:
                if line.startswith("#"):
                    continue
                f.write(line.replace(METADATA[dataset]["delimiter"], COMMON_DELIMITER))
        print(f"[graph generation script]: finished!")

    elif dataset == "toy_erdos":
        n, p = METADATA[dataset]["n"], METADATA[dataset]["p"]
        g = nx.gnp_random_graph(n, p)
        while not nx.is_connected(g):
            p = min(1, 1.01 * p)
            g = nx.gnp_random_graph(n, p)

        with open(edgelist_path, "w") as f:
            for e in g.edges:
                f.write(f"{e[0]}{COMMON_DELIMITER}{e[1]}\n")

    elif dataset == "toy_watts_strogatz":
        n, k, p = METADATA[dataset]["n"], METADATA[dataset]["k"], METADATA[dataset]["p"]
        g = nx.connected_watts_strogatz_graph(n, k, p)

        with open(edgelist_path, "w") as f:
            for e in g.edges:
                f.write(f"{e[0]}{COMMON_DELIMITER}{e[1]}\n")

    else:
        raise Exception(
            f"unknown dataset '{dataset}', known datasets: {list(METADATA)}"
        )
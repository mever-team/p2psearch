import os
import numpy as np
import networkx as nx

from scipy import sparse
from scipy.sparse.linalg import inv


def search(nodes, query_embedding):
    return max(
        [doc for node in nodes for doc in node.docs.values()],
        key=lambda doc: np.sum(doc.embedding * query_embedding),
    )


def analytic_ppr(adj, alpha, symmetric=True):
    """
    Calculates the personalized page rank diffusion matrix.
    Depending on the teleport probability (alpha), the computation is more efficient
    as an inverse matrix or power series operation.

    Arguments:
        adj (np.array): Adjacency matrix of the graph.
        alpha (float): Personalized page rank teleport probability.
        symmetric (bool): Selects the symmetric or assymetric version of the diffusion matrix.

    Returns:
        np.array: The diffusion matric of personalized page rank.
    """

    if alpha > 0.5:
        ppr_mat = power_analytic_ppr(adj, alpha, symmetric).toarray()
    else:
        ppr_mat = exact_analytic_ppr(adj, alpha, symmetric).toarray()
    return ppr_mat


def power_analytic_ppr(adj, alpha, symmetric=True):
    """Calculates the personalized page rank diffusion matrix as a power series."""

    def powerseries(W, tol=1e-4, max_tries=500):
        n = W.shape[0]
        I = sparse.identity(n)
        S_old = I
        S = S_old
        for i in range(max_tries):

            S = I + W @ S_old
            dS = np.max(np.abs(S - S_old))
            # print(f"iter {i + 1} diff {dS}")
            if dS < tol:
                break
            S_old = S
        return S

    D = np.array(adj.sum(axis=1)).squeeze()
    if symmetric:
        invsqrtD = sparse.diags(D**-0.5)
        trans_mat = invsqrtD @ adj.transpose() @ invsqrtD
    else:
        invD = sparse.diags(D**-1.0)
        trans_mat = adj.transpose() @ invD

    ppr_mat = alpha * powerseries((1 - alpha) * trans_mat)
    return ppr_mat


def exact_analytic_ppr(adj, alpha, symmetric=True):
    """Calculates the personalized page rank diffusion matrix as matrix inversion."""
    n = adj.shape[0]
    I = sparse.identity(n, format="csc")
    D = np.array(adj.sum(axis=1)).squeeze()
    if symmetric:
        invsqrtD = sparse.diags(D**-0.5)
        trans_mat = invsqrtD @ adj.transpose() @ invsqrtD
    else:
        invD = sparse.diags(D**-1.0)
        trans_mat = adj.transpose() @ invD
    return alpha * inv(I - (1 - alpha) * trans_mat)


if __name__ == "__main__":

    # Delay analysis of the calculation of PPR.

    from loader import load_graph
    from nodes import Node
    import random
    import time
    import os
    import matplotlib.pyplot as plt

    graph_name = "fb"
    graph = load_graph(Node, graph_name)
    n = graph.number_of_nodes()
    pers = np.zeros((n, 50))
    idxs = random.sample(range(n), k=n // 5)
    pers[idxs] = np.random.normal(0, 1, (len(idxs), pers.shape[1]))

    alpha_vals = np.arange(0.1, 0.91, 0.1)
    elapsed_power = []
    elapsed_exact = []
    for alpha in alpha_vals:
        print(f"for alpha {alpha}")
        start = time.time()
        ppr1 = power_analytic_ppr(nx.adjacency_matrix(graph), alpha, pers)
        elapsed = time.time() - start
        elapsed_power.append(elapsed)
        print(f"power method {elapsed} secs")

        start = time.time()
        ppr2 = exact_analytic_ppr(nx.adjacency_matrix(graph), alpha, pers)
        elapsed = time.time() - start
        elapsed_exact.append(elapsed)
        print(f"exact method {elapsed} secs")

        print(f"difference {np.max(np.abs(ppr1-ppr2))}")

    plt.figure()
    plt.grid()
    plt.plot(alpha_vals, elapsed_power, label="power")
    plt.plot(alpha_vals, elapsed_exact, label="exact")
    plt.legend()
    plt.xlabel("PPR alpha")
    plt.ylabel("Time (secs)")
    plt.title(f"PPR time analysis for graph {graph_name}")

    imgs_path = os.path.join(os.path.dirname(__file__), "img")
    figsfolder_path = os.path.join(imgs_path, "ppr_delay_analysis")
    if not os.path.exists(figsfolder_path):
        os.mkdir(figsfolder_path)
    fig_path = os.path.join(figsfolder_path, graph_name)
    plt.savefig(fig_path)

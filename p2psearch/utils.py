import numpy as np

from scipy import sparse
from scipy.sparse.linalg import inv


def analytic_ppr(adj, alpha, symmetric=True):
    """
    Calculates the personalized page rank diffusion matrix.
    Depending on the teleport probability (alpha), the computation is more efficient
    as an inverse matrix or power series operation.

    Arguments:
        adj (np.array): Adjacency matrix of a graph.
        alpha (float): Personalized page rank teleport probability.
        symmetric (bool): Selects the symmetric or asymmetric version of the diffusion matrix.

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
        for _ in range(max_tries):
            S = I + W @ S_old
            dS = np.max(np.abs(S - S_old))
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

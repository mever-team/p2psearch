import random

from uuid import uuid4
from ir import load_dataset
from network import load_network
from network.nodes import HardSumEmbeddingNode
from datatypes import *
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt

import numpy as np
import random
import networkx as nx


def distance(emb_mat_1, emb_mat_2):
    return np.linalg.norm(emb_mat_1 - emb_mat_2, axis=-1, ord=1).max(axis=-1)


class EmbeddingMonitor:

    def __init__(self, nodes, tolerance):
        self.nodes = nodes
        self.embs = [np.array([node.embedding for node in nodes])]
        self.tolerance = tolerance

    @property
    def embeddings(self):
        return np.array(self.embs)

    def __call__(self):
        self.embs.append(np.array([node.embedding for node in self.nodes]))
        emb_diff = distance(self.embs[-1], self.embs[-2])
        print(
            f"[Asynchronous diffusion monitor]: embedding variation in an epoch {emb_diff}"
        )
        return emb_diff > self.tolerance


class Simulation:

    def __init__(self, dataset_name, graph_name, ppr_a, n_docs, max_epochs, tolerance):
        self.sim_id = uuid4()
        self.dset = load_dataset(dataset=dataset_name)
        self.network = load_network(
            dataset=graph_name,
            init_node=lambda name: HardSumEmbeddingNode(name, self.dset.dim),
            ppr_a=ppr_a,
        )

        self.n_docs = n_docs
        self.max_epochs = max_epochs
        self.tolerance = tolerance

        self.params = {
            "dataset_name": dataset_name,
            "graph_name": graph_name,
            "ppr_a": ppr_a,
            "n_docs": n_docs,
            "max_epochs": max_epochs,
            "tolerance": 10**-10,
        }

    def run(self):

        self.network.clear()

        docs = self.dset.sample_other_docs(self.n_docs)

        self.network.scatter_docs(docs)

        monitor = EmbeddingMonitor(self.network.nodes, self.tolerance)
        self.network.diffuse_embeddings(epochs=self.max_epochs, monitor=monitor)
        async_embeddings = monitor.embeddings

        self.network.diffuse_fast_embeddings()
        exact_embeddings = self.network.embeddings
        return {"emb_diffs": distance(async_embeddings, exact_embeddings)}

    def save(self, results):
        runs_path = Path(__file__).parent / "runs"
        runs_path.mkdir(exist_ok=True)
        
        with open(runs_path / f"{self.sim_id}.txt", "w") as f:

            f.write("PARAMETERS\n")
            f.write("----------\n")
            for param_name, param_value in self.params.items():
                f.write(f"{param_name}: {param_value}\n")

            f.write("\n")

            f.write("RESULTS\n")
            f.write("-------\n")
            for res_name, res_value in results.items():
                f.write(f"{res_name}: {res_value}\n")

    def __call__(self, save=True):
        results = self.run()
        if save:
            self.save(results)
        return results

    def print(self, text):
        print(f"[simulation {self.sim_id}]: {text}")


parser = ArgumentParser()
parser.add_argument("-ni", "--n-iters", type=int, default=4)
parser.add_argument("-nd", "--n-docs", type=int, default=10)
parser.add_argument("-g", "--graph-name", type=str, default="fb")
parser.add_argument("-d", "--dataset-name", type=str, default="glove")
parser.add_argument("-a", "--ppr-a", type=float, default=0.5)
parser.add_argument("-me", "--max-epochs", type=int, default=100)
parser.add_argument("-t", "--tolerance", type=int, default=10**-10)

args = parser.parse_args()

sim = Simulation(
    dataset_name=args.dataset_name,
    graph_name=args.graph_name,
    ppr_a=args.ppr_a,
    n_docs=args.n_docs,
    max_epochs=args.max_epochs,
    tolerance=args.tolerance,
)

results = sim()
print(results)

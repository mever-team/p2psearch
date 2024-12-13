from p2psearch.ir import load_dataset
from p2psearch.network import load_network
from p2psearch.network import HardSumEmbeddingNode
from p2psearch.datatypes import *
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from .common import set_seed


def distance(emb_mat_1, emb_mat_2):
    return np.linalg.norm(emb_mat_1 - emb_mat_2, axis=-1, ord=1).max(axis=-1)


class DiffusionMonitorWithEarlyStop:
    """
    A utility class that stores the embeddings in each diffusion round.
    It also provides an early stop criterion when the embeddings have converged.
    """

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
        # print(f"[Diffusion monitor]: embedding variation in a round {emb_diff}")
        return emb_diff > self.tolerance


class Simulation:
    """
    A simulation that validates the convergence of the asynchronous personalized page rank
    diffusion embeddings to their exact values, computed analytically.

    In each iteration, simulation scatters a given number of documents and diffuses the
    embeddings asynchronously until convergence. The convergence is checked via a distance
    metric that captures the deviations from the exact values, which are tracked as output.
    The results from all iterations are summarized in a text file and plotted.
    """

    def __init__(
        self, dataset_name, graph_name, ppr_a, n_docs, n_iters, max_epochs, tolerance, seed
    ):
        self.sim_id = str(uuid4())
        self.dset = load_dataset(dataset=dataset_name)
        self.network = load_network(
            dataset=graph_name,
            init_node=lambda name: HardSumEmbeddingNode(name, self.dset.dim),
            ppr_a=ppr_a,
        )

        self.n_docs = n_docs
        self.n_iters = n_iters
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.seed = seed
        set_seed(seed)

        self.params = {
            "dataset_name": dataset_name,
            "graph_name": graph_name,
            "ppr_a": ppr_a,
            "n_docs": n_docs,
            "n_iters": n_iters,
            "max_epochs": max_epochs,
            "tolerance": tolerance,
            "seed": seed
        }

    def iterate(self):

        self.network.clear()

        docs = self.dset.sample_other_docs(self.n_docs)

        self.network.scatter_docs(docs)

        monitor = DiffusionMonitorWithEarlyStop(self.network.nodes, self.tolerance)
        self.network.diffuse_embeddings(epochs=self.max_epochs, monitor=monitor)
        async_embeddings = monitor.embeddings

        self.network.diffuse_fast_embeddings()
        exact_embeddings = self.network.embeddings
        return {"emb_diffs": distance(async_embeddings, exact_embeddings)}

    def run(self):
        all_emb_diffs = []
        for _ in tqdm(range(self.n_iters)):
            all_emb_diffs.append(self.iterate()["emb_diffs"])
        return {"all_emb_diffs": all_emb_diffs}

    def save(self, results):
        run_path = Path(__file__).parent / "check_ppr_convergence/runs" / str(self.sim_id)
        run_path.mkdir(exist_ok=True, parents=True)

        with open(run_path / f"results.txt", "w") as f:

            f.write("PARAMETERS\n")
            f.write("----------\n")
            for param_name, param_value in self.params.items():
                f.write(f"{param_name}: {param_value}\n")

            f.write("\n")

            f.write("RESULTS\n")
            f.write("-------\n")
            f.write("embedding deviation from analytic values\n")
            for i, emb_diffs in enumerate(results["all_emb_diffs"]):
                f.write(
                    f"iter {i}: {emb_diffs[0]} -> {emb_diffs[-1]} in {len(emb_diffs)} epochs\n"
                )

        fig, ax = plt.subplots(figsize=(6, 5))
        for diffs in results["all_emb_diffs"]:
            ax.plot(diffs, "-", ms=7, lw=1.0)
            ax.grid()
            ax.set_xlabel("Epochs", family="serif", size=16)
            ax.set_ylabel("Embeddings convergence metric", family="serif", size=16)
            ax.legend(prop={"family": "serif", "size": 13})
            fig.savefig(run_path / "plot.png")

    def __call__(self, save=True):
        results = self.run()
        if save:
            self.save(results)
        return results

    def print(self, text):
        print(f"[simulation {self.sim_id}]: {text}")


parser = ArgumentParser()
parser.add_argument(
    "-ni",
    "--n-iters",
    type=int,
    help="Number of times to check convergence in the same network.",
)
parser.add_argument(
    "-nd", "--n-docs", type=int, help="Number of documents to scatter in the network."
)
parser.add_argument(
    "-a", "--ppr-a", type=float, help="Diffusion parameter of personalized page rank."
)
parser.add_argument(
    "-g", "--graph-name", type=str, default="fb", help="Name of the network graph."
)
parser.add_argument(
    "-d",
    "--dataset-name",
    type=str,
    default="glove",
    help="Name of the retrieval dataset.",
)
parser.add_argument(
    "-me",
    "--max-epochs",
    type=int,
    default=500,
    help="Maximum number of epochs to wait for convergence.",
)
parser.add_argument(
    "-tol",
    "--tolerance",
    type=float,
    default=1.0e-10,
    help="Tolerance for convergence.",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=1,
    help="Simulation seed.",
)
args = parser.parse_args()

sim = Simulation(
    dataset_name=args.dataset_name,
    graph_name=args.graph_name,
    ppr_a=args.ppr_a,
    n_docs=args.n_docs,
    n_iters=args.n_iters,
    max_epochs=args.max_epochs,
    tolerance=args.tolerance,
    seed=args.seed,
)

results = sim(save=True)
print(results)

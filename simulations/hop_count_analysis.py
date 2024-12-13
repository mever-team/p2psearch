from p2psearch.ir import load_dataset
from p2psearch.network import load_network
from p2psearch.network import HardSumEmbeddingNode
from p2psearch.datatypes import *
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path


class Simulation:

    def __init__(
        self, dataset_name, graph_name, ppr_a, n_docs, n_iters, n_searches_per_iter, ttl
    ):
        self.sim_id = uuid4()
        self.dset = load_dataset(dataset=dataset_name)
        self.network = load_network(
            dataset=graph_name,
            init_node=lambda name: HardSumEmbeddingNode(name, self.dset.dim),
            ppr_a=ppr_a,
        )

        self.n_docs = n_docs
        self.n_iters = n_iters
        self.n_searches_per_iter = n_searches_per_iter
        self.ttl = ttl

        self.params = {
            "dataset_name": dataset_name,
            "graph_name": graph_name,
            "ppr_a": ppr_a,
            "n_docs": n_docs,
            "n_iters": n_iters,
            "n_searches_per_iter": n_searches_per_iter,
            "ttl": ttl,
        }

    def iterate(self):

        # self.print("clearing network")
        self.network.clear()

        # self.print("sampling queries and documents")
        query, gold_doc = self.dset.sample_gold_pair()
        other_docs = self.dset.sample_other_docs(self.n_docs - 1)

        # self.print("scattering documents")
        self.network.scatter_doc(gold_doc)
        self.network.scatter_docs(other_docs)

        # self.print("diffusing embeddings")
        self.network.diffuse_fast_embeddings()

        # self.print("scattering query messages")
        searches = [QuerySearch(query) for _ in range(self.n_searches_per_iter)]
        messages = [search.spawn_message(self.ttl) for search in searches]
        self.network.scatter_messages(messages)

        # self.print("forwarding query messages")
        _ = self.network.forward_messages(epochs=5 * self.ttl, monitor=None)

        # self.print("computing stats")
        successful_searches = [
            search for search in searches if search.candidate_doc == gold_doc
        ]
        hops = [search.hops_to_reach_candidate_doc for search in successful_searches]

        return {
            "n_total": len(searches),
            "n_success": len(successful_searches),
            "hops_success": hops,
        }

    def run(self):

        n_total = 0
        n_success = 0
        hops_success = []
        for _ in tqdm(range(self.n_iters)):
            results = self.iterate()
            n_total += results["n_total"]
            n_success += results["n_success"]
            hops_success.extend(results["hops_success"])

        return {
            "n_total": n_total,
            "n_success": n_success,
            "hops_success": hops_success,
        }

    def postprocess(self, results):
        return {
            "n_total": results["n_success"],
            "n_success": results["n_total"],
            "p_success": results["n_success"] / results["n_total"],
            "median_hops": int(np.median(results["hops_success"])),
            "mean_hops": np.mean(results["hops_success"]),
            "std_hops": np.std(results["hops_success"]),
        }

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
        results = self.postprocess(results)
        if save:
            self.save(results)
        return results

    def print(self, text):
        print(f"[simulation {self.sim_id}]: {text}")


parser = ArgumentParser()
parser.add_argument("-ni", "--n-iters", type=int, help="Number of iterations.")
parser.add_argument(
    "-nd", "--n-docs", type=int, help="Number of documents in the network."
)
parser.add_argument(
    "-nm", "--n-messages", type=int, help="Number of messages per query in the network."
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
    "-a", "--ppr-a", type=float, help="Diffusion parameter of personalized page rank."
)
parser.add_argument("-t", "--ttl", type=int, help="Time-to-live of the query messages.")


args = parser.parse_args()

sim = Simulation(
    dataset_name=args.dataset_name,
    graph_name=args.graph_name,
    ppr_a=args.ppr_a,
    n_docs=args.n_docs,
    n_iters=args.n_iters,
    n_searches_per_iter=args.n_messages,
    ttl=args.ttl,
)

results = sim()
print(results)

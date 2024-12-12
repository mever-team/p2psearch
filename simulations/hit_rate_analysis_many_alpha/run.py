import random 
import itertools

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



class Simulation:

    def __init__(
        self, dataset_name, graph_name, all_ppr_a, n_docs, n_searches_per_iter, ttl
    ):
        self.sim_id = uuid4()
        self.dset = load_dataset(dataset=dataset_name)
        self.network = load_network(
            dataset=graph_name,
            init_node=lambda name: HardSumEmbeddingNode(name, self.dset.dim),
            ppr_a=None,
        )

        self.n_docs = n_docs
        self.n_searches_per_iter = n_searches_per_iter
        self.ttl = ttl
        self.all_ppr_a = all_ppr_a

        self.params = {
            "dataset_name": dataset_name,
            "graph_name": graph_name,
            "all_ppr_a": all_ppr_a,
            "n_docs": n_docs,
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
        gold_node = self.network.sample_node()
        gold_node.add_doc(gold_doc)
        self.network.scatter_docs(other_docs)

        # self.print("diffusing embeddings")
        self.network.diffuse_fast_embeddings()

        # self.print("scattering query messages")
        hop2node = {hop: random.choice(nodes_at_hop) for hop, nodes_at_hop in enumerate(self.network.stream_hops(start_node=gold_node))}
        
        hop2search = {}
        for hop, node in hop2node.items():
            search = QuerySearch(query) 
            hop2search[hop] = search
            node.add_query(search.spawn_message(self.ttl))

        # self.print("forwarding query messages")
        _ = self.network.forward_queries(epochs=5 * self.ttl, monitor=None)

        # self.print("computing stats")
        hop2success = {hop:int(search.candidate_doc == gold_doc) for hop, search in hop2search.items()}
    
        return {"hop2success": hop2success}

    def run_for_single_alpha(self, n_iters, ppr_a):
     
        self.network.set_ppr_a(ppr_a=ppr_a)

        hop2success = defaultdict(lambda: [])
        for _ in tqdm(range(n_iters)):
            results = self.iterate()
            for hop, success in results["hop2success"].items():
                hop2success[hop].append(success)
        
        return {
            "hop2success": hop2success
        }

    def run(self, n_iters):

        alpha2results = {}

        for ppr_a in self.all_ppr_a:
            results = self.run_for_single_alpha(n_iters, ppr_a=ppr_a)
            alpha2results[ppr_a] = self.postprocess(results)
        
        return alpha2results

    def postprocess(self, results):
        hops = sorted(list(results["hop2success"]))
        hit_rates = [np.mean(results["hop2success"][hop])  for hop in hops]
        return {
            "hops": hops,
            "hit_rates": hit_rates,
        }

    def save(self, alpha2results):
        run_path = Path(__file__).parent / "runs" / str(self.sim_id)
        run_path.mkdir(exist_ok=True)
        
        with open(run_path / f"results.txt", "w") as f:

            f.write("PARAMETERS\n")
            f.write("----------\n")
            for param_name, param_value in self.params.items():
                f.write(f"{param_name}: {param_value}\n")

            f.write("\n")

            f.write("RESULTS\n")
            f.write("-------\n")
            for ppr_a, results in alpha2results.items():
                f.write(f"ppr_a: {ppr_a}\n")
                for res_name, res_value in results.items():
                    f.write(f"{res_name}: {res_value}\n")

        marker = itertools.cycle(('+', '*', 'o', '.')) 
        
        fig, ax = plt.subplots(figsize=(6, 5))
        for alpha, results in alpha2results.items():
            hops, hit_rates = results["hops"], results["hit_rates"]
            ax.plot(hops, hit_rates, "k-"+next(marker), label=rf"$\alpha = {alpha}$", ms=7, lw=1.0)
        ax.grid()
        ax.set_xlabel("TTL (hops)", family="serif", size=16)
        ax.set_ylabel("Accuracy (%)", family="serif", size=16)
        ax.legend(prop={'family':"serif", 'size': 13})
        fig.savefig(run_path / "plot.png")

    def __call__(self, n_iters, save=True):
        
        alpha2results = self.run(n_iters=n_iters)
        if save:
            self.save(alpha2results)
        return alpha2results

    def print(self, text):
        print(f"[simulation {self.sim_id}]: {text}")


# n_success, p_success, hops = sim(graph_name, dataset_name, ppr_a, n_iters, n_docs)

parser = ArgumentParser()
parser.add_argument("-ni", "--n-iters", type=int, default=4)
parser.add_argument("-nd", "--n-docs", type=int, default=10)
parser.add_argument("-nm", "--n-messages", type=int, default=10)
parser.add_argument("-g", "--graph-name", type=str, default="fb")
parser.add_argument("-d", "--dataset-name", type=str, default="glove")
parser.add_argument("-a", "--all-ppr-a", type=list[float], default=[0.1, 0.5, 0.9])
parser.add_argument("-t", "--ttl", type=int, default=50)

args = parser.parse_args()

alpha2results = {}


sim = Simulation(
    dataset_name=args.dataset_name,
    graph_name=args.graph_name,
    all_ppr_a=args.all_ppr_a,
    n_docs=args.n_docs,
    n_searches_per_iter=args.n_messages,
    ttl=args.ttl,
)

results = sim(args.n_iters)
print(results)
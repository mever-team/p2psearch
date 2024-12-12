import random 

from uuid import uuid4
from ir import load_dataset
from network import load_network
from network.nodes import HardSumEmbeddingNode
from datatypes import *
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path





class Simulation:

    def __init__(
        self, dataset_name, graph_name, ppr_a, n_docs, n_searches_per_iter, ttl
    ):
        self.sim_id = uuid4()
        self.dset = load_dataset(dataset=dataset_name)
        self.network = load_network(
            dataset=graph_name,
            init_node=lambda name: HardSumEmbeddingNode(name, self.dset.dim),
            ppr_a=ppr_a,
        )

        self.n_docs = n_docs
        self.n_searches_per_iter = n_searches_per_iter
        self.ttl = ttl

        self.params = {
            "dataset_name": dataset_name,
            "graph_name": graph_name,
            "ppr_a": ppr_a,
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

    def run(self, n_iters):
        from collections import defaultdict
        hop2success = defaultdict(lambda: [])
        for _ in tqdm(range(n_iters)):
            results = self.iterate()
            for hop, success in results["hop2success"].items():
                hop2success[hop].append(success)
        return {
            "hop2success": hop2success
        }

    def postprocess(self, results):
        hops = sorted(list(results["hop2success"]))
        hit_rates = [np.mean(results["hop2success"][hop])  for hop in hops]
        return {
            "hops": hops,
            "hit_rates": hit_rates,
        }


    def save(self, results):
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
            for res_name, res_value in results.items():
                f.write(f"{res_name}: {res_value}\n")

        hops, hit_rates = np.array(results["hops"]), np.array(results["hit_rates"])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(hops, hit_rates, "k-", ms=7, lw=1.0)
        ax.grid()
        ax.set_xlabel("TTL (hops)", family="serif", size=16)
        ax.set_ylabel("Accuracy (%)", family="serif", size=16)
        ax.legend(prop={'family':"serif", 'size': 13})
        fig.savefig(run_path / "plot.png")

    def __call__(self, n_iters, save=True):
        results = self.run(n_iters=n_iters)
        results = self.postprocess(results)
        if save:
            self.save(results)
        return results

    def print(self, text):
        print(f"[simulation {self.sim_id}]: {text}")


# n_success, p_success, hops = sim(graph_name, dataset_name, ppr_a, n_iters, n_docs)

parser = ArgumentParser()
parser.add_argument("-ni", "--n-iters", type=int, default=4)
parser.add_argument("-nd", "--n-docs", type=int, default=10)
parser.add_argument("-nm", "--n-messages", type=int, default=10)
parser.add_argument("-g", "--graph-name", type=str, default="fb")
parser.add_argument("-d", "--dataset-name", type=str, default="glove")
parser.add_argument("-a", "--ppr-a", type=float, default=0.5)
parser.add_argument("-t", "--ttl", type=int, default=50)

args = parser.parse_args()

sim = Simulation(
    dataset_name=args.dataset_name,
    graph_name=args.graph_name,
    ppr_a=args.ppr_a,
    n_docs=args.n_docs,
    n_searches_per_iter=args.n_messages,
    ttl=args.ttl,
)

results = sim(args.n_iters)
print(results)

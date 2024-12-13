# p2pSearch
This repository contains the code for the research paper "A Graph Diffusion Scheme for Decentralized Content Search based on Personalized PageRank", which was presented in the 2022 IEEE ICDCSW workshop.

## Paper Overview
The topic of our paper is _decentralized search algorithms_, which we believe are direly needed as the calls for a decentralized Web are mounting. Indeed, the Web is now a highly centralized place, marked by the dominance of global-scale social media networks and platforms. It wasn't always like this. In the 2000s, decentralized applications and peer-to-peer networks were popular with pioneers like Gnutella, BitTorrent, and Freenet, but they have since fallen out of favor. The reasons were partly technical, as lack of coordination brings forth significant challenges. Resource discovery discovery in particular forces queries to traverse the network based on limited knowledge of what resources are stored at each node. Distributed hash tables can bring structure and increase scalability but are best at locating resources with their identifiers at the expense of richer queries.

In this context, our work revisits the classic problem of decentralized search and gives it a fresh twist. Inspired by the emergence of _graph signal processing_ and _embedding-based retrieval_, we propose a graph diffusion scheme based on the _personalized pagerank_ algorithm for the nodes to advertise their stored content to their peers in the form of _embeddings_, i.e., vector representations of resources. This formulation is attractive as the diffused embeddings can be proven to converge to stable values even if the nodes operate asynchronously, as is typical of peer-to-peer applications. The diffused embeddings can further guide queries towards more promising nodes, increasing the efficiency of the search. The performance of our scheme is evaluated via the simulations contained in this repository.

## Repository Overview
Our code simulates an information retrieval application over a peer-to-peer network. It is written in Python, making extensive use of the ```numpy``` and ```networkx``` libraries, and contains the following modules:
- **ir**: Generates and loads an information retrieval dataset. In particular, we emulate retrieval data by carefully selecting Glove embeddings as queries and documents, as well as generating their judgement values.
- **network**: Loads a peer-to-peer network based on a selection of graph datasets. It abstracts the network and the nodes via the ```P2PNetwork``` and ```Node``` classes where the latter be extended to support different node behaviors.
- **datatypes**: Contains the main data objects of our simulation, the documents and the queries, through the ```Document```, ```Query```, ```QuerySearch```, and ```QueryMessage``` classes. It is worth highlighting the difference between the 3 query classes, which can be a point of confusion. ```Query``` represents a query sample from a dataset, ```QuerySearch``` a search operation in the network associated with a query, and ```QueryMessage``` the actual messages that are passed among the nodes. This division allows defining different search operations for the same query that can bifurcate into multiple parallel walks.
- **simulations**: Contains scripts for the experiments conducted in the paper, easily executed with arguments passed via the ```argparse``` interface. The scripts are: 
– ```check_ppr_convergence.py```: A simulation that verifies the convergence of the asynchronously diffused embeddings to the ones computed analytically. This allows using the analytical embeddings to significantly speed up the simulation.
– ```hit_rate_analysis.py```: A simulation that computes the search hit rate of queries as a function of the hop distance between the query and the gold document.
– ```hop_count_analysis.py```: A simulation that computes descriptive statistics for the hop count of successful walks from the querying node until the node with the gold document.


### Attributions
```
Our publication - available via Arxiv https://arxiv.org/abs/2204.12902
N. Giatsoglou, E. Krasanakis, S. Papadopoulos and I. Kompatsiaris,
"A Graph Diffusion Scheme for Decentralized Content Search based on Personalized PageRank," 2022 IEEE 42nd International Conference on Distributed Computing Systems Workshops (ICDCSW), Bologna, Italy, 2022, pp. 53-59
EU project acknowledgements: AI4Media (GA 951911), MediaVerse (GA 957252) and HELIOS (GA 825585)
```

```
Glove dataset - available via https://nlp.stanford.edu/projects/glove/ and the Gensim library
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. 
```

```
Facebook combined ego graph - http://snap.stanford.edu/data/ego-Facebook.html
J. Leskovec en R. Sosič,
“SNAP: A General-Purpose Network Analysis and Graph-Mining Library”,
ACM Transactions on Intelligent Systems and Technology (TIST), vol 8, no 1, bl 1, 2016.
```
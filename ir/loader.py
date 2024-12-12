import os
import numpy as np
import random as rnd

from pathlib import Path
from datatypes import Document, Query


class NamedEmbeddings:

    def __init__(self, names: np.array, embeddings: np.array):
        self.names = names
        self.embeddings = embeddings

    def sample_one(self, return_type=Document):
        idx = rnd.choice(range(len(self.names)))
        return return_type(self.names[idx], self.embeddings[idx])

    def sample_many(self, k, return_type=Document):
        idxs = rnd.sample(range(len(self.names)), k=k)
        return [return_type(self.names[idx], self.embeddings[idx]) for idx in idxs]
    
    def iterate(self, return_type=Document):
        return (return_type(_id, emb) for _id, emb in zip(self.names, self.embeddings))

class Dataset:

    def __init__(
        self,
        query_embeddings: NamedEmbeddings,
        doc_embeddings: NamedEmbeddings,
        other_doc_embeddings: NamedEmbeddings,
        qrels: dict[str, str],
    ):
        self.qrels = qrels
        self.name2query = {query.name: query  for query in query_embeddings.iterate(return_type=Query)}
        self.name2doc = {doc.name: doc for doc in doc_embeddings.iterate(return_type=Document)}
        self.other_doc_embeddings = other_doc_embeddings # other docs >> docs, do not convert to Document objects to avoid overhead
        self.dim = len(self.other_doc_embeddings.embeddings[0].flatten())
    
        self._query_names = list(self.name2query)
        self._doc_names = list(self.name2doc)

    def sample_query(self):
        return self.name2query[rnd.choice(self._query_names)]

    def sample_queries(self, k):
        return [self.name2query[name] for name in rnd.sample(self._query_names, k)]
    
    def sample_other_docs(self, k):
        return self.other_doc_embeddings.sample_many(k, return_type=Document)

    def sample_gold_pair(self):
        query = self.sample_query()
        doc = self.name2doc[self.qrels[query.name]]
        return query, doc

    def sample_gold_pairs(self, k):
        queries = self.sample_queries(k)
        return [(query, self.name2doc[self.qrels[query.name]]) for query in queries]
       

def read_qrels_file(path):
    with open(path, "r", encoding="utf8") as f:
        results = dict()
        for line in f:
            que_id, doc_id, _ = line.strip().split("\t")
            results[que_id] = doc_id
    return results


def read_embeddings_file(path):
    arrays = np.load(path)
    return arrays["ids"], arrays["embs"]


def load_dataset(dataset: str):

    dset_path = Path(__file__).parent / str(dataset)
    if not dset_path.exists():
        raise Exception(f'Dataset "{dataset}" folder does not exist.')
    
    generation_script_path = dset_path / "generate.py"
    if not generation_script_path.exists():
        raise Exception(f'Dataset "{dataset}" generation script does not exist.')
    
    try:
        qrels = read_qrels_file(dset_path / "qrels.txt")
        query_embeddings = NamedEmbeddings(*read_embeddings_file(dset_path / "queries_embs.npz"))
        doc_embeddings = NamedEmbeddings(*read_embeddings_file(dset_path / "docs_embs.npz"))
        other_doc_embeddings = NamedEmbeddings(*read_embeddings_file(dset_path / "other_docs_embs.npz"))

    except FileNotFoundError:
        
        os.system(f"python {generation_script_path}")

        qrels = read_qrels_file(dset_path / "qrels.txt")
        query_embeddings = NamedEmbeddings(*read_embeddings_file(dset_path / "queries_embs.npz"))
        doc_embeddings = NamedEmbeddings(*read_embeddings_file(dset_path / "docs_embs.npz"))
        other_doc_embeddings = NamedEmbeddings(*read_embeddings_file(dset_path / "other_docs_embs.npz"))

    return Dataset(
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        other_doc_embeddings=other_doc_embeddings,
        qrels=qrels,
    )


# def load_texts(dataset="glove", type="docs"):
#     """
#     Loads the texts of a retrieval dataset.
#     If the dataset does not exist locally, it is downloaded and cached,
#     hence the first time may be slow.

#     Arguments:
#         dataset (str): The name of a retrieval dataset.
#         type (str): The type of the embeddings. Available types are "queries", "documents", "other_docs",
#             respesenting queries, relevant / gold documents, irrelevant / other documents.
#     Returns:
#         dict[str, str]: A dictionary of texts indexed by the query or document name.
#     """
#     filepath = ir.get_texts_path(dataset, type)
#     if not os.path.exists(filepath):
#         ir.download(dataset)
#     with open(filepath, "r", encoding="utf8") as f:
#         texts = dict()
#         for line in f:
#             idx, text = line.strip().split("\t")
#             texts[idx] = text
#     return texts

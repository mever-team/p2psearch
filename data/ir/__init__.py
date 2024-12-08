import os


DATA_DIR = os.path.dirname(__file__)

# utility aliases
TYPE2FILENAME_DICT = {
    "q": "queries",
    "query": "queries",
    "queries": "queries",
    "d": "docs",
    "doc": "docs",
    "docs": "docs",
    "document": "docs",
    "documents": "docs",
    "o": "other_docs",
    "other": "other_docs",
    "others": "other_docs",
    "other_docs": "other_docs",
}


def get_dataset_path(dataset):
    """
    Returns the standard path for a dataset used for information retrieval.

    Arguments:
        dataset (str): The name of the dataset.

    Returns:
        str: The standard absolute path for the dataset.
    """
    return os.path.join(DATA_DIR, dataset)


def get_qrels_path(dataset):
    """
    Returns the standard path for the ground truth of a retrieval dataset.

    Arguments:
        dataset (str): The name of the dataset.

    Returns:
        str: The standard absolute path for the ground truth data.
    """
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, "qrels.txt")


def get_texts_path(dataset, type):
    """
    Returns the standard path for the texts of a retrieval dataset.

    Arguments:
        dataset (str): The name of the dataset.
        type (str): The type of the text. Available types are "queries", "documents", "other_docs",
            respesenting queries, relevant / gold documents, irrelevant / other documents.

    Returns:
        str: The standard absolute path for the ground truth data.
    """
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, TYPE2FILENAME_DICT[type] + ".txt")


def get_embeddings_path(dataset, type):
    """
    Returns the standard path for the embeddings of a retrieval dataset.

    Arguments:
        dataset (str): The name of the dataset.
        type (str): The type of the text. Available types are "queries", "documents", "other_docs",
            respesenting queries, relevant / gold documents, irrelevant / other documents.

    Returns:
        str: The standard absolute path for the ground truth data.
    """
    dset_path = get_dataset_path(dataset)
    return os.path.join(dset_path, TYPE2FILENAME_DICT[type] + "_embs.npz")


def download(dataset):
    """
    Downloads a dataset and converts it a form suitable for information retrieval.

    Arguments:
        dataset (str): The name of the dataset.
    """
    dset_path = get_dataset_path(dataset)
    script_path = os.path.join(dset_path, "generate_script.py")
    os.system(f"python {script_path} --path {dset_path}")

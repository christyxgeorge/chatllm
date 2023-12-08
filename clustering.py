"""Implementing Kmeans clustering on PDF chunks"""
# flake8: noqa
import logging
import os
import re

from typing import Literal

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import (
    pairwise_distances_argmin_min,
    silhouette_samples,
    silhouette_score,
)
from transformers import AutoTokenizer

from chatllm.loaders import SmartPDFLoader

logger = logging.getLogger(__name__)

# ====================================================================================
# Checking effective means of Summarization
# a) Do a KMeans clustering of the documents/paragraphs. For each cluster, figure
#    out the closest document to the cluster center. This is the most representative
#    document for the cluster.
# b) Identifying number of clusters is still an issue. Can it be a factor of the
#    number of documents? [Silhouette coefficient]
# c) Check if there are alternatives to KMeans clustering [Density based clustering like
#    DBSCAN or Spectral Clustering]
# d) Reference: https://pashpashpash.substack.com/p/tackling-the-challenge-of-document
# ====================================================================================


def dbscan_clusters(X):
    """Generate clusters using DBSCAN"""
    db = DBSCAN(eps=0.3, min_samples=4).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)


def mbkmeans_clusters(X, k, mb, verbose=False, print_silhouette_values=False):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb, n_init="auto").fit(X)
    sil_score = silhouette_score(X, km.labels_)
    if verbose:
        print(f"For n_clusters = {k}")
        print(f"\tSilhouette coefficient: {sil_score:0.2f}")
        print(f"\tInertia:{km.inertia_}")

    sample_silhouette_values = silhouette_samples(X, km.labels_)
    if print_silhouette_values:
        print("Silhouette values:")
    silhouette_values = []
    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
        # print(f"Cluster Silhouette Values = {cluster_silhouette_values}")
        if cluster_silhouette_values.any():
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        else:
            silhouette_values.append((i, 0, 0, 0, 0))
    silhouette_values = sorted(silhouette_values, key=lambda tup: tup[2], reverse=True)
    if print_silhouette_values:
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]:02d}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    return km, km.labels_, sil_score, silhouette_values, list(closest)


def process_pdf(file_name):
    smart_docs = SmartPDFLoader().load_data(file_name)
    texts = [doc.text for doc in smart_docs]
    metadata = [doc.metadata for doc in smart_docs]

    return texts, metadata


def print_cluster_info(df_clusters, verbose=False):
    test_cluster = df_clusters["cluster"][0]
    print()
    print(df_clusters.head(10))
    print()

    if verbose:
        most_representative_docs = np.argsort(
            np.linalg.norm(vectorized_docs - clustering.cluster_centers_[test_cluster], axis=1)
        )
        print(f"---Most representative docs for cluster {test_cluster}----------")
        for d in most_representative_docs[:3]:
            print(docs[d])
            print("-------------")


def initialize_logging(verbose=False, debug=False) -> None:
    # ===========================================================================================
    # Logging Setup
    # Django method = logging.config.dictConfig(config)
    # ===========================================================================================
    log_format = "{asctime}.{msecs:03.0f} {levelname} [{name}]: {message}"
    log_style: Literal["%", "{", "$"] = "{"
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARN)
    logging.basicConfig(format=log_format, level=log_level, datefmt="%I:%M:%S", style=log_style)


if __name__ == "__main__":
    initialize_logging(True, False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # Handle libiomp and libomp conflicts

    file = "/Users/christy/Desktop/interesting/Indian renaissance read.pdf"
    in_docs, metadata = process_pdf(file)
    docs = [re.sub(r"^.*\n", "", x) for x in in_docs]  # Remove the first line of each doc

    ids = [f"{i+1}" for i in range(len(docs))]  # Create IDs for each doc

    # for i, (d, id) in enumerate(zip(docs, in_docs), 1):
    #     print(f"Document {i}:{len(d)} => {len(id)}")

    logger.info("Starting Tokenization and Embedding")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    vectorized_docs = embedding_model.encode(docs)
    tokens = tokenizer(docs, padding=True, truncation=True, return_tensors="np")
    tokenized_docs = tokens["input_ids"]
    logger.info("Completed Tokenization and Embedding")

    # dbscan_clusters(vectorized_docs)

    selected_k = 0
    max_silhouette_score = 0
    selected_cluster_labels = None
    selected_sil_sizes = []
    selected_closest = []
    max_num_clusters = min(50, len(vectorized_docs) / 2)
    logger.info(f"=====> Checking KMeans Clustering upto {max_num_clusters} clusters")
    for k in range(2, max_num_clusters):
        # logger.info(f"Running KMeans with k={k}")
        clustering, cluster_labels, sil_score, sil_values, closest = mbkmeans_clusters(
            X=vectorized_docs, k=k, mb=500, verbose=False, print_silhouette_values=False
        )
        sil_sizes = [t[0] for t in sil_values]
        size_sd = np.std(sil_sizes)
        # logger.info(f"== Silhouette Sizes: {np.std(sil_sizes)} // {sil_sizes}")
        # logger.info(f"== Silhouette Averages:   {[t[2] for t in sil_values]}")
        comp = ">" if sil_score > max_silhouette_score else "<"
        message = f"== Silhouette score for k={k} = {sil_score:.3f} {comp} {max_silhouette_score:.3f} // SD: {size_sd:.2f}"
        if sil_score > max_silhouette_score:
            message = message + f" // Old k = {selected_k}"
            selected_sil_sizes = sil_sizes
            selected_closest = closest
            max_silhouette_score = sil_score
            selected_k = k
            selected_cluster_labels = cluster_labels
        logger.info(message)

    logger.info(
        f"Selected Number of Clusters = {selected_k}, Silhouette Score = {max_silhouette_score:.3f} // SD: {np.std(selected_sil_sizes)}"
    )
    logger.info(f"== Silhouette Sizes: {selected_sil_sizes}")
    logger.info(f"== Selected Closest document = {selected_closest}")

    df_clusters = pd.DataFrame(
        {
            "text": docs,
            "cluster": selected_cluster_labels,
            "id": ids,
            "length": [len(doc) for doc in docs],
            "metadata": metadata,
            # "tokens": [" ".join(str(token)) for token in tokenized_docs],
            "closest": [True if id in selected_closest else False for id, doc in enumerate(docs)],
        }
    )
    print_cluster_info(df_clusters)

    df_closest = df_clusters[df_clusters["closest"]]
    print("Selected documents:")
    for index, row in df_closest.text.items():
        print(f"{index+1} => {len(row)} bytes")

    print("End of KMeans")

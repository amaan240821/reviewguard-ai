import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def cosine_matrix(embeddings, chunk_size=256):
    n = embeddings.shape[0]
    m = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk_size):
        m[start : start + chunk_size] = np.clip(
            cosine_similarity(embeddings[start : start + chunk_size], embeddings), 0, 1
        )
    np.fill_diagonal(m, 1)
    return m


def lexical_matrix(texts):
    vectors = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 5) if max(map(len, texts)) < 5 else (5, 5)
    ).fit_transform(texts)
    return cosine_matrix(vectors)


def near_duplicates(matrix, threshold=0.85):
    return [
        (int(i), int(j), float(matrix[i, j]))
        for i, j in zip(*np.where(np.triu(matrix, 1) >= threshold))
    ]


def nearest_neighbors(matrix, index, k=5):
    return [
        (int(i), float(matrix[index, i]))
        for i in np.argsort(-matrix[index])
        if i != index
    ][:k]

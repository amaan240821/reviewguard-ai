import hashlib, json
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .config import ROOT, EMBEDDING_REVISION


@lru_cache(maxsize=1)
def sentence_model():
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(2)
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(ROOT / "models/hf_cache"),
        revision=EMBEDDING_REVISION,
        device="cpu",
    )


def embed_texts(texts, backend="lsa"):
    texts = list(texts)
    cache = ROOT / "models/embedding_cache"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(
        json.dumps([backend, texts], ensure_ascii=False).encode()
    ).hexdigest()
    path = cache / f"{key}.npy"
    meta = cache / f"{key}.json"
    if path.exists() and meta.exists():
        return np.load(path), json.loads(meta.read_text())["backend"]
    actual = "tfidf-lsa"
    if backend == "sentence-transformer":
        try:
            values = cached_sentence_embeddings(texts)
            actual = "all-MiniLM-L6-v2"
        except Exception:
            values = None
    else:
        values = None
    if values is None:
        vec = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
        )
        matrix = vec.fit_transform(texts)
        if min(matrix.shape) > 3 and matrix.shape[1] > 65:
            values = TruncatedSVD(
                n_components=min(64, min(matrix.shape) - 1), random_state=42
            ).fit_transform(matrix)
        else:
            values = matrix.toarray()
        values = normalize(values)
    np.save(path, values)
    meta.write_text(json.dumps({"backend": actual}))
    return values, actual


def cached_sentence_embeddings(texts):
    cache = ROOT / "models/text_embeddings"
    cache.mkdir(exist_ok=True)
    values = [None] * len(texts)
    missing = []
    for i, text in enumerate(texts):
        path = cache / (hashlib.sha256(text.encode()).hexdigest() + ".npy")
        if path.exists():
            values[i] = np.load(path)
        else:
            missing.append((i, text, path))
    if missing:
        encoded = sentence_model().encode(
            [t for _, t, _ in missing],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for (i, _, path), value in zip(missing, encoded):
            np.save(path, value)
            values[i] = value
    return np.array(values)

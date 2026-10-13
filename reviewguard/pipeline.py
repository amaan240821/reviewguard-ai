from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from .config import Config
from .data import validate_batch
from .preprocess import clean_text, normalize
from .sentiment import analyze_sentiments
from .embeddings import embed_texts
from .similarity import cosine_matrix, nearest_neighbors, lexical_matrix
from .classifier import predict_deception
from .linguistic import linguistic_features, matched_spans
from .clustering import build_clusters
from .scoring import score_signals, rating_mismatch, duplication_signal
from .explain import explanations, counter_evidence, influential_terms


@dataclass
class ReviewResult:
    text: str
    cleaned_text: str
    score: float
    band: str
    signals: dict
    contributions: dict
    sentiment: dict
    features: dict
    reasons: list
    neighbors: list
    embedding_backend: str
    spans: list
    warnings: list
    counter_evidence: list
    influential_terms: list
    effective_weights: dict

    def to_dict(self):
        return asdict(self)


@dataclass
class BatchResult:
    reviews: pd.DataFrame
    results: list
    similarity: np.ndarray
    clusters: list
    edges: list
    embedding_backend: str
    pair_evidence: list
    weights: dict

    def to_dict(self):
        return dict(
            reviews=json.loads(self.reviews.to_json(orient="records")),
            results=[r.to_dict() for r in self.results],
            similarity=self.similarity.tolist(),
            clusters=self.clusters,
            edges=self.edges,
            pair_evidence=self.pair_evidence,
            embedding_backend=self.embedding_backend,
            weights=self.weights,
            summary={
                "reviews": len(self.reviews),
                "mean_suspicion": float(self.reviews.suspicion_score.mean()),
                "clusters": len(self.clusters),
                "near_duplicate_pairs": len(self.edges),
            },
        )


def _result(text, rating, c, d, config, neighbors, backend, sentiment):
    features = linguistic_features(text)
    signals = dict(
        D=d,
        C=float(c) if c is not None else None,
        L=features["score"],
        R=rating_mismatch(sentiment["compound"], rating),
        E=float(np.clip((abs(sentiment["compound"]) - 0.8) / 0.2, 0, 1))
        * min(1, 30 / max(features["word_count"], 1)),
    )
    score, band, contributions = score_signals(signals, config)
    warnings = []
    if features["word_count"] < 15:
        warnings.append("Short text: low reliability.")
    if sentiment.get("warning"):
        warnings.append(sentiment["warning"])
    if "MiniLM" not in backend:
        warnings.append(
            "Semantic transformer inactive: TF-IDF + LSA fallback, or no reference pool."
        )
    if c is None:
        warnings.append(
            "Classifier unavailable; run make train. Available weights renormalized."
        )
    return ReviewResult(
        text,
        clean_text(text),
        score,
        band,
        signals,
        contributions,
        sentiment,
        features,
        explanations(signals, rating is not None and np.isfinite(rating)),
        neighbors,
        backend,
        matched_spans(text),
        warnings,
        counter_evidence(features, text),
        influential_terms(clean_text(text)),
        {
            k: (
                v / 100 / signals[k]
                if signals[k]
                else (
                    config.weights[k]
                    / sum(
                        config.weights[j]
                        for j in config.weights
                        if signals[j] is not None
                    )
                    if signals[k] is not None
                    else 0.0
                )
            )
            for k, v in contributions.items()
        },
    )


def analyze_review(
    text: str,
    rating: float | None = None,
    reference: pd.DataFrame | list[str] | None = None,
    config: Config | None = None,
) -> ReviewResult:
    """Analyze a review with optional rating and comparison pool."""
    config = config or Config()
    if not isinstance(text, str) or not normalize(text):
        raise ValueError("Enter a review containing words.")
    if len(text) > 20000:
        raise ValueError("Review exceeds 20,000 characters.")
    if rating is not None and (not np.isfinite(rating) or not 1 <= rating <= 5):
        raise ValueError("Rating must be between 1 and 5.")
    refs = (
        []
        if reference is None
        else (
            reference["text"].astype(str).tolist()
            if isinstance(reference, pd.DataFrame)
            else list(reference)
        )
    )
    if len(refs) > config.max_batch_size:
        raise ValueError("Reference corpus exceeds batch limit.")
    neighbors = []
    d = None
    backend = "not used (no reference corpus)"
    if refs:
        values, backend = embed_texts(
            [clean_text(text)] + [clean_text(t) for t in refs], config.embedding_backend
        )
        matrix = cosine_matrix(values)
        lexical = lexical_matrix([clean_text(text)] + [clean_text(t) for t in refs])
        neighbors = [
            {
                "index": i - 1,
                "similarity": float(matrix[0, i]),
                "lexical_similarity": float(lexical[0, i]),
                "text": refs[i - 1],
            }
            for i, s in nearest_neighbors(np.maximum(matrix, lexical), 0)
        ]
        d = duplication_signal(
            max(matrix[0, 1:]),
            max(lexical[0, 1:]),
            sum(
                np.maximum(matrix[0, 1:], lexical[0, 1:]) >= config.duplicate_threshold
            ),
        )
    try:
        c = predict_deception([clean_text(text)])[0]
    except RuntimeError:
        c = None
    return _result(
        text,
        rating,
        c,
        d,
        config,
        neighbors,
        backend,
        analyze_sentiments([clean_text(text)], config.sentiment_backend)[0],
    )


def analyze_batch(
    df: pd.DataFrame, mapping: dict, config: Config | None = None, progress_cb=None
) -> BatchResult:
    """Validate, score, cluster, and explain a bounded review batch."""
    config = config or Config()
    out = validate_batch(df, mapping, config.max_batch_size)
    texts = out.text.map(clean_text).tolist()
    values, backend = embed_texts(texts, config.embedding_backend)
    matrix = cosine_matrix(values)
    lexical = lexical_matrix(texts)
    combined = np.maximum(matrix, lexical)
    labels, clusters, edges = build_clusters(combined, config.duplicate_threshold)
    try:
        probs = predict_deception(texts)
    except RuntimeError:
        probs = [None] * len(texts)
    sentiments = analyze_sentiments(texts, config.sentiment_backend)
    results = []
    for i, (text, rating, d) in enumerate(zip(out.text, out.rating, probs)):
        neighbors = [
            {
                "index": j,
                "similarity": float(matrix[i, j]),
                "lexical_similarity": float(lexical[i, j]),
                "text": out.text.iloc[j],
            }
            for j, s in nearest_neighbors(combined, i)
        ]
        others = [j for j in range(len(texts)) if j != i]
        dup = (
            duplication_signal(
                float(max(matrix[i, others], default=0)),
                float(max(lexical[i, others], default=0)),
                sum(combined[i, others] >= config.duplicate_threshold),
            )
            if others
            else None
        )
        r = _result(
            text,
            None if pd.isna(rating) else float(rating),
            d,
            dup,
            config,
            neighbors,
            backend,
            sentiments[i],
        )
        results.append(r)
        if progress_cb:
            progress_cb((i + 1) / len(texts))
    out["suspicion_score"] = [r.score for r in results]
    out["risk_band"] = [r.band for r in results]
    out["cluster_id"] = labels
    for signal in "DCLRE":
        out[signal] = [r.signals[signal] for r in results]
    out["sentiment"] = [r.sentiment["label"] for r in results]
    out["explanation"] = [" ".join(r.reasons) for r in results]
    for cluster in clusters:
        members = cluster["members"]
        cluster["mean_suspicion"] = float(out.iloc[members].suspicion_score.mean())
        cluster["coordination_score"] = float(
            cluster["mean_similarity"] * min(cluster["size"] / 5, 1) * 100
        )
        from collections import Counter
        from .preprocess import tokens

        cluster["shared_terms"] = [
            t
            for t, c in Counter(
                t for i in members for t in set(tokens(texts[i], True))
            ).most_common(10)
            if c >= 2
        ]
    clusters.sort(key=lambda c: -c["coordination_score"])
    pair_evidence = [
        {
            "a": a,
            "b": b,
            "semantic_similarity": float(matrix[a, b]),
            "lexical_similarity": float(lexical[a, b]),
        }
        for a, b, _ in edges
    ]
    return BatchResult(
        out, results, combined, clusters, edges, backend, pair_evidence, config.weights
    )

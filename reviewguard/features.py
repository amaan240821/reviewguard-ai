"""Fold-fitted scikit-learn feature transformers, serializable in joblib."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .linguistic import linguistic_features
from .embeddings import sentence_model


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        try:
            self.model_ = sentence_model()
            self.backend_ = "MiniLM"
        except Exception:
            self.model_ = None
            self.backend_ = "LSA fallback"
            self.vectorizer_ = TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2), min_df=2
            )
            mat = self.vectorizer_.fit_transform(X)
            self.svd_ = TruncatedSVD(
                n_components=min(64, min(mat.shape) - 1), random_state=42
            ).fit(mat)
        return self

    def transform(self, X):
        if self.model_ is not None:
            from .embeddings import cached_sentence_embeddings

            return cached_sentence_embeddings(list(X))
        return normalize(self.svd_.transform(self.vectorizer_.transform(X)))

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            state["model_"] = "reload"
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.__dict__.get("model_") == "reload":
            self.model_ = sentence_model()


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                [
                    f["score"],
                    min(f["word_count"] / 500, 1),
                    f["lexical_diversity"],
                    f["uppercase_ratio"],
                    f["punctuation_ratio"],
                    min(f["detail_hits"] / 20, 1),
                ]
                for f in map(linguistic_features, X)
            ]
        )

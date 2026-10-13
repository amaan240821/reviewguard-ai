"""Select models on training-only CV; calibrate without touching holdout."""

import json, joblib, logging, hashlib
from functools import lru_cache
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from .config import ROOT
from .data import load_dataset, split_data
from .features import EmbeddingFeatures, LinguisticFeatures

logger = logging.getLogger(__name__)
MODEL = ROOT / "models/deception_clf.joblib"


def tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True
    )


def candidates():
    lr = lambda: LogisticRegression(C=3, max_iter=2000, random_state=42)
    return {
        "TF-IDF + Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", lr())]),
        "TF-IDF + Linear SVM": Pipeline(
            [("tfidf", tfidf()), ("clf", LinearSVC(C=1, random_state=42))]
        ),
        "MiniLM + Logistic Regression": Pipeline(
            [("embeddings", EmbeddingFeatures()), ("clf", lr())]
        ),
        "Stacked TF-IDF + embeddings + linguistic": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("tfidf", tfidf()),
                            ("embeddings", EmbeddingFeatures()),
                            ("linguistic", LinguisticFeatures()),
                        ]
                    ),
                ),
                ("clf", lr()),
            ]
        ),
    }


def train_model():
    df = load_dataset()
    train, test = split_data(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(train.text, train.label.astype(str) + "_" + train.polarity))
    models = candidates()
    rows = []
    for name, model in models.items():
        values = cross_val_score(
            model, train.text, train.label, cv=splits, scoring="f1", n_jobs=1
        )
        rows.append(
            dict(
                model=name,
                cv_f1=float(values.mean()),
                cv_std=float(values.std()),
                fold_scores=values.tolist(),
            )
        )
        logger.info("%s CV F1: %.4f", name, values.mean())
    best = max(rows, key=lambda r: r["cv_f1"])["model"]
    model = CalibratedClassifierCV(models[best], method="sigmoid", cv=3)
    oof = cross_val_predict(
        model, train.text, train.label, cv=splits, method="predict_proba"
    )[:, 1]
    np.savez(
        ROOT / "models/train_oof.npz", indices=train.index.to_numpy(), probability=oof
    )
    model.fit(train.text, train.label)
    joblib.dump(model, MODEL)
    card = dict(
        algorithm=best,
        seed=42,
        train_rows=len(train),
        test_rows=len(test),
        train_hotels=sorted(train.hotel.unique().tolist()),
        test_hotels=sorted(test.hotel.unique().tolist()),
        calibration="3-fold sigmoid calibration inside each outer training fold",
        dataset="Ott Deceptive Opinion Spam Corpus v1.4",
        dataset_sha256=hashlib.sha256(
            (ROOT / "data/raw/deceptive-opinion.csv").read_bytes()
        ).hexdigest(),
        split="80/20 stratified by label × polarity",
        selection="Training-only 5-fold CV F1; heldout test used only for evaluation",
        scope="English hotel reviews; research screening only",
        model_comparison=rows,
    )
    (ROOT / "models/model_card.json").write_text(json.dumps(card, indent=2))
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables/model_comparison.csv", index=False)
    train.to_csv(ROOT / "data/processed/train.csv", index=False)
    test.to_csv(ROOT / "data/processed/test.csv", index=False)
    load_model.cache_clear()
    return card


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL) if MODEL.exists() else None


def predict_deception(texts):
    model = load_model()
    if model is None:
        raise RuntimeError("No trained classifier. Run make train first.")
    return model.predict_proba(list(texts))[:, 1]
